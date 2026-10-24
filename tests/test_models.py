from copy import deepcopy
from pathlib import Path
import unittest
import numpy as np
import torch

from clinical_multimodal.data.collate import collate_samples
from clinical_multimodal.data.fixed_dataset import FixedPreparedDataset as PreparedDataset
from clinical_multimodal.models.predictor import MultimodalRiskModel
from clinical_multimodal.models.pooling import masked_softmax
from clinical_multimodal.training.losses import masked_bce, training_pos_weight
from clinical_multimodal.training.trainer import Trainer
from tests.helpers import fixture, small_config, cache, TestWorkspace


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = TestWorkspace()
        self.addCleanup(self.temp.cleanup)
        _, self.bundle, self.processor = fixture(self.temp.name)
        self.cfg = small_config(self.temp.name)
        self.dataset = PreparedDataset(self.bundle, self.bundle.rows("train"), self.processor, cache(self.temp.name), 6)
        self.batch = collate_samples([self.dataset[i % len(self.dataset)] for i in range(4)])
        self.model = MultimodalRiskModel(14,2,32,self.cfg.model).eval()

    def test_causal_prefix_invariance(self):
        x = self.batch["inputs"]
        modified = x["vitals"].clone()
        modified[:,20:] = torch.randn_like(modified[:,20:]) * 100
        with torch.no_grad():
            a = self.model.vital.sequence(x["vitals"], x["time_mask"])
            b = self.model.vital.sequence(modified, x["time_mask"])
        torch.testing.assert_close(a[:,:20], b[:,:20], rtol=0,atol=0)

    def test_time_padding_and_chunk_padding_isolation(self):
        x = self.batch["inputs"]
        changed = {k:v.clone() for k,v in x.items()}
        changed["vitals"][~x["time_mask"]] = float("nan")
        changed["note_chunks"][~x["chunk_mask"]] = float("nan")
        with torch.no_grad():
            torch.testing.assert_close(self.model(x)["logits"], self.model(changed)["logits"], rtol=0,atol=0)

    def test_batch_invariance(self):
        with torch.no_grad():
            batch = self.model(self.batch["inputs"])["logits"]
            single = torch.cat([self.model(collate_samples([self.dataset[i % len(self.dataset)]])["inputs"])["logits"] for i in range(4)])
        torch.testing.assert_close(batch, single, rtol=1e-5,atol=1e-6)

    def test_all_modality_combinations_and_backward(self):
        for mode in ("concat","gated"):
            self.cfg.model.fusion = mode
            model = MultimodalRiskModel(14,2,32,self.cfg.model)
            for code in range(8):
                with self.subTest(mode=mode, code=code):
                    x = {k:v.clone() for k,v in self.batch["inputs"].items()}
                    available = torch.tensor([(code>>j)&1 for j in range(3)],dtype=torch.bool).expand(4,-1).clone()
                    x["available"] = available
                    if not (code & 4):
                        x["note_mask"][:] = False
                        x["chunk_mask"][:] = False
                    output = model(x)
                    self.assertEqual(output["logits"].shape,(4,))
                    self.assertTrue(torch.isfinite(output["logits"]).all())
                    self.assertEqual(bool(output["valid"].all()),code>0)
                    self.assertTrue((output["gate_weights"][~available] == 0).all())
                    if mode == "gated" and code:
                        torch.testing.assert_close(output["gate_weights"].sum(dim=1),torch.ones(4))
                    model.zero_grad()
                    output["logits"].sum().backward()
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_missing_branch_returns_zero(self):
        x = self.batch["inputs"]
        absent = torch.zeros(4,dtype=torch.bool)
        self.assertTrue((self.model.vital(x["vitals"],x["time_mask"],absent)==0).all())
        self.assertTrue((self.model.lab(x["labs"],absent)==0).all())
        self.assertTrue((self.model.note(x["note_chunks"],torch.zeros_like(x["chunk_mask"]),
                                       x["note_age"],torch.zeros_like(x["note_mask"]))==0).all())

    def test_modality_dropout_keeps_one_existing(self):
        fusion = self.model.fusion.train()
        fusion.drop_probability = .999999
        available = torch.tensor([[1,0,0],[0,1,1],[0,0,0]],dtype=torch.bool)
        for _ in range(50):
            used = fusion.drop_modalities(available)
            self.assertFalse((used & ~available).any())
            torch.testing.assert_close(used.any(dim=1),available.any(dim=1))
        fusion.eval()
        torch.testing.assert_close(fusion.drop_modalities(available),available)

    def test_all_masked_softmax_finite_gradient(self):
        x = torch.randn(3,4,requires_grad=True)
        y = masked_softmax(x,torch.zeros(3,4,dtype=torch.bool))
        y.sum().backward()
        self.assertTrue((y==0).all())
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_invalid_labels_filtered_before_loss(self):
        logits = torch.tensor([0.,float("nan"),0.],requires_grad=True)
        labels = torch.tensor([1.,-1.,float("nan")])
        loss = masked_bce(logits,labels,torch.tensor([True,False,False]))
        self.assertAlmostEqual(loss.item(),np.log(2),places=6)
        loss.backward()
        torch.testing.assert_close(logits.grad[1:],torch.zeros(2))

    def test_no_valid_labels_no_optimizer_update(self):
        trainer = Trainer(self.model,self.cfg.training,Path(self.temp.name))
        self.batch["label_mask"][:] = False
        before = deepcopy(self.model.state_dict())
        self.assertIsNone(trainer.train_batch(self.batch))
        self.assertEqual(len(trainer.optimizer.state),0)
        for key,value in self.model.state_dict().items():
            torch.testing.assert_close(value,before[key],rtol=0,atol=0)

    def test_single_class_weight_safe(self):
        self.assertIsNone(training_pos_weight(np.zeros(4),np.ones(4,dtype=bool)))
        self.assertEqual(training_pos_weight(np.array([0,0,1]),np.ones(3,dtype=bool)),2.)

    def test_extreme_finite_observations(self):
        x = {k:v.clone() for k,v in self.batch["inputs"].items()}
        x["labs"] *= 1e6
        x["vitals"] *= 1e6
        x["static"] *= 1e6
        self.assertTrue(torch.isfinite(self.model(x)["logits"]).all())

    def test_all_encoders_receive_gradients(self):
        self.model.train()
        self.model(self.batch["inputs"])["logits"].sum().backward()
        for module in (self.model.vital,self.model.lab,self.model.note,self.model.static):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters()))
