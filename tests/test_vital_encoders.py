"""Causality, padding and the GRU-D missing-data recurrence."""
from copy import deepcopy
import unittest
import torch

from clinical_multimodal.config import ModelConfig
from clinical_multimodal.models.predictor import MultimodalRiskModel
from clinical_multimodal.models.vital_grud import GRUDCell, VitalGRUD


class VitalEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_encoders_causal_masked_and_differentiable(self):
        for name in ("tcn", "lstm", "gru", "grud"):
            with self.subTest(model=name):
                torch.manual_seed(37)
                cfg = ModelConfig(vital_encoder=name, embedding_dim=8, rnn_hidden=6,
                                  tcn_channels=6, tcn_blocks=3, max_time_steps=16, dropout=0)
                encoder = MultimodalRiskModel(14, 2, 16, cfg).vital.eval()
                x = torch.randn(3, 16, 20)
                x[..., 14:20] = (torch.rand(3, 16, 6) > .4).float()
                mask = torch.arange(16)[None] < torch.tensor([12, 8, 0])[:, None]
                available = torch.tensor([True, True, False])
                changed = x.clone()
                changed[~mask] = float("nan")
                a = encoder(x, mask, available)
                b = encoder(changed, mask, available)
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                self.assertTrue((a[2] == 0).all())
                self.assertTrue((encoder(x, mask, torch.zeros_like(available)) == 0).all())
                single = encoder(x[:1, :12], mask[:1, :12], available[:1])
                torch.testing.assert_close(a[:1], single, rtol=1e-5, atol=1e-6)
                changed = x.clone()
                changed[:, 6:, :14] += 100
                torch.testing.assert_close(encoder.sequence(x, mask)[:, :6],
                                           encoder.sequence(changed, mask)[:, :6], rtol=0, atol=0)
                encoder.train()
                encoder(x, mask, available).square().sum().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in encoder.parameters() if p.grad is not None))
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters()))

    def test_grud_decay_matches_paper_imputation(self):
        cell = GRUDCell(1, 1)
        with torch.no_grad():
            for parameter in cell.parameters():
                parameter.zero_()
            cell.input_decay_weight.fill_(.5)
            cell.candidate.weight[0, 0] = 1
        values, previous, h = torch.tensor([[999.]]), torch.tensor([[2.]]), torch.zeros(1, 1)
        delta = torch.tensor([[2.]])
        missing = torch.zeros(1, 1, dtype=torch.bool)
        result = cell(values, missing, delta, previous, h)
        expected = .5 * torch.tanh(torch.tensor([[2.]]) * torch.exp(torch.tensor(-1.)))
        torch.testing.assert_close(result, expected)
        observed = cell(torch.tensor([[3.]]), ~missing, delta, previous, h)
        torch.testing.assert_close(observed, .5 * torch.tanh(torch.tensor([[3.]])))
        # Longer gaps decay the hidden state even when all inputs are missing.
        with torch.no_grad():
            cell.candidate.weight.zero_()
            cell.hidden_decay.weight.fill_(.5)
        short = cell(values, missing, torch.ones(1, 1), previous, torch.ones(1, 1))
        long = cell(values, missing, torch.full((1, 1), 3.), previous, torch.ones(1, 1))
        self.assertTrue((long < short).all())

    def test_grud_elapsed_minutes_reset_after_actual_observation(self):
        cfg = ModelConfig(vital_encoder="grud", rnn_hidden=4)
        encoder = VitalGRUD(cfg, input_channels=20)
        x = torch.zeros(1, 5, 20)
        x[:, [0, 3], 14:20] = 1
        intervals = []
        handle = encoder.cell.register_forward_pre_hook(lambda module, args: intervals.append(args[2].detach().clone()))
        encoder.sequence(x, torch.ones(1, 5, dtype=torch.bool))
        handle.remove()
        torch.testing.assert_close(torch.stack(intervals)[:, 0, 0], torch.tensor([0., 1., 2., 3., 1.]))
        # A forward-filled value without an actual observation is ignored.
        changed = x.clone()
        changed[:, [1, 2, 4], :14] = 999
        torch.testing.assert_close(encoder.sequence(x, torch.ones(1, 5, dtype=torch.bool)),
                                   encoder.sequence(changed, torch.ones(1, 5, dtype=torch.bool)))

    def test_recurrent_masks_reject_internal_padding(self):
        cfg = ModelConfig(vital_encoder="gru")
        encoder = MultimodalRiskModel(14, 2, 16, cfg).vital
        with self.assertRaisesRegex(ValueError, "right padding"):
            encoder.sequence(torch.zeros(1, 3, 20), torch.tensor([[True, False, True]]))
