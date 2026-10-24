"""Final-cohort regression tests: leakage, masks, streaming and artifact reload."""
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from clinical_multimodal.data.fixed_adapter import VITAL_ORDER
from clinical_multimodal.data.fixed_adapter import FixedCohortBundle
from clinical_multimodal.data.fixed_dataset import FixedPreparedDataset as PreparedDataset
from clinical_multimodal.data.fixed_dataset import ShardBatchSampler
from clinical_multimodal.preprocessing.fixed_numeric import FixedNumericPreprocessor as NumericPreprocessor
from tests.helpers import DemoTextBackend, small_config
from clinical_multimodal.preprocessing.text_cache import TextCache
from clinical_multimodal.config import Config, DataConfig, TextConfig, ModelConfig, TrainingConfig
from clinical_multimodal.pipelines.train import train_pipeline
from clinical_multimodal.inference.predictor import Predictor
from clinical_multimodal.io import write_json
from tests.helpers import TestWorkspace


from tests.fixed_fixture import make_fixed


class FixedCohortTests(unittest.TestCase):
    def setUp(self):
        self.temp=TestWorkspace()
        self.addCleanup(self.temp.cleanup)
        self.root=make_fixed(Path(self.temp.name)/"fixed")
        self.bundle=FixedCohortBundle(self.root)

    def mutate_shard(self, index, mutate):
        path=self.root/self.bundle.schema["shards"][index]["file"]
        with np.load(path) as z: a={k:z[k] for k in z.files}
        mutate(a)
        np.savez_compressed(path,**a)

    def test_training_only_statistics_and_units(self):
        p=NumericPreprocessor.fit(self.bundle,self.bundle.rows("train"))
        self.assertEqual(p.active_lab_features,[True,False])
        self.assertEqual(p.vital_mean[:4],[85.]*4)
        self.assertEqual(p.lab_mean[0],11.)
        self.assertNotEqual(self.bundle.contract.lab_names[0],self.bundle.contract.lab_names[1])
        self.mutate_shard(2,lambda a:a["lab_features"].fill(1e9))
        other=FixedCohortBundle(self.root)
        self.assertEqual(p.to_dict(),NumericPreprocessor.fit(other,other.rows("train")).to_dict())
        with self.assertRaisesRegex(ValueError,"training rows"):
            NumericPreprocessor.fit(other,other.rows("test"))

    def test_batch_order_duplicates_and_no_target_leakage(self):
        a,texts=self.bundle.materialize([5,0,5,2])
        self.assertEqual([len([s for s in row if s]) for row in texts],[2,0,2,0])
        np.testing.assert_array_equal(a["vital_features"][0],a["vital_features"][2])
        p=NumericPreprocessor.fit(self.bundle,self.bundle.rows("train"))
        x=p.transform_arrays(self.bundle,a)
        self.assertEqual(x["vitals"].shape,(4,120,20))
        self.assertFalse(any("event" in k or k=="y" for k in x))
        self.assertTrue(all(np.isfinite(v).all() for v in x.values()))
        self.assertEqual(len(self.bundle._shards),2)

    def test_future_note_and_misaligned_label_rejected(self):
        self.mutate_shard(0,lambda a:a["y"].fill(0))
        with self.assertRaisesRegex(ValueError,"label misalignment"):
            self.bundle.shard(0)
        self.mutate_shard(1,lambda a:a["prediction_second"].fill(1))
        with self.assertRaisesRegex(ValueError,"time misalignment"):
            self.bundle.shard(1)
        self.bundle.note_seconds[:]=2**40
        with self.assertRaisesRegex(ValueError,"Future document"):
            self.bundle.shard(2)

    def test_patient_leakage_rejected(self):
        path=self.root/"sample_index.csv.gz"
        m=pd.read_csv(path)
        m.loc[2,"patient_group_id"]="PAT_1"
        m.to_csv(path,index=False)
        with self.assertRaisesRegex(ValueError,"leakage"):
            FixedCohortBundle(self.root)

    def test_masks_and_unavailable_blood_pressure(self):
        p=NumericPreprocessor.fit(self.bundle,self.bundle.rows("train"))
        a,_=self.bundle.materialize([0])
        a["vital_features"][:,:,:4]=0
        a["vital_features"][:,:,4:8]=1
        a["vital_features"][:,:,[10,13]]=0
        a["vital_features"][:,:,[8,11]]=999  # Invalid blood pressure filler must disappear.
        a["vital_observed_mask"][:]=False
        a["lab_mask"][:]=False
        x=p.transform_arrays(self.bundle,a)
        self.assertFalse(x["available"].any())
        self.assertTrue((x["vitals"][:,:,[0,1,2,3,8,11]]==0).all())

    def test_sampler_and_artifact_end_to_end(self):
        p=NumericPreprocessor.fit(self.bundle,self.bundle.rows("train"))
        cache=TextCache(Path(self.temp.name)/"cache",DemoTextBackend(32))
        d=PreparedDataset(self.bundle,np.arange(6),p,cache,6)
        sampler=ShardBatchSampler(d,2,123)
        self.assertEqual(sorted(i for batch in sampler for i in batch),list(range(6)))
        cfg = small_config(self.temp.name)
        cfg.data.root = str(self.root)
        artifact=train_pipeline(cfg)
        predictor=Predictor(artifact)
        results=predictor.predict(self.bundle,rows=[5,0],batch_size=2)
        self.assertEqual([r["sample_id"] for r in results],["5","0"])
        self.assertTrue(all(r["risk"] is not None for r in results))
        self.assertEqual(predictor.preprocessor.to_dict(),p.to_dict())
        single=[predictor.predict(self.bundle,rows=[i])[0]["risk"] for i in [5,0]]
        np.testing.assert_allclose(single,[r["risk"] for r in results],rtol=1e-5,atol=1e-6)

    def test_inference_without_labels_and_all_horizons(self):
        for horizon in [1,2,3,6]:
            y,mask=self.bundle.labels([0,1],horizon)
            np.testing.assert_array_equal(y,[0,1])
            self.assertTrue(mask.all())
        path=self.root/"sample_index.csv.gz"
        frame=pd.read_csv(path).drop(columns=[f"label_{h}h" for h in [1,2,3,6]])
        frame.to_csv(path,index=False)
        self.mutate_shard(0,lambda a:a.pop("y"))
        b=FixedCohortBundle(self.root,require_targets=False)
        a,_=b.materialize([0,1])
        self.assertNotIn("y",a)
        self.assertNotIn("y",b.targets)

    def test_missing_horizon_and_split_file_rejected(self):
        with self.assertRaisesRegex(ValueError,"supports horizons"):
            self.bundle.labels([0],24)
        np.save(self.root/"splits/train_sample_ids.npy",np.array([0,2]))
        with self.assertRaisesRegex(ValueError,"IDs disagree"):
            FixedCohortBundle(self.root)
