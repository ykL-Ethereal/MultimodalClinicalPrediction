import json
from pathlib import Path
import numpy as np
import pandas as pd
from clinical_multimodal.data.fixed_adapter import VITAL_ORDER
from clinical_multimodal.io import write_json


def make_fixed(root):
    root.mkdir()
    (root / "tensors").mkdir()
    (root / "splits").mkdir()
    schema = dict(version="test-fixed", samples=6, admissions=3, patients=3, horizons_hours=[1,2,3,6],
        history_minutes=120, vital_feature_order=VITAL_ORDER, lab_feature_count=2,
        lab_channels=["latest_value","latest_minus_previous"], lab_lookback_hours=72,
        static_feature_order=["age_years","sex_female","sex_male"],
        normalization="none; fit on training patients after splitting", notes=2,
        shards=[dict(file=f"tensors/shard_{i:06d}.npz",start=2*i,rows=2) for i in range(3)])
    write_json(root / "dataset_schema.json", schema)
    pd.DataFrame(dict(code=["A","A"], unit=["mg/L","mmol/L"], feature_index=[0,1],name=["A","B"],
                      source_unit_spellings=["mg/L","mmol/L"])).to_csv(root / "lab_feature_catalog.csv",index=False)
    splits = ["train","validation","test"]
    frame = pd.DataFrame(dict(sample_id=np.arange(6),patient_group_id=np.repeat(["PAT_1","PAT_2","PAT_3"],2),
        encounter_id=np.repeat(["ENC_1","ENC_2","ENC_3"],2),
        prediction_time=["2026-01-01 02:00:00"]*6, split=np.repeat(splits,2),
        **{f"label_{h}h": [0,1]*3 for h in [1,2,3,6]}))
    frame.to_csv(root / "sample_index.csv.gz",index=False)
    frame[["patient_group_id","split"]].drop_duplicates().to_csv(root / "patient_splits.csv",index=False)
    second = int(pd.Timestamp("2026-01-01 02:00:00").timestamp())
    offsets = []
    with (root / "note_text.jsonl").open("wb") as f:
        for text in ["病情稳定。", "继续观察，记录生命体征。"]:
            offsets.append(f.tell())
            f.write((json.dumps(text,ensure_ascii=False)+"\n").encode("utf-8"))
    np.savez(root / "note_index.npz",offsets=np.array(offsets),seconds=np.array([second-3600,second-60]))
    for i,split in enumerate(splits):
        np.save(root / "splits" / f"{split}_sample_ids.npy",np.arange(2*i,2*i+2))
        v=np.zeros((2,120,14),np.float32)
        v[:,:,:4]=80+i*100; v[1,:,:4]+=10
        v[:,:,[8,11]]=110; v[:,:,[10,13]]=1
        observed=np.ones((2,120,6),bool)
        labs=np.ones((2,2,2),np.float32)*(10+i*100)
        labs[1]+=2
        labmask=np.ones_like(labs,bool)
        if i==0: labmask[:,1]=False  # Holdout-only code/unit must stay disabled.
        a=dict(vital_features=v,vital_observed_mask=observed,lab_features=labs,lab_mask=labmask,
            lab_age_hours=np.ones((2,2),np.float32),lab_interval_hours=np.ones((2,2),np.float32),
            static_features=np.array([[40,1,0],[60,0,1]],np.float32),static_mask=np.ones((2,3),bool),
            note_start=np.zeros(2,np.int64),note_end=np.array([0,2]),prediction_second=np.full(2,second,np.int64),
            y=np.array([[0]*4,[1]*4],np.int8))
        np.savez_compressed(root / schema["shards"][i]["file"],**a)
    return root


