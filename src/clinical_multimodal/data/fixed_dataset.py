"""Batch materialization and bounded document embedding cache for sharded data."""
from collections import OrderedDict
import numpy as np
from torch.utils.data import Dataset, Sampler


class FixedPreparedDataset(Dataset):
    def __init__(self, bundle, rows, preprocessor, cache, horizon, *, with_targets=True, defer_numeric=False):
        self.bundle, self.rows, self.preprocessor, self.cache = bundle, np.asarray(rows, dtype=int), preprocessor, cache
        self.vital_only = getattr(preprocessor, "vital_only", False)
        self.text_dimension = 0 if self.vital_only else cache.backend.dimension
        self.defer_numeric = defer_numeric
        from .schema import Contract
        from .fixed_adapter import vital_contract
        Contract(**preprocessor.contract).assert_matches(vital_contract(bundle.schema) if self.vital_only else bundle.contract)
        self.encoded = OrderedDict()
        if with_targets:
            self.y, self.label_mask = bundle.labels(self.rows, horizon)
        else:
            self.y, self.label_mask = np.full(len(rows),-1,np.float32), np.zeros(len(rows),bool)

    @property
    def available(self):
        result = self.bundle.availability(self.rows).copy()
        if self.vital_only:
            result[:, 1:] = False
            return result
        # A source lab-only sample can become unavailable under the training feature policy.
        candidates = np.flatnonzero(result[:,1] & ~result[:,0] & ~result[:,2])
        active = np.asarray(self.preprocessor.active_lab_features, bool)
        for start in range(0, len(candidates), 256):
            pos = candidates[start:start+256]
            a = self.bundle.get_numeric(self.rows[pos])
            result[pos,1] = (a["lab_mask"][:,:,0] & active).any(axis=1)
        return result

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.__getitems__([index])[0]

    def __getitems__(self, indices):
        rows = self.rows[indices]
        if self.vital_only:
            a = self.bundle.get_numeric(rows)
            numeric = ({k: a[k] for k in ("vital_features", "vital_observed_mask")} if self.defer_numeric
                       else self.preprocessor.transform_arrays(self.bundle, a))
            samples = [{"inputs": {k: v[i] for k, v in numeric.items()}, "target": float(self.y[index]),
                        "label_mask": bool(self.label_mask[index]),
                        "metadata": self.bundle.metadata.iloc[rows[i]].to_dict()}
                       for i, index in enumerate(indices)]
            if self.defer_numeric:
                for sample in samples:
                    sample["numeric_preprocessor"] = self.preprocessor
            return samples
        a, texts = self.bundle.materialize(rows)
        if self.defer_numeric:
            keys = ("vital_features", "vital_observed_mask", "lab_features", "lab_mask",
                    "lab_age_hours", "lab_interval_hours", "static_features", "static_mask")
            numeric = {k: a[k] for k in keys}
            numeric.update(note_age=a["note_age_hours"], note_mask=a["note_doc_mask"])
        else:
            numeric = self.preprocessor.transform_arrays(self.bundle, a)
        samples = []
        for i, index in enumerate(indices):
            documents = []
            for j, present in enumerate(a["note_doc_mask"][i]):
                if not present:
                    documents.append(None)
                    continue
                note_id = int(a["note_start"][i] + j)
                if note_id not in self.encoded:
                    self.encoded[note_id] = self.cache.get(texts[i][j])
                self.encoded.move_to_end(note_id)
                documents.append(self.encoded[note_id])
                while len(self.encoded) > 256:
                    self.encoded.popitem(last=False)
            samples.append({"inputs": {k:v[i] for k,v in numeric.items()}, "documents": documents,
                "text_dimension": self.text_dimension, "target": float(self.y[index]),
                "label_mask": bool(self.label_mask[index]), "metadata": self.bundle.metadata.iloc[rows[i]].to_dict()})
            if self.defer_numeric:
                samples[-1]["numeric_preprocessor"] = self.preprocessor
        return samples


class ShardBatchSampler(Sampler):
    """Shuffle shard order and rows within shards, visiting every sample once per epoch."""
    def __init__(self, dataset, batch_size, seed):
        self.batch_size, self.rng = batch_size, np.random.default_rng(seed)
        ids = dataset.bundle.shard_ids(dataset.rows)
        order = np.argsort(ids, kind="stable")
        boundaries = np.r_[0, np.flatnonzero(np.diff(ids[order])) + 1, len(order)]
        self.groups = [order[a:b] for a,b in zip(boundaries[:-1], boundaries[1:]) if b>a]

    def __len__(self):
        return sum((len(g)+self.batch_size-1)//self.batch_size for g in self.groups)

    def __iter__(self):
        for group in self.rng.permutation(len(self.groups)):
            rows = self.rng.permutation(self.groups[group])
            for start in range(0,len(rows),self.batch_size):
                yield rows[start:start+self.batch_size].tolist()
