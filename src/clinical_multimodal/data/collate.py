import numpy as np
import torch


def collate_samples(samples):
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    x = {k: torch.from_numpy(np.stack([s["inputs"][k] for s in samples]))
         for k in samples[0]["inputs"] if k not in ("note_age", "note_mask")}
    if "documents" not in samples[0]:
        batch = {"inputs": x, "target": torch.tensor([s["target"] for s in samples], dtype=torch.float32),
                 "label_mask": torch.tensor([s["label_mask"] for s in samples], dtype=torch.bool),
                 "metadata": [s["metadata"] for s in samples]}
        if "numeric_preprocessor" in samples[0]:
            batch["numeric_preprocessor"] = samples[0]["numeric_preprocessor"]
        return batch
    d = max(1, max(len(s["documents"]) for s in samples))
    c = max(1, max((len(v) for s in samples for v in s["documents"] if v is not None), default=0))
    e = samples[0]["text_dimension"]
    chunks = torch.zeros(len(samples), d, c, e)
    mask = torch.zeros(len(samples), d, c, dtype=torch.bool)
    note_age = torch.zeros(len(samples), d)
    note_mask = torch.zeros(len(samples), d, dtype=torch.bool)
    for i, sample in enumerate(samples):
        for j, vectors in enumerate(sample["documents"]):
            if vectors is not None:
                chunks[i, j, :len(vectors)] = torch.from_numpy(vectors)
                mask[i, j, :len(vectors)] = True
                note_age[i, j] = float(sample["inputs"]["note_age"][j])
                note_mask[i, j] = True
    x.update(note_chunks=chunks, chunk_mask=mask, note_age=note_age, note_mask=note_mask)
    batch = {"inputs": x, "target": torch.tensor([s["target"] for s in samples], dtype=torch.float32),
            "label_mask": torch.tensor([s["label_mask"] for s in samples], dtype=torch.bool),
            "metadata": [s["metadata"] for s in samples]}
    if "numeric_preprocessor" in samples[0]:
        batch["numeric_preprocessor"] = samples[0]["numeric_preprocessor"]
    return batch


def move_batch(batch, device):
    non_blocking = torch.device(device).type == "cuda"
    inputs = {k: v.to(device, non_blocking=non_blocking) for k, v in batch["inputs"].items()}
    if "numeric_preprocessor" in batch:
        inputs = batch["numeric_preprocessor"].transform_tensors(inputs)
    return {**batch, "inputs": inputs,
            "target": batch["target"].to(device, non_blocking=non_blocking),
            "label_mask": batch["label_mask"].to(device, non_blocking=non_blocking)}

