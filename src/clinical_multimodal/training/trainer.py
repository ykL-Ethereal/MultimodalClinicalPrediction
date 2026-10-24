from __future__ import annotations

from dataclasses import dataclass
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.collate import collate_samples, move_batch
from ..evaluation.calibration import sigmoid
from ..evaluation.metrics import binary_metrics
from ..io import write_json
from .losses import masked_bce, training_pos_weight


def seed_everything(seed, num_threads=4):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(num_threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def make_loader(dataset, batch_size, *, shuffle=False, seed=0):
    generator = torch.Generator().manual_seed(seed)
    from ..data.fixed_dataset import FixedPreparedDataset, ShardBatchSampler
    if shuffle and isinstance(dataset, FixedPreparedDataset):
        return DataLoader(dataset, batch_sampler=ShardBatchSampler(dataset, batch_size, seed),
                          num_workers=0, collate_fn=collate_samples, generator=generator)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      collate_fn=collate_samples, generator=generator)


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    logits, targets, masks, valid, metadata = [], [], [], [], []
    for raw in loader:
        batch = move_batch(raw, device)
        output = model(batch["inputs"])
        logits.extend(output["logits"].cpu().tolist())
        targets.extend(raw["target"].tolist())
        masks.extend(raw["label_mask"].tolist())
        valid.extend(output["valid"].cpu().tolist())
        metadata.extend(raw["metadata"])
    return {"logits": np.asarray(logits), "targets": np.asarray(targets),
            "label_mask": np.asarray(masks, dtype=bool), "valid": np.asarray(valid, dtype=bool),
            "metadata": metadata}


@dataclass
class TrainingResult:
    best_epoch: int
    history: list
    selection_metric: str
    pos_weight: float | None


class Trainer:
    def __init__(self, model, config, output_dir):
        self.config = config
        self.device = torch.device(config.device)
        self.model = model.to(self.device)
        self.output_dir = output_dir
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    def train_batch(self, batch, pos_weight=None):
        batch = move_batch(batch, self.device)
        selected = batch["label_mask"] & batch["inputs"]["available"].any(dim=1)
        if not selected.any():
            return None
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(batch["inputs"])
        loss = masked_bce(output["logits"], batch["target"], batch["label_mask"], valid=output["valid"], pos_weight=pos_weight)
        if loss is None:
            return None
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip, error_if_nonfinite=True)
        self.optimizer.step()
        return float(loss.detach()), int(selected.sum()), float(norm)

    def fit(self, train_dataset, validation_dataset):
        labels = train_dataset.y
        mask = train_dataset.label_mask & train_dataset.available.any(axis=1)
        if not mask.any():
            raise ValueError("No valid labelled training samples with a dynamic modality")
        weight = training_pos_weight(labels, mask) if self.config.class_weight else None
        pos_weight = torch.tensor(weight, device=self.device) if weight is not None else None
        val_mask = validation_dataset.label_mask & validation_dataset.available.any(axis=1)
        both_validation = len(np.unique(validation_dataset.y[val_mask])) == 2
        both_training = len(np.unique(labels[mask])) == 2
        if not (both_validation and both_training):
            raise ValueError("Training requires both classes in training and selection validation")
        metric = "auprc" if both_validation else "negative_log_loss" if val_mask.any() else "last_epoch_no_validation"
        train_loader = make_loader(train_dataset, self.config.batch_size, shuffle=True, seed=self.config.seed)
        val_loader = make_loader(validation_dataset, self.config.batch_size)
        best, best_epoch, stale, history = -float("inf"), -1, 0, []
        for epoch in range(self.config.epochs):
            total, count, skipped = 0., 0, 0
            for step_index, batch in enumerate(train_loader, 1):
                step = self.train_batch(batch, pos_weight)
                if step is None:
                    skipped += 1
                    continue
                loss, n, _ = step
                total += loss * n
                count += n
                if step_index % self.config.log_every_steps == 0:
                    print(f"epoch={epoch + 1} batch={step_index}/{len(train_loader)} "
                          f"samples={count} train_loss={total/count:.6f}", flush=True)
            if count == 0:
                raise ValueError("No optimizer updates in this epoch")
            predictions = collect_predictions(self.model, val_loader, self.device)
            eligible = predictions["label_mask"] & predictions["valid"]
            metrics = binary_metrics(predictions["targets"][eligible], sigmoid(predictions["logits"][eligible]), self.config.threshold)
            score = metrics["auprc"] if both_validation else -metrics["log_loss"] if metrics["log_loss"] is not None else float(epoch)
            record = {"epoch": epoch + 1, "train_loss": total / count, "trained_samples": count,
                      "skipped_batches": skipped, "validation": metrics, "selection_score": score}
            history.append(record)
            if score > best:
                best, best_epoch, stale = score, epoch + 1, 0
                torch.save(self.model.state_dict(), self.output_dir / "best.pt")
            else:
                stale += 1
            torch.save({"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                        "epoch": epoch + 1}, self.output_dir / "last.pt")
            write_json(self.output_dir / "history.json", history)
            print(f"epoch={epoch + 1} train_loss={total/count:.6f} validation_status={metrics['status']} selection={score:.6f}", flush=True)
            if stale >= self.config.patience:
                break
        self.model.load_state_dict(torch.load(self.output_dir / "best.pt", map_location=self.device, weights_only=True))
        self.model.eval()
        return TrainingResult(best_epoch, history, metric, weight)

