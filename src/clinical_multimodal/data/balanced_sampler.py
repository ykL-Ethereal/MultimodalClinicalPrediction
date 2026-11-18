"""Patient and pre-event-time balanced neural training, with explicit exposure caps."""
import numpy as np
from torch.utils.data import Sampler


class UniformBudgetBatchSampler(Sampler):
    """Label-blind reference with the same optimizer-step budget as balanced sampling."""
    def __init__(self, dataset, batch_size, seed, steps):
        self.indices = np.flatnonzero(dataset.label_mask & dataset.available.any(axis=1))
        if not len(self.indices):
            raise ValueError("No eligible training windows")
        self.batch_size, self.seed, self.steps, self.epoch = batch_size, seed, steps, 0

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        queue = []
        for _ in range(self.steps):
            while len(queue) < self.batch_size:
                queue.extend(rng.permutation(self.indices).tolist())
            batch, queue = queue[-self.batch_size:], queue[:-self.batch_size]
            yield batch


class PatientTimeBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, seed, *, steps, positive_fraction, max_positive_repeats):
        self.dataset, self.batch_size, self.seed = dataset, batch_size, seed
        self.steps, self.max_repeats = steps, max_positive_repeats
        self.n_positive = max(1, round(batch_size * positive_fraction))
        if self.n_positive >= batch_size:
            raise ValueError("Balanced batch must include both classes")
        self.frame = (dataset.metadata if hasattr(dataset, "metadata") else
                      dataset.bundle.metadata.iloc[dataset.rows].reset_index(drop=True))
        if not self.frame.split.eq("train").all():
            raise ValueError("Balanced sampling is restricted to original training patients")
        eligible = dataset.label_mask & dataset.available.any(axis=1)
        self.groups = {}
        for label in (0, 1):
            selected = np.flatnonzero(eligible & (dataset.y == label))
            if not len(selected):
                raise ValueError("Balanced sampling needs both eligible classes")
            groups = {}
            for patient, positions in self.frame.iloc[selected].groupby("patient_id", sort=True).groups.items():
                groups[patient] = np.asarray(list(positions), dtype=np.int64)
            self.groups[label] = groups
        if sum(map(len, self.groups[1].values())) * self.max_repeats < self.steps * self.n_positive:
            raise ValueError("Positive exposure budget exceeds max_positive_repeats; reduce steps/fraction")
        self.epoch = 0
        self.audit = {}

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        counts = np.zeros(len(self.dataset), np.int64)
        queues, bins = {}, {}
        for label, groups in self.groups.items():
            for patient, indices in groups.items():
                if label and "time_to_event_minutes" in self.frame:
                    tte = self.frame.iloc[indices].time_to_event_minutes.to_numpy(float)
                    strata = np.where(np.isfinite(tte), np.searchsorted([60, 180], tte, side="left"), 3)
                else:
                    strata = np.zeros(len(indices), int)
                bins[label, patient] = {}
                for group in np.unique(strata):
                    members = indices[strata == group]
                    bins[label, patient][int(group)] = members
                    queues[label, patient, int(group)] = list(rng.permutation(members))
        patient_cycles = {0: [], 1: []}
        bin_cycles = {}

        def draw(label, used):
            if label == 0:
                cycle = patient_cycles[0]
                if not cycle:
                    cycle.extend(rng.permutation(list(self.groups[0])))
                # Pop O(1) in the common case; rotate already-used patients only.
                skipped = []
                while len(cycle) > 1 and cycle[-1] in used:
                    skipped.append(cycle.pop())
                patient = cycle.pop()
                if skipped:
                    cycle[:0] = skipped
                queue = queues[0, patient, 0]
                if not queue:
                    queue.extend(rng.permutation(self.groups[0][patient]))
                index = int(queue.pop())
                counts[index] += 1
                used.add(patient)
                return index
            eligible_patients = [p for p, ix in self.groups[label].items()
                                 if label == 0 or (counts[ix] < self.max_repeats).any()]
            if not eligible_patients:
                raise ValueError("Positive exposure cap exhausted")
            # Prefer distinct patients within a batch; only reuse after exhausting them.
            distinct = [p for p in eligible_patients if p not in used]
            allowed = set(distinct or eligible_patients)
            cycle = patient_cycles[label]
            if not any(p in allowed for p in cycle):
                cycle[:] = list(rng.permutation(eligible_patients))
            at = next(i for i, p in enumerate(cycle) if p in allowed)
            patient = cycle.pop(at)
            candidates = [b for b, ix in bins[label, patient].items()
                          if label == 0 or (counts[ix] < self.max_repeats).any()]
            key = (label, patient)
            bc = bin_cycles.setdefault(key, [])
            if not any(b in candidates for b in bc):
                bc[:] = list(rng.permutation(candidates))
            b = bc.pop(next(i for i, b in enumerate(bc) if b in candidates))
            queue = queues[label, patient, b]
            permitted = bins[label, patient][b]
            if label:
                permitted = permitted[counts[permitted] < self.max_repeats]
            queue[:] = [i for i in queue if label == 0 or counts[i] < self.max_repeats]
            if not queue:
                queue[:] = list(rng.permutation(permitted))
            index = int(queue.pop())
            counts[index] += 1
            used.add(patient)
            return index

        for _ in range(self.steps):
            used = set()
            batch = [draw(1, used) for _ in range(self.n_positive)]
            # Avoid positive-patient negatives when alternatives are available.
            batch += [draw(0, used) for _ in range(self.batch_size - self.n_positive)]
            rng.shuffle(batch)
            yield batch
        self.exposure_counts = counts
        patient_counts = {}
        for patient, ix in self.frame.groupby("patient_id").indices.items():
            patient_counts[str(patient)] = int(counts[ix].sum())
        self.audit = {"epoch": self.epoch, "draws": int(counts.sum()), "unique_windows": int((counts > 0).sum()),
                      "max_window_repeats": int(counts.max()),
                      "max_positive_repeats": int(counts[self.dataset.y == 1].max()),
                      "patient_draws": patient_counts,
                      "time_bins_minutes": [0, 60, 180, "horizon"],
                      "event_time_is_approximate_and_used_only_for_training_sampling": True}
