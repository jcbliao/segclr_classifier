"""Experimental batch mixing for the native presynaptic comparison.

Repartition neighboring single-cell batches into mixed-cell batches. The
number of optimizer steps and the set of windows per epoch stay unchanged.
"""

from __future__ import annotations

from data.dataset_presynaptic import AttentionBudgetBatchSampler


class MixedCellBatchSampler(AttentionBudgetBatchSampler):
    def __init__(self, *args, cells_per_batch: int = 16, **kwargs):
        if cells_per_batch < 2:
            raise ValueError("cells_per_batch must be at least 2")
        super().__init__(*args, **kwargs)
        self.cells_per_batch = cells_per_batch

    def _build_plan(self) -> list[list[int]]:
        if self._plan is not None:
            return self._plan
        original = super()._build_plan()
        mixed = []
        for start in range(0, len(original), self.cells_per_batch):
            group = original[start:start + self.cells_per_batch]
            outputs = [[] for _ in group]
            offset = 0
            for source_batch in group:
                for index in source_batch:
                    outputs[offset % len(outputs)].append(index)
                    offset += 1
            for batch in outputs:
                if len(batch) > self.max_windows:
                    raise ValueError("mixed batch exceeds max_windows")
                largest = max(int(self.dataset.window_sizes[index]) for index in batch) + 1
                if len(batch) * largest**2 > self.attention_budget:
                    raise ValueError("mixed batch exceeds attention_budget")
            mixed.extend(outputs)
        self._plan = mixed
        return mixed
