"""
PyTorch Dataset backed by ExperimentStore.

Each sample is a window of length L from one (experiment, output_channel) pair.

Input:  matrix of shape (4, L) — rows are [x, |x|, x², x³] of the raw input slice.
Output: vector of shape (L,) — raw output slice for one channel.

The index space is: (experiment, output_channel, window_start).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from experiment_store import ExperimentStore


class ExperimentDataset(Dataset):
    """Sliding-window dataset over experiments stored in an ExperimentStore.

    Parameters
    ----------
    store : ExperimentStore
        Opened store with saved experiments.
    exp_ids : list[str] | None
        Which experiments to include.  None = all experiments in the store.
    window_len : int
        Length of each time window (L).  Default 5.
    stride : int
        Step between consecutive windows.  Default 1 (fully overlapping).
    """

    def __init__(
        self,
        store: ExperimentStore,
        exp_ids: list[str] | None = None,
        window_len: int = 5,
        stride: int = 1,
    ):
        self.store = store
        self.window_len = window_len
        self.stride = stride

        if exp_ids is None:
            exp_ids = store.list_experiments()

        # Build a flat index: list of (exp_id, channel_idx, num_windows)
        # so that __getitem__ can map a global int index to a specific sample.
        self._index: list[tuple[str, int, int]] = []  # (exp_id, ch, n_windows)
        self._cumulative: list[int] = []  # cumulative sample count
        total = 0

        # Cache metadata per experiment (avoids zarr reads in __getitem__)
        self._metadata: dict[str, dict] = {}

        catalog = store.get_catalog()
        for eid in exp_ids:
            row = catalog.loc[eid]
            T = int(row["output.T"])
            C_out = int(row["output.C"])
            n_windows = max(0, (T - window_len) // stride + 1)
            if n_windows == 0:
                continue
            if eid not in self._metadata:
                self._metadata[eid] = store.read_metadata(eid)
            for ch in range(C_out):
                self._index.append((eid, ch, n_windows))
                total += n_windows
                self._cumulative.append(total)

        self._len = total

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        if idx < 0:
            idx += self._len
        if idx < 0 or idx >= self._len:
            raise IndexError(f"index {idx} out of range for dataset of size {self._len}")

        # Binary search to find which (exp_id, channel) block this index falls into
        lo, hi = 0, len(self._cumulative) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._cumulative[mid] <= idx:
                lo = mid + 1
            else:
                hi = mid
        block_idx = lo

        exp_id, ch, n_windows = self._index[block_idx]
        local_idx = idx - (self._cumulative[block_idx - 1] if block_idx > 0 else 0)
        t_start = local_idx * self.stride

        t_slice = slice(t_start, t_start + self.window_len)

        # Read input (single channel → shape (L,)) and output channel
        inp = self.store.read_arrays(exp_id, array_names=["input"],
                                     time_slice=t_slice, channels=0)["input"]
        out = self.store.read_arrays(exp_id, array_names=["output"],
                                     time_slice=t_slice, channels=ch)["output"]

        # Build (4, L) feature matrix: [x, |x|, x², x³]
        inp_matrix = np.stack([inp, np.abs(inp), inp ** 2, inp ** 3], axis=0)

        meta = self._metadata[exp_id]
        meta_sample = {**meta, "_exp_id": exp_id, "_channel": ch, "_t_start": t_start}

        return torch.from_numpy(inp_matrix), torch.from_numpy(out.copy()), meta_sample
