"""
Zarr v3 storage backend for experiment dataclasses.

Storage layout:
    data.zarr/
    ├── {exp_id}/              # one group per experiment
    │   ├── input              # array (T, C_in), chunks=(CHUNK_T, 1)
    │   ├── output             # array (T, C_out), chunks=(CHUNK_T, 1)
    │   └── .zattrs            # all non-array fields as JSON metadata
    ├── ...
    └── .zattrs                # cached flat catalog for filtering

Arrays are always stored as 2D (T, C) with chunks=(CHUNK_T, 1).
  - 1D arrays are reshaped to (T, 1) on save and can be squeezed on read.
  - Chunking along C=1 means each channel is independently readable.
  - Time-slicing reads only the chunks that overlap the requested window.
"""

from __future__ import annotations

import json
from dataclasses import fields, dataclass
from typing import Any

import numpy as np
import pandas as pd
import zarr

CHUNK_T = 10_000  # chunk size along the time axis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_json_safe(obj: Any) -> Any:
    """Recursively convert an object to JSON-serializable form."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    return obj


def _flatten_metadata(meta: dict, prefix: str = "") -> dict:
    """Flatten nested dicts with dot-separated keys.

    Only scalar (filterable) values are kept.
    Example: {"cond": {"temp": 300}} -> {"cond.temp": 300}
    """
    flat: dict[str, Any] = {}
    for k, v in meta.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten_metadata(v, prefix=f"{key}."))
        elif isinstance(v, (int, float, str, bool, type(None))):
            flat[key] = v
    return flat


def _is_array_field(val: Any) -> bool:
    """Decide whether a dataclass field should be stored as a zarr array."""
    return isinstance(val, np.ndarray)


# ---------------------------------------------------------------------------
# Main store
# ---------------------------------------------------------------------------

class ExperimentStore:
    """Read/write experiment dataclasses backed by a single zarr v3 store.

    Parameters
    ----------
    path : str
        Path to the zarr directory store (e.g. "data.zarr").
    chunk_t : int
        Chunk size along the time axis. Default 10 000.
    compressor : optional
        Zarr v3 codec for compression (e.g. zarr.codecs.BloscCodec()).
        None uses zarr defaults (no explicit compression).

    Usage
    -----
    >>> store = ExperimentStore("data.zarr")
    >>> store.save(exp_id="exp_000", instance=my_dataclass)
    >>> data = store.read_arrays("exp_000", time_slice=slice(0, 1000))
    >>> meta = store.read_metadata("exp_000")
    >>> ids  = store.query(experiment_type="A", temperature=(">", 300))
    """

    def __init__(self, path: str, chunk_t: int = CHUNK_T, compressor=None):
        self.path = path
        self.chunk_t = chunk_t
        self.compressor = compressor
        self.root = zarr.open_group(path, mode="a")
        self._catalog_cache: pd.DataFrame | None = None

    def __repr__(self) -> str:
        n = len(self.list_experiments())
        return f"ExperimentStore({self.path!r}, {n} experiments)"

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def _save_one(self, exp_id: str, instance) -> None:
        """Write a single experiment group (arrays + metadata)."""
        grp = self.root.create_group(exp_id, overwrite=True)

        metadata: dict[str, Any] = {}

        for f in fields(instance):
            val = getattr(instance, f.name)

            if _is_array_field(val):
                arr = val if val.ndim == 2 else val.reshape(-1, 1)
                kwargs: dict[str, Any] = dict(
                    name=f.name,
                    data=arr,
                    chunks=(self.chunk_t, 1),
                    overwrite=True,
                )
                if self.compressor is not None:
                    kwargs["compressors"] = self.compressor
                grp.create_array(**kwargs)
            else:
                metadata[f.name] = _to_json_safe(val)

        grp.attrs.update(metadata)

    def save(self, exp_id: str, instance) -> None:
        """Save (or overwrite) a single experiment from a dataclass instance.

        - np.ndarray fields  -> zarr arrays (2D, chunked along time)
        - everything else    -> group attrs (JSON metadata)
        """
        self._save_one(exp_id, instance)
        self._rebuild_catalog()

    def save_many(self, experiments: dict[str, Any]) -> None:
        """Save multiple experiments. Keys are exp_ids, values are dataclass instances."""
        for exp_id, instance in experiments.items():
            self._save_one(exp_id, instance)
        # rebuild catalog once after all writes
        self._rebuild_catalog()

    def delete(self, exp_id: str) -> None:
        """Delete an experiment from the store and update the catalog."""
        del self.root[exp_id]
        self._rebuild_catalog()

    # ------------------------------------------------------------------
    # Reading — single experiment
    # ------------------------------------------------------------------

    def read_metadata(self, exp_id: str) -> dict[str, Any]:
        """Return the full (possibly nested) metadata dict for one experiment."""
        return dict(self.root[exp_id].attrs)

    def read_arrays(
        self,
        exp_id: str,
        array_names: list[str] | None = None,
        time_slice: slice | None = None,
        channels: int | list[int] | None = None,
    ) -> dict[str, np.ndarray]:
        """Read array fields from one experiment with optional subwindowing.

        Parameters
        ----------
        array_names : list of field names to read. None = all arrays.
        time_slice  : e.g. slice(0, 1000) — reads only those time steps.
        channels    : int — returns 1D array (T,) for that channel.
                      list of ints — returns 2D array (T, len(channels)).
                      None — all channels, returns 2D (T, C).

        Returns
        -------
        dict mapping field name -> np.ndarray
        """
        grp = self.root[exp_id]
        t_sel = time_slice if time_slice is not None else slice(None)
        squeeze = isinstance(channels, int)

        if channels is not None:
            c_sel = [channels] if isinstance(channels, int) else channels
        else:
            c_sel = None

        names = array_names or self._array_names(grp)
        result: dict[str, np.ndarray] = {}
        for name in names:
            arr = grp[name]
            if c_sel is not None:
                data = arr[t_sel, c_sel]
                if squeeze:
                    data = data.squeeze(axis=1)
                result[name] = data
            else:
                result[name] = arr[t_sel, :]
        return result

    def read_experiment(
        self,
        exp_id: str,
        time_slice: slice | None = None,
        channels: int | list[int] | None = None,
    ) -> dict[str, Any]:
        """Read everything (arrays + metadata) for one experiment.

        Returns a dict with separate "metadata" and "arrays" keys to
        avoid collisions between metadata field names and array names.
        """
        return {
            "metadata": self.read_metadata(exp_id),
            "arrays": self.read_arrays(exp_id, time_slice=time_slice, channels=channels),
        }

    # ------------------------------------------------------------------
    # Reading — multiple experiments (for building PyTorch datasets)
    # ------------------------------------------------------------------

    def read_many(
        self,
        exp_ids: list[str],
        array_names: list[str] | None = None,
        time_slice: slice | None = None,
        channels: int | list[int] | None = None,
        include_metadata: bool = True,
    ) -> list[dict[str, Any]]:
        """Read several experiments at once. Convenient for dataset construction.

        Returns a list of dicts, one per experiment, each containing
        arrays (as np.ndarray) and optionally metadata.
        """
        results = []
        for eid in exp_ids:
            item: dict[str, Any] = {"_exp_id": eid}
            if include_metadata:
                item.update(self.read_metadata(eid))
            item.update(
                self.read_arrays(eid, array_names=array_names,
                                 time_slice=time_slice, channels=channels)
            )
            results.append(item)
        return results

    # ------------------------------------------------------------------
    # Catalog & filtering
    # ------------------------------------------------------------------

    def get_catalog(self) -> pd.DataFrame:
        """Load the flat metadata catalog as a DataFrame (cached).

        Nested metadata fields are flattened with dot notation,
        e.g. {"conditions": {"temp": 300}} -> column "conditions.temp".
        Only scalar (str/int/float/bool/None) fields are included.

        Also includes array shape info as `{array_name}.T` and
        `{array_name}.C` columns for convenience.
        """
        if self._catalog_cache is not None:
            return self._catalog_cache
        if "_catalog" not in dict(self.root.attrs):
            self._rebuild_catalog()
        cat = json.loads(self.root.attrs["_catalog"])
        self._catalog_cache = pd.DataFrame(cat).set_index("_exp_id")
        return self._catalog_cache

    def query(self, **filters) -> list[str]:
        """Return experiment ids matching all filters.

        Scalar filters:     query(experiment_type="A")
        Range filters:      query(temperature=(">", 300))
        Combine freely:     query(experiment_type="A", temperature=(">=", 250))

        Supported operators: ">", "<", ">=", "<=", "==", "!=".
        """
        df = self.get_catalog()
        mask = pd.Series(True, index=df.index)

        for key, val in filters.items():
            if key not in df.columns:
                raise KeyError(f"Unknown catalog field: {key!r}. "
                               f"Available: {list(df.columns)}")
            if isinstance(val, tuple) and len(val) == 2:
                op, v = val
                ops = {
                    ">":  lambda s, v=v: s > v,
                    "<":  lambda s, v=v: s < v,
                    ">=": lambda s, v=v: s >= v,
                    "<=": lambda s, v=v: s <= v,
                    "==": lambda s, v=v: s == v,
                    "!=": lambda s, v=v: s != v,
                }
                if op not in ops:
                    raise ValueError(f"Unsupported operator: {op!r}")
                mask &= ops[op](df[key])
            else:
                mask &= df[key] == val

        return df.index[mask].tolist()

    def list_experiments(self) -> list[str]:
        """Return all experiment ids in the store."""
        return [name for name, _ in self.root.members() if not name.startswith("_")]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _array_names(self, grp: zarr.Group) -> list[str]:
        """List zarr array members of a group."""
        return [name for name, node in grp.members() if isinstance(node, zarr.Array)]

    def _catalog_row(self, exp_id: str) -> dict[str, Any]:
        """Build a single flat catalog row for one experiment."""
        grp = self.root[exp_id]
        meta = dict(grp.attrs)
        flat = _flatten_metadata(meta)
        flat["_exp_id"] = exp_id

        for arr_name in self._array_names(grp):
            arr = grp[arr_name]
            flat[f"{arr_name}.T"] = arr.shape[0]
            flat[f"{arr_name}.C"] = arr.shape[1]

        return flat

    def _rebuild_catalog(self) -> None:
        """Scan all experiments and cache a flat metadata table in root attrs.

        Called automatically on save. For <100 experiments this is instant.
        """
        rows = [self._catalog_row(eid) for eid in self.list_experiments()]

        # build column-oriented dict for DataFrame reconstruction
        if rows:
            all_keys = sorted({k for r in rows for k in r})
            catalog = {k: [r.get(k) for r in rows] for k in all_keys}
        else:
            catalog = {}

        # store as JSON string in attrs (zarr attrs must be JSON-serializable)
        self.root.attrs["_catalog"] = json.dumps(catalog)
        self._catalog_cache = None


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    @dataclass
    class Experiment:
        name: str
        experiment_type: str
        sample_rate: float
        conditions: dict          # nested metadata example
        input: np.ndarray         # shape (T, C_in)  — time-series input
        output: np.ndarray        # shape (T, C_out) — time-series output

    # --- create some fake experiments ---
    store = ExperimentStore("demo_data.zarr")

    for i in range(5):
        T = np.random.randint(800_000, 1_200_000)
        C_in = np.random.randint(1, 4)
        C_out = 1
        exp = Experiment(
            name=f"run_{i:03d}",
            experiment_type=np.random.choice(["A", "B"]),
            sample_rate=1000.0 + i * 100,
            conditions={"temperature": 300 + i * 10, "pressure": 1.0},
            input=np.random.randn(T, C_in).astype(np.float32),
            output=np.random.randn(T, C_out).astype(np.float32),
        )
        store.save(exp_id=f"exp_{i:03d}", instance=exp)
        print(f"Saved exp_{i:03d}  T={T}  C_in={C_in}")

    # --- catalog & filtering ---
    print("\nCatalog:")
    print(store.get_catalog())

    type_a_ids = store.query(experiment_type="A")
    print(f"\nType A experiments: {type_a_ids}")

    warm_ids = store.query(**{"conditions.temperature": (">=", 320)})
    print(f"Warm experiments (temp >= 320): {warm_ids}")

    # --- read with subwindowing ---
    data = store.read_arrays("exp_000", time_slice=slice(0, 5000), channels=0)
    print(f"\nSubwindowed read shape: input={data['input'].shape}, output={data['output'].shape}")

    # --- read many for dataset construction ---
    batch = store.read_many(type_a_ids, time_slice=slice(0, 10_000))
    for item in batch:
        print(f"  {item['_exp_id']}: input={item['input'].shape}, type={item['experiment_type']}")
