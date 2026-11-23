# Copyright (c) 2026, Oak Ridge National Laboratory.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Check the Fed-LIDC-IDRI datamodule: build this rank's loaders from
``conf/datamodule/flamby/lidc-idri-natural.yaml`` and draw one batch from each.

The data must be downloaded first (hours; resumable)::

    OMNIFED_FLAMBY_ACCEPT_LICENSE=1 python -m src.omnifed.data.skewed.flamby_download fed_lidc_idri

Then, from the repo root (DataLoader workers included, as in training)::

    OMNIFED_NUM_FEDERATED_CLIENTS=4 OMNIFED_FEDERATED_CLIENT_INDEX=0 \\
        python -m src.omnifed.data.skewed.check_lidc

It reports how many of FLamby's 1,009 scans are on disk per center (and any
failures listed in ``lidc_failed.txt``), then one train and one eval batch:
FLamby samples 128^3 patches per scan (train: 2 random, mostly around nodules;
eval: all patches tiling the 384^3 volume), so a batch of one scan is
``(patches, 1, 128, 128, 128)`` CT intensities in [0, 1] and a same-shaped
binary nodule mask.

The client count and this rank's client come from the env vars, as in a real
run (default 4 clients / client 0 when unset; ``-1`` = server).
``OMNIFED_DATA_DIR`` defaults to ``<repo>/datasets``. Extra arguments are Hydra
overrides, e.g.::

    python -m src.omnifed.data.skewed.check_lidc datamodule.num_workers=0
    python -m src.omnifed.data.skewed.check_lidc \\
        datamodule/flamby/scheme@datamodule.partitioner=pooled-dirichlet-quantity

Run it as a module or a file, not through stdin (``python - <<EOF``): on macOS
DataLoader workers re-import the main script, which needs a real file.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[4]  # skewed/ data/ omnifed/ src/ <repo>
CONF_DIR = REPO_ROOT / "conf"
CONFIG = "flamby/lidc-idri-natural"
DATASET = "fed_lidc_idri"
NUM_CLIENTS_ENV = "OMNIFED_NUM_FEDERATED_CLIENTS"
CLIENT_INDEX_ENV = "OMNIFED_FEDERATED_CLIENT_INDEX"
DATA_DIR_ENV = "OMNIFED_DATA_DIR"
PATCH = (1, 128, 128, 128)  # FLamby's Sampler default patch, with the channel axis from collate_fn


def _describe(batch: Any) -> str:
    """Shapes / dtypes of a (nested) batch."""
    if hasattr(batch, "shape") and hasattr(batch, "dtype"):
        return f"{tuple(batch.shape)} {str(batch.dtype).replace('torch.', '')}"
    if isinstance(batch, (list, tuple)):
        return "(" + ", ".join(_describe(b) for b in batch) + ")"
    return type(batch).__name__


def _centers(subset: Any) -> list:
    """Centers present in a Subset of a FlambyPooledDataset (or of the pooled set itself)."""
    import numpy as np

    pooled = getattr(subset, "dataset", subset)
    ids = np.asarray(pooled.center_ids)
    if hasattr(subset, "indices"):
        ids = ids[np.asarray(subset.indices, dtype=np.int64)]
    return sorted(set(ids.tolist()))


def _report_download() -> None:
    """Scans on disk vs FLamby's metadata, per center and split."""
    import csv
    from collections import Counter

    from . import flamby_download as fd

    root = fd.flamby_data_root() / DATASET
    print(f"[check_lidc] data: {root} (ready: {fd.is_flamby_downloaded(DATASET)})")
    meta = fd.flamby_package_dir() / "datasets" / DATASET / "metadata" / "metadata.csv"
    with open(meta, newline="") as f:  # all 1,018 LIDC CT series, minus the 9 FLamby leaves out
        rows = [r for r in csv.DictReader(f) if r["SeriesInstanceUID"] not in fd.LIDC_EXCLUDED]
    expected = Counter((r["Manufacturer"], r["Split"]) for r in rows)
    have = Counter(
        (r["Manufacturer"], r["Split"]) for r in rows if fd._lidc_converted(root / r["SeriesInstanceUID"])
    )
    for center in sorted({m for m, _ in expected}):
        cells = [f"{split} {have[(center, split)]}/{expected[(center, split)]}" for split in ("train", "test")]
        print(f"             center {center}: {', '.join(cells)}")
    print(f"             total {sum(have.values())}/{len(rows)} scans converted")
    failed = root / "lidc_failed.txt"
    if failed.is_file():
        lines = failed.read_text().splitlines()
        print(f"[check_lidc] {len(lines)} scans failed in the last download run (see {failed}); e.g.:")
        for line in lines[:3]:
            print(f"             {line[:160]}")


def _check_batch(split: str, x: Any, y: Any) -> None:
    import torch

    assert x.dim() == 5 and tuple(x.shape[1:]) == PATCH, f"expected (patches, {PATCH}), got {tuple(x.shape)}"
    assert tuple(y.shape) == tuple(x.shape), f"mask shape {tuple(y.shape)} != image shape {tuple(x.shape)}"
    values = torch.unique(y).tolist()
    assert set(values) <= {0, 1}, f"mask must be binary, has {values[:10]}"
    fg = float((y > 0).float().mean())
    print(f"       x in [{float(x.min()):.2f}, {float(x.max()):.2f}] (FLamby's ClipNorm: [0, 1]); "
          f"mask values {values}, nodule voxels {100 * fg:.3f}%"
          + ("" if fg > 0 or split == "eval" else "  (no nodule in these patches)"))


def main(overrides: Optional[Sequence[str]] = None) -> int:
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    overrides = list(sys.argv[1:] if overrides is None else overrides)
    for var, default in ((NUM_CLIENTS_ENV, "4"), (CLIENT_INDEX_ENV, "0"), (DATA_DIR_ENV, str(REPO_ROOT / "datasets"))):
        if not os.environ.get(var):
            os.environ[var] = default
            print(f"[check_lidc] {var} not set; using {default}")
    print(f"[check_lidc] client {os.environ[CLIENT_INDEX_ENV]} of {os.environ[NUM_CLIENTS_ENV]}; "
          f"config datamodule={CONFIG} {' '.join(overrides)}".rstrip())
    _report_download()

    with initialize_config_dir(config_dir=str(CONF_DIR), version_base=None):
        cfg = compose(config_name="base", overrides=[f"datamodule={CONFIG}", *overrides])
    t0 = time.perf_counter()
    dm = instantiate(cfg.datamodule)
    print(f"[check_lidc] datamodule built in {time.perf_counter() - t0:.1f}s")

    for split, loader in (("train", dm.train), ("eval", dm.eval)):
        if loader is None:
            print(f"{split:5s}: None (server)")
            continue
        print(f"{split:5s}: {type(loader).__name__}, {len(loader.dataset)} scans, centers {_centers(loader.dataset)}, "
              f"batch_size={loader.batch_size}, num_workers={loader.num_workers}, {len(loader)} batches")
        t0 = time.perf_counter()
        x, y = next(iter(loader))[:2]
        print(f"       first batch in {time.perf_counter() - t0:.1f}s: x {_describe(x)}, y {_describe(y)}")
        _check_batch(split, x, y)
    print("[check_lidc] OK")
    return 0


if __name__ == "__main__":  # required on macOS: DataLoader workers re-import this module
    sys.exit(main())
