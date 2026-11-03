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
Check the Fed-ISIC2019 datamodule: build this rank's loaders from
``conf/datamodule/flamby/isic2019-natural.yaml`` and draw one batch from each.

The data must be downloaded first::

    OMNIFED_FLAMBY_ACCEPT_LICENSE=1 python -m src.omnifed.data.skewed.flamby_download fed_isic2019

Then, from the repo root (DataLoader workers included, as in training)::

    OMNIFED_NUM_FEDERATED_CLIENTS=6 OMNIFED_FEDERATED_CLIENT_INDEX=0 \\
        python -m src.omnifed.data.skewed.check_isic

The client count and this rank's client come from the env vars, as in a real
run (they default to 6 clients / client 0 here when unset; ``-1`` = server).
Extra arguments are Hydra overrides, e.g.::

    python -m src.omnifed.data.skewed.check_isic datamodule.num_workers=0
    python -m src.omnifed.data.skewed.check_isic \\
        datamodule/flamby/scheme@datamodule.partitioner=pooled-dirichlet-label

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
CONFIG = "flamby/isic2019-natural"
NUM_CLIENTS_ENV = "OMNIFED_NUM_FEDERATED_CLIENTS"
CLIENT_INDEX_ENV = "OMNIFED_FEDERATED_CLIENT_INDEX"

os.environ["OMNIFED_FLAMBY_ACCEPT_LICENSE"] = str(1)
os.environ["OMNIFED_NUM_FEDERATED_CLIENTS"] = str(6)
os.environ["OMNIFED_FEDERATED_CLIENT_INDEX"] = str(0)
os.environ["OMNIFED_DATA_DIR"] = str(REPO_ROOT / "datasets")  # <repo>/datasets


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


def main(overrides: Optional[Sequence[str]] = None) -> int:
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    overrides = list(sys.argv[1:] if overrides is None else overrides)
    for var, default in ((NUM_CLIENTS_ENV, "6"), (CLIENT_INDEX_ENV, "0")):
        if not os.environ.get(var):
            os.environ[var] = default
            print(f"[check_isic] {var} not set; using {default}")
    print(f"[check_isic] client {os.environ[CLIENT_INDEX_ENV]} of {os.environ[NUM_CLIENTS_ENV]}; "
          f"config datamodule={CONFIG} {' '.join(overrides)}".rstrip())

    with initialize_config_dir(config_dir=str(CONF_DIR), version_base=None):
        cfg = compose(config_name="base", overrides=[f"datamodule={CONFIG}", *overrides])
    t0 = time.perf_counter()
    dm = instantiate(cfg.datamodule)
    print(f"[check_isic] datamodule built in {time.perf_counter() - t0:.1f}s")

    for split, loader in (("train", dm.train), ("eval", dm.eval)):
        if loader is None:
            print(f"{split:5s}: None (server)")
            continue
        print(f"{split:5s}: {type(loader).__name__}, {len(loader.dataset)} images, centers {_centers(loader.dataset)}, "
              f"batch_size={loader.batch_size}, num_workers={loader.num_workers}, {len(loader)} batches")
        t0 = time.perf_counter()
        x, y = next(iter(loader))[:2]
        labels = y.flatten().tolist()
        print(f"       first batch in {time.perf_counter() - t0:.1f}s: x {_describe(x)}, y {_describe(y)}, "
              f"labels {sorted(set(labels))} (first 8: {labels[:8]})")
        assert tuple(x.shape[1:]) == (3, 200, 200), f"expected 3x200x200 images, got {tuple(x.shape)}"
        assert all(0 <= v < 8 for v in labels), "labels must be in 0..7"
    print("[check_isic] OK")
    return 0


if __name__ == "__main__":  # required on macOS: DataLoader workers re-import this module
    sys.exit(main())
