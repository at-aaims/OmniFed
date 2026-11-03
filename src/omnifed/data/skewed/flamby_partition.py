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
FLamby datasets as OmniFed federated datamodules: real cross-silo healthcare
data whose natural clients are the data-holding centers (Ogier du Terrail et
al., NeurIPS 2022 D&B, https://github.com/owkin/FLamby).

Two steps, two modules:

1. :mod:`.flamby_download` puts a dataset into ``$OMNIFED_DATA_DIR/flamby/<name>``
   if it is not there yet (the only code that downloads)::

       OMNIFED_FLAMBY_ACCEPT_LICENSE=1 python -m src.omnifed.data.skewed.flamby_download fed_ixi

2. This module loads downloaded data and partitions it for the rank, driven by
   ``conf/datamodule/flamby/flamby.yaml`` (dataset) and
   ``conf/datamodule/flamby/scheme/`` (partitioning scheme)::

       python main.py datamodule=flamby/flamby datamodule.dataset=fed_ixi
       python main.py datamodule=flamby/flamby datamodule.dataset=fed_heart_disease \\
           datamodule/flamby/scheme@datamodule.partitioner=pooled-dirichlet-label

   or a config file naming both (e.g. ``conf/datamodule/flamby/ixi-natural-dirichlet-quantity.yaml``).

The number of clients and this rank's client come only from
``$OMNIFED_NUM_FEDERATED_CLIENTS`` / ``$OMNIFED_FEDERATED_CLIENT_INDEX`` (set by
the runners; ``-1`` = server).

Pieces
------
* :class:`FlambyPooledDataset`: all centers' train (or test) data as one
  dataset, with ``center_ids`` per sample and cheap ``targets`` when the
  dataset has class labels (``has_class_labels``).
* :class:`FlambyPartition`: maps centers to FL clients by rank, including when
  the number of clients differs from the number of centers (below).
* :func:`build_flamby_datamodule`: Hydra target; this rank's loaders. Batch
  size, workers and FLamby's ``collate_fn`` default per dataset.

Schemes (``conf/datamodule/flamby/scheme/``)
---------------------------------------------
``natural`` (:class:`FlambyPartition`, default), ``natural-dirichlet-quantity`` /
``natural-dirichlet-label`` (natural, with shared centers split non-IID), or
``pooled-iid`` / ``pooled-dirichlet-quantity`` / ``pooled-dirichlet-label``
(ignore the centers and re-partition the pooled data with an OmniFed scheme).
Label schemes need class labels (Heart Disease, ISIC 2019, Camelyon16) and
refuse the others. Any OmniFed partitioner works as ``partitioner``.

Clients vs centers
------------------
Each client gets a list of centers; a center listed for several clients is
split among them. Automatically, with ``K`` clients and ``C`` centers:

* ``K == C``: client ``i`` gets center ``i`` (the FLamby benchmark setting).
* ``K < C`` (merge): every center goes to exactly one client.
  ``merge="balanced"`` (default) gives the largest remaining center to the
  currently smallest client, so client sizes end up close; ``"round_robin"``
  gives center ``c`` to client ``c % K``.
* ``K > C`` (split): every client gets one center; a center shared by several
  clients is split among them by ``within_center`` (any OmniFed index scheme,
  default IID). ``allocation="proportional"`` (default) gives each center at
  least one client plus extra clients in proportion to its size;
  ``"equal"`` spreads clients evenly. Clients are dealt to centers in
  round-robin order, so clients ``0..C-1`` always get centers ``0..C-1``.
* ``client_centers=[[0, 3], [1], [2, 4, 5]]``: explicit mapping, one list per
  client; overrides the above. Centers listed nowhere are left out.
* ``on_mismatch="error"``: refuse ``K != C`` (strict benchmark runs).

Evaluation uses FLamby's per-center test splits: each trainer is evaluated on
the test data of its own centers (``eval_scope="client"``, default) or on all
centers (``"pooled"``). The server keeps the pooled test set.

Compatibility: FLamby's 2024 code is run unmodified; :mod:`.flamby_compat`
restores ``np.NaN`` (NumPy 2) and MONAI's ``AddChannel`` (MONAI >= 1.3) before
it is imported.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import warnings
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from . import flamby_compat
from .base import SkewedPartitioner, build_skewed_datamodule, load_labels, require_class_labels
from .flamby_download import (
    DATA_DIR_ENV,
    FLAMBY_DATASETS,
    FlambySpec,
    _ensure_importable,
    _spec,
    flamby_dataset_dir,
    record_flamby_agreement,
)
from .label_distribution import IIDLabelDistribution
from .quantity_skew import apportion

__all__ = [
    "FLAMBY_DATASETS",
    "FlambyPooledDataset",
    "FlambyPartition",
    "plan_client_centers",
    "flamby_object",
    "build_flamby_datamodule",
]


# ════════════════════════════════════════════════════════════════════════════
# Importing FLamby
# ════════════════════════════════════════════════════════════════════════════


def flamby_object(path: str) -> Any:
    """
    Import ``"package.module.attr"`` from FLamby (submodule or installed), e.g.
    a dataset's ``collate_fn`` in a Hydra config::

        collate_fn:
          _target_: src.omnifed.data.skewed.flamby_partition.flamby_object
          path: flamby.datasets.fed_lidc_idri.collate_fn
    """
    module, _, attr = path.rpartition(".")
    _ensure_importable(module.split(".")[0])
    flamby_compat.apply()
    return getattr(importlib.import_module(module), attr)


def _import_flamby(spec: FlambySpec):
    _ensure_importable(spec.module.split(".")[0])
    flamby_compat.apply(spec.module)
    try:
        module = importlib.import_module(spec.module)
        flamby_compat.after_import(spec.module)
        return module
    except ImportError as e:
        raise ImportError(
            f"{spec.module} is not importable ({e}). FLamby comes from the submodule "
            "third_party/FLamby (or $OMNIFED_FLAMBY_DIR); install its "
            "light dependencies: pip install pandas scikit-learn matplotlib seaborn pyyaml "
            f"tqdm wget lifelines" + (f", plus for this dataset: pip install {spec.pip}" if spec.pip else "") + "."
        ) from e


def _check_load_licence(name: str, spec: FlambySpec) -> None:
    """
    TCGA-BRCA ships with FLamby, which asks for its licence on stdin every time
    the dataset is built until the agreement is recorded. Record it (with
    ``$OMNIFED_FLAMBY_ACCEPT_LICENSE=1``) or fail clearly instead of blocking.
    """
    if spec.bundled:
        record_flamby_agreement(name)


# ════════════════════════════════════════════════════════════════════════════
# Pooled dataset with center ids
# ════════════════════════════════════════════════════════════════════════════


def _cheap_labels(parts: Sequence[Dataset]) -> Optional[np.ndarray]:
    """Class labels from dataset attributes (no sample loading), or None."""
    out = []
    for part in parts:
        found = None
        for attr in ("features_labels", "targets", "labels", "target", "y"):
            val = getattr(part, attr, None)
            if val is None or callable(val):
                continue
            if hasattr(val, "detach"):
                val = val.detach().cpu().numpy()
            elif isinstance(val, (list, tuple)) and val and hasattr(val[0], "detach"):
                val = [v.detach().cpu().numpy() for v in val]  # e.g. FedHeartDisease.labels
            try:
                arr = np.asarray(val, dtype=float)
            except (TypeError, ValueError):
                continue
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr[:, 0]
            if arr.ndim == 1 and arr.size == len(part) and np.all(np.mod(arr, 1) == 0):
                found = arr.astype(np.int64)
                break
        if found is None:
            return None
        out.append(found)
    return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


class FlambyPooledDataset(Dataset):
    """
    A FLamby dataset's centers as one dataset (FLamby's train or test split).

    Attributes:
        center_ids: Center of each sample (``np.ndarray``).
        targets: Class labels when the dataset exposes them cheaply (e.g. Heart
            Disease), else zeros (segmentation, survival, ...), so label
            statistics are only meaningful for classification datasets.
        has_class_labels: Whether ``targets`` are real class labels; label
            schemes refuse datasets where it is ``False``.
        centers: The centers included, in order.

    Pickling (DataLoader workers under ``spawn``, the macOS default) sends only
    the constructor arguments; the worker rebuilds the FLamby datasets from
    disk, after the same import path and compatibility setup as here.

    Args:
        name: Key of :data:`FLAMBY_DATASETS`, e.g. ``"fed_heart_disease"``.
        train: FLamby's train split (``True``) or test split.
        centers: Subset of centers to load (default: all).
        data_dir: Data root (default ``$OMNIFED_DATA_DIR``); the dataset must
            already be in ``<data_dir>/flamby/<name>`` (:mod:`.flamby_download`),
            else ``FileNotFoundError`` with the download command.
        debug: FLamby's debug mode (small subset), where supported.
        **dataset_kwargs: Passed to the FLamby class (e.g. ``transform``).
    """

    def __init__(
        self,
        name: str,
        train: bool = True,
        centers: Optional[Sequence[int]] = None,
        data_dir: Optional[str] = None,
        debug: bool = False,
        **dataset_kwargs: Any,
    ) -> None:
        data_dir = data_dir or os.environ.get(DATA_DIR_ENV)
        self._args = dict(
            name=name, train=train, centers=centers, data_dir=data_dir, debug=debug, **dataset_kwargs,
        )
        spec = _spec(name)
        path = flamby_dataset_dir(name, data_dir)  # never downloads
        module = _import_flamby(spec)
        _check_load_licence(name, spec)
        cls = getattr(module, spec.cls)
        n = int(getattr(module, "NUM_CLIENTS", spec.num_centers))
        self.name, self.train = name, bool(train)
        self.centers = list(range(n)) if centers is None else [int(c) for c in centers]
        kwargs = dict(dataset_kwargs)
        if spec.takes_data_path and path is not None:
            kwargs["data_path"] = str(path)
        if debug:
            kwargs["debug"] = True
        self.parts = [cls(center=c, train=self.train, pooled=False, **kwargs) for c in self.centers]
        self._concat = ConcatDataset(self.parts)
        self.center_ids = np.concatenate(
            [np.full(len(p), c, dtype=np.int64) for c, p in zip(self.centers, self.parts)]
        )
        labels = _cheap_labels(self.parts)
        self.has_class_labels = labels is not None
        self.targets = labels if labels is not None else np.zeros(len(self.center_ids), dtype=np.int64)

    def __getstate__(self) -> Dict[str, Any]:
        return {"_args": self._args}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__init__(**state["_args"])

    def __len__(self) -> int:
        return len(self._concat)

    def __getitem__(self, i: int):
        return self._concat[i]

    def __repr__(self) -> str:
        split = "train" if self.train else "test"
        return f"FlambyPooledDataset({self.name}, {split}, centers={self.centers}, n={len(self)})"


# ════════════════════════════════════════════════════════════════════════════
# Centers -> clients
# ════════════════════════════════════════════════════════════════════════════


def plan_client_centers(
    center_sizes: Dict[int, int],
    num_clients: int,
    merge: str = "balanced",
    allocation: str = "proportional",
) -> List[List[int]]:
    """
    Which centers each client gets (see the module docstring for the rules).
    Deterministic: depends only on the center sizes and ``num_clients``.
    """
    centers = sorted(center_sizes)
    c, k = len(centers), int(num_clients)
    if k < 1 or c < 1:
        raise ValueError("need at least one client and one center")
    if k == c:
        return [[cid] for cid in centers]
    if k < c:  # merge centers into fewer clients
        plan: List[List[int]] = [[] for _ in range(k)]
        if merge == "round_robin":
            for j, cid in enumerate(centers):
                plan[j % k].append(cid)
        elif merge == "balanced":  # largest center first, to the smallest client
            totals = np.zeros(k, dtype=np.int64)
            for cid in sorted(centers, key=lambda x: (-center_sizes[x], x)):
                target = int(np.argmin(totals))
                plan[target].append(cid)
                totals[target] += center_sizes[cid]
            plan = [sorted(p) for p in plan]
        else:
            raise ValueError("merge must be 'balanced' or 'round_robin'")
        return plan
    # k > c: split centers among several clients
    if allocation == "proportional":
        extra = apportion([center_sizes[x] for x in centers], k - c)
        slots = {cid: 1 + int(e) for cid, e in zip(centers, extra)}
    elif allocation == "equal":
        slots = {cid: k // c + (1 if j < k % c else 0) for j, cid in enumerate(centers)}
    else:
        raise ValueError("allocation must be 'proportional' or 'equal'")
    plan, left = [], dict(slots)
    while len(plan) < k:  # deal clients to centers round-robin
        for cid in centers:
            if left[cid] > 0 and len(plan) < k:
                plan.append([cid])
                left[cid] -= 1
    return plan


class FlambyPartition(SkewedPartitioner):
    """
    Natural partition by data-holding center (for :class:`FlambyPooledDataset`,
    or any dataset exposing per-sample ``center_ids``).

    This rank's client index comes from ``$OMNIFED_FEDERATED_CLIENT_INDEX`` like
    every OmniFed partitioner; the number of clients from ``num_clients`` or
    ``$OMNIFED_NUM_FEDERATED_CLIENTS``.

    Args:
        client_centers: Explicit centers per client (overrides the automatic
            rules); a center listed for several clients is split among them.
        merge: ``K < C`` rule: ``"balanced"`` or ``"round_robin"``.
        allocation: ``K > C`` rule: ``"proportional"`` or ``"equal"``.
        within_center: Index scheme splitting a shared center among its clients
            (default ``IIDLabelDistribution()``; e.g. ``DirichletLabelSkew``).
        on_mismatch: ``"adapt"`` (default) or ``"error"`` when ``K != C``
            and no explicit mapping is given.
        eval_scope: ``"client"``: trainers evaluate on their own centers' test
            data; ``"pooled"``: on all centers.
        **kwargs: :class:`~.base.SkewedPartitioner` options (``num_clients``,
            ``seed``, ``min_samples_per_client``, ``stateful_loader``, ...).
    """

    def __init__(
        self,
        client_centers: Optional[Sequence[Sequence[int]]] = None,
        merge: str = "balanced",
        allocation: str = "proportional",
        within_center: Optional[SkewedPartitioner] = None,
        on_mismatch: str = "adapt",
        eval_scope: str = "client",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if merge not in ("balanced", "round_robin"):
            raise ValueError("merge must be 'balanced' or 'round_robin'")
        if allocation not in ("proportional", "equal"):
            raise ValueError("allocation must be 'proportional' or 'equal'")
        if on_mismatch not in ("adapt", "error"):
            raise ValueError("on_mismatch must be 'adapt' or 'error'")
        if eval_scope not in ("client", "pooled"):
            raise ValueError("eval_scope must be 'client' or 'pooled'")
        self.client_centers = (
            None if client_centers is None else tuple(tuple(int(c) for c in cs) for cs in client_centers)
        )
        self.merge, self.allocation = merge, allocation
        self.within_center = within_center if within_center is not None else IIDLabelDistribution()
        self.on_mismatch, self.eval_scope = on_mismatch, eval_scope
        self._plan: Optional[List[List[int]]] = None  # plan of the last partition
        self._center_sizes: Dict[int, int] = {}

    # ── mapping ────────────────────────────────────────────────────────────

    def plan(self, center_sizes: Dict[int, int], num_clients: int) -> List[List[int]]:
        """Centers of every client, for these center sizes and client count."""
        if self.client_centers is not None:
            if len(self.client_centers) != num_clients:
                raise ValueError(
                    f"client_centers has {len(self.client_centers)} entries but num_clients={num_clients}"
                )
            unknown = sorted({c for cs in self.client_centers for c in cs} - set(center_sizes))
            if unknown:
                raise ValueError(f"client_centers names unknown centers {unknown}; have {sorted(center_sizes)}")
            if any(len(cs) == 0 for cs in self.client_centers):
                raise ValueError("every client needs at least one center")
            unused = sorted(set(center_sizes) - {c for cs in self.client_centers for c in cs})
            if unused:
                warnings.warn(f"centers {unused} are assigned to no client and left out", stacklevel=3)
            return [sorted(set(cs)) for cs in self.client_centers]
        if num_clients != len(center_sizes) and self.on_mismatch == "error":
            raise ValueError(
                f"{num_clients} clients but {len(center_sizes)} centers; set on_mismatch='adapt' "
                "or give client_centers"
            )
        return plan_client_centers(center_sizes, num_clients, self.merge, self.allocation)

    def _partition(self, y, num_classes, num_clients, rng):
        raise TypeError("FlambyPartition needs a dataset with per-sample center_ids")

    def _partition_dataset(self, dataset, y, num_classes, num_clients, rng):
        if not hasattr(dataset, "center_ids"):
            raise TypeError(f"{type(dataset).__name__} has no center_ids; use FlambyPooledDataset")
        require_class_labels(self.within_center, dataset)
        centers = np.asarray(dataset.center_ids)
        sizes = {int(c): int(n) for c, n in zip(*np.unique(centers, return_counts=True))}
        plan = self.plan(sizes, num_clients)
        sharers: Dict[int, List[int]] = {}
        for client, cs in enumerate(plan):
            for c in cs:
                sharers.setdefault(c, []).append(client)

        parts: List[List[np.ndarray]] = [[] for _ in range(num_clients)]
        for c in sorted(sharers):
            idx = np.flatnonzero(centers == c)
            clients = sharers[c]
            if len(clients) == 1:
                parts[clients[0]].append(idx)
                continue
            _, y_local = np.unique(y[idx], return_inverse=True)
            sub = self.within_center._partition(
                y_local.reshape(-1), int(y_local.max()) + 1 if idx.size else 0, len(clients), rng
            )
            for client, s in zip(clients, sub):
                parts[client].append(idx[np.asarray(s, dtype=np.int64)])
        self._plan, self._center_sizes = plan, sizes
        return [np.concatenate(p) if p else np.empty(0, np.int64) for p in parts]

    # ── reporting / eval ───────────────────────────────────────────────────

    def _require_plan(self) -> List[List[int]]:
        if self._plan is None:
            raise RuntimeError("partition the dataset first (e.g. partition_info(dataset))")
        return self._plan

    def describe_client(self, client_idx: int, num_clients: int, classes=None) -> str:
        if self._plan is None:
            return ""
        plan = self._plan
        cs = plan[client_idx]
        shared = [c for c in cs if sum(c in p for p in plan) > 1]
        if shared:
            c = shared[0]
            sharing = [i for i, p in enumerate(plan) if c in p]
            return f"center {c} (part {sharing.index(client_idx) + 1}/{len(sharing)})"
        return f"centers={cs}" if len(cs) > 1 else f"center {cs[0]}"

    def _fingerprint_extra(self, num_clients: int, classes=None) -> str:
        return f"FlambyPartition|plan={self._plan}|within={self.within_center!r}"

    def client_eval_dataset(self, eval_dataset, client_idx, num_clients, classes):
        if self.eval_scope == "pooled" or not hasattr(eval_dataset, "center_ids"):
            return eval_dataset
        mine = self._require_plan()[client_idx]
        idx = np.flatnonzero(np.isin(np.asarray(eval_dataset.center_ids), mine))
        return Subset(eval_dataset, idx.tolist())

    def summary(self, dataset: Dataset, *, num_clients=None, labels=None) -> Dict[str, Any]:
        """Base summary plus ``client_centers`` and ``center_sizes``."""
        out = super().summary(dataset, num_clients=num_clients, labels=labels)
        if self._plan is None:  # summary came from cache built by another object
            y, _ = load_labels(dataset, labels, self.label_fn)
            self._partition_dataset(dataset, y, 0, self.resolve_num_clients(num_clients), np.random.default_rng(0))
        out["client_centers"] = self._plan
        out["center_sizes"] = self._center_sizes
        return out


# ════════════════════════════════════════════════════════════════════════════
# Hydra target
# ════════════════════════════════════════════════════════════════════════════


def build_flamby_datamodule(
    dataset: str,
    partitioner: Optional[SkewedPartitioner] = None,
    batch_size: Optional[int] = None,
    eval_batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    pin_memory: bool = False,
    data_dir: Optional[str] = None,
    centers: Optional[Sequence[int]] = None,
    debug: bool = False,
    dataset_kwargs: Optional[Dict[str, Any]] = None,
    num_federated_clients: Optional[int] = None,
    verbose: bool = True,
    **_: Any,
):
    """
    This rank's loaders for FLamby ``dataset`` (``conf/datamodule/flamby/flamby.yaml``).

    1. Load FLamby's per-center train and test splits from
       ``$OMNIFED_DATA_DIR/flamby/<dataset>`` and pool them. The data must be
       downloaded already (:mod:`.flamby_download`); this never downloads.
    2. Partition the train split with ``partitioner`` (default: natural
       :class:`FlambyPartition`, ``StatefulDataLoader``) for the client in
       ``$OMNIFED_FEDERATED_CLIENT_INDEX`` out of ``$OMNIFED_NUM_FEDERATED_CLIENTS``.

    The server (index -1) gets ``train=None`` and the pooled test set; it does
    not load the train split. ``batch_size`` / ``num_workers`` default to the
    dataset's FLamby benchmark values; FLamby's ``collate_fn`` is used where the
    dataset has one (LIDC-IDRI, Camelyon16).
    """
    spec = _spec(dataset)
    partitioner = partitioner if partitioner is not None else FlambyPartition(stateful_loader=True)
    num_clients = partitioner.resolve_num_clients(num_federated_clients)
    is_server = partitioner.resolve_client_idx(num_clients) is None

    workers = spec.num_workers if num_workers is None else int(num_workers)
    loader_kwargs: Dict[str, Any] = dict(num_workers=workers, pin_memory=pin_memory, persistent_workers=workers > 0)
    if spec.collate_fn:
        loader_kwargs["collate_fn"] = flamby_object(spec.collate_fn)
    flamby_dataset_dir(dataset, data_dir)  # fail early, with the download command, if missing
    ds_kwargs = dict(data_dir=data_dir, centers=centers, debug=debug, **(dataset_kwargs or {}))
    bs = int(batch_size or spec.batch_size)

    train = None
    if not is_server:
        train_ds = FlambyPooledDataset(dataset, train=True, **ds_kwargs)
        train = DataLoader(train_ds, batch_size=bs, shuffle=True, **loader_kwargs)
    eval_ds = FlambyPooledDataset(dataset, train=False, **ds_kwargs)
    eval_loader = DataLoader(eval_ds, batch_size=int(eval_batch_size or bs), shuffle=False, **loader_kwargs)
    if verbose:
        print(f"[flamby] {dataset}: {spec.task}; {spec.num_centers} centers -> {num_clients} clients", flush=True)
        if is_server:
            print(f"[datamodule] server: train=None, eval=pooled test set ({len(eval_ds)})", flush=True)
    return build_skewed_datamodule(
        train=train, eval=eval_loader, partitioner=partitioner,
        num_federated_clients=num_clients, verbose=verbose,
    )
