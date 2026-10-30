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
Shared building blocks for the non-IID partitioners in ``omnifed.data.skewed``.

* Label reading: :func:`extract_labels`, :func:`load_labels`.
* Partition checks: :func:`validate_partition`.
* Loader construction: :func:`build_dataloader` (``torch`` ``DataLoader`` or
  ``torchdata`` ``StatefulDataLoader``), :func:`loader_kwargs_from`.
* Sampling: :func:`sample_dirichlet`.
* Reproducibility: :func:`resolve_partition_seed` (one seed shared by every
  scheme) and :func:`partition_fingerprint` (short ID to compare across ranks).
* :class:`SkewedPartitioner`: abstract base for every scheme. Subclasses
  implement only ``_partition``.
* :func:`build_skewed_datamodule`: Hydra entry point for any partitioner.

Contains no scheme logic, so label, quantity, feature, and concept-shift
partitioners all reuse it.
"""

from __future__ import annotations

import hashlib
import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset

from ..datamodule import DataModule
from ..federated_shards import (
    FEDERATED_CLIENT_INDEX_ENV,
    resolve_federated_client_index,
    resolve_num_federated_clients,
)
from ...utils import print

__all__ = [
    "SkewedPartitioner",
    "PartitionResult",
    "build_skewed_datamodule",
    "PARTITION_SEED_ENV",
    "DEFAULT_PARTITION_SEED",
    "resolve_partition_seed",
    "partition_fingerprint",
    "sample_dirichlet",
    "extract_labels",
    "load_labels",
    "validate_partition",
    "build_dataloader",
    "loader_kwargs_from",
]


# ════════════════════════════════════════════════════════════════════════════
# Label extraction
# ════════════════════════════════════════════════════════════════════════════

# Attribute names used by torchvision and common custom datasets, in priority order.
#   targets: MNIST, CIFAR, ImageFolder, ...   labels: SVHN, STL10
#   _labels: Flowers102, Food101, ...          y: Caltech101/256
_LABEL_ATTRS: Tuple[str, ...] = ("targets", "labels", "_labels", "y")
# Keys for dict-style samples / HuggingFace columns.
_LABEL_KEYS: Tuple[str, ...] = ("label", "labels", "target", "targets", "y")


def _as_1d_int_array(values: Any) -> Optional[np.ndarray]:
    """Convert a list / ndarray / tensor of scalar labels to 1-D int64, else None."""
    if hasattr(values, "detach"):  # torch.Tensor
        values = values.detach().cpu().numpy()
    try:
        arr = np.asarray(values)
    except Exception:
        return None
    if arr.ndim != 1 or arr.size == 0:
        return None
    if arr.dtype.kind in "iu":
        return arr.astype(np.int64, copy=False)
    if arr.dtype.kind == "f" and np.all(np.mod(arr, 1) == 0):
        return arr.astype(np.int64)
    if arr.dtype.kind in "OUS":  # string/object class names -> keep, mapped later
        return arr
    return None


def _label_of(sample: Any) -> Any:
    if isinstance(sample, Mapping):
        for key in _LABEL_KEYS:
            if key in sample:
                return _scalar(sample[key])
        raise KeyError(f"no label key among {_LABEL_KEYS} in sample keys {list(sample)}")
    if isinstance(sample, (tuple, list)) and len(sample) >= 2:
        return _scalar(sample[1])
    raise TypeError(
        f"cannot infer label from sample of type {type(sample).__name__}; "
        "pass `labels=` or `label_fn=`"
    )


def _scalar(x: Any) -> Any:
    return x.item() if hasattr(x, "item") else x


def extract_labels(
    dataset: Dataset, label_fn: Optional[Callable[[Any], Any]] = None
) -> np.ndarray:
    """
    Return the label of every sample in ``dataset`` as a 1-D array.

    Resolution order:
      1. ``label_fn(sample)`` for each sample, if given;
      2. ``Subset`` -> labels of the parent dataset gathered at ``subset.indices``;
      3. a label attribute (``targets``, ``labels``, ``_labels``, ``y``);
      4. a HuggingFace ``datasets.Dataset`` label column;
      5. fallback: iterate ``dataset[i]`` (slow: loads every sample) with a warning.
    """
    n = len(dataset)  # type: ignore[arg-type]
    if label_fn is not None:
        return np.asarray([_scalar(label_fn(dataset[i])) for i in range(n)])

    if isinstance(dataset, Subset):
        parent = extract_labels(dataset.dataset)
        return parent[np.asarray(dataset.indices, dtype=np.int64)]

    for attr in _LABEL_ATTRS:
        val = getattr(dataset, attr, None)
        if val is None or callable(val):
            continue
        arr = _as_1d_int_array(val)
        if arr is not None and arr.shape[0] == n:
            return arr

    column_names = getattr(dataset, "column_names", None)  # HF datasets
    if column_names:
        for key in _LABEL_KEYS:
            if key in column_names:
                arr = _as_1d_int_array(dataset[key])  # type: ignore[index]
                if arr is not None and arr.shape[0] == n:
                    return arr

    warnings.warn(
        f"{type(dataset).__name__} exposes no label attribute; iterating all {n} "
        "samples to read labels (this loads every sample). Pass `labels=` or "
        "`label_fn=` to avoid this.",
        stacklevel=2,
    )
    return np.asarray([_label_of(dataset[i]) for i in range(n)])


# ════════════════════════════════════════════════════════════════════════════
# DataLoader construction
# ════════════════════════════════════════════════════════════════════════════


def build_dataloader(dataset: Dataset, *, stateful: bool = False, **kwargs: Any) -> DataLoader:
    """
    Build a ``torch.utils.data.DataLoader`` or, if ``stateful``, a
    ``torchdata.stateful_dataloader.StatefulDataLoader`` (a ``DataLoader`` subclass
    with ``state_dict()`` / ``load_state_dict()`` for mid-epoch resumption).

    Worker-only options are dropped when ``num_workers == 0`` because
    ``DataLoader`` rejects them in that case.
    """
    kwargs = dict(kwargs)
    if int(kwargs.get("num_workers", 0) or 0) == 0:
        kwargs.pop("persistent_workers", None)
        kwargs.pop("prefetch_factor", None)
    if stateful:
        try:
            from torchdata.stateful_dataloader import StatefulDataLoader
        except ImportError as e:  # pragma: no cover - depends on env
            raise ImportError(
                "stateful_loader=True requires torchdata>=0.8 "
                "(torchdata.stateful_dataloader.StatefulDataLoader)"
            ) from e
        return StatefulDataLoader(dataset, **kwargs)
    return DataLoader(dataset, **kwargs)


def loader_kwargs_from(loader: DataLoader) -> Dict[str, Any]:
    """Recover constructor kwargs from an existing loader (e.g. one Hydra built)."""
    batch_size = loader.batch_size
    if batch_size is None and loader.batch_sampler is not None:
        batch_size = getattr(loader.batch_sampler, "batch_size", None)
        warnings.warn(
            "template loader uses a custom batch_sampler; only its batch_size "
            f"({batch_size}) is carried over",
            stacklevel=2,
        )
    kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": isinstance(loader.sampler, RandomSampler),
        "num_workers": loader.num_workers,
        "pin_memory": loader.pin_memory,
        "drop_last": loader.drop_last,
        "collate_fn": loader.collate_fn,
        "timeout": loader.timeout,
        "worker_init_fn": loader.worker_init_fn,
    }
    if int(loader.num_workers) > 0:
        kwargs["persistent_workers"] = loader.persistent_workers
        kwargs["prefetch_factor"] = loader.prefetch_factor
    return kwargs


def load_labels(
    dataset: Dataset,
    labels: Optional[Sequence[Any]] = None,
    label_fn: Optional[Callable[[Any], Any]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Labels of ``dataset`` encoded as contiguous ints.

    Args:
        dataset: Dataset to read labels from (see :func:`extract_labels`).
        labels: Explicit per-sample labels; skips extraction when given.
        label_fn: ``sample -> label`` used by :func:`extract_labels`.

    Returns:
        ``(y, classes)`` where ``y[j]`` in ``[0, len(classes))`` is the label id of
        sample ``j`` and ``classes[y[j]]`` is its original label value.
    """
    raw = np.asarray(labels) if labels is not None else extract_labels(dataset, label_fn)
    n = len(dataset)  # type: ignore[arg-type]
    if raw.shape != (n,):
        raise ValueError(f"expected {n} labels, got shape {raw.shape}")
    classes, y = np.unique(raw, return_inverse=True)
    return y.astype(np.int64).reshape(-1), classes


# ════════════════════════════════════════════════════════════════════════════
# Partition validation
# ════════════════════════════════════════════════════════════════════════════


def validate_partition(
    parts: Sequence[Any],
    num_samples: int,
    num_clients: int,
    *,
    min_samples_per_client: int = 0,
    name: str = "partitioner",
) -> List[np.ndarray]:
    """
    Check a partition and return it normalized to sorted 1-D int64 arrays.

    Raises ``RuntimeError`` for bugs in a scheme (wrong number of parts,
    out-of-range or duplicated indices) and ``ValueError`` when a client has
    fewer than ``min_samples_per_client`` samples (a configuration problem).
    Samples left unassigned are allowed (e.g. long-tail subsampling).
    """
    out = [np.sort(np.asarray(p, dtype=np.int64).reshape(-1)) for p in parts]
    if len(out) != num_clients:
        raise RuntimeError(f"{name} returned {len(out)} parts for {num_clients} clients")
    flat = np.concatenate(out) if out else np.empty(0, dtype=np.int64)
    if flat.size and (flat.min() < 0 or flat.max() >= num_samples):
        raise RuntimeError(f"{name} produced out-of-range indices")
    if np.unique(flat).size != flat.size:
        raise RuntimeError(f"{name} assigned a sample to multiple clients")
    small = [i for i, p in enumerate(out) if p.size < min_samples_per_client]
    if small:
        raise ValueError(
            f"{name}: clients {small} have fewer than "
            f"min_samples_per_client={min_samples_per_client} samples "
            f"(sizes={[out[i].size for i in small]}). Use fewer clients, a less "
            "skewed setting, or lower min_samples_per_client."
        )
    return out


# ════════════════════════════════════════════════════════════════════════════
# Seeding and fingerprints
# ════════════════════════════════════════════════════════════════════════════

#: Env var holding the partition seed shared by every scheme. Set it once per job
#: (it must be identical on every rank) to re-draw all partitions.
PARTITION_SEED_ENV = "OMNIFED_PARTITION_SEED"
#: Seed used when neither ``seed=`` nor ``$OMNIFED_PARTITION_SEED`` is given.
DEFAULT_PARTITION_SEED = 0


def resolve_partition_seed(seed: Optional[int] = None) -> int:
    """Seed precedence: explicit ``seed`` > ``$OMNIFED_PARTITION_SEED`` > 0."""
    if seed is not None:
        return int(seed)
    raw = os.environ.get(PARTITION_SEED_ENV)
    if raw not in (None, ""):
        return int(raw)
    return DEFAULT_PARTITION_SEED


def partition_fingerprint(
    parts: Sequence[np.ndarray],
    y: np.ndarray,
    classes: np.ndarray,
    length: int = 12,
    extra: str = "",
) -> str:
    """
    Short hex ID of a global partition. Identical on two ranks iff they
    produced the same per-client indices from the same label sequence.

    The labels are hashed too: a pure index split (e.g. IID) would otherwise
    match across ranks whose datasets list samples in different orders.
    ``extra`` covers anything else that defines a client's data, such as a
    feature-skew scheme's per-client transforms. Empty by default, which
    leaves IDs of schemes that do not use it unchanged.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(np.int64(len(parts)).astype("<i8").tobytes())
    h.update(repr(np.asarray(classes).tolist()).encode())
    h.update(np.ascontiguousarray(y, dtype="<i8").tobytes())
    for p in parts:
        p = np.ascontiguousarray(p, dtype="<i8")
        h.update(np.int64(p.size).astype("<i8").tobytes())
        h.update(p.tobytes())
    if extra:
        h.update(b"|extra|" + extra.encode())
    return h.hexdigest()[:length]


def sample_dirichlet(rng: np.random.Generator, alpha: np.ndarray) -> np.ndarray:
    """``rng.dirichlet`` guarded against NaN/degenerate draws at tiny ``alpha``."""
    p = rng.dirichlet(alpha)
    if not np.all(np.isfinite(p)) or p.sum() <= 0:
        p = np.zeros_like(alpha, dtype=float)
        p[rng.choice(alpha.size, p=alpha / alpha.sum())] = 1.0
    return p


# ════════════════════════════════════════════════════════════════════════════
# Base partitioner
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class PartitionResult:
    """
    What a rank retains after partitioning: its own indices plus small global
    metadata. The full per-client index lists are discarded after computing.
    """

    partition_id: str  # fingerprint of the global partition; same on every rank
    seed: int  # resolved seed that produced it
    num_samples: int  # len(dataset)
    classes: np.ndarray  # original label value of each contiguous class id
    class_counts: np.ndarray  # [num_clients, num_classes] for all clients
    client_idx: Optional[int]  # client whose indices are kept (None: stats only)
    indices: Optional[np.ndarray]  # sorted int64 dataset indices of that client

    @property
    def num_clients(self) -> int:
        return int(self.class_counts.shape[0])


_STATS_ONLY: Any = object()  # sentinel: caller needs metadata, not a client's indices


class SkewedPartitioner(ABC):
    """
    Base class for all index-based non-IID partitioners (label, quantity,
    feature, concept-shift skew).

    Subclasses implement only :meth:`_partition`. Everything else is shared:
    label reading, seeding, validation, fingerprinting, caching, per-client
    ``Subset`` / loader construction, and heterogeneity statistics. Schemes that
    change the samples themselves (feature or concept skew) also override
    :meth:`client_dataset` to wrap the client's ``Subset``.

    Typical use::

        part = DirichletLabelSkew(alpha=0.1, num_clients=8)
        loader = part.client_loader(train_set, batch_size=64)  # this rank's client
        part.partition_id(train_set)   # e.g. '3f9c0a1b7e42', same on every rank
        part.summary(train_set)        # sizes, class coverage, Hellinger

    **Determinism.** All randomness comes from one ``numpy`` Generator seeded with
    :func:`resolve_partition_seed` (``seed`` > ``$OMNIFED_PARTITION_SEED`` > 0),
    independent of the global ``random`` / ``np.random`` / ``torch`` state. Every
    rank computes the same global partition and keeps only its own slice, so
    ranks must agree on the seed, ``num_clients``, the dataset's sample order,
    and the ``numpy`` version. Compare :meth:`partition_id` across ranks to
    confirm they did.

    **Which client am I?** Only ``$OMNIFED_FEDERATED_CLIENT_INDEX`` decides
    (:meth:`resolve_client_idx`), the same rule as
    ``build_federated_vision_datamodule``. The Slurm/Ray runners set it before
    the datamodule is instantiated: ``0..num_clients-1`` for trainers, ``-1``
    for the server. It is never part of a config, so every rank of a job can
    share one config yet take a different slice. Outside the runners
    (notebooks, tests) set the env var, or use :meth:`partition` to see all
    clients.

    **Memory.** The full partition exists only transiently while it is
    computed (a few int64 arrays of ``len(dataset)``; ~60 MB peak for
    ImageNet-1k's 1.28M samples). Afterwards only this client's indices, the
    ``[num_clients, num_classes]`` count table, and the fingerprint are cached.

    Args:
        num_clients: Number of FL clients. ``None`` reads
            ``OMNIFED_NUM_FEDERATED_CLIENTS`` (set by the runners).
        seed: Partition seed. ``None`` uses the shared seed (see above).
        min_samples_per_client: Raise if any client receives fewer samples.
        stateful_loader: Return ``torchdata`` ``StatefulDataLoader`` instead of
            ``torch.utils.data.DataLoader`` from :meth:`client_loader`.
        loader_kwargs: Default ``DataLoader`` kwargs (``batch_size``, ...).
        label_fn: Optional ``sample -> label`` used when the dataset has no
            label attribute.
    """

    def __init__(
        self,
        num_clients: Optional[int] = None,
        seed: Optional[int] = None,
        min_samples_per_client: int = 1,
        stateful_loader: bool = False,
        loader_kwargs: Optional[Mapping[str, Any]] = None,
        label_fn: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        if num_clients is not None and int(num_clients) < 1:
            raise ValueError(f"num_clients must be >= 1, got {num_clients}")
        if int(min_samples_per_client) < 0:
            raise ValueError("min_samples_per_client must be >= 0")
        self.num_clients = None if num_clients is None else int(num_clients)
        self.seed = None if seed is None else int(seed)
        self.min_samples_per_client = int(min_samples_per_client)
        self.stateful_loader = bool(stateful_loader)
        self.loader_kwargs: Dict[str, Any] = dict(loader_kwargs or {})
        self.label_fn = label_fn
        # id(dataset) -> (dataset, PartitionResult). Holding the dataset reference
        # keeps its id from being reused while cached.
        self._cache: Dict[int, Tuple[Any, PartitionResult]] = {}

    # ── the one method subclasses implement ────────────────────────────────

    @abstractmethod
    def _partition(
        self,
        y: np.ndarray,
        num_classes: int,
        num_clients: int,
        rng: np.random.Generator,
    ) -> List[np.ndarray]:
        """
        Split samples among clients.

        Args:
            y: Labels as contiguous ints in ``[0, num_classes)``; ``y[j]`` is the
                label of dataset index ``j``.
            num_classes: Number of distinct labels.
            num_clients: Number of clients.
            rng: Seeded generator. Use it for *all* randomness (never the global
                ``np.random`` / ``random`` / ``torch`` RNGs) so the result is
                reproducible and identical on every rank.

        Returns:
            ``num_clients`` disjoint int arrays of dataset indices, one per
            client. Samples may be left unassigned (e.g. long-tail subsampling)
            but never duplicated.
        """

    # ── resolution of seed / client count / client id ──────────────────────

    def resolve_seed(self) -> int:
        return resolve_partition_seed(self.seed)

    def resolve_num_clients(self, num_clients: Optional[int] = None) -> int:
        if num_clients is not None:
            return int(num_clients)
        if self.num_clients is not None:
            return self.num_clients
        return resolve_num_federated_clients(None)

    def resolve_client_idx(self, num_clients: int) -> Optional[int]:
        """
        This rank's client index from ``$OMNIFED_FEDERATED_CLIENT_INDEX``, or
        ``None`` for the server (index -1, no partition). Raises if the variable
        is unset with ``num_clients > 1`` or out of range.
        """
        return resolve_federated_client_index(num_clients, env_var=FEDERATED_CLIENT_INDEX_ENV)

    # ── optional hooks for schemes that change samples, not just indices ──

    def describe_client(self, client_idx: int, num_clients: int) -> str:
        """
        Short description of what this scheme does to one client's samples
        (e.g. ``"rotate(90)"``). Empty for index-only schemes. Logged by the
        builder and reported by :meth:`summary`.
        """
        return ""

    def _fingerprint_extra(self, num_clients: int) -> str:
        """Extra text folded into :meth:`partition_id` (see ``partition_fingerprint``)."""
        return ""

    # ── public API ─────────────────────────────────────────────────────────

    def partition(
        self,
        dataset: Dataset,
        *,
        num_clients: Optional[int] = None,
        labels: Optional[Sequence[Any]] = None,
    ) -> List[np.ndarray]:
        """
        Index arrays for *all* clients. For analysis/debugging only: not cached
        and O(len(dataset)) memory. Training code uses :meth:`client_indices`.
        """
        parts, _, _ = self._compute(dataset, self.resolve_num_clients(num_clients), labels)
        return parts

    def partition_info(
        self,
        dataset: Dataset,
        *,
        num_clients: Optional[int] = None,
        labels: Optional[Sequence[Any]] = None,
    ) -> PartitionResult:
        """Cached metadata: fingerprint, seed, class counts for every client."""
        return self._result(dataset, num_clients, labels, _STATS_ONLY)

    def partition_id(self, dataset: Dataset, **kwargs: Any) -> str:
        """Fingerprint of the global partition. Compare it across ranks."""
        return self.partition_info(dataset, **kwargs).partition_id

    def client_indices(
        self,
        dataset: Dataset,
        *,
        num_clients: Optional[int] = None,
        labels: Optional[Sequence[Any]] = None,
    ) -> np.ndarray:
        """Sorted dataset indices of this rank's client (see :meth:`resolve_client_idx`)."""
        k = self.resolve_num_clients(num_clients)
        cid = self.resolve_client_idx(k)
        if cid is None:
            raise ValueError(
                f"this rank is the server ({FEDERATED_CLIENT_INDEX_ENV}=-1) and has no partition"
            )
        return self._result(dataset, k, labels, cid).indices  # type: ignore[return-value]

    def client_dataset(
        self,
        dataset: Dataset,
        *,
        num_clients: Optional[int] = None,
        labels: Optional[Sequence[Any]] = None,
    ) -> Dataset:
        """The client's local dataset. Override to wrap samples (e.g. feature skew)."""
        idx = self.client_indices(dataset, num_clients=num_clients, labels=labels)
        return Subset(dataset, idx.tolist())

    def client_loader(
        self,
        dataset: Dataset,
        *,
        template: Optional[DataLoader] = None,
        num_clients: Optional[int] = None,
        labels: Optional[Sequence[Any]] = None,
        **overrides: Any,
    ) -> DataLoader:
        """
        Loader over the client's partition.

        Kwargs precedence (low -> high): ``template`` loader settings,
        ``self.loader_kwargs``, ``overrides``. ``shuffle`` defaults to True.
        """
        kwargs: Dict[str, Any] = {"shuffle": True}
        if template is not None:
            kwargs.update(loader_kwargs_from(template))
        kwargs.update(self.loader_kwargs)
        kwargs.update(overrides)
        local = self.client_dataset(dataset, num_clients=num_clients, labels=labels)
        return build_dataloader(local, stateful=self.stateful_loader, **kwargs)

    def class_counts(
        self,
        dataset: Dataset,
        *,
        num_clients: Optional[int] = None,
        labels: Optional[Sequence[Any]] = None,
    ) -> np.ndarray:
        """``[num_clients, num_classes]`` matrix of per-client label counts."""
        return self.partition_info(dataset, num_clients=num_clients, labels=labels).class_counts

    def summary(
        self,
        dataset: Dataset,
        *,
        num_clients: Optional[int] = None,
        labels: Optional[Sequence[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Realized heterogeneity of the partition: client sizes, classes present,
        and Hellinger distance between each client's label distribution and the
        global one (0 = identical, 1 = disjoint support).
        """
        info = self.partition_info(dataset, num_clients=num_clients, labels=labels)
        counts = info.class_counts
        sizes = counts.sum(axis=1)
        glob = counts.sum(axis=0) / max(int(counts.sum()), 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            local = counts / sizes[:, None]
        hell = np.sqrt(0.5 * np.sum((np.sqrt(local) - np.sqrt(glob)) ** 2, axis=1))
        features = [self.describe_client(i, info.num_clients) for i in range(info.num_clients)]
        out = {
            "scheme": repr(self),
            "partition_id": info.partition_id,
            "seed": info.seed,
            "num_clients": info.num_clients,
            "num_classes": int(counts.shape[1]),
            "coverage": float(sizes.sum() / max(info.num_samples, 1)),
            "sizes": sizes.tolist(),
            "classes_per_client": (counts > 0).sum(axis=1).tolist(),
            "hellinger": [None if np.isnan(h) else float(h) for h in hell],
            "mean_hellinger": float(np.nanmean(hell)) if np.any(~np.isnan(hell)) else float("nan"),
            "class_counts": counts.tolist(),
        }
        if any(features):  # feature-skew schemes: per-client sample transforms
            out["client_features"] = features
        return out

    def __repr__(self) -> str:
        skip = {"loader_kwargs", "label_fn"}
        params = ", ".join(
            f"{k}={v!r}" for k, v in vars(self).items() if not k.startswith("_") and k not in skip
        )
        return f"{type(self).__name__}({params})"

    # ── internals ──────────────────────────────────────────────────────────

    def _compute(
        self,
        dataset: Dataset,
        num_clients: int,
        labels: Optional[Sequence[Any]],
    ) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        """Full partition: (per-client indices, contiguous labels, classes)."""
        y, classes = load_labels(dataset, labels, self.label_fn)
        rng = np.random.default_rng(self.resolve_seed())
        parts = validate_partition(
            self._partition(y, int(classes.size), num_clients, rng),
            num_samples=y.size,
            num_clients=num_clients,
            min_samples_per_client=self.min_samples_per_client,
            name=type(self).__name__,
        )
        return parts, y, classes

    def _result(
        self,
        dataset: Dataset,
        num_clients: Optional[int],
        labels: Optional[Sequence[Any]],
        client_idx: Any,
    ) -> PartitionResult:
        """
        Cached :class:`PartitionResult`. ``client_idx`` is the resolved index of
        the client whose indices must be kept, or ``_STATS_ONLY``.
        """
        k = self.resolve_num_clients(num_clients)
        hit = self._cache.get(id(dataset))
        if (
            hit is not None
            and labels is None
            and hit[0] is dataset
            and hit[1].num_clients == k
            and hit[1].seed == self.resolve_seed()
            and (client_idx is _STATS_ONLY or hit[1].client_idx == client_idx)
        ):
            return hit[1]

        parts, y, classes = self._compute(dataset, k, labels)
        keep = None if client_idx is _STATS_ONLY else int(client_idx)
        result = PartitionResult(
            partition_id=partition_fingerprint(parts, y, classes, extra=self._fingerprint_extra(k)),
            seed=self.resolve_seed(),
            num_samples=int(y.size),
            classes=classes,
            class_counts=np.stack([np.bincount(y[p], minlength=classes.size) for p in parts]),
            client_idx=keep,
            indices=None if keep is None else parts[keep],
        )
        del parts, y  # drop every other client's indices and the label array
        if labels is None:  # explicit labels are one-off; don't let them shadow the dataset's
            self._cache[id(dataset)] = (dataset, result)
        return result


# ════════════════════════════════════════════════════════════════════════════
# Hydra entry point
# ════════════════════════════════════════════════════════════════════════════


def build_skewed_datamodule(
    train: Optional[DataLoader] = None,
    eval: Optional[DataLoader] = None,
    partitioner: Optional[SkewedPartitioner] = None,
    num_federated_clients: Optional[int] = None,
    verbose: bool = True,
    **_: Any,
) -> DataModule:
    """
    Hydra target: partition the train set with any :class:`SkewedPartitioner`.

    Drop-in alternative to ``federated_vision.build_federated_vision_datamodule``.
    The Hydra-built ``train`` loader is used as a template (batch size, workers,
    ...) and rebuilt over this client's partition. The client is identified
    only by ``$OMNIFED_FEDERATED_CLIENT_INDEX`` (set by the runners), so the
    yaml does not name it. The server
    (index -1) gets ``train=None``. ``eval`` is passed through unchanged (global
    test set). Each training rank logs ``partition_id``; it must be identical
    on every rank.

    Example ``conf/datamodule/cifar10_dirichlet.yaml``::

        defaults:
          - cifar10
          - _self_
        _target_: src.omnifed.data.skewed.base.build_skewed_datamodule
        partitioner:
          _target_: src.omnifed.data.skewed.label_distribution.DirichletLabelSkew
          alpha: 0.5
          # seed: 0            # optional; default $OMNIFED_PARTITION_SEED, else 0
          stateful_loader: false
    """
    if partitioner is None:
        raise ValueError("build_skewed_datamodule requires a `partitioner`")
    n_cli = partitioner.resolve_num_clients(num_federated_clients)

    if train is not None:
        cid = partitioner.resolve_client_idx(n_cli)
        if cid is None:
            if verbose:
                print(f"[datamodule] server: train=None (num_federated_clients={n_cli})", flush=True)
            train = None
        else:
            full = train.dataset
            train = partitioner.client_loader(full, template=train, num_clients=n_cli)
            if verbose:
                info = partitioner.partition_info(full, num_clients=n_cli)
                own = info.class_counts[cid]
                present = {_scalar(info.classes[c]): int(v) for c, v in enumerate(own) if v}
                feature = partitioner.describe_client(cid, n_cli)
                print(
                    f"[datamodule] {type(partitioner).__name__} partition_id={info.partition_id} "
                    f"seed={info.seed} client_idx={cid}/{n_cli} "
                    f"len={int(own.sum())}/{len(full)} classes={present} "
                    + (f"feature={feature} " if feature else "")
                    + f"loader={type(train).__name__}",
                    flush=True,
                )

    return DataModule(train=train, eval=eval)
