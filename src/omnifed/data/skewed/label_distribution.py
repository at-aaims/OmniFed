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
Label distribution skew (prior shift) partitioners.

Every scheme here makes ``P(y)`` differ across clients while ``P(x | y)`` stays
the same. Implemented schemes:

* ``IIDLabelDistribution``: uniform random split (homogeneous baseline).
* ``ShardLabelSkew``: sort by label, cut into shards, k shards per client
  (McMahan et al., 2017).
* ``QuantityLabelSkew``: each client sees exactly k classes, "#C=k"
  (Li et al., 2022, NIID-Bench).
* ``DirichletLabelSkew``: per-class ``Dir(alpha)`` split over clients
  (Yurochkin et al., 2019; NIID-Bench).
* ``ClientDirichletLabelSkew``: per-client class mix ``q_i ~ Dir(alpha * p)``
  (Hsu et al., 2019).
* ``SimilarityLabelSkew``: fraction ``s`` IID, rest sorted by label
  (Karimireddy et al., 2020, SCAFFOLD).
* ``LongTailLabelSkew``: global long-tail subsample, then an inner scheme
  (Cao et al., 2019; CReFF).

All schemes extend :class:`LabelDistribution`, itself a thin subclass of
:class:`~.base.SkewedPartitioner`, which provides label reading, seeding,
validation, caching, ``DataLoader`` / ``StatefulDataLoader`` construction, and
statistics. Each scheme here only implements ``_partition``. Schemes compose
(``LongTailLabelSkew(base=DirichletLabelSkew(...))``). For Hydra, use
``base.build_skewed_datamodule`` with any of these as ``partitioner``.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np

from .base import SkewedPartitioner, sample_dirichlet

__all__ = [
    "LabelDistribution",
    "IIDLabelDistribution",
    "ShardLabelSkew",
    "QuantityLabelSkew",
    "DirichletLabelSkew",
    "ClientDirichletLabelSkew",
    "SimilarityLabelSkew",
    "LongTailLabelSkew",
    "LABEL_DISTRIBUTIONS",
    "make_label_distribution",
]


# ════════════════════════════════════════════════════════════════════════════
# Family base class
# ════════════════════════════════════════════════════════════════════════════


class LabelDistribution(SkewedPartitioner):
    """
    Label distribution skew (prior shift): ``P(y)`` differs across clients while
    ``P(x | y)`` is unchanged.

    Adds nothing to :class:`~.base.SkewedPartitioner`; it marks the family so
    configs and type hints can require "a label-skew scheme". Subclasses
    implement ``_partition(y, num_classes, num_clients, rng)``.
    """


# ════════════════════════════════════════════════════════════════════════════
# Schemes
# ════════════════════════════════════════════════════════════════════════════


class IIDLabelDistribution(LabelDistribution):
    """Homogeneous baseline: uniform random split into near-equal parts."""

    def _partition(self, y, num_classes, num_clients, rng):
        return np.array_split(rng.permutation(y.size), num_clients)


class ShardLabelSkew(LabelDistribution):
    """
    Pathological shard split (McMahan et al., 2017, FedAvg).

    Sort samples by label, cut into ``num_clients * shards_per_client`` equal
    shards, and give each client ``shards_per_client`` random shards. With 2
    shards/client each client sees at most ~2 classes (a shard can straddle a
    class boundary). Order within a class is randomized.
    """

    def __init__(self, shards_per_client: int = 2, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if int(shards_per_client) < 1:
            raise ValueError("shards_per_client must be >= 1")
        self.shards_per_client = int(shards_per_client)

    def _partition(self, y, num_classes, num_clients, rng):
        s = self.shards_per_client
        num_shards = num_clients * s
        if num_shards > y.size:
            raise ValueError(f"{num_shards} shards requested for only {y.size} samples")
        order = np.lexsort((rng.random(y.size), y))  # by label, random within label
        shards = np.array_split(order, num_shards)
        assign = rng.permutation(num_shards)
        return [
            np.concatenate([shards[j] for j in assign[i * s : (i + 1) * s]])
            for i in range(num_clients)
        ]


class QuantityLabelSkew(LabelDistribution):
    """
    Quantity-based label imbalance, ``#C=k`` (NIID-Bench, Li et al., 2022).

    Each client is assigned exactly ``classes_per_client`` distinct classes;
    client ``i`` always gets class ``i % C`` so every class is covered when
    ``num_clients >= C``. Each class's samples are split evenly among the
    clients that own it. Classes owned by no client are dropped (warned).
    """

    def __init__(self, classes_per_client: int = 2, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if int(classes_per_client) < 1:
            raise ValueError("classes_per_client must be >= 1")
        self.classes_per_client = int(classes_per_client)

    def _partition(self, y, num_classes, num_clients, rng):
        k, c_all = self.classes_per_client, num_classes
        if k > c_all:
            raise ValueError(f"classes_per_client={k} exceeds num_classes={c_all}")
        owners: List[List[int]] = [[] for _ in range(c_all)]
        for i in range(num_clients):
            first = i % c_all
            others = rng.choice(np.delete(np.arange(c_all), first), size=k - 1, replace=False)
            for c in (first, *others.tolist()):
                owners[c].append(i)

        buckets: List[List[np.ndarray]] = [[] for _ in range(num_clients)]
        dropped = []
        for c in range(c_all):
            if not owners[c]:
                dropped.append(c)
                continue
            idx = rng.permutation(np.flatnonzero(y == c))
            for owner, chunk in zip(owners[c], np.array_split(idx, len(owners[c]))):
                buckets[owner].append(chunk)
        if dropped:
            warnings.warn(
                f"QuantityLabelSkew: classes {dropped} are owned by no client and were "
                f"dropped (num_clients={num_clients} * k={k} < num_classes={c_all})",
                stacklevel=3,
            )
        return [np.concatenate(b) if b else np.empty(0, np.int64) for b in buckets]


class DirichletLabelSkew(LabelDistribution):
    """
    Distribution-based label imbalance (Yurochkin et al., 2019; NIID-Bench).

    For every class ``c`` draw ``p_c ~ Dir(alpha * 1_K)`` and split that class's
    samples among the ``K`` clients in proportions ``p_c``. Smaller ``alpha`` means
    more skew; ``alpha -> inf`` approaches IID. Common values: 0.1, 0.3, 0.5, 1.0.

    The whole draw is repeated until every client holds at least
    ``min_samples_per_client`` samples (NIID-Bench uses 10).

    Args:
        alpha: Dirichlet concentration (> 0).
        balance: NIID-Bench trick: while splitting a class, give zero share to
            clients that already hold ``>= N / K`` samples. Keeps client sizes
            closer to equal, so label skew is not confounded with quantity skew.
        max_retries: Give up after this many full redraws.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        balance: bool = True,
        min_samples_per_client: int = 10,
        max_retries: int = 1000,
        **kwargs: Any,
    ) -> None:
        super().__init__(min_samples_per_client=min_samples_per_client, **kwargs)
        if float(alpha) <= 0:
            raise ValueError("alpha must be > 0")
        self.alpha = float(alpha)
        self.balance = bool(balance)
        self.max_retries = int(max_retries)
        self.attempts_: Optional[int] = None  # draws needed by the last partition

    def _partition(self, y, num_classes, num_clients, rng):
        n, k = y.size, num_clients
        if k * self.min_samples_per_client > n:
            raise ValueError(
                f"cannot give {k} clients >= {self.min_samples_per_client} samples each "
                f"from {n} samples"
            )
        class_idx = [np.flatnonzero(y == c) for c in range(num_classes)]
        cap = n / k
        conc = np.full(k, self.alpha)
        for attempt in range(1, self.max_retries + 1):
            buckets: List[List[np.ndarray]] = [[] for _ in range(k)]
            sizes = np.zeros(k, dtype=np.int64)
            for c in range(num_classes):
                idx = rng.permutation(class_idx[c])
                p = sample_dirichlet(rng, conc)
                if self.balance:
                    p = p * (sizes < cap)
                    total = p.sum()
                    p = p / total if total > 0 else np.full(k, 1.0 / k)
                cuts = (np.cumsum(p) * idx.size).astype(np.int64)[:-1]
                for i, chunk in enumerate(np.split(idx, cuts)):
                    buckets[i].append(chunk)
                    sizes[i] += chunk.size
            if sizes.min() >= self.min_samples_per_client:
                self.attempts_ = attempt
                return [np.concatenate(b) for b in buckets]
        raise RuntimeError(
            f"DirichletLabelSkew(alpha={self.alpha}): no draw gave all {k} clients >= "
            f"{self.min_samples_per_client} samples in {self.max_retries} attempts. "
            "Increase alpha, use fewer clients, or lower min_samples_per_client."
        )


class ClientDirichletLabelSkew(LabelDistribution):
    """
    Per-client Dirichlet class mixture (Hsu, Qi & Brown, 2019).

    Each client draws its own class distribution ``q_i ~ Dir(alpha * C * prior)``
    and then samples ``samples_per_client`` points from it without replacement.
    Unlike :class:`DirichletLabelSkew` (which fixes how each *class* is spread
    over clients), this fixes each *client's* size and draws its label mix.

    ``alpha`` is per-class concentration, so for balanced data
    ``Dir(alpha * C * prior) == Dir(alpha * 1_C)`` and values are comparable to
    :class:`DirichletLabelSkew`. (Hsu et al. report the total ``alpha * C``.)

    When a client's preferred classes are exhausted, its remaining samples come
    from the classes still available, in proportion to availability. Clients are
    filled in random order so this effect is not tied to client id.

    Args:
        alpha: Concentration per class (> 0).
        samples_per_client: Samples per client. ``None`` uses all data
            (``N // K``, remainder spread over the first clients).
        prior: ``"global"`` (dataset class frequencies) or ``"uniform"``.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        samples_per_client: Optional[int] = None,
        prior: str = "global",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if float(alpha) <= 0:
            raise ValueError("alpha must be > 0")
        if prior not in ("global", "uniform"):
            raise ValueError("prior must be 'global' or 'uniform'")
        self.alpha = float(alpha)
        self.samples_per_client = None if samples_per_client is None else int(samples_per_client)
        self.prior = prior

    def _partition(self, y, num_classes, num_clients, rng):
        n, k, c_all = y.size, num_clients, num_classes
        counts = np.bincount(y, minlength=c_all)
        if self.samples_per_client is None:
            sizes = np.full(k, n // k, dtype=np.int64)
            sizes[: n % k] += 1
        else:
            if self.samples_per_client * k > n:
                raise ValueError(
                    f"samples_per_client={self.samples_per_client} x {k} clients > {n} samples"
                )
            sizes = np.full(k, self.samples_per_client, dtype=np.int64)

        prior = counts / n if self.prior == "global" else np.full(c_all, 1.0 / c_all)
        conc = np.maximum(self.alpha * c_all * prior, 1e-12)
        pools = [rng.permutation(np.flatnonzero(y == c)) for c in range(c_all)]
        avail = counts.astype(np.int64)
        ptr = np.zeros(c_all, dtype=np.int64)

        parts: List[np.ndarray] = [np.empty(0, np.int64)] * k
        for i in rng.permutation(k):
            q = sample_dirichlet(rng, conc)
            need = int(sizes[i])
            take = np.zeros(c_all, dtype=np.int64)
            while need > 0:
                remaining = avail - take
                w = q * (remaining > 0)
                if w.sum() <= 0:  # preferred classes exhausted
                    w = remaining.astype(float)
                draw = np.minimum(rng.multinomial(need, w / w.sum()), remaining)
                take += draw
                need -= int(draw.sum())
            chunks = []
            for c in np.flatnonzero(take):
                chunks.append(pools[c][ptr[c] : ptr[c] + take[c]])
                ptr[c] += take[c]
            avail -= take
            parts[i] = np.concatenate(chunks) if chunks else np.empty(0, np.int64)
        return parts


class SimilarityLabelSkew(LabelDistribution):
    """
    Similarity-controlled split (Karimireddy et al., 2020, SCAFFOLD).

    A fraction ``similarity`` of the data is split IID; the rest is sorted by
    label and cut into contiguous blocks, one per client. ``similarity=1`` is
    IID; ``similarity=0`` gives each client ~``C/K`` contiguous classes.
    Block-to-client assignment is randomized.
    """

    def __init__(self, similarity: float = 0.1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not 0.0 <= float(similarity) <= 1.0:
            raise ValueError("similarity must be in [0, 1]")
        self.similarity = float(similarity)

    def _partition(self, y, num_classes, num_clients, rng):
        perm = rng.permutation(y.size)
        n_iid = int(round(self.similarity * y.size))
        iid, rest = perm[:n_iid], perm[n_iid:]
        rest = rest[np.argsort(y[rest], kind="stable")]
        iid_parts = np.array_split(iid, num_clients)
        sorted_parts = np.array_split(rest, num_clients)
        block_of = rng.permutation(num_clients)
        return [
            np.concatenate([iid_parts[i], sorted_parts[block_of[i]]]) for i in range(num_clients)
        ]


class LongTailLabelSkew(LabelDistribution):
    """
    Global long-tail subsample followed by an inner scheme (Cao et al., 2019;
    CIFAR-10/100-LT as used by CReFF / FedLT).

    Class ``r``-th in ``class_order`` keeps ``n_max * imbalance_factor^(-r/(C-1))``
    samples (exponential profile), so the head/tail size ratio is
    ``imbalance_factor``. The kept subset is then partitioned by ``base``.

    Args:
        base: Inner scheme. Its ``_partition`` is called with this object's RNG
            and client count; its own ``seed`` / ``num_clients`` / loader
            options are ignored. Defaults to ``DirichletLabelSkew(alpha=0.5)``.
        imbalance_factor: ``n_max / n_min`` (>= 1). 1 means no subsampling.
        class_order: ``"index"`` (class 0 is the head, standard CIFAR-LT) or
            ``"random"``.
    """

    def __init__(
        self,
        base: Optional[LabelDistribution] = None,
        imbalance_factor: float = 100.0,
        class_order: str = "index",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if float(imbalance_factor) < 1:
            raise ValueError("imbalance_factor must be >= 1")
        if class_order not in ("index", "random"):
            raise ValueError("class_order must be 'index' or 'random'")
        self.base = base if base is not None else DirichletLabelSkew(alpha=0.5)
        self.imbalance_factor = float(imbalance_factor)
        self.class_order = class_order

    def _partition(self, y, num_classes, num_clients, rng):
        counts = np.bincount(y, minlength=num_classes)
        n_max = int(counts.max())
        order = np.arange(num_classes) if self.class_order == "index" else rng.permutation(num_classes)
        keep = []
        for r, c in enumerate(order):
            frac = self.imbalance_factor ** (-r / (num_classes - 1)) if num_classes > 1 else 1.0
            target = int(np.clip(int(n_max * frac), 1, counts[c]))
            keep.append(rng.choice(np.flatnonzero(y == c), size=target, replace=False))
        kept = np.sort(np.concatenate(keep))
        inner = self.base._partition(y[kept], num_classes, num_clients, rng)
        return [kept[np.asarray(p, dtype=np.int64)] for p in inner]


# ════════════════════════════════════════════════════════════════════════════
# Registry
# ════════════════════════════════════════════════════════════════════════════

LABEL_DISTRIBUTIONS: Dict[str, type] = {
    "iid": IIDLabelDistribution,
    "shards": ShardLabelSkew,
    "quantity": QuantityLabelSkew,
    "dirichlet": DirichletLabelSkew,
    "client_dirichlet": ClientDirichletLabelSkew,
    "similarity": SimilarityLabelSkew,
    "long_tail": LongTailLabelSkew,
}


def make_label_distribution(name: str, **kwargs: Any) -> LabelDistribution:
    """Construct a scheme by short name, e.g. ``make_label_distribution("dirichlet", alpha=0.1)``."""
    try:
        cls = LABEL_DISTRIBUTIONS[name]
    except KeyError:
        raise ValueError(
            f"unknown label distribution {name!r}; choose from {sorted(LABEL_DISTRIBUTIONS)}"
        ) from None
    return cls(**kwargs)
