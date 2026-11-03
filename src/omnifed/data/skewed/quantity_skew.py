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
Quantity skew partitioners: clients hold different amounts of data, while each
client's label mix still matches the global one (IID labels, unequal sizes).

Label skew schemes can also produce unequal sizes, but only as a side effect
of skewing labels; these schemes control size on its own. Implemented:

* ``DirichletQuantitySkew``: size proportions ``q ~ Dir(beta * 1_K)``, redrawn
  until every client has ``min_samples_per_client`` (Li et al., 2022, NIID-Bench).
* ``LogNormalQuantitySkew``: sizes proportional to ``exp(N(0, sigma^2))``
  (LEAF-style unbalanced splits; FedLab's ``unbalance_sgm``).
* ``PowerLawQuantitySkew``: the client of rank ``r`` gets a share proportional to
  ``(r + 1) ** -exponent``: a few large clients, a long tail of small ones.
* ``ExplicitQuantitySkew``: proportions given directly, e.g. to mirror the data
  volumes of specific sites in a cross-silo deployment.

Every scheme draws per-client sizes that sum to the dataset size (largest-
remainder rounding), then deals a seeded random permutation of the samples out
in those sizes. All data is used. Subclasses implement only
:meth:`QuantitySkew.client_proportions`; seeding, validation, caching, loaders
and statistics come from :class:`~.base.SkewedPartitioner`, so ``summary()``
reports the sizes and confirms the label mix stays near-global.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Sequence

import numpy as np

from .base import SkewedPartitioner, sample_dirichlet

__all__ = [
    "QuantitySkew",
    "DirichletQuantitySkew",
    "LogNormalQuantitySkew",
    "PowerLawQuantitySkew",
    "ExplicitQuantitySkew",
    "apportion",
    "QUANTITY_SKEWS",
    "make_quantity_skew",
]


def apportion(proportions: Sequence[float], total: int) -> np.ndarray:
    """
    Integer sizes ``>= 0`` summing exactly to ``total``, as close as possible to
    ``proportions * total`` (largest-remainder method; ties go to the lower
    index, so the result is deterministic).
    """
    q = np.asarray(proportions, dtype=float)
    raw = q / q.sum() * int(total)
    sizes = np.floor(raw).astype(np.int64)
    short = int(total) - int(sizes.sum())
    if short:
        order = np.argsort(-(raw - sizes), kind="stable")
        sizes[order[:short]] += 1
    return sizes


class QuantitySkew(SkewedPartitioner):
    """
    Base class for quantity skew: unequal client sizes, IID labels.

    Subclasses implement :meth:`client_proportions`. Random schemes are
    redrawn (up to ``max_retries`` times) until every client gets at least
    ``min_samples_per_client`` samples; deterministic ones raise right away if
    a client would fall short.

    Args:
        min_samples_per_client: Smallest allowed client (NIID-Bench uses 10).
        max_retries: Redraws allowed for random schemes.
        **kwargs: :class:`~.base.SkewedPartitioner` options (``num_clients``,
            ``seed``, ``stateful_loader``, ...).
    """

    #: Whether :meth:`client_proportions` is random (and worth redrawing).
    RANDOM_PROPORTIONS = True

    def __init__(self, min_samples_per_client: int = 10, max_retries: int = 1000, **kwargs: Any) -> None:
        super().__init__(min_samples_per_client=min_samples_per_client, **kwargs)
        if int(max_retries) < 1:
            raise ValueError("max_retries must be >= 1")
        self.max_retries = int(max_retries)

    @abstractmethod
    def client_proportions(self, num_clients: int, rng: np.random.Generator) -> np.ndarray:
        """Positive weights, one per client; normalized by the caller."""

    def client_sizes(self, num_samples: int, num_clients: int, rng: np.random.Generator) -> np.ndarray:
        """Per-client sample counts summing to ``num_samples``."""
        n, k, floor = int(num_samples), int(num_clients), self.min_samples_per_client
        if k * floor > n:
            raise ValueError(f"cannot give {k} clients >= {floor} samples each from {n} samples")
        attempts = self.max_retries if self.RANDOM_PROPORTIONS else 1
        for _ in range(attempts):
            q = np.asarray(self.client_proportions(k, rng), dtype=float)
            if q.shape != (k,) or not np.all(np.isfinite(q)) or np.any(q < 0) or q.sum() <= 0:
                raise RuntimeError(f"{type(self).__name__} produced invalid proportions {q}")
            sizes = apportion(q, n)
            if sizes.min() >= floor:
                return sizes
        if self.RANDOM_PROPORTIONS:
            raise RuntimeError(
                f"{type(self).__name__}: no draw gave all {k} clients >= {floor} samples in "
                f"{self.max_retries} attempts. Make sizes less uneven, use fewer clients, or "
                "lower min_samples_per_client."
            )
        raise ValueError(
            f"{type(self).__name__}: the smallest client gets {int(sizes.min())} < "
            f"min_samples_per_client={floor} samples (sizes={sizes.tolist()}). Make sizes "
            "less uneven or lower min_samples_per_client."
        )

    def _partition(self, y, num_classes, num_clients, rng):
        sizes = self.client_sizes(y.size, num_clients, rng)
        order = rng.permutation(y.size)  # IID: labels are not looked at
        return np.split(order, np.cumsum(sizes)[:-1])


# ════════════════════════════════════════════════════════════════════════════
# Schemes
# ════════════════════════════════════════════════════════════════════════════


class DirichletQuantitySkew(QuantitySkew):
    """
    Dirichlet quantity skew (Li et al., 2022, NIID-Bench).

    Size proportions ``q ~ Dir(beta * 1_K)``. Smaller ``beta`` = more unequal;
    ``beta -> inf`` approaches equal sizes. NIID-Bench uses ``beta = 0.5``.

    Args:
        beta: Dirichlet concentration (> 0).
    """

    def __init__(self, beta: float = 0.5, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if float(beta) <= 0:
            raise ValueError("beta must be > 0")
        self.beta = float(beta)

    def client_proportions(self, num_clients, rng):
        return sample_dirichlet(rng, np.full(num_clients, self.beta))


class LogNormalQuantitySkew(QuantitySkew):
    """
    Log-normal quantity skew (LEAF-style unbalanced splits; FedLab ``unbalance_sgm``).

    Client sizes proportional to ``exp(N(0, sigma^2))``: ``sigma = 0`` gives equal
    sizes; larger ``sigma`` gives a heavier right tail of large clients.

    Args:
        sigma: Standard deviation of the log-sizes (>= 0).
    """

    def __init__(self, sigma: float = 1.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if float(sigma) < 0:
            raise ValueError("sigma must be >= 0")
        self.sigma = float(sigma)

    def client_proportions(self, num_clients, rng):
        return np.exp(rng.normal(0.0, self.sigma, size=num_clients))


class PowerLawQuantitySkew(QuantitySkew):
    """
    Power-law quantity skew.

    The client of rank ``r`` (0 = largest) gets a share proportional to
    ``(r + 1) ** -exponent``, so the largest client holds ``K ** exponent``
    times the smallest. ``exponent = 0`` gives equal sizes; 1 is Zipf-like.

    Args:
        exponent: Power-law exponent (>= 0).
        shuffle_order: Assign ranks to clients in a seeded random order
            (default). ``False``: client 0 is the largest, client K-1 the
            smallest.
    """

    RANDOM_PROPORTIONS = False  # shuffling the order does not change the sizes

    def __init__(self, exponent: float = 1.0, shuffle_order: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if float(exponent) < 0:
            raise ValueError("exponent must be >= 0")
        self.exponent = float(exponent)
        self.shuffle_order = bool(shuffle_order)

    def client_proportions(self, num_clients, rng):
        shares = np.arange(1, num_clients + 1, dtype=float) ** -self.exponent
        return rng.permutation(shares) if self.shuffle_order else shares


class ExplicitQuantitySkew(QuantitySkew):
    """
    Explicit quantity skew: client ``i`` gets ``proportions[i]`` of the data.

    Proportions are normalized, so ``[2, 1, 1]`` and ``[0.5, 0.25, 0.25]`` are
    the same. There must be exactly one entry per client.

    Args:
        proportions: One positive weight per client.
    """

    RANDOM_PROPORTIONS = False

    def __init__(self, proportions: Sequence[float] = (0.4, 0.3, 0.2, 0.1), **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.proportions = tuple(float(p) for p in proportions)
        if not self.proportions or any(p <= 0 for p in self.proportions):
            raise ValueError("proportions must be a non-empty list of positive numbers")

    def client_proportions(self, num_clients, rng):
        if len(self.proportions) != num_clients:
            raise ValueError(
                f"proportions has {len(self.proportions)} entries but num_clients={num_clients}"
            )
        return np.asarray(self.proportions)


# ════════════════════════════════════════════════════════════════════════════
# Registry
# ════════════════════════════════════════════════════════════════════════════

QUANTITY_SKEWS: Dict[str, type] = {
    "dirichlet": DirichletQuantitySkew,
    "lognormal": LogNormalQuantitySkew,
    "power_law": PowerLawQuantitySkew,
    "explicit": ExplicitQuantitySkew,
}


def make_quantity_skew(name: str, **kwargs: Any) -> QuantitySkew:
    """Construct a scheme by short name, e.g. ``make_quantity_skew("dirichlet", beta=0.1)``."""
    try:
        cls = QUANTITY_SKEWS[name]
    except KeyError:
        raise ValueError(f"unknown quantity skew {name!r}; choose from {sorted(QUANTITY_SKEWS)}") from None
    return cls(**kwargs)
