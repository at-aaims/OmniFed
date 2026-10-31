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
Concept shift partitioners: the same inputs carry different labels on
different clients, i.e. ``P(y | x)`` differs. Implemented schemes:

* ``LabelSwapConceptShift``: clients form ``num_groups`` groups; each group
  swaps its own random set of class pairs (clustered-FL benchmarks, e.g. CFL,
  Sattler et al., 2020).
* ``LabelPermutationConceptShift``: each group relabels every class with its
  own random permutation (clustered / multi-task FL, IFCA- and FedEM-style).
* ``LabelNoiseConceptShift``: a fraction of clients is noisy; each noisy client
  flips a share of its labels, symmetric or pair-flip (noisy-label FL, e.g.
  FedCorr, Xu et al., 2022).

The other form, same label but different inputs (``P(x | y)`` differs), is
feature skew: see :mod:`.feature_distribution`.

Design
------
* Which samples a client gets is decided by ``split`` (any index scheme,
  default ``IIDLabelDistribution``), so concept shift combines with label skew.
* What its labels become is a per-client target transform applied on the fly
  by :class:`~.base.ClientTransformDataset`. Inputs are untouched.
* Mappings act on the dataset's own label values (``classes``), so any label
  type works. Labels the mapping has never seen pass through unchanged.
* Deterministic: a group's mapping comes from (seed, group); which clients are
  noisy and how noisy from (seed, num_clients) and (seed, client); each noise
  flip from (seed, client, sample index). ``partition_id`` covers it all.
* Evaluation: with ``eval_follows_concept`` (default for swap / permutation)
  each trainer's eval loader is relabelled with its group's mapping, which is
  what clustered / personalised FL reports. Label noise defaults to clean eval,
  the noisy-label protocol. The server always keeps the clean eval set.
* Pure integer operations (numpy): no torch needed for the label logic.
"""

from __future__ import annotations

import hashlib
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from torch.utils.data import Dataset, Subset

from .base import ClientTransformDataset, SkewedPartitioner, load_labels
from .label_distribution import IIDLabelDistribution

__all__ = [
    "ConceptShift",
    "GroupConceptShift",
    "LabelSwapConceptShift",
    "LabelPermutationConceptShift",
    "LabelNoiseConceptShift",
    "LabelMap",
    "NoisyLabels",
    "CONCEPT_SHIFTS",
    "make_concept_shift",
]

# Random streams derived from the partition seed (independent of the split).
_GROUP_STREAM = 0xC0C3
_CLIENT_STREAM = 0xC11E
_NOISY_SET_STREAM = 0x5E7
_SAMPLE_STREAM = 0x5A3F


def _entropy(*keys: int) -> List[int]:
    return [int(k) % (2**63) for k in keys]


def _rng(*keys: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(_entropy(*keys)))


def _scalar(y: Any) -> Any:
    return y.item() if hasattr(y, "item") else y


def _like(y: Any, value: Any) -> Any:
    """Return ``value`` in the same container type as the incoming label ``y``."""
    if hasattr(y, "dtype") and hasattr(y, "new_tensor"):  # torch.Tensor
        return y.new_tensor(value)
    if isinstance(y, np.generic):
        return np.asarray(value, dtype=y.dtype)[()]
    return value


def _digest(perm: np.ndarray) -> str:
    return hashlib.blake2b(np.ascontiguousarray(perm, dtype="<i8").tobytes(), digest_size=4).hexdigest()


# ════════════════════════════════════════════════════════════════════════════
# Target transforms (picklable)
# ════════════════════════════════════════════════════════════════════════════


class LabelMap:
    """
    Fixed relabelling ``classes[i] -> classes[perm[i]]`` for one client.
    Callable as ``(y, index) -> y'``; labels outside ``classes`` pass through.
    """

    def __init__(self, classes: np.ndarray, perm: np.ndarray, label: str) -> None:
        self.classes = np.asarray(classes)
        self.perm = np.asarray(perm, dtype=np.int64)
        self.label = label
        self._map: Dict[Any, Any] = {
            _scalar(self.classes[i]): _scalar(self.classes[j]) for i, j in enumerate(self.perm)
        }

    def __call__(self, y: Any, index: int) -> Any:
        key = _scalar(y)
        return _like(y, self._map.get(key, key))

    @property
    def changed(self) -> int:
        """Number of classes whose label changes."""
        return int(np.sum(self.perm != np.arange(self.perm.size)))

    def describe(self) -> str:
        if self.changed == 0:
            return f"{self.label}(identity)"
        return f"{self.label}(changed={self.changed}, map={_digest(self.perm)})"

    def __repr__(self) -> str:
        return f"LabelMap({self.describe()})"


class NoisyLabels:
    """
    Flip each label with probability ``rate``, independently per sample but
    fixed by (seed, client, sample index). ``symmetric``: to a uniformly random
    other class; ``pairflip``: class ``k`` to ``k + 1`` (wrapping). ``rate == 0``
    is a clean client.
    """

    def __init__(self, classes: np.ndarray, rate: float, noise_type: str, key: Tuple[int, ...]) -> None:
        self.classes = np.asarray(classes)
        self.rate, self.noise_type, self.key = float(rate), str(noise_type), tuple(key)
        self._pos: Dict[Any, int] = {_scalar(c): i for i, c in enumerate(self.classes)}

    def flipped_position(self, pos: int, index: int) -> int:
        if self.rate == 0.0 or self.classes.size < 2:
            return pos
        rng = _rng(*self.key, index)
        if rng.random() >= self.rate:
            return pos
        if self.noise_type == "pairflip":
            return (pos + 1) % self.classes.size
        return (pos + int(rng.integers(1, self.classes.size))) % self.classes.size  # any other class

    def __call__(self, y: Any, index: int) -> Any:
        key = _scalar(y)
        pos = self._pos.get(key)
        if pos is None:  # unseen label
            return y
        return _like(y, _scalar(self.classes[self.flipped_position(pos, int(index))]))

    def describe(self) -> str:
        return "clean" if self.rate == 0.0 else f"{self.noise_type}_noise(rate={self.rate:.3f})"

    def __repr__(self) -> str:
        return f"NoisyLabels({self.describe()})"


# ════════════════════════════════════════════════════════════════════════════
# Family base classes
# ════════════════════════════════════════════════════════════════════════════


class ConceptShift(SkewedPartitioner):
    """
    Concept shift: a per-client label transform on top of an index split.

    Subclasses implement :meth:`make_client_target_transform`. Seeding,
    caching, validation, loaders and statistics come from
    :class:`~.base.SkewedPartitioner`; the index split is delegated to ``split``.

    Args:
        split: Index scheme deciding which samples each client gets. Defaults
            to ``IIDLabelDistribution()``. Called with this object's RNG and
            client count; its own seed / client count / loader options are
            ignored.
        eval_follows_concept: Relabel each trainer's eval set with its client's
            label transform (the server keeps the clean eval set).
        **kwargs: :class:`~.base.SkewedPartitioner` options (``num_clients``,
            ``seed``, ``min_samples_per_client``, ``stateful_loader``, ...).
    """

    def __init__(
        self,
        split: Optional[SkewedPartitioner] = None,
        eval_follows_concept: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.split = split if split is not None else IIDLabelDistribution()
        self.eval_follows_concept = bool(eval_follows_concept)

    @abstractmethod
    def make_client_target_transform(self, client_idx: int, num_clients: int, classes: np.ndarray) -> Any:
        """The label transform ``(y, index) -> y'`` for one client."""

    # ── shared machinery ───────────────────────────────────────────────────

    def _partition(self, y, num_classes, num_clients, rng):
        return self.split._partition(y, num_classes, num_clients, rng)

    def client_target_transform(self, client_idx: int, num_clients: int, classes: Sequence[Any]) -> Any:
        if not 0 <= int(client_idx) < int(num_clients):
            raise ValueError(f"client_idx={client_idx} out of range for {num_clients} clients")
        return self.make_client_target_transform(int(client_idx), int(num_clients), np.asarray(classes))

    def client_dataset(
        self,
        dataset: Dataset,
        *,
        num_clients: Optional[int] = None,
        labels: Optional[Sequence[Any]] = None,
    ) -> Dataset:
        """This rank's ``Subset`` with its labels transformed on the fly."""
        subset = super().client_dataset(dataset, num_clients=num_clients, labels=labels)
        k = self.resolve_num_clients(num_clients)
        cid = self.resolve_client_idx(k)  # not None: super() already rejected the server
        classes = self.partition_info(dataset, num_clients=k, labels=labels).classes
        return ClientTransformDataset(subset, target_transform=self.client_target_transform(cid, k, classes))

    def client_eval_dataset(self, eval_dataset, client_idx, num_clients, classes):
        if not self.eval_follows_concept:
            return eval_dataset
        transform = self.client_target_transform(client_idx, num_clients, classes)
        everything = Subset(eval_dataset, list(range(len(eval_dataset))))
        return ClientTransformDataset(everything, target_transform=transform)

    def describe_client(self, client_idx: int, num_clients: int, classes=None) -> str:
        if classes is None:
            return self.describe_abstract(client_idx, num_clients)
        return self.client_target_transform(client_idx, num_clients, classes).describe()

    def describe_abstract(self, client_idx: int, num_clients: int) -> str:
        """Description when the label set is not known yet."""
        return type(self).__name__

    def _fingerprint_extra(self, num_clients: int, classes=None) -> str:
        return f"{type(self).__name__}|eval={self.eval_follows_concept}|" + "|".join(
            self.describe_client(i, num_clients, classes) for i in range(num_clients)
        )

    def summary(self, dataset: Dataset, *, num_clients=None, labels=None) -> Dict[str, Any]:
        """
        :meth:`~.base.SkewedPartitioner.summary` plus what clients actually see:
        ``observed_class_counts`` (labels after the transform) and
        ``relabelled_fraction`` per client. Recomputes the full partition, so
        it is meant for analysis, not training.
        """
        out = super().summary(dataset, num_clients=num_clients, labels=labels)
        k = self.resolve_num_clients(num_clients)
        y, classes = load_labels(dataset, labels, self.label_fn)
        pos = {_scalar(c): i for i, c in enumerate(classes)}
        observed, relabelled = [], []
        for cid, idx in enumerate(self.partition(dataset, num_clients=k, labels=labels)):
            t = self.client_target_transform(cid, k, classes)
            new = np.fromiter((pos[_scalar(t(classes[y[j]], int(j)))] for j in idx), dtype=np.int64, count=idx.size)
            observed.append(np.bincount(new, minlength=classes.size).tolist())
            relabelled.append(float(np.mean(new != y[idx])) if idx.size else 0.0)
        out["observed_class_counts"] = observed
        out["relabelled_fraction"] = relabelled
        return out


class GroupConceptShift(ConceptShift):
    """
    Concept shift by groups: client ``i`` belongs to group ``i % num_groups`` and
    every client of a group shares one label mapping (drawn from the seed and
    the group). Subclasses implement :meth:`group_permutation`.

    Args:
        num_groups: Number of concepts (>= 1).
        include_identity: Group 0 keeps the original labels.
    """

    def __init__(self, num_groups: int = 2, include_identity: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if int(num_groups) < 1:
            raise ValueError("num_groups must be >= 1")
        self.num_groups = int(num_groups)
        self.include_identity = bool(include_identity)

    def client_group(self, client_idx: int) -> int:
        return int(client_idx) % self.num_groups

    @abstractmethod
    def group_permutation(self, num_classes: int, rng: np.random.Generator) -> np.ndarray:
        """A permutation of ``range(num_classes)`` for one (non-identity) group."""

    def make_client_target_transform(self, client_idx, num_clients, classes):
        g = self.client_group(client_idx)
        c = int(classes.size)
        if self.include_identity and g == 0:
            perm = np.arange(c)
        else:
            perm = self.group_permutation(c, _rng(self.resolve_seed(), _GROUP_STREAM, g, c))
        return LabelMap(classes, perm, label=f"group{g}")

    def describe_abstract(self, client_idx, num_clients):
        return f"group{self.client_group(client_idx)}"


# ════════════════════════════════════════════════════════════════════════════
# Schemes
# ════════════════════════════════════════════════════════════════════════════


class LabelSwapConceptShift(GroupConceptShift):
    """
    Label-swap concept shift (clustered-FL benchmarks, e.g. CFL, Sattler et al., 2020).

    Each group swaps ``swap_pairs`` random, disjoint pairs of classes (``a <-> b``);
    group 0 keeps the original labels when ``include_identity``. Clients of a
    group share the swaps, so the groups are the ground-truth clusters.

    Args:
        swap_pairs: Pairs swapped per group; needs ``2 * swap_pairs <= num_classes``.
        num_groups, include_identity: See :class:`GroupConceptShift`.
        eval_follows_concept: Default ``True``: each trainer is evaluated with
            its group's labels.
    """

    def __init__(self, swap_pairs: int = 1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if int(swap_pairs) < 1:
            raise ValueError("swap_pairs must be >= 1")
        self.swap_pairs = int(swap_pairs)

    def group_permutation(self, num_classes, rng):
        if 2 * self.swap_pairs > num_classes:
            raise ValueError(f"swap_pairs={self.swap_pairs} needs {2 * self.swap_pairs} classes, have {num_classes}")
        chosen = rng.choice(num_classes, size=2 * self.swap_pairs, replace=False)
        perm = np.arange(num_classes)
        a, b = chosen[0::2], chosen[1::2]
        perm[a], perm[b] = b, a
        return perm


class LabelPermutationConceptShift(GroupConceptShift):
    """
    Label-permutation concept shift (clustered / multi-task FL).

    Each group relabels every class with its own random derangement (no class
    keeps its label); group 0 keeps the original labels when
    ``include_identity``. The strongest form of concept shift: no input has the
    same label in two groups.

    Args:
        num_groups, include_identity: See :class:`GroupConceptShift`.
        eval_follows_concept: Default ``True``.
    """

    def group_permutation(self, num_classes, rng):
        if num_classes < 2:
            return np.arange(num_classes)
        while True:  # rejection sampling: ~e draws on average
            perm = rng.permutation(num_classes)
            if not np.any(perm == np.arange(num_classes)):
                return perm


class LabelNoiseConceptShift(ConceptShift):
    """
    Label-noise concept shift (noisy-label FL, e.g. FedCorr, Xu et al., 2022).

    ``round(noisy_client_fraction * K)`` clients, chosen by the seed, are noisy.
    Each noisy client flips its labels with its own rate: ``noise_rate`` if a
    number, or uniform in ``[low, high]`` if a pair. Flips are fixed per sample.

    Args:
        noise_type: ``"symmetric"`` (to a random other class) or ``"pairflip"``
            (class ``k`` to ``k + 1``, by sorted label order).
        noisy_client_fraction: Share of clients that are noisy, in [0, 1].
        noise_rate: Flip probability, or a ``[low, high]`` range per client.
        eval_follows_concept: Default ``False``: evaluate on clean labels.
    """

    def __init__(
        self,
        noise_type: str = "symmetric",
        noisy_client_fraction: float = 0.5,
        noise_rate: Union[float, Sequence[float]] = 0.4,
        eval_follows_concept: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(eval_follows_concept=eval_follows_concept, **kwargs)
        if noise_type not in ("symmetric", "pairflip"):
            raise ValueError("noise_type must be 'symmetric' or 'pairflip'")
        if not 0.0 <= float(noisy_client_fraction) <= 1.0:
            raise ValueError("noisy_client_fraction must be in [0, 1]")
        if isinstance(noise_rate, (int, float, np.floating)):
            rate: Union[float, Tuple[float, float]] = float(noise_rate)
            bounds = (rate, rate)
        else:
            rate = tuple(float(r) for r in noise_rate)
            if len(rate) != 2 or rate[0] > rate[1]:
                raise ValueError("noise_rate must be a number or [low, high] with low <= high")
            bounds = rate
        if not (0.0 <= bounds[0] and bounds[1] <= 1.0):
            raise ValueError("noise_rate must be within [0, 1]")
        self.noise_type = noise_type
        self.noisy_client_fraction = float(noisy_client_fraction)
        self.noise_rate = rate

    def noisy_clients(self, num_clients: int) -> np.ndarray:
        """Sorted indices of the noisy clients (same on every rank)."""
        n = int(round(self.noisy_client_fraction * num_clients))
        rng = _rng(self.resolve_seed(), _NOISY_SET_STREAM, num_clients)
        return np.sort(rng.choice(num_clients, size=n, replace=False))

    def client_noise_rate(self, client_idx: int, num_clients: int) -> float:
        if client_idx not in set(self.noisy_clients(num_clients).tolist()):
            return 0.0
        if isinstance(self.noise_rate, float):
            return self.noise_rate
        low, high = self.noise_rate
        return float(_rng(self.resolve_seed(), _CLIENT_STREAM, client_idx).uniform(low, high))

    def make_client_target_transform(self, client_idx, num_clients, classes):
        key = (self.resolve_seed() % (2**63), _SAMPLE_STREAM, client_idx)
        return NoisyLabels(classes, self.client_noise_rate(client_idx, num_clients), self.noise_type, key)

    def describe_abstract(self, client_idx, num_clients):
        rate = self.client_noise_rate(client_idx, num_clients)
        return "clean" if rate == 0.0 else f"{self.noise_type}_noise(rate={rate:.3f})"


# ════════════════════════════════════════════════════════════════════════════
# Registry
# ════════════════════════════════════════════════════════════════════════════

CONCEPT_SHIFTS: Dict[str, type] = {
    "label_swap": LabelSwapConceptShift,
    "label_permutation": LabelPermutationConceptShift,
    "label_noise": LabelNoiseConceptShift,
}


def make_concept_shift(name: str, **kwargs: Any) -> ConceptShift:
    """Construct a scheme by short name, e.g. ``make_concept_shift("label_swap", swap_pairs=8)``."""
    try:
        cls = CONCEPT_SHIFTS[name]
    except KeyError:
        raise ValueError(f"unknown concept shift {name!r}; choose from {sorted(CONCEPT_SHIFTS)}") from None
    return cls(**kwargs)
