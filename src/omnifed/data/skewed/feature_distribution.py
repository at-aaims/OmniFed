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
Feature distribution skew (covariate shift) partitioners.

Every scheme here makes ``P(x)`` differ across clients while the labelling rule
``P(y | x)`` is shared: each client sees the same kind of data, altered by its
own fixed transform. Implemented schemes:

* ``GaussianNoiseFeatureSkew``: client ``i`` adds ``N(0, sigma_i^2)`` noise,
  ``sigma_i = sigma * i / (K - 1)`` (Li et al., 2022, NIID-Bench).
* ``RotationFeatureSkew``: client ``i`` rotates by ``angles[i % len(angles)]``
  (rotated MNIST/CIFAR, e.g. IFCA, Ghosh et al., 2020).
* ``PermutationFeatureSkew``: a fixed per-client shuffle of pixels or patches
  (permuted MNIST).
* ``CorruptionFeatureSkew``: client ``i`` gets one corruption type at a fixed
  severity (ImageNet-C / CIFAR-10-C style, Hendrycks & Dietterich, 2019).
* ``StyleFeatureSkew``: fixed per-client brightness / contrast / saturation /
  hue shift, optionally grayscale.

Design
------
* Which samples a client gets is decided by ``split``, any index scheme from
  :mod:`.label_distribution` (default ``IIDLabelDistribution``). So feature
  skew combines with label skew, e.g. rotation on top of a Dirichlet split.
* What happens to them is a per-client transform, applied by
  :class:`ClientTransformDataset`, which wraps the client's ``Subset`` (the
  ``client_dataset`` hook of :class:`~.base.SkewedPartitioner`).
* Transforms act on the tensors the dataset returns, so the dataset transform
  must include ``ToTensor``. If it also normalizes, set ``input_mean`` /
  ``input_std`` to the same values: the transform then runs in ``[0, 1]``
  pixel space (undo Normalize, transform, clamp, redo Normalize). Corruption
  and style schemes need this; without it they assume inputs already in [0, 1].
* Deterministic: per-client parameters (angles, permutations, style factors)
  come from the partition seed and the client index; per-sample randomness
  (noise) from the seed, client index and the sample's dataset index. Every
  rank and every run sees the same data, and ``partition_id`` covers the
  per-client transforms too.
* Transforms are small picklable classes, so DataLoader worker processes
  (spawned on macOS) can use them.
* The eval set is not touched: clients train on shifted data and are
  evaluated on the shared clean test set.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from torchvision.io import decode_jpeg, encode_jpeg
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .base import SkewedPartitioner
from .label_distribution import IIDLabelDistribution

__all__ = [
    "FeatureDistribution",
    "GaussianNoiseFeatureSkew",
    "RotationFeatureSkew",
    "PermutationFeatureSkew",
    "CorruptionFeatureSkew",
    "StyleFeatureSkew",
    "ClientTransform",
    "ClientTransformDataset",
    "CORRUPTIONS",
    "FEATURE_DISTRIBUTIONS",
    "make_feature_distribution",
]

# Independent random streams derived from the partition seed, so that per-client
# parameters and per-sample noise never depend on how much randomness the split
# scheme consumed.
_PARAM_STREAM = 0xFEA7
_SAMPLE_STREAM = 0x5A3E


def _seed_entropy(*keys: int) -> list:
    return [int(k) % (2**63) for k in keys]


def _sample_generator(*keys: int) -> torch.Generator:
    """A torch Generator seeded deterministically from integer keys."""
    state = np.random.SeedSequence(_seed_entropy(*keys)).generate_state(1, dtype=np.uint64)[0]
    return torch.Generator().manual_seed(int(state) % (2**63))


def _float_tuple(values: Optional[Sequence[float]]) -> Optional[Tuple[float, ...]]:
    return None if values is None else tuple(float(v) for v in values)


# ════════════════════════════════════════════════════════════════════════════
# Per-client transforms (picklable)
# ════════════════════════════════════════════════════════════════════════════


class ClientTransform:
    """
    One client's sample transform: ``(x, index) -> x'`` on a ``[C, H, W]``
    tensor, where ``index`` is the sample's index in the full dataset (used to
    seed per-sample randomness). Subclasses are plain picklable objects.
    """

    def __call__(self, x: torch.Tensor, index: int) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def describe(self) -> str:  # pragma: no cover
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.describe()})"


class Identity(ClientTransform):
    def __call__(self, x, index):
        return x

    def describe(self) -> str:
        return "identity"


class AddGaussianNoise(ClientTransform):
    def __init__(self, sigma: float, key: Tuple[int, ...], fixed_per_sample: bool) -> None:
        self.sigma, self.key, self.fixed_per_sample = float(sigma), tuple(key), bool(fixed_per_sample)

    def __call__(self, x, index):
        if self.sigma == 0.0:
            return x
        if self.fixed_per_sample:
            g = _sample_generator(*self.key, index)
            noise = torch.randn(x.shape, generator=g, dtype=x.dtype)
        else:  # fresh noise on every access (NIID-Bench); not reproducible
            noise = torch.randn_like(x)
        return x + self.sigma * noise

    def describe(self) -> str:
        return f"gaussian_noise(sigma={self.sigma:.4g})"


class Rotate(ClientTransform):
    def __init__(self, angle: float) -> None:
        self.angle = float(angle)

    def __call__(self, x, index):
        a = self.angle % 360.0
        if a == 0.0:
            return x
        if a % 90.0 == 0.0 and x.shape[-1] == x.shape[-2]:  # exact, no interpolation
            return torch.rot90(x, int(a // 90.0), dims=(-2, -1))
        return TF.rotate(x, a, interpolation=InterpolationMode.BILINEAR)

    def describe(self) -> str:
        return f"rotate({self.angle:g})"


class PermutePatches(ClientTransform):
    """Shuffle non-overlapping ``patch_size`` squares with one fixed permutation."""

    def __init__(self, patch_size: int, key: Tuple[int, ...]) -> None:
        self.patch_size, self.key = int(patch_size), tuple(key)
        self._perms: Dict[int, torch.Tensor] = {}  # number of patches -> permutation

    def _perm(self, n: int) -> torch.Tensor:
        if n not in self._perms:
            rng = np.random.default_rng(np.random.SeedSequence(_seed_entropy(*self.key, n)))
            self._perms[n] = torch.as_tensor(rng.permutation(n), dtype=torch.long)
        return self._perms[n]

    def __call__(self, x, index):
        c, h, w = x.shape[-3:]
        p = self.patch_size
        if h % p or w % p:
            raise ValueError(f"image {h}x{w} is not divisible into {p}x{p} patches")
        gh, gw = h // p, w // p
        patches = x.reshape(c, gh, p, gw, p).permute(1, 3, 0, 2, 4).reshape(gh * gw, c, p, p)
        patches = patches[self._perm(gh * gw)]
        return patches.reshape(gh, gw, c, p, p).permute(2, 0, 3, 1, 4).reshape(c, h, w)

    def describe(self) -> str:
        return f"permute(patch={self.patch_size}, key={self.key[-1]})"


# ── corruptions on [0, 1] images (severity tables from ImageNet-C) ─────────


def _gaussian_noise(x, c, g):
    return x + c * torch.randn(x.shape, generator=g, dtype=x.dtype)


def _shot_noise(x, c, g):
    return torch.poisson(x.clamp(0.0, 1.0) * c, generator=g) / c


def _impulse_noise(x, c, g):  # salt and pepper, per element
    u = torch.rand(x.shape, generator=g, dtype=x.dtype)
    out = x.clone()
    out[u < c / 2] = 0.0
    out[(u >= c / 2) & (u < c)] = 1.0
    return out


def _speckle_noise(x, c, g):
    return x + x * c * torch.randn(x.shape, generator=g, dtype=x.dtype)


def _gaussian_blur(x, c, g):
    k = 2 * math.ceil(3 * c) + 1
    return TF.gaussian_blur(x, kernel_size=[k, k], sigma=[float(c), float(c)])


def _contrast(x, c, g):
    mean = x.mean(dim=(-2, -1), keepdim=True)
    return (x - mean) * c + mean


def _brightness(x, c, g):
    # HSV value + c with hue/saturation fixed == scaling RGB by (V + c) / V
    v = x.max(dim=-3, keepdim=True).values
    scaled = x * ((v + c).clamp(max=1.0) / v.clamp(min=1e-6))
    gray = (v + c).clamp(max=1.0).expand_as(x)  # black pixels have no hue: become gray
    return torch.where(v < 1e-6, gray, scaled)


def _pixelate(x, c, g):
    h, w = x.shape[-2:]
    small = F.interpolate(x.unsqueeze(0), size=(max(1, int(h * c)), max(1, int(w * c))), mode="area")
    return F.interpolate(small, size=(h, w), mode="nearest").squeeze(0)


def _jpeg_compression(x, c, g):
    u8 = (x.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    return decode_jpeg(encode_jpeg(u8, quality=int(c))).to(x.dtype) / 255.0


#: name -> (function(x, c, generator), parameter for severities 1..5).
#: Values follow Hendrycks & Dietterich (2019), ImageNet-C.
CORRUPTIONS: Dict[str, Tuple[Callable, Tuple[float, ...]]] = {
    "none": (lambda x, c, g: x, (0, 0, 0, 0, 0)),
    "gaussian_noise": (_gaussian_noise, (0.08, 0.12, 0.18, 0.26, 0.38)),
    "shot_noise": (_shot_noise, (60, 25, 12, 5, 3)),
    "impulse_noise": (_impulse_noise, (0.03, 0.06, 0.09, 0.17, 0.27)),
    "speckle_noise": (_speckle_noise, (0.15, 0.2, 0.35, 0.45, 0.6)),
    "gaussian_blur": (_gaussian_blur, (1, 2, 3, 4, 6)),
    "contrast": (_contrast, (0.4, 0.3, 0.2, 0.1, 0.05)),
    "brightness": (_brightness, (0.1, 0.2, 0.3, 0.4, 0.5)),
    "pixelate": (_pixelate, (0.6, 0.5, 0.4, 0.3, 0.25)),
    "jpeg_compression": (_jpeg_compression, (25, 18, 15, 10, 7)),
}


class Corrupt(ClientTransform):
    def __init__(self, name: str, severity: int, key: Tuple[int, ...]) -> None:
        if name not in CORRUPTIONS:
            raise ValueError(f"unknown corruption {name!r}; choose from {sorted(CORRUPTIONS)}")
        if not 1 <= int(severity) <= 5:
            raise ValueError("severity must be in 1..5")
        self.name, self.severity, self.key = name, int(severity), tuple(key)

    def __call__(self, x, index):
        fn, params = CORRUPTIONS[self.name]
        return fn(x, params[self.severity - 1], _sample_generator(*self.key, index))

    def describe(self) -> str:
        return "none" if self.name == "none" else f"{self.name}(severity={self.severity})"


class Style(ClientTransform):
    def __init__(self, brightness: float, contrast: float, saturation: float, hue: float, grayscale: bool) -> None:
        self.brightness, self.contrast = float(brightness), float(contrast)
        self.saturation, self.hue, self.grayscale = float(saturation), float(hue), bool(grayscale)

    def __call__(self, x, index):
        x = TF.adjust_brightness(x, self.brightness)
        x = TF.adjust_contrast(x, self.contrast)
        if x.shape[-3] == 3:
            x = TF.adjust_saturation(x, self.saturation)
            x = TF.adjust_hue(x, self.hue)
            if self.grayscale:
                x = TF.rgb_to_grayscale(x, num_output_channels=3)
        return x

    def describe(self) -> str:
        return (
            f"style(brightness={self.brightness:.3f}, contrast={self.contrast:.3f}, "
            f"saturation={self.saturation:.3f}, hue={self.hue:+.3f}"
            + (", grayscale" if self.grayscale else "")
            + ")"
        )


# ════════════════════════════════════════════════════════════════════════════
# Client dataset wrapper
# ════════════════════════════════════════════════════════════════════════════


class ClientTransformDataset(Dataset):
    """
    A client's ``Subset`` with its :class:`ClientTransform` applied to the
    input (first element of each ``(x, y, ...)`` sample).

    Exposes ``dataset`` and ``indices`` like ``Subset``, so code that inspects
    a client's data works for label and feature skew alike.
    """

    def __init__(
        self,
        subset: Subset,
        transform: ClientTransform,
        input_mean: Optional[Sequence[float]] = None,
        input_std: Optional[Sequence[float]] = None,
    ) -> None:
        self.subset = subset
        self.transform = transform
        self._mean = None if input_mean is None else torch.tensor(input_mean, dtype=torch.float32).view(-1, 1, 1)
        self._std = None if input_std is None else torch.tensor(input_std, dtype=torch.float32).view(-1, 1, 1)

    @property
    def dataset(self) -> Dataset:
        return self.subset.dataset

    @property
    def indices(self) -> Sequence[int]:
        return self.subset.indices

    def __len__(self) -> int:
        return len(self.subset)

    def apply(self, x: torch.Tensor, index: int) -> torch.Tensor:
        """The client transform on one input, in pixel space when mean/std are set."""
        if not torch.is_tensor(x):
            raise TypeError(
                f"feature-skew transforms act on tensors, got {type(x).__name__}; "
                "include ToTensor in the dataset transform"
            )
        if self._mean is None:
            return self.transform(x, index)
        mean, std = self._mean.to(x.dtype), self._std.to(x.dtype)
        pixels = x * std + mean
        return (self.transform(pixels, index).clamp(0.0, 1.0) - mean) / std

    def __getitem__(self, i: int):
        sample = self.subset[i]
        if not isinstance(sample, (tuple, list)):
            raise TypeError("expected (input, target, ...) samples")
        x = self.apply(sample[0], int(self.subset.indices[i]))
        return (x, *sample[1:])


# ════════════════════════════════════════════════════════════════════════════
# Family base class
# ════════════════════════════════════════════════════════════════════════════


class FeatureDistribution(SkewedPartitioner):
    """
    Feature distribution skew: a per-client transform on top of an index split.

    Subclasses implement :meth:`make_client_transform`. Seeding, caching,
    validation, loaders and statistics come from
    :class:`~.base.SkewedPartitioner`; the index split is delegated to
    ``split``.

    Args:
        split: Index scheme deciding which samples each client gets (any
            :class:`~.base.SkewedPartitioner`, typically a label scheme).
            Defaults to ``IIDLabelDistribution()``. Called with this object's
            RNG and client count; its own seed / client count / loader options
            are ignored.
        input_mean, input_std: The dataset's ``Normalize`` statistics. When set,
            transforms run in ``[0, 1]`` pixel space and the result is clamped
            and re-normalized. Leave unset if inputs are not normalized.
        **kwargs: :class:`~.base.SkewedPartitioner` options (``num_clients``,
            ``seed``, ``min_samples_per_client``, ``stateful_loader``, ...).
    """

    def __init__(
        self,
        split: Optional[SkewedPartitioner] = None,
        input_mean: Optional[Sequence[float]] = None,
        input_std: Optional[Sequence[float]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if (input_mean is None) != (input_std is None):
            raise ValueError("set both input_mean and input_std, or neither")
        self.split = split if split is not None else IIDLabelDistribution()
        self.input_mean = _float_tuple(input_mean)
        self.input_std = _float_tuple(input_std)
        if self.input_std is not None and any(s <= 0 for s in self.input_std):
            raise ValueError("input_std must be positive")

    # ── the one method subclasses implement ────────────────────────────────

    @abstractmethod
    def make_client_transform(self, client_idx: int, num_clients: int) -> ClientTransform:
        """
        The transform for one client. Draw any per-client randomness from
        :meth:`client_params_rng` so it is identical on every rank.
        """

    # ── shared machinery ───────────────────────────────────────────────────

    def _partition(self, y, num_classes, num_clients, rng):
        return self.split._partition(y, num_classes, num_clients, rng)

    def client_params_rng(self, client_idx: int) -> np.random.Generator:
        """Generator for one client's fixed parameters (seed, client index)."""
        return np.random.default_rng(
            np.random.SeedSequence(_seed_entropy(self.resolve_seed(), _PARAM_STREAM, client_idx))
        )

    def sample_key(self, client_idx: int) -> Tuple[int, ...]:
        """Key for per-sample randomness; the sample index is appended at use."""
        return (self.resolve_seed() % (2**63), _SAMPLE_STREAM, int(client_idx))

    def client_transform(self, client_idx: int, num_clients: Optional[int] = None) -> ClientTransform:
        k = self.resolve_num_clients(num_clients)
        if not 0 <= int(client_idx) < k:
            raise ValueError(f"client_idx={client_idx} out of range for {k} clients")
        return self.make_client_transform(int(client_idx), k)

    def client_dataset(
        self,
        dataset: Dataset,
        *,
        num_clients: Optional[int] = None,
        labels: Optional[Sequence[Any]] = None,
    ) -> Dataset:
        """This rank's ``Subset`` wrapped with its client transform."""
        subset = super().client_dataset(dataset, num_clients=num_clients, labels=labels)
        k = self.resolve_num_clients(num_clients)
        cid = self.resolve_client_idx(k)  # not None: super() already rejected the server
        return ClientTransformDataset(subset, self.client_transform(cid, k), self.input_mean, self.input_std)

    def describe_client(self, client_idx: int, num_clients: int) -> str:
        return self.client_transform(client_idx, num_clients).describe()

    def _fingerprint_extra(self, num_clients: int) -> str:
        space = "pixel" if self.input_mean is not None else "tensor"
        return f"{type(self).__name__}|{space}|" + "|".join(
            self.describe_client(i, num_clients) for i in range(num_clients)
        )


# ════════════════════════════════════════════════════════════════════════════
# Schemes
# ════════════════════════════════════════════════════════════════════════════


class GaussianNoiseFeatureSkew(FeatureDistribution):
    """
    Noise-based feature skew (Li et al., 2022, NIID-Bench).

    Client ``i`` of ``K`` adds Gaussian noise with standard deviation
    ``sigma * i / (K - 1)``: client 0 is clean, the last client gets ``sigma``.
    ``sigma`` is in pixel units when ``input_mean``/``input_std`` are set, else in
    the units of the dataset's tensors (NIID-Bench adds it after Normalize).

    Args:
        sigma: Noise level of the noisiest client (>= 0).
        fixed_per_sample: ``True`` (default): each sample's noise is fixed by
            the seed, client and sample index, so runs and resumed loaders are
            reproducible. ``False``: fresh noise on every access, as in
            NIID-Bench (uses the global torch RNG, not reproducible).
    """

    def __init__(self, sigma: float = 0.1, fixed_per_sample: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if float(sigma) < 0:
            raise ValueError("sigma must be >= 0")
        self.sigma = float(sigma)
        self.fixed_per_sample = bool(fixed_per_sample)

    def make_client_transform(self, client_idx, num_clients):
        level = self.sigma * client_idx / (num_clients - 1) if num_clients > 1 else 0.0
        return AddGaussianNoise(level, self.sample_key(client_idx), self.fixed_per_sample)


class RotationFeatureSkew(FeatureDistribution):
    """
    Rotation-based feature skew (rotated MNIST/CIFAR; e.g. IFCA, Ghosh et al., 2020).

    Client ``i`` rotates every image counter-clockwise by
    ``angles[i % len(angles)]`` degrees. Multiples of 90 on square images are
    exact (``torch.rot90``); other angles use bilinear interpolation, keep the
    image size, and fill uncovered corners with black (zeros in pixel space).

    Args:
        angles: Rotation per client group, in degrees.
    """

    def __init__(self, angles: Sequence[float] = (0, 90, 180, 270), **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.angles = _float_tuple(angles)
        if not self.angles:
            raise ValueError("angles must not be empty")

    def make_client_transform(self, client_idx, num_clients):
        return Rotate(self.angles[client_idx % len(self.angles)])


class PermutationFeatureSkew(FeatureDistribution):
    """
    Permutation-based feature skew (permuted MNIST).

    Each client applies its own fixed random permutation to the image's
    ``patch_size x patch_size`` patches (``patch_size=1``: individual pixels,
    the classic setting; larger patches keep local structure, like a jigsaw).
    The same permutation is used for every sample and channel of a client.

    Args:
        patch_size: Side of the square patches; must divide the image size.
        include_identity: Client 0 keeps the original layout.
    """

    def __init__(self, patch_size: int = 1, include_identity: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if int(patch_size) < 1:
            raise ValueError("patch_size must be >= 1")
        self.patch_size = int(patch_size)
        self.include_identity = bool(include_identity)

    def make_client_transform(self, client_idx, num_clients):
        if self.include_identity and client_idx == 0:
            return Identity()
        key = (self.resolve_seed() % (2**63), _PARAM_STREAM, client_idx)
        return PermutePatches(self.patch_size, key)


class CorruptionFeatureSkew(FeatureDistribution):
    """
    Corruption-based feature skew (ImageNet-C / CIFAR-10-C style;
    Hendrycks & Dietterich, 2019).

    Client ``i`` applies ``corruptions[i % len(corruptions)]`` at severity
    ``severity[i % len(severity)]`` (1 = mild .. 5 = severe). Available:
    ``none``, ``gaussian_noise``, ``shot_noise``, ``impulse_noise``,
    ``speckle_noise``, ``gaussian_blur``, ``contrast``, ``brightness``,
    ``pixelate``, ``jpeg_compression``, with ImageNet-C's severity parameters,
    implemented on tensors. Noise draws are fixed per sample (seed, client,
    sample index). Runs in [0, 1] pixel space: set ``input_mean``/``input_std``
    if the dataset normalizes.

    Args:
        corruptions: Corruption per client group.
        severity: One severity for all clients, or one per client group.
    """

    def __init__(
        self,
        corruptions: Sequence[str] = (
            "gaussian_noise",
            "shot_noise",
            "impulse_noise",
            "gaussian_blur",
            "contrast",
            "brightness",
            "pixelate",
            "jpeg_compression",
        ),
        severity: Union[int, Sequence[int]] = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.corruptions = tuple(str(c) for c in corruptions)
        self.severity = (int(severity),) if isinstance(severity, (int, np.integer)) else tuple(int(s) for s in severity)
        if not self.corruptions or not self.severity:
            raise ValueError("corruptions and severity must not be empty")
        bad = [c for c in self.corruptions if c not in CORRUPTIONS]
        if bad:
            raise ValueError(f"unknown corruptions {bad}; choose from {sorted(CORRUPTIONS)}")
        if any(not 1 <= s <= 5 for s in self.severity):
            raise ValueError("severity must be in 1..5")

    def make_client_transform(self, client_idx, num_clients):
        return Corrupt(
            self.corruptions[client_idx % len(self.corruptions)],
            self.severity[client_idx % len(self.severity)],
            self.sample_key(client_idx),
        )


class StyleFeatureSkew(FeatureDistribution):
    """
    Style / colour feature skew.

    Each client draws one fixed set of factors, applied to all its images:
    brightness and contrast and saturation factors uniform in
    ``[1 - x, 1 + x]``, hue shift uniform in ``[-hue, hue]``, and grayscale with
    probability ``grayscale_prob``. Runs in [0, 1] pixel space: set
    ``input_mean``/``input_std`` if the dataset normalizes.

    Args:
        brightness, contrast, saturation: Spread of the factors, in [0, 1).
        hue: Spread of the hue shift, in [0, 0.5].
        grayscale_prob: Chance that a client sees grayscale images.
    """

    def __init__(
        self,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        hue: float = 0.1,
        grayscale_prob: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        for name, v in (("brightness", brightness), ("contrast", contrast), ("saturation", saturation)):
            if not 0.0 <= float(v) < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        if not 0.0 <= float(hue) <= 0.5:
            raise ValueError("hue must be in [0, 0.5]")
        if not 0.0 <= float(grayscale_prob) <= 1.0:
            raise ValueError("grayscale_prob must be in [0, 1]")
        self.brightness, self.contrast, self.saturation = float(brightness), float(contrast), float(saturation)
        self.hue, self.grayscale_prob = float(hue), float(grayscale_prob)

    def make_client_transform(self, client_idx, num_clients):
        rng = self.client_params_rng(client_idx)
        return Style(
            brightness=1.0 + rng.uniform(-self.brightness, self.brightness),
            contrast=1.0 + rng.uniform(-self.contrast, self.contrast),
            saturation=1.0 + rng.uniform(-self.saturation, self.saturation),
            hue=rng.uniform(-self.hue, self.hue),
            grayscale=bool(rng.random() < self.grayscale_prob),
        )


# ════════════════════════════════════════════════════════════════════════════
# Registry
# ════════════════════════════════════════════════════════════════════════════

FEATURE_DISTRIBUTIONS: Dict[str, type] = {
    "gaussian_noise": GaussianNoiseFeatureSkew,
    "rotation": RotationFeatureSkew,
    "permutation": PermutationFeatureSkew,
    "corruption": CorruptionFeatureSkew,
    "style": StyleFeatureSkew,
}


def make_feature_distribution(name: str, **kwargs: Any) -> FeatureDistribution:
    """Construct a scheme by short name, e.g. ``make_feature_distribution("rotation")``."""
    try:
        cls = FEATURE_DISTRIBUTIONS[name]
    except KeyError:
        raise ValueError(
            f"unknown feature distribution {name!r}; choose from {sorted(FEATURE_DISTRIBUTIONS)}"
        ) from None
    return cls(**kwargs)
