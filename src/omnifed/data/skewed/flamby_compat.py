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
Run FLamby's 2024 code unmodified on current libraries.

FLamby (pinned submodule) uses things newer releases removed:

* ``np.NaN`` (removed in NumPy 2.0), in Fed-Heart-Disease;
* ``monai.transforms.AddChannel`` (removed in MONAI 1.3), imported by Fed-IXI
  (dataset class and, through the package ``__init__``, its download script);
* albumentations' ``Flip`` (removed in 1.4.24), ``always_apply=`` and
  ``CoarseDropout(max_holes, max_height, max_width)`` (removed in 2.0), used by
  Fed-ISIC2019 to build its augmentations.

* ``pydicom.read_file`` (removed in pydicom 3.0), used by Fed-LIDC-IDRI's
  DICOM -> NIfTI conversion.

It also lets Fed-ISIC2019 load without ``efficientnet-pytorch``: FLamby's
package imports its EfficientNet baseline model (``model.py``) even though the
data code never uses it, so when the package is missing a stand-in module is
registered; only building that model then fails, with an install hint.

:func:`apply` restores the first two before FLamby is imported and
:func:`after_import` gives Fed-ISIC2019 an albumentations with the old names
(same transforms, new spellings). They are used in-process by
:mod:`.flamby_partition`, and FLamby's download scripts (separate processes)
are started through this file so they get the same patches::

    python flamby_compat.py <flamby download script> [script args...]

This file deliberately has no OmniFed imports, so it runs as a plain script.
"""

from __future__ import annotations

import importlib.util
import runpy
import sys
import types
from pathlib import Path
from typing import Any

__all__ = ["apply", "after_import"]


class _AddChannel:
    """``monai.transforms.AddChannel`` (removed in MONAI 1.3): prepend a channel axis."""

    def __call__(self, img: Any) -> Any:
        return img[None]


def _numpy2_compat() -> None:
    import numpy as np

    if not hasattr(np, "NaN"):
        np.NaN = np.nan


def _monai_compat() -> None:
    if importlib.util.find_spec("monai") is None:
        return
    import monai.transforms as mt

    if not hasattr(mt, "AddChannel"):
        mt.AddChannel = _AddChannel


class _MissingEfficientNet:
    """Stand-in for ``efficientnet_pytorch.EfficientNet`` when the package is not installed."""

    _HINT = (
        "FLamby's Fed-ISIC2019 baseline model needs efficientnet-pytorch: pip install efficientnet-pytorch "
        "(loading the data does not)"
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise ImportError(self._HINT)

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> Any:
        raise ImportError(cls._HINT)

    @classmethod
    def from_name(cls, *args: Any, **kwargs: Any) -> Any:
        raise ImportError(cls._HINT)


def _efficientnet_placeholder() -> None:
    """Register a stand-in ``efficientnet_pytorch`` if it is not installed (Fed-ISIC2019's model import)."""
    if "efficientnet_pytorch" in sys.modules or importlib.util.find_spec("efficientnet_pytorch") is not None:
        return
    placeholder = types.ModuleType("efficientnet_pytorch")
    placeholder.EfficientNet = _MissingEfficientNet
    placeholder.__omnifed_placeholder__ = True
    sys.modules["efficientnet_pytorch"] = placeholder


def _pydicom_compat() -> None:
    if importlib.util.find_spec("pydicom") is None:
        return
    import pydicom

    if not hasattr(pydicom, "read_file"):
        pydicom.read_file = pydicom.dcmread


def apply(module: str = "") -> None:
    """Patch what FLamby needs; ``module`` (e.g. ``flamby.datasets.fed_ixi``) limits the patches to that dataset."""
    _numpy2_compat()
    if not module or module.endswith("fed_ixi"):
        _monai_compat()
    if not module or module.endswith("fed_isic2019"):
        _efficientnet_placeholder()
    if not module or module.endswith("fed_lidc_idri"):
        _pydicom_compat()


class _AlbumentationsCompat(types.ModuleType):
    """
    ``albumentations`` as Fed-ISIC2019 calls it (the <= 1.4.23 API), on a newer
    release. Everything else is passed through unchanged.
    """

    def __init__(self, albumentations: types.ModuleType) -> None:
        super().__init__(albumentations.__name__)
        self.__dict__["_A"] = albumentations

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__dict__["_A"], name)

    def Flip(self, p: float = 0.5, always_apply: Any = None):  # noqa: N802 - old API name
        """Old ``Flip``: horizontal, vertical or both, equally likely."""
        A = self._A
        both = A.Sequential([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)], p=1.0)
        return A.OneOf([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0), both], p=1.0 if always_apply else p)

    def Normalize(self, *args: Any, always_apply: Any = None, **kwargs: Any):  # noqa: N802
        if always_apply:
            kwargs["p"] = 1.0
        return self._A.Normalize(*args, **kwargs)

    def CoarseDropout(self, max_holes: int = 8, max_height: int = 8, max_width: int = 8, **kwargs: Any):  # noqa: N802
        """Old positional form: exactly ``max_holes`` holes of ``max_height`` x ``max_width`` pixels."""
        return self._A.CoarseDropout(
            num_holes_range=(max_holes, max_holes),
            hole_height_range=(max_height, max_height),
            hole_width_range=(max_width, max_width),
            **kwargs,
        )


def _albumentations_compat() -> None:
    mod = sys.modules.get("flamby.datasets.fed_isic2019.dataset")
    A = getattr(mod, "albumentations", None)
    if A is None or isinstance(A, _AlbumentationsCompat) or hasattr(A, "Flip"):
        return  # not imported, already patched, or an old enough release
    mod.albumentations = _AlbumentationsCompat(A)


def after_import(module: str = "") -> None:
    """Patches that need the FLamby module itself (call after importing ``module``)."""
    if not module or module.endswith("fed_isic2019"):
        _albumentations_compat()


def _run_script(argv: list) -> None:
    """Run ``argv[0]`` as ``__main__`` with its own folder first on ``sys.path``, after :func:`apply`."""
    script = str(Path(argv[0]).resolve())
    parts = Path(script).parts
    apply(next((p for p in ("fed_ixi", "fed_isic2019") if p in parts), "flamby"))
    sys.argv = [script, *argv[1:]]
    sys.path[0] = str(Path(script).parent)  # the scripts import siblings (e.g. color_constancy)
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python flamby_compat.py <script.py> [args...]")
    _run_script(sys.argv[1:])
