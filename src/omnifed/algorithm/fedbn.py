# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.
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
FedBN, with module-type-based detection of normalization-layer state
(parameters *and* buffers).

Detection works on modules, not on parameter-name substrings, in three tiers:

1. ``isinstance`` against the torch.nn norm base classes (BatchNorm*d, SyncBatchNorm,
   LazyBatchNorm*d, LayerNorm, GroupNorm, InstanceNorm*d, RMSNorm).
2. Class-name match for third-party / custom norms (``LlamaRMSNorm``, ``LayerNorm2d``,
   ``FrozenBatchNorm2d`` ...). The name must *end* in ``...Norm[<digits>][d]`` so that
   ``LayerNormLinear`` and similar fused layers are not swallowed.
3. Attribute-name match (``bn1``, ``ln_f``, ``input_layernorm``, ``norm_out`` ...), applied
   only to custom modules, never to built-in torch.nn layers, and only to the module's own
   attribute name, matched on ``_`` token boundaries (so ``dbnet`` and ``abnormal`` do not
   match, and an ``nn.Linear`` called ``norm_proj`` is never treated as a norm).

Tier 3 cannot tell which *kind* of norm a bare ``norm``/``norm1`` is, so those are reported
as "unclassified" and are only kept local when ``"all"`` is requested.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, ClassVar, NamedTuple

import rich.repr
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm

from ..communicator import AggregationOp, BaseCommunicator
from . import utils
from .base import BaseAlgorithm

__all__ = ["FedBN", "NORM_KINDS", "NormState", "find_norm_state"]

_RMSNorm = getattr(nn, "RMSNorm", None)  # torch >= 2.4

# Tier 1 ------------------------------------------------------------------------------
_NORM_TYPES: dict[str, tuple[type, ...]] = {
    "batch": (_BatchNorm,),  # BatchNorm1d/2d/3d, SyncBatchNorm, LazyBatchNorm*d
    "layer": (nn.LayerNorm,),
    "group": (nn.GroupNorm,),
    "instance": (_InstanceNorm,),
    "rms": (_RMSNorm,) if _RMSNorm is not None else (),
}

NORM_KINDS: tuple[str, ...] = tuple(_NORM_TYPES)
_UNCLASSIFIED = "unclassified"


# Tier 2 ------------------------------------------------------------------------------
def _class_re(long: str, short: str | None = None) -> re.Pattern[str]:
    pattern = rf"(?:{long}\d*d?$)"
    if short:
        pattern += rf"|(?:^{short}\d*d?$)"
    return re.compile(pattern, re.IGNORECASE)


_NORM_CLASS_RE: dict[str, re.Pattern[str]] = {
    "batch": _class_re("batch_?norm", "bn"),
    "layer": _class_re("layer_?norm", "ln"),
    "group": _class_re("group_?norm", "gn"),
    "instance": _class_re("instance_?norm"),
    "rms": _class_re("rms_?norm"),
}


# Tier 3 ------------------------------------------------------------------------------
def _name_re(tokens: str) -> re.Pattern[str]:
    """Match `tokens` as a whole `_`-delimited token, with an optional numeric suffix."""
    return re.compile(rf"(?:^|_)(?:{tokens})(?:_?\d+)?(?:$|_)", re.IGNORECASE)


_NORM_NAME_RE: dict[str, re.Pattern[str]] = {
    "batch": _name_re("bn|batch_?norm"),
    "layer": _name_re("ln|layer_?norm"),
    "group": _name_re("gn|group_?norm"),
    "instance": _name_re("instance_?norm"),
    "rms": _name_re("rms_?norm"),
}
_GENERIC_NORM_NAME_RE = _name_re("norm")


class NormState(NamedTuple):
    """Fully-qualified names (as in ``named_parameters`` / ``named_buffers``)."""

    params: frozenset[str]
    buffers: frozenset[str]
    unclassified: frozenset[str]  # norm-looking, but kind unknown and not selected


def _classify(module_name: str, module: nn.Module) -> str | None:
    """Return the norm kind of `module`, ``"unclassified"``, or None if it is not a norm."""
    for kind, types in _NORM_TYPES.items():
        if types and isinstance(module, types):
            return kind

    cls = type(module)
    for kind, pattern in _NORM_CLASS_RE.items():
        if pattern.search(cls.__name__):
            return kind

    # Built-in torch.nn layers (Linear, Conv2d, ...) are never classified by attribute name.
    if cls.__module__.startswith("torch.nn."):
        return None

    leaf = module_name.rsplit(".", 1)[-1]
    for kind, pattern in _NORM_NAME_RE.items():
        if pattern.search(leaf):
            return kind
    if _GENERIC_NORM_NAME_RE.search(leaf):
        return _UNCLASSIFIED
    return None


def _resolve_kinds(kinds: str | Iterable[str]) -> set[str]:
    if isinstance(kinds, str):
        kinds = (kinds,)
    wanted = set(kinds)
    if "all" in wanted:
        return set(NORM_KINDS) | {_UNCLASSIFIED}
    unknown = wanted - set(NORM_KINDS)
    if unknown:
        raise ValueError(
            f"Unknown norm kind(s) {sorted(unknown)}; valid: {list(NORM_KINDS)} or 'all'."
        )
    return wanted


def find_norm_state(
    model: nn.Module, kinds: str | Iterable[str] = ("batch",)
) -> NormState:
    """
    Collect the parameter and buffer names owned by normalization layers of the given kinds.

    Args:
        model: Model to scan.
        kinds: Any of ``"batch" | "layer" | "group" | "instance" | "rms"``, or ``"all"``
            (which also includes norm-looking modules whose kind cannot be determined).

    Buffers (``running_mean``, ``running_var``, ``num_batches_tracked``) are included:
    they are not returned by ``named_parameters()`` but are exactly the client-specific
    statistics FedBN is meant to keep local.
    """
    wanted = _resolve_kinds(kinds)
    params: set[str] = set()
    buffers: set[str] = set()
    unclassified: set[str] = set()

    for mod_name, mod in model.named_modules():
        kind = _classify(mod_name, mod)
        if kind is None:
            continue
        prefix = f"{mod_name}." if mod_name else ""
        p_names = {prefix + n for n, _ in mod.named_parameters(recurse=False)}
        b_names = {prefix + n for n, _ in mod.named_buffers(recurse=False)}
        if kind in wanted:
            params |= p_names
            buffers |= b_names
        elif kind == _UNCLASSIFIED:
            unclassified |= p_names | b_names

    return NormState(frozenset(params), frozenset(buffers), frozenset(unclassified))


# ======================================================================================


@rich.repr.auto
class FedBN(BaseAlgorithm):
    """
    Federated Batch Normalization (FedBN) algorithm implementation.

    FedBN aggregates only non-batch normalization parameters across clients,
    allowing each client to maintain its own batch normalization parameters and
    running statistics for improved personalization.

    [FedBN](https://arxiv.org/abs/2102.07623) | Xiaoxiao Li | 2021-02-15

    Attributes:
        LOCAL_NORM_KINDS: Which normalization layers stay local. The default,
            ``("batch",)``, is the original FedBN. Layer/group/instance/RMS norms have no
            data-dependent running statistics, so they are aggregated like any other
            weight unless you opt in, e.g. ``("batch", "layer")`` or ``("all",)``. Override
            in a subclass. Keeping non-BN norms local is a personalization *extension*
            of FedBN, so report it as such.
    """

    LOCAL_NORM_KINDS: ClassVar[tuple[str, ...]] = ("batch",)
    # to extend FedBN to norm layers on LLaMA, Qwen, DeepSeek, use "rms", and "layer" on GPT-models.

    def _configure_local_optimizer(self, local_lr: float) -> torch.optim.Optimizer:
        """
        SGD optimizer for local updates.
        """
        return torch.optim.SGD(self.local_model.parameters(), lr=local_lr)

    def _compute_loss(self, batch: Any) -> torch.Tensor:
        """
        Perform a forward pass and compute the loss for a single batch.
        """
        inputs, targets = batch
        outputs = self.local_model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        return loss

    def _aggregate_within_group(
        self, comm: BaseCommunicator, weight: float
    ) -> nn.Module:
        """
        FedBN aggregation: aggregate non-norm state while keeping selected norm layers local.

        Both the learnable parameters (weight/bias) and the buffers (running_mean,
        running_var, num_batches_tracked) of the local norm layers are saved before
        aggregation and restored afterwards, so they never mix across clients.
        """
        local_params, local_buffers = self._local_norm_state()

        # Save local norm parameters and buffers before aggregation
        saved_params = {
            name: p.detach().clone()
            for name, p in self.local_model.named_parameters()
            if name in local_params
        }
        saved_buffers = {
            name: b.detach().clone()
            for name, b in self.local_model.named_buffers()
            if name in local_buffers
        }

        # Scale only the shared parameters by data proportion (all nodes participate)
        utils.scale_params(
            self.local_model,
            weight,
            filter_fn=lambda name, tensor: name not in local_params,
        )

        # Aggregate shared parameters
        aggregated_model = comm.aggregate(
            self.local_model,
            reduction=AggregationOp.SUM,
        )

        # Restore local norm parameters and buffers
        with torch.no_grad():
            for name, p in aggregated_model.named_parameters():
                if name in saved_params:
                    p.copy_(saved_params[name])
            for name, b in aggregated_model.named_buffers():
                if name in saved_buffers:
                    b.copy_(saved_buffers[name])

        return aggregated_model

    def _local_norm_state(self) -> tuple[frozenset[str], frozenset[str]]:
        """
        Names of (parameters, buffers) kept local. Computed once from module types and
        cached; see `find_norm_state` for the detection rules.
        """
        cached = self.__dict__.get("_norm_state_cache")
        if cached is not None:
            return cached

        state = find_norm_state(self.local_model, self.LOCAL_NORM_KINDS)
        if not state.params and not state.buffers:
            print(
                f"[FedBN] WARNING: no {'/'.join(self.LOCAL_NORM_KINDS)} normalization layers "
                "found in the model; aggregation is identical to FedAvg."
            )
        if state.unclassified:
            print(
                f"[FedBN] WARNING: {len(state.unclassified)} tensors belong to norm-looking "
                f"modules of unknown type and are being AGGREGATED "
                f"(e.g. {sorted(state.unclassified)[:3]}). "
                "Use LOCAL_NORM_KINDS=('all',) to keep them local."
            )
        print(
            f"[FedBN] keeping {len(state.params)} parameters and {len(state.buffers)} "
            f"buffers local (kinds={self.LOCAL_NORM_KINDS})."
        )
        self.__dict__["_norm_state_cache"] = (state.params, state.buffers)
        return self.__dict__["_norm_state_cache"]

    def _is_bn_layer(self, param_name: str) -> bool:
        """
        Kept for backward compatibility: True if `param_name` is kept local.
        """
        return param_name in self._local_norm_state()[0]