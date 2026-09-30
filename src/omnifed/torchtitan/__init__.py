from __future__ import annotations

from typing import Any

from .backend import TorchTitanBackend
from .roles import (
    TorchTitanRole,
    resolve_torchtitan_role,
)
from .state_chunks import (
    ModelStateChunkAssembler,
    iter_model_state_chunks,
)


def create_torchtitan_backend(
    cfg: Any,
    **kwargs: Any,
) -> TorchTitanBackend:
    return TorchTitanBackend(
        cfg=cfg,
        **kwargs,
    )


__all__ = [
    "TorchTitanBackend",
    "TorchTitanRole",
    "ModelStateChunkAssembler",
    "create_torchtitan_backend",
    "iter_model_state_chunks",
    "resolve_torchtitan_role",
]