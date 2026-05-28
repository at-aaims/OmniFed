"""Shared state between hybrid MPI (facility) and gRPC (leader) paths."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["HybridCommBridge"]


@dataclass
class HybridCommBridge:
    """Filled during sync Phase 1 for use in Phase 2 (gRPC batch_samples)."""

    last_group_total_samples: int = 0
