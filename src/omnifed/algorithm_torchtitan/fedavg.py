from __future__ import annotations

from typing import Mapping

import torch

from .base import BaseTorchTitanAlgorithm


class FedAvg(BaseTorchTitanAlgorithm):
    """
    Token-weighted FedAvg for TorchTitan models.

    TorchTitan performs local LLM training. After each client leader
    consolidates the distributed model, FedAvg averages the resulting
    ordinary state dictionaries.
    """

    def __init__(
        self,
        weighting: str = "tokens",
    ) -> None:
        super().__init__()

        self.weighting = weighting.lower()

        if self.weighting != "tokens":
            raise ValueError(
                "TorchTitan FedAvg currently supports only "
                f"weighting='tokens'; got {weighting!r}"
            )

    @property
    def name(self) -> str:
        return "fedavg"

    def client_weight(
        self,
        local_units: float,
        total_units: float,
        round_id: int,
    ) -> float:
        if total_units <= 0:
            return 0.0

        return local_units / total_units

    def prepare_client_chunk(
        self,
        chunk: Mapping[str, torch.Tensor],
        weight: float,
        round_id: int,
    ) -> dict[str, torch.Tensor]:
        return {
            name: tensor * weight
            for name, tensor in chunk.items()
        }