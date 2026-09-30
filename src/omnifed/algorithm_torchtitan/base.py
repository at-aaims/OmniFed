from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import torch


StateDict = Mapping[str, torch.Tensor]


class BaseTorchTitanAlgorithm(ABC):
    """
    Base class for federated algorithms using TorchTitan local training.

    Responsibilities are separated as follows:

    TorchTitanBackend:
        - Builds the distributed model.
        - Runs PP/TP/DP local training.
        - Saves distributed checkpoints.
        - Consolidates checkpoints.
        - Loads the returned global model.

    BaseTorchTitanAlgorithm:
        - Controls algorithm-specific local training hooks.
        - Computes client aggregation weights.
        - Prepares client model chunks.
        - Provides the server collective contribution.
        - Applies algorithm-specific server updates.
    """

    def __init__(self) -> None:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    def validate(self) -> None:
        """
        Validate algorithm configuration before training.
        """

    def before_local_training(
        self,
        backend: Any,
        global_model_path: str,
        round_id: int,
    ) -> None:
        """
        Optional hook before TorchTitan local training.

        FedProx, SCAFFOLD and similar algorithms may use this hook
        to configure the local training objective.
        """

    def train_local(
        self,
        backend: Any,
        local_steps: int,
        round_id: int,
    ) -> dict[str, Any]:
        """
        Default TorchTitan local training implementation.
        """

        return backend.train_local_steps(
            steps=local_steps,
        )

    def after_local_training(
        self,
        backend: Any,
        train_result: dict[str, Any],
        round_id: int,
    ) -> None:
        """
        Optional hook after local training and before checkpoint export.
        """

    @abstractmethod
    def client_weight(
        self,
        local_units: float,
        total_units: float,
        round_id: int,
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    def prepare_client_chunk(
        self,
        chunk: StateDict,
        weight: float,
        round_id: int,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def prepare_server_chunk(
        self,
        chunk: StateDict,
        round_id: int,
    ) -> dict[str, torch.Tensor]:
        """
        Default server contribution to a collective SUM.
        """

        return {
            name: torch.zeros_like(tensor)
            for name, tensor in chunk.items()
        }

    def finalize_global_state(
        self,
        aggregated_state: dict[str, torch.Tensor],
        previous_global_state: StateDict,
        round_id: int,
    ) -> dict[str, torch.Tensor]:
        """
        Optional server-side update after collecting client models.

        FedAvg returns the averaged model unchanged.
        FedMom can apply server momentum here.
        """

        return aggregated_state