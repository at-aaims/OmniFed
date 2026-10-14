from __future__ import annotations

import math
from typing import Any

import torch

from .base import BaseTorchTitanAlgorithm, StateDict


class DiLoCo(BaseTorchTitanAlgorithm):
    """
    TorchTitan local training with server-side outer momentum.

    Uses the ordinary-momentum formula from algorithm/diloco.py
    when nesterov=False.

    Clients send uniformly weighted parameter chunks.
    The server computes deltas from their averaged parameters.

    Local optimizer resetting remains controlled by TorchTitanBackend.
    """

    def __init__(
        self,
        num_clients: int,
        outer_lr: float = 0.7,
        outer_momentum: float = 0.9,
        nesterov: bool = False,
    ) -> None:
        super().__init__()

        self.num_clients = int(num_clients)
        self.outer_lr = float(outer_lr)
        self.outer_momentum = float(outer_momentum)
        self.nesterov = bool(nesterov)

        # Parameter names, or existing parameter-name/slice-offset keys.
        self.velocity: dict[str, torch.Tensor] = {}

        self.validate()

    @property
    def name(self) -> str:
        return "diloco"

    def validate(self) -> None:
        if self.num_clients < 1:
            raise ValueError("num_clients must be positive")

        if not math.isfinite(self.outer_lr) or self.outer_lr < 0:
            raise ValueError("outer_lr must be finite and nonnegative")

        if (
            not math.isfinite(self.outer_momentum)
            or not 0 <= self.outer_momentum < 1
        ):
            raise ValueError("outer_momentum must be in [0, 1)")

    def client_weight(
        self,
        local_units: float,
        total_units: float,
        round_id: int,
    ) -> float:
        # Count logical clients, not TorchTitan ranks.
        # The dedicated server is not a training client.
        return 1.0 / self.num_clients

    def prepare_client_chunk(
        self,
        chunk: StateDict,
        weight: float,
        round_id: int,
    ) -> dict[str, torch.Tensor]:
        prepared = {}

        for name, tensor in chunk.items():
            if not tensor.is_floating_point():
                raise TypeError(
                    f"DiLoCo parameter exchange requires floating tensors; "
                    f"received {name} with dtype {tensor.dtype}"
                )

            prepared[name] = tensor.detach().cpu().float() * weight

        return prepared

    # Inherit prepare_server_chunk(): contribute zeros to SUM.

    @torch.no_grad()
    def server_step(
        self,
        averaged_parameters: StateDict,
        previous_global_parameters: StateDict,
        round_id: int,
    ) -> dict[str, torch.Tensor]:
        if set(averaged_parameters) != set(previous_global_parameters):
            raise ValueError("Averaged and previous parameter keys differ")

        updated_parameters = {}
        updated_velocity = {}

        for name, averaged in averaged_parameters.items():
            previous = previous_global_parameters[name]

            if not previous.is_floating_point():
                raise TypeError(
                    f"Cannot apply DiLoCo momentum to {name}: "
                    f"dtype={previous.dtype}"
                )

            if averaged.shape != previous.shape:
                raise ValueError(
                    f"Shape mismatch for {name}: "
                    f"{tuple(averaged.shape)} versus {tuple(previous.shape)}"
                )

            previous = previous.detach().cpu().float()
            averaged = averaged.detach().cpu().float()

            delta = averaged - previous

            old_velocity = self.velocity.get(name)
            if old_velocity is None:
                old_velocity = torch.zeros_like(previous)
            elif old_velocity.shape != previous.shape:
                raise ValueError(
                    f"Velocity shape mismatch for {name}. "
                    "Do not change chunk layout during a run or resume."
                )

            # Same ordinary-momentum convention as algorithm/diloco.py:
            #
            # v_new = momentum * v_old + outer_lr * delta
            # W_new = W_old + v_new
            velocity = (
                self.outer_momentum * old_velocity
                + self.outer_lr * delta
            )

            if self.nesterov:
                # Learning-rate-scaled velocity convention.
                # Equivalent to SGD Nesterov with fixed outer_lr.
                update = (
                    self.outer_lr * delta
                    + self.outer_momentum * velocity
                )
            else:
                update = velocity

            updated_parameters[name] = previous + update
            updated_velocity[name] = velocity

        # Commit after validating and computing the complete chunk.
        self.velocity.update(updated_velocity)

        return updated_parameters

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_clients": self.num_clients,
            "outer_lr": self.outer_lr,
            "outer_momentum": self.outer_momentum,
            "nesterov": self.nesterov,
            "velocity": {
                name: tensor.detach().cpu().clone()
                for name, tensor in self.velocity.items()
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {
            "num_clients": self.num_clients,
            "outer_lr": self.outer_lr,
            "outer_momentum": self.outer_momentum,
            "nesterov": self.nesterov,
        }

        for key, value in expected.items():
            if state[key] != value:
                raise ValueError(
                    f"Checkpoint {key}={state[key]!r} "
                    f"does not match configuration {value!r}"
                )

        self.velocity = {
            name: tensor.detach().cpu().float().clone()
            for name, tensor in state["velocity"].items()
        }