from typing import Optional, Union

import rich.repr
import torch
import torch.nn as nn

from .BaseCommunicator import Communicator


# ======================================================================================


@rich.repr.auto
class DummyCommunicator(Communicator):
    """
    Mock communicator for development with no-ops.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        group_name: str = "default",
        **kwargs,  # Accept additional parameters for compatibility with other communicators
    ):
        self.rank = rank
        self.world_size = world_size
        self.group_name = group_name
        print(f"DummyCommunicator | rank={rank}/{world_size} | group={group_name}")

    def setup(self) -> None:
        """
        Initialize the dummy communicator.
        This is a no-op for the mock implementation.
        """
        print("DummyCommunicator setup complete")

    def broadcast(self, *, model: nn.Module, src: int = 0) -> nn.Module:
        print(f"DummyCommunicator broadcast from rank {src}")
        return model

    def aggregate(
        self,
        *,
        obj: Union[nn.Module, torch.Tensor],
        mean: bool = True,
        num_samples: Optional[int] = None,
    ) -> Union[nn.Module, torch.Tensor]:
        print(f"DummyCommunicator aggregate | mean={mean}")
        return obj

    def close(self):
        print("DummyCommunicator close")
