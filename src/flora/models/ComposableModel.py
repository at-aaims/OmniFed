from abc import ABC, abstractmethod

import torch
import torch.nn as nn

# ======================================================================================


class BaseHead(nn.Module, ABC):
    """
    Abstract base class for CNN heads that defines the interface.
    """

    @property
    @abstractmethod
    def in_channels(self) -> int:
        """
        Get the number of input channels to the head.
        This should match the output channels of the backbone.
        """
        pass


class BaseBackbone(nn.Module, ABC):
    """
    Abstract base class for CNN backbones that defines the interface.
    """

    @property
    @abstractmethod
    def out_channels(self) -> int:
        """
        Get the number of output channels from the backbone.
        This should match the input channels of the head.
        """
        pass


class ComposableModel(nn.Module):
    """
    Flexible composable model with swappable backbone and head components.
    """

    def __init__(
        self,
        backbone: BaseBackbone,
        head: BaseHead,
    ):
        super().__init__()

        # Ensure compatible
        if backbone.out_channels != head.in_channels:
            raise ValueError(
                f"Backbone output channels ({backbone.out_channels}) don't match "
                f"head input channels ({head.in_channels})"
            )

        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features with backbone
        features = self.backbone(x)

        # Process features with head
        output = self.head(features)

        return output
