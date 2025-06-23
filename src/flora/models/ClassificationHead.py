from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops

from .ComposableModel import BaseHead

# ======================================================================================


class ClassificationHead(BaseHead):
    """
    Classification head that converts features to class predictions.

    - Optional feature transform layers (1x1 convs)
    - Global pooling to convert spatial features to vectors
    - Final projection to class logits
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        feature_layers: List[int] = [],
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        dropout_rate: float = 0.0,
    ):
        """
        Initialize classification head.

        Args:
            in_channels: Number of input channels from the backbone
            num_classes: Number of output classes
            feature_layers: List of feature transform layer sizes
            norm_layer: Optional normalization to apply after convolutions
            activation_layer: Activation function to use
            dropout_rate: Dropout rate for feature layers
        """
        super().__init__()
        self.__in_channels = in_channels

        # Build feature layers
        layers = []
        prev_channels = in_channels

        for feature_size in feature_layers:
            # Add 1x1 convolution
            layers.append(nn.Conv2d(prev_channels, feature_size, kernel_size=1))

            # Add normalization if specified
            if norm_layer is not None:
                layers.append(norm_layer(feature_size))

            # Add activation if specified
            if activation_layer is not None:
                layers.append(activation_layer(inplace=True))

            # Add dropout if specified
            if dropout_rate > 0:
                layers.append(nn.Dropout2d(dropout_rate))

            prev_channels = feature_size

        self.feature_transform = nn.Sequential(*layers) if layers else nn.Identity()

        # Classification output layer
        self.classifier = nn.Conv2d(prev_channels, num_classes, kernel_size=1)

    @property
    def in_channels(self) -> int:
        return self.__in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply feature transform
        x = self.feature_transform(x)

        # Apply classifier
        x = self.classifier(x)

        # Global average pooling and flatten
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)  # [batch_size, num_classes]

        return x
