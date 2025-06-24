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

from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn
from torchvision import ops

from .ComposableModel import BaseBackbone


class SimpleCNNBackbone(BaseBackbone):
    """
    A simple & generic CNN backbone.
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: List[int] = [32, 64],
        kernel_sizes: Union[int, List[int]] = 3,
        paddings: Optional[Union[int, List[int]]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        pool_layer: Optional[Callable[..., nn.Module]] = nn.MaxPool2d,
    ):
        """
        Initialize the CNN backbone.

        Args:
            in_channels: Number of input channels (1 for grayscale, 3 for RGB)
            conv_channels: List of channel sizes for each convolutional block
            kernel_sizes: Kernel size(s) for convolutional layers (single int or list)
            paddings: Padding(s) for convolutional layers (single int, list, or None to auto-calculate)
            norm_layer: Normalization layer to use (default: BatchNorm2d)
            activation_layer: Activation function to use (default: ReLU)
            pool_layer: Pooling layer to use (default: MaxPool2d)
        """
        super().__init__()

        # Normalize kernel_sizes and paddings to lists if they are integers
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(conv_channels)
        if isinstance(paddings, int):
            paddings = [paddings] * len(conv_channels)

        # Create convolutional and pooling stages
        self.stages = nn.ModuleList()

        prev_channels = in_channels
        for i, out_channels in enumerate(conv_channels):
            # Create a sequential block with Conv2dNormActivation and pooling
            stage = nn.Sequential()

            # Add Conv2dNormActivation
            stage.append(
                ops.Conv2dNormActivation(
                    in_channels=prev_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_sizes[i],
                    padding=paddings[i] if paddings is not None else None,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

            # Add pooling if specified
            if pool_layer is not None:
                stage.append(
                    pool_layer(2)
                )  # NOTE: Assuming fixed 2x2 pooling (for now)

            self.stages.append(stage)
            prev_channels = out_channels

        # Store output channels for head compatibility check
        self.__out_channels = prev_channels

    @property
    def out_channels(self) -> int:
        """
        Get the number of output channels from the backbone.
        This should match the input channels of the head.
        """
        return self.__out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process through stages
        for stage in self.stages:
            x = stage(x)

        return x
