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

from abc import ABC, abstractmethod

import rich.repr
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


@rich.repr.auto
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
