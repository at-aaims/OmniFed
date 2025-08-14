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

from dataclasses import dataclass
from typing import Dict, List, Optional

from omegaconf import MISSING, DictConfig

from ..communicator import BaseCommunicatorConfig


@dataclass
class BaseTopologyConfig:
    """Base configuration for all topology types."""

    _target_: str = "src.omnifed.topology.BaseTopology"


@dataclass
class CentralizedTopologyConfig(BaseTopologyConfig):
    """Configuration for CentralizedTopology."""

    _target_: str = "src.omnifed.topology.CentralizedTopology"

    # Required parameters
    num_clients: int = MISSING
    local_comm: BaseCommunicatorConfig = MISSING

    # Optional parameters
    overrides: Optional[Dict[int, DictConfig]] = None


@dataclass
class HierarchicalTopologyConfig(BaseTopologyConfig):
    """Configuration for HierarchicalTopology."""

    _target_: str = "src.omnifed.topology.HierarchicalTopology"

    # Required parameters
    groups: List[DictConfig] = MISSING
    global_comm: BaseCommunicatorConfig = MISSING
