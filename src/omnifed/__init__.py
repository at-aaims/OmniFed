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

from __future__ import annotations

import importlib
from typing import Any, Tuple

__all__ = (
    "Engine",
    "EngineConfig",
    "RayConfig",
    "Node",
    "NodeConfig",
    "RayActorConfig",
    "utils",
)


def __getattr__(name: str) -> Any:
    if name == "utils":
        # Must not use ``from . import utils`` here: that re-enters ``__getattr__``
        # and overflows the stack (utils is a subpackage, not a re-export).
        return importlib.import_module(".utils", __name__)
    if name == "Engine":
        from .engine import Engine

        return Engine
    if name == "EngineConfig":
        from .engine import EngineConfig

        return EngineConfig
    if name == "RayConfig":
        from .engine import RayConfig

        return RayConfig
    if name == "Node":
        from .node import Node

        return Node
    if name == "NodeConfig":
        from .node import NodeConfig

        return NodeConfig
    if name == "RayActorConfig":
        from .node import RayActorConfig

        return RayActorConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> Tuple[str, ...]:
    return __all__
