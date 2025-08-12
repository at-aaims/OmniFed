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

import inspect
from typing import Optional

from rich import print as rich_print
from rich.color import ANSI_COLOR_NAMES
from rich.rule import Rule

ANSI_COLOR_NAMES_LIST = [
    name
    for name in ANSI_COLOR_NAMES.keys()
    if not any(excluded in name for excluded in ["gray", "grey", "black"])
]


def _get_color_for_prefix(prefix: str) -> str:
    """Get consistent color for a prefix using hash."""
    color_names = ANSI_COLOR_NAMES_LIST
    color_hash = hash(prefix) % len(color_names)
    return color_names[color_hash]


def _get_caller_prefix() -> str:
    """Get caller function/class name for logging prefix."""
    # Get caller information to include function/class name if available
    stack = inspect.stack()
    # Skip this function and get the actual caller
    caller_function = stack[2].function
    caller_frame = stack[2].frame

    # Check if called from within a class method (self exists)
    if "self" in caller_frame.f_locals:
        class_name = caller_frame.f_locals["self"].__class__.__name__
        prefix = f"{class_name}->{caller_function}"
    else:
        # Just use function name if no class context
        prefix = caller_function

    return prefix


def print_rule(msg: Optional[str] = None, characters: str = "━") -> None:
    """Print a separator line with caller context.

    Args:
        msg: Optional message to display with the rule
        characters: Characters to use for the rule
    """
    prefix = _get_caller_prefix()
    color = _get_color_for_prefix(prefix.split("->")[0])
    prefix_colored = f"[bold {color}]{prefix}[/bold {color}]"

    rich_print()

    # Print prefix rule first (always center-aligned)
    rich_print(
        Rule(
            prefix_colored,
            style=color,
            characters=characters,
            align="center",
        )
    )

    # Print message below if provided
    if msg:
        rich_print(
            Rule(
                msg,
                style=color,
                characters=characters,
                align="center",
            )
        )


def print(*args, **kwargs) -> None:
    """Print with caller context prefix."""
    prefix = _get_caller_prefix()
    color = _get_color_for_prefix(prefix.split("->")[0])
    prefix = f"[bold {color}]{prefix}[/bold {color}]"
    rich_print(f"[{prefix}]", *args, **kwargs)
