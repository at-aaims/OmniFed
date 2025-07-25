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

from dataclasses import dataclass, field
from typing import Optional, List

# ======================================================================================


@dataclass
class Schedule:
    """
    Schedule for validation/evaluation execution in federated learning.

    Supports frequency-based (every N steps) and milestone-based (at specific steps) scheduling.
    Steps are interpreted based on the algorithm's aggregation level (rounds, epochs, or iterations).
    """

    enabled: bool = False
    every: Optional[int] = None
    at: List[int] = field(default_factory=list)

    def __post_init__(self):
        """Validate schedule configuration."""
        if self.enabled:
            # Check for valid configuration
            if self.every is None and not self.at:
                raise ValueError(
                    "Schedule enabled but no frequency (every) or milestones (at) specified"
                )

            # Validate frequency
            if self.every is not None and self.every <= 0:
                raise ValueError(
                    f"Schedule frequency 'every' must be positive, got {self.every}"
                )

            # Validate and sort milestones
            if self.at:
                invalid_milestones = [m for m in self.at if m < 0]
                if invalid_milestones:
                    raise ValueError(
                        f"Schedule milestones must be non-negative, got: {invalid_milestones}"
                    )
                self.at = sorted(self.at)  # Sort for efficiency

    def should_run(self, current_step: int) -> bool:
        """
        Check if execution should happen at the current step.

        Args:
            current_step: Current step index (round, epoch, or iteration based on context)

        Returns:
            True if should execute at this step, False otherwise
        """
        if not self.enabled:
            return False

        # Check frequency-based execution
        # Allow evaluation at step 0 to enable initial evaluation
        if (
            self.every is not None
            and current_step >= 0
            and current_step % self.every == 0
        ):
            return True

        # Check milestone-based execution
        if current_step in self.at:
            return True

        return False


@dataclass
class EvalSchedule:
    """
    Configuration for evaluation scheduling in federated learning.

    Provides four evaluation points covering practical FL evaluation needs:
    - experiment_start/end: Boolean flags for initial and final evaluations
    - pre/post_aggregation: Schedules for local and global model evaluations

    Example:
        EvalSchedule(
            pre_aggregation=Schedule(enabled=True, every=5),
            post_aggregation=Schedule(enabled=True, at=[10, 20, 50])
        )
    """

    # Essential FL evaluation points
    experiment_start: bool = True
    pre_aggregation: Schedule = field(default_factory=lambda: Schedule(enabled=False))
    post_aggregation: Schedule = field(
        default_factory=lambda: Schedule(enabled=True, every=1)
    )
    experiment_end: bool = True
