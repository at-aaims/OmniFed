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

    Special cases:
    - every=0: Run only at step 0 (initial evaluation)
    - every=None, at=[]: Never run (natural disabled state)
    """

    every: Optional[int] = None
    at: List[int] = field(default_factory=list)

    def __post_init__(self):
        """Validate schedule configuration."""
        # Validate frequency (allow 0 for "step 0 only", reject negative)
        if self.every is not None and self.every < 0:
            raise ValueError(
                f"Schedule frequency 'every' must be non-negative, got {self.every}"
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
        # Natural disabled state: no frequency AND no milestones
        if self.every is None and not self.at:
            return False

        # Check frequency-based execution
        if self.every is not None and current_step >= 0:
            # Special case: every=0 means "run only at step 0"
            if self.every == 0:
                return current_step == 0
            # Normal case: run every N steps (including step 0)
            else:
                return current_step % self.every == 0

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
            pre_aggregation=Schedule(every=5),
            post_aggregation=Schedule(at=[10, 20, 50])
        )
    """

    # Essential FL evaluation points
    experiment_start: bool = True
    pre_aggregation: Schedule = field(default_factory=Schedule)
    post_aggregation: Schedule = field(default_factory=Schedule)
    experiment_end: bool = True
