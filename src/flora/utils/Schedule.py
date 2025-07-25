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

    Supports both frequency-based (every N steps) and milestone-based (at specific steps) scheduling.
    Steps are interpreted based on the algorithm's aggregation level (rounds, epochs, or iterations).

    Attributes:
        enabled: Whether this schedule is active
        every: Run every N steps (e.g., every 5 rounds)
        at: Run at specific step milestones (e.g., at rounds [50, 100, 200])
    """

    enabled: bool = False
    every: Optional[int] = None
    at: List[int] = field(default_factory=list)

    def __post_init__(self):
        """Validate schedule configuration."""
        if self.enabled:
            # Check for valid configuration
            if self.every is None and not self.at:
                print(
                    "[WARNING] Schedule enabled but no frequency (every) or milestones (at) specified",
                    flush=True,
                )

            # Validate frequency
            if self.every is not None and self.every <= 0:
                print(
                    f"[WARNING] Schedule frequency 'every' must be positive, got {self.every}",
                    flush=True,
                )
                self.every = None

            # Validate milestones
            if self.at:
                valid_milestones = [m for m in self.at if m >= 0]
                if len(valid_milestones) < len(self.at):
                    print(
                        f"[WARNING] Negative milestones ignored in schedule: {[m for m in self.at if m < 0]}",
                        flush=True,
                    )
                self.at = sorted(valid_milestones)  # Sort for efficiency

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
        # Allow evaluation at step 0 to enable initial/baseline evaluation
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

    Supports flexible evaluation timing with frequency-based and milestone-based scheduling.
    Can combine both approaches for maximum research flexibility.

    Attributes:
        local_model: Local model evaluation schedule configuration
        global_model: Global model evaluation schedule configuration
    """

    local_model: Schedule = field(default_factory=lambda: Schedule(enabled=False))
    global_model: Schedule = field(default_factory=lambda: Schedule(enabled=True, every=1))

    def should_run_local(self, current_step: int) -> bool:
        """Check if local evaluation should run at current step."""
        return self.local_model.should_run(current_step)

    def should_run_global(self, current_step: int) -> bool:
        """Check if global evaluation should run at current step."""
        return self.global_model.should_run(current_step)
