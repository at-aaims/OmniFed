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
from typing import Optional, List, Literal

from hydra.core.config_store import ConfigStore

# ======================================================================================

# Type definitions for better IDE support and type safety
TriggerName = Literal[
    "experiment_start",
    "experiment_end",
    "round_start",
    "round_end",
    "epoch_start",
    "epoch_end",
    "batch_start",
    "batch_end",
    "pre_aggregation",
    "post_aggregation",
]


@dataclass
class Trigger:
    """Trigger configuration with Hydra support."""

    _target_: str = field(default="src.flora.utils.Scheduling.Trigger", init=False)
    """
    Trigger for validation/evaluation execution in federated learning.

    Supports frequency-based (every N steps) and milestone-based (at specific steps) scheduling.
    Steps are interpreted based on the algorithm's aggregation level (rounds, epochs, or iterations).

    Special cases:
    - every=0: Run only at step 0 (initial evaluation)
    - every=None, at=[]: Never run (natural disabled state)

    Examples:
        >>> # Every 5 steps
        >>> Trigger(every=5)

        >>> # At specific milestones
        >>> Trigger(at=[1, 5, 10])

        >>> # Only at start
        >>> Trigger(every=0)

        >>> # Never run
        >>> Trigger()

        >>> # Convenience methods
        >>> Trigger.always()  # every=1
        >>> Trigger.never()   # disabled
        >>> Trigger.once_at_start()  # every=0
    """

    every: Optional[int] = None
    at: List[int] = field(default_factory=list)

    def __post_init__(self):
        """Validate trigger configuration."""
        # Validate frequency (allow 0 for "step 0 only", reject negative)
        if self.every is not None and self.every < 0:
            raise ValueError(
                f"Trigger frequency 'every' must be non-negative, got {self.every}. "
                f"Use every=0 for 'step 0 only' or omit for disabled trigger."
            )

        # Validate and sort milestones
        if self.at:
            invalid_milestones = [m for m in self.at if m < 0]
            if invalid_milestones:
                raise ValueError(
                    f"Trigger milestones must be non-negative, got: {invalid_milestones}. "
                    f"Milestones represent step indices starting from 0."
                )
            # Remove duplicates and sort for efficiency
            self.at = sorted(set(self.at))

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

    @staticmethod
    def should_run_any(*triggers: "Trigger", current_step: int) -> bool:
        """True if any of the provided triggers should run at current_step."""
        return any(trigger.should_run(current_step) for trigger in triggers)

    @staticmethod
    def should_run_all(*triggers: "Trigger", current_step: int) -> bool:
        """True if all of the provided triggers should run at current_step."""
        return all(trigger.should_run(current_step) for trigger in triggers)


@dataclass
class LifecycleTriggers:
    """LifecycleTriggers configuration with Hydra support."""

    _target_: str = field(
        default="src.flora.utils.Scheduling.LifecycleTriggers", init=False
    )
    """
    Defines triggers for an operation at different lifecycle points in federated learning.

    This class provides a unified interface for scheduling operations across the entire
    federated learning lifecycle, from experiment start to individual batch processing.

    Lifecycle Points:
    - experiment_start/end: Boolean flags for experiment boundaries
    - round_start/end: Per-round triggers using round_idx
    - epoch_start/end: Per-epoch triggers using epoch_idx
    - batch_start/end: Per-batch triggers using batch_idx
    - pre/post_aggregation: Aggregation-related triggers (can use multiple indices)

    Examples:
        >>> # Evaluation after every aggregation
        >>> triggers = LifecycleTriggers(
        ...     experiment_start=True,
        ...     post_aggregation=Trigger.always(),
        ...     experiment_end=True
        ... )

        >>> # Aggregation every 2nd round, 3rd epoch
        >>> triggers = LifecycleTriggers(
        ...     round_end=Trigger(every=2),
        ...     epoch_end=Trigger(every=3)
        ... )

        >>> # Check if should run
        >>> if triggers.should_run('post_aggregation', round_idx=5, epoch_idx=2):
        ...     print("Run evaluation!")
    """

    experiment_start: bool = False
    experiment_end: bool = False
    round_start: Trigger = field(default_factory=Trigger)
    round_end: Trigger = field(default_factory=Trigger)
    epoch_start: Trigger = field(default_factory=Trigger)
    epoch_end: Trigger = field(default_factory=Trigger)
    batch_start: Trigger = field(default_factory=Trigger)
    batch_end: Trigger = field(default_factory=Trigger)
    # For evaluation-specific triggers
    pre_aggregation: Trigger = field(default_factory=Trigger)
    post_aggregation: Trigger = field(default_factory=Trigger)

    def should_run(self, trigger_name: TriggerName, **indices) -> bool:
        """
        Check if operation should run at the specified lifecycle trigger.

        Args:
            trigger_name: Name of the lifecycle trigger (e.g., 'round_end', 'batch_start')
            **indices: Named indices for the trigger (e.g., round_idx=5, epoch_idx=2)

        Returns:
            True if should execute at this trigger, False otherwise

        Raises:
            ValueError: If trigger_name is unknown or required indices are missing
        """
        # Handle boolean triggers
        if trigger_name in ("experiment_start", "experiment_end"):
            result = getattr(self, trigger_name)
            if result:
                print(f"[TRIG_{trigger_name.upper()}] Trigger activated")
            return result

        # Get trigger object
        trigger_obj = getattr(self, trigger_name, None)
        if trigger_obj is None:
            # Filter out internal fields for cleaner error message
            valid_triggers = [
                name
                for name in self.__dataclass_fields__.keys()
                if not name.startswith("_")
            ]
            raise ValueError(
                f"Unknown trigger '{trigger_name}'. Valid triggers: {valid_triggers}"
            )

        # For aggregation triggers, check all provided indices
        if trigger_name in ("pre_aggregation", "post_aggregation"):
            if not indices:
                raise ValueError(
                    f"Aggregation trigger '{trigger_name}' requires at least one index "
                    f"(e.g., round_idx=0, epoch_idx=1, batch_idx=5)"
                )
            result = any(trigger_obj.should_run(idx) for idx in indices.values())
            if result:
                matching_indices = {
                    k: v for k, v in indices.items() if trigger_obj.should_run(v)
                }
                print(
                    f"[TRIG_{trigger_name.upper()}] Trigger activated at indices: {matching_indices}"
                )
            return result

        # For lifecycle triggers, use the semantically correct index
        index_mapping = {
            "round_start": "round_idx",
            "round_end": "round_idx",
            "epoch_start": "epoch_idx",
            "epoch_end": "epoch_idx",
            "batch_start": "batch_idx",
            "batch_end": "batch_idx",
        }

        expected_index = index_mapping[trigger_name]
        if expected_index not in indices:
            provided = list(indices.keys()) if indices else ["none"]
            raise ValueError(
                f"Trigger '{trigger_name}' requires '{expected_index}' parameter. "
                f"Provided: {provided}. "
                f"Usage: should_run('{trigger_name}', {expected_index}=<value>)"
            )

        result = trigger_obj.should_run(indices[expected_index])
        if result:
            print(
                f"[TRIG_{trigger_name.upper()}] Trigger activated at {expected_index}={indices[expected_index]}"
            )
        return result


@dataclass
class Schedules:
    """Schedules configuration with Hydra support."""

    _target_: str = field(default="src.flora.utils.Scheduling.Schedules", init=False)
    """
    Container for all scheduled operations in federated learning.

    This class organizes scheduling for the two main federated learning operations:
    aggregation (model parameter synchronization) and evaluation (metric computation).

    Default Behavior:
    - aggregation: Round-end aggregation enabled (Trigger(every=1))
    - evaluation: All evaluation disabled (empty LifecycleTriggers)

    Examples:
        >>> # Use defaults
        >>> schedules = Schedules()

        >>> # Custom scheduling
        >>> schedules = Schedules(
        ...     aggregation=LifecycleTriggers(
        ...         round_end=Trigger.always(),
        ...         epoch_end=Trigger(every=2)
        ...     ),
        ...     evaluation=LifecycleTriggers(
        ...         experiment_start=True,
        ...         post_aggregation=Trigger.always(),
        ...         experiment_end=True
        ...     )
        ... )
    """

    aggregation: LifecycleTriggers = field(
        default_factory=lambda: LifecycleTriggers(
            round_end=Trigger(every=1)  # Default: aggregate every round
        )
    )
    evaluation: LifecycleTriggers = field(default_factory=LifecycleTriggers)


# Register with ConfigStore for clean YAML config composition
cs = ConfigStore.instance()

# One good default schedule pattern
cs.store(
    group="schedules",
    name="base",
    node=Schedules(
        aggregation=LifecycleTriggers(round_end=Trigger(every=1)),
        evaluation=LifecycleTriggers(
            experiment_start=True,
            post_aggregation=Trigger(every=1),
            experiment_end=True,
        ),
    ),
)
