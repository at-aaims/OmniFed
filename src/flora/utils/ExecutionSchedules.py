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

from typing import Optional, List, Union

from typeguard import typechecked

# ======================================================================================


class Trigger:
    """
    Defines when an operation should execute during federated learning.

    Each trigger maintains its own call counter and determines execution
    based on how many times it has been called.

    Examples:
        Trigger(every=1)     # Every call
        Trigger(every=5)     # Every 5th call
        Trigger(at=[0,10])   # At calls 0 and 10
        Trigger()            # Never
    """

    def __init__(self, every: Optional[int] = None, at: Optional[List[int]] = None):
        self.every = every
        self.at = at or []
        self._step_counter = 0

        if self.every is not None and self.every < 0:
            raise ValueError(f"'every' must be non-negative, got {self.every}")
        if self.at:
            invalid = [m for m in self.at if m < 0]
            if invalid:
                raise ValueError(f"'at' values must be non-negative, got: {invalid}")
            self.at = sorted(set(self.at))

    def __call__(self) -> bool:
        """Check if should run and increment internal counter."""
        should_run = self._should_run(self._step_counter)
        self._step_counter += 1
        return should_run

    def _should_run(self, step: int) -> bool:
        """Check if should run at given step."""
        if self.every is None and not self.at:
            return False
        if self.every is not None:
            if self.every == 0:
                return False
            return step % self.every == 0
        return step in self.at

    def __repr__(self):
        return f"Trigger(every={self.every}, at={self.at})"

    @classmethod
    def always(cls) -> "Trigger":
        """Create a trigger that always runs."""
        return cls(every=1)

    @classmethod
    def never(cls) -> "Trigger":
        """Create a trigger that never runs."""
        return cls(every=0, at=[])


@typechecked
def resolve_trigger(val: Union[Trigger, bool, None]) -> Trigger:
    """Convert config value to Trigger instance."""
    if val is None or val is False:
        return Trigger.never()
    if val is True:
        return Trigger.always()
    return val


class AggregationTriggers:
    """Aggregation schedule."""

    def __init__(
        self,
        round_end: Union[Trigger, bool, None] = True,
        epoch_end: Union[Trigger, bool, None] = True,
        batch_end: Union[Trigger, bool, None] = True,
    ):
        self.round_end: Trigger = resolve_trigger(round_end)
        self.epoch_end: Trigger = resolve_trigger(epoch_end)
        self.batch_end: Trigger = resolve_trigger(batch_end)


class EvaluationTriggers:
    """Evaluation schedule."""

    def __init__(
        self,
        experiment_start: Union[Trigger, bool, None] = True,
        experiment_end: Union[Trigger, bool, None] = False,
        pre_aggregation: Union[Trigger, bool, None] = False,
        post_aggregation: Union[Trigger, bool, None] = False,
    ):
        self.experiment_start: Trigger = resolve_trigger(experiment_start)
        self.experiment_end: Trigger = resolve_trigger(experiment_end)
        self.pre_aggregation: Trigger = resolve_trigger(pre_aggregation)
        self.post_aggregation: Trigger = resolve_trigger(post_aggregation)


class ExecutionSchedules:
    """
    Federated learning execution scheduler.
    """

    def __init__(
        self,
        aggregation: Optional[AggregationTriggers] = None,
        evaluation: Optional[EvaluationTriggers] = None,
    ):
        self.aggregation: AggregationTriggers = aggregation or AggregationTriggers()
        self.evaluation: EvaluationTriggers = evaluation or EvaluationTriggers()
