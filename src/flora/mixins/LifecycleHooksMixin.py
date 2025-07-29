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

from abc import ABC


class LifecycleHooksMixin(ABC):
    """
    Optional lifecycle hooks for federated learning algorithm customization.

    Provides standardized extension points during FL execution for algorithm-specific
    logic like custom metrics, logging, or state management.

    All hooks are optional - override only what your algorithm needs.
    Default implementations are no-ops.

    Hook execution order:
    ```
    _round_start()
      _train_epoch_start()
        _train_batch_start()
        _train_batch_end()
      _train_epoch_end()
      _eval_epoch_start()
        _eval_batch_start()
        _eval_batch_end()
      _eval_epoch_end()
    _round_end()
    ```
    """

    def _round_start(self) -> None:
        """Called at start of each FL round."""
        pass

    def _round_end(self) -> None:
        """Called at end of each FL round."""
        pass

    def _train_epoch_start(self) -> None:
        """Called at start of each training epoch."""
        pass

    def _train_epoch_end(self) -> None:
        """Called at end of each training epoch."""
        pass

    def _train_batch_start(self) -> None:
        """Called before processing each training batch."""
        pass

    def _train_batch_end(self) -> None:
        """Called after processing each training batch."""
        pass

    def _eval_epoch_start(self) -> None:
        """Called at start of each evaluation epoch."""
        pass

    def _eval_epoch_end(self) -> None:
        """Called at end of each evaluation epoch."""
        pass

    def _eval_batch_start(self) -> None:
        """Called before processing each evaluation batch."""
        pass

    def _eval_batch_end(self) -> None:
        """Called after processing each evaluation batch."""
        pass
