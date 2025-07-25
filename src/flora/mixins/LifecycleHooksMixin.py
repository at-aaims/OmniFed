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
    Mixin providing standardized lifecycle hooks for federated learning components.
    
    Defines optional override points throughout the federated learning execution
    lifecycle for algorithm-specific customization.
    """

    def _experiment_start(self) -> None:
        """
        Optional lifecycle hook: Called at the start of the federated learning experiment.
        
        Override in subclasses for experiment initialization logic such as
        initial model evaluation or baseline metrics collection.
        """
        pass

    def _experiment_end(self) -> None:
        """
        Optional lifecycle hook: Called at the end of the federated learning experiment.
        
        Override in subclasses for experiment finalization logic such as
        final model evaluation or results summarization.
        """
        pass

    def _round_start(self) -> None:
        """
        Optional lifecycle hook: Called at the start of each federated learning round.

        Override in subclasses for algorithm-specific round initialization logic
        such as model synchronization and state reset.
        """
        pass

    def _round_end(self) -> None:
        """
        Optional lifecycle hook: Called at the end of each federated learning round.

        Override in subclasses for custom round finalization logic.
        """
        pass

    def _train_epoch_start(self) -> None:
        """
        Optional lifecycle hook: Called at the start of each local training epoch.

        Override in subclasses for custom epoch initialization logic.
        """
        pass

    def _train_epoch_end(self) -> None:
        """
        Optional lifecycle hook: Called at the end of each local training epoch.

        Override in subclasses for custom epoch finalization logic.
        """
        pass

    def _on_sync_start(self) -> None:
        """
        Optional lifecycle hook: Called before synchronization operations begin.
        
        Override in subclasses for custom pre-synchronization logic.
        """
        pass

    def _on_sync_end(self) -> None:
        """
        Optional lifecycle hook: Called after synchronization operations complete.
        
        Override in subclasses for custom post-synchronization logic.
        """
        pass

    def _train_batch_start(self) -> None:
        """
        Optional lifecycle hook: Called before processing each training batch.

        Override in subclasses for custom batch initialization logic.
        """
        pass

    def _train_batch_end(self) -> None:
        """
        Optional lifecycle hook: Called after processing each training batch.

        Override in subclasses for custom batch finalization logic.
        """
        pass

    def _eval_epoch_start(self) -> None:
        """
        Optional lifecycle hook: Called at the start of each evaluation epoch.

        Override in subclasses for custom evaluation initialization logic.
        """
        pass

    def _eval_epoch_end(self) -> None:
        """
        Optional lifecycle hook: Called after evaluation epoch completion.

        Override in subclasses for custom evaluation finalization logic.
        """
        pass

    def _eval_batch_start(self) -> None:
        """
        Optional lifecycle hook: Called before processing each evaluation batch.

        Override in subclasses for custom evaluation batch initialization logic.
        """
        pass

    def _eval_batch_end(self) -> None:
        """
        Optional lifecycle hook: Called after processing each evaluation batch.

        Override in subclasses for custom evaluation batch finalization logic.
        """
        pass