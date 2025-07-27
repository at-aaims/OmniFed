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

from typing import Any, Optional
import rich
from torch.utils.data import DataLoader
import rich.repr
from typeguard import typechecked

# ======================================================================================


@rich.repr.auto
class DataModule:
    """
    Base class for data modules in FLORA.
    This class encapsulates the training and evaluation DataLoaders.
    """

    @typechecked
    def __init__(
        self,
        train: Optional[DataLoader[Any]] = None,
        eval: Optional[DataLoader[Any]] = None,
    ):
        """
        Initializes the DataModule with train and evaluation DataLoaders.

        :param train: DataLoader for training data
        :param eval: DataLoader for evaluation data
        """
        print("[DATAMODULE-INIT]")

        self.train: Optional[DataLoader[Any]] = train
        self.eval: Optional[DataLoader[Any]] = eval

        # if self.train is None:
        #     print("NOTE: Training DataLoader is not provided.")
        # if self.eval is None:
        #     print("NOTE: Evaluation DataLoader is not provided.")

        # ---

    # def __str__(self) -> str:
    #     """
    #     Returns a compact, single-line string representation of the DataModule.
    #     """

    #     def loader_info(name, loader):
    #         if loader is not None:
    #             num_samples = len(loader.dataset) if hasattr(loader, "dataset") else "?"
    #             batch_size = loader.batch_size if hasattr(loader, "batch_size") else "?"
    #             return f"{name}: {num_samples} samples, batch_size={batch_size}"
    #         else:
    #             return f"{name}: None"

    #     return (
    #         f"{self.__class__.__name__}("
    #         f"{loader_info('train', self.train)} | "
    #         f"{loader_info('eval', self.eval)} | "
    #         f"{loader_info('test', self.test)})"
    #     )
