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

from typing import Optional
from torch.utils.data import DataLoader


# ======================================================================================


class DataModule:
    """
    Base class for data modules in FLORA.
    This class encapsulates the training, validation, and test DataLoaders.
    """

    def __init__(
        self,
        train: DataLoader,
        val: Optional[DataLoader] = None,
        test: Optional[DataLoader] = None,
    ):
        """
        Initializes the DataModule with train, validation, and test DataLoaders.

        :param train: DataLoader for training data
        :param val: DataLoader for validation data
        :param test: DataLoader for test data
        """
        print(f"{self.__class__.__name__} init...")

        self.train: DataLoader = train
        self.val: Optional[DataLoader] = val
        self.test: Optional[DataLoader] = test

        if self.val is None:
            print("Warning: Validation DataLoader is not provided.")
        if self.test is None:
            print("Warning: Test DataLoader is not provided.")
