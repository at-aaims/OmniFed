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

import torch
import numpy as np
import random
from _random import Random

def set_seed(seed=1234, determinism=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    rng = Random()
    rng.seed(seed)
    torch.use_deterministic_algorithms(determinism)

class Partition(object):
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class SplitData(object):
    """Splits training data evenly based on the total number of clients."""
    def __init__(self, data, total_clients):
        self.data = data
        self.partitions = []
        # partition data equally among the trainers
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        random.shuffle(indexes)

        partitions = [1 / (total_clients) for _ in range(0, total_clients)]
        print(f"partitions are {partitions}")

        for part in partitions:
            part_len = int(part * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


# TODO: fix the split based on id annotation across clients and server
class UnsplitData(object):
    """Unlike split partitioner which evenly distributes data among all workers, unsplit partitioner keeps all
    data on each worker and shuffles it in a different order"""
    def __init__(self, data, client_id):
        self.data = data
        self.partitions = []
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        random.seed(client_id)
        random.shuffle(indexes)
        self.partitions.append(indexes)

    def use(self, partition_ix):
        return Partition(self.data, self.partitions[partition_ix])