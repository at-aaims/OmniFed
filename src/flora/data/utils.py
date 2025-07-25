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

import random
from _random import Random

import numpy as np
import torch


def set_seed(seed=1234, determinism=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    rng = Random()
    rng.seed(seed)
    torch.use_deterministic_algorithms(determinism)


def split_into_chunks(dataset, client_id=0, total_clients=1):
    """
    :param dataset: A torchvision dataset
    :param total_clients: total number of clients/world-size
    :return: unique data subset for each client
    """
    num_samples = len(dataset)
    chunk_size = num_samples // total_clients
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    partitions = []
    start_index = client_id * chunk_size
    if client_id == total_clients - 1:
        chunk_indices = indices[start_index:]
    else:
        chunk_indices = indices[start_index : start_index + chunk_size]

    # Create a subset for the current chunk
    chunk_dataset = torch.utils.data.Subset(dataset, chunk_indices)
    partitions.append(chunk_dataset)
    partitioned_dataset = partioneDataset(data=partitions)
    del partitions

    return partitioned_dataset


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


class partioneDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        """
        Initialize the dataset with a list.

        Args:
            data (list): A list of tuples (image, label).
        """
        self.data = data

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a sample by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the data (e.g., image) and the label.
        """
        image, label = self.data[idx]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(
            label, dtype=torch.long
        )


class DataSubsetNonIID(torch.utils.data.Dataset):
    def __init__(self, dataset, labels_per_rank):
        self.dataset = dataset
        self.indices = [
            i
            for i, label in enumerate(self.dataset.targets)
            if label in labels_per_rank
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return self.dataset[real_idx]


def get_labels_per_client(number_of_labels=1, total_clients=1):
    partitions = []
    for i in range(total_clients):
        start = (i * number_of_labels) // total_clients
        end = ((i + 1) * number_of_labels) // total_clients
        partitions.append(list(range(start, end)))

    return partitions
