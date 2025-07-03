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

import numpy as np
import torch
import torchtext

from ..utils import set_seed

# TODO: fix C++ ABI mismatch between torch==2.6.0 and torchtext==0.18.0 (version mismatch between torchtext 0.17.1 also)


def imdbReviewsData(
    client_id=0,
    total_clients=1,
    datadir="~/",
    partition_dataset=True,
    train_bsz=32,
    test_bsz=32,
    is_test=True,
    get_training_dataset=False,
):
    """
    :param client_id: id/rank of client or server
    :param total_clients: total number of clients/world-size
    :param datadir: where to download/read the data
    :param partition_dataset: whether to partition dataset among clients in unique chunks or just shuffle
    entire dataset across clients in different order
    :param train_bsz: training batch size
    :param test_bsz: test batch size
    :param is_test: process test/validation set dataloader or send None value
    :param get_training_dataset: whether to get training dataset or train/test dataloader
    :return: train and test dataloaders
    """
    set_seed(seed=total_clients)
    g = torch.Generator()
    g.manual_seed(total_clients)

    # Tokenization and lowercasing
    TEXT = torchtext.data.Field(tokenize="spacy", lower=True)
    # Labels as floating-point numbers
    LABEL = torchtext.data.LabelField(dtype=torch.float)

    # Load the IMDb dataset
    train_data, test_data = torchtext.datasets.IMDB.splits(TEXT, LABEL)

    # Build the vocabulary (includes padding and unknown tokens)
    TEXT.build_vocab(
        train_data,
        max_size=25000,
        vectors="glove.6B.100d",
        unk_init=torch.Tensor.normal_,
    )
    LABEL.build_vocab(train_data)

    if partition_dataset:
        num_samples = len(train_data)
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
        train_data = torch.utils.data.Subset(train_data, chunk_indices)

    if get_training_dataset:
        return train_data
    else:
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=train_bsz,
            shuffle=True,
            worker_init_fn=set_seed(client_id),
            generator=g,
            num_workers=4,
        )
        del train_data

        if is_test:
            test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=test_bsz, shuffle=True, generator=g, num_workers=4
            )
        else:
            test_loader = None

        # iterate over the training dataset
        # for batch in train_loader:
        #     text, text_lengths = batch.text
        #     labels = batch.label
        #     print(text.shape)  # Shape of the text data
        #     print(labels.shape)  # Shape of the labels
        #     print(text_lengths.shape)  # Lengths of the text sequences
        #     break

        return train_loader, test_loader
