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
import torchvision
import torchvision.transforms as transforms

from ..utils import SplitData, UnsplitData, set_seed

# TODO: adjust num_workers in torch.utils.data.DataLoader based on total threads available on a client
# TODO: EMNIST, FMNIST, etc. from torchvision.datasets module


def food101Data(
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

    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    training_set = torchvision.datasets.Food101(
        root=datadir, split="train", transform=transform, download=True
    )
    if partition_dataset:
        training_set = SplitData(data=training_set, total_clients=total_clients)
        training_set = training_set.use(client_id)
    else:
        training_set = UnsplitData(data=training_set, client_id=client_id)
        training_set = training_set.use(0)

    if get_training_dataset:
        return training_set
    else:
        train_loader = torch.utils.data.DataLoader(
            training_set,
            batch_size=train_bsz,
            shuffle=True,
            worker_init_fn=set_seed(client_id),
            generator=g,
            num_workers=4,
        )

        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        if is_test:
            test_set = torchvision.datasets.Food101(
                root=datadir, split="test", transform=transform, download=True
            )
            test_loader = torch.utils.data.DataLoader(
                test_set, batch_size=test_bsz, shuffle=True, generator=g, num_workers=4
            )
            del test_set
        else:
            test_loader = None

        return train_loader, test_loader


def places365Data(
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

    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    training_set = torchvision.datasets.Places365(
        root=datadir, split="train", transform=transform, download=True
    )

    if partition_dataset:
        training_set = SplitData(data=training_set, total_clients=total_clients)
        training_set = training_set.use(client_id)
    else:
        training_set = UnsplitData(data=training_set, client_id=client_id)
        training_set = training_set.use(0)

    if get_training_dataset:
        return training_set
    else:
        train_loader = torch.utils.data.DataLoader(
            training_set,
            batch_size=train_bsz,
            shuffle=True,
            worker_init_fn=set_seed(client_id),
            generator=g,
            num_workers=4,
        )
        del training_set

        if is_test:
            test_set = torchvision.datasets.Places365(
                root=datadir, split="test", transform=transform, download=True
            )
            test_loader = torch.utils.data.DataLoader(
                test_set, batch_size=test_bsz, shuffle=True, generator=g, num_workers=4
            )
            del test_set
        else:
            test_loader = None

        return train_loader, test_loader


def emnistData(
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
    EMNIST dataset for handwritten digits
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

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]
    )
    training_set = torchvision.datasets.EMNIST(
        root=datadir, split="letters", train=True, download=True, transform=transform
    )

    if partition_dataset:
        training_set = SplitData(data=training_set, total_clients=total_clients)
        training_set = training_set.use(client_id)
    else:
        training_set = UnsplitData(data=training_set, client_id=client_id)
        training_set = training_set.use(0)

    if get_training_dataset:
        return training_set
    else:
        train_loader = torch.utils.data.DataLoader(
            training_set,
            batch_size=train_bsz,
            shuffle=True,
            worker_init_fn=set_seed(client_id),
            generator=g,
            num_workers=4,
        )
        del training_set

        if is_test:
            test_set = torchvision.datasets.EMNIST(
                root=datadir,
                split="letters",
                train=False,
                download=True,
                transform=transform,
            )
            test_loader = torch.utils.data.DataLoader(
                test_set, batch_size=test_bsz, shuffle=True, generator=g, num_workers=4
            )
            del test_set
        else:
            test_loader = None

        return train_loader, test_loader


def fmnistData(
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
    Fashion MNIST dataset
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

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]
    )
    training_set = torchvision.datasets.FashionMNIST(
        root=datadir, train=True, download=True, transform=transform
    )
    if partition_dataset:
        training_set = SplitData(data=training_set, total_clients=total_clients)
        training_set = training_set.use(client_id)
    else:
        training_set = UnsplitData(data=training_set, client_id=client_id)
        training_set = training_set.use(0)

    if get_training_dataset:
        return training_set
    else:
        train_loader = torch.utils.data.DataLoader(
            training_set,
            batch_size=train_bsz,
            shuffle=True,
            worker_init_fn=set_seed(client_id),
            generator=g,
            num_workers=4,
        )
        del training_set

        if is_test:
            test_set = torchvision.datasets.FashionMNIST(
                root=datadir, train=False, download=True, transform=transform
            )
            test_loader = torch.utils.data.DataLoader(
                test_set, batch_size=test_bsz, shuffle=True, generator=g, num_workers=4
            )
            del test_set
        else:
            test_loader = None

        return train_loader, test_loader
