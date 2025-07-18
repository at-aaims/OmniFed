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

import os

import torch
import torchvision
import torchvision.transforms as transforms

from src.flora.datasets.image_classification import set_seed


class NonIIDSubsetData(torch.utils.data.Dataset):
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


def get_labels_perclient(number_of_labels=1, total_clients=1):
    partitions = []
    for i in range(total_clients):
        start = (i * number_of_labels) // total_clients
        end = ((i + 1) * number_of_labels) // total_clients
        partitions.append(list(range(start, end)))

    return partitions


def noniid_cifar10Data(
    client_id=0,
    total_clients=1,
    datadir="~/",
    train_bsz=32,
    test_bsz=32,
    number_of_labels=10,
):
    labels_per_client = get_labels_perclient(
        number_of_labels=number_of_labels, total_clients=total_clients
    )

    set_seed(seed=total_clients)
    g = torch.Generator()
    g.manual_seed(total_clients)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            normalize,
        ]
    )
    training_set = torchvision.datasets.CIFAR10(
        root=datadir, train=True, download=True, transform=transform
    )

    noniid_dataset = NonIIDSubsetData(
        dataset=training_set, labels_per_rank=labels_per_client[client_id]
    )
    del training_set

    noniid_train_loader = torch.utils.data.DataLoader(
        noniid_dataset,
        batch_size=train_bsz,
        shuffle=True,
        worker_init_fn=set_seed(client_id),
        generator=g,
        num_workers=1,
    )

    transform = transforms.Compose([transforms.ToTensor(), normalize])
    test_set = torchvision.datasets.CIFAR10(
        root=datadir, train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=test_bsz, shuffle=True, generator=g, num_workers=1
    )
    del test_set

    return noniid_train_loader, test_loader


def noniid_cifar100Data(
    client_id=0,
    total_clients=1,
    datadir="~/",
    train_bsz=32,
    test_bsz=32,
    number_of_labels=100,
):
    labels_per_client = get_labels_perclient(
        number_of_labels=number_of_labels, total_clients=total_clients
    )

    set_seed(seed=total_clients)
    g = torch.Generator()
    g.manual_seed(total_clients)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            normalize,
        ]
    )
    training_set = torchvision.datasets.CIFAR100(
        root=datadir, train=True, download=True, transform=transform
    )

    noniid_dataset = NonIIDSubsetData(
        dataset=training_set, labels_per_rank=labels_per_client[client_id]
    )
    del training_set

    noniid_train_loader = torch.utils.data.DataLoader(
        noniid_dataset,
        batch_size=train_bsz,
        shuffle=True,
        worker_init_fn=set_seed(client_id),
        generator=g,
        num_workers=4,
    )

    transform = transforms.Compose([transforms.ToTensor(), normalize])
    test_set = torchvision.datasets.CIFAR100(
        root=datadir, train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=test_bsz, shuffle=True, generator=g, num_workers=4
    )
    del test_set

    return noniid_train_loader, test_loader


def noniid_caltech101Data(
    client_id=0,
    total_clients=1,
    datadir="~/",
    train_bsz=32,
    test_bsz=32,
    number_of_labels=101,
):
    labels_per_client = get_labels_perclient(
        number_of_labels=number_of_labels, total_clients=total_clients
    )

    set_seed(seed=total_clients)
    g = torch.Generator()
    g.manual_seed(total_clients)
    size = (224, 256)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose(
        [
            transforms.Resize(size=size[1]),
            transforms.CenterCrop(size=size[0]),
            transforms.ToTensor(),
            normalize,
        ]
    )

    training_set = torchvision.datasets.ImageFolder(
        os.path.join(datadir, "Caltech101", "train"), transform=transform
    )

    noniid_dataset = NonIIDSubsetData(
        dataset=training_set, labels_per_rank=labels_per_client[client_id]
    )
    del training_set

    noniid_train_loader = torch.utils.data.DataLoader(
        noniid_dataset,
        batch_size=train_bsz,
        shuffle=True,
        worker_init_fn=set_seed(client_id),
        generator=g,
        num_workers=4,
    )
    del training_set

    test_set = torchvision.datasets.ImageFolder(
        os.path.join(datadir, "Caltech101", "test"), transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=test_bsz, shuffle=True, generator=g, num_workers=4
    )
    del test_set

    return noniid_train_loader, test_loader


def noniid_caltech256Data(
    client_id=0,
    total_clients=1,
    datadir="~/",
    train_bsz=32,
    test_bsz=32,
    number_of_labels=256,
):
    labels_per_client = get_labels_perclient(
        number_of_labels=number_of_labels, total_clients=total_clients
    )

    set_seed(seed=total_clients)
    g = torch.Generator()
    g.manual_seed(total_clients)
    size = (224, 256)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose(
        [
            transforms.Resize(size=size[1]),
            transforms.CenterCrop(size=size[0]),
            transforms.ToTensor(),
            normalize,
        ]
    )

    training_set = torchvision.datasets.ImageFolder(
        os.path.join(datadir, "Caltech256", "train"), transform=transform
    )

    noniid_dataset = NonIIDSubsetData(
        dataset=training_set, labels_per_rank=labels_per_client[client_id]
    )
    del training_set

    noniid_train_loader = torch.utils.data.DataLoader(
        noniid_dataset,
        batch_size=train_bsz,
        shuffle=True,
        worker_init_fn=set_seed(client_id),
        generator=g,
        num_workers=4,
    )
    del training_set

    test_set = torchvision.datasets.ImageFolder(
        os.path.join(datadir, "Caltech256", "test"), transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=test_bsz, shuffle=True, generator=g, num_workers=4
    )
    del test_set

    return noniid_train_loader, test_loader
