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
import random
import shutil
import zipfile
import tarfile
import rich.repr

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets.utils import download_url

from src.flora.data.utils import set_seed, SplitData, UnsplitData, get_labels_per_client, DataSubsetNonIID


# ======================================================================================


@rich.repr.auto
class DataLoaderIID:
    def __init__(
        self,
        client_id: int = 0,
        total_clients: int = 1,
        data_dir: str = "~/",
        partition: bool = False,
        train_bsz: int = 32,
        test_bsz: int = 32,
    ):
        set_seed(seed=total_clients)
        self.g = torch.Generator()
        self.g.manual_seed(total_clients)
        self.client_id = client_id
        self.total_clients = total_clients
        self.data_dir = data_dir
        self.partition = partition
        self.train_bsz = train_bsz
        self.test_bsz = test_bsz

    def CIFAR10(self):
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
            root=self.data_dir, train=True, download=True, transform=transform
        )

        if self.partition:
            training_set = SplitData(
                data=training_set, total_clients=self.total_clients
            )
            training_set = training_set.use(self.client_id)
        else:
            training_set = UnsplitData(data=training_set, client_id=self.client_id)
            training_set = training_set.use(0)

        train_loader = torch.utils.data.DataLoader(
            training_set,
            batch_size=self.train_bsz,
            shuffle=True,
            worker_init_fn=set_seed(self.client_id),
            generator=self.g,
            num_workers=1,
        )

        del training_set

        # test_set = torchvision.datasets.CIFAR10(
        #     root=self.data_dir, train=False, download=True, transform=transform
        # )
        # test_loader = torch.utils.data.DataLoader(
        #     test_set,
        #     batch_size=self.test_bsz,
        #     shuffle=True,
        #     generator=self.g,
        #     num_workers=1,
        # )
        # del test_set

        return train_loader

    def CIFAR100(self):
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
            root=self.data_dir, train=True, download=True, transform=transform
        )

        if self.partition:
            training_set = SplitData(
                data=training_set, total_clients=self.total_clients
            )
            training_set = training_set.use(self.client_id)
        else:
            training_set = UnsplitData(data=training_set, client_id=self.client_id)
            training_set = training_set.use(0)

        train_loader = torch.utils.data.DataLoader(
            training_set,
            batch_size=self.train_bsz,
            shuffle=True,
            worker_init_fn=set_seed(self.client_id),
            generator=self.g,
            num_workers=1,
        )
        del training_set

        # transform = transforms.Compose([transforms.ToTensor(), normalize])
        # test_set = torchvision.datasets.CIFAR100(
        #     root=self.data_dir, train=False, download=True, transform=transform
        # )
        # test_loader = torch.utils.data.DataLoader(
        #     test_set, batch_size=self.test_bsz, shuffle=True, generator=g, num_workers=1
        # )
        # del test_set

        return train_loader

    def CalTechDataSplit(
        self, train_ratio: float = 0.8, dataset_name: str = "Caltech101"
    ):
        if dataset_name == "Caltech101":
            download_dir = "101_ObjectCategories"
        elif dataset_name == "Caltech256":
            download_dir = "256_ObjectCategories"
        else:
            raise ValueError("Unknown dataset")

        caltech_dir = os.path.join(self.data_dir, download_dir)
        print("going to split CalTech dataset into train and test set")
        train_dir = os.path.join(self.data_dir, dataset_name, "train")
        test_dir = os.path.join(self.data_dir, dataset_name, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        random.seed(self.total_clients)
        for category in sorted(os.listdir(caltech_dir)):
            category_path = os.path.join(caltech_dir, category)
            if not os.path.isdir(category_path):
                continue

            images = sorted(
                [
                    f
                    for f in os.listdir(category_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
            )
            random.shuffle(images)
            split_idx = int(len(images) * train_ratio)
            train_images = images[:split_idx]
            test_images = images[split_idx:]

            train_category_path = os.path.join(train_dir, category)
            test_category_path = os.path.join(test_dir, category)
            os.makedirs(train_category_path, exist_ok=True)
            os.makedirs(test_category_path, exist_ok=True)

            for img in train_images:
                shutil.copy(
                    os.path.join(category_path, img),
                    os.path.join(train_category_path, img),
                )

            for img in test_images:
                shutil.copy(
                    os.path.join(category_path, img),
                    os.path.join(test_category_path, img),
                )

    def Caltech101(self):
        if not os.path.isdir(os.path.join(self.data_dir, "101_ObjectCategories")):
            curr_dir = os.getcwd()
            zip_url = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1"
            print("Going to download CalTech-101 dataset...")
            download_url(url=zip_url, filename="caltech101.zip", root=self.data_dir)
            print(f"Downloaded Caltech101 zip file to {self.data_dir}")
            with zipfile.ZipFile(os.path.join(self.data_dir, "caltech101.zip"), "r") as zip_ref:
                zip_ref.extractall(os.path.join(self.data_dir, "Caltech101"))

            print("Unzipped CalTech-101 dataset")
            filename = "101_ObjectCategories.tar.gz"
            print("Going to untar 101_ObjectCategories.tar.gz")
            tar = tarfile.open(
                os.path.join(self.data_dir, "Caltech101", "caltech-101", filename), "r"
            )
            os.chdir(self.data_dir)
            tar.extractall()
            tar.close()
            os.remove(os.path.join(self.data_dir, "caltech101.zip"))
            # split dataset into 80% training and 20% testing
            self.CalTechDataSplit(train_ratio=0.8, dataset_name="Caltech101")
            os.chdir(curr_dir)

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
        training_set = datasets.ImageFolder(
            os.path.join(self.data_dir, "Caltech101", "train"), transform=transform
        )
        if self.partition:
            training_set = SplitData(data=training_set, total_clients=self.total_clients)
            training_set = training_set.use(self.client_id)
        else:
            training_set = UnsplitData(data=training_set, client_id=self.client_id)
            training_set = training_set.use(0)

        train_loader = torch.utils.data.DataLoader(
            training_set,
            batch_size=self.train_bsz,
            shuffle=True,
            worker_init_fn=set_seed(self.client_id),
            generator=self.g,
            num_workers=1,
        )
        del training_set

        # test_set = datasets.ImageFolder(
        #     os.path.join(self.data_dir, "Caltech101", "test"), transform=transform
        # )
        # test_loader = torch.utils.data.DataLoader(
        #     test_set, batch_size=self.test_bsz, shuffle=True, generator=self.g, num_workers=1
        # )
        # del test_set

        return train_loader

    def Caltech256(self):
        if not os.path.isdir(os.path.join(self.data_dir, "256_ObjectCategories")):
            print("CalTech-256 dataset not present. Going to download it...")
            url = "https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar?download=1"
            filename = "256_ObjectCategories.tar"
            md5 = "67b4f42ca05d46448c6bb8ecd2220f6d"
            curr_dir = os.getcwd()
            download_url(url=url, filename=filename, md5=md5, root=self.data_dir)
            tar = tarfile.open(os.path.join(self.data_dir, filename), "r")
            os.chdir(self.data_dir)
            tar.extractall()
            tar.close()
            # split dataset into 80% training and 20% testing
            self.CalTechDataSplit(train_ratio=0.8, dataset_name="Caltech256")
            os.chdir(curr_dir)

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

        training_set = datasets.ImageFolder(
            os.path.join(self.data_dir, "Caltech256", "train"), transform=transform
        )
        if self.partition:
            training_set = SplitData(data=training_set, total_clients=self.total_clients)
            training_set = training_set.use(self.client_id)
        else:
            training_set = UnsplitData(data=training_set, client_id=self.client_id)
            training_set = training_set.use(0)

        train_loader = torch.utils.data.DataLoader(
            training_set,
            batch_size=self.train_bsz,
            shuffle=True,
            worker_init_fn=set_seed(self.client_id),
            generator=self.g,
            num_workers=1,
        )
        del training_set

        # test_set = datasets.ImageFolder(
        #     os.path.join(self.data_dir, "Caltech256", "test"), transform=transform
        # )
        # test_loader = torch.utils.data.DataLoader(
        #     test_set, batch_size=self.test_bsz, shuffle=True, generator=self.g, num_workers=1
        # )
        # del test_set

        return train_loader


@rich.repr.auto
class DataLoaderNonIID:
    def __init__(
        self,
        total_labels: int,
        client_id: int = 0,
        total_clients: int = 1,
        data_dir: str = "~/",
        train_bsz: int = 32,
        test_bsz: int = 32
    ):
        set_seed(seed=total_clients)
        self.g = torch.Generator()
        self.g.manual_seed(total_clients)
        self.total_labels = total_labels
        self.client_id = client_id
        self.total_clients = total_clients
        self.data_dir = data_dir
        self.train_bsz = train_bsz
        self.test_bsz = test_bsz

    def CIFAR10(self):

        labels_per_client = get_labels_per_client(
            number_of_labels=self.total_labels, total_clients=self.total_clients
        )

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
            root=self.data_dir, train=True, download=True, transform=transform
        )

        noniid_dataset = DataSubsetNonIID(
            dataset=training_set, labels_per_rank=labels_per_client[self.client_id]
        )
        del training_set

        noniid_train_loader = torch.utils.data.DataLoader(
            noniid_dataset,
            batch_size=self.train_bsz,
            shuffle=True,
            worker_init_fn=set_seed(self.client_id),
            generator=self.g,
            num_workers=1,
        )

        # transform = transforms.Compose([transforms.ToTensor(), normalize])
        # test_set = torchvision.datasets.CIFAR10(
        #     root=self.data_dir, train=False, download=True, transform=transform
        # )
        # test_loader = torch.utils.data.DataLoader(
        #     test_set, batch_size=self.test_bsz, shuffle=True, generator=self.g, num_workers=1
        # )
        # del test_set

        return noniid_train_loader

    def CIFAR100(self):

        labels_per_client = get_labels_per_client(
            number_of_labels=self.total_labels, total_clients=self.total_clients
        )

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
            root=self.data_dir, train=True, download=True, transform=transform
        )

        noniid_dataset = DataSubsetNonIID(
            dataset=training_set, labels_per_rank=labels_per_client[self.client_id]
        )
        del training_set

        noniid_train_loader = torch.utils.data.DataLoader(
            noniid_dataset,
            batch_size=self.train_bsz,
            shuffle=True,
            worker_init_fn=set_seed(self.client_id),
            generator=self.g,
            num_workers=1,
        )

        # transform = transforms.Compose([transforms.ToTensor(), normalize])
        # test_set = torchvision.datasets.CIFAR100(
        #     root=self.data_dir, train=False, download=True, transform=transform
        # )
        # test_loader = torch.utils.data.DataLoader(
        #     test_set, batch_size=self.test_bsz, shuffle=True, generator=self.g, num_workers=1
        # )
        # del test_set

        return noniid_train_loader

    def CalTech101(self):

        labels_per_client = get_labels_per_client(
            number_of_labels=self.total_labels, total_clients=self.total_clients
        )

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
            os.path.join(self.data_dir, "Caltech101", "train"), transform=transform
        )

        noniid_dataset = DataSubsetNonIID(
            dataset=training_set, labels_per_rank=labels_per_client[self.client_id]
        )
        del training_set

        noniid_train_loader = torch.utils.data.DataLoader(
            noniid_dataset,
            batch_size=self.train_bsz,
            shuffle=True,
            worker_init_fn=set_seed(self.client_id),
            generator=self.g,
            num_workers=1,
        )
        del training_set

        test_set = torchvision.datasets.ImageFolder(
            os.path.join(self.data_dir, "Caltech101", "test"), transform=transform
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=self.test_bsz, shuffle=True, generator=self.g, num_workers=1
        )
        del test_set

        return noniid_train_loader, test_loader

    def CalTech256(self):

        labels_per_client = get_labels_per_client(
            number_of_labels=self.total_labels, total_clients=self.total_clients
        )

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
            os.path.join(self.data_dir, "Caltech256", "train"), transform=transform
        )

        noniid_dataset = DataSubsetNonIID(
            dataset=training_set, labels_per_rank=labels_per_client[self.client_id]
        )
        del training_set

        noniid_train_loader = torch.utils.data.DataLoader(
            noniid_dataset,
            batch_size=self.train_bsz,
            shuffle=True,
            worker_init_fn=set_seed(self.client_id),
            generator=self.g,
            num_workers=1,
        )
        del training_set

        # test_set = torchvision.datasets.ImageFolder(
        #     os.path.join(self.data_dir, "Caltech256", "test"), transform=transform
        # )
        # test_loader = torch.utils.data.DataLoader(
        #     test_set, batch_size=self.test_bsz, shuffle=True, generator=g, num_workers=4
        # )
        # del test_set

        return noniid_train_loader
