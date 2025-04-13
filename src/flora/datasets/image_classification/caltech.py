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
import tarfile
import random
import shutil
import requests
import zipfile

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets.utils import download_url

from src.flora.datasets.image_classification import set_seed, SplitData, UnsplitData

# TODO: adjust num_workers in torch.utils.data.DataLoader based on total threads available on a client
# TODO: give layout of directory structure for ease of use

def caltechDataSplit(datadir, seed=1234, train_ratio=0.8, dataset_name='Caltech101'):
    """
    :param datadir: where to download/read the data
    :param seed: random seed
    :param train_ratio: splits data into 80% training and 20% test sets
    :param dataset_name: either Caltech101 or Caltech256
    :return: None
    """
    if dataset_name == 'Caltech101':
        download_dir = '101_ObjectCategories'
    elif dataset_name == 'Caltech256':
        download_dir = '256_ObjectCategories'
    else:
        raise ValueError('Unknown dataset')

    caltech_dir = os.path.join(datadir, download_dir)
    print(f'going to split CalTech dataset into train and test set')
    train_dir = os.path.join(datadir, dataset_name, 'train')
    test_dir = os.path.join(datadir, dataset_name, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    random.seed(seed)
    for category in sorted(os.listdir(caltech_dir)):
        category_path = os.path.join(caltech_dir, category)
        if not os.path.isdir(category_path):
            continue

        images = sorted([f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        test_images = images[split_idx:]

        train_category_path = os.path.join(train_dir, category)
        test_category_path = os.path.join(test_dir, category)
        os.makedirs(train_category_path, exist_ok=True)
        os.makedirs(test_category_path, exist_ok=True)

        for img in train_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(train_category_path, img))

        for img in test_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(test_category_path, img))


def caltech256Data(client_id=0, total_clients=1, datadir='~/', partition_dataset=True, train_bsz=32, test_bsz=32,
                   is_test=True):
    """
    :param client_id: id/rank of client/server
    :param total_clients: total number of clients/world-size
    :param datadir: where to download/read the data
    :param partition_dataset: whether to partition dataset among clients in unique chunks or just shuffle
    entire dataset across clients in different order
    :param train_bsz: training batch size
    :param test_bsz: test batch size
    :param is_test: process test/validation set dataloader or send None value
    :return: train and test dataloaders
    """
    if not os.path.isdir(os.path.join(datadir, '256_ObjectCategories')):
        print("CalTech-256 dataset not present. Going to download it...")
        url = "https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar?download=1"
        filename = "256_ObjectCategories.tar"
        md5 = "67b4f42ca05d46448c6bb8ecd2220f6d"
        curr_dir = os.getcwd()
        download_url(url=url, filename=filename, md5=md5, root=datadir)
        tar = tarfile.open(os.path.join(datadir, filename), "r")
        os.chdir(datadir)
        tar.extractall()
        tar.close()
        caltechDataSplit(datadir=datadir, seed=client_id, dataset_name='Caltech256')
        os.chdir(curr_dir)

    set_seed(seed=total_clients)
    g = torch.Generator()
    g.manual_seed(seed=total_clients)
    size = (224, 256)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize(size=size[1]), transforms.CenterCrop(size=size[0]),
                                    transforms.ToTensor(), normalize])

    training_set = datasets.ImageFolder(os.path.join(datadir, 'Caltech256', 'train'), transform=transform)

    if partition_dataset:
        training_set = SplitData(data=training_set, total_clients=total_clients)
        training_set = training_set.use(client_id)
    else:
        training_set = UnsplitData(data=training_set, client_id=client_id)
        training_set = training_set.use(0)

    train_loader = torch.utils.data.DataLoader(training_set, batch_size=train_bsz, shuffle=True,
                                               worker_init_fn=set_seed(client_id), generator=g, num_workers=4)
    del training_set

    if is_test:
        test_set = datasets.ImageFolder(os.path.join(datadir, 'Caltech256', 'test'), transform=transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_bsz, shuffle=True, generator=g, num_workers=4)
        del test_set
    else:
        test_loader = None

    return train_loader, test_loader


def caltech101Data(client_id=0, total_clients=1, datadir='~/', partition_dataset=True, train_bsz=32, test_bsz=32,
                   is_test=True):
    """
    :param client_id: id/rank of client or server
    :param total_clients: total number of clients/world-size
    :param datadir: where to download/read the data
    :param partition_dataset: whether to partition dataset among clients in unique chunks or just shuffle
    entire dataset across clients in different order
    :param train_bsz: training batch size
    :param test_bsz: test batch size
    :param is_test: process test/validation set dataloader or send None value
    :return: train and test dataloaders
    """
    if not os.path.isdir(os.path.join(datadir, '101_ObjectCategories')):
        curr_dir = os.getcwd()
        zip_url = 'https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1'
        print('Going to download CalTech-101 dataset...')
        download_url(url=zip_url, filename='caltech101.zip', root=datadir)
        print(f"Downloaded Caltech101 zip file to {datadir}")
        with zipfile.ZipFile(os.path.join(datadir, 'caltech101.zip'), 'r') as zip_ref:
            zip_ref.extractall(os.path.join(datadir, 'Caltech101'))

        print(f'Unzipped CalTech-101 dataset')
        filename = "101_ObjectCategories.tar.gz"
        print(f'Going to untar 101_ObjectCategories.tar.gz')
        tar = tarfile.open(os.path.join(datadir, 'Caltech101', 'caltech-101', filename), "r")
        os.chdir(datadir)
        tar.extractall()
        tar.close()
        os.remove(os.path.join(datadir, 'Caltech101'))
        os.remove(os.path.join(datadir, 'caltech101.zip'))
        caltechDataSplit(datadir=datadir, seed=client_id, dataset_name='Caltech101')
        os.chdir(curr_dir)

    set_seed(seed=total_clients)
    g = torch.Generator()
    g.manual_seed(total_clients)
    size = (224, 256)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize(size=size[1]), transforms.CenterCrop(size=size[0]),
                                    transforms.ToTensor(), normalize])

    training_set = datasets.ImageFolder(os.path.join(datadir, 'Caltech101', 'train'), transform=transform)

    if partition_dataset:
        training_set = SplitData(data=training_set, total_clients=total_clients)
        training_set = training_set.use(client_id)
    else:
        training_set = UnsplitData(data=training_set, client_id=client_id)
        training_set = training_set.use(0)

    train_loader = torch.utils.data.DataLoader(training_set, batch_size=train_bsz, shuffle=True,
                                               worker_init_fn=set_seed(client_id), generator=g, num_workers=4)
    del training_set

    if is_test:
        test_set = datasets.ImageFolder(os.path.join(datadir, 'Caltech101', 'test'), transform=transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_bsz, shuffle=True, generator=g, num_workers=4)
        del test_set
    else:
        test_loader = None

    return train_loader, test_loader