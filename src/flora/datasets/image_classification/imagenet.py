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
import json
import shutil

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from src.flora.datasets.image_classification import set_seed

# TODO: adjust num_workers in torch.utils.data.DataLoader based on total threads available on a client

def processImageNet(datadir='~/'):
    """
    :param datadir: points to the ImageNet folder after download and untar operation.
    :return:
    """
    train_datadir = os.path.join(datadir, 'train')
    val_datadir = os.path.join(datadir, 'val')
    imagenet_classes_file = os.path.join(datadir, 'imagenet_classes.json')

    imagenet_traindir = os.path.join(os.path.dirname(datadir), 'imagenet', 'train')
    imagenet_valdir = os.path.join(os.path.dirname(datadir), 'imagenet', 'val')
    os.makedirs(imagenet_traindir, exist_ok=True)
    os.makedirs(imagenet_valdir, exist_ok=True)

    # Load class index mapping (if available)
    with open(imagenet_classes_file, 'r') as f:
        class_mapping = json.load(f)

    # Organize training data
    for class_id, class_name in class_mapping.items():
        source_dir = os.path.join(train_datadir, class_name)
        if os.path.exists(source_dir):
            dest_dir = os.path.join(imagenet_traindir, class_name)
            os.makedirs(dest_dir, exist_ok=True)

            # Move images to the respective class directory
            for img in os.listdir(source_dir):
                shutil.move(os.path.join(source_dir, img), dest_dir)

    # Organize validation data
    # Assuming the validation data has been organized similarly into folders
    for class_id, class_name in class_mapping.items():
        source_dir = os.path.join(val_datadir, class_name)
        if os.path.exists(source_dir):
            dest_dir = os.path.join(imagenet_valdir, class_name)
            os.makedirs(dest_dir, exist_ok=True)

            # Move images to the respective class directory
            for img in os.listdir(source_dir):
                shutil.move(os.path.join(source_dir, img), dest_dir)

    print(f'ImageNet dataset ready to be processed as dataloader for training/testing.')


def imagenetData(client_id=0, datadir='~/', train_bsz=32, test_bsz=32, is_test=True):
    """
    :param client_id: id/rank of client/server
    :param datadir: where to download/read the data
    :param train_bsz: training batch size
    :param test_bsz: test batch size
    :param is_test: process test/validation set dataloader or send None value
    :return: train and test dataloaders
    """
    # if processed imagenet directory does not exist, we assume the ImageNet data is downloaded and untar
    # and points to 'datadir' directory
    if not os.path.exists(datadir):
        processImageNet(datadir=datadir)

    set_seed(seed=client_id)
    g = torch.Generator()
    g.manual_seed(seed=client_id)

    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    training_set = ImageFolder(os.path.join(datadir, 'train'), transform=transform)
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=train_bsz, shuffle=True,
                                               worker_init_fn=set_seed(client_id), generator=g, num_workers=4)
    del training_set

    if is_test:
        val_set = ImageFolder(os.path.join(datadir, 'val'), transform=transform)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=test_bsz, shuffle=False, num_workers=4)
        del val_set
    else:
        val_loader = None

    return train_loader, val_loader