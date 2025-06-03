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
import os.path
import shutil

import kagglehub

# Download and process COCO-2017 dataset:
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F

from src.flora.datasets.image_classification import set_seed, split_into_chunks

# @FUTURE WORK: use pycocotools for non-IID distribution of this dataset
# TODO: partition training data into unique chunks based on total number of clients (check if this works)


class Rescale(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image):
        return F.resize(image, self.output_size)


class CocoDatasetObject(CocoDetection):
    def __init__(self, root, annFile, transform=None):
        super(CocoDatasetObject, self).__init__(root, annFile)
        self.transform = transform

    def __getitem__(self, idx):
        img, target = super(CocoDatasetObject, self).__getitem__(idx)
        if self.transform:
            img = self.transform(img)
        return img, target


def coco2017Data(
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
    :param partition_dataset: whether to partition dataset among clients in unique chunks or just shuffle whole dataset
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
    if "coco2017".lower() not in datadir.lower():
        print(
            "Going to download COCO 2017 dataset from Kaggle. This may take a while..."
        )
        coco2017_install_pth = kagglehub.dataset_download("awsaf49/coco-2017-dataset")
        if not os.path.exists(datadir):
            os.makedirs(datadir)

        shutil.move(src=coco2017_install_pth, dst=datadir)

    coco_train_dir = os.path.join(datadir, "train2017")
    coco_ann_train_file = os.path.join(
        datadir, "annotations", "instances_train2017.json"
    )

    coco_val_dir = os.path.join(datadir, "val2017")
    coco_ann_val_file = os.path.join(datadir, "annotations", "instances_val2017.json")

    transform = transforms.Compose([Rescale((256, 256)), transforms.ToTensor()])
    training_set = CocoDatasetObject(
        root=coco_train_dir, annFile=coco_ann_train_file, transform=transform
    )

    if partition_dataset:
        training_set = split_into_chunks(
            dataset=training_set, client_id=client_id, total_clients=total_clients
        )

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
            val_set = CocoDatasetObject(
                root=coco_val_dir, annFile=coco_ann_val_file, transform=transform
            )
            val_loader = torch.utils.data.DataLoader(
                val_set, batch_size=test_bsz, shuffle=True, generator=g, num_workers=4
            )
            del val_set
        else:
            val_loader = None

        return train_loader, val_loader


# if __name__ == "__main__":
#     path = kagglehub.dataset_download("awsaf49/coco-2017-dataset")
#     print("Path to dataset files:", path)
