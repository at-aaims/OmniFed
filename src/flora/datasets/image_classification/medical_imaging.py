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
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image

from src.flora.datasets.image_classification import set_seed

# TODO: partition ISIC-Archive data into training and test sets

class ISICData(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # Filter out invalid entries
        self.valid_indices = [idx for idx in range(len(self.data_frame))
                              if os.path.isfile(os.path.join(self.root_dir, self.data_frame.iloc[idx, 0] + '.jpg'))]

    def __len__(self):
        # Return the count of filtered valid entries
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Use the filtered index to get the original dataframe index
        actual_idx = self.valid_indices[idx]
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[actual_idx, 0] + '.jpg')
        image = Image.open(img_name).convert("RGB")

        label = self.data_frame.iloc[actual_idx, 1]
        if self.transform:
            image = self.transform(image)

        return image, label


def isicArchiveData(client_id=0, total_clients=1, datadir='~/', partition_dataset=True, train_bsz=1, test_bsz=32,
                    is_test=True, csv_filename='challenge-2019-training_metadata_2025-04-13.csv', get_training_dataset=False):
    """
    :param client_id: id/rank of client or server
    :param total_clients: total number of clients/world-size
    :param datadir: where to download/read the data
    :param partition_dataset: whether to partition dataset among clients in unique chunks or just shuffle
    entire dataset across clients in different order
    :param train_bsz: training batch size
    :param test_bsz: test batch size
    :param is_test: process test/validation set dataloader or send None value
    :param csv_filename: name of csv file containing metadata like image id and corresponding label
    :param get_training_dataset: whether to get training dataset or train/test dataloader
    :return: train and test dataloaders
    """
    set_seed(seed=total_clients)
    g = torch.Generator()
    g.manual_seed(total_clients)

    csv_file_pth = os.path.join(datadir, csv_filename)
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])])

    training_set = ISICData(csv_file=csv_file_pth, root_dir=os.path.join(datadir, 'ISIC-images'), transform=transform)
    if get_training_dataset:
        return training_set
    else:
        train_loader = torch.utils.data.DataLoader(training_set, batch_size=train_bsz, shuffle=True,
                                                   worker_init_fn=set_seed(client_id), generator=g, num_workers=4)

        return train_loader

# if __name__ == '__main__':
#     print("Torch version:", torch.__version__)
#     print("Torchvision version:", torchvision.__version__)
#     isicArchiveDataset(datadir='/Users/ssq/Desktop/datasets/')