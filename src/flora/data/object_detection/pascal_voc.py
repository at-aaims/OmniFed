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
import re
import xml.etree.ElementTree as ET

import torch
import torchvision.transforms as transforms
from PIL import Image

from ..utils import SplitData, UnsplitData, set_seed

# Download and process PASCAL VOC-2012 dataset: https://www.kaggle.com/code/kuongan/vgg16

VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
class_to_idx = {cls_name: i for i, cls_name in enumerate(VOC_CLASSES)}


def extract_classes_from_annotation_regex(name_str, voc_classes):
    """
    Use regex to extract classes from the string name_str. Return a list of found classes (without duplicates).
    """
    s = name_str.strip().lower()
    sorted_classes = sorted(voc_classes, key=lambda x: -len(x))
    pattern = re.compile("|".join(sorted_classes))
    matches = pattern.findall(s)
    unique_matches = []
    for m in matches:
        if m not in unique_matches:
            unique_matches.append(m)
    return unique_matches


class PascalVOCDatasetObject(torch.utils.data.Dataset):
    def __init__(self, root, image_set="train", transform=None):
        """
        root: The path to the root directory of Pascal VOC, for example: /path/to/VOC2012_train_val
        image_set: "train", "val", "trainval", ...
        transform: Image transformations (for example: torchvision.transforms)

        This dataset generates samples in the form of (image, single_label). If an image has multiple classes,
        we create multiple samples (one sample for each class that appears).
        """
        super().__init__()
        self.root = root
        self.transform = transform
        if image_set == "train":
            file_list = os.path.join(root, "ImageSets", "Segmentation", "train.txt")
        elif image_set == "val":
            file_list = os.path.join(root, "ImageSets", "Segmentation", "val.txt")
        else:
            file_list = os.path.join(
                root, "ImageSets", "Segmentation", f"{image_set}.txt"
            )
        with open(file_list, "r") as f:
            self.image_ids = [line.strip() for line in f if line.strip()]

        self.samples = []
        for image_id in self.image_ids:
            ann_path = os.path.join(root, "Annotations", image_id + ".xml")
            try:
                tree = ET.parse(ann_path)
                root_xml = tree.getroot()
            except Exception as e:
                print(f"Error parsing {ann_path}: {e}")
                continue
            objects = root_xml.findall("object")
            classes_in_image = set()
            for obj in objects:
                name_str = obj.find("name").text
                extracted = extract_classes_from_annotation_regex(name_str, VOC_CLASSES)
                for cls in extracted:
                    classes_in_image.add(cls)
            for cls in classes_in_image:
                self.samples.append((image_id, cls))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_id, cls = self.samples[index]
        img_path = os.path.join(self.root, "JPEGImages", image_id + ".jpg")
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        label = class_to_idx[cls]
        return image, label


def pascalvocData(
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
    :param client_id: id/rank of client/server
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
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    training_set = PascalVOCDatasetObject(
        root=datadir, image_set="train", transform=transform
    )
    # TODO: check if data is evenly partitioned among clients and 'SplitData' works correctly
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

        if is_test:
            test_set = PascalVOCDatasetObject(
                root=datadir, image_set="val", transform=transform
            )
            test_loader = torch.utils.data.DataLoader(
                test_set, batch_size=test_bsz, shuffle=True, generator=g, num_workers=4
            )
            del test_set
        else:
            test_loader = None

        return train_loader, test_loader
