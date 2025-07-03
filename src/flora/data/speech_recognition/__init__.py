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

import math
import os
import urllib.request
import zipfile

import torch
import torchaudio
from torchaudio.datasets import LIBRISPEECH
from torchaudio.transforms import Resample

from ..utils import set_seed

# TODO: torchaudio backend didn't support .flac files, so install ffmpeg with 'conda install -c conda-forge ffmpeg';
#  works with torch==2.5.1 and torchaudio==2.5.1
# TODO: verify training data is split into unique chunks. make sure shuffle based on client_id happens
#  in partition_dataset=False


def get_data_chunk(dataset, client_id=0, total_clients=1):
    indices = list(range(len(dataset)))
    chunk_size = math.ceil(len(dataset) / total_clients)
    start_idx = client_id * chunk_size
    end_idx = min(start_idx + chunk_size, len(dataset))

    chunk_indices = indices[start_idx:end_idx]
    subset = torch.utils.data.Subset(dataset, chunk_indices)

    return subset


# Optional: transform to resample all audio to a target rate
class ResampleTransform:
    def __init__(self, orig_freq, new_freq):
        self.new_freq = new_freq
        self.orig_freq = orig_freq
        self.resampler = Resample(orig_freq, new_freq)

    def __call__(self, waveform):
        if self.orig_freq != self.new_freq:
            return self.resampler(waveform)
        return waveform


# Custom collate function to pad variable-length audio
def librispeech_collate_fn(batch):
    waveforms, sample_rates, transcripts = zip(*batch)

    lengths = [w.shape[1] for w in waveforms]
    max_len = max(lengths)

    padded_waveforms = torch.zeros(len(waveforms), 1, max_len)
    for i, w in enumerate(waveforms):
        padded_waveforms[i, :, : w.shape[1]] = w

    return padded_waveforms, lengths, transcripts


# Apply transformation in a wrapper dataset
class LibriSpeechWrapped(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        waveform, sample_rate, transcript, *_ = self.dataset[idx]
        if self.transform:
            waveform = self.transform(waveform)
        return waveform, sample_rate, transcript

    def __len__(self):
        return len(self.dataset)


def libriSpeechData(
    client_id=0,
    total_clients=1,
    datadir="~/",
    partition_dataset=True,
    train_bsz=32,
    test_bsz=32,
    is_test=True,
    orig_sample_rate=16000,
    target_sample_rate=16000,
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
    :param orig_sample_rate: original sample frequency
    :param target_sample_rate: target sample frequency after transformation
    :param get_training_dataset: whether to get training dataset or train/test dataloader
    :return: train and test dataloaders
    """
    torchaudio.set_audio_backend("ffmpeg")
    set_seed(seed=total_clients)
    g = torch.Generator()
    g.manual_seed(total_clients)

    transform = ResampleTransform(
        orig_freq=orig_sample_rate, new_freq=target_sample_rate
    )

    training_set = LIBRISPEECH(datadir, url="train-clean-100", download=True)
    training_set = LibriSpeechWrapped(training_set, transform=transform)

    if partition_dataset:
        training_set = get_data_chunk(
            dataset=training_set, client_id=client_id, total_clients=total_clients
        )

    if get_training_dataset:
        return training_set
    else:
        train_loader = torch.utils.data.DataLoader(
            training_set,
            batch_size=train_bsz,
            shuffle=True,
            collate_fn=librispeech_collate_fn,
        )
        del training_set

        if is_test:
            test_set = LIBRISPEECH(datadir, url="test-clean", download=True)
            test_set = LibriSpeechWrapped(test_set, transform=transform)
            test_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=test_bsz,
                shuffle=False,
                collate_fn=librispeech_collate_fn,
            )
            del test_set
        else:
            test_loader = None

        for i, (waveforms, lengths, transcripts) in enumerate(train_loader):
            print("Waveforms shape:", waveforms.shape)
            print("Lengths:", lengths)
            print("Transcripts:", transcripts)
            break

        return train_loader, test_loader


def commonVoice_collate_fn(batch):
    """Collate function to handle variable-length audio and transcripts."""
    waveforms = []
    sample_rates = []
    transcripts = []

    for waveform, sample_rate, transcript in batch:
        waveforms.append(waveform)
        sample_rates.append(sample_rate)
        transcripts.append(transcript)

    return waveforms, sample_rates, transcripts


def commonVoiceData(
    client_id=0,
    total_clients=1,
    datadir="~/",
    partition_dataset=True,
    train_bsz=32,
    test_bsz=32,
    is_test=True,
    version="cv-corpus-13.0-delta-2023-03-09",
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
    :param version: version of CommonVoice dataset downloaded from https://commonvoice.mozilla.org/en/datasets. The
    default version of dataset downloaded is "cv-corpus-13.0-delta-2023-03-09"
    :param get_training_dataset: whether to get training dataset or train/test dataloader
    :return: train and test dataloaders
    """
    set_seed(seed=total_clients)
    g = torch.Generator()
    g.manual_seed(total_clients)

    training_set = torchaudio.datasets.COMMONVOICE(root=datadir, tsv="train.tsv")
    if partition_dataset:
        training_set = get_data_chunk(
            dataset=training_set, client_id=client_id, total_clients=total_clients
        )

    if get_training_dataset:
        return training_set
    else:
        train_loader = torch.utils.data.DataLoader(
            training_set,
            batch_size=train_bsz,
            shuffle=True,
            num_workers=4,
            collate_fn=commonVoice_collate_fn,
        )
        del training_set

        if is_test:
            test_set = torchaudio.datasets.COMMONVOICE(root=datadir, tsv="test.tsv")
            test_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=test_bsz,
                shuffle=False,
                collate_fn=commonVoice_collate_fn,
            )
        else:
            test_loader = None

        for waveforms, sample_rates, transcripts in train_loader:
            print(waveforms[0].shape)
            print(sample_rates[0])
            print(transcripts[0])
            print("inside here...")
            break

        return train_loader, test_loader
