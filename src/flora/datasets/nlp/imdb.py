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


# OSError: dlopen(/Users/ssq/Documents/github/ornl_projects/FLORA/pyvenv3_11_8/lib/python3.11/site-packages/torchtext/lib/libtorchtext.so, 0x0006):
# Symbol not found: __ZN3c105ErrorC1ENSt3__112basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEES7_PKv
#  Referenced from: <5436ECC1-6F45-386E-B542-D5F76A22B52C> /Users/ssq/Documents/github/ornl_projects/FLORA/pyvenv3_11_8/lib/python3.11/site-packages/torchtext/lib/libtorchtext.so
#  Expected in:     <5445D2E4-6D7A-39F2-9003-F3A3F854555A> /Users/ssq/Documents/github/ornl_projects/FLORA/pyvenv3_11_8/lib/python3.11/site-packages/torch/lib/libc10.dylib

import torch

from src.flora.datasets.nlp import imdbReviewsDataset

if __name__ == '__main__':
    print(torch.__file__)
    pth='/Users/ssq/Desktop/datasets/'
    _, _ = imdbReviewsDataset(datadir=pth, partition_dataset=False)