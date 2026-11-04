#!/bin/sh

# Copyright (c) 2026, Oak Ridge National Laboratory.  All rights reserved.
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

# error and log traces
# =========================================

export OMNIFED_DEBUG=1 # NOTE: currently unused

export PYTHONUNBUFFERED=1 # Immediate Python output
export HYDRA_FULL_ERROR=1 # Full Hydra error traces
export RAY_DEDUP_LOGS=0   # Show all FL node logs

# for data loading, partitioning, and storing training/test/eval datasets
# =========================================

# for skewed and unbalanced data partitioning schemes in src/omnifed/data/skewed
export OMNIFED_PARTITION_SEED=123

# for testing w/ C4 dataset
export OMNIFED_TOKENIZER_DIR=""
export OMNIFED_C4_DISK=""
export OMNIFED_DATA_DIR="/Users/ssq/Desktop/datasets/"  # replace with "./datasets" (used in .gitignore)

# To use  FLamby cross-silo datasets
export OMNIFED_FLAMBY_DIR=""  # FLamby repo dir to use if not using third_party/FLamby
export OMNIFED_FLAMBY_ACCEPT_LICENSE=1
export OMNIFED_FLAMBY_IXI_URL="https://data.mendeley.com/datasets/7kd5wj7v7p/3"
export OMNIFED_FLAMBY_LIDC_LIMIT=16 # 16 for testing, then all 1018 scans
export OMNIFED_FLAMBY_LIDC_URL="https://www.cancerimagingarchive.net/wp-content/uploads/LIDC-XML-only.zip"