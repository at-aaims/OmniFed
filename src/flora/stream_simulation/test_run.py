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

from src.flora.stream_simulation import DataStreamSimulator, TrainingDataset

if __name__ == "__main__":
    strm = DataStreamSimulator(
        dataset_type=TrainingDataset.CIFAR10,
        datadir="/Users/ssq/Desktop/datasets/",
        kafka_dir="/Users/ssq/Desktop/datasets/kafka_2.12-3.2.0",
        total_clients=2,
    )
    strm.end_streaming()
