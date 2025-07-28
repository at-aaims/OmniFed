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

from src.flora.stream_simulation.data_streaming import DataStreamPublisher


if __name__ == "__main__":
    kafka_dir = "/Users/ssq/Desktop/datasets/kafka_2.12-3.2.0/"
    data_dir = "/Users/ssq/Desktop/datasets/flora_test/"
    streamer = DataStreamPublisher(
        dataset_type="cifar10",
        kafka_host="127.0.0.1",
        kafka_port=9092,
        stream_rate=32,
        datadir=data_dir,
        total_clients=1,
    )
    streamer.publish_data_to_clients()
