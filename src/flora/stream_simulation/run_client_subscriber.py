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

import argparse

from src.flora.stream_simulation.data_streaming import DataStreamSubscriber


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kafka-dir", type=str, default="/Users/ssq/Desktop/datasets/kafka_2.12-3.2.0/"
    )
    parser.add_argument(
        "--data-dir", type=str, default="/Users/ssq/Desktop/datasets/flora_test/"
    )
    parser.add_argument(
        "--log-dir", type=str, default="/Users/ssq/Desktop/datasets/flora_test/"
    )
    parser.add_argument("--kafka-host", type=str, default="127.0.0.1")
    parser.add_argument("--kafka-port", type=int, default=9092)
    parser.add_argument("--kafka-topic", type=str, default="client-0")
    parser.add_argument("--client-id", type=int, default=0)
    args = parser.parse_args()

    stream = DataStreamSubscriber(
        kafka_host=args.kafka_host,
        kafka_port=args.kafka_port,
        kafka_dir=args.kafka_dir,
        client_id=args.client_id,
        log_dir=args.log_dir,
    )
    stream.create_topic(args.kafka_topic)
    stream.stream_data()
