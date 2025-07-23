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

import time
from typing import Dict

import grpc
import rich.repr
import torch

from . import ReductionType, grpc_communicator_pb2
from . import grpc_communicator_pb2_grpc
from .utils import get_msg_info, proto_to_tensordict, tensordict_to_proto


@rich.repr.auto
class GrpcClient:
    """
    gRPC client for federated learning communication.

    Handles server registration, broadcast state retrieval, and
    aggregation operations with automatic retry and timeout logic.
    """

    def __init__(
        self,
        client_id: str,
        master_addr: str,
        master_port: int,
        max_send_message_length: int,
        max_receive_message_length: int,
        retry_delay: float = 5.0,  # Seconds between retries
        max_retries: int = 3,  # Maximum retry attempts
        client_timeout: float = 60,  # Seconds to wait for server responses
    ):
        print(f"[COMM-INIT] Client | addr={master_addr}:{master_port}")

        self.client_id = client_id
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        self.client_timeout = client_timeout
        self.master_addr = master_addr
        self.master_port = master_port
        self.max_send_message_length = max_send_message_length
        self.max_receive_message_length = max_receive_message_length
        self.channel = None
        self.stub = None

        for attempt in range(1, self.max_retries + 1):
            try:
                self.channel = grpc.insecure_channel(
                    self.master_addr + ":" + str(self.master_port),
                    options=[
                        (
                            "grpc.max_receive_message_length",
                            self.max_receive_message_length,
                        ),
                        (
                            "grpc.max_send_message_length",
                            self.max_send_message_length,
                        ),
                    ],
                )
                self.stub = grpc_communicator_pb2_grpc.CentralServerStub(self.channel)
                response = self.stub.RegisterClient(
                    grpc_communicator_pb2.ClientInfo(client_id=self.client_id),
                )
                print(f"[COMM-CLIENT] Register | success={response.success}")
                return

            except grpc.RpcError as e:
                if attempt >= self.max_retries:
                    print(
                        f"[COMM-ERROR] Connection failed | {self.max_retries} attempts"
                    )
                    raise e

                print(
                    f"[COMM-ERROR] Connection retry | attempt {attempt}/{self.max_retries} | {self.retry_delay}s delay"
                )
                time.sleep(self.retry_delay)

    def get_broadcast_state(self) -> Dict[str, torch.Tensor]:
        """Retrieve broadcast state from server with retries."""
        print(f"[BCAST-REQUEST] Waiting for server to broadcast model")

        poll_count = 0
        error_count = 0

        while True:
            try:
                request = grpc_communicator_pb2.ClientInfo(client_id=self.client_id)
                response = self.stub.GetBroadcastState(request)
                if response.is_ready:
                    tensordict = proto_to_tensordict(response.tensor_dict)
                    print(f"[BCAST-RECEIVED] {get_msg_info(tensordict)}")
                    return tensordict
                poll_count += 1
                print(
                    f"[COMM-BCAST] Waiting | poll {poll_count} | retry in {self.retry_delay}s"
                )
                time.sleep(self.retry_delay)

            except grpc.RpcError as e:
                error_count += 1
                if error_count > self.max_retries:
                    raise RuntimeError(
                        f"Max retries ({self.max_retries}) exceeded for broadcast state"
                    )
                print(
                    f"[COMM-ERROR] Broadcast fetch | error {error_count}/{self.max_retries} | retry in {self.retry_delay}s"
                )
                time.sleep(self.retry_delay)

    def submit_for_aggregation(
        self, tensordict: Dict[str, torch.Tensor], reduction_type: ReductionType
    ):
        """Submit tensors to server for aggregation."""
        try:
            proto_tensordict = tensordict_to_proto(tensordict)
            request = grpc_communicator_pb2.AggregationRequest(
                client_id=self.client_id,
                tensor_dict=proto_tensordict,
                reduction_type=reduction_type.value,
            )
            response = self.stub.SubmitForAggregation(request)
            if response.success:
                print(f"[AGG-SUBMIT] Successfully sent local model to server")
            else:
                print(f"[COMM-ERROR] Submit failed")
        except grpc.RpcError as e:
            print(f"[COMM-ERROR] Submit exception | {e}")

    def get_aggregation_result(self) -> Dict[str, torch.Tensor]:
        """Retrieve aggregation result from server with timeout."""
        print(
            f"[AGG-WAIT] Waiting for server to aggregate models (timeout={self.client_timeout}s)"
        )

        start_time = time.time()
        poll_count = 0
        error_count = 0

        while True:
            elapsed = time.time() - start_time
            if elapsed > self.client_timeout:
                raise RuntimeError(f"Aggregation timeout ({self.client_timeout}s)")
            try:
                request = grpc_communicator_pb2.ClientInfo(client_id=self.client_id)
                response = self.stub.GetAggregationResult(request)
                if response.is_ready:
                    tensordict = proto_to_tensordict(response.tensor_dict)
                    print(
                        f"[AGG-RECEIVED] {get_msg_info(tensordict)} (waited {elapsed:.1f}s)"
                    )
                    return tensordict
                poll_count += 1
                remaining = self.client_timeout - elapsed
                print(
                    f"[COMM-AGG] Waiting | poll {poll_count} | {remaining:.1f}s remaining"
                )
                time.sleep(min(self.retry_delay, remaining))

            except grpc.RpcError as e:
                error_count += 1
                if error_count > self.max_retries:
                    raise RuntimeError(
                        f"Max retries ({self.max_retries}) exceeded for aggregation result"
                    )
                print(
                    f"[COMM-ERROR] Aggregation fetch | error {error_count}/{self.max_retries} | retry in {self.retry_delay}s"
                )
                time.sleep(self.retry_delay)
