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

"""Secure variant of ``grpc_client.GrpcClient`` (mTLS channel, explicit RPC deadlines).

Retry / polling / chunking / compression logic is the existing OmniFed code. What differs:

* ``grpc.secure_channel`` with mutual TLS: the server must present a certificate for the host
  we dialled, signed by our CA, and we present our own client certificate.
* The request messages carry no client id. At start-up the server tells us which participant
  it derived from our certificate; if that is not our rank we stop.
* Every RPC has a deadline (``rpc_timeout``); connecting has its own (``connect_timeout``).
* UNAUTHENTICATED / PERMISSION_DENIED are not retried. A "server busy" refusal (the server's
  concurrent-RPC cap) is retried after ``retry_delay``.
"""

from __future__ import annotations

import time
from typing import Dict

import grpc
import rich.repr
import torch

from ...utils import print
from ..base import AggregationOp
from . import secure_grpc_pb2 as pb2
from . import secure_grpc_pb2_grpc, security
from ..grpc_chunking import (
    assemble_payload_chunks,
    iter_payload_chunks,
    packed_message_bytes,
    should_chunk_payload,
)
from ..grpc_limits import (
    GRPC_CHUNK_PAYLOAD_BYTES,
    GRPC_CHUNK_THRESHOLD_BYTES,
    GRPC_CLIENT_TIMEOUT_SEC,
)
from ..utils import get_msg_info, proto_to_tensordict, proto_to_tensordict_extended
from ..utils import (
    aggregation_metric_for_communicate_params,
    compress_message_tensors,
    compressor_proto_name,
)
from ...utils import MetricLogger
from .secure_codec import tensordict_to_proto as _encode_tensordict

from contextlib import nullcontext

from src.omnifed.summary.per_iteration import accumulate_iter_comm


def _to_proto(tensordict, compression_type=None):
    """Encode into the *secure* proto's TensorDict."""
    return _encode_tensordict(pb2, tensordict, compression_type)


@rich.repr.auto
class SecureGrpcClient:
    """
    gRPC client for federated learning communication coordination over mutual TLS.

    Connects to a central secure gRPC server for broadcast and aggregation operations.
    Provides automatic retry logic, per-RPC deadlines, and error recovery
    for robust distributed communication.

    Used by: SecureGrpcCommunicator for client-side operations
    """

    def __init__(
        self,
        client_id: str,
        master_addr: str,
        master_port: int,
        max_send_message_length: int,
        max_receive_message_length: int,
        channel_credentials: grpc.ChannelCredentials,
        retry_delay: float = 5.0,
        max_retries: int = 3,
        client_timeout: float = 60,
        compressor=None,
        communicate_params: bool = True,
        agg_device: torch.device | str | None = None,
        chunk_threshold_bytes: int | None = None,
        chunk_payload_bytes: int | None = None,
        rpc_timeout: float = GRPC_CLIENT_TIMEOUT_SEC,
        connect_timeout: float = 30.0,
        server_name_override: str | None = None,
    ):
        """
        Initialize gRPC client with connection and retry settings.

        Args:
            client_id: This client's rank as a string. NOT sent to the server (the server
                derives the identity from our certificate); it is compared with what the
                server says it derived, to catch a certificate/rank mix-up at start-up.
            master_addr: gRPC server address
            master_port: gRPC server port
            max_send_message_length: Maximum outbound message size in bytes
            max_receive_message_length: Maximum inbound message size in bytes
            channel_credentials: mTLS credentials (security.channel_credentials)
            retry_delay: Seconds between connection retry attempts
            max_retries: Maximum connection retry attempts
            client_timeout: Seconds to wait for server responses
            rpc_timeout: Deadline set on every data-carrying RPC (seconds)
            connect_timeout: Seconds to wait for the TLS channel and for RegisterClient
            server_name_override: TESTING ONLY, name checked against the server certificate
        """
        print(f"addr={master_addr}:{master_port}")

        # Store configuration
        self.client_id = client_id
        self.master_addr = master_addr
        self.master_port = master_port
        self.max_send_message_length = max_send_message_length
        self.max_receive_message_length = max_receive_message_length
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        self.client_timeout = client_timeout
        self.channel_credentials = channel_credentials
        self.rpc_timeout = float(rpc_timeout)
        self.connect_timeout = float(connect_timeout)
        self.server_name_override = server_name_override
        # self.compressor = TopKCompression(compress_ratio=0.01)
        self.compressor = compressor
        self.communicate_params = bool(communicate_params)
        self.agg_device = torch.device(agg_device or "cpu")
        # self.compressor = None
        self.last_tensordict_submitted = None
        self.logger = None
        self.chunk_threshold_bytes = int(
            GRPC_CHUNK_THRESHOLD_BYTES
            if chunk_threshold_bytes is None
            else chunk_threshold_bytes
        )
        self.chunk_payload_bytes = int(
            GRPC_CHUNK_PAYLOAD_BYTES
            if chunk_payload_bytes is None
            else chunk_payload_bytes
        )
        self._used_chunk_stream = False

        # Initialize connection state
        self.channel = None
        self.stub = None

        # Establish connection with retry logic
        self._establish_connection()

    @property
    def aggregation_metric(self) -> str:
        return aggregation_metric_for_communicate_params(self.communicate_params)

    def set_logger(self, logger: MetricLogger):
        self.logger = logger

    def _establish_connection(self):
        """Open the mTLS channel, wait for the handshake, register (with retries).

        Refusals that retrying cannot fix (certificate not allowed, wrong rank) raise at once.
        """
        target = f"{self.master_addr}:{self.master_port}"
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            self.channel = grpc.secure_channel(
                target,
                self.channel_credentials,
                options=security.client_options(
                    self.max_send_message_length,
                    self.max_receive_message_length,
                    server_name_override=self.server_name_override,
                ),
            )
            try:
                ready = grpc.channel_ready_future(self.channel)
                try:
                    ready.result(timeout=self.connect_timeout)
                finally:
                    ready.cancel()
                self.stub = secure_grpc_pb2_grpc.SecureGrpcServerStub(self.channel)
                response = self.stub.RegisterClient(
                    pb2.CallerRequest(), timeout=self.connect_timeout
                )
            except grpc.FutureTimeoutError as e:
                last_error = e
            except grpc.RpcError as e:
                if security.is_auth_error(e):
                    self.channel.close()
                self._raise_if_auth_error(e)
                last_error = e
            else:
                if not response.success or str(response.participant_id) != str(self.client_id):
                    self.channel.close()
                    raise RuntimeError(
                        f"The server registered this client as participant "
                        f"{response.participant_id!r} but this is rank {self.client_id!r}: the "
                        "certificate presented does not belong to this rank."
                    )
                print(
                    f"Register | success={response.success} | "
                    f"server identified me as participant {response.participant_id}"
                )
                return

            self.channel.close()
            if attempt >= self.max_retries:
                raise RuntimeError(
                    f"Failed to connect to server {target} after {self.max_retries} attempts. "
                    + security.tls_failure_hint(self.master_addr)
                ) from last_error
            print(f"Retry {attempt}/{self.max_retries} | {self.retry_delay}s delay")
            time.sleep(self.retry_delay)

    @staticmethod
    def _raise_if_auth_error(err) -> None:
        """UNAUTHENTICATED / PERMISSION_DENIED: retrying cannot help, say what to check."""
        if security.is_auth_error(err):
            raise RuntimeError(
                f"Server refused this client ({err.code().name}: {err.details()}). Not "
                "retrying. Check that this rank uses the certificate whose CN is allowed "
                "for it, and that it was issued by the CA the server trusts."
            ) from err

    def get_broadcast_state(self) -> Dict[str, torch.Tensor]:
        """
        Retrieve broadcast state from server with polling and retry logic.

        Continuously polls server until broadcast state is available.
        Used during broadcast operations to receive global model.

        Returns:
            Dictionary mapping parameter names to tensor values
        """
        print("Waiting for server to broadcast model")

        poll_count = 0
        error_count = 0

        while True:
            try:
                request = pb2.CallerRequest()
                response = self.stub.GetBroadcastState(request, timeout=self.rpc_timeout)
                if response.is_ready and int(response.n_chunks) > 1:
                    print(
                        f"Broadcast needs {response.n_chunks} chunks; using broadcast stream"
                    )
                    return self._get_broadcast_state_stream()
                if response.is_ready:
                    print(f"Retrieving response on the client side; client_id = {self.client_id}")
                    tensordict = proto_to_tensordict(response.tensor_dict)
                    with torch.no_grad():
                        for key, tensor in tensordict.items():
                            pass
                    print(f"Received {get_msg_info(tensordict)}")
                    return tensordict
                poll_count += 1
                print(f"Polling | {poll_count} total | {self.retry_delay}s delay")
                time.sleep(self.retry_delay)

            except grpc.RpcError as e:
                self._raise_if_auth_error(e)
                error_count += 1
                if error_count > self.max_retries:
                    raise RuntimeError(
                        f"Failed to get broadcast state after {self.max_retries} retries"
                    ) from e
                print(
                    f"Retry {error_count}/{self.max_retries} | {self.retry_delay}s delay"
                )
                time.sleep(self.retry_delay)

    def _get_broadcast_state_stream(self) -> Dict[str, torch.Tensor]:
        """Assemble streamed broadcast chunks into one tensordict."""
        error_count = 0
        while True:
            try:
                request = pb2.CallerRequest()
                decoded_chunks = []
                stream_ready = False
                for response in self.stub.GetBroadcastStateStream(request, timeout=self.rpc_timeout):
                    if not response.is_ready:
                        stream_ready = False
                        decoded_chunks = []
                        break
                    stream_ready = True
                    part, _ = proto_to_tensordict_extended(
                        response.tensor_dict,
                        overlay_base=None,
                        compute_device=self.agg_device,
                    )
                    decoded_chunks.append(part)
                if not stream_ready:
                    print(
                        f"Broadcast stream not ready | {self.retry_delay}s delay"
                    )
                    time.sleep(self.retry_delay)
                    continue
                tensordict = assemble_payload_chunks(decoded_chunks)
                print(
                    f"Received {get_msg_info(tensordict)} "
                    f"from {len(decoded_chunks)} broadcast chunk(s)"
                )
                return tensordict
            except grpc.RpcError as e:
                self._raise_if_auth_error(e)
                error_count += 1
                if error_count > self.max_retries:
                    raise RuntimeError(
                        f"Failed to stream broadcast state after {self.max_retries} retries"
                    ) from e
                print(
                    f"Broadcast stream | error {error_count}/{self.max_retries} | "
                    f"{self.retry_delay}s delay"
                )
                time.sleep(self.retry_delay)

    def submit_for_aggregation(
        self,
        tensordict: Dict[str, torch.Tensor],
        reduction_type: AggregationOp,
        num_samples: int = 0,
        compress: bool | None = None,
    ) -> bool:
        """
        Submit local tensors to server for distributed aggregation.

        Args:
            tensordict: Local tensors to contribute to aggregation
            reduction_type: SUM, MEAN, or MAX aggregation operation
            num_samples: Training samples represented by this contribution
        """
        try:
            # Compress only when the caller asks (grad/param hop). BN stays dense.
            if compress is None:
                compress = int(num_samples) > 0
            active_compressor = self.compressor if compress else None
            ctx = self.logger.log_duration("training_compression_time") if self.logger else nullcontext()
            t0 = time.perf_counter()
            with ctx:
                compressed_tensordict = compress_message_tensors(
                    tensordict, active_compressor, self.aggregation_metric
                )
                compressor_name = compressor_proto_name(active_compressor)
                if isinstance(tensordict, torch.Tensor):
                    compressor_name = None
                    compressed_tensordict = tensordict
                self.last_tensordict_submitted = tensordict
            # pack = Top-K/QSGD (if any); protobuf encode happens per unary/chunk below
            if self.logger:
                accumulate_iter_comm(
                    self.logger, "grpc_pack_s", time.perf_counter() - t0
                )
            # encode_end = time.time()
            # upload_start = time.time()
            ctx = self.logger.log_duration("training_upstream_upload_time") if self.logger else nullcontext()
            t0 = time.perf_counter()
            with ctx:
                ok = self._send_packed_payload(
                    compressed_tensordict,
                    compressor_name,
                    reduction_type,
                    int(num_samples),
                )
            if self.logger:
                accumulate_iter_comm(
                    self.logger, "grpc_upstream_s", time.perf_counter() - t0
                )
            if ok:
                payload = "params" if self.communicate_params else "grads"
                print(f"Successfully sent local {payload} to server")
                return True
            print("Submit failed")
            return False
        except grpc.RpcError as e:
            self._raise_if_auth_error(e)
            print(f"Submit exception | {e}")
            return False

    def _send_packed_payload(
        self,
        compressed_tensordict,
        compressor_name,
        reduction_type: AggregationOp,
        num_samples: int,
    ) -> bool:
        """Unary if packed size fits; otherwise one streaming RPC of chunks."""
        self._used_chunk_stream = False
        use_stream = should_chunk_payload(
            compressed_tensordict, self.chunk_threshold_bytes
        )
        if not use_stream:
            proto_tensordict = _to_proto(
                compressed_tensordict, compressor_name
            )
            request = pb2.AggregationRequest(
                tensor_dict=proto_tensordict,
                reduction_type=reduction_type.value,
                num_samples=int(num_samples),
                n_chunks=1,
            )
            if packed_message_bytes(request) <= self.max_send_message_length:
                response = self.stub.SubmitForAggregation(request, timeout=self.rpc_timeout)
                return bool(response.success)
            use_stream = True
            print(
                "Packed unary exceeds channel cap; falling back to chunk stream"
            )

        chunks = list(
            iter_payload_chunks(
                compressed_tensordict,
                min(self.chunk_payload_bytes, int(self.max_send_message_length * 0.85)),
            )
        )
        n_chunks = len(chunks)
        self._used_chunk_stream = True
        print(
            f"Uplink packed estimate needs {n_chunks} chunk(s) "
            f"(threshold={self.chunk_threshold_bytes})"
        )

        def request_iter():
            for index, chunk in enumerate(chunks):
                yield pb2.AggregationRequest(
                    tensor_dict=_to_proto(chunk, compressor_name),
                    reduction_type=reduction_type.value,
                    num_samples=int(num_samples) if index == 0 else 0,
                    chunk_index=index,
                    n_chunks=n_chunks,
                )

        response = self.stub.SubmitAggregationStream(
            request_iter(), timeout=self.rpc_timeout
        )
        return bool(response.success)

    def get_aggregation_result(self) -> Dict[str, torch.Tensor]:
        """
        Retrieve aggregated result from server with timeout and polling.

        Waits for server to complete aggregation across all clients,
        then returns the aggregated tensors.

        Returns:
            Dictionary mapping parameter names to aggregated tensor values

        Raises:
            RuntimeError: If aggregation times out or max retries exceeded
        """
        print(
            f"Waiting for server to aggregate models (timeout={self.client_timeout}s)"
        )
        if self._used_chunk_stream:
            return self._get_aggregation_result_stream()

        start_time = time.time()
        poll_count = 0
        error_count = 0

        while True:
            elapsed = time.time() - start_time
            if elapsed > self.client_timeout:
                raise RuntimeError(f"Aggregation timeout ({self.client_timeout}s)")
            try:
                request = pb2.CallerRequest()
                ctx = self.logger.log_duration("training_downstream_download_time") if self.logger else nullcontext()
                t0 = time.perf_counter()
                with ctx:
                    response = self.stub.GetAggregationResult(request, timeout=self.rpc_timeout)
                if self.logger:
                    accumulate_iter_comm(
                        self.logger, "grpc_downstream_s", time.perf_counter() - t0
                    )
                if response.is_ready and int(response.n_chunks) > 1:
                    print(
                        f"Result needs {response.n_chunks} chunks; using result stream"
                    )
                    return self._get_aggregation_result_stream()
                if response.is_ready:
                    ctx = self.logger.log_duration("training_decompression_time") if self.logger else nullcontext()
                    t0 = time.perf_counter()
                    with ctx:
                        tensordict, is_model_communicated = proto_to_tensordict_extended(
                            response.tensor_dict,
                            overlay_base=None,
                            compute_device=self.agg_device,
                        )
                    if self.logger:
                        accumulate_iter_comm(
                            self.logger, "grpc_unpack_s", time.perf_counter() - t0
                        )
                    print(
                        f"Received {get_msg_info(tensordict)} (waited {elapsed:.1f}s)"
                    )
                    return tensordict
                poll_count += 1
                remaining = self.client_timeout - elapsed
                print(f"Waiting | poll {poll_count} | {remaining:.1f}s remaining")
                time.sleep(min(self.retry_delay, remaining))

            except grpc.RpcError as e:
                self._raise_if_auth_error(e)
                if security.is_busy_error(e):
                    # Server is at its maximum_concurrent_rpcs; back off and ask again.
                    error_count += 1
                    if error_count > self.max_retries:
                        raise RuntimeError(
                            f"Server stayed at its concurrent-RPC limit for {self.max_retries} retries"
                        ) from e
                    print(f"Server busy | retry in {self.retry_delay}s")
                    time.sleep(self.retry_delay)
                    continue
                if e.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:
                    print("Unary result too large; using result stream")
                    return self._get_aggregation_result_stream()
                error_count += 1
                if error_count > self.max_retries:
                    raise RuntimeError(
                        f"Failed to get aggregation result after {self.max_retries} retries"
                    ) from e
                print(
                    f"Aggregation fetch | error {error_count}/{self.max_retries} | retry in {self.retry_delay}s"
                )
                time.sleep(self.retry_delay)

    def _get_aggregation_result_stream(self) -> Dict[str, torch.Tensor]:
        """Assemble streamed result chunks into one tensordict (one logical Get)."""
        start_time = time.time()
        error_count = 0
        while True:
            elapsed = time.time() - start_time
            if elapsed > self.client_timeout:
                raise RuntimeError(f"Aggregation timeout ({self.client_timeout}s)")
            try:
                request = pb2.CallerRequest()
                ctx = self.logger.log_duration("training_downstream_download_time") if self.logger else nullcontext()
                t0 = time.perf_counter()
                decoded_chunks = []
                stream_ready = False
                with ctx:
                    for response in self.stub.GetAggregationResultStream(request, timeout=self.rpc_timeout):
                        if not response.is_ready:
                            stream_ready = False
                            decoded_chunks = []
                            break
                        stream_ready = True
                        part, _ = proto_to_tensordict_extended(
                            response.tensor_dict,
                            overlay_base=None,
                            compute_device=self.agg_device,
                        )
                        decoded_chunks.append(part)
                if self.logger:
                    accumulate_iter_comm(
                        self.logger, "grpc_downstream_s", time.perf_counter() - t0
                    )
                if not stream_ready:
                    remaining = self.client_timeout - elapsed
                    print(f"Waiting | stream not ready | {remaining:.1f}s remaining")
                    time.sleep(min(self.retry_delay, remaining))
                    continue
                t1 = time.perf_counter()
                tensordict = assemble_payload_chunks(decoded_chunks)
                if self.logger:
                    accumulate_iter_comm(
                        self.logger, "grpc_unpack_s", time.perf_counter() - t1
                    )
                print(
                    f"Received {get_msg_info(tensordict)} "
                    f"from {len(decoded_chunks)} chunks (waited {elapsed:.1f}s)"
                )
                return tensordict
            except grpc.RpcError as e:
                self._raise_if_auth_error(e)
                error_count += 1
                if error_count > self.max_retries:
                    raise RuntimeError(
                        f"Failed to stream aggregation result after {self.max_retries} retries"
                    ) from e
                print(
                    f"Aggregation stream | error {error_count}/{self.max_retries} | "
                    f"retry in {self.retry_delay}s"
                )
                time.sleep(self.retry_delay)
