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

"""Secure variant of ``grpc.GrpcCommunicator``: the same centralized broadcast / aggregate
protocol over mutual TLS.

Message handling (``_extract_tensordict_from_msg`` / ``_apply_tensordict_to_msg``), the
server-side participation in aggregation, compression, chunking and the public API
(``setup``, ``broadcast``, ``aggregate``, ``close``, ...) are the existing OmniFed code.

What is new is in ``__init__`` (security arguments) and ``_setup``: the server opens a
*secure port* that requires a client certificate, the clients open a *secure channel* that
verifies the server, and the server decides who a caller is from the certificate.

The file is deliberately NOT called ``grpc.py``: a module of that name next to scripts
shadows the grpcio package.
"""

from __future__ import annotations

import warnings
from concurrent import futures

import grpc
import rich.repr
import torch
from torch import nn
from ...utils import print
from ...utils import MetricLogger
from ..base import AggregationOp, BaseCommunicator
from ..compression import resolve_compressor
from ..grpc_limits import (
    GRPC_AGGREGATION_TIMEOUT_SEC,
    GRPC_CHUNK_PAYLOAD_BYTES,
    GRPC_CHUNK_THRESHOLD_BYTES,
    GRPC_CLIENT_TIMEOUT_SEC,
    GRPC_MAX_MESSAGE_BYTES,
)
from ..utils import get_msg_info
from . import secure_grpc_pb2, secure_grpc_pb2_grpc, security
from .secure_codec import check_wire_compat
from .secure_grpc_client import SecureGrpcClient
from .secure_grpc_server import SecureGrpcServer


@rich.repr.auto
class SecureGrpcCommunicator(BaseCommunicator):
    """
    mTLS gRPC communication backend with centralized coordination.

    Uses client-server architecture where rank 0 acts as coordinator for
    broadcast and aggregation operations, exactly like ``GrpcCommunicator``,
    but every connection is mutually authenticated: the server proves itself to the
    clients with its certificate, each client proves itself with its own, and the server
    only serves callers whose certificate name is on its allowlist.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        master_addr: str = "127.0.0.1",
        master_port: int = 50051,
        max_workers: int = 10,
        max_send_message_length: int = GRPC_MAX_MESSAGE_BYTES,
        max_receive_message_length: int = GRPC_MAX_MESSAGE_BYTES,
        aggregation_timeout: float = GRPC_AGGREGATION_TIMEOUT_SEC,
        client_timeout: float = GRPC_CLIENT_TIMEOUT_SEC,
        retry_delay: float = 5.0,
        max_retries: int = 5,
        compressor=None,
        client_compressor=None,
        server_compressor=None,
        communicate_params: bool = True,
        normalize_by_total_samples: bool = False,
        backend: object = None,
        chunk_threshold_bytes: int = GRPC_CHUNK_THRESHOLD_BYTES,
        chunk_payload_bytes: int = GRPC_CHUNK_PAYLOAD_BYTES,
        # --- security (all optional; see secure_grpc/SECURE_GRPC.md) ---
        tls_dir: str | None = None,
        tls_ca_cert: str | None = None,
        tls_cert: str | None = None,
        tls_key: str | None = None,
        tls_server_name_override: str | None = None,
        tls_allow_loose_key_permissions: bool = False,
        client_cn_prefix: str = security.DEFAULT_CLIENT_CN_PREFIX,
        client_identities: dict | None = None,
        bind_addr: str = "[::]",
        rpc_timeout: float = GRPC_CLIENT_TIMEOUT_SEC,
        connect_timeout: float = 30.0,
        long_poll_sec: float = 30.0,
        require_client_deadline: bool = True,
        max_concurrent_rpcs: int | None = None,
        max_concurrent_streams: int | None = 16,
        keepalive_sec: float | None = 60.0,
        handshake_timeout_sec: float | None = 20.0,
        idle_timeout_sec: float | None = None,
        **kwargs: object,
    ) -> None:
        """
        Initialize the mTLS gRPC federated learning communicator.

        ``backend`` is accepted (and ignored) when Hydra merges TorchDist leftovers.

        Security arguments:
            tls_dir: directory laid out like gen_certs.sh output (``ca.crt``,
                ``server/server.crt|key``, ``clients/<name>.crt|key``). One shared config then
                works on every rank. Individual ``tls_*`` paths override it; environment
                variables OMNIFED_TLS_CA / _CERT / _KEY are the last fallback. Paths may use
                ``{rank}``.
            tls_server_name_override: TESTING ONLY (127.0.0.1 with a cert for another name).
            client_cn_prefix / client_identities: which certificate names may call. Default:
                CN ``<prefix><rank>`` is rank ``rank``. ``client_identities={"alice": 1}`` maps
                arbitrary CNs to ranks instead.
            bind_addr: interface the server listens on (``[::]`` = all).
            rpc_timeout: deadline on every data RPC. connect_timeout: TLS connect + register.
            long_poll_sec: how long the server holds one result request open before answering
                "not ready" (the client re-asks). require_client_deadline: refuse calls with
                no deadline.
            max_concurrent_rpcs: server cap on in-flight RPCs (default: enough for every
                client plus headroom). Excess calls get RESOURCE_EXHAUSTED.
            max_concurrent_streams / keepalive_sec / handshake_timeout_sec / idle_timeout_sec:
                server connection hygiene (see security.server_options).
        """
        if backend is not None:
            warnings.warn(
                f"SecureGrpcCommunicator ignores TorchDist-only argument backend={backend!r}",
                stacklevel=2,
            )
        if kwargs:
            warnings.warn(
                f"SecureGrpcCommunicator ignoring unused keyword arguments: {sorted(kwargs)}",
                stacklevel=2,
            )

        # slurm_worker uses ``backend`` only to pick CUDA vs CPU (no torch.distributed PG).
        self.backend = "nccl" if torch.cuda.is_available() else "gloo"

        super().__init__(rank, world_size, master_addr, master_port)
        print(f"rank={rank}/{world_size} | addr={master_addr}:{master_port}")

        # Core distributed parameters
        # self.rank: int = rank
        # self.world_size: int = world_size
        # self.master_addr: str = master_addr
        # self.master_port: int = master_port

        # gRPC configuration
        self.max_workers = max_workers
        self.max_send_message_length = max_send_message_length
        self.max_receive_message_length = max_receive_message_length

        # Timeout and retry settings
        self.aggregation_timeout = aggregation_timeout
        self.client_timeout = client_timeout
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        self.logger = None

        # Runtime components (initialized in _setup)
        self._server = None
        self._client = None
        self._servicer = None
        self.compressor = resolve_compressor(
            compressor, client_compressor, server_compressor
        )
        self.communicate_params = bool(communicate_params)
        self.normalize_by_total_samples = bool(normalize_by_total_samples)
        self._aggregation_num_samples = 0
        self._aggregation_compress = False
        self._agg_device = torch.device("cpu")
        self.chunk_threshold_bytes = int(chunk_threshold_bytes)
        self.chunk_payload_bytes = int(chunk_payload_bytes)

        # Security configuration (validated now so a bad config fails before any socket opens)
        self._identities = security.normalize_identities(
            dict(client_identities) if client_identities else None,
            world_size,
            client_cn_prefix,
        )
        self.client_cn_prefix = client_cn_prefix
        self.client_identities = client_identities
        if tls_dir:
            d_ca, d_cert, d_key = security.tls_paths_from_dir(tls_dir, rank, self._identities)
            tls_ca_cert, tls_cert, tls_key = (
                tls_ca_cert or d_ca,
                tls_cert or d_cert,
                tls_key or d_key,
            )
        self.tls_dir = tls_dir
        self.tls_ca_cert = tls_ca_cert  # paths only; key material is read in _setup
        self.tls_cert = tls_cert
        self.tls_key = tls_key
        self.tls_server_name_override = tls_server_name_override
        self.tls_allow_loose_key_permissions = bool(tls_allow_loose_key_permissions)
        self.bind_addr = bind_addr
        self.rpc_timeout = float(rpc_timeout)
        self.connect_timeout = float(connect_timeout)
        self.long_poll_sec = float(long_poll_sec)
        self.require_client_deadline = bool(require_client_deadline)
        self.max_concurrent_rpcs = max_concurrent_rpcs
        self.max_concurrent_streams = max_concurrent_streams
        self.keepalive_sec = keepalive_sec
        self.handshake_timeout_sec = handshake_timeout_sec
        self.idle_timeout_sec = idle_timeout_sec
        for name in ("rpc_timeout", "connect_timeout", "long_poll_sec"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.long_poll_sec >= self.rpc_timeout:
            warnings.warn(
                f"long_poll_sec={self.long_poll_sec} >= rpc_timeout={self.rpc_timeout}: the "
                "server will cut each wait short to stay inside the client's deadline",
                stacklevel=2,
            )

    def set_agg_device(self, device: torch.device | str) -> None:
        """Wire resolver ``agg_device`` into client compress/decompress and server SUM."""
        self._agg_device = torch.device(device)
        if getattr(self, "_servicer", None) is not None:
            self._servicer.set_agg_device(self._agg_device)
        if getattr(self, "_client", None) is not None:
            self._client.agg_device = self._agg_device
        print(f"[grpc] agg_device={self._agg_device}")

    @property
    def client(self):
        """
        gRPC client instance for non-server ranks.

        Returns:
            GrpcClient instance for communication with server

        Raises:
            RuntimeError: If accessed before setup() or on server rank
        """
        if self._client is None:
            raise RuntimeError("gRPC client not initialized. Call setup() first.")
        return self._client

    @property
    def server(self):
        """
        gRPC server instance for rank 0.

        Returns:
            grpc.Server instance handling client connections

        Raises:
            RuntimeError: If accessed before setup() or on client rank
        """
        if self._server is None:
            raise RuntimeError("gRPC server not initialized. Call setup() first.")
        return self._server

    @property
    def servicer(self):
        """
        gRPC servicer instance containing FL coordination logic.

        Returns:
            CentralServerServicer managing aggregation and broadcast state

        Raises:
            RuntimeError: If accessed before setup() or on client rank
        """
        if self._servicer is None:
            raise RuntimeError("gRPC servicer not initialized. Call setup() first.")
        return self._servicer

    @property
    def is_server(self) -> bool:
        """
        True if this rank acts as the central gRPC server.

        Returns:
            True for rank 0, False for all other ranks
        """
        return self.rank == 0

    def _setup(self):
        """
        Initialize the mTLS gRPC server (rank 0) or client (other ranks).

        Server setup: Loads certificates, creates the gRPC server (concurrency-capped),
            the servicer, opens the secure port and starts listening
        Client setup: Loads this rank's certificate, opens the secure channel and registers
        """
        check_wire_compat(secure_grpc_pb2)
        if self.is_server:
            material = security.load_tls_material(
                self.tls_ca_cert,
                self.tls_cert,
                self.tls_key,
                rank=self.rank,
                allow_loose_key_permissions=self.tls_allow_loose_key_permissions,
            )
            cap, workers = security.plan_concurrency(
                self.world_size, self.max_workers, self.max_concurrent_rpcs
            )
            self._server = grpc.server(
                futures.ThreadPoolExecutor(
                    max_workers=workers, thread_name_prefix="secure-grpc"
                ),
                maximum_concurrent_rpcs=cap,
                options=security.server_options(
                    self.max_send_message_length,
                    self.max_receive_message_length,
                    max_concurrent_streams=self.max_concurrent_streams,
                    keepalive_sec=self.keepalive_sec,
                    handshake_timeout_sec=self.handshake_timeout_sec,
                    idle_timeout_sec=self.idle_timeout_sec,
                ),
            )

            self._servicer = SecureGrpcServer(
                world_size=self.world_size,
                identities=self._identities,
                compressor=self.compressor,
                communicate_params=self.communicate_params,
                normalize_by_total_samples=self.normalize_by_total_samples,
                agg_device=getattr(self, "_agg_device", torch.device("cpu")),
                chunk_threshold_bytes=self.chunk_threshold_bytes,
                chunk_payload_bytes=self.chunk_payload_bytes,
                long_poll_sec=self.long_poll_sec,
                require_client_deadline=self.require_client_deadline,
            )
            secure_grpc_pb2_grpc.add_SecureGrpcServerServicer_to_server(
                self._servicer, self._server
            )

            address = f"{self.bind_addr}:{self.master_port}"
            try:
                bound = self._server.add_secure_port(
                    address, security.server_credentials(material)
                )
            except RuntimeError as exc:  # newer grpcio raises instead of returning 0
                raise RuntimeError(
                    f"Cannot listen on {address} (port in use, or no permission?)"
                ) from exc
            if not bound:
                raise RuntimeError(f"Cannot listen on {address} (port in use, or no permission?)")
            print(
                f"Secure server listening on {address} | mTLS required | "
                f"allowed clients={sorted(self._identities)} | "
                f"max_concurrent_rpcs={cap} workers={workers}"
            )
            self._server.start()
        else:
            material = security.load_tls_material(
                self.tls_ca_cert,
                self.tls_cert,
                self.tls_key,
                rank=self.rank,
                expected_common_name=security.expected_cn_for_rank(self._identities, self.rank),
                allow_loose_key_permissions=self.tls_allow_loose_key_permissions,
            )
            self._client = SecureGrpcClient(
                client_id=str(self.rank),
                master_addr=self.master_addr,
                master_port=self.master_port,
                max_send_message_length=self.max_send_message_length,
                max_receive_message_length=self.max_receive_message_length,
                channel_credentials=security.channel_credentials(material),
                retry_delay=self.retry_delay,
                max_retries=self.max_retries,
                client_timeout=self.client_timeout,
                compressor=self.compressor,
                communicate_params=self.communicate_params,
                agg_device=getattr(self, "_agg_device", torch.device("cpu")),
                chunk_threshold_bytes=self.chunk_threshold_bytes,
                chunk_payload_bytes=self.chunk_payload_bytes,
                rpc_timeout=self.rpc_timeout,
                connect_timeout=self.connect_timeout,
                server_name_override=self.tls_server_name_override,
            )
            if self.logger is not None:
                self._client.set_logger(self.logger)

    def broadcast(
        self,
        msg: BaseCommunicator.MsgT,
        src: int = 0,
    ) -> BaseCommunicator.MsgT:
        """
        Broadcast message from server to all clients via gRPC.

        Server (rank 0): Stores message in broadcast state for client retrieval
        Clients: Retrieve broadcast state from server via polling

        Args:
            msg: Model, tensor dict, or tensor to broadcast
            src: Source rank (unused in centralized gRPC - always rank 0)

        Returns:
            Broadcasted message with updated values
        """
        print(f"{get_msg_info(msg)} | src={src}")
        if self.is_server:
            # Server: Store broadcast state for client retrieval
            tensordict = self._extract_tensordict_from_msg(msg)
            self.servicer.set_broadcast_state(tensordict)
            return msg
        else:
            # Client: Retrieve broadcast state from server
            tensordict = self.client.get_broadcast_state()
            return self._apply_tensordict_to_msg(msg, tensordict)

    def set_logger(self, logger: MetricLogger):
        """Attach metrics logger; forward to gRPC client when it exists."""
        self.logger = logger
        if self._client is not None:
            self._client.set_logger(logger)

    def set_aggregation_num_samples(self, num_samples: int) -> None:
        """n_i for the next ``aggregate()`` (sample-weighted SUM)."""
        self._aggregation_num_samples = max(int(num_samples), 0)

    def set_aggregation_compress(self, enabled: bool) -> None:
        """Whether Top-K/QSGD applies on the next ``aggregate()``."""
        self._aggregation_compress = bool(enabled)

    def aggregate(
        self,
        msg: BaseCommunicator.MsgT,
        reduction: AggregationOp,
    ) -> BaseCommunicator.MsgT:
        """
        Aggregate message across all ranks via central gRPC server.

        All ranks submit their data to the server, which performs aggregation
        when all participants have contributed, then distributes results.

        Args:
            msg: Model, tensor dict, or tensor to aggregate
            reduction: SUM, MEAN, or MAX aggregation operation

        Returns:
            Aggregated message with combined values from all ranks
        """
        try:
            tensordict = self._extract_tensordict_from_msg(msg)
            print(f"{get_msg_info(msg)} | reduction={reduction}")
            aggregated_tensordict = self._grpc_aggregate(tensordict, reduction)
            return self._apply_tensordict_to_msg(msg, aggregated_tensordict)
        finally:
            self._aggregation_num_samples = 0
            self._aggregation_compress = False

    def _extract_tensordict_from_msg(
        self,
        msg: BaseCommunicator.MsgT,
        # TODO: Consider adding control arguments like utils.scale_params:
        # requires_grad: Optional[bool] = None,
        # include_buffers: bool = True,
        # filter_fn: Optional[Callable[[str, torch.Tensor], bool]] = None
    ) -> BaseCommunicator.MsgT:
        """
        Extract tensors from message for gRPC serialization.

        Args:
            msg: Model, tensor dict, or single tensor

        Returns:
            Dictionary mapping parameter names to tensor values
        """
        if isinstance(msg, nn.Module):
            result = dict()
            for name, param in msg.named_parameters():
                if param.requires_grad:
                    if self.communicate_params:
                        result[name] = param.data
                    elif param.grad is not None:
                        result[name] = param.grad.detach()
                    else:
                        result[name] = torch.zeros_like(param.data)
            # Param FedAvg only: BN running stats are sample-weighted separately in
            # grad mode (including them in SUM / sample-weighted grad agg corrupts BN).
            if self.communicate_params:
                for name, buffer in msg.named_buffers():
                    if buffer is None:  # type: ignore
                        warnings.warn(
                            f"Buffer '{name}' is None, skipping serialization"
                        )
                        continue
                    if not buffer.dtype.is_floating_point:
                        continue  # Skip integer buffers like num_batches_tracked
                    result[name] = buffer.data
            return result
        elif isinstance(msg, dict):
            return msg
        else:
            return {"tensor": msg}

    def _apply_tensordict_to_msg(
        self,
        msg: BaseCommunicator.MsgT,
        tensordict: BaseCommunicator.MsgT,
        # TODO: Consider adding control arguments like utils.scale_params:
        # requires_grad: Optional[bool] = None,
        # include_buffers: bool = True,
        # filter_fn: Optional[Callable[[str, torch.Tensor], bool]] = None
    ) -> BaseCommunicator.MsgT:
        """
        Apply deserialized tensors back to original message format.

        Args:
            msg: Original message providing structure and device info
            tensordict: Deserialized tensors from gRPC

        Returns:
            Message with updated tensor values and proper device placement
        """
        if isinstance(msg, nn.Module):
            with torch.no_grad():
                # Apply parameters or aggregated gradients
                for name, param in msg.named_parameters():
                    if param.requires_grad and name in tensordict:
                        tensor = tensordict[name].to(param.device)
                        if self.communicate_params:
                            param.data.copy_(tensor)
                        else:
                            param.grad = tensor
                if self.communicate_params:
                    for name, buffer in msg.named_buffers():
                        if buffer is None:  # type: ignore
                            warnings.warn(
                                f"Buffer '{name}' is None, skipping deserialization"
                            )
                            continue
                        if not buffer.dtype.is_floating_point:
                            continue  # Skip integer buffers like num_batches_tracked
                        if name not in tensordict:
                            continue  # Buffer not in received data
                        tensor = tensordict[name].to(buffer.device)
                        buffer.data.copy_(tensor)
            return msg
        elif isinstance(msg, dict):
            return tensordict
        else:
            return tensordict.get("tensor", msg)

    def _grpc_aggregate(self, tensordict: dict, reduction: AggregationOp) -> dict:
        """
        Perform distributed aggregation via gRPC protocol.

        Server: Submits data to local session and waits for aggregation
        Clients: Submit data to server and retrieve aggregated result

        Args:
            tensordict: Local tensors to contribute
            reduction: Aggregation operation type

        Returns:
            Aggregated tensor dictionary
        """
        if self.is_server:
            current_session = self._submit_server_data(tensordict, reduction)
            return self._wait_for_aggregation_result(current_session)
        else:
            if not self.client.submit_for_aggregation(
                tensordict,
                reduction,
                num_samples=self._aggregation_num_samples,
                compress=self._aggregation_compress,
            ):
                raise RuntimeError(
                    "gRPC aggregation submit failed (see server log for session errors)"
                )
            return self.client.get_aggregation_result()

    def _submit_server_data(self, tensordict: dict, reduction: AggregationOp) -> int:
        """
        Submit server's local data to current aggregation session.

        Args:
            tensordict: Server's local tensors
            reduction: Aggregation operation type

        Returns:
            Session ID for tracking aggregation progress
        """
        with self.servicer.lock:
            current_session = self.servicer.current_aggregation_session
            session_state = self.servicer.aggregation_state[current_session]

            # Set reduction type
            reduction_str = reduction.value
            if session_state["reduction_type"] is None:
                session_state["reduction_type"] = reduction_str
            elif session_state["reduction_type"] != reduction_str:
                raise ValueError(
                    f"Reduction type mismatch - expected {session_state['reduction_type']}, got {reduction_str}"
                )

            # Running SUM/MAX into one accumulator (device = servicer agg_device).
            if isinstance(tensordict, dict):
                payload = tensordict
            elif torch.is_tensor(tensordict):
                payload = tensordict
            else:
                payload = tensordict
            if self._aggregation_compress:
                session_state["compress_result"] = True
            self.servicer._accumulate_into_session(
                session_state,
                "server",
                payload,
                num_samples=int(self._aggregation_num_samples),
            )
            session_state["total_samples"] = int(
                session_state.get("total_samples", 0)
            ) + int(self._aggregation_num_samples)
            data_count = self.servicer._submitted_count(session_state)

            print(
                f"Server submit | session={current_session} | {data_count}/{self.servicer.world_size}"
            )

            # Trigger aggregation if ready
            self.servicer.perform_aggregation_if_ready(session_state, current_session)

            return current_session

    def _wait_for_aggregation_result(self, session_id: int) -> dict:
        """
        Wait for aggregation to complete and return aggregated tensors.

        Args:
            session_id: Aggregation session identifier

        Returns:
            Aggregated tensor dictionary

        Raises:
            RuntimeError: If aggregation times out or fails
        """
        session_state = self.servicer.aggregation_state[session_id]

        print(f"Server wait | session={session_id}")

        wait_result = session_state["event"].wait(timeout=self.aggregation_timeout)
        if not wait_result:
            raise RuntimeError(
                f"Aggregation timeout ({self.aggregation_timeout}s) for session {session_id}"
            )

        if session_state["result"] is None:
            raise RuntimeError(f"No result available for session {session_id}")

        result = session_state["result"]
        self.servicer.mark_aggregation_result_delivered(session_id, "server")
        return result

    def close(self):
        """
        Clean up gRPC resources and close connections.

        Gracefully shuts down server (rank 0) or closes client connections.
        Should be called when communication is no longer needed.
        """
        print()
        if self._server is not None:
            self._server.stop(grace=15)
        if self._client is not None:
            self._client.channel.close()
