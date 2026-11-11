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

"""Secure variant of ``grpc_server.GrpcServer`` (served over mTLS, caller identity from the cert).

Aggregation, sessions, chunking and compression are the existing OmniFed logic, unchanged.
What differs:

* Every RPC starts with ``_caller(context)``: the caller must have presented a client
  certificate that chains to the configured CA (enforced by the TLS layer) AND whose CN is in
  ``identities``. The participant id used for sessions is derived from that CN. Requests
  carry no ``client_id`` at all, so one rank cannot speak for another.
* ``GetAggregationResult`` / ``GetAggregationResultStream`` no longer block forever: they wait
  at most ``long_poll_sec`` (and never past the client's own deadline), wake early if the
  client hangs up, then answer "not ready" and the client asks again.
* Calls without a client deadline are rejected (``require_client_deadline``).

Serve it with ``grpc.server(..., maximum_concurrent_rpcs=...)`` and ``add_secure_port``; see
``secure_grpc_communicator.py``.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any, Dict, Mapping
import warnings
import traceback

import rich.repr
import torch

from ...utils import print
from . import secure_grpc_pb2 as pb2
from . import secure_grpc_pb2_grpc, security
from ..base import AggregationOp
from ..grpc_chunking import (
    assemble_payload_chunks,
    chunk_count,
    iter_payload_chunks,
    should_chunk_payload,
)
from ..grpc_limits import GRPC_CHUNK_PAYLOAD_BYTES, GRPC_CHUNK_THRESHOLD_BYTES
from ..utils import get_msg_info, proto_to_tensordict_extended
from ..utils import (
    aggregation_metric_for_communicate_params,
    compress_message_tensors,
    compressor_proto_name,
)
from .secure_codec import tensordict_to_proto as _encode_tensordict
from src.omnifed.device_resolver import is_cuda_oom, mapping_to_device


def _to_proto(tensordict, compression_type=None):
    """Encode into the *secure* proto's TensorDict."""
    return _encode_tensordict(pb2, tensordict, compression_type)


@rich.repr.auto
class SecureGrpcServer(secure_grpc_pb2_grpc.SecureGrpcServerServicer):
    """
    gRPC server for federated learning coordination, reachable only by authenticated clients.

    Coordinates broadcast and aggregation operations across multiple clients.
    Handles session management, client synchronization, and tensor aggregation
    with support for multiple concurrent FL rounds. Callers are identified by the
    CN of their verified TLS client certificate.
    """

    def __init__(
        self,
        world_size: int,
        identities: Mapping[str, str],
        compressor=None,
        communicate_params: bool = True,
        normalize_by_total_samples: bool = False,
        agg_device: torch.device | str | None = None,
        chunk_threshold_bytes: int | None = None,
        chunk_payload_bytes: int | None = None,
        long_poll_sec: float = 30.0,
        require_client_deadline: bool = True,
    ):
        """
        Initialize gRPC server for federated learning coordination.

        Args:
            world_size: Total number of FL participants (including server)
            identities: {certificate CN: participant id}. Only these certificate names may
                call; the participant id (the rank, as a string) is what aggregation
                sessions are keyed on. The id comes from the certificate, never from
                the request.
            communicate_params: Aggregate model parameters when True, else gradients
            normalize_by_total_samples: Ignored. SUM with total_samples>=1 always
                does sum(n_i * x_i) / sum(n_i). Kept for Hydra constructor compat.
            long_poll_sec: longest a GetAggregationResult call is held open waiting for
                the aggregate; the client then simply asks again. Never exceeds the
                client's own deadline.
            require_client_deadline: reject calls that carry no deadline.
        """
        print(f"world_size={world_size} | allowed clients={sorted(identities)}")

        # Core configuration
        self.world_size = world_size
        self.communicate_params = bool(communicate_params)
        self.normalize_by_total_samples = bool(normalize_by_total_samples)
        self.registered_clients = set()
        # self.compressor = QSGDQuantCompression()
        self.lock = threading.Lock()
        self.model = None

        # Aggregation session management
        self.current_aggregation_session = 0
        self.aggregation_state: dict[int, dict[str, Any]] = defaultdict(
            self._new_aggregation_session_state
        )

        self.compressor = compressor
        self.result_processors = {}
        # self.compressor = None
        self._broadcast_state = {}
        self.agg_device = torch.device(agg_device or "cpu")
        self._compute_device = self.agg_device
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

        # Security: who may call, and how long one call may hold a worker thread.
        self.identities = dict(identities)
        self._policy = security.IdentityPolicy(self.identities)
        self.long_poll_sec = float(long_poll_sec)
        self.require_client_deadline = bool(require_client_deadline)

    def set_agg_device(self, device: torch.device | str) -> None:
        """Where decompress / running SUM / recompress run (OOM may demote to CPU)."""
        self.agg_device = torch.device(device)
        self._compute_device = self.agg_device
        print(f"[grpc_server] agg_device={self.agg_device}", flush=True)

    @property
    def aggregation_metric(self) -> str:
        return aggregation_metric_for_communicate_params(self.communicate_params)

    @staticmethod
    def _new_aggregation_session_state() -> dict[str, Any]:
        return {
            "data": {},
            "accum": None,
            "result": None,
            "event": threading.Event(),
            "reduction_type": None,
            "total_samples": 0,
            "participants": set(),
            "results_delivered": set(),
            "compress_result": False,
        }

    def _session_has_participant(
        self, session_state: dict[str, Any], participant_id: str
    ) -> bool:
        if participant_id in session_state["data"]:
            return True
        return participant_id in session_state.get("participants", ())

    def _release_aggregation_inputs(self, session_state: dict[str, Any]) -> None:
        """Drop submitted client/server tensors once aggregation result is ready."""
        if session_state["data"]:
            session_state["participants"] = set(session_state["data"].keys()) | set(
                session_state.get("participants") or ()
            )
        session_state["data"].clear()
        session_state["accum"] = None

    def mark_aggregation_result_delivered(
        self, session_id: int, participant_id: str
    ) -> None:
        """Record that a participant fetched the result; drop session when all have."""
        with self.lock:
            self._mark_aggregation_result_delivered_locked(session_id, participant_id)

    def _mark_aggregation_result_delivered_locked(
        self, session_id: int, participant_id: str
    ) -> None:
        session_state = self.aggregation_state.get(session_id)
        if session_state is None:
            return
        delivered = session_state.setdefault("results_delivered", set())
        delivered.add(participant_id)
        participants = session_state.get("participants") or set()
        if participants and delivered >= participants:
            self._drop_aggregation_session(session_id)

    def _drop_aggregation_session(self, session_id: int) -> None:
        """Remove a completed aggregation session and free retained tensors."""
        session_state = self.aggregation_state.pop(session_id, None)
        if session_state is None:
            return
        session_state["data"].clear()
        session_state["accum"] = None
        session_state["result"] = None
        session_state["participants"] = set()
        session_state["results_delivered"] = set()
        session_state["event"].clear()
        print(f"Dropped aggregation session {session_id}")

    def _reset_aggregation_session(self, session_id: int) -> None:
        """Clear a stuck/partial aggregation session so the next op can proceed."""
        session_state = self.aggregation_state[session_id]
        session_state["data"].clear()
        session_state["accum"] = None
        session_state["reduction_type"] = None
        session_state["total_samples"] = 0
        session_state["result"] = None
        session_state["participants"] = set()
        session_state["results_delivered"] = set()
        session_state["compress_result"] = False
        session_state["event"].clear()

    def _abandon_current_session(self) -> None:
        """Drop the active session and advance (partial MAX/SUM mix or failed submit)."""
        sid = int(self.current_aggregation_session)
        self._reset_aggregation_session(sid)
        self.current_aggregation_session = sid + 1
        print(f"Abandoned aggregation session {sid}; next session={self.current_aggregation_session}")

    @staticmethod
    def _coerce_tensor_dict(data: Any) -> Dict[str, Any]:
        if isinstance(data, dict):
            return data
        if torch.is_tensor(data):
            return {"tensor": data}
        return {"value": data}

    def _fallback_compute_to_cpu(self, session_state: dict[str, Any]) -> None:
        print(
            "[grpc_server] agg GPU OOM; SUM/compress falling back to CPU",
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._compute_device = torch.device("cpu")
        accum = session_state.get("accum")
        if accum is not None:
            session_state["accum"] = mapping_to_device(accum, "cpu")
        result = session_state.get("result")
        if result is not None:
            session_state["result"] = mapping_to_device(result, "cpu")

    def _to_compute_device(
        self, payload: Dict[str, Any], session_state: dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return mapping_to_device(payload, self._compute_device)
        except Exception as exc:
            if self._compute_device.type == "cuda" and is_cuda_oom(exc):
                self._fallback_compute_to_cpu(session_state)
                return mapping_to_device(payload, self._compute_device)
            raise

    def _accumulate_into_session(
        self,
        session_state: dict[str, Any],
        participant_id: str,
        data: Any,
        num_samples: int = 0,
    ) -> None:
        """Running SUM/MEAN/MAX: keep one accumulator, drop this participant's copy."""
        if participant_id in session_state["participants"]:
            return
        payload = self._to_compute_device(self._coerce_tensor_dict(data), session_state)
        reduction = session_state.get("reduction_type")
        ns = int(num_samples)
        if (
            ns > 0
            and reduction != AggregationOp.MAX.value
            and reduction != AggregationOp.MEAN.value
        ):
            with torch.no_grad():
                payload = {
                    key: (tensor * ns if torch.is_tensor(tensor) else tensor)
                    for key, tensor in payload.items()
                }
        accum = session_state.get("accum")
        with torch.no_grad():
            if accum is None:
                session_state["accum"] = {
                    key: (
                        tensor.detach().clone()
                        if torch.is_tensor(tensor)
                        else tensor
                    )
                    for key, tensor in payload.items()
                }
            elif reduction == AggregationOp.MAX.value:
                for key, tensor in payload.items():
                    if key in session_state["accum"] and torch.is_tensor(tensor):
                        torch.maximum(
                            session_state["accum"][key],
                            tensor,
                            out=session_state["accum"][key],
                        )
            else:
                for key, tensor in payload.items():
                    if key in session_state["accum"] and torch.is_tensor(tensor):
                        session_state["accum"][key] += tensor
        session_state["participants"].add(participant_id)

    def _submitted_count(self, session_state: dict[str, Any]) -> int:
        n = len(session_state.get("participants") or ())
        if n:
            return n
        return len(session_state.get("data") or ())

    def get_broadcast_state(self) -> Dict[str, torch.Tensor]:
        """
        Get current broadcast state with thread-safe tensor cloning.

        Returns:
            Deep copy of broadcast tensors to prevent race conditions
        """
        with self.lock:
            return {
                key: tensor.clone() for key, tensor in self._broadcast_state.items()
            }

    def set_broadcast_state(self, tensordict: Dict[str, torch.Tensor]):
        """
        Store broadcast state for distribution to clients.

        Args:
            tensordict: Tensors to broadcast (typically global model)
        """
        with self.lock:
            self._broadcast_state = tensordict
            self.model = tensordict
        print(get_msg_info(tensordict))

    def set_next_result_processor(self, processor) -> int:
        """
        Register a result transformation for the next aggregation.

        Called by the server's worker thread before it submits its
        contribution. Aggregation cannot finish without that contribution.
        """
        with self.lock:
            session_id = self.current_aggregation_session
            session = self.aggregation_state[session_id]

            if session["result"] is not None:
                raise RuntimeError(
                    f"Aggregation session {session_id} already completed"
                )

            if session_id in self.result_processors:
                raise RuntimeError(
                    f"Aggregation session {session_id} already has a processor"
                )

            self.result_processors[session_id] = processor
            return session_id

    def perform_aggregation_if_ready(
        self,
        session_state: Dict,
        current_session: int,
        is_model_communicated=False,
    ) -> bool:
        """
        Execute aggregation when all clients have submitted data.
        SUM/MEAN/MAX use a running accumulator. Tests that still fill
        ``session_state["data"]`` are folded into that accumulator here.
        """

        # Never reapply an outer update to an already completed session.
        if session_state["result"] is not None:
            return True

        if session_state.get("accum") is None and session_state.get("data"):
            for pid, payload in list(session_state["data"].items()):
                self._accumulate_into_session(session_state, pid, payload)
            session_state["data"].clear()

        submitted_count = self._submitted_count(session_state)
        print(
            f"Waiting for clients "
            f"({submitted_count}/{self.world_size} ready)"
        )

        if submitted_count != self.world_size:
            return False

        if session_state.get("accum") is None:
            return False

        print(
            f"All {self.world_size} clients ready - beginning aggregation"
        )

        reduction_type = session_state["reduction_type"]
        if reduction_type is None:
            raise ValueError(
                f"No reduction type set for session {current_session}"
            )

        aggregated_tensors = session_state["accum"]

        with torch.no_grad():
            if reduction_type == AggregationOp.MEAN.value:
                for tensor in aggregated_tensors.values():
                    if torch.is_tensor(tensor):
                        tensor /= self.world_size

            elif reduction_type == AggregationOp.SUM.value:
                total_samples = int(session_state.get("total_samples", 0))
                if total_samples >= 1:
                    for tensor in aggregated_tensors.values():
                        if torch.is_tensor(tensor):
                            tensor /= total_samples

            elif reduction_type not in (
                AggregationOp.SUM.value,
                AggregationOp.MEAN.value,
                AggregationOp.MAX.value,
            ):
                raise ValueError(
                    f"Unknown reduction type: {reduction_type}"
                )

        # For Torchtitan algorithms (Diloco, FedAvg), apply any registered result processor to the aggregated tensors.
        processor = self.result_processors.get(current_session)
        if processor is not None:
            with torch.no_grad():
                aggregated_tensors = processor(aggregated_tensors)
            self.result_processors.pop(current_session, None)

        if is_model_communicated:
            self.model = aggregated_tensors

        session_state["result"] = aggregated_tensors
        self._release_aggregation_inputs(session_state)
        session_state["event"].set()

        print(
            f"Aggregated {len(aggregated_tensors)} tensors "
            f"using {reduction_type}"
        )

        self.current_aggregation_session += 1
        return True

    def _caller(self, context) -> str:
        """Participant id of the authenticated caller; aborts the RPC if it is not allowed.

        The identity is the CN of the verified TLS client certificate, looked up in
        ``identities`` -- never a field of the request. Every RPC calls this first, before
        its own try/except, so an abort cannot be swallowed by ``except Exception``.
        """
        participant = self._policy.resolve(context)
        security.enforce_deadline(context, self.require_client_deadline)
        return participant

    def _wait_for_result(self, session_state: dict, context, budget: float) -> bool:
        """Wait up to ``budget`` s for the session's result; True if it is there.

        Wakes in 0.5 s slices so a caller that hung up (cancelled / deadline) releases its
        worker thread promptly instead of pinning it until the aggregation finishes.
        """
        end = time.monotonic() + max(0.0, budget)
        while True:
            left = end - time.monotonic()
            if left <= 0:
                return session_state["event"].is_set()
            if session_state["event"].wait(min(0.5, left)):
                return True
            if not context.is_active():
                return False

    def _respond_with_result(self, client_id: str, target_session: int, session_state: dict):
        """Unary result response for ``client_id`` (marks it delivered); None if no result yet."""
        with self.lock:
            if session_state["result"] is None:
                return None
            print(f"Sending aggregated model to client {client_id}")
            response = self._create_aggregation_result_response(target_session)
            if int(response.n_chunks) <= 1:
                self._mark_aggregation_result_delivered_locked(target_session, client_id)
            return response

    def GetBroadcastState(self, request, context):
        """
        gRPC endpoint: Send broadcast state to requesting client.

        Args:
            request: CallerRequest (carries nothing; identity comes from the certificate)
            context: gRPC context; context.auth_context() identifies the caller

        Returns:
            OperationResponse with tensor data or not-ready status
        """
        client_id = self._caller(context)
        print(f"GetBroadcastState | client={client_id}")

        with self.lock:
            if self._broadcast_state:
                if should_chunk_payload(
                    self._broadcast_state, self.chunk_threshold_bytes
                ):
                    n_chunks = max(
                        chunk_count(
                            self._broadcast_state, self.chunk_payload_bytes
                        ),
                        2,
                    )
                    print(
                        f"Broadcast packed estimate needs {n_chunks} chunk(s) "
                        f"for client {client_id}"
                    )
                    return pb2.OperationResponse(
                        is_ready=True, n_chunks=n_chunks
                    )
                proto_tensordict = _to_proto(self._broadcast_state)
                return pb2.OperationResponse(
                    tensor_dict=proto_tensordict, is_ready=True, n_chunks=1
                )
            return pb2.OperationResponse(is_ready=False)

    def GetBroadcastStateStream(self, request, context):
        """Stream packed broadcast chunks for one logical GetBroadcastState."""
        client_id = self._caller(context)
        with self.lock:
            state = self._broadcast_state
        if not state:
            yield pb2.OperationResponse(is_ready=False)
            return
        try:
            chunks = list(
                iter_payload_chunks(state, self.chunk_payload_bytes)
            )
            n_chunks = len(chunks)
            print(
                f"Streaming {n_chunks} broadcast chunk(s) to client {client_id}"
            )
            for index, chunk in enumerate(chunks):
                yield pb2.OperationResponse(
                    tensor_dict=_to_proto(chunk),
                    is_ready=True,
                    chunk_index=index,
                    n_chunks=n_chunks,
                )
        except Exception as e:
            warnings.warn(
                f"Failed to stream broadcast state for client {client_id} | {e}",
                RuntimeWarning,
            )
            yield pb2.OperationResponse(is_ready=False)

    def _create_aggregation_result_response(self, session_id: int):
        """
        Create gRPC response containing aggregated tensor results.

        Args:
            session_id: Aggregation session identifier

        Returns:
            OperationResponse with aggregated tensors or not-ready status
        """
        if session_id in self.aggregation_state:
            session_state = self.aggregation_state[session_id]
            if session_state["result"] is not None:
                compressed_tensordict, compressor_name = (
                    self._compressed_aggregation_result(session_state)
                )
                if should_chunk_payload(
                    compressed_tensordict, self.chunk_threshold_bytes
                ):
                    n_chunks = sum(
                        1
                        for _ in iter_payload_chunks(
                            compressed_tensordict, self.chunk_payload_bytes
                        )
                    )
                    return pb2.OperationResponse(
                        is_ready=True, n_chunks=max(n_chunks, 2)
                    )
                proto_tensordict = _to_proto(
                    compressed_tensordict, compressor_name
                )
                return pb2.OperationResponse(
                    tensor_dict=proto_tensordict, is_ready=True, n_chunks=1
                )
        return pb2.OperationResponse(is_ready=False)

    def _compressed_aggregation_result(self, session_state: dict[str, Any]):
        aggregated_tensors = session_state["result"]
        active_compressor = (
            self.compressor if session_state.get("compress_result") else None
        )
        try:
            compressed_tensordict = compress_message_tensors(
                aggregated_tensors,
                active_compressor,
                self.aggregation_metric,
            )
        except Exception as exc:
            if self._compute_device.type == "cuda" and is_cuda_oom(exc):
                self._fallback_compute_to_cpu(session_state)
                aggregated_tensors = session_state["result"]
                compressed_tensordict = compress_message_tensors(
                    aggregated_tensors,
                    active_compressor,
                    self.aggregation_metric,
                )
            else:
                raise
        compressor_name = compressor_proto_name(active_compressor)
        if isinstance(aggregated_tensors, torch.Tensor):
            compressor_name = None
            compressed_tensordict = aggregated_tensors
        return compressed_tensordict, compressor_name

    def _ingest_client_payload(
        self,
        client_id: str,
        data: Any,
        reduction_type: str,
        num_samples: int,
        compressed_in: bool,
        is_model_communicated: bool,
    ) -> None:
        """Lock must be held. One assembled payload per client per session."""
        current_session = self.current_aggregation_session
        session_state = self.aggregation_state[current_session]
        if compressed_in:
            session_state["compress_result"] = True

        if (
            session_state["reduction_type"] is not None
            and session_state["reduction_type"] != reduction_type
        ):
            if self._submitted_count(session_state) < self.world_size:
                print(
                    f"Partial session {current_session} reduction mismatch "
                    f"(expected={session_state['reduction_type']} "
                    f"got={reduction_type}) — resetting session"
                )
                self._reset_aggregation_session(current_session)
                session_state = self.aggregation_state[current_session]
            else:
                raise ValueError(
                    f"Reduction mismatch | expected={session_state['reduction_type']} "
                    f"got={reduction_type}"
                )

        if session_state["reduction_type"] is None:
            session_state["reduction_type"] = reduction_type

        self._accumulate_into_session(
            session_state,
            client_id,
            data,
            num_samples=int(num_samples),
        )
        session_state["total_samples"] = int(
            session_state.get("total_samples", 0)
        ) + int(num_samples)
        data_count = self._submitted_count(session_state)
        print(
            f"Received from client {client_id} ({data_count}/{self.world_size} ready)"
        )
        self.perform_aggregation_if_ready(
            session_state, current_session, is_model_communicated
        )

    def SubmitForAggregation(self, request, context):
        """
        gRPC endpoint: Receive client tensors for distributed aggregation.

        Stores client contributions and triggers aggregation when all
        clients have submitted their data.

        Args:
            request: AggregationRequest with client data and reduction type
            context: gRPC context; context.auth_context() identifies the caller

        Returns:
            StatusResponse indicating success or failure
        """
        # Authenticate before the try-block: an abort must not be swallowed by `except Exception`.
        client_id = self._caller(context)
        with self.lock:
            try:
                data, is_model_communicated, compressed_in = self._decode_tensor_dict(
                    request.tensor_dict
                )
                self._ingest_client_payload(
                    client_id,
                    data,
                    request.reduction_type,
                    int(request.num_samples),
                    compressed_in,
                    is_model_communicated,
                )
                return pb2.StatusResponse(success=True)
            except Exception as e:
                print(f"Error is: {e}")
                traceback.print_exc()
                warnings.warn(
                    f"Failed to process aggregation submission from client {client_id} | {e}",
                    RuntimeWarning,
                )
                try:
                    session_state = self.aggregation_state[
                        self.current_aggregation_session
                    ]
                    if self._submitted_count(session_state) < self.world_size:
                        self._abandon_current_session()
                except Exception:
                    pass
                return pb2.StatusResponse(success=False)

    def _decode_tensor_dict(self, tensor_dict):
        data, is_model_communicated = proto_to_tensordict_extended(
            tensor_dict,
            overlay_base=None,
            compute_device=self._compute_device,
        )
        compressed_in = any(
            bool(getattr(entry, "compression_type", "") or "")
            for entry in tensor_dict.entries
        )
        return data, is_model_communicated, compressed_in

    def SubmitAggregationStream(self, request_iterator, context):
        """One logical submit: assemble chunks, then the same SUM as unary."""
        client_id = self._caller(context)
        reduction_type = None
        num_samples = 0
        decoded_chunks: list[dict] = []
        compressed_in = False
        is_model_communicated = False
        try:
            for request in request_iterator:
                if not decoded_chunks:
                    reduction_type = request.reduction_type
                    num_samples = int(request.num_samples)
                data, model_flag, chunk_compressed = self._decode_tensor_dict(
                    request.tensor_dict
                )
                decoded_chunks.append(data)
                compressed_in = compressed_in or chunk_compressed
                is_model_communicated = is_model_communicated or model_flag
            if not decoded_chunks or client_id is None:
                return pb2.StatusResponse(success=False)
            assembled = assemble_payload_chunks(decoded_chunks)
            print(
                f"Assembled {len(decoded_chunks)} uplink chunks from client {client_id}"
            )
            with self.lock:
                self._ingest_client_payload(
                    client_id,
                    assembled,
                    reduction_type or "",
                    num_samples,
                    compressed_in,
                    is_model_communicated,
                )
            return pb2.StatusResponse(success=True)
        except Exception as e:
            print(f"Error is: {e}")
            traceback.print_exc()
            warnings.warn(
                f"Failed to process streamed aggregation from client {client_id} | {e}",
                RuntimeWarning,
            )
            with self.lock:
                try:
                    session_state = self.aggregation_state[
                        self.current_aggregation_session
                    ]
                    if self._submitted_count(session_state) < self.world_size:
                        self._abandon_current_session()
                except Exception:
                    pass
            return pb2.StatusResponse(success=False)

    def GetAggregationResult(self, request, context):
        """
        gRPC endpoint: Send aggregation result to requesting client.

        Long-polls: waits up to ``long_poll_sec`` (and never past the client's own deadline)
        for the aggregation to finish, then returns the aggregated tensors or
        ``is_ready=False``, in which case the client simply asks again. This bounds how long
        one caller can hold a worker thread.

        Args:
            request: CallerRequest (carries nothing; identity comes from the certificate)
            context: gRPC context; context.auth_context() identifies the caller

        Returns:
            OperationResponse with aggregated tensors or not-ready status
        """
        client_id = self._caller(context)

        with self.lock:
            target_session = self._find_undelivered_session(client_id)
            if target_session is None:
                warnings.warn(
                    f"Client {client_id} has no data submitted for aggregation",
                    RuntimeWarning,
                )
                return pb2.OperationResponse(is_ready=False)

        print(f"Client {client_id} requesting aggregation result")

        try:
            session_state = self.aggregation_state[target_session]
            response = self._respond_with_result(client_id, target_session, session_state)
            if response is not None:
                return response

            budget = security.wait_budget(context, self.long_poll_sec)
            print(f"Client {client_id} waiting up to {budget:.0f}s for aggregation to complete")
            if self._wait_for_result(session_state, context, budget):
                print(f"Aggregation complete for client {client_id}")
                response = self._respond_with_result(client_id, target_session, session_state)
                if response is not None:
                    return response
            return pb2.OperationResponse(is_ready=False)

        except Exception as e:
            warnings.warn(
                f"Failed to get aggregation result for client {client_id} | {e}",
                RuntimeWarning,
            )
            return pb2.OperationResponse(is_ready=False)

    def _find_undelivered_session(self, client_id: str) -> int | None:
        for session_id in sorted(self.aggregation_state.keys(), reverse=True):
            session_state = self.aggregation_state[session_id]
            if not self._session_has_participant(session_state, client_id):
                continue
            if client_id in session_state.get("results_delivered", set()):
                continue
            return session_id
        return None

    def GetAggregationResultStream(self, request, context):
        """Stream packed result chunks for one logical Get."""
        client_id = self._caller(context)
        with self.lock:
            target_session = self._find_undelivered_session(client_id)
        if target_session is None:
            yield pb2.OperationResponse(is_ready=False)
            return

        session_state = self.aggregation_state[target_session]
        if session_state["result"] is None:
            budget = security.wait_budget(context, self.long_poll_sec)
            if not self._wait_for_result(session_state, context, budget):
                yield pb2.OperationResponse(is_ready=False)
                return

        try:
            with self.lock:
                if session_state["result"] is None:
                    packed = None
                else:
                    compressed_tensordict, compressor_name = (
                        self._compressed_aggregation_result(session_state)
                    )
                    chunks = list(
                        iter_payload_chunks(
                            compressed_tensordict, self.chunk_payload_bytes
                        )
                    )
                    packed = [
                        _to_proto(chunk, compressor_name) for chunk in chunks
                    ]
            if packed is None:
                yield pb2.OperationResponse(is_ready=False)
                return
            n_chunks = len(packed)
            print(
                f"Streaming {n_chunks} result chunks to client {client_id}"
            )
            for index, proto_tensordict in enumerate(packed):
                yield pb2.OperationResponse(
                    tensor_dict=proto_tensordict,
                    is_ready=True,
                    chunk_index=index,
                    n_chunks=n_chunks,
                )
            with self.lock:
                self._mark_aggregation_result_delivered_locked(
                    target_session, client_id
                )
        except Exception as e:
            warnings.warn(
                f"Failed to stream aggregation result for client {client_id} | {e}",
                RuntimeWarning,
            )
            yield pb2.OperationResponse(is_ready=False)

    def RegisterClient(self, request, context):
        """
        gRPC endpoint: Register client connection and track participant count.

        The participant id is the one the server derived from the client certificate and is
        returned so the client can confirm it is registered as the rank it thinks it is.

        Args:
            request: CallerRequest (carries nothing; identity comes from the certificate)
            context: gRPC context; context.auth_context() identifies the caller

        Returns:
            RegisterResponse with success and the derived participant id
        """
        client_id = self._caller(context)
        with self.lock:
            self.registered_clients.add(client_id)
            total_clients = len(self.registered_clients)
            print(f"{total_clients}/{self.world_size} total | registered {client_id} from {context.peer()}")
        return pb2.RegisterResponse(success=True, participant_id=client_id)
