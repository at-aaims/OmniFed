# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.

"""Classic gRPC channel limits (centralized ``GrpcCommunicator``).

Not model-specific: one unary RPC cap for CIFAR and Llama. grpcio ChannelArgs
are signed int32, so exactly ``2 * 1024**3`` overflows; use INT32_MAX.
Timeouts are wait budgets for one aggregation, not per-architecture knobs.
CIFAR finishes in seconds; the long default only matters on hang or Llama-scale
payloads (~1.6 GiB fp32 for Llama-400M).
"""

# Signed INT32_MAX (~2 GiB - 1 B). Same value as hybrid ``global_grpc_limits``.
GRPC_MAX_MESSAGE_BYTES = 2147483647

# Unary vs streamed chunks: compare packed-size *estimate* to this, not model name.
# 0.9 * INT32_MAX (~1.93 GiB) keeps Llama-400M dense (~1.6 GiB) on one unary.
GRPC_CHUNK_THRESHOLD_BYTES = int(GRPC_MAX_MESSAGE_BYTES * 0.9)

# Each streamed message body stays further under the cap (protobuf overhead).
GRPC_CHUNK_PAYLOAD_BYTES = 1610612736  # 1.5 GiB

# Match hybrid Llama-400M ``server_sec_per_round``: one classic agg can include
# 6× ~1.6 GiB uploads plus download. CIFAR yaml may still set a shorter override.
GRPC_AGGREGATION_TIMEOUT_SEC = 5000.0
GRPC_CLIENT_TIMEOUT_SEC = 5000.0
