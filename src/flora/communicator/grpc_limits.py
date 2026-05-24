# Centralized gRPC max message sizes for Flora ``GrpcCommunicator`` / ``GrpcClient``.
#
# Former default (100 MiB) fits small CNN payloads; Llama-class ``SendUpdate``
# protobufs carry full fp32 tensors and exceed 100 MiB (e.g. ~860 MiB for 150 M-class LMs).
#
# grpcio passes these ``ChannelArgs`` through a signed 32-bit conversion in the C extension.
# Exactly ``2 * 1024**3`` (2147483648) overflows and raises ``OverflowError`` when creating the
# server/channel (Frontier job ``4646079``). Stay at signed ``INT32_MAX`` (~2 GiB − 1 B).


GRPC_MAX_MESSAGE_BYTES = 2147483647

