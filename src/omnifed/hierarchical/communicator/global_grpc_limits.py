# Centralized gRPC max message sizes for hybrid global GrpcCommunicator / GrpcClient.
#
# Former default (100 MiB) fits small CNN payloads; Llama-class SendUpdate protobufs
# carry full fp32 tensors and exceed 100 MiB (e.g. ~860 MiB for 150M-class LMs).
#
# grpcio passes these ChannelArgs through a signed 32-bit conversion in the C extension.
# Exactly 2 * 1024**3 overflows; stay at signed INT32_MAX (~2 GiB - 1 B).

GRPC_MAX_MESSAGE_BYTES = 2147483647
