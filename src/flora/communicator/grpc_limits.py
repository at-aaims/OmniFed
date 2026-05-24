# Centralized gRPC max message sizes for Flora ``GrpcCommunicator`` / ``GrpcClient``.
#
# Former default (100 MiB) fits small CNN payloads; Llama-class ``SendUpdate``
# protobufs carry full fp32 tensors and exceed 100 MiB (e.g. ~860 MiB for 150 M-class LMs).


GRPC_MAX_MESSAGE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB
