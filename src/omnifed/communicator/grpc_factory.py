"""Select matching gRPC communicator and model-exchange implementations."""

from .grpc import GrpcCommunicator


def validate_transport_mode(transport_mode: str) -> str:
    if transport_mode not in {"legacy", "optimized"}:
        raise ValueError(
            "transport_mode must be 'legacy' or 'optimized'; "
            f"received {transport_mode!r}"
        )
    return transport_mode


def create_grpc_communicator(
    *,
    transport_mode: str = "legacy",
    **kwargs,
):
    # mode = validate_transport_mode(transport_mode)

    if transport_mode == "legacy":
        communicator = GrpcCommunicator(**kwargs)
    elif transport_mode == "optimized":
        raise NotImplementedError(
            "Optimized gRPC transport is not implemented yet."
        )
    else:
        raise ValueError(
            f"Unknown transport_mode: {transport_mode!r}"
        )

    communicator.transport_mode = transport_mode
    return communicator


def create_model_exchange(communicator, *, chunk_size_mb: int):
    mode = validate_transport_mode(
        communicator.transport_mode
    )

    if mode == "optimized":
        raise NotImplementedError(
            "Optimized model exchange is not implemented yet."
        )

    from .exchange.legacy import LegacyModelExchange

    return LegacyModelExchange(
        communicator,
        chunk_size_mb=chunk_size_mb,
    )