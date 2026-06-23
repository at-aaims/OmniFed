"""Unit tests for hybrid global TopK compression and gRPC layer encode/decode."""

import numpy as np
import torch

import src.omnifed.hybrid.communicator.global_grpc_pb2 as global_grpc_pb2
from src.omnifed.hybrid.compression.topk import TOPK_COMPRESSION_NAME, TopKCompression
from src.omnifed.hybrid.communicator.global_grpc_compression import (
    decode_layer_tensor,
    encode_layer_state,
)


def test_topk_roundtrip_with_error_feedback():
    compressor = TopKCompression(device="cpu", compress_ratio=0.25)
    x = torch.randn(32)
    (values, indices), ctx = compressor.compress(x.clone(), name="layer0")
    assert values.numel() == max(1, int(32 * 0.25))
    restored = compressor.decompress((values, indices), ctx)
    assert restored.shape == x.shape


def test_layer_state_sparse_encode_decode_overlay():
    compressor = TopKCompression(device="cpu", compress_ratio=0.1)
    base = torch.randn(4, 4)
    layer = encode_layer_state("conv.weight", base, compressor)
    assert layer.compression_type == TOPK_COMPRESSION_NAME
    assert len(layer.param_update) == 0
    assert layer.values_data
    assert layer.indices_data

    decoded = decode_layer_tensor(layer, base_tensor=base)
    assert decoded.shape == base.shape
    flat_base = base.reshape(-1)
    flat_dec = decoded.reshape(-1)
    mask = torch.ones_like(flat_base, dtype=torch.bool)
    indices = np.frombuffer(layer.indices_data, dtype=np.int64)
    mask[indices] = False
    assert torch.allclose(flat_dec[mask], flat_base[mask])


def test_layer_state_dense_legacy_path():
    t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    layer = encode_layer_state("dense", t, None)
    assert layer.compression_type == ""
    out = decode_layer_tensor(layer)
    assert torch.allclose(out, t)
