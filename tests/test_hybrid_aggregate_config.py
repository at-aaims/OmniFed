"""Unit tests for hybrid aggregate_payload config helper."""

from __future__ import annotations

import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.omnifed.hybrid.hybrid_aggregate_config import (
    AGGREGATE_PAYLOAD_GRADIENTS,
    AGGREGATE_PAYLOAD_PARAMS,
    hybrid_aggregate_payload_from_cfg,
    hybrid_communicate_params_from_cfg,
    normalize_aggregate_payload,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestHybridAggregateConfig(unittest.TestCase):
    def test_normalize_defaults_and_aliases(self) -> None:
        self.assertEqual(normalize_aggregate_payload(None), AGGREGATE_PAYLOAD_PARAMS)
        self.assertEqual(normalize_aggregate_payload("params"), AGGREGATE_PAYLOAD_PARAMS)
        self.assertEqual(normalize_aggregate_payload("weights"), AGGREGATE_PAYLOAD_PARAMS)
        self.assertEqual(normalize_aggregate_payload("gradients"), AGGREGATE_PAYLOAD_GRADIENTS)
        self.assertEqual(normalize_aggregate_payload("grad"), AGGREGATE_PAYLOAD_GRADIENTS)

    def test_normalize_rejects_invalid(self) -> None:
        with self.assertRaises(ValueError):
            normalize_aggregate_payload("mixed")

    def test_compose_resnet_qsgd_default_is_params(self) -> None:
        conf_dir = str(REPO_ROOT / "conf")
        with initialize_config_dir(version_base=None, config_dir=conf_dir):
            cfg = compose(config_name="test_hybrid_layout_fedavg_cifar10_resnet18_grpc_qsgd")
        self.assertEqual(hybrid_aggregate_payload_from_cfg(cfg), AGGREGATE_PAYLOAD_PARAMS)
        self.assertTrue(hybrid_communicate_params_from_cfg(cfg))

    def test_compose_override_gradients(self) -> None:
        conf_dir = str(REPO_ROOT / "conf")
        with initialize_config_dir(version_base=None, config_dir=conf_dir):
            cfg = compose(
                config_name="test_hybrid_layout_fedavg_cifar10_resnet18_grpc_qsgd",
                overrides=["engine.hybrid.aggregate_payload=gradients"],
            )
        self.assertEqual(hybrid_aggregate_payload_from_cfg(cfg), AGGREGATE_PAYLOAD_GRADIENTS)
        self.assertFalse(hybrid_communicate_params_from_cfg(cfg))

    def test_base_yaml_has_knob(self) -> None:
        conf_dir = str(REPO_ROOT / "conf")
        with initialize_config_dir(version_base=None, config_dir=conf_dir):
            cfg = compose(config_name="base")
        self.assertEqual(
            OmegaConf.select(cfg, "engine.hybrid.aggregate_payload"),
            "params",
        )

    def test_communicate_params_tracks_aggregate_payload(self) -> None:
        conf_dir = str(REPO_ROOT / "conf")
        with initialize_config_dir(version_base=None, config_dir=conf_dir):
            params_cfg = compose(
                config_name="test_hybrid_layout_fedavg_cifar10_resnet18_grpc_qsgd",
            )
            grad_cfg = compose(
                config_name="test_hybrid_layout_fedavg_cifar10_resnet18_grpc_qsgd",
                overrides=["engine.hybrid.aggregate_payload=gradients"],
            )
        self.assertTrue(hybrid_communicate_params_from_cfg(params_cfg))
        self.assertFalse(hybrid_communicate_params_from_cfg(grad_cfg))


if __name__ == "__main__":
    unittest.main()
