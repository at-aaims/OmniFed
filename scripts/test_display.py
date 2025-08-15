#!/usr/bin/env python3
"""
Test script for OmniFed experiment results display.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Add OmniFed root to path and import modules
script_dir = Path(__file__).parent
omnifed_root = script_dir.parent
sys.path.insert(0, str(omnifed_root))

from src.omnifed.utils import print  # noqa: E402
from src.omnifed.utils.results_display import ResultsDisplay  # noqa: E402
from src.omnifed.utils.rich_helpers import print_rule  # noqa: E402


def print_test_configuration(
    test_name: str, results: List[Any], total_nodes: int, rounds: int, duration: float
) -> None:
    """Print test configuration summary."""

    contexts: Set[str] = set()
    nodes_with_data: Dict[str, int] = {}
    sample_metrics: Dict[str, List[str]] = {}
    excluded_keys = {"global_step", "round_idx", "epoch_idx", "batch_idx"}
    total_measurements = 0

    for node_data in results:
        if not isinstance(node_data, dict):
            continue

        for context, context_data in node_data.items():
            contexts.add(context)
            if isinstance(context_data, list) and context_data:
                total_measurements += len(context_data)
                nodes_with_data[context] = nodes_with_data.get(context, 0) + 1
                if context not in sample_metrics:
                    sample_metrics[context] = [
                        k for k in context_data[0].keys() if k not in excluded_keys
                    ]

    config = {
        "test_name": test_name,
        "nodes": total_nodes,
        "rounds": rounds,
        "duration_s": duration,
        "contexts": sorted(contexts),
        "total_measurements": total_measurements,
        "nodes_with_data": {
            k: f"{v}/{total_nodes}" for k, v in nodes_with_data.items()
        },
        "sample_metrics": sample_metrics,
    }

    print(f"\n📋 TEST CONFIG: {test_name}")
    print(json.dumps(config, indent=2))


# Test Data Generation Constants
DEFAULTS = {
    "final_step": 3,
    "round": 0,
    "epoch": 1,
    "batch": 1,
    "duration": 10.0,
    "eval_num_batches": 10,
    "eval_num_samples": 1000,
    "train_num_batches": 20,
}

# Metric progression patterns
METRIC_PARAMS = {
    "eval": {
        "accuracy": {"base": 0.7, "progress": 0.25, "noise": 0.02},
        "precision": {"base": 0.65, "progress": 0.3, "noise": 0.02},
        "recall": {"base": 0.68, "progress": 0.27, "noise": 0.02},
        "f1": {"base": 0.66, "progress": 0.29, "noise": 0.02},
        "loss": {"base": 0.5, "noise": 0.05},  # Loss decreases with progress
        "batch_time": {"base": 0.008, "noise": 0.004},
    },
    "train": {
        "accuracy": {"base": 0.65, "progress": 0.28, "noise": 0.02},
        "loss": {"base": 0.6, "noise": 0.06},  # Loss decreases with progress
        "batch_time": {"base": 0.012, "noise": 0.006},
    },
}


def create_measurement(
    context: str = "eval",
    round_idx: Optional[int] = None,
    epoch_idx: Optional[int] = None,
    batch_idx: Optional[int] = None,
    global_step: Optional[int] = None,
    metrics: Optional[Dict[str, Any]] = None,
    **extra_metrics: Any,
) -> Dict[str, Any]:
    """Create measurement with standard structure."""
    measurement = {
        "round_idx": round_idx or DEFAULTS["round"],
        "epoch_idx": epoch_idx or DEFAULTS["epoch"],
        "batch_idx": batch_idx or DEFAULTS["batch"],
        "global_step": global_step or DEFAULTS["final_step"],
    }

    # Add context-specific defaults
    if context == "eval":
        measurement.update(
            {
                "eval/num_batches": DEFAULTS["eval_num_batches"],
                "eval/num_samples": DEFAULTS["eval_num_samples"],
            }
        )
    else:  # train
        measurement.update(
            {
                "train/num_batches": DEFAULTS["train_num_batches"],
            }
        )

    # Add the provided metrics
    if metrics:
        measurement.update(metrics)
    measurement.update(extra_metrics)
    return measurement


def create_node_structure(
    eval_measurements: Optional[Any] = None, train_measurements: Optional[Any] = None
) -> Dict[str, Any]:
    """Create node structure with eval/train contexts."""
    node = {}
    for context, measurements in [
        ("eval", eval_measurements),
        ("train", train_measurements),
    ]:
        if measurements is not None:
            node[context] = (
                measurements if isinstance(measurements, list) else [measurements]
            )
    return node


def find_and_load_experiment(experiment_path: str) -> List[Any]:
    """Find and load experiment data."""
    path = Path(experiment_path)

    # Search for experiment results
    found_dirs: Set[Path] = set()
    for pattern in [
        "*/*/engine/node_results/node_*_results.pkl",
        "*/*/node_results/node_*_results.pkl",
        "*/engine/node_results/node_*_results.pkl",
        "*/node_results/node_*_results.pkl",
    ]:
        found_dirs.update(pkl_file.parent for pkl_file in path.glob(pattern))

    if not found_dirs:
        raise FileNotFoundError(f"No experiment data found in {path}")

    # Use the most recent results directory if multiple found
    results_dir = max(found_dirs, key=lambda x: x.stat().st_mtime)
    if len(found_dirs) > 1:
        print(f"Found {len(found_dirs)} result directories, using: {results_dir}")

    pkl_files = sorted(
        results_dir.glob("node_*_results.pkl"), key=lambda x: int(x.stem.split("_")[1])
    )
    if not pkl_files:
        raise FileNotFoundError(f"No node result files in {results_dir}")

    results = []
    for pkl_file in pkl_files:
        with open(pkl_file, "rb") as f:
            results.append(pickle.load(f))

    if not results:
        raise FileNotFoundError(
            f"No valid result files could be loaded from {results_dir}"
        )

    print(f"Loaded {len(results)} node files")
    return results


def generate_metric_value(metric_name: str, context: str, progress: float) -> float:
    """Generate metric value with progression and noise."""
    if context not in METRIC_PARAMS:
        raise ValueError(
            f"Unknown context '{context}'. Available: {list(METRIC_PARAMS.keys())}"
        )
    if metric_name not in METRIC_PARAMS[context]:
        raise ValueError(
            f"Unknown metric '{metric_name}' for context '{context}'. Available: {list(METRIC_PARAMS[context].keys())}"
        )
    if not (0.0 <= progress <= 1.0):
        raise ValueError(f"Progress must be in [0.0, 1.0], got {progress}")

    params = METRIC_PARAMS[context][metric_name]
    value = (
        params["base"] + params["progress"] * progress
        if "progress" in params
        else params["base"] * (1 - progress)
    )
    return value + params["noise"] * np.random.randn()


def generate_progression_data(
    context: str, rounds: int, epochs: int, batches: int
) -> List[Dict[str, Any]]:
    """Generate progression data for a node."""
    if rounds < 1 or epochs < 1 or batches < 1:
        raise ValueError("All progression parameters must be positive integers")

    data, global_step = [], 0
    total_steps = rounds * epochs * batches

    for round_idx in range(rounds):
        for epoch_idx in range(epochs):
            for batch_idx in range(batches):
                measurement = create_measurement(
                    context=context,
                    round_idx=round_idx,
                    epoch_idx=epoch_idx,
                    batch_idx=batch_idx,
                    global_step=global_step,
                )

                # Add context-specific metrics
                progress = global_step / total_steps
                measurement.update(
                    {
                        f"{context}/{metric}": generate_metric_value(
                            metric, context, progress
                        )
                        for metric in METRIC_PARAMS[context]
                    }
                )

                data.append(measurement)
                global_step += 1

    return data


def generate_test_data(
    rounds: int = 2, epochs: int = 3, batches: int = 2, num_nodes: int = 4
) -> List[Dict[str, Any]]:
    """Generate clean test data with normal progression nodes only."""
    if rounds < 1 or epochs < 1 or batches < 1:
        raise ValueError("All parameters must be positive")
    if num_nodes < 1:
        raise ValueError("Number of nodes must be positive")

    return [
        {
            "eval": generate_progression_data("eval", rounds, epochs, batches),
            "train": generate_progression_data("train", rounds, epochs, batches),
        }
        for _ in range(num_nodes)
    ]


def generate_single_node_test() -> List[Dict[str, Any]]:
    """Single node test with progression over time."""
    return [
        create_node_structure(
            eval_measurements=[
                create_measurement(
                    context="eval",
                    round_idx=0,
                    epoch_idx=0,
                    batch_idx=0,
                    global_step=0,
                    metrics={
                        "eval/accuracy": 0.75,
                        "eval/precision": 0.73,
                        "eval/recall": 0.77,
                        "eval/f1": 0.75,
                        "eval/loss": 0.35,
                        "eval/batch_time": 0.012,
                    },
                ),
                create_measurement(
                    context="eval",
                    round_idx=1,
                    epoch_idx=0,
                    batch_idx=0,
                    global_step=1,
                    metrics={
                        "eval/accuracy": 0.88,
                        "eval/precision": 0.86,
                        "eval/recall": 0.90,
                        "eval/f1": 0.88,
                        "eval/loss": 0.22,
                        "eval/batch_time": 0.008,
                    },
                ),
            ],
            train_measurements=[
                create_measurement(
                    context="train",
                    round_idx=0,
                    epoch_idx=0,
                    batch_idx=0,
                    global_step=0,
                    metrics={
                        "train/accuracy": 0.72,
                        "train/loss": 0.38,
                        "train/batch_time": 0.015,
                    },
                ),
                create_measurement(
                    context="train",
                    round_idx=1,
                    epoch_idx=0,
                    batch_idx=0,
                    global_step=1,
                    metrics={
                        "train/accuracy": 0.85,
                        "train/loss": 0.28,
                        "train/batch_time": 0.012,
                    },
                ),
            ],
        )
    ]


def generate_duplicate_measurements_test() -> List[Dict[str, Any]]:
    """Test duplicate measurements at the same step.

    Three nodes:
    - Node 0: Single measurement
    - Node 1: Multiple measurements at same step (3 eval, 2 train)
    - Node 2: Single measurement
    """
    node_specs = [
        # Node 0: Control node with single measurements
        {
            "eval_measurements": create_measurement(
                context="eval", metrics={"eval/accuracy": 0.85, "eval/loss": 0.25}
            ),
            "train_measurements": create_measurement(
                context="train",
                metrics={"train/accuracy": 0.80, "train/loss": 0.30},
            ),
        },
        # Node 1: Multiple measurements per step
        {
            "eval_measurements": [
                create_measurement(
                    context="eval", metrics={"eval/accuracy": 0.70, "eval/loss": 0.45}
                ),
                create_measurement(
                    context="eval", metrics={"eval/accuracy": 0.88, "eval/loss": 0.18}
                ),
                create_measurement(
                    context="eval", metrics={"eval/accuracy": 0.92, "eval/loss": 0.12}
                ),
            ],
            "train_measurements": [
                create_measurement(
                    context="train",
                    metrics={"train/accuracy": 0.75, "train/loss": 0.40},
                ),
                create_measurement(
                    context="train",
                    metrics={"train/accuracy": 0.89, "train/loss": 0.18},
                ),
            ],
        },
        # Node 2: Control node with single measurements
        {
            "eval_measurements": create_measurement(
                context="eval", metrics={"eval/accuracy": 0.90, "eval/loss": 0.15}
            ),
            "train_measurements": create_measurement(
                context="train",
                metrics={"train/accuracy": 0.87, "train/loss": 0.22},
            ),
        },
    ]
    return [create_node_structure(**spec) for spec in node_specs]


def generate_invalid_values_test() -> List[Dict[str, Any]]:
    """Nodes with invalid values (NaN, Inf)."""
    node_specs = [
        # Pure NaN corruption
        {
            "eval_measurements": create_measurement(
                context="eval",
                metrics={
                    "eval/accuracy": float("nan"),
                    "eval/precision": 0.85,
                    "eval/recall": 0.83,
                    "eval/loss": 0.25,
                },
            ),
            "train_measurements": create_measurement(
                context="train",
                metrics={
                    "train/accuracy": 0.80,
                    "train/loss": float("nan"),
                },
            ),
        },
        # Pure Inf corruption
        {
            "eval_measurements": create_measurement(
                context="eval",
                metrics={
                    "eval/accuracy": 0.87,
                    "eval/precision": float("inf"),
                    "eval/loss": 0.22,
                },
            ),
            "train_measurements": create_measurement(
                context="train",
                metrics={
                    "train/accuracy": 0.82,
                    "train/loss": float("inf"),
                },
            ),
        },
        # Negative Inf corruption
        {
            "eval_measurements": create_measurement(
                context="eval",
                metrics={
                    "eval/accuracy": 0.89,
                    "eval/precision": 0.88,
                    "eval/loss": float("-inf"),
                },
            ),
            "train_measurements": create_measurement(
                context="train",
                metrics={
                    "train/accuracy": 0.85,
                    "train/loss": float("-inf"),
                },
            ),
        },
    ]
    return [create_node_structure(**spec) for spec in node_specs]


def generate_out_of_range_test() -> List[Dict[str, Any]]:
    """Nodes with out-of-range values."""
    node_specs = [
        # Accuracy/precision/recall > 1.0
        {
            "eval_measurements": create_measurement(
                context="eval",
                metrics={
                    "eval/accuracy": 1.15,  # > 1.0
                    "eval/precision": 1.05,  # > 1.0
                    "eval/recall": 0.95,
                    "eval/loss": 0.18,
                },
            ),
            "train_measurements": create_measurement(
                context="train",
                metrics={
                    "train/accuracy": 1.08,  # > 1.0
                    "train/loss": 0.22,
                },
            ),
        },
        # Negative values where they shouldn't be
        {
            "eval_measurements": create_measurement(
                context="eval",
                metrics={
                    "eval/accuracy": -0.2,  # < 0.0
                    "eval/precision": -0.1,  # < 0.0
                    "eval/batch_time": -0.001,  # < 0.0
                    "eval/loss": 0.25,
                },
            ),
            "train_measurements": create_measurement(
                context="train",
                metrics={
                    "train/accuracy": -0.05,  # < 0.0
                    "train/loss": -1.2,
                    "train/batch_time": 0.008,
                },
            ),
        },
    ]
    return [create_node_structure(**spec) for spec in node_specs]


def generate_wrong_types_test() -> List[Dict[str, Any]]:
    """Nodes with wrong data types."""
    node_specs = [
        {
            "eval_measurements": create_measurement(
                context="eval",
                metrics={
                    "eval/accuracy": 0.88,
                    "eval/loss": 0.15,
                    "eval/num_batches": 10.5,  # Float instead of int
                    "eval/num_samples": 1000.75,  # Float instead of int
                },
            ),
            "train_measurements": create_measurement(
                context="train",
                metrics={
                    "train/accuracy": 0.85,
                    "train/loss": 0.20,
                    "train/num_batches": 20.3,  # Float instead of int
                },
            ),
        },
    ]
    return [create_node_structure(**spec) for spec in node_specs]


def generate_extreme_values_test() -> List[Dict[str, Any]]:
    """Nodes with extreme values."""
    node_specs = [
        # Extremely high values
        {
            "eval_measurements": create_measurement(
                context="eval",
                metrics={
                    "eval/accuracy": 0.92,
                    "eval/loss": 5000.0,  # Extremely high loss
                    "eval/batch_time": 300.0,  # 5 minutes per batch
                },
            ),
            "train_measurements": create_measurement(
                context="train",
                metrics={
                    "train/accuracy": 0.90,
                    "train/loss": 1000.0,  # Extremely high loss
                    "train/batch_time": 120.0,  # 2 minutes per batch
                },
            ),
        },
        # Extremely low values
        {
            "eval_measurements": create_measurement(
                context="eval",
                metrics={
                    "eval/accuracy": 0.001,  # Near-zero accuracy
                    "eval/loss": 0.000001,  # Near-zero loss
                    "eval/batch_time": 0.000001,  # Microsecond timing
                },
            ),
            "train_measurements": create_measurement(
                context="train",
                metrics={
                    "train/accuracy": 0.002,  # Near-zero accuracy
                    "train/loss": 0.000001,  # Near-zero loss
                    "train/batch_time": 0.000001,  # Microsecond timing
                },
            ),
        },
    ]
    return [create_node_structure(**spec) for spec in node_specs]


def generate_bad_data_test() -> List[Dict[str, Any]]:
    """Various bad data scenarios.

    Combines different types of problematic data:
    - Invalid values (NaN/Inf)
    - Out-of-range values
    - Wrong data types
    - Extreme values
    """
    # Combine all bad data types
    all_nodes = []
    all_nodes.extend(generate_invalid_values_test())
    all_nodes.extend(generate_out_of_range_test())
    all_nodes.extend(generate_wrong_types_test())
    all_nodes.extend(generate_extreme_values_test())

    return all_nodes


def generate_variability_thresholds_test() -> List[Dict[str, Any]]:
    """Test variability threshold boundaries for color coding.

    Creates nodes with specific CV values to check threshold classification:
    - CV < 20% = Normal
    - CV 20-50% = Moderate
    - CV 50-100% = High
    - CV > 100% = Extreme
    """

    boundary_specs = [
        # Normal variability (CV ~15%)
        {
            "eval_measurements": [
                create_measurement(
                    context="eval", metrics={"eval/accuracy": 0.85, "eval/loss": 0.20}
                ),
                create_measurement(
                    context="eval", metrics={"eval/accuracy": 0.87, "eval/loss": 0.21}
                ),
                create_measurement(
                    context="eval", metrics={"eval/accuracy": 0.83, "eval/loss": 0.19}
                ),
            ],
            "train_measurements": [
                create_measurement(
                    context="train",
                    metrics={"train/accuracy": 0.80, "train/loss": 0.25},
                ),
                create_measurement(
                    context="train",
                    metrics={"train/accuracy": 0.82, "train/loss": 0.26},
                ),
                create_measurement(
                    context="train",
                    metrics={"train/accuracy": 0.78, "train/loss": 0.24},
                ),
            ],
        },
        # Moderate variability (CV ~25%)
        {
            "eval_measurements": [
                create_measurement(
                    context="eval", metrics={"eval/accuracy": 0.70, "eval/loss": 0.30}
                ),
                create_measurement(
                    context="eval", metrics={"eval/accuracy": 0.85, "eval/loss": 0.25}
                ),
                create_measurement(
                    context="eval", metrics={"eval/accuracy": 0.55, "eval/loss": 0.35}
                ),
            ],
            "train_measurements": [
                create_measurement(
                    context="train",
                    metrics={"train/accuracy": 0.65, "train/loss": 0.40},
                ),
                create_measurement(
                    context="train",
                    metrics={"train/accuracy": 0.80, "train/loss": 0.30},
                ),
                create_measurement(
                    context="train",
                    metrics={"train/accuracy": 0.50, "train/loss": 0.50},
                ),
            ],
        },
        # High variability (CV ~75%)
        {
            "eval_measurements": [
                create_measurement(
                    context="eval", metrics={"eval/accuracy": 0.50, "eval/loss": 0.50}
                ),
                create_measurement(
                    context="eval", metrics={"eval/accuracy": 0.80, "eval/loss": 0.20}
                ),
                create_measurement(
                    context="eval", metrics={"eval/accuracy": 0.20, "eval/loss": 0.80}
                ),
            ],
            "train_measurements": [
                create_measurement(
                    context="train",
                    metrics={"train/accuracy": 0.40, "train/loss": 0.60},
                ),
                create_measurement(
                    context="train",
                    metrics={"train/accuracy": 0.70, "train/loss": 0.30},
                ),
                create_measurement(
                    context="train",
                    metrics={"train/accuracy": 0.10, "train/loss": 0.90},
                ),
            ],
        },
        # Extreme variability (CV >150%)
        {
            "eval_measurements": [
                create_measurement(
                    context="eval", metrics={"eval/accuracy": 0.90, "eval/loss": 0.10}
                ),
                create_measurement(
                    context="eval", metrics={"eval/accuracy": 0.50, "eval/loss": 0.50}
                ),
                create_measurement(
                    context="eval", metrics={"eval/accuracy": 0.05, "eval/loss": 0.95}
                ),
            ],
            "train_measurements": [
                create_measurement(
                    context="train",
                    metrics={"train/accuracy": 0.85, "train/loss": 0.15},
                ),
                create_measurement(
                    context="train",
                    metrics={"train/accuracy": 0.40, "train/loss": 0.60},
                ),
                create_measurement(
                    context="train",
                    metrics={"train/accuracy": 0.01, "train/loss": 0.99},
                ),
            ],
        },
    ]

    return [create_node_structure(**spec) for spec in boundary_specs]


def generate_empty_data_test() -> List[Dict[str, Any]]:
    """Test missing/empty data scenarios.

    Various cases of missing data:
    - Empty nodes
    - Empty contexts
    - Missing eval or train data
    - Mixed empty and valid data
    """
    return [
        # Completely empty node (no data at all)
        {},
        # Node with empty contexts
        {
            "eval": [],
            "train": [],
        },
        # Node with only empty eval context
        {
            "eval": [],
        },
        # Node with only empty train context
        {
            "train": [],
        },
        # Eval-only node (missing train context entirely)
        create_node_structure(
            eval_measurements=create_measurement(
                context="eval",
                metrics={
                    "eval/accuracy": 0.85,
                    "eval/loss": 0.20,
                },
            )
        ),
        # Train-only node (missing eval context entirely)
        create_node_structure(
            train_measurements=create_measurement(
                context="train",
                metrics={
                    "train/accuracy": 0.80,
                    "train/loss": 0.25,
                },
            )
        ),
        # Normal node for comparison
        create_node_structure(
            eval_measurements=create_measurement(
                context="eval",
                metrics={
                    "eval/accuracy": 0.88,
                    "eval/precision": 0.87,
                    "eval/recall": 0.89,
                    "eval/loss": 0.18,
                },
            ),
            train_measurements=create_measurement(
                context="train",
                metrics={
                    "train/accuracy": 0.85,
                    "train/loss": 0.22,
                },
            ),
        ),
    ]


def generate_measurement_patterns_test() -> List[Dict[str, Any]]:
    """Test different measurement patterns.

    Different measurement patterns:
    - High density: many measurements, same final step
    - Low density: few measurements, same final step
    - Sparse irregular: scattered measurement points
    - Early stopping: lower final step but high density
    """
    return [
        # High-density node (many measurements, same final step)
        create_node_structure(
            eval_measurements=[
                create_measurement(
                    context="eval",
                    global_step=0,
                    metrics={"eval/accuracy": 0.70, "eval/loss": 0.50},
                ),
                create_measurement(
                    context="eval",
                    global_step=1,
                    metrics={"eval/accuracy": 0.75, "eval/loss": 0.45},
                ),
                create_measurement(
                    context="eval",
                    global_step=2,
                    metrics={"eval/accuracy": 0.80, "eval/loss": 0.40},
                ),
                create_measurement(
                    context="eval",
                    global_step=3,
                    metrics={"eval/accuracy": 0.85, "eval/loss": 0.35},
                ),
                create_measurement(
                    context="eval",
                    global_step=4,
                    metrics={"eval/accuracy": 0.90, "eval/loss": 0.30},
                ),
            ],
            train_measurements=[
                create_measurement(
                    context="train",
                    global_step=0,
                    metrics={"train/accuracy": 0.65, "train/loss": 0.55},
                ),
                create_measurement(
                    context="train",
                    global_step=1,
                    metrics={"train/accuracy": 0.70, "train/loss": 0.50},
                ),
                create_measurement(
                    context="train",
                    global_step=2,
                    metrics={"train/accuracy": 0.75, "train/loss": 0.45},
                ),
                create_measurement(
                    context="train",
                    global_step=3,
                    metrics={"train/accuracy": 0.80, "train/loss": 0.40},
                ),
                create_measurement(
                    context="train",
                    global_step=4,
                    metrics={"train/accuracy": 0.85, "train/loss": 0.35},
                ),
            ],
        ),
        # Low-density node (few measurements, same final step)
        create_node_structure(
            eval_measurements=[
                create_measurement(
                    context="eval",
                    global_step=0,
                    metrics={"eval/accuracy": 0.72, "eval/loss": 0.48},
                ),
                create_measurement(
                    context="eval",
                    global_step=4,
                    metrics={"eval/accuracy": 0.88, "eval/loss": 0.32},
                ),
            ],
            train_measurements=[
                create_measurement(
                    context="train",
                    global_step=0,
                    metrics={"train/accuracy": 0.68, "train/loss": 0.52},
                ),
                create_measurement(
                    context="train",
                    global_step=4,
                    metrics={"train/accuracy": 0.83, "train/loss": 0.37},
                ),
            ],
        ),
        # Sparse irregular node (measurements at scattered steps)
        create_node_structure(
            eval_measurements=[
                create_measurement(
                    context="eval",
                    global_step=1,
                    metrics={"eval/accuracy": 0.74, "eval/loss": 0.46},
                ),
                create_measurement(
                    context="eval",
                    global_step=3,
                    metrics={"eval/accuracy": 0.86, "eval/loss": 0.34},
                ),
            ],
            train_measurements=[
                create_measurement(
                    context="train",
                    global_step=0,
                    metrics={"train/accuracy": 0.69, "train/loss": 0.51},
                ),
                create_measurement(
                    context="train",
                    global_step=2,
                    metrics={"train/accuracy": 0.77, "train/loss": 0.43},
                ),
            ],
        ),
        # Single measurement node (minimal data)
        create_node_structure(
            eval_measurements=create_measurement(
                context="eval",
                global_step=4,
                metrics={
                    "eval/accuracy": 0.92,
                    "eval/loss": 0.28,
                },
            ),
            train_measurements=create_measurement(
                context="train",
                global_step=4,
                metrics={
                    "train/accuracy": 0.87,
                    "train/loss": 0.33,
                },
            ),
        ),
        # Early-stopping node (lower final step, high density)
        create_node_structure(
            eval_measurements=[
                create_measurement(
                    context="eval",
                    global_step=0,
                    metrics={"eval/accuracy": 0.71, "eval/loss": 0.49},
                ),
                create_measurement(
                    context="eval",
                    global_step=1,
                    metrics={"eval/accuracy": 0.78, "eval/loss": 0.42},
                ),
                create_measurement(
                    context="eval",
                    global_step=2,
                    metrics={"eval/accuracy": 0.84, "eval/loss": 0.36},
                ),
            ],
            train_measurements=[
                create_measurement(
                    context="train",
                    global_step=0,
                    metrics={"train/accuracy": 0.67, "train/loss": 0.53},
                ),
                create_measurement(
                    context="train",
                    global_step=1,
                    metrics={"train/accuracy": 0.74, "train/loss": 0.46},
                ),
                create_measurement(
                    context="train",
                    global_step=2,
                    metrics={"train/accuracy": 0.81, "train/loss": 0.39},
                ),
            ],
        ),
    ]


def generate_identical_values_test() -> List[Dict[str, Any]]:
    """Test identical values (CV=0 cases).

    Different identical value scenarios:
    - Identical positive values
    - All zeros
    - Identical very small values
    - Identical large values
    - Mix of identical and varied metrics
    """
    return [
        # All identical positive accuracy values
        create_node_structure(
            eval_measurements=[
                create_measurement(
                    context="eval", metrics={"eval/accuracy": 0.85, "eval/loss": 0.20}
                ),
                create_measurement(
                    context="eval", metrics={"eval/accuracy": 0.85, "eval/loss": 0.20}
                ),
                create_measurement(
                    context="eval", metrics={"eval/accuracy": 0.85, "eval/loss": 0.20}
                ),
            ],
            train_measurements=[
                create_measurement(
                    context="train",
                    metrics={"train/accuracy": 0.80, "train/loss": 0.25},
                ),
                create_measurement(
                    context="train",
                    metrics={"train/accuracy": 0.80, "train/loss": 0.25},
                ),
                create_measurement(
                    context="train",
                    metrics={"train/accuracy": 0.80, "train/loss": 0.25},
                ),
            ],
        ),
        # All identical zero values (CV undefined case)
        create_node_structure(
            eval_measurements=[
                create_measurement(
                    context="eval", metrics={"eval/accuracy": 0.0, "eval/loss": 0.0}
                ),
                create_measurement(
                    context="eval", metrics={"eval/accuracy": 0.0, "eval/loss": 0.0}
                ),
                create_measurement(
                    context="eval", metrics={"eval/accuracy": 0.0, "eval/loss": 0.0}
                ),
            ],
            train_measurements=[
                create_measurement(
                    context="train", metrics={"train/accuracy": 0.0, "train/loss": 0.0}
                ),
                create_measurement(
                    context="train", metrics={"train/accuracy": 0.0, "train/loss": 0.0}
                ),
                create_measurement(
                    context="train", metrics={"train/accuracy": 0.0, "train/loss": 0.0}
                ),
            ],
        ),
        # All identical very small values (near-zero mean, zero variance)
        create_node_structure(
            eval_measurements=[
                create_measurement(
                    context="eval",
                    metrics={"eval/accuracy": 1e-10, "eval/batch_time": 1e-6},
                ),
                create_measurement(
                    context="eval",
                    metrics={"eval/accuracy": 1e-10, "eval/batch_time": 1e-6},
                ),
                create_measurement(
                    context="eval",
                    metrics={"eval/accuracy": 1e-10, "eval/batch_time": 1e-6},
                ),
            ],
            train_measurements=[
                create_measurement(context="train", metrics={"train/batch_time": 1e-5}),
                create_measurement(context="train", metrics={"train/batch_time": 1e-5}),
                create_measurement(context="train", metrics={"train/batch_time": 1e-5}),
            ],
        ),
        # All identical large values
        create_node_structure(
            eval_measurements=[
                create_measurement(
                    context="eval",
                    metrics={"eval/loss": 100.0, "eval/batch_time": 10.0},
                ),
                create_measurement(
                    context="eval",
                    metrics={"eval/loss": 100.0, "eval/batch_time": 10.0},
                ),
                create_measurement(
                    context="eval",
                    metrics={"eval/loss": 100.0, "eval/batch_time": 10.0},
                ),
            ],
            train_measurements=[
                create_measurement(
                    context="train",
                    metrics={"train/loss": 150.0, "train/batch_time": 12.0},
                ),
                create_measurement(
                    context="train",
                    metrics={"train/loss": 150.0, "train/batch_time": 12.0},
                ),
                create_measurement(
                    context="train",
                    metrics={"train/loss": 150.0, "train/batch_time": 12.0},
                ),
            ],
        ),
        # Mix of identical and varied values (some metrics with CV=0, others with CV>0)
        create_node_structure(
            eval_measurements=[
                create_measurement(
                    context="eval",
                    metrics={
                        "eval/accuracy": 0.88,
                        "eval/loss": 0.15,
                        "eval/precision": 0.90,
                    },
                ),
                create_measurement(
                    context="eval",
                    metrics={
                        "eval/accuracy": 0.88,
                        "eval/loss": 0.25,
                        "eval/precision": 0.85,
                    },
                ),
                create_measurement(
                    context="eval",
                    metrics={
                        "eval/accuracy": 0.88,
                        "eval/loss": 0.35,
                        "eval/precision": 0.80,
                    },
                ),
                # accuracy: CV=0% (identical), loss: CV>0% (varied), precision: CV>0% (varied)
            ],
            train_measurements=[
                create_measurement(
                    context="train",
                    metrics={"train/accuracy": 0.82, "train/loss": 0.30},
                ),
                create_measurement(
                    context="train",
                    metrics={"train/accuracy": 0.82, "train/loss": 0.30},
                ),
                create_measurement(
                    context="train",
                    metrics={"train/accuracy": 0.82, "train/loss": 0.30},
                ),
                # Both metrics: CV=0% (identical)
            ],
        ),
    ]


def generate_mixed_issues_test() -> List[Dict[str, Any]]:
    """Test nodes with different completion levels AND bad data.

    Combines completion issues with data problems:
    - Node 1: Full completion with invalid values
    - Node 2: Early stop with out-of-range values
    - Node 3: Partial completion with extreme values
    - Node 4: Full completion with mixed problems
    - Node 5: Very early stop with duplicate measurements
    - Node 6: Clean reference node
    """
    return [
        # Node 1: Completed successfully but with some NaN values
        create_node_structure(
            eval_measurements=[
                create_measurement(
                    context="eval",
                    global_step=0,
                    metrics={"eval/accuracy": 0.70, "eval/loss": 0.45},
                ),
                create_measurement(
                    context="eval",
                    global_step=1,
                    metrics={"eval/accuracy": 0.80, "eval/loss": 0.35},
                ),
                create_measurement(
                    context="eval",
                    global_step=2,
                    metrics={"eval/accuracy": float("nan"), "eval/loss": 0.25},
                ),  # Invalid value at final step
                create_measurement(
                    context="eval",
                    global_step=3,
                    metrics={"eval/accuracy": 0.90, "eval/loss": 0.20},
                ),
            ],
            train_measurements=[
                create_measurement(
                    context="train",
                    global_step=0,
                    metrics={"train/accuracy": 0.65, "train/loss": 0.50},
                ),
                create_measurement(
                    context="train",
                    global_step=1,
                    metrics={"train/accuracy": 0.75, "train/loss": 0.40},
                ),
                create_measurement(
                    context="train",
                    global_step=2,
                    metrics={"train/accuracy": 0.85, "train/loss": 0.30},
                ),
                create_measurement(
                    context="train",
                    global_step=3,
                    metrics={"train/accuracy": 0.88, "train/loss": 0.25},
                ),
            ],
        ),
        # Node 2: Stopped early (step 1) with range violations
        create_node_structure(
            eval_measurements=[
                create_measurement(
                    context="eval",
                    global_step=0,
                    metrics={"eval/accuracy": 0.72, "eval/loss": 0.43},
                ),
                create_measurement(
                    context="eval",
                    global_step=1,
                    metrics={"eval/accuracy": 1.15, "eval/loss": -0.1},
                ),  # Out-of-range values
            ],
            train_measurements=[
                create_measurement(
                    context="train",
                    global_step=0,
                    metrics={"train/accuracy": 0.68, "train/loss": 0.47},
                ),
                create_measurement(
                    context="train",
                    global_step=1,
                    metrics={"train/accuracy": -0.05, "train/loss": 0.38},
                ),  # Out-of-range value
            ],
        ),
        # Node 3: Moderate completion (step 2) with extreme outliers
        create_node_structure(
            eval_measurements=[
                create_measurement(
                    context="eval",
                    global_step=0,
                    metrics={"eval/accuracy": 0.74, "eval/loss": 0.41},
                ),
                create_measurement(
                    context="eval",
                    global_step=1,
                    metrics={"eval/accuracy": 0.82, "eval/loss": 5000.0},
                ),  # Extreme value
                create_measurement(
                    context="eval",
                    global_step=2,
                    metrics={"eval/accuracy": 0.001, "eval/batch_time": 300.0},
                ),  # Mixed extreme values
            ],
            train_measurements=[
                create_measurement(
                    context="train",
                    global_step=0,
                    metrics={"train/accuracy": 0.71, "train/loss": 0.44},
                ),
                create_measurement(
                    context="train",
                    global_step=1,
                    metrics={"train/accuracy": 0.78, "train/loss": 0.39},
                ),
                create_measurement(
                    context="train",
                    global_step=2,
                    metrics={"train/accuracy": 0.84, "train/batch_time": 0.000001},
                ),  # Extreme value
            ],
        ),
        # Node 4: Full completion but with mixed corruption (Inf + type errors)
        create_node_structure(
            eval_measurements=[
                create_measurement(
                    context="eval",
                    global_step=0,
                    metrics={"eval/accuracy": 0.76, "eval/loss": 0.39},
                ),
                create_measurement(
                    context="eval",
                    global_step=1,
                    metrics={"eval/accuracy": 0.84, "eval/loss": float("inf")},
                ),  # Inf corruption
                create_measurement(
                    context="eval",
                    global_step=2,
                    metrics={
                        "eval/accuracy": 0.89,
                        "eval/loss": 0.28,
                        "eval/num_batches": 10.5,
                    },
                ),  # Type error
                create_measurement(
                    context="eval",
                    global_step=3,
                    metrics={"eval/accuracy": 0.92, "eval/loss": 0.22},
                ),
            ],
            train_measurements=[
                create_measurement(
                    context="train",
                    global_step=0,
                    metrics={"train/accuracy": 0.73, "train/loss": 0.42},
                ),
                create_measurement(
                    context="train",
                    global_step=1,
                    metrics={"train/accuracy": 0.79, "train/loss": 0.36},
                ),
                create_measurement(
                    context="train",
                    global_step=2,
                    metrics={"train/accuracy": 0.86, "train/loss": float("-inf")},
                ),  # Negative Inf
                create_measurement(
                    context="train",
                    global_step=3,
                    metrics={"train/accuracy": 0.91, "train/loss": 0.26},
                ),
            ],
        ),
        # Node 5: Very early stopping (step 0 only) with coordinator pollution
        create_node_structure(
            eval_measurements=[
                create_measurement(
                    context="eval",
                    global_step=0,
                    metrics={"eval/accuracy": 0.69, "eval/loss": 0.48},
                ),
                create_measurement(
                    context="eval",
                    global_step=0,
                    metrics={"eval/accuracy": 0.73, "eval/loss": 0.52},
                ),  # Multiple measurements at step 0
                create_measurement(
                    context="eval",
                    global_step=0,
                    metrics={"eval/accuracy": 0.71, "eval/loss": 0.50},
                ),
            ],
            train_measurements=[
                create_measurement(
                    context="train",
                    global_step=0,
                    metrics={"train/accuracy": 0.66, "train/loss": 0.51},
                ),
                create_measurement(
                    context="train",
                    global_step=0,
                    metrics={"train/accuracy": 0.70, "train/loss": 0.49},
                ),  # Multiple measurements
            ],
        ),
        # Node 6: Normal completion for comparison
        create_node_structure(
            eval_measurements=[
                create_measurement(
                    context="eval",
                    global_step=0,
                    metrics={"eval/accuracy": 0.75, "eval/loss": 0.40},
                ),
                create_measurement(
                    context="eval",
                    global_step=1,
                    metrics={"eval/accuracy": 0.83, "eval/loss": 0.32},
                ),
                create_measurement(
                    context="eval",
                    global_step=2,
                    metrics={"eval/accuracy": 0.87, "eval/loss": 0.27},
                ),
                create_measurement(
                    context="eval",
                    global_step=3,
                    metrics={"eval/accuracy": 0.91, "eval/loss": 0.23},
                ),
            ],
            train_measurements=[
                create_measurement(
                    context="train",
                    global_step=0,
                    metrics={"train/accuracy": 0.72, "train/loss": 0.43},
                ),
                create_measurement(
                    context="train",
                    global_step=1,
                    metrics={"train/accuracy": 0.80, "train/loss": 0.35},
                ),
                create_measurement(
                    context="train",
                    global_step=2,
                    metrics={"train/accuracy": 0.85, "train/loss": 0.29},
                ),
                create_measurement(
                    context="train",
                    global_step=3,
                    metrics={"train/accuracy": 0.89, "train/loss": 0.24},
                ),
            ],
        ),
    ]


def generate_data_preservation_test() -> List[Dict[str, Any]]:
    """Test data preservation - no silent filtering.

    Includes various data types to ensure everything shows up in output:
    - Normal values
    - Edge cases
    - Problematic values
    - Out-of-range values
    """
    return [
        # Node with comprehensive data types to test all transformation paths
        create_node_structure(
            eval_measurements=[
                # Normal values
                create_measurement(
                    context="eval",
                    global_step=0,
                    metrics={"eval/accuracy": 0.85, "eval/loss": 0.22},
                ),
                # Edge case values
                create_measurement(
                    context="eval",
                    global_step=1,
                    metrics={
                        "eval/accuracy": 0.0,  # Exact zero
                        "eval/loss": 1.0,  # Exact one
                        "eval/precision": 0.999999,  # Near one
                        "eval/recall": 0.000001,  # Near zero
                        "eval/batch_time": 1e-10,  # Very small positive
                    },
                ),
                # Problematic values
                create_measurement(
                    context="eval",
                    global_step=2,
                    metrics={
                        "eval/accuracy": float("nan"),  # NaN
                        "eval/loss": float("inf"),  # Inf
                        "eval/precision": -0.1,  # Out of range
                        "eval/recall": 1.5,  # Out of range
                        "eval/f1": float("-inf"),  # Negative Inf
                    },
                ),
                # Extreme but valid values
                create_measurement(
                    context="eval",
                    global_step=3,
                    metrics={
                        "eval/accuracy": 1e-15,  # Extremely small but valid
                        "eval/loss": 1e15,  # Extremely large but valid
                        "eval/batch_time": 0.1,  # Valid timing
                    },
                ),
            ],
            train_measurements=[
                # Mixed valid and problematic train data
                create_measurement(
                    context="train",
                    global_step=0,
                    metrics={"train/accuracy": 0.80, "train/loss": 0.28},
                ),
                create_measurement(
                    context="train",
                    global_step=1,
                    metrics={
                        "train/accuracy": float("nan"),
                        "train/loss": 999999.9,  # Very large loss
                        "train/batch_time": -1.0,  # Impossible negative time
                    },
                ),
                create_measurement(
                    context="train",
                    global_step=2,
                    metrics={
                        "train/accuracy": 0.001,  # Very low but valid accuracy
                        "train/loss": 0.0,  # Perfect loss
                    },
                ),
            ],
        ),
        # Node with all identical values to test CV=0 preservation
        create_node_structure(
            eval_measurements=[
                create_measurement(
                    context="eval",
                    metrics={"eval/accuracy": 0.777777, "eval/loss": 0.333333},
                ),
                create_measurement(
                    context="eval",
                    metrics={"eval/accuracy": 0.777777, "eval/loss": 0.333333},
                ),
                create_measurement(
                    context="eval",
                    metrics={"eval/accuracy": 0.777777, "eval/loss": 0.333333},
                ),
            ],
        ),
        # Node with precision edge cases
        create_node_structure(
            eval_measurements=[
                create_measurement(
                    context="eval",
                    metrics={
                        "eval/accuracy": 0.123456789012345,  # High precision should be preserved in some form
                        "eval/loss": 1.0000000000001,  # Near-integer with tiny difference
                        "eval/precision": 0.9999999999999,  # Near-one with tiny difference
                    },
                ),
            ],
            train_measurements=[
                create_measurement(
                    context="train",
                    metrics={
                        "train/accuracy": 0.987654321987654,  # High precision
                        "train/loss": 1.23456789e-12,  # Scientific notation edge case
                    },
                ),
            ],
        ),
        # Node with repeated identical timestamps (duplicate measurements scenario)
        create_node_structure(
            eval_measurements=[
                create_measurement(
                    context="eval",
                    global_step=1,
                    metrics={"eval/accuracy": 0.70, "eval/loss": 0.40},
                ),
                create_measurement(
                    context="eval",
                    global_step=1,
                    metrics={"eval/accuracy": 0.75, "eval/loss": 0.38},
                ),  # Same step
                create_measurement(
                    context="eval",
                    global_step=1,
                    metrics={"eval/accuracy": 0.72, "eval/loss": 0.42},
                ),  # Same step
                # All measurements should be accounted for, not silently merged
            ],
        ),
    ]


def generate_node_selection_test() -> List[Dict[str, Any]]:
    """Test node selection for statistics.

    Various node types to test selection criteria:
    - Complete high-quality data
    - Incomplete data
    - Corrupted data
    - Empty/missing data
    - Mixed quality data
    """
    return [
        # Node 1: Complete, high-quality data
        create_node_structure(
            eval_measurements=[
                create_measurement(
                    context="eval",
                    global_step=0,
                    metrics={"eval/accuracy": 0.75, "eval/loss": 0.35},
                ),
                create_measurement(
                    context="eval",
                    global_step=1,
                    metrics={"eval/accuracy": 0.82, "eval/loss": 0.28},
                ),
                create_measurement(
                    context="eval",
                    global_step=2,
                    metrics={"eval/accuracy": 0.88, "eval/loss": 0.22},
                ),
                create_measurement(
                    context="eval",
                    global_step=3,
                    metrics={"eval/accuracy": 0.91, "eval/loss": 0.18},
                ),
            ],
            train_measurements=[
                create_measurement(
                    context="train",
                    global_step=0,
                    metrics={"train/accuracy": 0.70, "train/loss": 0.40},
                ),
                create_measurement(
                    context="train",
                    global_step=1,
                    metrics={"train/accuracy": 0.78, "train/loss": 0.32},
                ),
                create_measurement(
                    context="train",
                    global_step=2,
                    metrics={"train/accuracy": 0.85, "train/loss": 0.25},
                ),
                create_measurement(
                    context="train",
                    global_step=3,
                    metrics={"train/accuracy": 0.89, "train/loss": 0.20},
                ),
            ],
        ),
        # Node 2: Incomplete but valid data
        create_node_structure(
            eval_measurements=[
                create_measurement(
                    context="eval",
                    global_step=0,
                    metrics={"eval/accuracy": 0.74, "eval/loss": 0.36},
                ),
                create_measurement(
                    context="eval",
                    global_step=1,
                    metrics={"eval/accuracy": 0.81, "eval/loss": 0.29},
                ),
                # Stopped early at step 1
            ],
            train_measurements=[
                create_measurement(
                    context="train",
                    global_step=0,
                    metrics={"train/accuracy": 0.69, "train/loss": 0.41},
                ),
                create_measurement(
                    context="train",
                    global_step=1,
                    metrics={"train/accuracy": 0.77, "train/loss": 0.33},
                ),
            ],
        ),
        # Node 3: Complete but with some corrupted measurements
        create_node_structure(
            eval_measurements=[
                create_measurement(
                    context="eval",
                    global_step=0,
                    metrics={"eval/accuracy": 0.73, "eval/loss": 0.37},
                ),
                create_measurement(
                    context="eval",
                    global_step=1,
                    metrics={"eval/accuracy": float("nan"), "eval/loss": 0.30},
                ),  # NaN accuracy
                create_measurement(
                    context="eval",
                    global_step=2,
                    metrics={"eval/accuracy": 0.87, "eval/loss": 0.23},
                ),
                create_measurement(
                    context="eval",
                    global_step=3,
                    metrics={"eval/accuracy": 0.90, "eval/loss": 0.19},
                ),
            ],
            train_measurements=[
                create_measurement(
                    context="train",
                    global_step=0,
                    metrics={"train/accuracy": 0.68, "train/loss": 0.42},
                ),
                create_measurement(
                    context="train",
                    global_step=1,
                    metrics={"train/accuracy": 0.76, "train/loss": 0.34},
                ),
                create_measurement(
                    context="train",
                    global_step=2,
                    metrics={"train/accuracy": 0.84, "train/loss": 0.26},
                ),
                create_measurement(
                    context="train",
                    global_step=3,
                    metrics={"train/accuracy": 0.88, "train/loss": 0.21},
                ),
            ],
        ),
        # Node 4: Single measurement node (minimal data - enough for statistics?)
        create_node_structure(
            eval_measurements=create_measurement(
                context="eval",
                global_step=3,
                metrics={
                    "eval/accuracy": 0.92,
                    "eval/loss": 0.17,
                },
            ),
            train_measurements=create_measurement(
                context="train",
                global_step=3,
                metrics={
                    "train/accuracy": 0.90,
                    "train/loss": 0.19,
                },
            ),
        ),
        # Node 5: Multiple measurements at same step (duplicate measurements)
        create_node_structure(
            eval_measurements=[
                create_measurement(
                    context="eval",
                    global_step=2,
                    metrics={"eval/accuracy": 0.76, "eval/loss": 0.34},
                ),
                create_measurement(
                    context="eval",
                    global_step=2,
                    metrics={"eval/accuracy": 0.84, "eval/loss": 0.27},
                ),  # Same step
                create_measurement(
                    context="eval",
                    global_step=2,
                    metrics={"eval/accuracy": 0.80, "eval/loss": 0.30},
                ),  # Same step
                create_measurement(
                    context="eval",
                    global_step=3,
                    metrics={"eval/accuracy": 0.89, "eval/loss": 0.21},
                ),
            ],
            train_measurements=[
                create_measurement(
                    context="train",
                    global_step=2,
                    metrics={"train/accuracy": 0.72, "train/loss": 0.38},
                ),
                create_measurement(
                    context="train",
                    global_step=3,
                    metrics={"train/accuracy": 0.87, "train/loss": 0.23},
                ),
            ],
        ),
        # Node 6: Completely empty contexts
        {
            "eval": [],
            "train": [],
        },
        # Node 7: Only train data, no eval
        create_node_structure(
            train_measurements=[
                create_measurement(
                    context="train",
                    global_step=0,
                    metrics={"train/accuracy": 0.71, "train/loss": 0.39},
                ),
                create_measurement(
                    context="train",
                    global_step=1,
                    metrics={"train/accuracy": 0.79, "train/loss": 0.31},
                ),
                create_measurement(
                    context="train",
                    global_step=2,
                    metrics={"train/accuracy": 0.86, "train/loss": 0.24},
                ),
                create_measurement(
                    context="train",
                    global_step=3,
                    metrics={"train/accuracy": 0.91, "train/loss": 0.18},
                ),
            ],
        ),
        # Node 8: Extreme outlier node (should outliers be included or excluded from statistics?)
        create_node_structure(
            eval_measurements=[
                create_measurement(
                    context="eval",
                    global_step=0,
                    metrics={"eval/accuracy": 0.01, "eval/loss": 5.0},
                ),  # Extreme outlier
                create_measurement(
                    context="eval",
                    global_step=1,
                    metrics={"eval/accuracy": 0.02, "eval/loss": 4.8},
                ),  # Extreme outlier
                create_measurement(
                    context="eval",
                    global_step=2,
                    metrics={"eval/accuracy": 0.03, "eval/loss": 4.5},
                ),  # Extreme outlier
                create_measurement(
                    context="eval",
                    global_step=3,
                    metrics={"eval/accuracy": 0.05, "eval/loss": 4.2},
                ),  # Extreme outlier
            ],
            train_measurements=[
                create_measurement(
                    context="train",
                    global_step=0,
                    metrics={"train/accuracy": 0.02, "train/loss": 4.9},
                ),
                create_measurement(
                    context="train",
                    global_step=1,
                    metrics={"train/accuracy": 0.04, "train/loss": 4.6},
                ),
                create_measurement(
                    context="train",
                    global_step=2,
                    metrics={"train/accuracy": 0.06, "train/loss": 4.3},
                ),
                create_measurement(
                    context="train",
                    global_step=3,
                    metrics={"train/accuracy": 0.08, "train/loss": 4.0},
                ),
            ],
        ),
    ]


def generate_completion_patterns_test() -> List[Dict[str, Any]]:
    """Test various completion patterns.

    Mix of completion scenarios:
    - Zero completion (failed at start)
    - Partial completion (stopped partway)
    - Different final steps across nodes
    """
    # Zero common final step scenarios
    zero_completion_specs = [
        (
            0,
            0,
            0,
            0,
            {"eval/accuracy": 0.65, "eval/loss": 0.45},
            {"train/accuracy": 0.60, "train/loss": 0.50},
        ),
        (
            0,
            0,
            1,
            1,
            {"eval/accuracy": 0.70, "eval/loss": 0.40},
            {"train/accuracy": 0.68, "train/loss": 0.42},
        ),
        (
            0,
            1,
            0,
            2,
            {"eval/accuracy": 0.75, "eval/loss": 0.35},
            {"train/accuracy": 0.73, "train/loss": 0.37},
        ),
    ]

    # Partial completion patterns
    test_nodes = generate_test_data(
        5, 4, 2
    )  # Generate base data with partial completion

    # Create zero completion nodes with proper structure
    zero_nodes = []
    for r, e, b, s, eval_metrics, train_metrics in zero_completion_specs:
        node = create_node_structure(
            eval_measurements=create_measurement(
                context="eval",
                round_idx=r,
                epoch_idx=e,
                batch_idx=b,
                global_step=s,
                metrics=eval_metrics,
            ),
            train_measurements=create_measurement(
                context="train",
                round_idx=r,
                epoch_idx=e,
                batch_idx=b,
                global_step=s,
                metrics=train_metrics,
            ),
        )
        zero_nodes.append(node)

    # Combine both types for comprehensive completion testing
    return zero_nodes + test_nodes[:7]


# Test scenario definitions
TEST_SCENARIOS = [
    ("01. Single Node Evaluation", generate_single_node_test),
    ("02. Completion Patterns", generate_completion_patterns_test),
    ("03. Bad Data Detection", generate_bad_data_test),
    ("04. Duplicate Measurements", generate_duplicate_measurements_test),
    ("05. Many Nodes Sampling", lambda: generate_test_data(25, 1, 1)),
    ("06. Variability Thresholds", generate_variability_thresholds_test),
    ("07. Missing Data Handling", generate_empty_data_test),
    ("08. Measurement Patterns", generate_measurement_patterns_test),
    ("09. Identical Values", generate_identical_values_test),
    ("10. Mixed Issues", generate_mixed_issues_test),
    ("11. Data Preservation", generate_data_preservation_test),
    ("12. Node Selection", generate_node_selection_test),
]


def run_test(test_name: str, results: List[Any]) -> None:
    """Run a single test."""
    print(f"🚀 Starting test: {test_name}")
    print_test_configuration(test_name, results, len(results), 2, DEFAULTS["duration"])
    ResultsDisplay().show_experiment_results(
        results, DEFAULTS["duration"], 2, len(results)
    )


def find_test_by_query(query: str) -> Tuple[Optional[str], Optional[Any]]:
    """Find test scenario by ID or name match."""
    query = query.strip().lower()
    for i, (name, generator) in enumerate(TEST_SCENARIOS, 1):
        if query in [str(i).zfill(2), str(i)] or query in name.lower():
            return name, generator
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Test OmniFed results display")
    parser.add_argument("--experiment", default="outputs", help="Experiment directory")
    parser.add_argument(
        "--synthetic-data", action="store_true", help="Generate synthetic test data"
    )
    parser.add_argument(
        "--all-tests", action="store_true", help="Run all test scenarios"
    )
    parser.add_argument("--test-by-name", help="Run specific test by name or ID")
    parser.add_argument(
        "--list-tests", action="store_true", help="List available tests"
    )
    parser.add_argument(
        "--rounds", type=int, default=2, help="Rounds for synthetic data"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Epochs for synthetic data"
    )
    parser.add_argument(
        "--batches", type=int, default=2, help="Batches for synthetic data"
    )
    args = parser.parse_args()

    # List tests
    if args.list_tests:
        print("Available test scenarios:")
        for name, _ in TEST_SCENARIOS:
            print(f"  {name}")
        print("\nUsage: --test-by-name '01' or --test-by-name 'Single Node'")
        return

    # Run specific test
    if args.test_by_name:
        test_name, generator = find_test_by_query(args.test_by_name)
        if not generator or not test_name:
            print(f"❌ No test found for '{args.test_by_name}'")
            return

        print(f"🎯 Running: {test_name}")
        run_test(test_name, generator())
        return

    # Run all tests
    if args.all_tests:
        print("🧪 RUNNING ALL TESTS")
        for i, (name, generator) in enumerate(TEST_SCENARIOS, 1):
            print_rule(f"TEST {i}/{len(TEST_SCENARIOS)}: {name}")
            run_test(name, generator())
            print(f"✅ Completed: {name}")
        print_rule("ALL TESTS COMPLETED")
        return

    # Generate synthetic data or load real experiment
    if args.synthetic_data:
        test_name = "Comprehensive Test"
        results = generate_test_data(args.rounds, args.epochs, args.batches)
        print(f"Generated synthetic data: {len(results)} nodes")
    else:
        test_name = "Real Experiment"
        results = find_and_load_experiment(args.experiment)

    run_test(test_name, results)


if __name__ == "__main__":
    main()
