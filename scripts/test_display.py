#!/usr/bin/env python3
"""
Simple test script for FLUX experiment results display.

Design note: This script intentionally avoids catching broad exceptions so that
any underlying errors inside the display pipeline surface with full tracebacks.
If something would have crashed, we let it crash to maximize debuggability.
"""

import argparse
import pickle
import sys
from pathlib import Path
import numpy as np

# Add FLUX root to path
script_dir = Path(__file__).parent
flux_root = script_dir.parent
sys.path.insert(0, str(flux_root))

from src.flora.utils.results_display import ResultsDisplayManager
from rich.console import Console
from rich.table import Table
from rich import box


def find_latest_experiment(outputs_dir="outputs"):
    """Find the most recent experiment with node results."""
    outputs_path = Path(outputs_dir)

    # Look for node_*_results.pkl files in various patterns
    patterns = [
        "*/*/engine/node_results/node_*_results.pkl",
        "*/*/node_results/node_*_results.pkl",
        "*/engine/node_results/node_*_results.pkl",
        "*/node_results/node_*_results.pkl",
    ]

    found_dirs = set()
    for pattern in patterns:
        for pkl_file in outputs_path.glob(pattern):
            found_dirs.add(pkl_file.parent)

    if not found_dirs:
        raise FileNotFoundError(f"No node_*_results.pkl files found in {outputs_path}")

    # Return most recent
    latest = max(found_dirs, key=lambda x: x.stat().st_mtime)
    print(f"Using: {latest}")
    return latest


def load_node_results(results_dir):
    """Load all node result files from directory."""
    pkl_files = list(results_dir.glob("node_*_results.pkl"))

    if not pkl_files:
        raise FileNotFoundError(f"No node_*_results.pkl files in {results_dir}")

    # Sort by node number - let natural exceptions show filename format issues
    pkl_files.sort(key=lambda x: int(x.stem.split("_")[1]))

    results = []
    for pkl_file in pkl_files:
        with open(pkl_file, "rb") as f:
            results.append(pickle.load(f))

    print(f"Loaded {len(results)} node files")
    return results


def print_test_config(test_name, results, total_nodes, rounds, duration, args=None):
    """Print detailed test configuration and data structure."""
    console = Console()

    # Main configuration table
    config_table = Table(title=f"🔧 {test_name} - Configuration", box=box.ROUNDED)
    config_table.add_column("Parameter", style="bold cyan", justify="left")
    config_table.add_column("Value", style="white", justify="left")

    config_table.add_row("Test Scenario", test_name)
    config_table.add_row("Total Nodes", str(total_nodes))
    config_table.add_row("Global Rounds", str(rounds))
    config_table.add_row("Duration (s)", f"{duration:.1f}")

    if args:
        config_table.add_row("Rounds (param)", str(args.rounds))
        config_table.add_row("Epochs (param)", str(args.epochs))
        config_table.add_row("Batches (param)", str(args.batches))

    console.print(config_table)

    # Data structure analysis
    structure_table = Table(title="📊 Test Data Structure Analysis", box=box.ROUNDED)
    structure_table.add_column("Node ID", style="bold blue", justify="center")
    structure_table.add_column("Contexts", style="cyan", justify="left")
    structure_table.add_column("Measurements", style="green", justify="center")
    structure_table.add_column("Final Step", style="yellow", justify="center")
    structure_table.add_column("Metrics Sample", style="dim", justify="left")

    for node_idx, node_data in enumerate(results):
        contexts = list(node_data.keys())
        total_measurements = sum(len(node_data[ctx]) for ctx in contexts)

        # Find max global step for this node
        max_step = 0
        sample_metrics = set()
        for context, measurements in node_data.items():
            for measurement in measurements:
                max_step = max(max_step, measurement.get("global_step", 0))
                sample_metrics.update(
                    [
                        k
                        for k in measurement.keys()
                        if k
                        not in {
                            "global_step",
                            "round_idx",
                            "epoch_idx",
                            "batch_idx",
                            "_node_id",
                        }
                    ]
                )

        sample_metrics_str = ", ".join(sorted(list(sample_metrics))[:3])
        if len(sample_metrics) > 3:
            sample_metrics_str += "..."

        structure_table.add_row(
            str(node_idx),
            ", ".join(contexts),
            str(total_measurements),
            str(max_step),
            sample_metrics_str or "none",
        )

    console.print(structure_table)

    # Metrics coverage analysis
    all_contexts = set()
    for node_data in results:
        all_contexts.update(node_data.keys())

    for context in sorted(all_contexts):
        metrics_table = Table(
            title=f"🎯 {context.title()} Context - Metrics Coverage", box=box.ROUNDED
        )
        metrics_table.add_column("Metric", style="bold cyan", justify="left")
        metrics_table.add_column("Reporting Nodes", style="green", justify="center")
        metrics_table.add_column("Sample Values", style="dim", justify="left")

        # Collect all metrics in this context
        context_metrics = set()
        for node_data in results:
            if context in node_data:
                for measurement in node_data[context]:
                    context_metrics.update(
                        [
                            k
                            for k in measurement.keys()
                            if k
                            not in {
                                "global_step",
                                "round_idx",
                                "epoch_idx",
                                "batch_idx",
                                "_node_id",
                            }
                        ]
                    )

        for metric in sorted(context_metrics):
            reporting_nodes = []
            sample_values = []

            for node_idx, node_data in enumerate(results):
                if context in node_data:
                    for measurement in node_data[context]:
                        if metric in measurement:
                            reporting_nodes.append(node_idx)
                            sample_values.append(str(measurement[metric]))
                            break  # Only take first occurrence per node

            unique_reporting = len(set(reporting_nodes))
            coverage_pct = (
                (unique_reporting / total_nodes * 100) if total_nodes > 0 else 0
            )
            coverage_str = f"{unique_reporting}/{total_nodes} ({coverage_pct:.0f}%)"

            sample_str = ", ".join(sample_values[:3])
            if len(sample_values) > 3:
                sample_str += "..."

            metrics_table.add_row(metric, coverage_str, sample_str)

        console.print(metrics_table)

    console.print()


def generate_test_data(rounds=2, epochs=3, batches=2):
    """Generate comprehensive test data covering all edge cases."""
    if rounds < 1 or epochs < 1 or batches < 1:
        raise ValueError("Rounds, epochs, and batches must be positive integers")

    print(
        f"Generating comprehensive test data: {rounds} rounds, {epochs} epochs/round, {batches} batches/epoch..."
    )

    def generate_normal_progression(
        node_id, context, max_rounds, max_epochs, max_batches
    ):
        """Generate normal progression data for a node/context."""
        data = []
        global_step = 0

        for round_idx in range(max_rounds):
            for epoch_idx in range(max_epochs):
                for batch_idx in range(max_batches):
                    # Create realistic improving metrics
                    progress = (
                        round_idx * max_epochs * max_batches
                        + epoch_idx * max_batches
                        + batch_idx
                    ) / (max_rounds * max_epochs * max_batches)

                    if context == "eval":
                        data.append(
                            {
                                "round_idx": round_idx,
                                "epoch_idx": epoch_idx,
                                "batch_idx": batch_idx,
                                "global_step": global_step,
                                "eval/accuracy": 0.7
                                + 0.25 * progress
                                + 0.02 * np.random.randn(),
                                "eval/precision": 0.65
                                + 0.3 * progress
                                + 0.02 * np.random.randn(),
                                "eval/recall": 0.68
                                + 0.27 * progress
                                + 0.02 * np.random.randn(),
                                "eval/f1": 0.66
                                + 0.29 * progress
                                + 0.02 * np.random.randn(),
                                "eval/loss": 0.5 * (1 - progress)
                                + 0.05 * np.random.randn(),
                                "eval/batch_time": 0.008 + 0.004 * np.random.randn(),
                                "eval/num_batches": 10,
                                "eval/num_samples": 1000,
                            }
                        )
                    else:  # train
                        data.append(
                            {
                                "round_idx": round_idx,
                                "epoch_idx": epoch_idx,
                                "batch_idx": batch_idx,
                                "global_step": global_step,
                                "train/accuracy": 0.65
                                + 0.28 * progress
                                + 0.02 * np.random.randn(),
                                "train/loss": 0.6 * (1 - progress)
                                + 0.06 * np.random.randn(),
                                "train/batch_time": 0.012 + 0.006 * np.random.randn(),
                                "train/num_batches": 20,
                            }
                        )

                    global_step += 1

        return data

    test_data = []

    # Node 0: Normal progression data
    node_0 = {
        "eval": generate_normal_progression(0, "eval", rounds, epochs, batches),
        "train": generate_normal_progression(0, "train", rounds, epochs, batches),
    }
    test_data.append(node_0)

    # Node 1: Validation edge cases (single measurement with various error types)
    final_step = rounds * epochs * batches - 1

    node_1 = {
        "eval": [
            {
                "round_idx": rounds - 1,
                "epoch_idx": epochs - 1,
                "batch_idx": batches - 1,
                "global_step": final_step,
                "eval/accuracy": 1.2,  # 🔴 impossible (>1.0)
                "eval/precision": -0.1,  # 🔴 impossible (negative accuracy)
                "eval/recall": float("inf"),  # 💥 implementation error
                "eval/f1": float("nan"),  # 💥 implementation error
                "eval/loss": -0.5,  # 🔴 impossible (negative loss)
                "eval/batch_time": -0.001,  # 🔴 impossible (negative time)
                "eval/num_batches": 10.5,  # 💥 implementation error (fractional count)
                "eval/num_samples": -500,  # 💥 implementation error (negative count)
            }
        ],
        "train": [
            {
                "round_idx": rounds - 1,
                "epoch_idx": epochs - 1,
                "batch_idx": batches - 1,
                "global_step": final_step,
                "train/accuracy": 0.91,
                "train/loss": 5000,  # ⚠️ anomaly (extreme loss)
                "train/batch_time": 0.0,
                "train/num_batches": 20,
            }
        ],
    }
    test_data.append(node_1)

    # Node 2: Statistical anomalies (only final step)
    node_2 = {
        "eval": [
            {
                "round_idx": rounds - 1,
                "epoch_idx": epochs - 1,
                "batch_idx": batches - 1,
                "global_step": rounds * epochs * batches - 1,
                "eval/accuracy": 0.85,
                "eval/precision": 0.84,
                "eval/recall": 0.86,
                "eval/f1": 0.85,
                "eval/loss": 0.0001,  # ⚠️ anomaly (order of magnitude difference)
                "eval/batch_time": 0.000001,  # ⚠️ anomaly (very small)
                "eval/num_batches": 10,
                "eval/num_samples": 1000,
            }
        ],
        "train": [
            {
                "round_idx": rounds - 1,
                "epoch_idx": epochs - 1,
                "batch_idx": batches - 1,
                "global_step": rounds * epochs * batches - 1,
                "train/accuracy": 0.88,
                "train/loss": 0.08,
                "train/batch_time": 0.012,
                "train/num_batches": 20,
            }
        ],
    }
    test_data.append(node_2)

    # Node 3: Partial progression (stopped early - different final steps for different contexts)
    # This creates a complex out-of-sync scenario where different contexts finish at different steps
    partial_rounds = max(1, rounds - 1)
    partial_epochs = max(1, epochs - 1)

    # Generate different final steps for eval vs train contexts
    eval_data = generate_normal_progression(
        3, "eval", partial_rounds, partial_epochs, batches
    )
    train_data = generate_normal_progression(
        3, "train", partial_rounds, max(1, partial_epochs - 1), batches
    )  # Even earlier for train

    node_3 = {"eval": eval_data, "train": train_data}
    test_data.append(node_3)

    # Node 4: Another partial node (creates multiple excluded nodes scenario)
    if rounds > 1:
        # This node stops at an intermediate point, creating a minority final step
        intermediate_rounds = max(1, rounds - 2) if rounds > 2 else 1
        intermediate_epochs = max(1, epochs - 2) if epochs > 2 else 1

        node_4 = {
            "eval": generate_normal_progression(
                4, "eval", intermediate_rounds, intermediate_epochs, batches
            ),
            "train": generate_normal_progression(
                4, "train", intermediate_rounds, intermediate_epochs, batches
            ),
        }
        test_data.append(node_4)

    # Node 5: Empty context scenario (has train but no eval data)
    node_5 = {
        "train": [
            {
                "round_idx": rounds - 1,
                "epoch_idx": epochs - 1,
                "batch_idx": batches - 1,
                "global_step": final_step,
                "train/accuracy": 0.82,
                "train/loss": 0.15,
                "train/batch_time": 0.008,
                "train/num_batches": 20,
            }
        ]
        # Note: no 'eval' context - tests asymmetric context completion
    }
    test_data.append(node_5)

    # Node 6: Missing metrics scenario (partial metric reporting)
    node_6 = {
        "eval": [
            {
                "round_idx": rounds - 1,
                "epoch_idx": epochs - 1,
                "batch_idx": batches - 1,
                "global_step": final_step,
                "eval/accuracy": 0.89,  # Has accuracy but missing precision/recall/f1
                "eval/loss": 0.18,
                "eval/num_batches": 10,
                "eval/num_samples": 1000,
            }
        ],
        "train": [
            {
                "round_idx": rounds - 1,
                "epoch_idx": epochs - 1,
                "batch_idx": batches - 1,
                "global_step": final_step,
                "train/loss": 0.12,  # Has loss but missing accuracy
                "train/batch_time": 0.009,
                "train/num_batches": 20,
            }
        ],
    }
    test_data.append(node_6)

    # Node 7: All identical values scenario (tests std=0, cv=0 edge case)
    node_7 = {
        "eval": [
            {
                "round_idx": rounds - 1,
                "epoch_idx": epochs - 1,
                "batch_idx": batches - 1,
                "global_step": final_step,
                "eval/accuracy": 0.85,
                "eval/precision": 0.85,
                "eval/recall": 0.85,  # All identical
                "eval/f1": 0.85,
                "eval/loss": 0.25,
                "eval/batch_time": 0.010,  # All identical
                "eval/num_batches": 10,
                "eval/num_samples": 1000,  # All identical
            }
        ],
        "train": [
            {
                "round_idx": rounds - 1,
                "epoch_idx": epochs - 1,
                "batch_idx": batches - 1,
                "global_step": final_step,
                "train/accuracy": 0.87,
                "train/loss": 0.22,  # All identical
                "train/batch_time": 0.011,
                "train/num_batches": 20,  # All identical
            }
        ],
    }
    test_data.append(node_7)

    # Node 8: Extreme values scenario (tests formatting limits)
    if rounds >= 2:  # Only add if we have enough rounds
        node_8 = {
            "eval": [
                {
                    "round_idx": rounds - 1,
                    "epoch_idx": epochs - 1,
                    "batch_idx": batches - 1,
                    "global_step": final_step,
                    "eval/accuracy": 0.92,
                    "eval/precision": 0.91,
                    "eval/recall": 0.93,
                    "eval/f1": 0.92,
                    "eval/loss": 1e-8,  # Very small number
                    "eval/batch_time": 1e-6,  # Microsecond timing
                    "eval/num_batches": 1000000,
                    "eval/num_samples": 1e9,  # Very large numbers
                }
            ],
            "train": [
                {
                    "round_idx": rounds - 1,
                    "epoch_idx": epochs - 1,
                    "batch_idx": batches - 1,
                    "global_step": final_step,
                    "train/accuracy": 0.94,
                    "train/loss": 1e-9,  # Very small loss
                    "train/batch_time": 1e-5,
                    "train/num_batches": 500000,  # Large batch count
                }
            ],
        }
        test_data.append(node_8)

    return test_data


def generate_single_node_test():
    """Generate test case with only one node (tests single node edge case)."""
    print("Generating single node test case...")
    return [
        {
            "eval": [
                {
                    "round_idx": 0,
                    "epoch_idx": 0,
                    "batch_idx": 0,
                    "global_step": 0,
                    "eval/accuracy": 0.88,
                    "eval/precision": 0.86,
                    "eval/recall": 0.90,
                    "eval/f1": 0.88,
                    "eval/loss": 0.22,
                    "eval/batch_time": 0.008,
                    "eval/num_batches": 10,
                    "eval/num_samples": 1000,
                }
            ],
            "train": [
                {
                    "round_idx": 0,
                    "epoch_idx": 0,
                    "batch_idx": 0,
                    "global_step": 0,
                    "train/accuracy": 0.85,
                    "train/loss": 0.28,
                    "train/batch_time": 0.012,
                    "train/num_batches": 20,
                }
            ],
        }
    ]


def generate_coordinator_pollution_test():
    """Generate test case with coordinator pollution (multiple measurements per node per step)."""
    print("Generating coordinator pollution test case...")
    final_step = 3

    return [
        # Node 0: Normal single measurement at final step
        {
            "eval": [
                {
                    "round_idx": 0,
                    "epoch_idx": 1,
                    "batch_idx": 1,
                    "global_step": final_step,
                    "eval/accuracy": 0.85,
                    "eval/loss": 0.25,
                    "eval/num_batches": 10,
                    "eval/num_samples": 1000,
                }
            ],
            "train": [
                {
                    "round_idx": 0,
                    "epoch_idx": 1,
                    "batch_idx": 1,
                    "global_step": final_step,
                    "train/accuracy": 0.80,
                    "train/loss": 0.30,
                    "train/num_batches": 20,
                }
            ],
        },
        # Node 1: COORDINATOR POLLUTION - Multiple measurements at the same final step
        {
            "eval": [
                # First measurement (older timestamp/measurement)
                {
                    "round_idx": 0,
                    "epoch_idx": 1,
                    "batch_idx": 1,
                    "global_step": final_step,
                    "eval/accuracy": 0.70,
                    "eval/loss": 0.45,  # Different values
                    "eval/num_batches": 10,
                    "eval/num_samples": 1000,
                },
                # Second measurement (newer timestamp/measurement) - this should be used
                {
                    "round_idx": 0,
                    "epoch_idx": 1,
                    "batch_idx": 1,
                    "global_step": final_step,
                    "eval/accuracy": 0.88,
                    "eval/loss": 0.18,  # Latest values
                    "eval/num_batches": 10,
                    "eval/num_samples": 1000,
                },
                # Third measurement (newest) - this should be the one used
                {
                    "round_idx": 0,
                    "epoch_idx": 1,
                    "batch_idx": 1,
                    "global_step": final_step,
                    "eval/accuracy": 0.92,
                    "eval/loss": 0.12,  # Final latest values
                    "eval/num_batches": 10,
                    "eval/num_samples": 1000,
                },
            ],
            "train": [
                # Multiple train measurements too
                {
                    "round_idx": 0,
                    "epoch_idx": 1,
                    "batch_idx": 1,
                    "global_step": final_step,
                    "train/accuracy": 0.75,
                    "train/loss": 0.40,
                    "train/num_batches": 20,
                },
                {
                    "round_idx": 0,
                    "epoch_idx": 1,
                    "batch_idx": 1,
                    "global_step": final_step,
                    "train/accuracy": 0.89,
                    "train/loss": 0.20,
                    "train/num_batches": 20,  # Latest should be used
                },
            ],
        },
        # Node 2: Normal single measurement
        {
            "eval": [
                {
                    "round_idx": 0,
                    "epoch_idx": 1,
                    "batch_idx": 1,
                    "global_step": final_step,
                    "eval/accuracy": 0.90,
                    "eval/loss": 0.15,
                    "eval/num_batches": 10,
                    "eval/num_samples": 1000,
                }
            ],
            "train": [
                {
                    "round_idx": 0,
                    "epoch_idx": 1,
                    "batch_idx": 1,
                    "global_step": final_step,
                    "train/accuracy": 0.87,
                    "train/loss": 0.22,
                    "train/num_batches": 20,
                }
            ],
        },
    ]


def generate_zero_completion_test():
    """Generate test where no nodes reach a common final step."""
    print("Generating zero completion test case...")
    return [
        # Node 0: Stops at step 0
        {
            "eval": [
                {
                    "round_idx": 0,
                    "epoch_idx": 0,
                    "batch_idx": 0,
                    "global_step": 0,
                    "eval/accuracy": 0.65,
                    "eval/loss": 0.45,
                    "eval/num_batches": 10,
                    "eval/num_samples": 1000,
                }
            ],
            "train": [
                {
                    "round_idx": 0,
                    "epoch_idx": 0,
                    "batch_idx": 0,
                    "global_step": 0,
                    "train/accuracy": 0.60,
                    "train/loss": 0.50,
                    "train/num_batches": 20,
                }
            ],
        },
        # Node 1: Stops at step 1
        {
            "eval": [
                {
                    "round_idx": 0,
                    "epoch_idx": 0,
                    "batch_idx": 1,
                    "global_step": 1,
                    "eval/accuracy": 0.70,
                    "eval/loss": 0.40,
                    "eval/num_batches": 10,
                    "eval/num_samples": 1000,
                }
            ],
            "train": [
                {
                    "round_idx": 0,
                    "epoch_idx": 0,
                    "batch_idx": 1,
                    "global_step": 1,
                    "train/accuracy": 0.68,
                    "train/loss": 0.42,
                    "train/num_batches": 20,
                }
            ],
        },
        # Node 2: Stops at step 2
        {
            "eval": [
                {
                    "round_idx": 0,
                    "epoch_idx": 1,
                    "batch_idx": 0,
                    "global_step": 2,
                    "eval/accuracy": 0.75,
                    "eval/loss": 0.35,
                    "eval/num_batches": 10,
                    "eval/num_samples": 1000,
                }
            ],
            "train": [
                {
                    "round_idx": 0,
                    "epoch_idx": 1,
                    "batch_idx": 0,
                    "global_step": 2,
                    "train/accuracy": 0.73,
                    "train/loss": 0.37,
                    "train/num_batches": 20,
                }
            ],
        },
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Test FLUX experiment results display system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--experiment",
        default="outputs",
        help="Experiment directory or direct path to node results (default: outputs)",
    )

    parser.add_argument(
        "--test-data",
        action="store_true",
        help="Generate comprehensive synthetic test data instead of loading real experiment",
    )

    parser.add_argument(
        "--single-node",
        action="store_true",
        help="Test single node scenario (edge case with only 1 node)",
    )

    parser.add_argument(
        "--zero-completion",
        action="store_true",
        help="Test zero completion scenario (all nodes fail at different steps)",
    )

    parser.add_argument(
        "--coordinator-pollution",
        action="store_true",
        help="Test coordinator pollution scenario (multiple measurements per node per step)",
    )

    parser.add_argument(
        "--all-tests",
        action="store_true",
        help="Run all test scenarios sequentially for comprehensive validation",
    )

    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print detailed configuration and test data structure along with tables",
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="Number of global rounds for test data (default: 2)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs per round for test data (default: 3)",
    )

    parser.add_argument(
        "--batches",
        type=int,
        default=2,
        help="Number of batches per epoch for test data (default: 2)",
    )

    args = parser.parse_args()

    # Handle --all-tests flag by running all scenarios
    if args.all_tests:
        test_scenarios = [
            ("Single Node Test", lambda: generate_single_node_test()),
            ("Zero Completion Test", lambda: generate_zero_completion_test()),
            (
                "Coordinator Pollution Test",
                lambda: generate_coordinator_pollution_test(),
            ),
            (
                "Comprehensive Test",
                lambda: generate_test_data(args.rounds, args.epochs, args.batches),
            ),
        ]

        print("🧪 RUNNING ALL TEST SCENARIOS")
        print("=" * 80)

        for i, (test_name, test_generator) in enumerate(test_scenarios, 1):
            print(f"\n🔍 TEST {i}/4: {test_name}")
            print("-" * 60)

            results = test_generator()
            total_nodes = len(results)
            duration = 10.0
            rounds = args.rounds if test_name == "Comprehensive Test" else 1

            print(
                f"Testing display with {total_nodes} nodes, {rounds} rounds, {duration}s duration"
            )

            if args.show_config:
                print_test_config(
                    test_name, results, total_nodes, rounds, duration, args
                )

            display = ResultsDisplayManager()
            display.show_experiment_results(
                results=results,
                duration=duration,
                global_rounds=rounds,
                total_nodes=total_nodes,
            )
            print(f"✅ {test_name} completed successfully!")

            if i < len(test_scenarios):
                print("\n" + "=" * 80)

        print(f"\n🎉 ALL {len(test_scenarios)} TEST SCENARIOS COMPLETED SUCCESSFULLY!")
        return

    # Determine test scenario and generate data
    test_scenarios = {
        "single_node": (generate_single_node_test, "Single Node Test"),
        "zero_completion": (generate_zero_completion_test, "Zero Completion Test"),
        "coordinator_pollution": (
            generate_coordinator_pollution_test,
            "Coordinator Pollution Test",
        ),
        "test_data": (
            lambda: generate_test_data(args.rounds, args.epochs, args.batches),
            "Comprehensive Test",
        ),
    }

    # Find which test scenario to run
    active_scenario = None
    for scenario_name, (generator, test_name) in test_scenarios.items():
        if getattr(args, scenario_name):
            results = generator()
            total_nodes = len(results)
            rounds = args.rounds if scenario_name == "test_data" else 1
            active_scenario = test_name
            print(f"Generated {test_name.lower()}: {total_nodes} nodes")
            break

    if active_scenario is None:
        # Load real experiment data
        experiment_path = Path(args.experiment)
        results_dir = (
            experiment_path
            if (experiment_path / "node_0_results.pkl").exists()
            else find_latest_experiment(args.experiment)
        )

        results = load_node_results(results_dir)
        total_nodes = len(results)
        active_scenario = "Real Experiment"

        # Infer rounds from data
        rounds = 1
        for node_result in results:
            for context_data in node_result.values():
                if isinstance(context_data, list):
                    for measurement in context_data:
                        if isinstance(measurement, dict):
                            rounds = max(rounds, measurement.get("round_idx", 0) + 1)

    duration = 10.0  # Standard duration for all tests

    # Test the display system - let it crash if there are issues
    print(
        f"\nTesting display with {total_nodes} nodes, {rounds} rounds, {duration}s duration"
    )
    print("=" * 80)

    # Print configuration if requested
    if args.show_config:
        print_test_config(active_scenario, results, total_nodes, rounds, duration, args)

    if args.test_data:
        print(
            "📊 Comprehensive test includes coordinator pollution: nodes with multiple measurements per step"
        )

    # Run the display system
    display = ResultsDisplayManager()
    display.show_experiment_results(
        results=results,
        duration=duration,
        global_rounds=rounds,
        total_nodes=total_nodes,
    )

    print("=" * 80)
    print("✅ Display test completed successfully!")


if __name__ == "__main__":
    main()
