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

import atexit
import csv
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Optional

import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import MeanMetric, SumMetric


class MetricAggType(str, Enum):
    """Enumeration of metric aggregation strategies."""

    MEAN = "mean"
    SUM = "sum"


class _MetricContext:
    """Context manager and decorator for metric context switching."""

    def __init__(
        self,
        logger: "MetricLogger",
        ctx_name: str,
        log_duration: bool = True,
        duration_key: Optional[str] = None,
        print_progress: bool = True,
    ):
        self._logger = logger
        self._context_name = ctx_name
        self._duration_key = duration_key or f"time/{ctx_name}"
        self._print_progress = print_progress
        self._log_duration = log_duration
        self.previous_context: Optional[str] = None
        self._start_time: Optional[float] = None

    def __enter__(self):
        self.previous_context = self._logger.current_context
        self._logger.current_context = self._context_name

        if self._print_progress:
            print(
                f"[{self._context_name}] START @ {self._logger.progress_info_str}",
                flush=True,
            )

        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Log duration if enabled
        if self._log_duration and self._start_time is not None:
            duration = time.time() - self._start_time
            self._logger.log_metric(self._duration_key, duration)

        # Flush metrics for this context
        flushed_metrics = self._logger.flush_metrics(self._context_name)

        if self._print_progress:
            print(
                f"[{self._context_name}] END @ {self._logger.progress_info_str} | {flushed_metrics}",
                flush=True,
            )

        # Restore previous context
        self._logger.current_context = self.previous_context

    def __call__(self, func):
        """Decorator functionality."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Use self directly but update duration key for function name if it's the default
            original_duration_key = self._duration_key
            if self._duration_key == f"time/{self._context_name}":
                self._duration_key = f"time/{func.__name__}"

            try:
                with self:
                    return func(*args, **kwargs)
            finally:
                # Restore original duration key
                self._duration_key = original_duration_key

        return wrapper


class MetricLogger:
    """Mixin for metrics collection and logging.

    Organizes metrics by context (train/eval/aggregate) and
    logs to TensorBoard and CSV files.

    Key features:
        - Separate contexts for different training phases
        - Built-in timing measurement
        - Multiple output formats
        - Dual API: decorators and context managers
        - Flexible coordinate extraction (round/epoch/batch/etc.)

    Note: Not thread-safe. Use one instance per process/actor.
    """

    def __init__(
        self,
        log_dir: str,
        global_step_fn: Callable[[], int],
        metadata_fields: Optional[Dict[str, Callable[[], Any]]] = None,
    ):
        """Initialize metrics collection system.

        Creates log_dir if needed. Overwrites existing metric files.
        Call this directly from inheriting classes, not via super().

        Args:
            log_dir: Directory for metric output files (TensorBoard logs, CSV files)
            global_step_fn: Lambda function that returns current global step
                           e.g., lambda: self.global_step
            metadata_fields: Optional dict mapping field names to lambda functions
                           e.g., {"round_idx": lambda: self.round_idx}
        """

        self._metric_log_dir = log_dir
        self._global_step_fn = global_step_fn
        self._metadata_fields = metadata_fields or {}

        self._tb_writer = SummaryWriter(log_dir)
        self._csv_file = open(os.path.join(log_dir, "metrics_all.csv"), "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)

        # Build dynamic CSV header based on metadata fields
        header = (
            ["global_step"]
            + list(self._metadata_fields.keys())
            + ["metric_name", "metric_value"]
        )
        self._csv_writer.writerow(header)

        self.current_context: str = "default"
        self._ctx_accumulators: Dict[str, Dict[MetricAggType, defaultdict]] = {}
        self._ctx_dataframes = {}

        atexit.register(self.close_metrics)

    def metric_context(
        self,
        name: str,
        *,
        log_duration: bool = True,
        duration_key: Optional[str] = None,
        print_progress: bool = True,
    ):
        """Create a metric context for organized logging (works as both decorator and context manager).

        Groups metrics and automatically measures duration. Flushes when exiting.

        Args:
            name: Context name for grouping metrics
            log_duration: Whether to automatically log execution duration (default: True)
            duration_key: Custom duration metric name (default: f"time/{name}")
            print_progress: Print start/end messages (default: True)

        Examples:
            # As context manager
            with self.metric_context("training"):
                self.log_metric("loss", 0.5)
                self.log_metric("accuracy", 0.9)

            with self.metric_context("evaluation", log_duration=False, print_progress=False):
                self.log_metric("eval_loss", 0.3)

            # As decorator
            @self.metric_context("training", duration_key="custom_time")
            def train_epoch(self):
                self.log_metric("loss", 0.5)
        """
        return _MetricContext(
            self,
            name,
            log_duration=log_duration,
            duration_key=duration_key,
            print_progress=print_progress,
        )

    @classmethod
    def context(
        cls,
        name: str,
        *,
        log_duration: bool = True,
        duration_key: Optional[str] = None,
        print_progress: bool = True,
    ):
        """Create a decorator for metric context switching (class-level decorator syntax).

        Args:
            name: Context name for grouping metrics
            log_duration: Whether to automatically log execution duration (default: True)
            duration_key: Custom duration metric name (default: uses function name)
            print_progress: Print start/end messages (default: True)

        Example:
            @MetricLogger.context("training", duration_key="train_time")
            def train_epoch(self):
                self.log_metric("loss", 0.5)
        """

        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                with self.metric_context(
                    name,
                    log_duration=log_duration,
                    duration_key=duration_key or f"time/{func.__name__}",
                    print_progress=print_progress,
                ):
                    return func(self, *args, **kwargs)

            return wrapper

        return decorator

    @property
    def progress_info_str(self) -> str:
        """Current status line with global step and all metadata."""
        global_step = self._global_step_fn()
        parts = []

        for field_name, field_fn in self._metadata_fields.items():
            try:
                value = field_fn()
                parts.append(f"{field_name}={value}")
            except Exception:
                pass

        return f"global_step={global_step} ({' '.join(parts)})"

    def log_metric(
        self,
        name: str,
        value: float,
        aggregation: MetricAggType = MetricAggType.MEAN,
    ) -> None:
        """Log a metric value in the current context.

        Args:
            name: Metric name (e.g., "loss/train", "accuracy/eval")
            value: Value to log
            aggregation: MEAN for averages, SUM for totals

        Example:
            self.log_metric("loss/train", 0.5)
            self.log_metric("samples/train", 32, MetricAggregation.SUM)
        """
        # Initialize current context if not exists
        if self.current_context not in self._ctx_accumulators:
            self._ctx_accumulators[self.current_context] = {
                MetricAggType.MEAN: defaultdict(MeanMetric),
                MetricAggType.SUM: defaultdict(SumMetric),
            }

        ctx = self._ctx_accumulators[self.current_context]
        match aggregation:
            case MetricAggType.MEAN:
                ctx[MetricAggType.MEAN][name].update(value)
            case MetricAggType.SUM:
                ctx[MetricAggType.SUM][name].update(value)
            case _:
                raise ValueError(f"Unknown aggregation type: {aggregation}")

    @contextmanager
    def log_duration(self, metric_key: str, *, print_progress: bool = True):
        """Time a code block and log the duration.

        Args:
            metric_key: Name for the timing metric
            print_progress: Print start/end messages (default: False)

        Example:
            with self.log_duration("time/training"):
                run_training_epoch()

            with self.log_duration("time/sync", print_progress=True):
                perform_synchronization()
        """
        if print_progress:
            print(f"[{metric_key}] START @ {self.progress_info_str}", flush=True)

        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            self.log_metric(metric_key, duration)

            if print_progress:
                print(
                    f"[{metric_key}] END @ {self.progress_info_str} | duration={duration:.3f}s",
                    flush=True,
                )

    def flush_metrics(self, context: Optional[str] = None) -> Dict[str, float]:
        """Write accumulated metrics and reset counters.

        Usually called automatically by @MetricLogger.context decorator.
        Call manually only when changing contexts without the decorator.

        Args:
            context: Context to flush (defaults to current context)

        Returns:
            Dictionary of computed metrics that were flushed

        Note:
            Skips contexts with no metrics. Resets all counters after writing.
        """
        context = context or self.current_context

        # Skip if context doesn't exist (no metrics were accumulated)
        if context not in self._ctx_accumulators:
            print(f"WARN: No metrics to flush for context '{context}'. Skipping.")
            return {}

        metrics = {}
        ctx = self._ctx_accumulators[context]

        # Compute and reset all metrics in this context
        for name, metric in ctx[MetricAggType.MEAN].items():
            if metric.update_count > 0:  # Only compute if metric has been updated
                metrics[name] = metric.compute().item()
            metric.reset()

        for name, metric in ctx[MetricAggType.SUM].items():
            if metric.update_count > 0:  # Only compute if metric has been updated
                metrics[name] = metric.compute().item()
            metric.reset()

        # Only log if we have metrics to log
        if metrics:
            self._write_metrics(metrics, context=context)

        return metrics

    def _extract_metadata(self, global_step: int) -> Dict[str, int | float]:
        """Extract metadata fields with graceful error handling."""
        metadata: Dict[str, int | float] = {"global_step": global_step}
        for field_name, field_fn in self._metadata_fields.items():
            try:
                metadata[field_name] = field_fn()
            except Exception:
                metadata[field_name] = -1
        return metadata

    def _write_metrics(
        self, metrics: Dict[str, float], context: Optional[str] = None
    ) -> None:
        """Write metrics to TensorBoard and CSV files.

        Args:
            metrics: Metric names and values to write
            tag: Optional tag for separate CSV file
        """
        print("CALLED WRITE METRICS")
        # Capture global_step and metadata once
        global_step = self._global_step_fn()
        metadata = self._extract_metadata(global_step)

        # Write to TensorBoard and long-format CSV
        metadata_values = [
            metadata[field]
            for field in ["global_step"] + list(self._metadata_fields.keys())
        ]

        for name, value in metrics.items():
            # Log to TensorBoard
            self._tb_writer.add_scalar(name, value, global_step)
            # Write to long format CSV (one row per metric)
            self._csv_writer.writerow(metadata_values + [name, value])

        self._csv_file.flush()

        # Write to tagged CSV file if tag is provided (one row per flush with all metrics)
        if context and metrics:
            # Initialize tagged data accumulator if needed
            if context not in self._ctx_dataframes:
                self._ctx_dataframes[context] = []

            # Build complete row with metadata and all metrics
            row = metadata.copy()
            row.update(metrics)

            # Accumulate row for later pandas processing
            self._ctx_dataframes[context].append(row)

    def get_experiment_data(self) -> Dict[str, Any]:
        """Extract experiment timeline data for display purposes.

        Returns:
            Dictionary containing all tagged dataframes data organized by context.
            Format: {context: [list of metric rows with metadata]}
        """
        return dict(self._ctx_dataframes)

    def close_metrics(self) -> None:
        """Close TensorBoard writer and write CSV files.

        Note:
            Called automatically on exit. Safe to call multiple times.
        """

        # Close TensorBoard writer
        self._tb_writer.close()

        # Close CSV file
        self._csv_file.close()

        # Write all tagged DataFrames to CSV files
        for tag, rows in self._ctx_dataframes.items():
            if rows:  # Only write if we have data
                df = pd.DataFrame(rows)
                csv_path = os.path.join(self._metric_log_dir, f"metrics_{tag}.csv")
                df.to_csv(csv_path, index=False)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close_metrics()
