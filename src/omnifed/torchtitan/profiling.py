from __future__ import annotations

import csv
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import torch


TIMING_COLUMNS = [
    "time",
    "role",
    "rank",
    "client_id",
    "round",
    "iteration",
    "global_step",
    "phase",
    "seconds",
    "extra",
]


def _sync_gpu() -> None:
    """Wait for outstanding GPU work before reading the clock."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class TimingRecorder:
    def __init__(
        self,
        root: str | os.PathLike[str],
        role: str,
        rank: int,
    ):
        self.role = role
        self.rank = rank
        self.client_id = os.environ.get("CLIENT_ID", "")

        timing_root = Path(root) / "timing"

        if role == "client":
            if self.client_id == "":
                raise ValueError(
                    "CLIENT_ID must be set before creating a "
                    "client TimingRecorder"
                )

            self.path = (
                timing_root
                / f"client_{self.client_id}"
                / f"rank_{rank}.csv"
            )
        else:
            self.path = timing_root / f"{role}_rank_{rank}.csv"

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_file()

    def _initialize_file(self) -> None:
        """Create the CSV file and write its header."""
        if self.path.exists() and self.path.stat().st_size > 0:
            with open(self.path, newline="") as file:
                existing_header = next(csv.reader(file), [])

            if existing_header != TIMING_COLUMNS:
                raise RuntimeError(
                    f"Timing file {self.path} uses the old schema.\n"
                    f"Existing columns: {existing_header}\n"
                    f"Expected columns: {TIMING_COLUMNS}\n"
                    "Use a new job directory or rename the old timing file."
                )

            return

        with open(self.path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(TIMING_COLUMNS)

    def record(
        self,
        phase: str,
        seconds: float,
        round_id: int | str = "",
        iteration: int | str = "",
        global_step: int | str = "",
        extra: str = "",
    ) -> None:
        """Write one completed timing measurement."""
        with open(self.path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    self.role,
                    self.rank,
                    self.client_id,
                    round_id,
                    iteration,
                    global_step,
                    phase,
                    f"{seconds:.6f}",
                    extra,
                ]
            )

    @contextmanager
    def measure(
        self,
        phase: str,
        round_id: int | str = "",
        extra: str = "",
        *,
        iteration: int | str = "",
        global_step: int | str = "",
    ):
        """Measure a CPU/GPU operation and record its elapsed wall time."""
        _sync_gpu()
        start = time.perf_counter()

        try:
            yield
        finally:
            _sync_gpu()
            seconds = time.perf_counter() - start

            self.record(
                phase=phase,
                seconds=seconds,
                round_id=round_id,
                iteration=iteration,
                global_step=global_step,
                extra=extra,
            )


def _communication_category(event_name: str) -> str | None:
    """Classify a Torch profiler event as a communication operation."""
    name = event_name.lower()

    if "all_reduce" in name or "allreduce" in name:
        return "all_reduce"

    if "all_gather" in name or "allgather" in name:
        return "all_gather"

    if "reduce_scatter" in name or "reducescatter" in name:
        return "reduce_scatter"

    if "broadcast" in name:
        return "broadcast"

    if (
        "send" in name
        or "recv" in name
        or "sendrecv" in name
        or "send_recv" in name
    ):
        return "send_recv"

    # PyTorch uses NCCL names with ROCm/RCCL in several profiler outputs.
    if "nccl" in name or "rccl" in name:
        return "other_collective"

    return None


def _event_device_time_us(event) -> float:
    """Read device time from different PyTorch profiler versions."""
    for attribute in (
        "self_device_time_total",
        "device_time_total",
        "self_cuda_time_total",
        "cuda_time_total",
    ):
        value = getattr(event, attribute, None)

        if value is not None and float(value) > 0:
            return float(value)

    return 0.0


def _event_interval_us(event) -> tuple[float, float] | None:
    """Return an event's profiler-timeline interval in microseconds."""
    time_range = getattr(event, "time_range", None)
    if time_range is None:
        return None

    start = getattr(time_range, "start", None)
    end = getattr(time_range, "end", None)
    if start is None or end is None:
        return None

    start = float(start)
    end = float(end)
    if end <= start:
        return None
    return start, end


def _is_device_event(event) -> bool:
    """Identify GPU events without depending on one PyTorch enum version."""
    device_type = str(getattr(event, "device_type", "")).lower()
    return any(name in device_type for name in ("cuda", "hip", "privateuse1"))


def _event_communication_category(event) -> str | None:
    """Classify a GPU kernel, consulting its CPU parent chain if needed."""
    current = event
    visited: set[int] = set()
    generic_category: str | None = None

    for _ in range(12):
        if current is None or id(current) in visited:
            break
        visited.add(id(current))

        name = str(
            getattr(current, "name", "")
            or getattr(current, "key", "")
        )
        category = _communication_category(name)
        if category is not None and category != "other_collective":
            return category
        if category == "other_collective":
            generic_category = category

        current = getattr(current, "cpu_parent", None)

    return generic_category


def _merge_intervals(
    intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Merge overlapping timeline intervals."""
    if not intervals:
        return []

    ordered = sorted(intervals)
    merged: list[list[float]] = [[ordered[0][0], ordered[0][1]]]

    for start, end in ordered[1:]:
        previous = merged[-1]
        if start <= previous[1]:
            previous[1] = max(previous[1], end)
        else:
            merged.append([start, end])

    return [(start, end) for start, end in merged]


def _interval_duration_us(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals)


def _intersection_duration_us(
    first: list[tuple[float, float]],
    second: list[tuple[float, float]],
) -> float:
    """Measure the intersection of two already-merged interval lists."""
    i = 0
    j = 0
    duration = 0.0

    while i < len(first) and j < len(second):
        start = max(first[i][0], second[j][0])
        end = min(first[i][1], second[j][1])
        if end > start:
            duration += end - start

        if first[i][1] <= second[j][1]:
            i += 1
        else:
            j += 1

    return duration


@contextmanager
def profile_torchtitan_communication(
    timer: TimingRecorder,
    round_id: int,
    iteration: int,
    global_step: int,
    enabled: bool = True,
):
    """
    Profile GPU communication during one TorchTitan training iteration.

    This captures TP, FSDP and PP communication visible to torch.profiler.
    """
    if not enabled:
        yield
        return

    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]

    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    _sync_gpu()

    with profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        yield

    _sync_gpu()

    # key_averages() sums event durations and can double count concurrent or
    # nested work. Keep that diagnostic separately from the wall-time metrics.
    totals_us: dict[str, float] = defaultdict(float)
    event_counts: dict[str, int] = defaultdict(int)

    for event in profiler.key_averages():
        category = _communication_category(event.key)

        if category is None:
            continue

        device_time_us = _event_device_time_us(event)

        # Device time is preferred. CPU time is only a fallback when the
        # profiler did not expose a device duration for this event.
        if torch.cuda.is_available():
            # On Frontier/ROCm, report GPU collective time.
            # Ignore CPU-only collective launch events.
            measured_us = device_time_us
        else:
            # CPU fallback for environments without a GPU.
            measured_us = float(
                getattr(event, "self_cpu_time_total", 0.0)
            )

        if measured_us <= 0:
            continue

        totals_us[category] += measured_us
        event_counts[category] += int(
            getattr(event, "count", 1)
        )

    total_communication_us = sum(totals_us.values())

    timer.record(
        phase="torchtitan_iteration_communication_accumulated",
        seconds=total_communication_us / 1_000_000.0,
        round_id=round_id,
        iteration=iteration,
        global_step=global_step,
        extra=f"num_categories={len(totals_us)}",
    )

    for category, microseconds in sorted(totals_us.items()):
        timer.record(
            phase=f"torchtitan_iteration_{category}_accumulated",
            seconds=microseconds / 1_000_000.0,
            round_id=round_id,
            iteration=iteration,
            global_step=global_step,
            extra=f"count={event_counts[category]}",
        )

    # Raw GPU intervals provide non-double-counted wall time. Communication
    # and compute are merged separately, then intersected to measure overlap.
    communication_intervals: list[tuple[float, float]] = []
    compute_intervals: list[tuple[float, float]] = []
    category_intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for event in profiler.events():
        if not _is_device_event(event):
            continue

        interval = _event_interval_us(event)
        if interval is None:
            continue

        category = _event_communication_category(event)
        if category is None:
            compute_intervals.append(interval)
        else:
            communication_intervals.append(interval)
            category_intervals[category].append(interval)

    merged_communication = _merge_intervals(communication_intervals)
    merged_compute = _merge_intervals(compute_intervals)
    communication_active_us = _interval_duration_us(merged_communication)
    compute_active_us = _interval_duration_us(merged_compute)
    overlap_us = _intersection_duration_us(
        merged_communication,
        merged_compute,
    )
    exposed_us = max(0.0, communication_active_us - overlap_us)
    overlap_ratio = (
        overlap_us / communication_active_us
        if communication_active_us > 0
        else 0.0
    )

    wall_metrics = {
        "torchtitan_iteration_communication_active_wall": communication_active_us,
        "torchtitan_iteration_communication_overlapped_wall": overlap_us,
        "torchtitan_iteration_communication_exposed_wall": exposed_us,
        "torchtitan_iteration_compute_active_wall": compute_active_us,
    }
    common_extra = (
        f"comm_intervals={len(communication_intervals)};"
        f"comm_merged={len(merged_communication)};"
        f"compute_intervals={len(compute_intervals)};"
        f"overlap_ratio={overlap_ratio:.6f}"
    )

    for phase, microseconds in wall_metrics.items():
        timer.record(
            phase=phase,
            seconds=microseconds / 1_000_000.0,
            round_id=round_id,
            iteration=iteration,
            global_step=global_step,
            extra=common_extra,
        )

    for category, intervals in sorted(category_intervals.items()):
        merged = _merge_intervals(intervals)
        timer.record(
            phase=f"torchtitan_iteration_{category}_active_wall",
            seconds=_interval_duration_us(merged) / 1_000_000.0,
            round_id=round_id,
            iteration=iteration,
            global_step=global_step,
            extra=f"events={len(intervals)};merged_intervals={len(merged)}",
        )
