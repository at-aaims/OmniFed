import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Literal, Callable, Set
import numpy as np
from rich import box, print
from rich.table import Table
from .rich_helpers import print_rule

DIM_DASH = "[dim]-[/dim]"

Position = Tuple[Optional[int], Optional[int], Optional[int], int]


@dataclass(frozen=True)
class TableStyles:
    """Terminal color constants for experiment result display."""

    VERY_GOOD: str = "bold bright_green"
    GOOD: str = "bright_yellow"
    MODERATE: str = "orange1"
    BAD: str = "orange_red1"
    VERY_BAD: str = "bold bright_red"

    NEUTRAL: str = "white"
    SUBTLE: str = "dim white"

    ROUND: str = "bright_magenta"
    EPOCH: str = "bright_blue"
    BATCH: str = "bright_green"
    STEP: str = "bright_yellow"


COLORS = TableStyles()


def _pct(numerator: float, denominator: float) -> float:
    return (numerator / denominator * 100.0) if denominator > 0 else 0.0


class MetricType(Enum):
    LOSS = "Loss & Error"
    PERFORMANCE = "Performance"
    TIME = "Timing"
    COUNT = "Dataset"
    PROGRESS = "Progress"
    LOCAL_AGG = "Local Agg"
    GLOBAL_AGG = "Global Agg"
    LOCAL_BCAST = "Local Bcast"
    OTHER = "Other"


class ProgressionPattern(Enum):
    ROUND_DOMINANT = "round_dominant"
    EPOCH_DOMINANT = "epoch_dominant"
    BATCH_DOMINANT = "batch_dominant"

    ROUND_EPOCH = "round_epoch"
    ROUND_BATCH = "round_batch"
    EPOCH_BATCH = "epoch_batch"

    BALANCED = "balanced"
    NONE = "none"
    MIXED = "mixed"


@dataclass(frozen=True)
class ThresholdSpec:
    thresholds: Dict[float, str]

    def get_value_for_threshold(self, value: float) -> str:
        for threshold in sorted(self.thresholds.keys(), reverse=True):
            if value >= threshold:
                return self.thresholds[threshold]
        return list(self.thresholds.values())[-1]


@dataclass(frozen=True)
class DisplayConfig:
    META_KEYS: frozenset[str] = frozenset(
        {"global_step", "round_idx", "epoch_idx", "batch_idx", "_node_id"}
    )
    DEFAULT_PRECISION: int = 4
    PERCENTAGE_CHANGE_PRECISION: int = 1

    EPSILON_THRESHOLD: float = 1e-10
    ZERO_LOG_VALUE: float = -10.0
    MIN_DATA_POINTS_REQUIRED: int = 2

    PATTERN_SIGNIFICANCE_THRESHOLD: float = 0.5
    DEFAULT_MAX_DATA_POINTS: int = 10
    STANDARD_SAMPLING_FRACTIONS: Tuple[float, ...] = (0.25, 0.5, 0.75)

    PARTICIPATION_THRESHOLDS: ThresholdSpec = field(
        default_factory=lambda: ThresholdSpec(
            {
                99.0: COLORS.VERY_GOOD,
                90.0: COLORS.GOOD,
                75.0: COLORS.MODERATE,
                50.0: COLORS.BAD,
                25.0: COLORS.VERY_BAD,
            }
        )
    )

    PARTICIPATION_ICONS: ThresholdSpec = field(
        default_factory=lambda: ThresholdSpec(
            {
                99.0: "🟢",
                90.0: "🟡",
                75.0: "🟠",
                50.0: "🔴",
                25.0: "🚨",
            }
        )
    )

    SAMPLING_ICONS: ThresholdSpec = field(
        default_factory=lambda: ThresholdSpec(
            {
                90.0: "🟢",
                75.0: "🟡",
                50.0: "🟠",
                25.0: "🔵",
                0.0: "📊",
            }
        )
    )

    CHANGE_IMPROVEMENT_THRESHOLDS: ThresholdSpec = field(
        default_factory=lambda: ThresholdSpec(
            {
                0.01: "green",
                0.0: COLORS.NEUTRAL,
            }
        )
    )

    CHANGE_REGRESSION_THRESHOLDS: ThresholdSpec = field(
        default_factory=lambda: ThresholdSpec(
            {
                0.01: "red",
                0.0: COLORS.NEUTRAL,
            }
        )
    )


CONFIG = DisplayConfig()


@dataclass(frozen=True)
class MetricRule:
    pattern: str
    precision: int = CONFIG.DEFAULT_PRECISION
    units: str = ""
    minimize: bool = False
    emoji: str = ":bar_chart:"
    metric_type: MetricType = MetricType.OTHER
    valid_range: Optional[Tuple[Optional[float], Optional[float]]] = None

    show_cv: bool = True
    show_median: bool = False
    show_sum: bool = False

    cv_thresholds: ThresholdSpec = field(
        default_factory=lambda: ThresholdSpec(
            {
                50.0: COLORS.VERY_BAD,
                25.0: COLORS.BAD,
                10.0: COLORS.MODERATE,
                2.5: COLORS.GOOD,
                0.0: COLORS.VERY_GOOD,
            }
        )
    )
    outlier_thresholds: ThresholdSpec = field(
        default_factory=lambda: ThresholdSpec(
            {
                3.0: COLORS.VERY_BAD,
                2.0: COLORS.BAD,
                1.0: COLORS.MODERATE,
                0.0: COLORS.NEUTRAL,
            }
        )
    )

    def is_valid_value(self, value: float) -> bool:
        if math.isnan(value) or math.isinf(value):
            return False

        if self.valid_range is None:
            return True

        min_val, max_val = self.valid_range
        return (min_val is None or value >= min_val) and (
            max_val is None or value <= max_val
        )


METRIC_RULES = [
    MetricRule(
        r"(loss|error|mse|mae|rmse)",
        precision=4,
        minimize=True,
        emoji=":chart_decreasing:",
        metric_type=MetricType.LOSS,
        show_median=True,
    ),
    MetricRule(
        r"(accuracy|precision|recall|f1)",
        precision=4,
        minimize=False,
        emoji=":dart:",
        metric_type=MetricType.PERFORMANCE,
        valid_range=(0.0, 1.0),
        show_median=True,
    ),
    MetricRule(
        r"(count|total|num_)",
        precision=0,
        emoji=":package:",
        metric_type=MetricType.COUNT,
        valid_range=(0.0, None),
        show_cv=False,
        show_sum=True,
    ),
    MetricRule(
        r"time",
        precision=4,
        units="s",
        minimize=True,
        emoji="",
        metric_type=MetricType.TIME,
        valid_range=(0.0, None),
    ),
MetricRule(
        r"local_agg",
        precision=4,
        emoji=":arrow_up:",
        metric_type=MetricType.LOCAL_AGG,
        show_cv=False,
    ),
    MetricRule(
        r"global_agg",
        precision=4,
        emoji=":globe_with_meridians:",
        metric_type=MetricType.GLOBAL_AGG,
        show_cv=False,
    ),
    MetricRule(
        r"local_bcast",
        precision=4,
        emoji=":arrow_down:",
        metric_type=MetricType.LOCAL_BCAST,
        show_cv=False,
    ),
    MetricRule(
        r"(progress|completion|percent)",
        precision=1,
        units="%",
        emoji=":chart_increasing:",
        metric_type=MetricType.PROGRESS,
        valid_range=(0.0, 100.0),
        show_cv=False,
    ),
]

DEFAULT_RULE = MetricRule(pattern=".*", emoji=":bar_chart:")


def detect_outliers(values: List[float]) -> List[int]:
    """Return indices of values that deviate >1.0 in log10 space from median."""
    if len(values) < CONFIG.MIN_DATA_POINTS_REQUIRED:
        return []
    log_arr = np.array(
        [math.log10(abs(v)) if v != 0 else CONFIG.ZERO_LOG_VALUE for v in values]
    )
    median = float(np.median(log_arr))
    diff = np.abs(log_arr - median)
    return [int(i) for i in np.where(diff > 1.0)[0]]


class DisplayFormatter:
    """Formats experiment metrics for terminal display with appropriate styling."""

    def __init__(self):
        self._rule_cache: Dict[str, MetricRule] = {}

    def format_text(self, text: str, color: str = "white", style: str = "") -> str:
        tag = f"{style} {color}" if style else color
        return f"[{tag}]{text}[/{tag}]"

    def find_rule(self, metric_name: str) -> MetricRule:
        """Return the formatting rule for the given metric name."""
        if metric_name in self._rule_cache:
            return self._rule_cache[metric_name]

        for rule in METRIC_RULES:
            try:
                if re.search(rule.pattern, metric_name, re.IGNORECASE):
                    self._rule_cache[metric_name] = rule
                    return rule
            except re.error as e:
                raise ValueError(
                    f"Invalid regex pattern '{rule.pattern}' for metric '{metric_name}': {e}"
                ) from e
        self._rule_cache[metric_name] = DEFAULT_RULE
        return DEFAULT_RULE

    def validate_and_format(self, metric_name: str, value: float) -> str:
        """Format numeric value according to its metric type with precision and units."""
        if np.isnan(value) or np.isinf(value):
            text = f"❌ {str(value)}"
            return self.format_text(text, COLORS.VERY_BAD)

        rule = self.find_rule(metric_name)

        if rule.metric_type == MetricType.COUNT:
            if not (math.isfinite(value) and value >= 0 and value < 1e15):
                return (
                    self.format_text(f"{value:.0f}{rule.units}", COLORS.MODERATE)
                    + " :warning:"
                )
            return f"{int(round(value)):,}{rule.units}"
        return f"{value:.{rule.precision}f}{rule.units}"

    def get_pct_change_color(
        self, metric_name: str, delta: float, prev_value: Optional[float] = None
    ) -> str:
        """Return color code based on whether change represents improvement or regression."""
        if np.isnan(delta) or np.isinf(delta):
            return COLORS.SUBTLE

        rule = self.find_rule(metric_name)
        is_improvement = (delta < 0 and rule.minimize) or (
            delta > 0 and not rule.minimize
        )

        if prev_value is not None and abs(prev_value) >= CONFIG.EPSILON_THRESHOLD:
            abs_pct_change = abs(delta / prev_value * 100) if prev_value != 0 else 0
            thresholds = (
                CONFIG.CHANGE_IMPROVEMENT_THRESHOLDS
                if is_improvement
                else CONFIG.CHANGE_REGRESSION_THRESHOLDS
            )
            return thresholds.get_value_for_threshold(abs_pct_change)
        return COLORS.VERY_GOOD if is_improvement else COLORS.VERY_BAD

    def format_cell(
        self,
        metric_name: str,
        value: float,
        color: str = COLORS.NEUTRAL,
        validate_range: bool = True,
        context_values: Optional[List[float]] = None,
    ) -> str:
        """Apply Rich formatting to metric value for table display."""
        if np.isnan(value) or np.isinf(value):
            return self.validate_and_format(metric_name, value)

        formatted_value = self.validate_and_format(metric_name, value)

        if validate_range and not self.find_rule(metric_name).is_valid_value(value):
            return self.format_text(formatted_value, COLORS.VERY_BAD)

        if context_values and len(context_values) >= CONFIG.MIN_DATA_POINTS_REQUIRED:
            try:
                value_index = context_values.index(value)
                outlier_indices = detect_outliers(context_values)
                if value_index in outlier_indices:
                    deviation = MetricStats.compute_log_deviation(
                        context_values, [value_index]
                    )
                    color = self.find_rule(
                        metric_name
                    ).outlier_thresholds.get_value_for_threshold(deviation)
            except ValueError:
                pass

        return self.format_text(formatted_value, color)

    def format_cv_cell(self, cv_value: float, thresholds: ThresholdSpec) -> str:
        """Format coefficient of variation percentage with threshold-based coloring."""
        if np.isnan(cv_value) or np.isinf(cv_value):
            text = f"❌ {str(cv_value)}"
            return self.format_text(text, COLORS.VERY_BAD)

        return self.format_text(
            f"{cv_value:.1f}%", thresholds.get_value_for_threshold(cv_value)
        )

    def format_outlier_cell(
        self, outlier_count: int, deviation: float, thresholds: ThresholdSpec
    ) -> str:
        """Format outlier count with severity-based coloring."""
        return (
            self.format_text("-", "dim")
            if outlier_count == 0
            else self.format_text(
                str(outlier_count), thresholds.get_value_for_threshold(deviation)
            )
        )

    def format_participation_cell(
        self,
        reporting_nodes: int,
        total_nodes: int,
        none_count: int = 0,
        description: str = "nodes",
    ) -> str:
        """Format node participation statistics with icons and percentage."""
        coverage_pct = _pct(reporting_nodes, total_nodes)
        icon = CONFIG.PARTICIPATION_ICONS.get_value_for_threshold(coverage_pct)
        base_display = self.format_text(
            f"{reporting_nodes}/{total_nodes} {description} ({coverage_pct:.0f}%)",
            CONFIG.PARTICIPATION_THRESHOLDS.get_value_for_threshold(coverage_pct),
        )
        formatted_with_icon = f"{icon} {base_display}"
        return (
            f"{formatted_with_icon} [dim]({none_count} None)[/dim]"
            if none_count > 0
            else formatted_with_icon
        )

    def format_pct_change_cell(
        self, pct_change: float, metric_name: str, prev_value: float
    ) -> str:
        """Format percentage change with directional coloring."""
        delta = pct_change / 100 * prev_value
        color = self.get_pct_change_color(metric_name, delta, prev_value)
        return self.format_text(
            f"{pct_change:+5.{CONFIG.PERCENTAGE_CHANGE_PRECISION}f}%", color, "italic"
        )

    def format_participation_caption(
        self, complete_count: int, total_count: int, context: str
    ) -> str:
        """Generate caption describing node participation completeness."""
        participation_info = self.format_participation_cell(
            complete_count, total_count, description="participants"
        )
        return f"{participation_info} reported all metrics in {context} context"

    def format_position_header(
        self, position: Tuple[Optional[int], Optional[int], Optional[int], int]
    ) -> str:
        """Format multi-line table header showing round/epoch/batch/step values."""
        round_idx, epoch_idx, batch_idx, global_step = position

        position_parts = [
            (round_idx, "R", COLORS.ROUND),
            (epoch_idx, "E", COLORS.EPOCH),
            (batch_idx, "B", COLORS.BATCH),
            (global_step, "S", COLORS.STEP),
        ]
        header_parts = [
            self.format_text(f"{prefix}{value}", color)
            for value, prefix, color in position_parts
            if value is not None
        ]
        return "\n".join(header_parts) if header_parts else "Unknown"


class MetricStats:
    """Statistical computations for experiment metric arrays."""

    @dataclass(frozen=True)
    class _ComputedStats:
        """Cached statistical computations for metric values."""

        mean: float
        std: float
        min_val: float
        max_val: float
        median: float
        cv: float

    @dataclass(frozen=True)
    class CountData:
        """Counts of data quality issues (NaN, infinite, invalid, outliers)."""

        nan_count: int
        inf_count: int
        invalid_count: int
        outlier_count: int

    def __init__(
        self,
        values: List[float],
        metric_name: Optional[str] = None,
        formatter: Optional[DisplayFormatter] = None,
    ):
        self.values = values
        self.metric_name = metric_name
        self.formatter = formatter

        self._clean_values: List[float] = []
        self._nan_indices: List[int] = []
        self._inf_indices: List[int] = []
        self._invalid_indices: List[int] = []

        for idx, value in enumerate(values):
            if np.isnan(value):
                self._nan_indices.append(idx)
            elif np.isinf(value):
                self._inf_indices.append(idx)
            else:
                self._clean_values.append(value)
                if metric_name and formatter:
                    rule = formatter.find_rule(metric_name)
                    if not rule.is_valid_value(value):
                        self._invalid_indices.append(len(self._clean_values) - 1)

        self._outlier_indices = None
        self._basic_stats = None

    def compute_stats(self) -> _ComputedStats:
        """Calculate mean, std, min, max, median, and CV with caching."""
        if self._basic_stats is None:
            if not self._clean_values:
                self._basic_stats = self._ComputedStats(
                    mean=0.0, std=0.0, min_val=0.0, max_val=0.0, median=0.0, cv=0.0
                )
            else:
                arr = np.array(self._clean_values)
                mean_val = float(np.mean(arr))
                std_val = float(np.std(arr))
                cv_val = _pct(std_val, abs(mean_val)) if mean_val else 0.0

                self._basic_stats = self._ComputedStats(
                    mean=mean_val,
                    std=std_val,
                    min_val=float(np.min(arr)),
                    max_val=float(np.max(arr)),
                    median=float(np.median(arr)),
                    cv=cv_val,
                )
        return self._basic_stats

    @property
    def outlier_indices(self) -> List[int]:
        """Indices of values identified as statistical outliers."""
        if self._outlier_indices is None:
            self._outlier_indices = (
                detect_outliers(self._clean_values)
                if len(self._clean_values) >= CONFIG.MIN_DATA_POINTS_REQUIRED
                else []
            )
        return self._outlier_indices

    @staticmethod
    def compute_log_deviation(values: List[float], indices: List[int]) -> float:
        """Maximum absolute deviation in log10 space from median for specified indices."""
        if not values or not indices:
            return 0.0
        log_values = [
            math.log10(abs(v)) if v != 0 else CONFIG.ZERO_LOG_VALUE for v in values
        ]
        log_median = float(np.median(log_values))
        return max(abs(log_values[i] - log_median) for i in indices)

    @property
    def counts(self) -> CountData:
        """Data quality issue counts for this metric."""
        return self.CountData(
            nan_count=len(self._nan_indices),
            inf_count=len(self._inf_indices),
            invalid_count=len(self._invalid_indices),
            outlier_count=len(self.outlier_indices),
        )


@dataclass(frozen=True)
class Measurement:
    """Metric values from one node at a specific experiment step."""

    global_step: int
    round_idx: Optional[int]
    epoch_idx: Optional[int]
    batch_idx: Optional[int]
    node_id: int
    metrics: Dict[str, Any]

    @staticmethod
    def from_raw(raw: Dict[str, Any], node_id: int) -> "Measurement":
        """Parse raw experiment data into structured measurement object."""
        return Measurement(
            global_step=raw["global_step"],
            round_idx=raw["round_idx"],
            epoch_idx=raw["epoch_idx"],
            batch_idx=raw["batch_idx"],
            node_id=node_id,
            metrics={k: v for k, v in raw.items() if k not in CONFIG.META_KEYS},
        )

    @property
    def position(self) -> Position:
        """Experiment position as (round, epoch, batch, global_step) tuple."""
        return (self.round_idx, self.epoch_idx, self.batch_idx, self.global_step)


@dataclass(frozen=True)
class StatDisplay:
    mean: str
    std: str
    median: str
    min: str
    max: str
    sum: str
    cv: str


@dataclass(frozen=True)
class TableColumn:
    """Rich table column definition with header, justification, and styling."""

    key: str
    header: str
    justify: Literal["left", "center", "right", "full", "default"] = "right"
    style: str = ""


class TableFactory:
    """Creates Rich tables with consistent styling for experiment results."""

    STATS_COLUMNS = (
        TableColumn("coverage", ":busts_in_silhouette: Coverage", "center"),
        TableColumn("bad_values", ":cross_mark: Corrupt", "center"),
        TableColumn("invalid", ":prohibited: Invalid", "center"),
        TableColumn("outliers", ":mag: Outliers", "center"),
        TableColumn("sum", ":heavy_plus_sign: Sum", "right"),
        TableColumn("mean", ":bar_chart: Mean", "right"),
        TableColumn("std", ":straight_ruler: Std", "right"),
        TableColumn("median", ":bar_chart: Median", "right"),
        TableColumn("min", ":red_triangle_pointed_down: Min", "right"),
        TableColumn("max", ":red_triangle_pointed_up: Max", "right"),
        TableColumn("cv", ":chart_increasing: CV", "right"),
    )

    @staticmethod
    def create_base_table(
        title: str,
        caption: Optional[str] = None,
        title_style: str = "bold",
        caption_style: str = "bright_white",
    ) -> Table:
        return Table(
            title=title,
            title_justify="left",
            caption=caption,
            caption_justify="right",
            caption_style=caption_style,
            box=box.ROUNDED,
            title_style=title_style,
            show_header=True,
        )

    @staticmethod
    def create_summary_table(title: str) -> Table:
        """Build two-column table for high-level experiment statistics."""
        table = TableFactory.create_base_table(title)
        table.add_column(":bar_chart: Metric", style="bold", justify="left")
        table.add_column(":clipboard: Value", justify="center")
        return table

    @staticmethod
    def create_stats_table(
        context: str,
        caption: Optional[str] = None,
        node_participation: Optional[str] = None,
    ) -> Table:
        """Build multi-column statistics table for final step metric analysis."""
        base_title = (
            f":busts_in_silhouette: {context.title()} Metrics - Node Statistics"
        )
        title = (
            f"{base_title} {node_participation}" if node_participation else base_title
        )
        table = TableFactory.create_base_table(title, caption)

        table.add_column(":bar_chart: Metric", style="bold", justify="left")

        for col in TableFactory.STATS_COLUMNS:
            table.add_column(col.header, justify=col.justify, style=col.style)

        return table


class ResultsDisplay:
    """Primary interface for formatting and displaying federated learning experiment results."""

    GROUP_PRIORITY = [
        MetricType.PROGRESS,
        MetricType.LOSS,
        MetricType.PERFORMANCE,
        MetricType.COUNT,
        MetricType.TIME,
        MetricType.LOCAL_AGG,
        MetricType.GLOBAL_AGG,
        MetricType.LOCAL_BCAST,
        MetricType.OTHER,
    ]

    def __init__(self):
        self.formatter = DisplayFormatter()

    def _metric_label(self, metric: str, rule: MetricRule) -> str:
        """Format metric name with type-specific emoji prefix."""
        return self.formatter.format_text(f"{rule.emoji} {metric}", COLORS.NEUTRAL)

    def _add_group_header(
        self, table: Table, group_name: str, count: int, span: int
    ) -> None:
        """Insert metric type header row (e.g., 'Loss & Error (3)')."""
        table.add_row(
            f"{self.formatter.format_text(group_name, 'blue', 'bold')} {self.formatter.format_text(f'({count})', 'white', 'dim')}",
            *[""] * span,
        )

    def _iterate_metric_groups(
        self,
        table: Table,
        metric_groups: Dict[str, List[str]],
        span: int,
        add_metric_fn: Callable[[str], None],
    ) -> None:
        """Process metric groups with headers and delegate row creation to callback."""
        multi = len(metric_groups) > 1
        for i, (group_name, group_metrics) in enumerate(metric_groups.items()):
            if i > 0:
                table.add_section()
            if multi:
                self._add_group_header(table, group_name, len(group_metrics), span)
            for metric in group_metrics:
                add_metric_fn(metric)

    def _add_no_data_metric_row(
        self,
        table: Table,
        metric: str,
        participating_nodes: int,
        reported: int,
    ) -> None:
        """Insert error row for metrics with no valid numeric data."""
        coverage_pct = (
            _pct(reported, participating_nodes) if participating_nodes else 0.0
        )
        coverage_str = self.formatter.format_text(
            f"{reported}/{participating_nodes} ({coverage_pct:.0f}%)", COLORS.VERY_BAD
        )
        no_data_text = self.formatter.format_text(
            ":rotating_light: No valid data", COLORS.VERY_BAD
        )

        # Fill remaining columns with "No Data"
        remaining_cols = len(TableFactory.STATS_COLUMNS) - 2
        filler = [
            self.formatter.format_text("No Data", COLORS.VERY_BAD)
        ] * remaining_cols

        table.add_row(f":bar_chart: {metric}", coverage_str, no_data_text, *filler)

    def show_experiment_results(
        self,
        results: List[Dict[str, Any]],
        duration: float,
        global_rounds: int,
        total_nodes: int,
    ) -> None:
        """Render complete federated learning experiment results to terminal."""
        self._show_summary(results, duration, global_rounds, total_nodes)

        for context in self._get_contexts(results):
            print_rule(f"{context.title()} Results", characters="-")
            print()
            self._show_context_results(context, results)

    def _show_summary(
        self,
        results: List[Dict[str, Any]],
        duration: float,
        global_rounds: int,
        total_nodes: int,
    ) -> None:
        """Display high-level experiment statistics and node participation."""
        context_coverage = self._calculate_context_coverage(results, total_nodes)
        final_step_completion = self._calculate_final_step_completion(
            results, total_nodes
        )

        table = TableFactory.create_summary_table("Experiment Summary")
        table.add_row(":wrench: Nodes Configured", str(total_nodes))

        if final_step_completion[0] is not None:
            table.add_row(
                ":trophy: Final Step Reached",
                self.formatter.format_participation_cell(
                    final_step_completion[0], total_nodes
                ),
            )

        for context, (nodes_with_data, _) in context_coverage.items():
            table.add_row(
                f":bar_chart: Node Participation - {context.title()}",
                self.formatter.format_participation_cell(nodes_with_data, total_nodes),
            )

        table.add_row(":arrows_counterclockwise: Total Rounds", str(global_rounds))
        table.add_row(":timer_clock:  Duration", f"{duration:.2f}s")
        print(table)
        print()

    def _calculate_context_coverage(
        self, results: List[Dict[str, Any]], total_nodes: int
    ) -> Dict[str, Tuple[int, float]]:
        context_coverage: Dict[str, Tuple[int, float]] = {}
        for context in self._get_contexts(results):
            nodes_with_data = sum(
                1
                for node_data in results
                if context in node_data and node_data[context]
            )
            coverage_pct = (
                _pct(nodes_with_data, total_nodes) if total_nodes > 0 else 0.0
            )
            context_coverage[context] = (nodes_with_data, coverage_pct)
        return context_coverage

    def _calculate_final_step_completion(
        self, results: List[Dict[str, Any]], total_nodes: int
    ) -> Tuple[Optional[int], float]:
        node_steps: Dict[int, int] = {}
        for node_index, node_data in enumerate(results):
            max_step = 0
            for context_data in node_data.values():
                if isinstance(context_data, list):
                    for m in context_data:
                        if "global_step" in m:
                            step_val = m["global_step"]
                            if (
                                isinstance(step_val, (int, float))
                                and step_val > max_step
                            ):
                                max_step = int(step_val)
            if max_step > 0:
                node_steps[node_index] = max_step

        if not node_steps:
            return (None, 0.0)

        final_step = max(node_steps.values())
        nodes_at_final_step = sum(
            1 for step in node_steps.values() if step == final_step
        )
        completion_pct = (
            _pct(nodes_at_final_step, total_nodes) if total_nodes > 0 else 0.0
        )
        return (nodes_at_final_step, completion_pct)

    def _show_context_results(self, context: str, results: List[Dict[str, Any]]):
        """Display statistics and progression tables for one context (train/eval)."""
        measurements = self._extract_measurements(context, results)
        if not measurements:
            print(f":warning:  No data available for {context}")
            return

        final_measurements = self._get_final_measurements(measurements)
        if not final_measurements:
            print(f":warning:  No final step data for {context}")
            return

        total_nodes = len(results)
        self._show_statistics_table(context, final_measurements, total_nodes)
        print()
        self._show_progression_table(context, measurements, total_nodes)
        print()

    def _extract_measurements(
        self, context: str, results: List[Dict[str, Any]]
    ) -> List[Measurement]:
        """Parse raw node results into structured Measurement objects for given context."""
        measurements = []
        skipped_nodes = 0
        invalid_data_nodes = 0

        for node_id, node_data in enumerate(results):
            if context not in node_data:
                skipped_nodes += 1
                continue
            context_data = node_data[context]
            if not context_data:
                skipped_nodes += 1
                continue

            if not isinstance(context_data, list):
                invalid_data_nodes += 1
                print(
                    f":warning:  Node {node_id}: Invalid data type for context '{context}' (expected list, got {type(context_data).__name__})"
                )
                continue

            for raw_measurement in context_data:
                measurements.append(Measurement.from_raw(raw_measurement, node_id))

        total_nodes = len(results)
        if skipped_nodes > 0:
            print(
                f":information_source:  Context '{context}': {skipped_nodes}/{total_nodes} nodes had no data"
            )
        if invalid_data_nodes > 0:
            print(
                f":warning:  Context '{context}': {invalid_data_nodes}/{total_nodes} nodes had invalid data format"
            )

        return measurements

    def _get_final_measurements(
        self, measurements: List[Measurement]
    ) -> List[Measurement]:
        """Filter measurements to only include those from the highest global_step."""
        if not measurements:
            return []

        final_step = max(m.global_step for m in measurements)
        return [m for m in measurements if m.global_step == final_step]

    def _show_statistics_table(
        self, context: str, measurements: List[Measurement], total_nodes: int
    ) -> None:
        """Display final-step statistics with coverage, data quality, and metric summaries."""
        metrics = self._get_metrics(measurements)
        if not metrics:
            return

        node_count = self._count_nodes(measurements)
        final_step = max(m.global_step for m in measurements) if measurements else 0
        nodes_at_final_step = len(
            {m.node_id for m in measurements if m.global_step == final_step}
        )

        # Calculate metric coverage percentages
        metric_coverages = [
            _pct(
                self._count_nodes(measurements, lambda m: metric in m.metrics),
                node_count,
            )
            if node_count > 0
            else 0
            for metric in metrics
        ]

        min_metric_coverage = min(metric_coverages) if metric_coverages else 0
        complete_nodes = (
            node_count
            if min_metric_coverage == 100
            else int(round(node_count * min_metric_coverage / 100))
        )

        # Create caption lines
        participation_line = f"{self.formatter.format_participation_cell(node_count, total_nodes, description='nodes')} participated in {context} metric context"
        metric_completeness_line = self.formatter.format_participation_caption(
            complete_nodes, node_count, context
        )

        # Final step line with position context
        final_step_info = self.formatter.format_participation_cell(
            nodes_at_final_step, node_count, description="participants"
        )
        final_measurements = [m for m in measurements if m.global_step == final_step]

        if final_measurements:
            position_text = self.formatter.format_position_header(
                final_measurements[0].position
            ).replace("\n", " ")
            final_step_line = (
                f"{final_step_info} reached final position {position_text}"
            )
        else:
            final_step_line = f"{final_step_info} reached final step {final_step}"

        combined_caption = (
            f"{participation_line}\n{metric_completeness_line}\n{final_step_line}"
        )
        table = TableFactory.create_stats_table(context, combined_caption)

        metric_groups = self._group_metrics(metrics)

        self._iterate_metric_groups(
            table,
            metric_groups,
            len(TableFactory.STATS_COLUMNS),
            lambda metric: self._add_metric_row(
                table, metric, measurements, node_count
            ),
        )

        print(table)

    def _get_metrics(self, measurements: List[Measurement]) -> List[str]:
        """Extract unique metric names from measurement data, excluding metadata fields."""
        metrics: Set[str] = set()
        for m in measurements:
            metrics.update(m.metrics.keys())
        return sorted(metrics - CONFIG.META_KEYS)

    def _group_metrics(self, metrics: List[str]) -> Dict[str, List[str]]:
        """Organize metrics by type (Loss, Performance, Time, etc.) for display grouping."""
        groups = defaultdict(list)

        for metric in metrics:
            rule = self.formatter.find_rule(metric)
            group_name = rule.metric_type.value
            groups[group_name].append(metric)

        return {
            group_type.value: sorted(groups[group_type.value])
            for group_type in self.GROUP_PRIORITY
            if group_type.value in groups
        }

    def _calculate_coverage_cell(
        self,
        metric: str,
        measurements: List[Measurement],
        participating_nodes: int,
        none_count: int,
    ) -> str:
        """Format participation cell showing nodes reporting this metric."""
        reporting_nodes = self._count_nodes(measurements, lambda m: metric in m.metrics)
        return self.formatter.format_participation_cell(
            reporting_nodes, participating_nodes, none_count
        )

    def _format_statistics(
        self, metric: str, stats: MetricStats, values: List[float]
    ) -> StatDisplay:
        """Generate formatted statistical summary cells for table display."""
        computed_stats = stats.compute_stats()
        rule = self.formatter.find_rule(metric)

        def _format_extreme(value: float) -> str:
            base = self.formatter.validate_and_format(metric, value)
            if not rule.is_valid_value(value):
                return self.formatter.format_text(base, COLORS.VERY_BAD)
            color = COLORS.NEUTRAL
            if len(values) >= CONFIG.MIN_DATA_POINTS_REQUIRED:
                try:
                    idx = values.index(value)
                except ValueError:
                    idx = -1
                if idx >= 0:
                    outlier_indices = stats.outlier_indices
                    if idx in outlier_indices:
                        deviation = MetricStats.compute_log_deviation(values, [idx])
                        color = rule.outlier_thresholds.get_value_for_threshold(
                            deviation
                        )
            return self.formatter.format_text(base, color)

        mean_str = self.formatter.format_cell(metric, computed_stats.mean)

        std_str = (
            self.formatter.format_cell(
                metric, computed_stats.std, COLORS.NEUTRAL, False
            )
            if len(values) > 1
            else DIM_DASH
        )

        min_str = _format_extreme(computed_stats.min_val)
        max_str = _format_extreme(computed_stats.max_val)

        cv_str = (
            self.formatter.format_cv_cell(computed_stats.cv, rule.cv_thresholds)
            if rule.show_cv and len(values) > 1
            else DIM_DASH
        )

        median_str = (
            _format_extreme(computed_stats.median)
            if rule.show_median and values
            else DIM_DASH
        )

        sum_str = (
            self.formatter.format_cell(metric, sum(values))
            if rule.show_sum and values
            else DIM_DASH
        )

        return StatDisplay(
            mean=mean_str,
            std=std_str,
            median=median_str,
            min=min_str,
            max=max_str,
            sum=sum_str,
            cv=cv_str,
        )

    def _add_metric_row(
        self,
        table: Table,
        metric: str,
        measurements: List[Measurement],
        participating_nodes: int,
    ) -> None:
        """Insert complete statistics row for one metric in final-step table."""
        all_values = [m.metrics[metric] for m in measurements if metric in m.metrics]
        none_count = sum(1 for v in all_values if v is None)
        values = [v for v in all_values if v is not None]

        if not values:
            self._add_no_data_metric_row(
                table, metric, participating_nodes, len(all_values)
            )
            return

        rule = self.formatter.find_rule(metric)
        stats = MetricStats(values, metric, self.formatter)
        coverage_display = self._calculate_coverage_cell(
            metric, measurements, participating_nodes, none_count
        )
        stat_display = self._format_statistics(metric, stats, values)
        counts = stats.counts
        # Combine nan and inf counts into one "bad values" column
        bad_count = counts.nan_count + counts.inf_count
        bad_display = str(bad_count) if bad_count else DIM_DASH
        invalid_display = (
            str(counts.invalid_count) if counts.invalid_count else DIM_DASH
        )
        outliers_display = (
            self.formatter.format_outlier_cell(
                counts.outlier_count,
                MetricStats.compute_log_deviation(values, stats.outlier_indices),
                rule.outlier_thresholds,
            )
            if counts.outlier_count
            else DIM_DASH
        )
        table.add_row(
            self._metric_label(metric, rule),
            coverage_display,
            bad_display,
            invalid_display,
            outliers_display,
            stat_display.sum,
            stat_display.mean,
            stat_display.std,
            stat_display.median,
            stat_display.min,
            stat_display.max,
            stat_display.cv,
        )

    def _count_nodes(
        self,
        measurements: List[Measurement],
        condition_fn: Optional[Callable[[Measurement], bool]] = None,
    ) -> int:
        """Count distinct node IDs in measurements, optionally with filtering."""
        if condition_fn is None:
            return len({m.node_id for m in measurements})
        return len({m.node_id for m in measurements if condition_fn(m)})

    def _get_contexts(self, results: List[Dict[str, Any]]) -> List[str]:
        """Extract unique context names (train, eval, etc.) from experiment results."""
        contexts = {k for result in results for k in result.keys()}
        return sorted(contexts)

    def _detect_progression_pattern(
        self, positions: List[Position]
    ) -> ProgressionPattern:
        """Analyze experiment positions to determine sampling strategy for progression table."""
        if not positions:
            return ProgressionPattern.NONE

        rounds = [pos[0] for pos in positions if pos[0] is not None]
        epochs = [pos[1] for pos in positions if pos[1] is not None]
        batches = [pos[2] for pos in positions if pos[2] is not None]

        round_range = max(rounds) - min(rounds) if len(set(rounds)) > 1 else 0
        epoch_range = max(epochs) - min(epochs) if len(set(epochs)) > 1 else 0
        batch_range = max(batches) - min(batches) if len(set(batches)) > 1 else 0

        total_variation = round_range + epoch_range + batch_range
        if total_variation == 0:
            return ProgressionPattern.NONE

        round_contrib = round_range / total_variation
        epoch_contrib = epoch_range / total_variation
        batch_contrib = batch_range / total_variation

        significant_threshold = CONFIG.PATTERN_SIGNIFICANCE_THRESHOLD
        if round_contrib > significant_threshold:
            return ProgressionPattern.ROUND_DOMINANT
        if epoch_contrib > significant_threshold:
            return ProgressionPattern.EPOCH_DOMINANT
        if batch_contrib > significant_threshold:
            return ProgressionPattern.BATCH_DOMINANT

        if (
            round_contrib > significant_threshold
            and epoch_contrib > significant_threshold
        ):
            return ProgressionPattern.ROUND_EPOCH
        if (
            round_contrib > significant_threshold
            and batch_contrib > significant_threshold
        ):
            return ProgressionPattern.ROUND_BATCH
        if (
            epoch_contrib > significant_threshold
            and batch_contrib > significant_threshold
        ):
            return ProgressionPattern.EPOCH_BATCH

        return ProgressionPattern.BALANCED

    def _explain_sampling_strategy(
        self, pattern: ProgressionPattern, total_points: int, shown_points: int
    ) -> str:
        """Create human-readable description of checkpoint sampling methodology."""
        if total_points <= shown_points:
            return "All data points shown"

        if pattern == ProgressionPattern.NONE:
            return f"Showing {shown_points} of {total_points} checkpoints - start and end only"

        elif pattern in [
            ProgressionPattern.ROUND_DOMINANT,
            ProgressionPattern.EPOCH_DOMINANT,
            ProgressionPattern.BATCH_DOMINANT,
        ]:
            dimension = pattern.value.replace("_dominant", "")
            return f"Showing {shown_points} of {total_points} checkpoints - uniform {dimension} spacing"

        elif pattern in [
            ProgressionPattern.ROUND_EPOCH,
            ProgressionPattern.ROUND_BATCH,
            ProgressionPattern.EPOCH_BATCH,
        ]:
            fractions = ", ".join(
                f"{int(f * 100)}%" for f in CONFIG.STANDARD_SAMPLING_FRACTIONS
            )
            dims = pattern.value.replace("_", "+")
            return f"Showing {shown_points} of {total_points} checkpoints - key {dims} transitions at {fractions}"

        elif pattern == ProgressionPattern.BALANCED:
            return f"Showing {shown_points} of {total_points} checkpoints - logarithmic sampling for complex progression"

        else:
            return f"Showing {shown_points} of {total_points} checkpoints - adaptive sampling"

    def _sample_checkpoints_adaptive(
        self,
        positions: List[Position],
        max_data_points: int = CONFIG.DEFAULT_MAX_DATA_POINTS,
    ) -> List[Position]:
        """Select representative subset of experiment positions for progression display."""
        if len(positions) <= max_data_points:
            return positions

        pattern = self._detect_progression_pattern(positions)

        def _sample_evenly_spaced(
            positions_list: List[Position],
            sampled_list: List[Position],
            max_data_points: int,
        ) -> None:
            """Add evenly distributed checkpoints to sample list."""
            step = max(1, len(positions_list) // (max_data_points - 2))
            for i in range(step, len(positions_list) - 1, step):
                sampled_list.append(positions_list[i])
                if len(sampled_list) >= max_data_points - 1:
                    break

        sampled = [positions[0]]

        if pattern == ProgressionPattern.NONE:
            return [positions[0], positions[-1]]

        if pattern in [
            ProgressionPattern.ROUND_DOMINANT,
            ProgressionPattern.EPOCH_DOMINANT,
            ProgressionPattern.BATCH_DOMINANT,
        ]:
            _sample_evenly_spaced(positions, sampled, max_data_points)

        elif pattern in [
            ProgressionPattern.ROUND_EPOCH,
            ProgressionPattern.ROUND_BATCH,
            ProgressionPattern.EPOCH_BATCH,
        ]:
            for fraction in CONFIG.STANDARD_SAMPLING_FRACTIONS:
                idx = int(len(positions) * fraction)
                if idx > 0 and idx < len(positions) - 1:
                    sampled.append(positions[idx])
                    if len(sampled) >= max_data_points - 1:
                        break

        else:
            indices = [
                int(i)
                for i in np.logspace(
                    0, np.log10(len(positions) - 1), max_data_points - 1
                )
            ]
            for idx in sorted(set(indices))[1:]:
                if idx < len(positions) - 1:
                    sampled.append(positions[idx])

        if sampled[-1] != positions[-1]:
            sampled.append(positions[-1])

        return sampled[:max_data_points]

    def _show_progression_table(
        self, context: str, measurements: List[Measurement], total_nodes: int
    ) -> None:
        """Display time-series progression table showing metric evolution over experiment steps."""
        if not measurements:
            return
        positions: Dict[Position, List[Measurement]] = {}
        for m in measurements:
            pos = m.position
            if pos not in positions:
                positions[pos] = []
            positions[pos].append(m)

        if len(positions.keys()) < CONFIG.MIN_DATA_POINTS_REQUIRED:
            return

        sorted_positions: List[Position] = sorted(
            positions.keys(), key=lambda x: tuple(-1 if v is None else v for v in x)
        )
        displayed_positions = self._sample_checkpoints_adaptive(sorted_positions)
        metrics = self._get_progression_metrics(positions, displayed_positions)
        if not metrics:
            return

        total_checkpoints = len(sorted_positions)
        sampling_applied = len(displayed_positions) < total_checkpoints
        pattern = self._detect_progression_pattern(sorted_positions)

        node_count = self._count_nodes(measurements)
        node_participation_info = self.formatter.format_participation_cell(
            node_count, total_nodes, description="nodes"
        )
        title = f":chart_increasing: {context.title()} Metrics - Experiment Progression"

        participation_line = (
            f"{node_participation_info} participated in {context} context"
        )

        if sampling_applied:
            sampling_explanation = self._explain_sampling_strategy(
                pattern, total_checkpoints, len(displayed_positions)
            )
            shown_count = len(displayed_positions)
            coverage_pct = (
                _pct(shown_count, total_checkpoints) if total_checkpoints > 0 else 0
            )
            status = CONFIG.SAMPLING_ICONS.get_value_for_threshold(coverage_pct)
            caption = f"{participation_line}\n{status} {sampling_explanation}"
        else:
            caption = participation_line

        table = TableFactory.create_base_table(title, caption)
        table.add_column(":bar_chart: Metric", style="bold", justify="left")
        for pos in displayed_positions:
            header = self.formatter.format_position_header(pos)
            table.add_column(header, justify="right", style="white")

        metric_groups = self._group_metrics(metrics)
        self._iterate_metric_groups(
            table,
            metric_groups,
            len(displayed_positions),
            lambda metric: self._add_progression_row(
                table, metric, positions, displayed_positions
            ),
        )
        print(table)

    def _get_progression_metrics(
        self,
        positions: Dict[Position, List[Measurement]],
        displayed_positions: List[Position],
    ) -> List[str]:
        """Filter to metrics with sufficient checkpoint coverage for progression analysis."""
        metric_coverage = defaultdict(int)

        for pos in displayed_positions:
            measurements = positions[pos]
            metrics_at_pos = set()
            for m in measurements:
                metrics_at_pos.update(m.metrics.keys())

            for metric in metrics_at_pos:
                metric_coverage[metric] += 1

        valid_metrics = [
            metric
            for metric, count in metric_coverage.items()
            if count >= CONFIG.MIN_DATA_POINTS_REQUIRED
            and metric not in CONFIG.META_KEYS
        ]
        return sorted(valid_metrics)

    def _collect_progression_values(
        self,
        metric: str,
        positions: Dict[Position, List[Measurement]],
        displayed_positions: List[Position],
    ) -> Tuple[List[Optional[float]], List[str]]:
        """Aggregate metric values at each checkpoint and format for table display."""
        values = []
        row_cells = []

        for pos in displayed_positions:
            measurements = positions[pos]
            pos_values = [
                v
                for v in (
                    m.metrics[metric] for m in measurements if metric in m.metrics
                )
                if v is not None
            ]

            if pos_values:
                avg_value = sum(pos_values) / len(pos_values)
                values.append(avg_value)
                row_cells.append(self.formatter.validate_and_format(metric, avg_value))
            else:
                values.append(None)
                row_cells.append(DIM_DASH)

        return values, row_cells

    def _create_change_row(
        self,
        metric: str,
        values: List[Optional[float]],
    ) -> List[str]:
        """Calculate and format percentage changes between consecutive progression values."""
        change_row = [""]
        for i, current in enumerate(values):
            if i == 0 or current is None:
                change_row.append(DIM_DASH)
                continue

            prev = next(
                (values[j] for j in range(i - 1, -1, -1) if values[j] is not None), None
            )

            if prev is None or abs(prev) < CONFIG.EPSILON_THRESHOLD:
                change_row.append(DIM_DASH)
                continue

            if prev == 0:
                pct_change = 0
            else:
                pct_change = (current - prev) / abs(prev) * 100

            change_row.append(
                self.formatter.format_pct_change_cell(pct_change, metric, prev)
            )
        return change_row

    def _add_progression_row(
        self,
        table: Table,
        metric: str,
        positions: Dict[Position, List[Measurement]],
        displayed_positions: List[Position],
    ) -> None:
        """Insert metric value row and optional change row in progression table."""
        rule = self.formatter.find_rule(metric)

        values, row_cells = self._collect_progression_values(
            metric, positions, displayed_positions
        )
        table.add_row(
            self._metric_label(metric, rule),
            *row_cells,
        )

        non_none_values = [v for v in values if v is not None]
        if len(non_none_values) >= CONFIG.MIN_DATA_POINTS_REQUIRED:
            change_row = self._create_change_row(metric, values)
            table.add_row(*change_row)
