"""
Styling and theming for FLUX experiment displays.

Centralized colors, emojis, table configurations, and column definitions.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ColorScheme:
    """Centralized color definitions for all display components."""

    # Primary colors for different data types
    round_color: str = "bright_blue"
    epoch_color: str = "bright_magenta"
    batch_color: str = "bright_yellow"
    step_color: str = "bright_green"

    # Table styling colors
    header_tertiary: str = "bold bright_cyan"

    # Text colors
    metric_name: str = "bright_white"
    metric_emoji: str = "bold bright_green"
    group_header: str = "bold bright_blue"
    group_count: str = "dim"

    # Status colors
    success: str = "bold bright_green"
    warning: str = "bold bright_yellow"
    error: str = "bold bright_red"
    info: str = "bright_cyan"

    # Data colors
    data_primary: str = "white"

    # Change indicators
    positive_change: str = "italic bright_green"
    negative_change: str = "italic bright_red"


@dataclass(frozen=True)
class EmojiScheme:
    """Centralized emoji definitions for consistent iconography."""

    # Table types
    progression: str = "📈"
    node_statistics: str = "👥"

    # Coordinate levels
    round: str = "🔄"
    epoch: str = "⏳"
    batch: str = "📦"
    step: str = "👣"

    # Metrics and data
    metric: str = "📊"
    sum: str = "➕"
    mean: str = "📊"
    std: str = "📏"
    min: str = "🔻"
    max: str = "🔺"
    median: str = "📈"
    cv: str = "📈"
    nodes: str = "👥"
    anomalies: str = "⚠️"

    # Status indicators
    success: str = "✅"
    warning: str = "⚠️"
    error: str = "❌"
    info: str = "ℹ️"

    # General
    value: str = "📋"
    duration: str = "⏱️"
    rounds: str = "🔄"
    
    # Additional status/action emojis
    alert: str = "🚨"
    blocked: str = "🚫"
    forbidden: str = "⛔"
    investigate: str = "🔍"
    processing: str = "🔄"
    system: str = "⚡"


@dataclass(frozen=True)
class ValidationConfiguration:
    """Simple, generic validation and display configuration."""

    # Display thresholds (centralized configuration)
    small_change_threshold: float = 0.1  # For ~0.0% display
    division_epsilon: float = 1e-10  # Avoid division by zero
    max_epochs_per_round: int = 10  # Column limits
    max_batches_per_epoch: int = 6  # Column limits

    # Coverage and completion thresholds
    coverage_threshold_good: float = 0.75  # 75% - good coverage
    coverage_threshold_perfect: float = 1.0  # 100% - perfect coverage
    completion_threshold: float = 0.50  # 50% - minimum completion rate

    # Text formatting limits
    error_message_truncation_limit: int = 30  # Character limit for error messages


# Global configuration instances
COLORS = ColorScheme()
EMOJIS = EmojiScheme()
VALIDATION = ValidationConfiguration()

# Display constants
DISPLAY_PLACEHOLDER = "[dim]-[/dim]"  # Standard placeholder for missing/empty data


