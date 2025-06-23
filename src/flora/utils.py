import logging

from rich.console import Console
from rich.logging import RichHandler
from rich.rule import Rule

console = Console()


def setup_rich_logging(level=logging.INFO) -> None:
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=True,
        markup=True,
        omit_repeated_times=True,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
    )

    formatter = logging.Formatter(fmt="[%(name)s] %(message)s")
    rich_handler.setFormatter(formatter)

    root_logger.addHandler(rich_handler)
    root_logger.setLevel(level)


def log_sep(title: str = "", style: str = "═", color: str = "yellow") -> None:
    """full-width separator for different logging stages"""
    if title:
        console.print()
        console.print(
            Rule(f"[bold {color}]{title}[/bold {color}]", style=color, characters=style)
        )
    else:
        console.print(Rule(style="dim", characters=style))
