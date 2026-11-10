"""The progress-bar layout every long-running pipeline step shares."""

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn


def bar(console: Console | None = None, *, disable: bool = False) -> Progress:
    """Items done, items in total and time remaining. `disable` silences the bar."""
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=disable,
    )
