from rich.console import Console
from rich.progress import TimeRemainingColumn

from leibovich_validation import progress


def test_bar_reports_items_done_total_and_time_remaining():
    console = Console(record=True, width=60)
    with progress.bar(console) as bar:
        task = bar.add_task("hashing", total=2)
        bar.advance(task)
    assert "1/2" in console.export_text()
    assert any(isinstance(column, TimeRemainingColumn) for column in bar.columns)
