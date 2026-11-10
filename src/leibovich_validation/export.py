"""The dated manuscript export: every declared table and figure plus an index.

`manuscript.yaml` declares the publication set: each `table` entry names a
results/tables stem the table stage wrote as CSV (with LaTeX and notes files
where the entry sets `latex` or `notes`), and each
`figure` entry names a figure the figure stage rendered as PDF and PNG under
its placement. `export` copies every declared file into one dated folder under
results/manuscript/, grouped into `main` and `supplement`, and writes
index.csv listing each item with its placement and caption. A
declared file that is missing stops the export: a publication set with a
silent gap is worse than no export.
"""

import datetime as dt
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import yaml


@dataclass(frozen=True)
class ExportItem:
    """One declared publication item and the files it owns."""

    label: str
    placement: str
    caption: str
    files: tuple[Path, ...]


def _items(manuscript: dict[str, Any], tables_dir: Path, figures_dir: Path) -> list[ExportItem]:
    """The declared tables and figures, main placement first, in file order."""
    items: list[ExportItem] = []
    for stem, entry in (manuscript.get("table") or {}).items():
        names = [f"{stem}.csv"]
        if entry.get("latex"):
            names.append(f"{stem}.tex")
        if entry.get("notes"):
            names.append(f"{stem}_notes.csv")
        items.append(
            ExportItem(
                label=str(entry["label"]),
                placement=str(entry["placement"]),
                caption=" ".join(str(entry["caption"]).split()),
                files=tuple(tables_dir / name for name in names),
            )
        )
    for name, entry in (manuscript.get("figure") or {}).items():
        placement = str(entry["placement"])
        items.append(
            ExportItem(
                label=str(entry["label"]),
                placement=placement,
                caption=" ".join(str(entry["caption"]).split()),
                files=tuple(
                    figures_dir / placement / f"{name}.{suffix}" for suffix in ("pdf", "png")
                ),
            )
        )
    return sorted(items, key=lambda item: item.placement != "main")


def export(
    out_dir: Path,
    manuscript_path: Path,
    tables_dir: Path,
    figures_dir: Path,
    *,
    today: dt.date | None = None,
) -> Path:
    """Copy the declared tables and figures into `<out_dir>/<date>/` with an index.

    Returns the dated folder. `out_dir` is rebuilt on every run, so it always
    holds exactly one dated export.
    """
    manuscript = yaml.safe_load(manuscript_path.read_text())
    items = _items(manuscript, tables_dir, figures_dir)
    for item in items:
        for source in item.files:
            if not source.is_file():
                msg = (
                    f"{item.label}: declared file {source} does not exist; "
                    "run the table and figure stages first"
                )
                raise FileNotFoundError(msg)
    date = today or dt.date.today()
    shutil.rmtree(out_dir, ignore_errors=True)
    dated = out_dir / date.isoformat()
    dated.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for item in items:
        target_dir = dated / item.placement
        target_dir.mkdir(parents=True, exist_ok=True)
        copied = []
        for source in item.files:
            target = target_dir / source.name
            shutil.copy2(source, target)
            copied.append(target.relative_to(dated).as_posix())
        rows.append(
            {
                "item": item.label,
                "placement": item.placement,
                "files": "; ".join(copied),
                "caption": item.caption,
            }
        )
    index = pl.DataFrame(rows, schema=["item", "placement", "files", "caption"])
    index.write_csv(dated / "index.csv")
    return dated
