"""Turn a manuscript.yaml figure entry and its summary tables into a PDF and PNG."""

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import matplotlib
import matplotlib.pyplot as plt
import polars as pl
import yaml
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from leibovich_validation.figures import panels

# The pipeline renders headless.
plt.switch_backend("Agg")

MM_TO_INCH = 1 / 25.4


def load_manuscript(path: Path) -> dict[str, Any]:
    """The shared `style` rcParams and the `figure` entries of the layout file."""
    return yaml.safe_load(path.read_text())


def _filter(
    table: pl.DataFrame,
    where: Mapping[str, Any] | None,
    drop: Mapping[str, Any] | None = None,
) -> pl.DataFrame:
    """Keep the rows matching `where` and drop the rows matching `drop`."""
    for column, value in (where or {}).items():
        table = table.filter(pl.col(column).cast(pl.String) == str(value))
    for column, value in (drop or {}).items():
        table = table.filter(pl.col(column).cast(pl.String) != str(value))
    return table


def _fill(
    template: str | None, table: pl.DataFrame, where: Mapping[str, Any] | None = None
) -> str | None:
    """Fill a label template such as "{stratum} (n = {n:,})" from the table's first row.

    Values named in the panel's `where` fill in first, so a title still makes
    sense when the slice is empty (a cohort whose results have not landed yet).
    If some placeholders still cannot be filled, the stratum alone is the title.
    """
    if template is None or "{" not in template:
        return template
    context = dict(where or {})
    if not table.is_empty():
        context.update(table.row(0, named=True))
    try:
        return template.format(**context)
    except KeyError:
        return str(where["stratum"]) if where and "stratum" in where else None


def panel_letter(letter: str) -> str:
    """The bold panel letter as it opens a panel title."""
    return rf"$\mathbf{{{letter}}}$"


def _levels(tables: Iterable[pl.DataFrame], column: str) -> list[Any]:
    seen = set()
    for table in tables:
        if column in table.columns:
            seen.update(table[column].unique().to_list())
    return sorted(seen)


def draw_figure(
    entry: Mapping[str, Any],
    tables: Mapping[str, pl.DataFrame],
    *,
    minimum_at_risk: int | None = None,
) -> Figure:
    """Draw one `figure` entry from its summary tables.

    Layout, sizes and the legend come from the entry; the caller applies the shared
    `style` rcParams so saved files keep TrueType fonts.
    """
    if entry.get("sharey") and any(p["draw"] == "dot" for p in entry["panels"].values()):
        # Shared axes share tick labels, and each dot panel labels its rows
        # with its own group sizes and event counts.
        msg = "dot panels label their own rows; sharey cannot be set"
        raise ValueError(msg)
    width_mm, height_mm = entry["size"]
    fig, axes = plt.subplot_mosaic(
        entry["mosaic"],
        figsize=(width_mm * MM_TO_INCH, height_mm * MM_TO_INCH),
        sharex=bool(entry.get("sharex")),
        sharey=bool(entry.get("sharey")),
        layout="constrained",
    )
    slices = {}
    for letter, panel in entry["panels"].items():
        try:
            table = tables[panel["table"]]
        except KeyError:
            msg = f"panel {letter} reads unknown summary table {panel['table']!r}"
            raise ValueError(msg) from None
        slices[letter] = panel, _filter(table, panel.get("where"), panel.get("drop"))
    colours = panels.series_colours(_levels((t for _, t in slices.values()), "series"))
    styles = panels.stratum_styles(_levels((t for _, t in slices.values()), "stratum"))
    lettered = len(entry["panels"]) > 1
    for letter, (panel, table) in slices.items():
        ax = axes[letter]
        try:
            geometry = panels.GEOMETRIES[panel["draw"]]
        except KeyError:
            msg = f"panel {letter} uses unregistered geometry {panel['draw']!r}"
            raise ValueError(msg) from None
        geometry(ax, table, colours=colours, styles=styles, minimum_at_risk=minimum_at_risk)
        title = _fill(panel.get("title"), table, panel.get("where")) or ""
        if lettered:
            # One left-aligned title: the bold letter, then the panel title.
            title = f"{panel_letter(letter)}  {title}".rstrip()
        if title:
            ax.set_title(title, loc="left")
        if xlabel := _fill(panel.get("xlabel"), table, panel.get("where")):
            ax.set_xlabel(xlabel)
        if ylabel := _fill(panel.get("ylabel"), table, panel.get("where")):
            ax.set_ylabel(ylabel)
    if all(panel["draw"] in ("dot", "calibration") for panel in entry["panels"].values()):
        # Panels drawn from one summary table share a risk scale, taken from
        # the rows the figure draws, so they stay comparable and a marker at the
        # data edge does not clip.
        # Time axes (the curve geometry) keep their own limits.
        peaks: dict[str, float] = {}
        for panel, table in slices.values():
            peaks[panel["table"]] = max(peaks.get(panel["table"], 0.0), _data_peak(table))
        for letter, panel in entry["panels"].items():
            peak = peaks[panel["table"]]
            if peak <= 0:
                continue
            ax = axes[letter]
            # Risks cannot pass 1, so the axis stops just beyond it.
            limit = min(peak * 1.15, 1.03)
            ax.set_xlim(0.0, limit)
            if panel["draw"] == "calibration":
                # A strip below zero holds the rug of predicted risks.
                ax.set_ylim(-0.1 * limit, limit)
    if "legend" in entry:
        legend = entry["legend"]
        # `labels` renames series in the legend; `stratum_label` is a template
        # for the line-style entries, such as "age {} years".
        names = legend.get("labels", {})
        stratum_label = legend.get("stratum_label", "{}")
        markers = all(panel["draw"] == "dot" for panel in entry["panels"].values())
        handles = [
            Line2D(
                [],
                [],
                color=colour,
                linestyle="None" if markers else "-",
                marker=panels.SERIES_MARKERS.get(series, "o") if markers else None,
                label=str(names.get(series, series)),
            )
            for series, colour in colours.items()
        ]
        if any(
            table["stratum"].n_unique() > 1
            for _, table in slices.values()
            if "stratum" in table.columns
        ):
            # Line styles only need a legend when a panel mixes strata.
            handles += [
                Line2D(
                    [],
                    [],
                    color="0.4",
                    linestyle=cast(Any, style),
                    label=stratum_label.format(stratum),
                )
                for stratum, style in styles.items()
            ]
        opts = {k: v for k, v in legend.items() if k not in ("loc", "labels", "stratum_label")}
        loc = legend.get("loc", "outside lower center")
        fig.legend(handles=handles, loc=loc, **opts)
    return fig


def _data_peak(table: pl.DataFrame) -> float:
    """The largest value on the risk axes of a summary table, or zero."""
    peak = 0.0
    for column in ("predicted", "estimate", "hi"):
        if column in table.columns:
            values = table[column].drop_nulls()
            if values.len():
                peak = max(peak, float(cast(Any, values.max())))
    return peak


def build_figure(
    name: str,
    manuscript: Path,
    summary_dir: Path,
    out_dir: Path,
    *,
    minimum_at_risk: int | None = None,
) -> list[Path]:
    """Render the `figure.<name>` entry to a vector PDF and a 300 dpi PNG."""
    spec = load_manuscript(manuscript)
    try:
        entry = spec["figure"][name]
    except KeyError:
        msg = f"{manuscript} has no figure entry {name!r}"
        raise ValueError(msg) from None
    tables = {}
    for panel in entry["panels"].values():
        if panel["table"] not in tables:
            tables[panel["table"]] = pl.read_csv(summary_dir / f"{panel['table']}.csv")
    with matplotlib.rc_context(spec.get("style") or {}):
        fig = draw_figure(entry, tables, minimum_at_risk=minimum_at_risk)
        placement_dir = out_dir / entry["placement"]
        placement_dir.mkdir(parents=True, exist_ok=True)
        paths = [placement_dir / f"{name}.{suffix}" for suffix in ("pdf", "png")]
        fig.savefig(paths[0])
        fig.savefig(paths[1], dpi=300)
        plt.close(fig)
    return paths
