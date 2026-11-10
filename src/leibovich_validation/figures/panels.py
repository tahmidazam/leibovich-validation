"""Panel functions: the registered `draw` geometries for manuscript figures.

A panel function takes an axis and a Polars table, uses only the fixed column roles and
draws into the axis it is given. Panel code holds no numbers of its own: colours, line
styles and thresholds arrive as arguments.
"""

import textwrap
from collections.abc import Callable, Iterable, Mapping
from typing import Any, cast

import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import blended_transform_factory

from leibovich_validation.predict import LEIBOVICH, LEIBOVICH_PLUS

# The Okabe-Ito palette, safe for colour-blind readers. A model keeps one colour
# in every figure.
PALETTE = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
]

SERIES_COLOURS = {
    LEIBOVICH: "#000000",
    LEIBOVICH_PLUS: "#D55E00",
    # The observed estimates in the `dot` geometry. Black is already the
    # original Leibovich model, and the Okabe-Ito orange is too close to
    # Leibovich Plus's vermillion at print size, so observed takes the blue.
    "observed": "#0072B2",
}

LINE_STYLES = ["-", "--", "-.", ":"]

# Marker order for the predicted series in the `dot` geometry: a different
# shape per model so the figure still reads in greyscale.
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]

# A series keeps its marker in every panel, whichever other series it shares
# the panel with.
SERIES_MARKERS = {"observed": "o", LEIBOVICH: "s", LEIBOVICH_PLUS: "^"}


def series_colours(levels: Iterable[str]) -> dict[str, str]:
    """One colour per `series` level: the fixed model colours first, the palette after."""
    return {
        level: SERIES_COLOURS.get(level, PALETTE[index % len(PALETTE)])
        for index, level in enumerate(sorted(levels))
    }


def stratum_styles(levels: Iterable[Any]) -> dict[Any, str]:
    """One line style per `stratum` level: a second grouping is a style, not a colour."""
    return {
        level: LINE_STYLES[index % len(LINE_STYLES)] for index, level in enumerate(sorted(levels))
    }


def curve(
    ax: Axes,
    table: pl.DataFrame,
    *,
    colours: Mapping[str, str] | None = None,
    styles: Mapping[Any, str] | None = None,
    minimum_at_risk: int | None = None,
) -> None:
    """Lines of `estimate` over `time`, coloured by `series` and styled by `stratum`.

    `lo` and `hi` columns, when present, draw an interval band around each line. When
    the table carries an `n` column of numbers at risk and `minimum_at_risk` is set,
    each line stops at the last time with at least that many at risk.
    """
    for role in ("time", "estimate"):
        if role not in table.columns:
            msg = f"the curve geometry needs a `{role}` column"
            raise ValueError(msg)
    series_levels = (
        table["series"].unique().sort().to_list() if "series" in table.columns else [None]
    )
    stratum_levels = (
        table["stratum"].unique().sort().to_list() if "stratum" in table.columns else [None]
    )
    colours = dict(colours or series_colours(s for s in series_levels if s is not None))
    styles = dict(styles or stratum_styles(s for s in stratum_levels if s is not None))
    banded = {"lo", "hi"} <= set(table.columns)
    for series in series_levels:
        for stratum in stratum_levels:
            subset = table.sort("time")
            if series is not None:
                subset = subset.filter(pl.col("series") == series)
            if stratum is not None:
                subset = subset.filter(pl.col("stratum") == stratum)
            if minimum_at_risk is not None and "n" in subset.columns:
                above = subset.filter(pl.col("n") >= minimum_at_risk)
                if above.is_empty():
                    continue
                subset = subset.filter(pl.col("time") <= above["time"].max())
            if subset.is_empty():
                continue
            colour = colours.get(series, PALETTE[0])
            linestyle = styles.get(stratum, "-")
            x = subset["time"].to_list()
            ax.plot(
                x,
                subset["estimate"].to_list(),
                color=colour,
                linestyle=linestyle,
            )
            if banded:
                ax.fill_between(
                    x,
                    subset["lo"].to_list(),
                    subset["hi"].to_list(),
                    color=colour,
                    alpha=0.2,
                    linewidth=0,
                )


def calibration(
    ax: Axes,
    table: pl.DataFrame,
    *,
    colours: Mapping[str, str] | None = None,
    styles: Mapping[Any, str] | None = None,
    minimum_at_risk: int | None = None,
) -> None:
    """Observed against predicted risk per `series`, with the identity line.

    Rows labelled "smooth" draw the smoothed calibration curve as a line.
    Rows labelled "bin" draw the grouped observed estimate against the mean
    predicted risk as points, with a vertical `lo` to `hi` interval when those
    columns carry values.
    Rows labelled "rug" draw a strip of predicted-risk ticks along the foot of
    the plot, one row per series, so the spread of predictions is visible. `styles` and
    `minimum_at_risk` are accepted for the shared panel signature; this
    geometry has no second grouping.
    """
    for role in ("predicted", "estimate", "kind"):
        if role not in table.columns:
            msg = f"the calibration geometry needs a `{role}` column"
            raise ValueError(msg)
    bins = table.filter(pl.col("kind") == "bin")
    rug = table.filter(pl.col("kind") == "rug")
    smooth = table.filter(pl.col("kind") == "smooth")
    series_levels = bins["series"].unique().sort().to_list() if "series" in bins.columns else [None]
    colours = dict(colours or series_colours(s for s in series_levels if s is not None))

    # The identity line spans the data, not the unit square: predicted risks
    # rarely fill the whole range, and a longer line only adds empty space.
    limit = 0.0
    for column in ("predicted", "estimate", "hi"):
        if column in table.columns:
            values = table[column].drop_nulls()
            if values.len():
                limit = max(limit, float(cast(Any, values.max())))
    ax.plot([0, limit], [0, limit], color="0.4", linewidth=0.8, linestyle="--", zorder=0)

    foot = blended_transform_factory(ax.transData, ax.transAxes)
    banded = {"lo", "hi"} <= set(bins.columns)
    for index, series in enumerate(series_levels):
        colour = colours.get(series, PALETTE[0])
        subset = bins if series is None else bins.filter(pl.col("series") == series)
        subset = subset.sort("predicted")
        if subset.height:
            x = subset["predicted"].to_list()
            y = subset["estimate"].to_list()
            if banded and subset["lo"].null_count() == 0 and subset["hi"].null_count() == 0:
                # The interval is drawn from its bounds, not as a symmetric
                # error: a percentile interval need not contain its estimate.
                below = [max(0.0, est - lo) for est, lo in zip(y, subset["lo"], strict=True)]
                above = [max(0.0, hi - est) for est, hi in zip(y, subset["hi"], strict=True)]
                ax.errorbar(
                    x,
                    y,
                    yerr=[below, above],
                    color=colour,
                    marker="o",
                    markersize=4.5,
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                    linestyle="None",
                    elinewidth=1.3,
                    capsize=0,
                    zorder=3,
                )
            else:
                ax.plot(x, y, color=colour, marker="o", markersize=4.5, linestyle="None", zorder=3)
        curve = smooth if series is None else smooth.filter(pl.col("series") == series)
        if curve.height:
            curve = curve.sort("predicted")
            ax.plot(
                curve["predicted"].to_list(),
                curve["estimate"].to_list(),
                color=colour,
                linewidth=0.8,
                alpha=0.55,
            )
        rug_subset = rug if series is None else rug.filter(pl.col("series") == series)
        if rug_subset.height:
            # One row of ticks per model along the foot of the plot, so the
            # models' spreads of predicted risk do not overprint. The figure
            # builder leaves this strip below zero on the observed axis.
            ax.plot(
                rug_subset["predicted"].to_list(),
                [0.025 + 0.035 * index] * rug_subset.height,
                transform=foot,
                marker="|",
                markersize=5,
                linestyle="None",
                color=colour,
            )


# Flow-diagram layout in axis-fraction coordinates: the funnel stacks on the
# left and the side boxes sit on the right, the exclusions beside the
# connector they leave and the field counts beside the last funnel box. Side
# boxes hold bulleted lists, left aligned and one point smaller than the funnel.
_FLOW_X_MAIN = 0.25
_FLOW_X_SIDE = 0.75
_FLOW_W_MAIN = 0.44
_FLOW_W_SIDE = 0.46
_FLOW_LINE = 0.043
_FLOW_SIDE_LINE = 0.0345
_FLOW_PAD = 0.025
_FLOW_MIN_PAD = 0.012
_FLOW_GAP = 0.075
# The rounded box outline sits this far outside the box's nominal edge.
_FLOW_OUTLINE = 0.006
_FLOW_TOP = 0.99
_FLOW_WRAP_MAIN = 26
_FLOW_WRAP_SIDE = 28


def _flow_box_height(text: str, line: float = _FLOW_LINE) -> float:
    """The box height for wrapped box text, in axis fractions."""
    return (text.count("\n") + 1) * line + 2 * _FLOW_PAD


def _flow_box(
    ax: Axes,
    centre_x: float,
    top: float,
    width: float,
    text: str,
    *,
    dashed: bool = False,
    side: bool = False,
    height: float | None = None,
) -> float:
    """Draw one rounded box with its text; returns its bottom edge.

    Funnel boxes centre their text; `side` boxes left-align a bulleted list.
    `height` overrides the height the text would take.
    """
    if height is None:
        height = _flow_box_height(text, _FLOW_SIDE_LINE if side else _FLOW_LINE)
    ax.add_patch(
        FancyBboxPatch(
            (centre_x - width / 2, top - height),
            width,
            height,
            boxstyle=f"round,pad={_FLOW_OUTLINE}",
            transform=ax.transAxes,
            facecolor="white",
            edgecolor="0.3",
            linestyle="--" if dashed else "-",
            linewidth=0.8,
        )
    )
    ax.text(
        centre_x - width / 2 + 0.025 if side else centre_x,
        top - height / 2,
        text,
        transform=ax.transAxes,
        ha="left" if side else "center",
        va="center",
        multialignment="left" if side else "center",
        fontsize="small" if side else None,
    )
    return top - height


def _flow_arrow(ax: Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    """One connector arrow between flow boxes, in axis-fraction coordinates."""
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            shrinkA=0,
            shrinkB=0,
            linewidth=0.8,
            color="0.3",
            transform=ax.transAxes,
        )
    )


def _flow_list(heading: str, rows: pl.DataFrame) -> str:
    """A side-box heading over one wrapped bullet per row, with its count."""
    items = [
        textwrap.fill(
            f"\u2022 {row['label']} ({row['n']:,})",
            _FLOW_WRAP_SIDE,
            subsequent_indent="   ",
        )
        for row in rows.iter_rows(named=True)
    ]
    return "\n".join([heading, *items])


def flow(
    ax: Axes,
    table: pl.DataFrame,
    *,
    colours: Mapping[str, str] | None = None,
    styles: Mapping[Any, str] | None = None,
    minimum_at_risk: int | None = None,
) -> None:
    """Participant-flow boxes and arrows from a counts table.

    `kind` rows of "main" stack down the left in table order, joined by
    arrows; `label` is the box text and `n` the count drawn under it. Rows of
    kind "excluded" collect into one side box level with the transition that
    removed them, with an arrow from the connector to the middle of the box.
    Rows of kind "note" with a non-zero count collect into a dashed box beside
    the last funnel box, tied to it by a dashed line; they record what the
    cohort build did without excluding anyone. The first funnel box sits at
    the top of the panel and the last at the bottom, with the dashed box on
    the same baseline, so panels side by side line up. `colours`, `styles`
    and `minimum_at_risk` are accepted for the shared panel signature; this
    geometry maps nothing to colour.
    """
    for role in ("kind", "label", "n"):
        if role not in table.columns:
            msg = f"the flow geometry needs a `{role}` column"
            raise ValueError(msg)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    main = table.filter(pl.col("kind") == "main")
    excluded = table.filter(pl.col("kind") == "excluded")
    notes = table.filter((pl.col("kind") == "note") & (pl.col("n") > 0))
    if main.is_empty():
        msg = "the flow geometry needs at least one `main` row"
        raise ValueError(msg)

    excluded_text = _flow_list("Excluded", excluded) if excluded.height else ""
    excluded_height = _flow_box_height(excluded_text, _FLOW_SIDE_LINE) if excluded_text else 0.0
    notes_text = _flow_list("Recorded, not excluded", notes) if notes.height else ""
    notes_height = _flow_box_height(notes_text, _FLOW_SIDE_LINE) if notes_text else 0.0
    main_texts = [
        textwrap.fill(str(row["label"]), _FLOW_WRAP_MAIN) + f"\nn = {row['n']:,}"
        for row in main.iter_rows(named=True)
    ]
    heights = [_flow_box_height(text) for text in main_texts]
    floor = 1 - _FLOW_TOP
    side_left = _FLOW_X_SIDE - _FLOW_W_SIDE / 2

    if len(main_texts) == 1:
        # Nothing to connect: the side boxes stack beside the single box.
        _flow_box(ax, _FLOW_X_MAIN, _FLOW_TOP, _FLOW_W_MAIN, main_texts[0])
        side_top = _FLOW_TOP
        if excluded_text:
            bottom = _flow_box(ax, _FLOW_X_SIDE, side_top, _FLOW_W_SIDE, excluded_text, side=True)
            side_top = bottom - _FLOW_GAP
        if notes_text:
            _flow_box(ax, _FLOW_X_SIDE, side_top, _FLOW_W_SIDE, notes_text, dashed=True, side=True)
        return

    # A short dashed box takes the height of the funnel box beside it, when
    # its text still fits with a slimmer margin.
    if notes_text and notes_height - 2 * (_FLOW_PAD - _FLOW_MIN_PAD) <= heights[-1]:
        notes_height = heights[-1]

    # The funnel: the last box sits on the floor and the connectors share the
    # height that is left; the last connector also holds the exclusions box.
    gaps = len(heights) - 1
    gap = max(_FLOW_GAP, (_FLOW_TOP - floor - sum(heights) - excluded_height) / gaps)
    tops = [_FLOW_TOP]
    for index in range(1, len(heights)):
        last = index == gaps
        tops.append(tops[-1] - heights[index - 1] - gap - (excluded_height if last else 0.0))
    bottoms = [top - height for top, height in zip(tops, heights, strict=True)]
    for index, text in enumerate(main_texts):
        _flow_box(ax, _FLOW_X_MAIN, tops[index], _FLOW_W_MAIN, text)
        if index:
            _flow_arrow(
                ax,
                (_FLOW_X_MAIN, bottoms[index - 1] - _FLOW_OUTLINE),
                (_FLOW_X_MAIN, tops[index] + _FLOW_OUTLINE),
            )

    notes_top = bottoms[-1] + notes_height
    if excluded_text:
        # Centred on the connector, or higher when the dashed box needs the room.
        centre = (bottoms[-2] + tops[-1]) / 2
        if notes_text:
            centre = max(centre, notes_top + _FLOW_GAP / 2 + excluded_height / 2)
        centre = min(centre, _FLOW_TOP - excluded_height / 2)
        _flow_box(
            ax,
            _FLOW_X_SIDE,
            centre + excluded_height / 2,
            _FLOW_W_SIDE,
            excluded_text,
            side=True,
        )
        _flow_arrow(ax, (_FLOW_X_MAIN, centre), (side_left - _FLOW_OUTLINE, centre))
    if notes_text:
        _flow_box(
            ax,
            _FLOW_X_SIDE,
            notes_top,
            _FLOW_W_SIDE,
            notes_text,
            dashed=True,
            side=True,
            height=notes_height,
        )
        # A dashed line, not an arrow: the counts describe the analysis set
        # and no patient moves between the boxes.
        link_y = (tops[-1] + bottoms[-1]) / 2
        ax.plot(
            [_FLOW_X_MAIN + _FLOW_W_MAIN / 2 + _FLOW_OUTLINE, side_left - _FLOW_OUTLINE],
            [link_y, link_y],
            transform=ax.transAxes,
            color="0.3",
            linestyle="--",
            linewidth=0.8,
        )


def _count_label(value: Any) -> str:
    """A group count for a tick label: whole when it is one, a decimal otherwise.

    Pooled group sizes and event counts can be fractional; showing the decimal
    keeps the flagging threshold honest next to the count.
    """
    number = float(value)
    return f"{number:,.0f}" if number.is_integer() else f"{number:.1f}"


def dot(
    ax: Axes,
    table: pl.DataFrame,
    *,
    colours: Mapping[str, str] | None = None,
    styles: Mapping[Any, str] | None = None,
    minimum_at_risk: int | None = None,
) -> None:
    """Observed risk with its interval beside each model's mean predicted risk.

    One row band per `label` group: `kind` "observed" rows draw `estimate`
    with a `lo` to `hi` interval and `kind` "predicted" rows draw `predicted`
    markers, dodged inside the band. Tick labels carry the group's `n` and
    `events` on a second line; a group marked `small` is drawn faded on a hatched band, flagged
    rather than removed. `group_order`, when present, orders the groups;
    otherwise they take the table's row order. `styles` and
    `minimum_at_risk` are accepted for the shared panel signature.
    """
    for role in ("label", "kind", "series", "estimate", "predicted", "n", "events"):
        if role not in table.columns:
            msg = f"the dot geometry needs a `{role}` column"
            raise ValueError(msg)
    if "group_order" in table.columns:
        labels = (
            table.group_by("label")
            .agg(pl.col("group_order").first())
            .sort("group_order")["label"]
            .to_list()
        )
    else:
        labels = table["label"].unique(maintain_order=True).to_list()
    observed = table.filter(pl.col("kind") == "observed")
    predicted = table.filter(pl.col("kind") == "predicted")
    series_levels = predicted["series"].unique().sort().to_list()
    colours = dict(colours or series_colours(table["series"].unique().to_list()))
    observed_colour = colours.get("observed", PALETTE[0])
    banded = {"lo", "hi"} <= set(table.columns)
    slots = 1 + len(series_levels)
    offsets = np.linspace(-0.22, 0.22, slots) if slots > 1 else np.zeros(1)
    ylabels = []
    for index, label in enumerate(labels):
        rows = table.filter(pl.col("label") == label)
        n, events = rows["n"][0], rows["events"][0]
        counts = (
            ""
            if n is None or events is None
            else f"\nn = {_count_label(n)}, {_count_label(events)} events"
        )
        ylabels.append(f"{label}{counts}")
        small = "small" in table.columns and rows["small"][0] is True
        alpha = 0.35 if small else 1.0
        if small:
            ax.axhspan(
                index - 0.45,
                index + 0.45,
                facecolor="0.95",
                edgecolor="0.8",
                hatch="//",
                linewidth=0,
                alpha=0.6,
                zorder=0,
            )
        for record in observed.filter(pl.col("label") == label).iter_rows(named=True):
            estimate = record["estimate"]
            if estimate is None:
                continue
            y = index + offsets[0]
            if banded and record["lo"] is not None and record["hi"] is not None:
                ax.errorbar(
                    [estimate],
                    [y],
                    xerr=[[estimate - record["lo"]], [record["hi"] - estimate]],
                    fmt="o",
                    markersize=4,
                    capsize=2.5,
                    linewidth=1.0,
                    color=observed_colour,
                    alpha=alpha,
                )
            else:
                ax.plot(
                    [estimate],
                    [y],
                    marker="o",
                    markersize=4,
                    linestyle="None",
                    color=observed_colour,
                    alpha=alpha,
                )
        for slot, series in enumerate(series_levels, start=1):
            subset = predicted.filter((pl.col("label") == label) & (pl.col("series") == series))
            for record in subset.iter_rows(named=True):
                if record["predicted"] is None:
                    continue
                ax.plot(
                    [record["predicted"]],
                    [index + offsets[slot]],
                    marker=SERIES_MARKERS.get(series, MARKERS[slot % len(MARKERS)]),
                    markersize=4,
                    linestyle="None",
                    color=colours.get(series, PALETTE[0]),
                    alpha=alpha,
                )
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(ylabels)
    # First group at the top, the forest-plot convention, with half a row of
    # headroom so the top group's markers do not clip against the spine.
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_xlim(left=0.0)


GEOMETRIES: dict[str, Callable[..., None]] = {
    "curve": curve,
    "calibration": calibration,
    "flow": flow,
    "dot": dot,
}
