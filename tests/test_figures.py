import itertools
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest
from matplotlib.lines import Line2D

from leibovich_validation import figures
from leibovich_validation.config import REPO_ROOT
from leibovich_validation.figures import panels

MANUSCRIPT = REPO_ROOT / "manuscript.yaml"


def synthetic_table() -> pl.DataFrame:
    rows = [
        (series, stratum, time, 0.02 * time + 0.1 * s + 0.01 * stratum)
        for s, series in enumerate(["version a", "version b"])
        for stratum in (60, 80)
        for time in range(11)
    ]
    return pl.DataFrame(rows, schema=["series", "stratum", "time", "estimate"], orient="row")


def test_curve_draws_one_line_per_series_and_stratum():
    fig, ax = plt.subplots()
    panels.curve(ax, synthetic_table())
    assert len(ax.lines) == 4
    assert ax.dataLim.xmin == 0
    assert ax.dataLim.xmax == 10
    assert len({line.get_color() for line in ax.lines}) == 2
    assert len({line.get_linestyle() for line in ax.lines}) == 2
    plt.close(fig)


def test_curve_draws_interval_bands():
    fig, ax = plt.subplots()
    table = synthetic_table().with_columns(
        (pl.col("estimate") - 0.02).alias("lo"), (pl.col("estimate") + 0.02).alias("hi")
    )
    panels.curve(ax, table)
    assert len(ax.collections) == 4
    plt.close(fig)


def test_curve_stops_where_numbers_at_risk_fall_below_the_minimum():
    fig, ax = plt.subplots()
    table = synthetic_table().with_columns(
        pl.when(pl.col("time") <= 5).then(30).otherwise(5).alias("n")
    )
    panels.curve(ax, table, minimum_at_risk=20)
    assert len(ax.lines) == 4
    assert all(np.asarray(line.get_xdata()).max() == 5 for line in ax.lines)
    plt.close(fig)


def test_curve_requires_the_time_and_estimate_roles():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="estimate"):
        panels.curve(ax, pl.DataFrame({"time": [1.0], "x": [0.1]}))
    plt.close(fig)


def synthetic_for_entry(entry: dict, table_name: str) -> pl.DataFrame:
    """Rows covering every panel's `where` slice, so no panel filters to empty."""
    wheres = [
        panel.get("where", {}) for panel in entry["panels"].values() if panel["table"] == table_name
    ]
    keys = sorted({key for where in wheres for key in where})
    values = {key: sorted({where[key] for where in wheres if key in where}) for key in keys}
    values.setdefault("series", ["version a", "version b"])
    values.setdefault("stratum", [60, 80])
    rows = []
    for index, combo in enumerate(itertools.product(*values.values())):
        base = dict(zip(values, combo, strict=True))
        for time in (0.0, 2.5, 5.0, 7.5, 10.0):
            estimate = 0.02 * time + 0.05 * index
            rows.append(
                {
                    **base,
                    "time": time,
                    "estimate": estimate,
                    "kind": "bin",
                    "predicted": estimate - 0.01,
                    "lo": estimate - 0.02,
                    "hi": estimate + 0.02,
                    "n": 40,
                    "events": 25,
                    "cohort_n": 400,
                    "label": base.get("label", "bin"),
                }
            )
        # The flow geometry reads kind, label and n; the other geometries
        # ignore these extra rows because their kind is not "bin" or "rug".
        for kind, label, n in (
            ("main", "Source records", 100),
            ("excluded", "Reason", 20),
            ("note", "Field count", 5),
            ("main", "Analysis set", 80),
        ):
            rows.append({**base, "kind": kind, "label": label, "n": n})
    return pl.DataFrame(rows)


def calibration_table() -> pl.DataFrame:
    """A synthetic calibration summary: two series, three bins plus rug rows."""
    rows = []
    for s, series in enumerate(["version a", "version b"]):
        for i, (pred, obs) in enumerate(
            [(0.05 + 0.1 * s, 0.06), (0.15 + 0.1 * s, 0.14), (0.3 + 0.1 * s, 0.28)]
        ):
            rows.append(
                {
                    "kind": "bin",
                    "series": series,
                    "predicted": pred,
                    "estimate": obs,
                    "lo": obs - 0.03,
                    "hi": obs + 0.03,
                    "n": 30 + i,
                    "cohort_n": 200,
                    "events": 40,
                    "stratum": "TranSORCE",
                }
            )
        for k in range(5):
            rows.append(
                {
                    "kind": "rug",
                    "series": series,
                    "predicted": 0.02 + 0.1 * s + 0.05 * k,
                    "estimate": None,
                    "lo": None,
                    "hi": None,
                    "n": None,
                    "cohort_n": 200,
                    "events": 40,
                    "stratum": "TranSORCE",
                }
            )
    for s, series in enumerate(["version a", "version b"]):
        for pred in (0.02, 0.1, 0.2, 0.35):
            rows.append(
                {
                    "kind": "smooth",
                    "series": series,
                    "predicted": pred + 0.1 * s,
                    "estimate": pred + 0.1 * s + 0.01,
                    "lo": None,
                    "hi": None,
                    "n": 40,
                    "cohort_n": 200,
                    "events": 40,
                    "stratum": "TranSORCE",
                }
            )
    return pl.DataFrame(rows)


def test_calibration_draws_curve_bins_bands_rug_and_identity():
    fig, ax = plt.subplots()
    panels.calibration(ax, calibration_table())
    # The identity line, then per series one smoothed curve, one marker-only
    # line of bins and one rug strip; the lo/hi intervals are collections.
    assert len(ax.lines) == 1 + 3 * 2
    identity = ax.lines[0]
    assert list(np.asarray(identity.get_xdata())) == list(np.asarray(identity.get_ydata()))
    assert len(ax.collections) == 2  # one set of bin intervals per series
    assert len({line.get_color() for line in ax.lines[1:]}) == 2
    curves = [line for line in ax.lines[1:] if line.get_linestyle() == "-"]
    assert len(curves) == 2
    assert all(np.asarray(line.get_xdata()).size == 4 for line in curves)
    bins = [line for line in ax.lines[1:] if line.get_marker() == "o"]
    assert len(bins) == 2
    assert all(
        line.get_linestyle() == "None" and np.asarray(line.get_xdata()).size == 3 for line in bins
    )
    plt.close(fig)


def test_calibration_requires_the_role_columns():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="predicted"):
        panels.calibration(ax, pl.DataFrame({"kind": ["bin"], "estimate": [0.1]}))
    plt.close(fig)


def flow_table() -> pl.DataFrame:
    """A synthetic flow summary: a funnel, two exclusions and one field count."""
    return pl.DataFrame(
        [
            {"kind": "main", "label": "Source records", "n": 100},
            {"kind": "main", "label": "Eligible", "n": 90},
            {"kind": "excluded", "label": "First reason", "n": 7},
            {"kind": "excluded", "label": "Second reason", "n": 3},
            {"kind": "note", "label": "Field count", "n": 12},
            {"kind": "main", "label": "Analysis set", "n": 80},
        ]
    )


def test_flow_draws_the_funnel_side_box_and_field_counts():
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots()
    panels.flow(ax, flow_table())
    boxes = [p for p in ax.patches if isinstance(p, FancyBboxPatch)]
    arrows = [p for p in ax.patches if isinstance(p, FancyArrowPatch)]
    # Three funnel boxes plus the exclusions box and the field-counts box.
    assert len(boxes) == 5
    # Two funnel connectors plus the side arrow to the exclusions box.
    assert len(arrows) == 3
    texts = "\n".join(text.get_text() for text in ax.texts)
    for needle in ("Source records\nn = 100", "Analysis set\nn = 80"):
        assert needle in texts
    for needle in ("First reason (7)", "Second reason (3)", "Field count (12)"):
        assert needle in texts
    # The diagram is uncluttered: no ticks and all spines hidden.
    assert ax.get_xticks().size == 0 and ax.get_yticks().size == 0
    assert all(not spine.get_visible() for spine in ax.spines.values())
    plt.close(fig)


def test_flow_draws_only_the_counts_from_the_table():
    fig, ax = plt.subplots()
    panels.flow(ax, flow_table().filter(pl.col("label") != "Eligible"))
    texts = "\n".join(text.get_text() for text in ax.texts)
    assert "Eligible" not in texts
    assert "n = 90" not in texts
    plt.close(fig)


def test_flow_requires_the_role_columns():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="n"):
        panels.flow(ax, pl.DataFrame({"kind": ["main"], "label": ["x"]}))
    plt.close(fig)


def test_flow_requires_a_main_row():
    fig, ax = plt.subplots()
    table = pl.DataFrame({"kind": ["note"], "label": ["x"], "n": [1]})
    with pytest.raises(ValueError, match="main"):
        panels.flow(ax, table)
    plt.close(fig)


def dot_table() -> pl.DataFrame:
    """A synthetic subgroup summary: three groups, three models, two flagged."""
    rows = []
    for label, order, n, events, small, estimate in (
        ("<60", 0, 120, 30, False, 0.12),
        ("60\u201369", 1, 90, 8, True, 0.22),
        ("70+", 2, 0, 0, True, None),
    ):
        rows.append(
            {
                "label": label,
                "group_order": order,
                "kind": "observed",
                "series": "observed",
                "estimate": estimate,
                "lo": estimate - 0.04 if estimate is not None else None,
                "hi": estimate + 0.04 if estimate is not None else None,
                "predicted": None,
                "n": n,
                "events": events,
                "small": small,
                "stratum": "TranSORCE",
                "bound": "pt3a",
            }
        )
        for slot, series in enumerate(("version a", "version b", "version c")):
            rows.append(
                {
                    "label": label,
                    "group_order": order,
                    "kind": "predicted",
                    "series": series,
                    "estimate": None,
                    "lo": None,
                    "hi": None,
                    "predicted": None if n == 0 else 0.10 + 0.05 * slot + 0.03 * order,
                    "n": n,
                    "events": events,
                    "small": small,
                    "stratum": "TranSORCE",
                    "bound": "pt3a",
                }
            )
    return pl.DataFrame(rows)


def test_dot_draws_labelled_groups_intervals_markers_and_flags():
    fig, ax = plt.subplots()
    panels.dot(ax, dot_table())
    labels = [tick.get_text() for tick in ax.get_yticklabels()]
    assert labels == [
        "<60\nn = 120, 30 events",
        "60\u201369\nn = 90, 8 events",
        "70+\nn = 0, 0 events",
    ]
    # One interval per group with an estimate; the empty group keeps its label.
    assert len(ax.containers) == 2
    # Each flagged group draws a hatched band.
    assert len([patch for patch in ax.patches if patch.get_hatch()]) == 2
    # The flagged groups' marks are faded, the unflagged group's are not.
    alphas = {line.get_alpha() for line in ax.lines}
    assert 0.35 in alphas
    assert 1.0 in alphas
    plt.close(fig)


def test_dot_draws_the_groups_in_group_order():
    fig, ax = plt.subplots()
    panels.dot(ax, dot_table().sort("group_order", descending=True))
    labels = [tick.get_text() for tick in ax.get_yticklabels()]
    assert [label.split("\n")[0] for label in labels] == ["<60", "60\u201369", "70+"]
    plt.close(fig)


def test_dot_requires_the_role_columns():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="predicted"):
        panels.dot(ax, dot_table().drop("predicted"))
    plt.close(fig)


def test_draw_figure_labels_panels_and_builds_the_legend():
    spec = figures.load_manuscript(MANUSCRIPT)
    entry = spec["figure"]["model_behaviour"]
    fig = figures.draw_figure(
        entry, {"model_behaviour": synthetic_for_entry(entry, "model_behaviour")}
    )
    assert len(fig.axes) == 3
    # The bold letter and the panel title share one left-aligned title.
    assert [ax.get_title(loc="left") for ax in fig.axes] == [
        f"{figures.build.panel_letter(letter)}  {title}"
        for letter, title in (
            ("A", "Low-risk profile"),
            ("B", "Intermediate-risk profile"),
            ("C", "High-risk profile"),
        )
    ]
    assert all(ax.get_title() == "" for ax in fig.axes)
    assert all(ax.get_xlabel() == "Years since surgery" for ax in fig.axes)
    assert fig.axes[0].get_ylabel() == "Predicted risk of distant metastasis"
    (legend,) = fig.legends
    assert len(legend.legend_handles) == 4
    plt.close(fig)


def test_every_manuscript_entry_builds_from_synthetic_tables(tmp_path):
    spec = figures.load_manuscript(MANUSCRIPT)
    for name, entry in spec["figure"].items():
        summary_dir = tmp_path / name
        summary_dir.mkdir()
        for table_name in {panel["table"] for panel in entry["panels"].values()}:
            synthetic_for_entry(entry, table_name).write_csv(summary_dir / f"{table_name}.csv")
        pdf, png = figures.build_figure(name, MANUSCRIPT, summary_dir, tmp_path / "out")
        assert pdf.parent.name == entry["placement"]
        assert pdf.read_bytes().startswith(b"%PDF")
        assert png.stat().st_size > 0


def test_dot_figure_legend_shows_each_series_marker_and_renamed_labels():
    entry = {
        "size": [120, 60],
        "mosaic": [["A"]],
        "legend": {"loc": "outside lower center", "labels": {"observed": "Observed incidence"}},
        "panels": {"A": {"draw": "dot", "table": "subgroups"}},
    }
    models = {"version a": "Leibovich", "version b": "Leibovich Plus", "version c": "Leibovich"}
    table = dot_table().with_columns(pl.col("series").replace(models))
    fig = figures.draw_figure(entry, {"subgroups": table})
    (legend,) = fig.legends
    labels = [text.get_text() for text in legend.get_texts()]
    markers = [cast(Line2D, handle).get_marker() for handle in legend.legend_handles]
    by_label = dict(zip(labels, markers, strict=True))
    assert by_label["Observed incidence"] == "o"
    assert by_label["Leibovich"] == panels.SERIES_MARKERS["Leibovich"]
    assert by_label["Leibovich Plus"] == panels.SERIES_MARKERS["Leibovich Plus"]
    plt.close(fig)


def test_dot_figures_refuse_a_shared_y_axis():
    entry = {
        "size": [120, 60],
        "mosaic": [["A", "B"]],
        "sharey": True,
        "panels": {
            "A": {"draw": "dot", "table": "subgroups"},
            "B": {"draw": "dot", "table": "subgroups"},
        },
    }
    with pytest.raises(ValueError, match="sharey"):
        figures.draw_figure(entry, {"subgroups": dot_table()})
