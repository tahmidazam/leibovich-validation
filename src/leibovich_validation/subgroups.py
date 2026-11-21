from collections.abc import Iterable

import polars as pl

from leibovich_validation import rbridge, report

PREFIX = "aj_group"

COLUMNS = ("grouping", "label", "horizon_years", "lower", "upper")


def definitions(
    age_cut_points: Iterable[int], leibovich_groups: dict[str, tuple[int, int]]
) -> dict[str, tuple[str, list[str]]]:
    """Each subgrouping's group-label column and its ordered labels."""
    return report.subgroup_definitions(age_cut_points, leibovich_groups)


def labelled(
    predictions: pl.DataFrame,
    completed: pl.DataFrame,
    age_cut_points: Iterable[int],
    leibovich_groups: dict[str, tuple[int, int]],
) -> pl.DataFrame:
    """The predictions with group-label columns from the row-aligned completed dataset."""
    columns = report.group_columns(age_cut_points, leibovich_groups)
    return predictions.hstack(completed.select(*columns.values()))


def measures(
    frame: pl.DataFrame, horizon: float, groupings: dict[str, tuple[str, list[str]]]
) -> dict[str, float]:
    """The Aalen-Johansen incidence of every non-empty group in one `labelled` frame."""
    out: dict[str, float] = {}
    for grouping, (column, labels) in groupings.items():
        for label in labels:
            subset = frame.filter(pl.col(column) == label)
            if subset.is_empty():
                continue
            outcomes = rbridge.Outcomes(subset["time"], subset["event"], subset["entry_time"])
            out[f"{PREFIX}|{grouping}|{label}"] = rbridge.observed_incidence(
                outcomes, horizon
            ).estimate
    return out


def rows(
    horizon: float, intervals: dict[str, rbridge.BootstrapInterval]
) -> list[dict[str, object]]:
    """One row per group at `horizon`, with the percentile interval clipped to [0, 1]."""
    out: list[dict[str, object]] = []
    for key, interval in intervals.items():
        if not key.startswith(f"{PREFIX}|"):
            continue
        _, grouping, label = key.split("|", 2)
        out.append(
            {
                "grouping": grouping,
                "label": label,
                "horizon_years": horizon,
                "lower": max(0.0, interval.lower),
                "upper": min(1.0, interval.upper),
            }
        )
    return out
