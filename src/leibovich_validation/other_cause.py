"""Other-cause O/E: the direct test of Leibovich Plus's competing-risk adjustment.

The observed side is the Aalen-Johansen incidence of death without the event
(the "competing" status) at the horizon; the expected side is the mean of the
model's other-cause risk over the same window, `predict.other_risk_column`.
O/E is computed overall and by age group, inside every resample, so the
intervals are percentile intervals from the same paired resamples as the
other measures.
Only the principal pT3a bound is summarised, as in the other subgroup tables.
"""

import math
from collections.abc import Iterable
from typing import Any, cast

import numpy as np
import polars as pl

from leibovich_validation import predict, rbridge, report

OVERALL = "Overall"
MODEL = predict.LEIBOVICH_PLUS
BOUND = "pt3a"
COMPETING = "competing"

COLUMNS = (
    "cohort",
    "horizon_years",
    "group",
    "model",
    "n",
    "events",
    "observed",
    "observed_lower",
    "observed_upper",
    "mean_predicted",
    "oe_ratio",
    "lower",
    "upper",
    "method",
)


def age_group(age: pl.Series, cut_points: Iterable[int]) -> pl.Series:
    """Each age's band label; bands are left-closed, so age 60 joins "60-69"."""
    cuts = sorted(int(cut) for cut in cut_points)
    labels = np.asarray(report.age_group_labels(cuts))
    return pl.Series("group", labels[np.searchsorted(cuts, age.to_numpy(), side="right")])


def _groups(frame: pl.DataFrame, cut_points: Iterable[int]) -> list[tuple[str, pl.DataFrame]]:
    """The whole frame, then one subframe per age band, empty bands included."""
    labels = report.age_group_labels(cut_points)
    band = age_group(frame["age_at_surgery"], cut_points)
    return [(OVERALL, frame)] + [(label, frame.filter(band == label)) for label in labels]


def observed_incidence(frame: pl.DataFrame, horizon: float) -> float:
    """Aalen-Johansen incidence of death without the event by `horizon`."""
    outcomes = rbridge.Outcomes(frame["time"], frame["event"], frame["entry_time"])
    return rbridge.observed_incidence(outcomes, horizon, COMPETING).estimate


def oe_ratio(observed: float, expected: float) -> float:
    """Observed over mean predicted other-cause risk; NaN when the expected risk is not positive."""
    return observed / expected if expected > 0 else math.nan


def measures(frame: pl.DataFrame, horizon: float, cut_points: Iterable[int]) -> dict[str, float]:
    """Observed other-cause incidence and O/E per group, for the bootstrap statistic.

    Keys are "aj_other|<group>" and "oe_other|<group>". A group with no
    patients, or a zero mean predicted risk, gives no O/E for that resample.
    """
    column = predict.other_risk_column(horizon, BOUND)
    out: dict[str, float] = {}
    for label, subset in _groups(frame, cut_points):
        if subset.is_empty():
            continue
        observed = observed_incidence(subset, horizon)
        expected = float(cast(Any, subset[column].mean()))
        out[f"aj_other|{label}"] = observed
        out[f"oe_other|{label}"] = oe_ratio(observed, expected)
    return out


def deaths(frame: pl.DataFrame, horizon: float) -> int:
    """Deaths without the event by `horizon`."""
    return int(((frame["event"] == COMPETING) & (frame["time"] <= horizon)).sum())


def pooled_expected(frames: Iterable[pl.DataFrame], horizon: float) -> pl.Series:
    """Each patient's other-cause risk averaged over row-aligned completed datasets."""
    column = predict.other_risk_column(horizon, BOUND)
    stacked = np.stack([frame[column].to_numpy() for frame in frames])
    return pl.Series(column, stacked.mean(axis=0))


def rows(
    cohort: str,
    horizon: float,
    patients: pl.DataFrame,
    expected: pl.Series,
    point: dict[str, float],
    intervals: dict[str, rbridge.BootstrapInterval],
    cut_points: Iterable[int],
) -> list[dict[str, object]]:
    """One row per group at `horizon`: size, competing deaths, observed, expected, O/E.

    `patients` holds the age, exit time and event of every scored patient and
    `expected` their predicted other-cause risk, row-aligned. `events` counts
    deaths without the event by the horizon. Percentile bounds come from
    `intervals`; the observed incidence is clipped to [0, 1].
    """
    frame = patients.select("age_at_surgery", "time", "event").with_columns(expected.alias("e"))
    out: list[dict[str, object]] = []
    for label, subset in _groups(frame, cut_points):
        observed = intervals.get(f"aj_other|{label}")
        oe = intervals.get(f"oe_other|{label}")
        mean = float(cast(Any, subset["e"].mean())) if subset.height else None
        out.append(
            {
                "cohort": cohort,
                "horizon_years": horizon,
                "group": label,
                "model": MODEL,
                "n": subset.height,
                "events": deaths(subset, horizon),
                "observed": point.get(f"aj_other|{label}"),
                "observed_lower": None if observed is None else max(0.0, observed.lower),
                "observed_upper": None if observed is None else min(1.0, observed.upper),
                "mean_predicted": mean,
                "oe_ratio": point.get(f"oe_other|{label}"),
                "lower": None if oe is None else oe.lower,
                "upper": None if oe is None else oe.upper,
                "method": "bootstrap",
            }
        )
    return out


def performance_row(
    cohort: str,
    horizon: float,
    point: dict[str, float],
    intervals: dict[str, rbridge.BootstrapInterval],
    n: int,
    events: int,
) -> dict[str, object]:
    """The overall other-cause O/E in the performance table's schema, for Table 2."""
    interval = intervals.get(f"oe_other|{OVERALL}")
    return {
        "cohort": cohort,
        "horizon_years": horizon,
        "bound": BOUND,
        "model": MODEL,
        "metric": "oe_other_cause",
        "estimate": point.get(f"oe_other|{OVERALL}"),
        "lower": None if interval is None else interval.lower,
        "upper": None if interval is None else interval.upper,
        "method": "bootstrap",
        "n": n,
        "events": events,
    }
