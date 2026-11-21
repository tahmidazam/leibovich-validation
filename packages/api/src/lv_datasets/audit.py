"""Aggregate tables shared by the adapters' `audit` hooks."""

from collections.abc import Mapping
from itertools import groupby

import polars as pl


def value_counts(
    frame: pl.DataFrame,
    columns: list[str],
    coverage: Mapping[str, frozenset[str]],
    source: str,
) -> pl.DataFrame:
    """One row per distinct value (including missing) per column, with mapping coverage."""
    rows = []
    for column in columns:
        counts = frame.group_by(column).agg(pl.len().alias("n")).sort(column)
        covered = coverage.get(column)
        for value, n in counts.iter_rows():
            rows.append(
                {
                    "source": source,
                    "column": column,
                    "value": "<missing>" if value is None else str(value),
                    "n": n,
                    "covered_by_mapping_rule": (
                        None if covered is None else value is None or str(value) in covered
                    ),
                }
            )
    return pl.DataFrame(rows)


def missing_counts(frame: pl.DataFrame, columns: list[str], source: str) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "source": source,
                "column": column,
                "n": frame.height,
                "n_missing": frame[column].null_count(),
            }
            for column in columns
        ]
    )


def reverse_km_median(times: list[float], censored: list[bool]) -> float | None:
    """Median follow-up: a Kaplan-Meier curve where censoring is the event."""
    pairs = sorted(zip(times, censored, strict=True))
    at_risk = len(pairs)
    survival = 1.0
    for t, group in groupby(pairs, key=lambda p: p[0]):
        group_list = list(group)
        d = sum(1 for _, c in group_list if c)
        if d:
            survival *= 1 - d / at_risk
        at_risk -= len(group_list)
        if survival <= 0.5:
            return t
    return None


def followup(
    times: list[float], censored: list[bool], horizons: list[float], dataset: str
) -> pl.DataFrame:
    """Follow-up summary from observed event-or-censor times in years."""
    rows = [
        {
            "cohort": dataset,
            "measure": "reverse_km_median_years",
            "value": reverse_km_median(times, censored),
        }
    ]
    rows += [
        {
            "cohort": dataset,
            "measure": f"at_risk_at_{horizon:g}y",
            "value": sum(1 for t in times if t >= horizon),
        }
        for horizon in horizons
    ]
    return pl.DataFrame(rows)


def score_agreement(computed: list[int | None], recorded: list[int | None]) -> pl.DataFrame:
    pairs = [
        (c, r) for c, r in zip(computed, recorded, strict=True) if c is not None and r is not None
    ]
    n = len(pairs)
    agree = sum(c == r for c, r in pairs)
    return pl.DataFrame(
        [
            {
                "n_compared": n,
                "n_agree": agree,
                "pct_agree": round(100 * agree / n, 2) if n else None,
            }
        ]
    )
