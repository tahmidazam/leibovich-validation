from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import polars as pl


class SchemaError(ValueError):
    """Raw or harmonised data does not match the declared schema."""


# The event labels match the coding the R bridge expects.
EVENT_TYPES = ("metastasis", "competing", "censored")
SEXES = ("male", "female")

# Leibovich points. `T_POINTS` also fixes the T-stage vocabulary.
T_POINTS: dict[str, int] = {
    "pT1a": 0,
    "pT1b": 2,
    "pT2": 3,
    "pT3a": 4,
    "pT3b": 4,
    "pT3c": 4,
    "pT4": 4,
}
T_STAGES = tuple(T_POINTS)
GRADE_POINTS: dict[int, int] = {1: 0, 2: 0, 3: 1, 4: 3}
NODE_POSITIVE_POINTS = 2
SIZE_GE_10CM_POINTS = 1
NECROSIS_POINTS = 1

# A tumour of at least this diameter counts as 10 cm or greater.
SIZE_THRESHOLD_MM = 100
DAYS_PER_YEAR = 365.25

# Column order of the harmonised table. Columns a dataset does not record stay null.
# `dfs_inconsistent` flags records whose relapse and DFS status disagree.
# `missing_predictors` lists predictor fields that were absent or unmappable, including
# defaulted ones such as nodes, for Table 1. Adapters supply every column except `cohort`
# and `leibovich_score_computed`, which `finalise` derives.
HARMONISED_COLUMNS = (
    "cohort",
    "patient_id",
    "age_at_surgery",
    "sex",
    "t_stage",
    "node_positive",
    "size_ge_10cm",
    "grade",
    "necrosis",
    "nephrectomy_type",
    "arm",
    "operation_year",
    "entry_time",
    "time",
    "event",
    "estimated_operation_date",
    "leibovich_score_computed",
    "leibovich_score_recorded",
    "dfs_inconsistent",
    "missing_predictors",
)

PREDICTOR_COLUMNS = ("t_stage", "node_positive", "size_ge_10cm", "grade", "necrosis")

# The predictor names `missing_predictors` can carry, in Table 1 order.
PREDICTOR_NAMES = ("t_stage", "node", "size", "grade", "necrosis")

# Table S2 sections in display order; `mapping_rows` rows name one of them.
S2_SECTIONS = (
    "Predictors",
    "Eligibility and design",
    "Timing",
    "Event and competing event",
    "Score agreement",
)


def leibovich_score(
    t_stage: pl.Expr,
    node_positive: pl.Expr,
    size_ge_10cm: pl.Expr,
    grade: pl.Expr,
    necrosis: pl.Expr,
) -> pl.Expr:
    """The Leibovich score of five predictor expressions; null if any is null."""
    return (
        t_stage.replace_strict(T_POINTS, return_dtype=pl.Int64)
        + NODE_POSITIVE_POINTS * node_positive.cast(pl.Int64)
        + SIZE_GE_10CM_POINTS * size_ge_10cm.cast(pl.Int64)
        + grade.replace_strict(GRADE_POINTS, return_dtype=pl.Int64)
        + NECROSIS_POINTS * necrosis.cast(pl.Int64)
    )


def score_of(columns: Mapping[str, Any]) -> int | None:
    """The Leibovich score of one patient's predictor columns; null if any is missing."""
    if any(columns[name] is None for name in PREDICTOR_COLUMNS):
        return None
    return pl.select(leibovich_score(*(pl.lit(columns[name]) for name in PREDICTOR_COLUMNS))).item()


FlowKind = Literal["main", "excluded", "note"]


@dataclass(frozen=True)
class Context:
    """What `harmonise` receives: the dataset's raw-data directory and parameter section."""

    data_dir: Path
    params: dict[str, Any]


@dataclass(frozen=True)
class Dataset:
    """One dataset: how to read and harmonise it, and how to label its flow.

    `harmonise` returns every raw row in the harmonised columns, an `excluded` column
    (null for retained rows, else the first exclusion step that removed it), an optional
    `score_explained_by` column (rule name for rows whose computed and recorded scores
    disagree), any extra columns to carry, and a dict of note counts.
    `flow_steps` maps each flow step to (kind, display label) in flow-table order; it
    holds every exclusion step, every note and `analysis_set`. A label is formatted with
    the dataset's parameter section.

    The hooks are optional. `audit(ctx, horizons)` returns the dataset's aggregate audit
    tables by output name. `mapping_rows(params, analysis)` returns its Table S2 rows
    (`section`, `item`, `source`, `raw_coding`, `rule`, `missing`) given its parameter
    section and the analysis parameters. `mapping_tables()` returns further mapping
    tables by output name.
    """

    name: str
    label: str
    flow_steps: dict[str, tuple[FlowKind, str]]
    harmonise: Callable[[Context], tuple[pl.DataFrame, dict[str, int]]]
    audit: Callable[[Context, list[float]], dict[str, pl.DataFrame]] | None = None
    mapping_rows: Callable[[dict[str, Any], dict[str, Any]], list[dict[str, str]]] | None = None
    mapping_tables: Callable[[], dict[str, pl.DataFrame]] | None = None


def validate_patient_table(frame: pl.DataFrame) -> None:
    """Check the harmonised patient table against its schema.

    Requires every column, unique identifiers, coded values in their allowed sets and
    0 <= entry < exit. Raises SchemaError listing every problem found.
    """
    missing = [c for c in HARMONISED_COLUMNS if c not in frame.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")
    problems: list[str] = []
    if frame["patient_id"].null_count():
        problems.append("patient_id has missing values")
    if frame["patient_id"].n_unique() != frame.height:
        problems.append("patient_id is not unique")
    for column, allowed in (("event", EVENT_TYPES), ("sex", SEXES), ("t_stage", T_STAGES)):
        bad = sorted(set(frame[column].drop_nulls()) - set(allowed))
        if bad:
            problems.append(f"{column} has values outside {sorted(allowed)}: {bad}")
    if frame.filter(pl.col("grade").is_not_null() & ~pl.col("grade").is_between(1, 4)).height:
        problems.append("grade is outside 1-4")
    if frame["time"].null_count() or frame["entry_time"].null_count():
        problems.append("time or entry_time has missing values")
    else:
        if (frame["time"] <= 0).any():
            problems.append("time is not positive in every row")
        if (frame["entry_time"] < 0).any():
            problems.append("entry_time is negative in some rows")
        if (frame["entry_time"] >= frame["time"]).any():
            problems.append("entry_time is not before time in every row")
    if problems:
        raise SchemaError("; ".join(problems))
