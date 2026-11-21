import datetime as dt
from pathlib import Path
from typing import Any, cast

import polars as pl
import yaml
from lv_datasets import DAYS_PER_YEAR, SIZE_THRESHOLD_MM, Context, SchemaError, score_of

from lv_transorce import raw

# TranSORCE writes "pT3a - 4" for pT3a; the main analysis gives it the pT3a
# coefficient and the pT3b bound runs alongside.
T_STAGE = {
    "pT1a": "pT1a",
    "pT1b": "pT1b",
    "pT2": "pT2",
    "pT3a - 4": "pT3a",
}

# Free-text "Other: Details" classes in other_details_mapping.yaml.
OTHER_DETAIL_CLASSES = frozenset({"distant_metastasis", "second_primary", "other"})
OTHER_DETAILS_MAPPING = Path(__file__).with_name("other_details_mapping.yaml")

# Sites the workbook records as structured flags. Nodal metastasis counts as
# the event. Contralateral kidney metastasis is a probable second primary, so
# it is not in this list.
DISTANT_FLAGS = ("nodal_metastasis", "lung_metastasis", "bone_metastasis", "liver_metastasis")

# Raw codings the predictor and event rules compare against. They are named
# constants because Table S2 quotes them: a changed literal must change the
# published mapping table too.
YES = "Yes"
NODE_POSITIVE = "pN1/ pN2"
SIZE_OVER_10CM = ">10"
SEX = {"M": "male", "F": "female"}

FLOW_STEPS = {
    "workbook_rows": ("main", "Rows in the TranSORCE workbook"),
    "eligible_trial_record": ("main", "Eligible by the trial record"),
    "excluded_missing_sex": ("excluded", "Missing sex"),
    "estimated_operation_date": ("note", "Operation date estimated"),
    "excluded_operation_outside_window": ("excluded", "Operation date outside the trial window"),
    "excluded_randomisation_lag_outside_range": (
        "excluded",
        "Randomisation lag outside {randomisation_lag_days[0]} to {randomisation_lag_days[1]} days",
    ),
    "analysis_set": ("main", "Analysis set"),
}


def load_other_details_mapping() -> dict[str, dict[str, Any]]:
    """The reviewed "Other: Details" mapping: each string's `class` and review flags.

    A value is a class (distant_metastasis, second_primary or other), or a mapping with
    `class` plus the `flagged` and `laterality` review flags. An unknown class raises
    SchemaError.
    """
    entries = {}
    for text, value in yaml.safe_load(OTHER_DETAILS_MAPPING.read_text())["mapping"].items():
        flags = value if isinstance(value, dict) else {"class": value}
        if flags["class"] not in OTHER_DETAIL_CLASSES:
            raise SchemaError(
                f"{OTHER_DETAILS_MAPPING.name}: {text!r} maps to unknown class "
                f"{flags['class']!r}; expected one of {sorted(OTHER_DETAIL_CLASSES)}"
            )
        entries[str(text)] = flags
    return entries


def predictors(
    t_stage: str | None,
    node_status: str | None,
    max_mm: float | None,
    size_cat: str | None,
    grade: int | None,
    necrosis: str | None,
) -> dict[str, Any]:
    """The Leibovich predictor columns under the main TranSORCE mapping.

    Measured diameter sets the 10 cm threshold, with the size category as fallback.
    ISUP grade stands in for Fuhrman grade and "pNx/ pN0" is node-negative. Missing
    necrosis and nodal fields read as absent. Every column is null if a required
    predictor is missing.
    """
    if t_stage not in T_STAGE or grade is None or (max_mm is None and size_cat is None):
        return dict.fromkeys(("t_stage", "node_positive", "size_ge_10cm", "grade", "necrosis"))
    return {
        "t_stage": T_STAGE[str(t_stage)],
        "node_positive": node_status == NODE_POSITIVE,
        "size_ge_10cm": (
            max_mm >= SIZE_THRESHOLD_MM if max_mm is not None else size_cat == SIZE_OVER_10CM
        ),
        "grade": int(grade),
        "necrosis": necrosis == YES,
    }


def _explained_by(fields: dict[str, Any], recorded: int | None) -> str | None:
    """The first alternative predictor reading that reproduces a disagreeing recorded score."""
    columns = predictors(**fields)
    if recorded is None or score_of(columns) in (None, recorded):
        return None
    by_category = predictors(**fields | {"max_mm": None})
    by_node = columns | {"node_positive": True}
    for rule, alternative in (
        ("size_threshold_from_recorded_category", by_category),
        ("node_pnx_counted_positive", by_node),
    ):
        if score_of(alternative) == recorded:
            return rule
    return None


def missing_predictors(fields: dict[str, Any]) -> list[str]:
    """Predictor fields with no usable recorded value, in Table 1 order.

    `fields` holds the raw fields passed to `predictors`. Node and necrosis absences are
    listed even though they score as negative and absent.
    """
    return [
        name
        for name, absent in (
            ("t_stage", fields["t_stage"] not in T_STAGE),
            ("node", fields["node_status"] is None),
            ("size", fields["max_mm"] is None and fields["size_cat"] is None),
            ("grade", fields["grade"] is None),
            ("necrosis", fields["necrosis"] is None),
        )
        if absent
    ]


def _other_details_class(details: str | None, mapping: dict[str, dict[str, Any]]) -> str | None:
    """The mapped class of one "Other: Details" string; unmapped strings raise."""
    if details is None:
        return None
    try:
        return mapping[details]["class"]
    except KeyError:
        raise SchemaError(
            f"Other: Details string {details!r} is not in the approved mapping; "
            "add it to other_details_mapping.yaml"
        ) from None


def event_of(
    dfs_status: int | None,
    dfs_date: dt.date | None,
    os_status: int | None,
    os_date: dt.date | None,
    distant: bool,
    other_class: str | None,
) -> tuple[str, dt.date]:
    """The event type and date of one patient.

    `distant` is true if a distant-site flag or a distant_metastasis "Other: Details"
    class applies, which makes the event metastasis at the DFS date. A DFS event with
    no distant site is a local recurrence or second primary and censors the patient,
    unless it is the death itself (competing event). With no DFS event, death is the
    competing event and survival is censoring at the last disease-free date.
    """
    if dfs_status is None or dfs_date is None:
        raise SchemaError("dfs_status or dfs_date is missing; the event cannot be derived")
    if dfs_status not in (0, 1):
        raise SchemaError(f"unexpected dfs_status {dfs_status!r}; expected 0 or 1")
    if os_status not in (0, 1):
        raise SchemaError(f"unexpected os_status {os_status!r}; expected 0 or 1")
    if dfs_status == 1:
        if distant:
            return "metastasis", dfs_date
        if other_class in {"second_primary", "other"}:
            return "censored", dfs_date
        if os_status == 1 and os_date == dfs_date:
            return "competing", dfs_date
        return "censored", dfs_date
    if os_status == 1:
        if os_date is None:
            raise SchemaError("os_status records a death but os_date is missing")
        return "competing", os_date
    return "censored", dfs_date


def _distant_metastasis_recorded(row: dict[str, Any], other_class: str | None) -> bool:
    """Whether the recurrence reaches a distant site.

    A distant-site flag counts, as does a free-text site mapped to distant_metastasis.
    An "Other Metastasis" flag with no detail string keeps the workbook's coding.
    """
    if any(row[flag] == YES for flag in DISTANT_FLAGS):
        return True
    if row["other_metastasis"] == YES:
        return other_class in (None, "distant_metastasis")
    return False


def harmonise(ctx: Context) -> tuple[pl.DataFrame, dict[str, int]]:
    """Every raw patient in the harmonised columns, with its exclusion and the note counts.

    Exclusions run in order: missing sex, operation date outside `operation_window`, then
    randomisation lag outside `randomisation_lag_days`. A missing operation date is
    estimated as randomisation minus the cohort-median lag and flagged; it skips both date
    checks. A free-text "Other: Details" string missing from the approved mapping stops the
    build.
    """
    clinical = raw.read_clinical(ctx.data_dir / "clinical.xlsx")
    mapping = load_other_details_mapping()
    start, end = (dt.date.fromisoformat(str(d)) for d in ctx.params["operation_window"])
    lo, hi = ctx.params["randomisation_lag_days"]

    both_dates = clinical.filter(
        pl.col("operation_date").is_not_null() & pl.col("randomisation_date").is_not_null()
    )
    if both_dates.is_empty():
        raise SchemaError("no patient has both operation and randomisation dates")
    lag_series = (both_dates["randomisation_date"] - both_dates["operation_date"]).dt.total_days()
    median_lag = float(cast(Any, lag_series.median()))

    notes = {
        "workbook_rows": clinical.height,
        "eligible_trial_record": clinical.height,
        "estimated_operation_date": 0,
    }
    rows = []
    for row in clinical.iter_rows(named=True):
        fields = {
            "t_stage": row["t_stage"],
            "node_status": row["node_status"],
            "max_mm": row["max_diameter_mm"],
            "size_cat": row["tumour_size_category"],
            "grade": row["isup_grade"],
            "necrosis": row["necrosis"],
        }
        out: dict[str, Any] = {
            "patient_id": str(row["trial_identifier"]),
            **predictors(**fields),
            "leibovich_score_recorded": row["leibovich_score"],
            "score_explained_by": _explained_by(fields, row["leibovich_score"]),
            "missing_predictors": missing_predictors(fields),
            "excluded": None,
        }
        rows.append(out)
        if row["sex"] is None:
            out["excluded"] = "excluded_missing_sex"
            continue

        operation_date = row["operation_date"]
        randomisation_date = row["randomisation_date"]
        estimated = operation_date is None
        if estimated:
            if randomisation_date is None:
                raise SchemaError(
                    "operation and randomisation dates are both missing; "
                    "the operation date cannot be estimated"
                )
            operation_date = randomisation_date - dt.timedelta(days=median_lag)
            notes["estimated_operation_date"] += 1
        elif not start <= operation_date <= end:
            out["excluded"] = "excluded_operation_outside_window"
            continue
        if randomisation_date is None:
            raise SchemaError(
                "randomisation date is missing but the operation date is "
                "present; the lag cannot be checked"
            )
        lag = (randomisation_date - operation_date).days
        if not estimated and not lo <= lag <= hi:
            out["excluded"] = "excluded_randomisation_lag_outside_range"
            continue

        other_class = _other_details_class(row["other_details"], mapping)
        event, event_date = event_of(
            row["dfs_status"],
            row["dfs_date"],
            row["os_status"],
            row["os_date"],
            _distant_metastasis_recorded(row, other_class),
            other_class,
        )
        out |= {
            "age_at_surgery": float(row["age_at_randomisation"]) - lag / DAYS_PER_YEAR,
            "sex": SEX[row["sex"]],
            "nephrectomy_type": None,  # not recorded in the workbook
            "arm": row["arm"],
            "operation_year": operation_date.year,
            "entry_time": lag / DAYS_PER_YEAR,
            "time": (event_date - operation_date).days / DAYS_PER_YEAR,
            "event": event,
            "estimated_operation_date": estimated,
            "dfs_inconsistent": False,
        }
    frame = pl.DataFrame(rows, infer_schema_length=None).with_columns(
        pl.col("missing_predictors").cast(pl.List(pl.String))
    )
    return frame, notes
