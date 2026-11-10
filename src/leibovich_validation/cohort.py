"""The harmonised patient table and the cohort builders.

One row per patient, with the shared columns both cohorts map to.

For TranSORCE: eligibility follows the trial record, missing operation dates
are estimated from the cohort-median randomisation lag, records failing the
date checks are excluded and counted, the event is the first recorded distant
site with second primaries excluded by docs/transorce-other-details-mapping.yaml,
a local-only recurrence or a second primary censors the patient at that event,
death without a distant metastasis is the competing event, and each patient
enters the risk set at randomisation (left truncation).

For the Vasudev cohort: overall stage IV, recorded M1, missing overall stage and
missing sex are excluded and counted. The event is relapse at the DFS time,
an RCC death without a recorded relapse counts as relapse at death, and death
without relapse is the competing event, with an unknown cause read as
other-cause. Where RELAPSE is missing, DFS_STATUS supplies the event status;
the DFS fields supply timing only, capped at the survival time. Patients
enter the risk set at surgery, so entry_time is zero.

The table stays in data/interim/: it is patient-level, so it never enters
git. `validate_patient_table` is the explicit schema check the stage runs
before writing; any violation raises SchemaError and the stage fails.
"""

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import polars as pl
import yaml

from leibovich_validation import leibovich, raw_vasudev

SchemaError = raw_vasudev.SchemaError

# The event labels match the coding the R bridge expects; they are part of
# both the interim schema and the scoring interface.
EVENT_TYPES = ("metastasis", "competing", "censored")
EventType = Literal["metastasis", "competing", "censored"]

SEXES = ("male", "female")

_DAYS_PER_YEAR = 365.25

# Column order of the harmonised table. Columns a source does not record
# stay null; `dfs_inconsistent` flags the Vasudev cohort records whose RELAPSE
# and DFS_STATUS disagree, always False elsewhere. `missing_predictors`
# lists the Leibovich predictor fields whose recorded value was absent or
# unmappable for that patient, including fields that have a default
# (missing nodes count as node-negative), so Table 1 can report
# missingness per predictor.
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

# The predictor names `missing_predictors` can carry, in Table 1 order.
PREDICTOR_NAMES = ("t_stage", "node", "size", "grade", "necrosis")

# TranSORCE writes "pT3a - 4" for pT3a; the main analysis gives it the pT3a
# coefficient and the pT3b bound runs alongside.
TRANSORCE_T_STAGE: dict[str, leibovich.TStage] = {
    "pT1a": "pT1a",
    "pT1b": "pT1b",
    "pT2": "pT2",
    "pT3a - 4": "pT3a",
}

# Free-text "Other: Details" classes in docs/transorce-other-details-mapping.yaml.
OTHER_DETAIL_CLASSES = frozenset({"distant_metastasis", "second_primary", "other"})

# Sites the workbook records as structured flags. Nodal metastasis counts as
# the event. Contralateral kidney metastasis is a probable second primary, so
# it is not in this list.
DISTANT_FLAGS = ("nodal_metastasis", "lung_metastasis", "bone_metastasis", "liver_metastasis")

# Raw codings the predictor and event rules compare against. They are named
# constants because Table S2 quotes them: a changed literal must change the
# published mapping table too.
SIZE_THRESHOLD_MM = 100
TRANSORCE_YES = "Yes"
TRANSORCE_NODE_POSITIVE = "pN1/ pN2"
TRANSORCE_SIZE_OVER_10CM = ">10"
TRANSORCE_SEX = {"M": "male", "F": "female"}
VASUDEV_NODE_POSITIVE = "NI"
VASUDEV_NECROSIS_PRESENT = "Present"
VASUDEV_SEX = {"Male": "male", "Female": "female"}


def load_other_details_mapping(path: Path) -> dict[str, str]:
    """The reviewed "Other: Details" string-to-class mapping.

    Each entry maps one distinct free-text string to distant_metastasis,
    second_primary or other. A value may be a plain string or a mapping with
    `class` plus the review flags `flagged` and `laterality`; only the class
    matters for the event derivation. Anything else raises SchemaError.
    """
    raw = yaml.safe_load(path.read_text())
    try:
        entries = raw["mapping"]
    except TypeError, KeyError:
        raise SchemaError(f"{path}: no 'mapping' section") from None
    classes: dict[str, str] = {}
    for text, value in entries.items():
        flags = value if isinstance(value, dict) else {}
        cls = flags.get("class", value)
        if cls not in OTHER_DETAIL_CLASSES:
            raise SchemaError(
                f"{path}: {text!r} maps to unknown class {cls!r}; "
                f"expected one of {sorted(OTHER_DETAIL_CLASSES)}"
            )
        classes[str(text)] = str(cls)
    return classes


def transorce_tumour(
    t_stage: str | None,
    node_status: str | None,
    max_mm: float | None,
    size_cat: str | None,
    grade: int | None,
    necrosis: str | None,
) -> leibovich.Tumour | None:
    """The Leibovich predictors under the main TranSORCE mapping.

    The measured maximum diameter drives the 10 cm threshold where present,
    with the recorded size category as fallback; ISUP grade stands in for
    Fuhrman grade; "pNx/ pN0" counts as node-negative. Missing necrosis and
    nodal fields read as absent, matching the audit's score computation.
    Returns None when a required predictor is missing, so the caller can
    count it.
    """
    if t_stage not in TRANSORCE_T_STAGE or grade is None:
        return None
    if max_mm is None and size_cat is None:
        return None
    size_ge_10 = (
        (max_mm >= SIZE_THRESHOLD_MM)
        if max_mm is not None
        else (size_cat == TRANSORCE_SIZE_OVER_10CM)
    )
    return leibovich.Tumour(
        t_stage=TRANSORCE_T_STAGE[str(t_stage)],
        node_positive=node_status == TRANSORCE_NODE_POSITIVE,
        size_ge_10cm=size_ge_10,
        grade=int(grade),
        necrosis=necrosis == TRANSORCE_YES,
    )


def transorce_missing_predictors(predictors: dict[str, Any]) -> list[str]:
    """The Leibovich predictor fields with no usable recorded value, in Table 1 order.

    `predictors` carries the raw fields passed to `transorce_tumour`. Node and
    necrosis absences are listed even though they score as negative and
    absent, because Table 1 reports missingness per predictor.
    """
    return [
        name
        for name, absent in (
            ("t_stage", predictors["t_stage"] not in TRANSORCE_T_STAGE),
            ("node", predictors["node_status"] is None),
            ("size", predictors["max_mm"] is None and predictors["size_cat"] is None),
            ("grade", predictors["grade"] is None),
            ("necrosis", predictors["necrosis"] is None),
        )
        if absent
    ]


def _other_details_class(details: str | None, mapping: dict[str, str]) -> str | None:
    """The mapped class of one "Other: Details" string; unmapped strings raise."""
    if details is None:
        return None
    try:
        return mapping[details]
    except KeyError:
        raise SchemaError(
            f"Other: Details string {details!r} is not in the approved mapping; "
            "add it to docs/transorce-other-details-mapping.yaml"
        ) from None


def transorce_event(
    dfs_status: int | None,
    dfs_date: dt.date | None,
    os_status: int | None,
    os_date: dt.date | None,
    distant: bool,
    other_class: str | None,
) -> tuple[EventType, dt.date]:
    """The event type and date of one TranSORCE patient.

    `distant` is whether any structured distant-site flag or a
    distant-metastasis "Other: Details" class applies. With a distant site the
    event is metastasis at the DFS date. A DFS event without a distant site is
    a local-only recurrence or a second primary, so the patient is censored at
    that date, unless the DFS event was the death itself, which is the
    competing event. Without a DFS event, death is the competing event and
    survival is censoring at the last disease-free date.
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
    """Whether the patient's recurrence reaches a distant site.

    Any structured distant-site flag counts; the free-text site counts when
    the approved mapping classes it distant_metastasis. Second primaries and
    "other" sites are not distant events. An "Other Metastasis" flag with no
    recorded detail string keeps the workbook's own coding.
    """
    if any(row[flag] == TRANSORCE_YES for flag in DISTANT_FLAGS):
        return True
    if row["other_metastasis"] == TRANSORCE_YES:
        return other_class in (None, "distant_metastasis")
    return False


@dataclass(frozen=True)
class CohortResult:
    """The harmonised patient table plus the aggregate stage outputs."""

    patients: pl.DataFrame
    flow: pl.DataFrame
    score_agreement: pl.DataFrame


# Alternative predictor readings used to explain score disagreements. Each
# maps a rule name to a function that recomputes the score under one changed
# reading; the first whose recomputed score matches the recorded one explains
# the disagreement. They are ordered, and only plausible single-predictor
# readings appear, so the attribution stays a count by rule and never a
# patient list. The Vasudev cohort has no explainers: its disagreements are reported
# as unexplained rather than guessed at.
_TRANSORCE_SCORE_EXPLAINERS: tuple[tuple[str, Callable[[dict[str, Any]], int | None]], ...] = (
    (
        "size_threshold_from_recorded_category",
        lambda record: _alt_score(**record["predictors"], size_from_category=True),
    ),
    (
        "node_pnx_counted_positive",
        lambda record: _alt_score(**record["predictors"], node_positive=True),
    ),
)


def _score_agreement_rows(
    records: list[dict[str, Any]],
    minimum: float,
    explainers: tuple[tuple[str, Callable[[dict[str, Any]], int | None]], ...] = (),
) -> pl.DataFrame:
    """Computed-versus-recorded Leibovich score agreement, counted by rule.

    Each disagreement is attributed to the first alternative predictor
    reading whose recomputed score matches the recorded one, or counted as
    unexplained. Agreement below `minimum` raises, as it signals a
    predictor mapping error.
    """
    compared = [r for r in records if r["computed"] is not None and r["recorded"] is not None]
    if not compared:
        raise SchemaError("no patient has both a computed and a recorded Leibovich score")
    rows = [
        {"rule": "total_compared", "n": len(compared)},
        {"rule": "agree", "n": sum(r["computed"] == r["recorded"] for r in compared)},
    ]
    disagree = [r for r in compared if r["computed"] != r["recorded"]]
    explained: dict[str, int] = {}
    unexplained = 0
    for record in disagree:
        for name, explain in explainers:
            if explain(record) == record["recorded"]:
                explained[name] = explained.get(name, 0) + 1
                break
        else:
            unexplained += 1
    for name, _ in explainers:
        rows.append({"rule": f"disagrees_as_{name}", "n": explained.get(name, 0)})
    rows.append({"rule": "disagrees_unexplained", "n": unexplained})
    agreement = rows[1]["n"] / rows[0]["n"]
    if agreement < minimum:
        raise SchemaError(
            f"computed-versus-recorded Leibovich score agreement is {agreement:.1%}, "
            f"below the {minimum:.0%} stop threshold; check the predictor "
            "mappings before scoring"
        )
    return pl.DataFrame(rows)


def _alt_score(
    *,
    t_stage: str | None,
    node_status: str | None,
    max_mm: float | None,
    size_cat: str | None,
    grade: int | None,
    necrosis: str | None,
    size_from_category: bool = False,
    node_positive: bool | None = None,
) -> int | None:
    """Recompute a Leibovich score under an alternative predictor reading."""
    if size_from_category:
        max_mm = None
    tumour = transorce_tumour(t_stage, node_status, max_mm, size_cat, grade, necrosis)
    if tumour is None:
        return None
    if node_positive is not None:
        tumour = leibovich.Tumour(
            t_stage=tumour.t_stage,
            node_positive=node_positive,
            size_ge_10cm=tumour.size_ge_10cm,
            grade=tumour.grade,
            necrosis=tumour.necrosis,
        )
    return leibovich.score(tumour)


def validate_patient_table(frame: pl.DataFrame) -> None:
    """The explicit schema check for the harmonised patient table.

    Every required column must be present, identifiers unique, coded values
    inside their allowed sets and times consistent with left truncation
    (0 <= entry < exit). Any violation raises SchemaError naming every
    problem found, so the stage fails loudly rather than writing a bad table.
    """
    problems: list[str] = []
    missing = [c for c in HARMONISED_COLUMNS if c not in frame.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
        raise SchemaError("; ".join(problems))
    if frame["patient_id"].null_count():
        problems.append("patient_id has missing values")
    if frame["patient_id"].n_unique() != frame.height:
        problems.append("patient_id is not unique")
    for column, allowed in (
        ("event", EVENT_TYPES),
        ("sex", SEXES),
        ("t_stage", tuple(leibovich.T_COEF)),
    ):
        bad = sorted(set(frame[column].drop_nulls()) - set(allowed))
        if bad:
            problems.append(f"{column} has values outside {sorted(allowed)}: {bad}")
    bad_grade = frame.filter(pl.col("grade").is_not_null() & ~pl.col("grade").is_between(1, 4))
    if bad_grade.height:
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


def reorigin_at_randomisation(patients: pl.DataFrame) -> pl.DataFrame:
    """The patient table on the randomisation clock.

    Follow-up becomes `time - entry_time` and entry becomes zero, so no delayed
    entry remains. The surgery-to-randomisation interval is kept as
    `prediction_offset`, which the conditional prediction conditions on and
    extends the horizon by. Age stays the age at surgery. Cohorts with no
    delayed entry pass through with a zero offset.
    """
    if (patients["entry_time"] >= patients["time"]).any():
        raise ValueError("follow-up must end after entry in every row")
    return patients.with_columns(
        (pl.col("time") - pl.col("entry_time")).alias("time"),
        pl.col("entry_time").alias("prediction_offset"),
        pl.lit(0.0).alias("entry_time"),
    )


def surgery_clock(patients: pl.DataFrame) -> pl.DataFrame:
    """The patient table on the surgery clock, for the surgery-clock sensitivity analysis.

    Follow-up stays timed from surgery, with entry at randomisation as
    `entry_time` (delayed entry), so the predictions are the unconditional
    risks from surgery and `prediction_offset` is zero.
    """
    if (patients["entry_time"] >= patients["time"]).any():
        raise ValueError("follow-up must end after entry in every row")
    return patients.with_columns(pl.lit(0.0).alias("prediction_offset"))


def _tumour_columns(tumour: leibovich.Tumour | None) -> dict[str, Any]:
    """The harmonised predictor columns; all missing for an incomplete tumour."""
    return {
        "t_stage": None if tumour is None else tumour.t_stage,
        "node_positive": None if tumour is None else tumour.node_positive,
        "size_ge_10cm": None if tumour is None else tumour.size_ge_10cm,
        "grade": None if tumour is None else tumour.grade,
        "necrosis": None if tumour is None else tumour.necrosis,
    }


def harmonise_transorce(
    clinical: pl.DataFrame,
    other_mapping: dict[str, str],
    operation_window: tuple[dt.date, dt.date],
    lag_days: tuple[int, int],
    score_agreement_min: float,
) -> CohortResult:
    """Build the harmonised TranSORCE patient table and its flow counts.

    `operation_window` and `lag_days` carry the audit date checks:
    an operation outside the window or a randomisation lag outside the range
    excludes the record. `other_mapping` comes from `load_other_details_mapping`; a
    free-text string it does not know stops the build.
    """
    start, end = operation_window
    lo, hi = lag_days

    both_dates = clinical.filter(
        pl.col("operation_date").is_not_null() & pl.col("randomisation_date").is_not_null()
    )
    if both_dates.is_empty():
        raise SchemaError("no patient has both operation and randomisation dates")
    lag_series = (both_dates["randomisation_date"] - both_dates["operation_date"]).dt.total_days()
    median_lag = float(cast(Any, lag_series.median()))

    rows: list[dict[str, Any]] = []
    score_records: list[dict[str, Any]] = []
    flow_counts = {
        "workbook_rows": clinical.height,
        "eligible_trial_record": clinical.height,
        "excluded_missing_sex": 0,
        "estimated_operation_date": 0,
        "excluded_operation_outside_window": 0,
        "excluded_randomisation_lag_outside_range": 0,
    }
    for row in clinical.iter_rows(named=True):
        predictors = {
            "t_stage": row["t_stage"],
            "node_status": row["node_status"],
            "max_mm": row["max_diameter_mm"],
            "size_cat": row["tumour_size_category"],
            "grade": row["isup_grade"],
            "necrosis": row["necrosis"],
        }
        tumour = transorce_tumour(**predictors)
        computed = None if tumour is None else leibovich.score(tumour)
        score_records.append(
            {
                "predictors": predictors,
                "computed": computed,
                "recorded": row["leibovich_score"],
            }
        )

        if row["sex"] is None:
            flow_counts["excluded_missing_sex"] += 1
            continue

        operation_date = row["operation_date"]
        estimated = False
        if operation_date is None:
            if row["randomisation_date"] is None:
                raise SchemaError(
                    "operation and randomisation dates are both missing; "
                    "the operation date cannot be estimated"
                )
            # Estimate: randomisation minus the cohort-median
            # operation-to-randomisation lag. Flagged in the patient table.
            operation_date = row["randomisation_date"] - dt.timedelta(days=median_lag)
            estimated = True
            flow_counts["estimated_operation_date"] += 1
        elif not (start <= operation_date <= end):
            flow_counts["excluded_operation_outside_window"] += 1
            continue

        randomisation_date = row["randomisation_date"]
        if randomisation_date is None:
            raise SchemaError(
                "randomisation date is missing but the operation date is "
                "present; the lag cannot be checked"
            )
        lag = (randomisation_date - operation_date).days
        if not estimated and not (lo <= lag <= hi):
            flow_counts["excluded_randomisation_lag_outside_range"] += 1
            continue

        other_class = _other_details_class(row["other_details"], other_mapping)
        distant = _distant_metastasis_recorded(row, other_class)
        event, event_date = transorce_event(
            row["dfs_status"],
            row["dfs_date"],
            row["os_status"],
            row["os_date"],
            distant,
            other_class,
        )

        try:
            sex = TRANSORCE_SEX[row["sex"]]
        except KeyError:
            raise SchemaError(f"unrecognised sex code {row['sex']!r}") from None

        rows.append(
            {
                "cohort": "transorce",
                "patient_id": str(row["trial_identifier"]),
                "age_at_surgery": float(row["age_at_randomisation"]) - lag / _DAYS_PER_YEAR,
                "sex": sex,
                **_tumour_columns(tumour),
                "nephrectomy_type": None,  # not recorded in the workbook
                "arm": row["arm"],
                "operation_year": operation_date.year,
                "entry_time": lag / _DAYS_PER_YEAR,
                "time": (event_date - operation_date).days / _DAYS_PER_YEAR,
                "event": event,
                "estimated_operation_date": estimated,
                "leibovich_score_computed": computed,
                "leibovich_score_recorded": row["leibovich_score"],
                "dfs_inconsistent": False,
                "missing_predictors": transorce_missing_predictors(predictors),
            }
        )

    flow_counts["analysis_set"] = len(rows)
    flow = pl.DataFrame([{"step": step, "n": n} for step, n in flow_counts.items()])
    agreement = _score_agreement_rows(
        score_records, score_agreement_min, explainers=_TRANSORCE_SCORE_EXPLAINERS
    )
    patients = pl.DataFrame(rows).with_columns(
        # An all-complete cohort would otherwise store the column as List(Null).
        pl.col("missing_predictors").cast(pl.List(pl.String))
    )
    validate_patient_table(patients)
    return CohortResult(patients=patients, flow=flow, score_agreement=agreement)


# Vasudev cohort pathological T stage to Leibovich stage. The cohort does not
# split pT1 or pT3: unsplit "TI" is missing, and unsplit "TIII" takes the
# pT3a coefficient in the main analysis, with the pT3b bound applied at
# prediction. The audit uses the same mapping.
VASUDEV_T_STAGE: dict[str, leibovich.TStage] = {
    "TIa": "pT1a",
    "TIb": "pT1b",
    "TII": "pT2",
    "TIII": "pT3a",
    "TIV": "pT4",
}


def vasudev_predictors(
    stage_t: str | None,
    stage_n: str | None,
    size_mm: float | None,
    grade: int | None,
    necrosis: str | None,
) -> dict[str, Any]:
    """Each recorded Vasudev cohort predictor, with a null where it is missing.

    Unsplit "TI" counts as a missing T stage. Unknown nodes score as
    node-negative, because Leibovich gives pNx and pN0 the same zero points and
    the cohort records almost no nodal status. An unrecorded necrosis stays null for the
    imputation step.
    """
    return {
        "t_stage": VASUDEV_T_STAGE.get(stage_t) if stage_t is not None else None,
        "node_positive": stage_n == VASUDEV_NODE_POSITIVE,
        "size_ge_10cm": None if size_mm is None else size_mm >= SIZE_THRESHOLD_MM,
        "grade": None if grade is None else int(grade),
        "necrosis": None if necrosis is None else necrosis == VASUDEV_NECROSIS_PRESENT,
    }


def vasudev_tumour(
    stage_t: str | None,
    stage_n: str | None,
    size_mm: float | None,
    grade: int | None,
    necrosis: str | None,
) -> leibovich.Tumour | None:
    """The complete Leibovich tumour under the main Vasudev cohort mapping.

    The score needs all five predictors, so this returns None when any
    predictor is still missing after `vasudev_predictors` defaults the nodes. The
    caller counts the incomplete sets.
    """
    columns = vasudev_predictors(stage_t, stage_n, size_mm, grade, necrosis)
    if any(value is None for value in columns.values()):
        return None
    return leibovich.Tumour(**columns)


def vasudev_missing_predictors(predictors: dict[str, Any]) -> list[str]:
    """The Leibovich predictor fields with no usable recorded value, in Table 1 order.

    `predictors` carries the raw fields passed to `vasudev_tumour`. Unsplit "TI"
    counts as a missing T stage. Node and necrosis absences are listed even
    though they score as negative and absent, because Table 1 reports missingness per predictor.
    """
    return [
        name
        for name, absent in (
            ("t_stage", predictors["stage_t"] not in VASUDEV_T_STAGE),
            ("node", predictors["stage_n"] is None),
            ("size", predictors["size_mm"] is None),
            ("grade", predictors["grade"] is None),
            ("necrosis", predictors["necrosis"] is None),
        )
        if absent
    ]


@dataclass(frozen=True)
class VasudevEvent:
    """One patient's outcome and the field-anomaly flags the flow counts need."""

    event: EventType
    time: float
    # RELAPSE records a relapse but DFS_STATUS codes the patient disease-free.
    inconsistent: bool
    capped_at_survival: bool
    dfs_missing_fallback: bool
    # RELAPSE is missing and DFS_STATUS supplied the event status.
    status_from_dfs: bool
    # RELAPSE and DFS_STATUS are both missing.
    status_underivable: bool


def _dfs_or_survival(dfs_years: float | None, rcc_years: float | None) -> tuple[float, bool, bool]:
    """(time, capped, fallback): DFS time capped at survival, with survival as fallback.

    The survival time is the timing backbone: a DFS time later than it is an
    internal inconsistency, so the event or censoring time is capped there.
    A missing DFS time falls back to the survival time.
    """
    if dfs_years is None:
        if rcc_years is None:
            raise SchemaError("DFS and survival times are both missing; no follow-up time")
        return rcc_years, False, True
    if rcc_years is not None and dfs_years > rcc_years:
        return rcc_years, True, False
    return dfs_years, False, False


def vasudev_event(
    relapse: str | None,
    dfs_status: str | None,
    dfs_years: float | None,
    os_status: str | None,
    rcc_related_death: str | None,
    rcc_status: str | None,
    rcc_years: float | None,
) -> VasudevEvent:
    """The event type and time of one Vasudev cohort patient.

    RELAPSE is the event record; where it is missing, DFS_STATUS supplies the
    event status. An RCC death without a recorded relapse counts as a relapse
    at the time of death. Any other death, including one of unknown cause, is
    the competing event at the survival time. The DFS fields supply timing
    only, because the audit suggests DFS_STATUS may track death rather than
    relapse: the event or censoring time is the DFS time capped at the
    survival time, falling back to the survival time when DFS time is
    missing. A record marked relapsed but coded disease-free counts as an
    event and is flagged. The event carries the harmonised "metastasis" code:
    it is the event of interest slot, which here holds the wider relapse
    endpoint.
    """
    status_from_dfs = relapse is None and dfs_status is not None
    if relapse == "Yes":
        relapsed = True
    elif relapse == "No":
        relapsed = False
    elif dfs_status == "1:Recurred/Progressed":
        relapsed = True
    elif dfs_status == "0:DiseaseFree":
        relapsed = False
    else:
        relapsed = None  # RELAPSE and DFS_STATUS are both missing
    inconsistent = relapse == "Yes" and dfs_status == "0:DiseaseFree"
    rcc_death = rcc_status == "1:DECEASED" or rcc_related_death == "Yes"
    deceased = os_status == "1:DECEASED" or rcc_death

    capped = False
    dfs_missing = False
    if relapsed is True:
        event: EventType = "metastasis"
        time, capped, dfs_missing = _dfs_or_survival(dfs_years, rcc_years)
    elif rcc_death:
        if rcc_years is None:
            raise SchemaError("an RCC death is recorded but the survival time is missing")
        # RCC death without a recorded relapse counts as relapse at death.
        event, time = "metastasis", rcc_years
    elif deceased:
        if rcc_years is None:
            raise SchemaError("a death is recorded but the survival time is missing")
        # Other-cause and unknown-cause deaths are the competing event.
        event, time = "competing", rcc_years
    else:
        event = "censored"
        time, capped, dfs_missing = _dfs_or_survival(dfs_years, rcc_years)
    return VasudevEvent(
        event=event,
        time=time,
        inconsistent=inconsistent,
        capped_at_survival=capped,
        dfs_missing_fallback=dfs_missing,
        status_from_dfs=status_from_dfs,
        status_underivable=relapsed is None,
    )


def harmonise_vasudev(
    patient: pl.DataFrame,
    sample: pl.DataFrame,
    score_agreement_min: float,
) -> CohortResult:
    """Build the harmonised Vasudev cohort patient table and its flow counts.

    `patient` and `sample` are the raw cBioPortal clinical tables; they join
    one-to-one on PATIENT_ID. Exclusions, in order, are missing overall stage, overall
    stage IV, recorded M1 and missing sex; the four DFS-field anomalies are
    counted among the patients who remain, and the inconsistent records are
    flagged for the censoring sensitivity analysis. Every recorded predictor is
    written and only the missing ones are null, so imputation fills the gaps
    alone, necrosis included. The computed score is defined only for complete
    tumours.
    """
    for name, frame in (("patient", patient), ("sample", sample)):
        if frame["PATIENT_ID"].null_count():
            raise SchemaError(f"data_clinical_{name}.txt has missing PATIENT_IDs")
        if frame["PATIENT_ID"].is_duplicated().any():
            raise SchemaError(f"data_clinical_{name}.txt has duplicated PATIENT_IDs")
    joined = patient.join(sample, on="PATIENT_ID", how="left")
    matched_samples = int(joined["SAMPLE_ID"].is_not_null().sum())

    flow_counts: dict[str, int] = {
        "patient_rows": patient.height,
        "sample_rows": sample.height,
        "patients_without_sample_row": patient.height - matched_samples,
        "sample_rows_without_patient": sample.height - matched_samples,
    }
    score_records: list[dict[str, Any]] = []
    predictor_columns: dict[str, dict[str, Any]] = {}
    computed_scores: dict[str, int | None] = {}
    predictor_sets: dict[str, dict[str, Any]] = {}
    for row in joined.iter_rows(named=True):
        predictors = {
            "stage_t": row["STAGE_T"],
            "stage_n": row["STAGE_N"],
            "size_mm": row["TUMOR_SIZE"],
            "grade": row["FFPE_GRADE"],
            "necrosis": row["NECROSIS"],
        }
        tumour = vasudev_tumour(**predictors)
        computed = None if tumour is None else leibovich.score(tumour)
        patient_id = str(row["PATIENT_ID"])
        predictor_columns[patient_id] = vasudev_predictors(**predictors)
        computed_scores[patient_id] = computed
        predictor_sets[patient_id] = predictors
        score_records.append(
            {
                "predictors": predictors,
                "computed": computed,
                "recorded": row["LEIBOVICH_SCORE"],
            }
        )

    eligible = joined
    for step, mask in (
        ("excluded_missing_overall_stage", pl.col("STAGE").is_null()),
        ("excluded_overall_stage_iv", pl.col("STAGE") == "IV"),
        # eq_missing keeps missing STAGE_M eligible.
        ("excluded_recorded_m1", pl.col("STAGE_M").eq_missing("MI")),
        ("excluded_missing_sex", pl.col("SEX").is_null()),
    ):
        flow_counts[step] = eligible.filter(mask).height
        eligible = eligible.filter(~mask)

    rows: list[dict[str, Any]] = []
    anomalies = {
        "missing_relapse_event_status_from_dfs_status": 0,
        "relapse_and_dfs_status_both_missing": 0,
        "relapsed_but_coded_disease_free": 0,
        "dfs_time_capped_at_survival_time": 0,
        "missing_dfs_time_used_survival_time": 0,
    }
    # A patient with no observable follow-up cannot enter the analysis, so it
    # is counted as the last exclusion.
    flow_counts["excluded_nonpositive_followup"] = 0
    for row in eligible.iter_rows(named=True):
        derived = vasudev_event(
            row["RELAPSE"],
            row["DFS_STATUS"],
            row["DFS_YEARS"],
            row["OS_STATUS"],
            row["RCC_RELATED_DEATH"],
            row["RCC_STATUS"],
            row["RCC_YEARS"],
        )
        anomalies["missing_relapse_event_status_from_dfs_status"] += derived.status_from_dfs
        anomalies["relapse_and_dfs_status_both_missing"] += derived.status_underivable
        anomalies["relapsed_but_coded_disease_free"] += derived.inconsistent
        anomalies["dfs_time_capped_at_survival_time"] += derived.capped_at_survival
        anomalies["missing_dfs_time_used_survival_time"] += derived.dfs_missing_fallback
        if derived.time <= 0:
            flow_counts["excluded_nonpositive_followup"] += 1
            continue

        patient_id = str(row["PATIENT_ID"])
        try:
            sex = VASUDEV_SEX[row["SEX"]]
        except KeyError:
            raise SchemaError(f"unrecognised sex code {row['SEX']!r}") from None
        rows.append(
            {
                "cohort": "vasudev",
                "patient_id": patient_id,
                "age_at_surgery": float(row["AGE"]),
                "sex": sex,
                **predictor_columns[patient_id],
                "nephrectomy_type": (
                    None if row["NEPHRECTOMY_TYPE"] is None else row["NEPHRECTOMY_TYPE"].lower()
                ),
                "arm": None,  # not a trial cohort
                "operation_year": None,  # the cohort records no surgery dates
                "entry_time": 0.0,  # surgery is the origin; no delayed entry
                "time": derived.time,
                "event": derived.event,
                "estimated_operation_date": False,
                "leibovich_score_computed": computed_scores[patient_id],
                "leibovich_score_recorded": row["LEIBOVICH_SCORE"],
                "dfs_inconsistent": derived.inconsistent,
                "missing_predictors": vasudev_missing_predictors(predictor_sets[patient_id]),
                # Feeds the country-matched WHO life-table
                # sensitivity analysis; patient-level, so interim only.
                "country": row["COUNTRY"],
            }
        )

    flow_counts["analysis_set"] = len(rows)
    flow_counts.update(anomalies)
    flow = pl.DataFrame([{"step": step, "n": n} for step, n in flow_counts.items()])
    agreement = _score_agreement_rows(score_records, score_agreement_min)
    patients = pl.DataFrame(rows).with_columns(
        pl.col("missing_predictors").cast(pl.List(pl.String))
    )
    validate_patient_table(patients)
    return CohortResult(patients=patients, flow=flow, score_agreement=agreement)
