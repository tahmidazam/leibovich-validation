from dataclasses import dataclass
from typing import Any

import polars as pl
from lv_datasets import SIZE_THRESHOLD_MM, Context, SchemaError

from lv_vasudev import raw

# Pathological T stage to Leibovich stage. The cohort does not split pT1 or pT3: unsplit
# "TI" is missing, and unsplit "TIII" takes the pT3a coefficient in the main analysis,
# with the pT3b bound applied at prediction.
T_STAGE = {
    "TIa": "pT1a",
    "TIb": "pT1b",
    "TII": "pT2",
    "TIII": "pT3a",
    "TIV": "pT4",
}
NODE_POSITIVE = "NI"
NECROSIS_PRESENT = "Present"
SEX = {"Male": "male", "Female": "female"}

FLOW_STEPS = {
    "patient_rows": ("main", "Patients in the clinical table"),
    "sample_rows": ("note", "Tumour sample rows"),
    "patients_without_sample_row": ("note", "Patients without a sample row"),
    "sample_rows_without_patient": ("note", "Sample rows without a patient"),
    "excluded_missing_overall_stage": ("excluded", "Missing overall stage"),
    "excluded_overall_stage_iv": ("excluded", "Overall stage IV"),
    "excluded_recorded_m1": ("excluded", "Recorded M1"),
    "excluded_missing_sex": ("excluded", "Missing sex"),
    "excluded_nonpositive_followup": ("excluded", "No observable follow-up"),
    "analysis_set": ("main", "Analysis set"),
    "missing_relapse_event_status_from_dfs_status": ("note", "Event status taken from DFS_STATUS"),
    "relapse_and_dfs_status_both_missing": ("note", "RELAPSE and DFS_STATUS both missing"),
    "relapsed_but_coded_disease_free": ("note", "Relapsed but coded disease-free"),
    "dfs_time_capped_at_survival_time": ("note", "DFS time capped at survival time"),
    "missing_dfs_time_used_survival_time": ("note", "Survival time used for missing DFS time"),
}


def predictors(
    stage_t: str | None,
    stage_n: str | None,
    size_mm: float | None,
    grade: int | None,
    necrosis: str | None,
) -> dict[str, Any]:
    """Each recorded predictor, null where missing.

    Unsplit "TI" is a missing T stage. Unknown nodes score as node-negative, since
    Leibovich gives pNx and pN0 the same points and nodal status is almost never
    recorded. Unrecorded necrosis stays null for imputation.
    """
    return {
        "t_stage": T_STAGE.get(stage_t) if stage_t is not None else None,
        "node_positive": stage_n == NODE_POSITIVE,
        "size_ge_10cm": None if size_mm is None else size_mm >= SIZE_THRESHOLD_MM,
        "grade": None if grade is None else int(grade),
        "necrosis": None if necrosis is None else necrosis == NECROSIS_PRESENT,
    }


def missing_predictors(stage_t, stage_n, size_mm, grade, necrosis) -> list[str]:
    """Predictor fields with no usable recorded value, in Table 1 order.

    Node and necrosis absences are listed even though they score as negative and absent.
    """
    return [
        name
        for name, absent in (
            ("t_stage", stage_t not in T_STAGE),
            ("node", stage_n is None),
            ("size", size_mm is None),
            ("grade", grade is None),
            ("necrosis", necrosis is None),
        )
        if absent
    ]


@dataclass(frozen=True)
class Event:
    """One patient's outcome and the field-anomaly flags the flow counts need."""

    event: str
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
    """(time, capped, fallback): DFS time capped at survival time.

    Falls back to survival time when DFS is missing. A DFS time later than survival is
    an inconsistency, so it is capped there.
    """
    if dfs_years is None:
        if rcc_years is None:
            raise SchemaError("DFS and survival times are both missing; no follow-up time")
        return rcc_years, False, True
    if rcc_years is not None and dfs_years > rcc_years:
        return rcc_years, True, False
    return dfs_years, False, False


def event_of(
    relapse: str | None,
    dfs_status: str | None,
    dfs_years: float | None,
    os_status: str | None,
    rcc_related_death: str | None,
    rcc_status: str | None,
    rcc_years: float | None,
) -> Event:
    """The event type and time of one patient.

    RELAPSE is the event record, with DFS_STATUS supplying the status where it is
    missing. An RCC death with no recorded relapse counts as a relapse at death. Any
    other death is the competing event at the survival time. DFS fields supply timing
    only, capped at survival time, because DFS_STATUS may track death rather than
    relapse. A record marked relapsed but coded disease-free counts as an event and is
    flagged. The event uses the "metastasis" code for the wider relapse endpoint.
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
        event = "metastasis"
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
    return Event(
        event=event,
        time=time,
        inconsistent=inconsistent,
        capped_at_survival=capped,
        dfs_missing_fallback=dfs_missing,
        status_from_dfs=status_from_dfs,
        status_underivable=relapsed is None,
    )


def _exclusion(row: dict[str, Any]) -> str | None:
    """The first exclusion step that removes the row: stage, then M1, then sex."""
    if row["STAGE"] is None:
        return "excluded_missing_overall_stage"
    if row["STAGE"] == "IV":
        return "excluded_overall_stage_iv"
    if row["STAGE_M"] == "MI":
        return "excluded_recorded_m1"
    if row["SEX"] is None:
        return "excluded_missing_sex"
    return None


def harmonise(ctx: Context) -> tuple[pl.DataFrame, dict[str, int]]:
    """Every raw patient in the harmonised columns, with its exclusion and the note counts.

    The patient and sample tables are joined on PATIENT_ID. Exclusions run in order:
    missing overall stage, stage IV, recorded M1, missing sex, then no observable
    follow-up. DFS-field anomalies are counted among the patients not excluded before
    follow-up, and inconsistent records are flagged for the censoring sensitivity
    analysis. Missing predictors are left null for imputation.
    """
    patient = raw.read_patient(ctx.data_dir / "data_clinical_patient.txt")
    sample = raw.read_sample(ctx.data_dir / "data_clinical_sample.txt")
    for name, frame in (("patient", patient), ("sample", sample)):
        if frame["PATIENT_ID"].null_count():
            raise SchemaError(f"data_clinical_{name}.txt has missing PATIENT_IDs")
        if frame["PATIENT_ID"].is_duplicated().any():
            raise SchemaError(f"data_clinical_{name}.txt has duplicated PATIENT_IDs")
    joined = patient.join(sample, on="PATIENT_ID", how="left")
    matched_samples = int(joined["SAMPLE_ID"].is_not_null().sum())
    notes = {
        "patient_rows": patient.height,
        "sample_rows": sample.height,
        "patients_without_sample_row": patient.height - matched_samples,
        "sample_rows_without_patient": sample.height - matched_samples,
        "missing_relapse_event_status_from_dfs_status": 0,
        "relapse_and_dfs_status_both_missing": 0,
        "relapsed_but_coded_disease_free": 0,
        "dfs_time_capped_at_survival_time": 0,
        "missing_dfs_time_used_survival_time": 0,
    }
    rows = []
    for row in joined.iter_rows(named=True):
        fields = {
            "stage_t": row["STAGE_T"],
            "stage_n": row["STAGE_N"],
            "size_mm": row["TUMOR_SIZE"],
            "grade": row["FFPE_GRADE"],
            "necrosis": row["NECROSIS"],
        }
        out: dict[str, Any] = {
            "patient_id": str(row["PATIENT_ID"]),
            **predictors(**fields),
            "leibovich_score_recorded": row["LEIBOVICH_SCORE"],
            "missing_predictors": missing_predictors(**fields),
            "excluded": _exclusion(row),
            "country": row["COUNTRY"],
        }
        if out["excluded"] is None:
            derived = event_of(
                row["RELAPSE"],
                row["DFS_STATUS"],
                row["DFS_YEARS"],
                row["OS_STATUS"],
                row["RCC_RELATED_DEATH"],
                row["RCC_STATUS"],
                row["RCC_YEARS"],
            )
            notes["missing_relapse_event_status_from_dfs_status"] += derived.status_from_dfs
            notes["relapse_and_dfs_status_both_missing"] += derived.status_underivable
            notes["relapsed_but_coded_disease_free"] += derived.inconsistent
            notes["dfs_time_capped_at_survival_time"] += derived.capped_at_survival
            notes["missing_dfs_time_used_survival_time"] += derived.dfs_missing_fallback
            if derived.time <= 0:
                out["excluded"] = "excluded_nonpositive_followup"
            else:
                out |= {
                    "age_at_surgery": float(row["AGE"]),
                    "sex": SEX[row["SEX"]],
                    "nephrectomy_type": (
                        None if row["NEPHRECTOMY_TYPE"] is None else row["NEPHRECTOMY_TYPE"].lower()
                    ),
                    "entry_time": 0.0,  # surgery is the origin; no delayed entry
                    "time": derived.time,
                    "event": derived.event,
                    "estimated_operation_date": False,
                    "dfs_inconsistent": derived.inconsistent,
                }
        rows.append(out)
    frame = pl.DataFrame(rows, infer_schema_length=None).with_columns(
        pl.col("missing_predictors").cast(pl.List(pl.String)),
        pl.lit(None).alias("arm"),  # not a trial cohort
        pl.lit(None).alias("operation_year"),  # the cohort records no surgery dates
    )
    return frame, notes
