"""Blind audit of the raw cohorts: counts and value lists only, blind to outcomes.

Runs structural checks fixed before any outcome is seen: distinct values of each
categorical field, missing values per predictor, the TranSORCE date checks, how often
each Vasudev cohort event rule fires, unsplit pT1 and pT3 counts, missing sex, agreement
between computed and recorded Leibovich scores, follow-up, and the distinct TranSORCE
"Other: Details" strings.

The audit never cross-tabulates a predictor against an outcome and never computes a
model prediction. Every written frame passes through ``assert_aggregate``, which
refuses anything shaped like one row per patient.
"""

import datetime as dt
from itertools import groupby
from pathlib import Path

import polars as pl

from leibovich_validation import cohort, leibovich, raw_transorce, raw_vasudev

_ID_LIKE = {
    "patient_id",
    "PATIENT_ID",
    "SAMPLE_ID",
    "STUDY_ID",
    "trial_identifier",
    "icaird_ref",
    "image_tag",
    "patient_id_number",
}
_MAX_ROWS = 100_000

# Predictor columns whose non-null values a mapping rule must explain. The
# covered values are the raw readers' declared categories, which already
# carry the values the cohort mappings work with (unsplit "TI", the
# "pT3a - 4" spelling of T3a); the readers reject anything outside them.
_VASUDEV_DECLARED = raw_vasudev.PATIENT_CATEGORIES | raw_vasudev.SAMPLE_CATEGORIES
_VASUDEV_COVERAGE: dict[str, frozenset[str]] = {
    column: _VASUDEV_DECLARED[column]
    for column in (
        "STAGE_T",
        "STAGE_N",
        "STAGE_M",
        "NECROSIS",
        "STAGE",
        "SEX",
        "NEPHRECTOMY_TYPE",
        "RELAPSE",
        "OS_STATUS",
        "RCC_RELATED_DEATH",
        "DFS_STATUS",
        "RCC_STATUS",
    )
}

_TRANSORCE_COVERAGE: dict[str, frozenset[str]] = {
    column: raw_transorce.CATEGORIES[column]
    for column in (
        "t_stage",
        "node_status",
        "tumour_size_category",
        "necrosis",
        "histology",
        "arm",
        "sex",
        "leibovich_risk_group",
    )
}


def assert_aggregate(frame: pl.DataFrame, name: str) -> None:
    offending = _ID_LIKE & set(frame.columns)
    if offending:
        raise ValueError(
            f"{name}: audit output would carry patient identifiers {sorted(offending)}"
        )
    if frame.height > _MAX_ROWS:
        raise ValueError(f"{name}: {frame.height} rows is not an aggregate table")


def _value_counts(
    frame: pl.DataFrame,
    columns: list[str],
    coverage: dict[str, frozenset[str]],
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


def _missing_counts(frame: pl.DataFrame, columns: list[str], source: str) -> pl.DataFrame:
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


def _at_risk(times: list[float], horizon: float) -> int:
    return sum(1 for t in times if t >= horizon)


def _followup(
    times: list[float], censored: list[bool], horizons: list[float], cohort: str
) -> pl.DataFrame:
    """Follow-up summary from observed event-or-censor times in years."""
    rows = [
        {
            "cohort": cohort,
            "measure": "reverse_km_median_years",
            "value": reverse_km_median(times, censored),
        }
    ]
    for horizon in horizons:
        rows.append(
            {
                "cohort": cohort,
                "measure": f"at_risk_at_{horizon:g}y",
                "value": _at_risk(times, horizon),
            }
        )
    return pl.DataFrame(rows)


def _score_agreement(computed: list[int | None], recorded: list[int | None]) -> pl.DataFrame:
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


def _vasudev_computed_score(
    t_stage: str | None,
    stage_n: str | None,
    size_mm: float | None,
    grade: int | None,
    necrosis: str | None,
) -> int | None:
    """Leibovich score under the main mapping in `cohort.vasudev_tumour`."""
    tumour = cohort.vasudev_tumour(t_stage, stage_n, size_mm, grade, necrosis)
    return None if tumour is None else leibovich.score(tumour)


def _transorce_computed_score(
    t_stage: str | None,
    node_status: str | None,
    max_mm: float | None,
    size_cat: str | None,
    grade: int | None,
    necrosis: str | None,
) -> int | None:
    """Leibovich score under the main mapping in `cohort.transorce_tumour`."""
    tumour = cohort.transorce_tumour(t_stage, node_status, max_mm, size_cat, grade, necrosis)
    return None if tumour is None else leibovich.score(tumour)


def vasudev_outputs(
    patient: pl.DataFrame, sample: pl.DataFrame, horizons: list[float]
) -> dict[str, pl.DataFrame]:
    outputs: dict[str, pl.DataFrame] = {}
    patient_cats = list(raw_vasudev.PATIENT_CATEGORIES)
    sample_cats = list(raw_vasudev.SAMPLE_CATEGORIES)
    outputs["vasudev_categorical_values"] = pl.concat(
        [
            _value_counts(patient, patient_cats, _VASUDEV_COVERAGE, "patient"),
            _value_counts(sample, sample_cats, _VASUDEV_COVERAGE, "sample"),
        ]
    )

    outputs["vasudev_missing"] = pl.concat(
        [
            _missing_counts(
                patient,
                [raw_vasudev.MONTHS_TO_YEARS.get(c, c) for c in raw_vasudev.PATIENT_COLUMNS],
                "patient",
            ),
            _missing_counts(sample, list(raw_vasudev.SAMPLE_COLUMNS), "sample"),
        ]
    )

    rule_counts = patient.select(
        (pl.col("RELAPSE") == "Yes").sum().alias("relapse_yes"),
        ((pl.col("RELAPSE") == "Yes") & (pl.col("DFS_STATUS") == "1:Recurred/Progressed"))
        .sum()
        .alias("relapse_at_dfs_time"),
        ((pl.col("RCC_RELATED_DEATH") == "Yes") & (pl.col("RELAPSE") == "No"))
        .sum()
        .alias("rcc_death_without_relapse"),
        ((pl.col("RELAPSE") == "Yes") & (pl.col("DFS_STATUS") == "0:DiseaseFree"))
        .sum()
        .alias("relapsed_but_coded_disease_free"),
        (pl.col("DFS_YEARS") > pl.col("RCC_YEARS")).sum().alias("dfs_time_later_than_survival"),
        (pl.col("OS_STATUS") == "1:DECEASED").sum().alias("deceased"),
        (pl.col("RCC_STATUS") == "1:DECEASED").sum().alias("rcc_deceased"),
        ((pl.col("OS_STATUS") == "1:DECEASED") & (pl.col("RCC_STATUS") == "0:LIVING"))
        .sum()
        .alias("other_cause_deceased"),
        (pl.col("STAGE") == "IV").sum().alias("overall_stage_iv"),
    )
    outputs["vasudev_event_rules"] = rule_counts.unpivot(variable_name="rule", value_name="n")

    unsplit = sample.group_by("STAGE_T").agg(pl.len().alias("n"))
    outputs["vasudev_unsplit_stage"] = pl.DataFrame(
        [
            {
                "stage_t": value,
                "n": n,
                "interpretation": {
                    "TI": "unsplit pT1, treated as missing",
                    "TIII": "unsplit pT3, pT3a coefficient in main analysis",
                }.get(str(value), ""),
            }
            for value, n in unsplit.iter_rows()
            if str(value) in {"TI", "TIII"}
        ],
        schema={"stage_t": pl.String, "n": pl.UInt32, "interpretation": pl.String},
    )

    joined = patient.join(sample, on="PATIENT_ID", how="left")
    computed = [
        _vasudev_computed_score(t, n, s, g, ne)
        for t, n, s, g, ne in joined.select(
            "STAGE_T", "STAGE_N", "TUMOR_SIZE", "FFPE_GRADE", "NECROSIS"
        ).iter_rows()
    ]
    recorded = joined["LEIBOVICH_SCORE"].to_list()
    outputs["vasudev_score_agreement"] = _score_agreement(computed, recorded)

    usable = patient.filter(pl.col("DFS_YEARS").is_not_null())
    outputs["vasudev_followup"] = _followup(
        usable["DFS_YEARS"].to_list(),
        (usable["DFS_STATUS"] == "0:DiseaseFree").fill_null(False).to_list(),
        horizons,
        "vasudev",
    )

    outputs["vasudev_counts"] = pl.DataFrame(
        [
            {"measure": "patients", "value": patient.height},
            {"measure": "samples", "value": sample.height},
            {"measure": "unique_patient_ids", "value": patient["PATIENT_ID"].n_unique()},
            {
                "measure": "unique_sample_patient_ids",
                "value": sample["PATIENT_ID"].n_unique(),
            },
            {
                "measure": "missing_sex",
                "value": patient["SEX"].null_count(),
            },
        ]
    )
    return outputs


def transorce_outputs(
    clinical: pl.DataFrame,
    operation_window: tuple[dt.date, dt.date],
    lag_days: tuple[int, int],
    horizons: list[float],
) -> dict[str, pl.DataFrame]:
    outputs: dict[str, pl.DataFrame] = {}
    clinical_cats = [
        canonical for canonical, kind in raw_transorce.CLINICAL_SCHEMA.values() if kind == "cat"
    ]
    outputs["transorce_categorical_values"] = _value_counts(
        clinical, clinical_cats, _TRANSORCE_COVERAGE, "clinical"
    )

    outputs["transorce_missing"] = _missing_counts(
        clinical,
        [canonical for canonical, _ in raw_transorce.CLINICAL_SCHEMA.values()],
        "clinical",
    )

    start, end = operation_window
    lo, hi = lag_days
    both_dates = pl.col("operation_date").is_not_null() & pl.col("randomisation_date").is_not_null()
    checks = clinical.select(
        pl.len().alias("n"),
        pl.col("operation_date").is_not_null().sum().alias("operation_present"),
        pl.col("operation_date")
        .is_between(start, end)
        .fill_null(False)
        .sum()
        .alias("operation_in_window"),
        both_dates.sum().alias("both_dates"),
        (pl.col("randomisation_date") - pl.col("operation_date"))
        .dt.total_days()
        .is_between(lo, hi)
        .fill_null(False)
        .sum()
        .alias("lag_in_range"),
    )
    n = checks["n"][0]
    outputs["transorce_date_checks"] = pl.DataFrame(
        [
            {
                "check": "operation_date_present",
                "n_applicable": n,
                "n_pass": checks["operation_present"][0],
                "n_fail": n - checks["operation_present"][0],
            },
            {
                "check": "operation_in_window",
                "n_applicable": checks["operation_present"][0],
                "n_pass": checks["operation_in_window"][0],
                "n_fail": checks["operation_present"][0] - checks["operation_in_window"][0],
            },
            {
                "check": "randomisation_lag_in_range",
                "n_applicable": checks["both_dates"][0],
                "n_pass": checks["lag_in_range"][0],
                "n_fail": checks["both_dates"][0] - checks["lag_in_range"][0],
            },
        ]
    )

    computed = [
        _transorce_computed_score(t, n, mm, cat, g, ne)
        for t, n, mm, cat, g, ne in clinical.select(
            "t_stage",
            "node_status",
            "max_diameter_mm",
            "tumour_size_category",
            "isup_grade",
            "necrosis",
        ).iter_rows()
    ]
    outputs["transorce_score_agreement"] = _score_agreement(
        computed, clinical["leibovich_score"].to_list()
    )

    details = clinical.filter(pl.col("other_details").is_not_null())
    outputs["transorce_other_details"] = (
        details.group_by("other_details").agg(pl.len().alias("n")).sort("other_details")
    )

    both = clinical.filter(
        pl.col("operation_date").is_not_null()
        & pl.col("dfs_date").is_not_null()
        & pl.col("dfs_status").is_not_null()
    )
    times = ((both["dfs_date"] - both["operation_date"]).dt.total_days() / 365.25).to_list()
    outputs["transorce_followup"] = _followup(
        times,
        (both["dfs_status"] == 0).to_list(),
        horizons,
        "transorce",
    )

    outputs["transorce_counts"] = pl.DataFrame(
        [
            {"measure": "clinical_rows", "value": clinical.height},
            {
                "measure": "missing_sex",
                "value": clinical["sex"].null_count(),
            },
            {
                "measure": "missing_operation_date",
                "value": clinical["operation_date"].null_count(),
            },
        ]
    )
    return outputs


def write_outputs(outputs: dict[str, pl.DataFrame], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, frame in outputs.items():
        assert_aggregate(frame, name)
        path = out_dir / f"{name}.csv"
        frame.write_csv(path)
        written.append(path)
    return written


def run(
    data_dir: Path,
    out_dir: Path,
    operation_window: tuple[dt.date, dt.date],
    lag_days: tuple[int, int],
    horizons: list[float],
) -> list[Path]:
    vasudev = data_dir / "vasudev"
    transorce = data_dir / "transorce"
    outputs = vasudev_outputs(
        raw_vasudev.read_patient(vasudev / "data_clinical_patient.txt"),
        raw_vasudev.read_sample(vasudev / "data_clinical_sample.txt"),
        horizons,
    )
    outputs.update(
        transorce_outputs(
            raw_transorce.read_clinical(transorce / "clinical.xlsx"),
            operation_window,
            lag_days,
            horizons,
        )
    )
    return write_outputs(outputs, out_dir)
