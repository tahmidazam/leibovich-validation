"""Manuscript tables: tidy Polars frames written as CSV.

`table_s1` fully specifies the two validated models: the Leibovich coefficients
and points with their source, the recovered baseline cumulative hazard and its
derivation, the life-table method and the equations of each version. Every value
comes from `leibovich` or the model parameters, so the table cannot drift from the
code it documents.

`table_s2` documents the variable mapping: how each cohort's source fields reach
the harmonised predictors, outcomes and design fields. Its codings, thresholds
and maps come from the raw schemas and the constants in `cohort`, so it cannot
drift from the loaders it documents either. `table_s2_other_details` is the
publication form of the approved TranSORCE "Other: Details" mapping.

`table_s3` summarises every sensitivity analysis the sensitivity stages have
written to results/sensitivity/, and the `table` command builds the 3-year
companion of Table 2 from the same summary table.
"""

from collections.abc import Iterable, Mapping
from pathlib import Path

import polars as pl
import yaml

from leibovich_validation import (
    cohort,
    leibovich,
    params,
    raw_transorce,
    raw_vasudev,
    report,
    sensitivity,
)
from leibovich_validation.config import REPO_ROOT

LEIBOVICH_2003 = "Leibovich et al. 2003 (https://doi.org/10.1002/cncr.11234)"
STIMPSON_2025 = "Stimpson et al. 2025 (https://doi.org/10.1101/2025.05.22.25328132)"
ONS_QX = "ONS period mortality rates, qx (https://www.ons.gov.uk)"
BONNEVILLE_2022 = "Bonneville et al. 2022 (https://doi.org/10.1177/09622802221102623)"
MICE = "R package mice 3.19.0"

COLUMNS = ["section", "item", "value", "source"]

# The Mayo development cohort column of Table 1. Values are transcribed once
# from Table 1 of Leibovich et al. 2003 (https://doi.org/10.1002/cncr.11234),
# with the event count (479 distant metastases) and median follow-up (5.4
# years, range 0-31) from the paper's results text. This is a cited reference
# column, not computed from our data; "not reported" marks rows the paper
# does not publish. Keys join on the `key` column of the report stage's
# table1 summary.
MAYO_COLUMN = "Mayo development cohort (Leibovich 2003)"
MAYO_2003_TABLE_1: dict[str, str] = {
    "n_patients": "1671",
    "age_mean_sd": "63.2 (11.3)",
    "age_median_range": "65 (24\u201389)",
    "sex_male": "1061 (63.5%)",
    "sex_female": "610 (36.5%)",
    "t_stage_pT1a": "384 (23.0%)",
    "t_stage_pT1b": "440 (26.3%)",
    "t_stage_pT2": "335 (20.0%)",
    "t_stage_pT3a": "154 (9.2%)",
    "t_stage_pT3b": "343 (20.5%)",
    "t_stage_pT3c": "10 (0.6%)",
    "t_stage_pT4": "5 (0.3%)",
    "node_positive": "66 (4.0%)",
    "node_negative": "1605 (96.0%)",
    "size_ge_10cm": "359 (21.5%)",
    "grade_1": "182 (10.9%)",
    "grade_2": "786 (47.0%)",
    "grade_3": "600 (35.9%)",
    "grade_4": "103 (6.2%)",
    "necrosis": "439 (26.3%)",
    "nephrectomy_radical": "1671 (100%)",
    "nephrectomy_partial": "0 (0%)",
    "risk_low": "not reported",
    "risk_intermediate": "not reported",
    "risk_high": "not reported",
    "followup_median_range": "5.4 (0\u201331)",
    "events_interest": "479",
    "events_competing": "not reported",
    "events_censored": "not reported",
    "missing_t_stage": "not reported",
    "missing_node": "not reported",
    "missing_size": "not reported",
    "missing_grade": "not reported",
    "missing_necrosis": "not reported",
    "missing_nephrectomy_type": "not reported",
    "missing_score": "not reported",
}

_COEFFICIENTS = "Leibovich linear predictor coefficients"
_POINTS = "Leibovich risk score points"
_PROFILES = "Preprint tumour profiles"
_BASELINE = "Baseline cumulative hazard"
_LIFE_TABLE = "Life table"
_EQUATIONS = "Model equations"


def _describe(tumour: leibovich.Tumour) -> str:
    """One-line summary of a tumour profile's predictor values."""
    return (
        f"{tumour.t_stage}, "
        f"{'node positive' if tumour.node_positive else 'node negative'}, "
        f"{'size 10 cm or greater' if tumour.size_ge_10cm else 'size under 10 cm'}, "
        f"grade {tumour.grade}, "
        f"{'necrosis' if tumour.necrosis else 'no necrosis'}"
    )


def table_s1(model: params.ModelParams) -> pl.DataFrame:
    """Table S1, the full model specification, one row per item.

    Columns: `section` groups the rows, `item` names the quantity, `value` carries
    the coefficient, hazard, setting or equation, and `source` gives the citation or
    derivation. With this table alone a reviewer can rebuild the model and reproduce
    Table 1 of the preprint.
    """
    rows: list[tuple[str, str, str, str]] = []
    coefficient_source = f"{LEIBOVICH_2003}, Table 3"
    point_source = f"{LEIBOVICH_2003}, Table 4"

    def coefficient(value: float) -> str:
        return "0 (reference)" if value == 0 else f"{value:.3f}"

    for stage, coef in leibovich.T_COEF.items():
        rows.append((_COEFFICIENTS, f"T stage {stage}", coefficient(coef), coefficient_source))
    rows.append(
        (
            _COEFFICIENTS,
            "Regional lymph node metastasis",
            f"{leibovich.NODE_POSITIVE_COEF:.3f}",
            coefficient_source,
        )
    )
    rows.append(
        (
            _COEFFICIENTS,
            "Tumour size 10 cm or greater",
            f"{leibovich.SIZE_GE_10CM_COEF:.3f}",
            coefficient_source,
        )
    )
    for grade, coef in leibovich.GRADE_COEF.items():
        rows.append(
            (_COEFFICIENTS, f"Nuclear grade {grade}", coefficient(coef), coefficient_source)
        )
    rows.append(
        (_COEFFICIENTS, "Tumour necrosis", f"{leibovich.NECROSIS_COEF:.3f}", coefficient_source)
    )

    for stage, points in leibovich.T_POINTS.items():
        rows.append((_POINTS, f"T stage {stage}", str(points), point_source))
    rows.append(
        (
            _POINTS,
            "Regional lymph node metastasis",
            str(leibovich.NODE_POSITIVE_POINTS),
            point_source,
        )
    )
    rows.append(
        (
            _POINTS,
            "Tumour size 10 cm or greater",
            str(leibovich.SIZE_GE_10CM_POINTS),
            point_source,
        )
    )
    for grade, points in leibovich.GRADE_POINTS.items():
        rows.append((_POINTS, f"Nuclear grade {grade}", str(points), point_source))
    rows.append((_POINTS, "Tumour necrosis", str(leibovich.NECROSIS_POINTS), point_source))

    for name, tumour in leibovich.PREPRINT_PROFILES.items():
        published = leibovich.PREPRINT_LEIBOVICH_RISK[name]
        risks = ", ".join(f"{years} y: {risk}%" for years, risk in published.items())
        rows.append(
            (
                _PROFILES,
                f"{name}-risk profile",
                f"{_describe(tumour)}; published Leibovich risks {risks}",
                f"{STIMPSON_2025}, Table 1",
            )
        )

    baseline_source = (
        f"Recovered from the unadjusted Leibovich risks in Table 1 of {STIMPSON_2025}: "
        "the geometric mean of -log(1 - R) / exp(linear predictor) across the three "
        "published tumour profiles"
    )
    for years, value in leibovich.BASELINE_CUMULATIVE_HAZARD.items():
        rows.append(
            (
                _BASELINE,
                f"Baseline cumulative hazard at {years} year{'' if years == 1 else 's'}",
                f"{value:.6g}",
                baseline_source,
            )
        )
    knots = ", ".join(f"{t:g}" for t in leibovich.BASELINE_TIMES)
    rows.append(
        (
            _BASELINE,
            "Hazard between the recovered times",
            f"piecewise constant; the cumulative hazard is linear between knots at {knots} years",
            baseline_source,
        )
    )

    rounding = "rounded to a whole percent" if model.round_other_cause_to_percent else "not rounded"
    rows += [
        (_LIFE_TABLE, "Mortality rates", model.life_table, ONS_QX),
        (_LIFE_TABLE, "Sex", "each patient's own sex", STIMPSON_2025),
        (
            _LIFE_TABLE,
            "Calendar years",
            f"one calendar year per follow-up year, from {model.start_year}",
            STIMPSON_2025,
        ),
        (_LIFE_TABLE, "Other-cause risk", rounding, STIMPSON_2025),
    ]

    rows += [
        (
            _EQUATIONS,
            "Leibovich (original)",
            "R_D(t) = 1 - exp(-H0(t) * exp(lp)); lp is the linear predictor, the sum "
            "of the Table 3 coefficients",
            LEIBOVICH_2003,
        ),
        (
            _EQUATIONS,
            "Leibovich Plus (published)",
            "R = 1 - (1 - R_D)(1 - R_P), split as R_met = R * R_D / (R_D + R_P) and "
            "R_death = R * R_P / (R_D + R_P); R_D is the Leibovich risk and R_P the "
            "life-table other-cause risk",
            f"{STIMPSON_2025}, equations 4, 5.1 and 5.2",
        ),
    ]
    return pl.DataFrame(rows, schema=COLUMNS, orient="row")


def table_s4(
    missingness: pl.DataFrame, analysis: params.AnalysisParams, labels: dict[str, str]
) -> pl.DataFrame:
    """Table S4: missing values per predictor per cohort and the imputation model.

    `missingness` is the concatenated impute-stage summary (columns cohort,
    predictor, n_total, n_missing, pct_missing). `labels` maps the cohort key
    to its display name. The model rows describe the chained-equations model,
    with the cause-specific event indicator and the
    Nelson-Aalen cumulative hazard of each competing cause as covariates.
    """
    rows: list[tuple[str, str, str, str]] = []
    for record in missingness.iter_rows(named=True):
        label = labels.get(str(record["cohort"]), str(record["cohort"]))
        rows.append(
            (
                "Missing values",
                f"{label}: {record['predictor']}",
                f"{record['n_missing']} of {record['n_total']} ({record['pct_missing']}%)",
                "harmonised patient table",
            )
        )
    rows += [
        (
            "Imputation model",
            "Method",
            f"multiple imputation by chained equations, {analysis.imputations} "
            f"completed datasets, seed {analysis.seed}",
            f"{MICE}; counts from params.yaml",
        ),
        (
            "Imputation model",
            "Outcome terms",
            "cause-specific event indicator and Nelson-Aalen cumulative hazard "
            "of each competing cause",
            BONNEVILLE_2022,
        ),
        (
            "Imputation model",
            "T stage",
            "polytomous logistic regression (polyreg)",
            MICE,
        ),
        (
            "Imputation model",
            "Nuclear grade",
            "proportional-odds logistic regression (polr)",
            MICE,
        ),
        (
            "Imputation model",
            "Size 10 cm or more, necrosis",
            "logistic regression (logreg)",
            MICE,
        ),
        (
            "Imputation model",
            "Nodal status",
            "not imputed: unrecorded nodal status (pNx) scores as pN0",
            "this study",
        ),
        (
            "Imputation model",
            "Other covariates",
            "age at surgery, sex, entry time",
            "this study",
        ),
        (
            "Imputation model",
            "Interval estimation",
            f"each bootstrap resample re-runs MICE with {analysis.bootstrap_imputations} "
            "completed datasets",
            "this study; count from params.yaml",
        ),
    ]
    return pl.DataFrame(rows, schema=COLUMNS, orient="row")


def _estimate_cell(estimate: float | None, lo: float | None, hi: float | None) -> str:
    """One table cell: the estimate with its interval in parentheses."""
    if estimate is None:
        return ""
    if lo is None or hi is None:
        return f"{estimate:.3f}"
    return f"{estimate:.3f} ({lo:.3f}\u2013{hi:.3f})"


def table_2(summary: pl.DataFrame) -> pl.DataFrame:
    """Manuscript Table 2: 5-year performance per cohort, model and bound.

    `summary` is the report stage's Table 2 source (columns label, series,
    stratum, bound, horizon, estimate, lo, hi, method). One row per cohort,
    coefficient bound, model and horizon; the observed column is the
    Aalen-Johansen cumulative incidence. Each cell carries the estimate and
    its 95% interval. `oe_other_cause` is Leibovich Plus's other-cause O/E,
    filled for the published model at the principal bound only;
    `other_cause_status` reads "descriptive" where the cohort's rows are.
    """
    observed = summary.filter(pl.col("label") == "aj_incidence")
    rows: list[dict[str, object]] = []
    metric_rows = summary.filter(pl.col("label").is_in(report.TABLE2_METRICS))
    groups = metric_rows.group_by(["stratum", "bound", "series", "horizon"], maintain_order=True)
    for (cohort_name, bound, model, horizon), group in groups:
        obs = observed.filter((pl.col("stratum") == cohort_name) & (pl.col("horizon") == horizon))
        row: dict[str, object] = {
            "cohort": cohort_name,
            "bound": bound,
            "model": model,
            "horizon_years": horizon,
            "observed_incidence": (
                _estimate_cell(obs["estimate"][0], obs["lo"][0], obs["hi"][0]) if obs.height else ""
            ),
        }
        for metric in report.TABLE2_METRICS:
            cell = group.filter(pl.col("label") == metric)
            row[metric] = (
                _estimate_cell(cell["estimate"][0], cell["lo"][0], cell["hi"][0])
                if cell.height
                else ""
            )
        cell = summary.filter(
            (pl.col("label") == report.OTHER_CAUSE_METRIC)
            & (pl.col("stratum") == cohort_name)
            & (pl.col("bound") == bound)
            & (pl.col("series") == model)
            & (pl.col("horizon") == horizon)
        )
        row["oe_other_cause"] = (
            _estimate_cell(cell["estimate"][0], cell["lo"][0], cell["hi"][0]) if cell.height else ""
        )
        row["other_cause_status"] = (
            "descriptive"
            if cell.height and str(cohort_name).lower() in report.DESCRIPTIVE_COHORTS
            else ""
        )
        rows.append(row)
    return pl.DataFrame(rows)


NOT_ESTIMABLE = "not estimable"
S3_METRICS = ("oe_ratio", "calibration_slope", "auc", "brier")


def _footnote_marker(variant: str) -> str:
    """The footnote letter of a variant that reports O/E only, in register order."""
    return f"({chr(ord('a') + list(sensitivity.NOT_ESTIMABLE_REASONS).index(variant))})"


def table_s6(summary: pl.DataFrame) -> pl.DataFrame:
    """Table S6: other-cause O/E of Leibovich Plus, overall and by age group.

    `summary` is the report stage's other-cause summary. One row per cohort,
    horizon and group: patients, deaths without the event by the horizon, the
    observed Aalen-Johansen incidence of those deaths, the mean predicted
    other-cause risk and their ratio, each with its 95% bootstrap percentile
    interval. `flag` reads "few events" for a group under
    `reporting.minimum_group_n` deaths; such a group stays in the table.
    """
    rows = [
        {
            "cohort": r["cohort"],
            "horizon_years": r["horizon_years"],
            "group": r["group"],
            "n": r["n"],
            "deaths": r["events"],
            "observed_incidence": _estimate_cell(
                r["observed"], r["observed_lower"], r["observed_upper"]
            ),
            "mean_predicted": _estimate_cell(r["mean_predicted"], None, None),
            "oe_other_cause": _estimate_cell(r["oe_ratio"], r["lower"], r["upper"]),
            "flag": "few events" if r["small"] else "",
            "status": r["status"] or "",
        }
        for r in summary.iter_rows(named=True)
    ]
    return pl.DataFrame(rows)


def table_s3(sensitivity_rows: pl.DataFrame, horizon: float) -> pl.DataFrame:
    """Table S3: the primary-horizon performance of every sensitivity analysis.

    `sensitivity_rows` is the concatenated content of results/sensitivity/
    (columns variant, cohort, horizon_years, bound, model, metric, estimate,
    lower, upper, method, n, events). One row per variant, cohort,
    coefficient bound and model at `horizon`, in the register's order; the
    observed column is the variant's Aalen-Johansen incidence and each measure
    cell carries its interval. The other-cause O/E column holds the
    Leibovich Plus pT3a ratio where a variant or the main analysis reports
    one, and is blank elsewhere. A measure a variant did not estimate reads
    "not estimable" with the footnote marker of its reason
    (`table_s3_footnotes`). A variant whose stage has not run has no rows.
    """
    frame = sensitivity_rows.filter(pl.col("horizon_years") == horizon)
    observed = frame.filter(pl.col("metric") == "aj_incidence")
    order = list(sensitivity.VARIANT_LABELS)
    rows: list[tuple[int, dict[str, object]]] = []
    oe = frame.filter(pl.col("metric") == "oe_ratio")
    groups = oe.group_by(["variant", "cohort", "bound", "model"], maintain_order=True)
    for (variant, cohort_name, bound, model), group in groups:
        variant = str(variant)
        obs = observed.filter((pl.col("variant") == variant) & (pl.col("cohort") == cohort_name))
        row: dict[str, object] = {
            "variant": sensitivity.VARIANT_LABELS.get(variant, variant),
            "cohort": cohort_name,
            "bound": bound,
            "model": model,
            "observed_incidence": (
                _estimate_cell(obs["estimate"][0], obs["lower"][0], obs["upper"][0])
                if obs.height
                else ""
            ),
        }
        for metric in S3_METRICS:
            cell = frame.filter(
                (pl.col("variant") == variant)
                & (pl.col("cohort") == cohort_name)
                & (pl.col("bound") == bound)
                & (pl.col("model") == model)
                & (pl.col("metric") == metric)
            )
            if cell.height:
                row[metric] = _estimate_cell(
                    cell["estimate"][0], cell["lower"][0], cell["upper"][0]
                )
            elif variant in sensitivity.NOT_ESTIMABLE_REASONS:
                row[metric] = f"{NOT_ESTIMABLE} {_footnote_marker(variant)}"
            else:
                row[metric] = ""
        other = frame.filter(
            (pl.col("variant") == variant)
            & (pl.col("cohort") == cohort_name)
            & (pl.col("bound") == bound)
            & (pl.col("model") == model)
            & (pl.col("metric") == "oe_other_cause")
        )
        row["oe_other_cause"] = (
            _estimate_cell(other["estimate"][0], other["lower"][0], other["upper"][0])
            if other.height
            else ""
        )
        row["n"] = group["n"][0]
        row["events"] = group["events"][0]
        rows.append((order.index(variant) if variant in order else len(order), row))
    rows.sort(key=lambda item: item[0])
    return pl.DataFrame([row for _, row in rows])


def table_s3_footnotes(table: pl.DataFrame) -> list[str]:
    """The footnotes of Table S3: the reason behind each not estimable marker it shows."""
    cells = [
        str(value) for column in S3_METRICS if column in table.columns for value in table[column]
    ]
    return [
        f"{_footnote_marker(variant)} {reason}"
        for variant, reason in sensitivity.NOT_ESTIMABLE_REASONS.items()
        if any(cell == f"{NOT_ESTIMABLE} {_footnote_marker(variant)}" for cell in cells)
    ]


S2_COLUMNS = ("section", "item", "cohort", "source", "raw_coding", "rule", "missing")

_TRANSORCE_HEADERS = {
    name: header for header, (name, _kind) in raw_transorce.CLINICAL_SCHEMA.items()
}

_S2_PREDICTORS = "Predictors"
_S2_DESIGN = "Eligibility and design"
_S2_TIMING = "Timing"
_S2_EVENT = "Event and competing event"
_S2_SCORE = "Score agreement"

_TRANSORCE = "TranSORCE"
_VASUDEV = "Vasudev cohort"


def _quoted(values: Iterable[str]) -> str:
    """A raw-coding cell: each allowed value in quotes, comma separated."""
    return ", ".join(f'"{value}"' for value in sorted(values, key=str.lower))


def _arrows(mapping: Mapping[str, str]) -> str:
    """A rule cell for a code-to-harmonised map, rendered raw -> harmonised."""
    return ", ".join(f"{raw} -> {mapped}" for raw, mapped in mapping.items())


def _english_list(items: Iterable[str], conjunction: str = "or") -> str:
    """'a, b or c' for a list of names."""
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + f" {conjunction} " + items[-1]


def table_s2(
    analysis: params.AnalysisParams,
    audit: params.AuditParams,
    other_details_map: Path,
) -> pl.DataFrame:
    """Table S2: every rule that maps a source field to a harmonised definition.

    One row per harmonised field per cohort, giving the source field's raw
    name and coding, the rule the loaders apply and the treatment of missing
    values. Codings, thresholds and code maps are read from the raw schemas
    and the constants in `cohort`, so the table cannot drift from the code it
    documents.
    """
    try:
        map_ref = other_details_map.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        map_ref = other_details_map.name
    rows: list[dict[str, str]] = []

    def add(
        section: str,
        item: str,
        cohort_label: str,
        source: str,
        raw_coding: str,
        rule: str,
        missing: str = "",
    ) -> None:
        rows.append(
            {
                "section": section,
                "item": item,
                "cohort": cohort_label,
                "source": source,
                "raw_coding": raw_coding,
                "rule": rule,
                "missing": missing,
            }
        )

    def t_source(*canonical: str) -> str:
        return "; ".join(_TRANSORCE_HEADERS[name] for name in canonical)

    incomplete = "leaves the predictor set incomplete, imputed in the main analysis"
    pt3_bound = "; the pT3b coefficient bound runs alongside in the principal analysis"

    add(
        _S2_PREDICTORS,
        "t_stage",
        _TRANSORCE,
        t_source("t_stage"),
        _quoted(raw_transorce.CATEGORIES["t_stage"]),
        _arrows(cohort.TRANSORCE_T_STAGE) + pt3_bound,
        f"missing or unrecorded stage {incomplete}",
    )
    unmapped = sorted(set(raw_vasudev.SAMPLE_CATEGORIES["STAGE_T"]) - set(cohort.VASUDEV_T_STAGE))
    if unmapped:
        vasudev_t_missing = (
            f"{_english_list(unmapped, 'and')} count{'s' if len(unmapped) == 1 else ''} "
            "as missing; the incomplete set is imputed"
        )
    else:
        vasudev_t_missing = "every recorded stage maps; a missing value is imputed"
    add(
        _S2_PREDICTORS,
        "t_stage",
        _VASUDEV,
        "STAGE_T",
        _quoted(raw_vasudev.SAMPLE_CATEGORIES["STAGE_T"]),
        _arrows(cohort.VASUDEV_T_STAGE) + pt3_bound,
        vasudev_t_missing,
    )
    add(
        _S2_PREDICTORS,
        "node_positive",
        _TRANSORCE,
        t_source("node_status"),
        _quoted(raw_transorce.CATEGORIES["node_status"]),
        f'"{cohort.TRANSORCE_NODE_POSITIVE}" counts as node-positive; the combined '
        "pNx/pN0 category counts as node-negative because pNx scores as pN0 "
        "",
        "missing counts as node-negative",
    )
    add(
        _S2_PREDICTORS,
        "node_positive",
        _VASUDEV,
        "STAGE_N",
        _quoted(raw_vasudev.SAMPLE_CATEGORIES["STAGE_N"]),
        f'"{cohort.VASUDEV_NODE_POSITIVE}" counts as node-positive; every other value '
        "counts as node-negative because pNx scores as pN0 "
        "",
        "missing counts as node-negative; the cohort records almost no nodal status",
    )
    add(
        _S2_PREDICTORS,
        "size_ge_10cm",
        _TRANSORCE,
        t_source("max_diameter_mm", "tumour_size_category"),
        f"millimetres; category {_quoted(raw_transorce.CATEGORIES['tumour_size_category'])}",
        f"a diameter of {cohort.SIZE_THRESHOLD_MM} mm or more counts as 10 cm or "
        "greater; the recorded category applies when the diameter is missing "
        "",
        f"diameter and category both missing {incomplete}",
    )
    add(
        _S2_PREDICTORS,
        "size_ge_10cm",
        _VASUDEV,
        "TUMOR_SIZE",
        "millimetres",
        f"a diameter of {cohort.SIZE_THRESHOLD_MM} mm or more counts as 10 cm or greater",
        f"missing size {incomplete}",
    )
    add(
        _S2_PREDICTORS,
        "grade",
        _TRANSORCE,
        t_source("isup_grade"),
        "integer grade",
        "ISUP grade stands in for Fuhrman grade",
        f"missing grade {incomplete}",
    )
    add(
        _S2_PREDICTORS,
        "grade",
        _VASUDEV,
        "FFPE_GRADE",
        "integer grade",
        "Fuhrman grade from the FFPE block, used directly",
        f"missing grade {incomplete}",
    )
    add(
        _S2_PREDICTORS,
        "necrosis",
        _TRANSORCE,
        t_source("necrosis"),
        _quoted(raw_transorce.CATEGORIES["necrosis"]),
        f'"{cohort.TRANSORCE_YES}" counts as necrosis',
        "missing counts as absent",
    )
    add(
        _S2_PREDICTORS,
        "necrosis",
        _VASUDEV,
        "NECROSIS",
        _quoted(raw_vasudev.SAMPLE_CATEGORIES["NECROSIS"]),
        f'"{cohort.VASUDEV_NECROSIS_PRESENT}" counts as necrosis',
        "missing counts as absent",
    )

    add(
        _S2_DESIGN,
        "patient_id",
        _TRANSORCE,
        t_source("trial_identifier"),
        "text",
        "kept as text",
        "required",
    )
    add(
        _S2_DESIGN,
        "patient_id",
        _VASUDEV,
        "PATIENT_ID",
        "text",
        "kept as text; joins the patient and sample tables",
        "missing or duplicated identifiers stop the build",
    )
    window = f"{audit.operation_window[0]} to {audit.operation_window[1]}"
    lag = f"{audit.randomisation_lag_days[0]} to {audit.randomisation_lag_days[1]} days"
    add(
        _S2_DESIGN,
        "eligibility",
        _TRANSORCE,
        t_source("operation_date", "randomisation_date"),
        "Excel dates",
        "every ccRCC workbook row is eligible by the trial record; an operation "
        f"outside {window} or an operation-to-randomisation lag outside {lag} is "
        "excluded and counted (audit date checks)",
        "a missing operation date is estimated; both dates missing stops the build",
    )
    add(
        _S2_DESIGN,
        "eligibility",
        _VASUDEV,
        "STAGE; STAGE_M",
        f"STAGE {_quoted(raw_vasudev.PATIENT_CATEGORIES['STAGE'])}; "
        f"STAGE_M {_quoted(raw_vasudev.SAMPLE_CATEGORIES['STAGE_M'])}",
        "overall stage IV, recorded MI and missing overall stage are excluded; "
        "missing STAGE_M at stage I to III stays eligible",
        "missing STAGE is excluded; missing STAGE_M stays eligible",
    )
    sex_missing = (
        f'missing-sex rule from params.yaml: "{analysis.missing_sex}" (the life-table '
        "risk needs sex)"
    )
    add(
        _S2_DESIGN,
        "sex",
        _TRANSORCE,
        t_source("sex"),
        _quoted(raw_transorce.CATEGORIES["sex"]),
        _arrows(cohort.TRANSORCE_SEX),
        sex_missing,
    )
    add(
        _S2_DESIGN,
        "sex",
        _VASUDEV,
        "SEX",
        _quoted(raw_vasudev.PATIENT_CATEGORIES["SEX"]),
        _arrows(cohort.VASUDEV_SEX),
        sex_missing,
    )
    add(
        _S2_DESIGN,
        "age_at_surgery",
        _TRANSORCE,
        t_source("age_at_randomisation"),
        "years",
        "age at randomisation minus the operation-to-randomisation lag",
        "a missing age stops the build",
    )
    add(
        _S2_DESIGN,
        "age_at_surgery",
        _VASUDEV,
        "AGE",
        "years",
        "recorded age, used directly",
        "a missing age stops the build",
    )
    add(
        _S2_DESIGN,
        "arm",
        _TRANSORCE,
        t_source("arm"),
        _quoted(raw_transorce.CATEGORIES["arm"]),
        "recorded as the randomised arm; no analysis uses it",
        "stays null",
    )
    add(
        _S2_DESIGN,
        "arm",
        _VASUDEV,
        "not recorded",
        "",
        "not a trial cohort; null",
        "always null",
    )
    add(
        _S2_DESIGN,
        "nephrectomy_type",
        _TRANSORCE,
        "not recorded",
        "",
        "the workbook records no operation type; null",
        "always null",
    )
    add(
        _S2_DESIGN,
        "nephrectomy_type",
        _VASUDEV,
        "NEPHRECTOMY_TYPE",
        _quoted(raw_vasudev.PATIENT_CATEGORIES["NEPHRECTOMY_TYPE"]),
        "lowercased; no analysis uses it",
        "stays null",
    )

    add(
        _S2_TIMING,
        "operation_year",
        _TRANSORCE,
        t_source("operation_date"),
        "Excel date",
        "the operation's calendar year",
        "a missing operation date is estimated as randomisation minus the "
        "cohort-median lag and flagged",
    )
    add(
        _S2_TIMING,
        "operation_year",
        _VASUDEV,
        "not recorded",
        "",
        "null; the cohort records no surgery dates, so the country-matched WHO life-table "
        f"sensitivity assigns every patient the mid-point year {analysis.vasudev_who_year} "
        "from params.yaml",
        "always null",
    )
    add(
        _S2_TIMING,
        "entry_time",
        _TRANSORCE,
        t_source("operation_date", "randomisation_date"),
        "Excel dates",
        "the operation-to-randomisation lag in years; the patient enters the risk "
        "set at randomisation (left truncation)",
        "estimated along with the operation date",
    )
    add(
        _S2_TIMING,
        "entry_time",
        _VASUDEV,
        "derived",
        "",
        "zero; surgery is the time origin with no delayed entry",
    )
    add(
        _S2_TIMING,
        "time",
        _TRANSORCE,
        t_source("dfs_date", "os_date", "operation_date"),
        "Excel dates",
        "event or censoring date minus the operation date, in years",
        "a missing DFS status or date stops the build",
    )
    add(
        _S2_TIMING,
        "time",
        _VASUDEV,
        "DFS_MONTHS; RCC_MONTHS",
        "months",
        "the DFS time in years, capped at the survival time; a missing DFS time "
        "falls back to the survival time",
        "both times missing stops the build",
    )

    distant_sites = _english_list(name.removesuffix("_metastasis") for name in cohort.DISTANT_FLAGS)
    add(
        _S2_EVENT,
        "event of interest",
        _TRANSORCE,
        t_source(
            "dfs_status",
            "dfs_date",
            *cohort.DISTANT_FLAGS,
            "other_metastasis",
            "other_details",
        ),
        '0/1 status; "No"/"Yes" flags; free text',
        f"the first recorded distant site: any of the {distant_sites} flags, or an "
        f"Other: Details string classed distant_metastasis by the approved mapping in "
        f"{map_ref}; the contralateral-kidney flag and second primaries are not the "
        "event",
        "a missing DFS status or date stops the build; an unmapped Other: Details "
        "string stops the build",
    )
    add(
        _S2_EVENT,
        "competing event",
        _TRANSORCE,
        t_source("os_status", "os_date"),
        "0/1 status; Excel date",
        "death without distant metastasis: a recorded death with no DFS event, or a "
        "DFS event that is the death itself",
        "a recorded death with no date stops the build",
    )
    add(
        _S2_EVENT,
        "censoring",
        _TRANSORCE,
        t_source("dfs_status", "dfs_date", "local_recurrence", "other_details"),
        '0/1 status; "No"/"Yes"; free text',
        "a DFS event with no distant site, a local-only recurrence or a second "
        "primary censors the patient at that event; otherwise the last "
        "disease-free date censors",
    )
    vasudev_event_raw = "; ".join(
        f"{name} {_quoted(raw_vasudev.PATIENT_CATEGORIES[name])}"
        for name in ("RELAPSE", "DFS_STATUS")
    )
    add(
        _S2_EVENT,
        "event of interest",
        _VASUDEV,
        "RELAPSE; DFS_STATUS; RCC_RELATED_DEATH; RCC_STATUS",
        vasudev_event_raw,
        "relapse at the DFS time, a wider endpoint than distant metastasis; a "
        "missing RELAPSE takes the status from DFS_STATUS; an RCC death without a "
        "recorded relapse counts as relapse at death; a relapsed but disease-free "
        "record counts as an event and is flagged",
        "RELAPSE and DFS_STATUS both missing leaves the status underivable; the "
        "patient is censored unless a death is recorded",
    )
    vasudev_death_raw = "; ".join(
        f"{name} {_quoted(raw_vasudev.PATIENT_CATEGORIES[name])}"
        for name in ("OS_STATUS", "RCC_STATUS", "RCC_RELATED_DEATH")
    )
    add(
        _S2_EVENT,
        "competing event",
        _VASUDEV,
        "OS_STATUS; RCC_STATUS; RCC_RELATED_DEATH; RCC_MONTHS",
        vasudev_death_raw,
        "death without relapse at the survival time; a death of unknown cause "
        "counts as other-cause",
        "a recorded death with no survival time stops the build",
    )

    score_rule = (
        "checked against the computed score; agreement below "
        f"{analysis.score_agreement_min:.0%} stops the cohort build"
    )
    add(
        _S2_SCORE,
        "leibovich_score_recorded",
        _TRANSORCE,
        t_source("leibovich_score"),
        "integer",
        score_rule,
        "missing scores are not compared",
    )
    add(
        _S2_SCORE,
        "leibovich_score_recorded",
        _VASUDEV,
        "LEIBOVICH_SCORE",
        "integer",
        score_rule,
        "missing scores are not compared",
    )
    return pl.DataFrame(rows, schema=list(S2_COLUMNS))


def table_s2_other_details(path: Path) -> pl.DataFrame:
    """The TranSORCE "Other: Details" mapping as a publication table.

    One row per distinct free-text string in the approved mapping file, with
    its class and the flagged and laterality marks. `load_other_details_mapping`
    validates the same file the loader reads, so the table cannot drift from
    the event derivation.
    """
    classes = cohort.load_other_details_mapping(path)
    entries = yaml.safe_load(path.read_text())["mapping"]
    rows = []
    for text, value in entries.items():
        flags = value if isinstance(value, dict) else {}
        rows.append(
            {
                "other_details": str(text),
                "class": classes[str(text)],
                "flagged": "yes" if flags.get("flagged") else "",
                "laterality": "yes" if flags.get("laterality") else "",
            }
        )
    return pl.DataFrame(rows)


def table_1(summary: pl.DataFrame) -> pl.DataFrame:
    """Manuscript Table 1: cohort characteristics side by side with the Mayo column.

    `summary` is the report stage's Table 1 source (columns key, section, label,
    stratum, value, n), computed from the harmonised patient tables. One output
    row per `key`: the Mayo development cohort values come from
    `MAYO_2003_TABLE_1`, the cited reference column, and each validation cohort
    fills a column of its own. A summary key the reference table does not know
    raises: the row specs and the reference column must move together.
    """
    strata = summary["stratum"].unique(maintain_order=True).to_list()
    rows: list[dict[str, object]] = []
    groups = summary.group_by(["key", "section", "label"], maintain_order=True)
    for (key, section, label), group in groups:
        try:
            mayo = MAYO_2003_TABLE_1[key]
        except KeyError:
            msg = f"table1 summary key {key!r} has no Mayo value in MAYO_2003_TABLE_1"
            raise ValueError(msg) from None
        row: dict[str, object] = {
            "section": section,
            "characteristic": label,
            MAYO_COLUMN: mayo,
        }
        for stratum in strata:
            cell = group.filter(pl.col("stratum") == stratum)
            row[stratum] = cell["value"][0] if cell.height else ""
        rows.append(row)
    return pl.DataFrame(rows)


def write_table(
    frame: pl.DataFrame, name: str, out_dir: Path, footnotes: Iterable[str] | None = None
) -> list[Path]:
    """Write `frame` as `<name>.csv` in `out_dir`.

    When `footnotes` is given, they go to `<name>_notes.csv`, one per row.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.csv"]
    frame.write_csv(paths[0])
    if footnotes is not None:
        paths.append(out_dir / f"{name}_notes.csv")
        pl.DataFrame({"note": list(footnotes)}, schema={"note": pl.String}).write_csv(paths[1])
    return paths
