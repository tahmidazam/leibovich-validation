import dataclasses
from collections.abc import Collection, Iterable, Mapping
from pathlib import Path
from typing import Any

import lv_datasets
import polars as pl

from leibovich_validation import (
    atomic,
    leibovich,
    params,
    report,
    sensitivity,
)

LEIBOVICH_2003 = "Leibovich et al. 2003 (https://doi.org/10.1002/cncr.11234)"
STIMPSON_2025 = "Stimpson et al. 2025 (https://doi.org/10.1101/2025.05.22.25328132)"
ONS_QX = "ONS period mortality rates, qx (https://www.ons.gov.uk)"
BONNEVILLE_2022 = "Bonneville et al. 2022 (https://doi.org/10.1177/09622802221102623)"
MICE = "R package mice 3.19.0"

COLUMNS = ["section", "item", "value", "source"]

# The Mayo development cohort column of Table 1, transcribed from Leibovich et al. 2003
# (https://doi.org/10.1002/cncr.11234), with the event count (479 distant metastases)
# and median follow-up (5.4 years, range 0-31) from its results text. A cited reference,
# not computed here; "not reported" marks rows the paper omits. Keys join on `key` in
# the report stage's table1 summary.
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

    `section` groups rows, `item` names the quantity, `value` carries the coefficient,
    hazard, setting or equation, and `source` the citation or derivation. It suffices to
    rebuild the model and reproduce Table 1 of the preprint.
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

    for stage, points in lv_datasets.T_POINTS.items():
        rows.append((_POINTS, f"T stage {stage}", str(points), point_source))
    rows.append(
        (
            _POINTS,
            "Regional lymph node metastasis",
            str(lv_datasets.NODE_POSITIVE_POINTS),
            point_source,
        )
    )
    rows.append(
        (
            _POINTS,
            "Tumour size 10 cm or greater",
            str(lv_datasets.SIZE_GE_10CM_POINTS),
            point_source,
        )
    )
    for grade, points in lv_datasets.GRADE_POINTS.items():
        rows.append((_POINTS, f"Nuclear grade {grade}", str(points), point_source))
    rows.append((_POINTS, "Tumour necrosis", str(lv_datasets.NECROSIS_POINTS), point_source))

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

    `missingness` is the concatenated impute-stage summary (cohort, predictor, n_total,
    n_missing, pct_missing). `labels` maps cohort keys to display names. Model rows
    describe the chained-equations model, with the cause-specific event indicator and the
    Nelson-Aalen cumulative hazard of each cause as covariates.
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


def table_2(summary: pl.DataFrame, descriptive: Collection[str]) -> pl.DataFrame:
    """Manuscript Table 2: 5-year performance per cohort, model and bound.

    `summary` is the report stage's Table 2 source (label, series, stratum, bound,
    horizon, estimate, lo, hi, method). One row per cohort, bound, model and horizon; the
    observed column is the Aalen-Johansen incidence. Cells carry the estimate and its 95%
    interval. `oe_other_cause` is filled for the published model at the principal bound
    only; `other_cause_status` reads "descriptive" for the cohorts in `descriptive`.
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
            "descriptive" if cell.height and cohort_name in descriptive else ""
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

    `summary` is the report stage's other-cause summary. One row per cohort, horizon and
    group: patients, deaths without the event, observed incidence, mean predicted
    other-cause risk and their ratio, each with a 95% bootstrap percentile interval.
    `flag` reads "few events" below `reporting.minimum_group_n` deaths; the group stays
    in the table.
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

    `sensitivity_rows` is the concatenated content of results/sensitivity/ (variant,
    cohort, horizon_years, bound, model, metric, estimate, lower, upper, method, n,
    events). One row per variant, cohort, bound and model at `horizon`, in register
    order; the observed column is the variant's Aalen-Johansen incidence. The
    other-cause O/E column holds the Leibovich Plus pT3a ratio where reported, else is
    blank. A measure a variant did not estimate reads "not estimable" with the footnote
    marker for its reason (`table_s3_footnotes`). A variant whose stage has not run has
    no rows.
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


def table_s2(
    analysis: params.AnalysisParams, dataset_params: Mapping[str, dict[str, Any]]
) -> pl.DataFrame:
    """Table S2: every rule that maps a source field to a harmonised definition.

    One row per harmonised field per dataset, from the datasets' `mapping_rows` hooks:
    the source field's raw name and coding, the rule the adapter applies and the
    treatment of missing values. Rows follow the section order, then the order items
    first appear. Datasets without the hook add nothing.
    """
    rows = [
        {**row, "cohort": dataset.label}
        for name, ctx_params in dataset_params.items()
        if (dataset := lv_datasets.get(name)).mapping_rows
        for row in dataset.mapping_rows(ctx_params, dataclasses.asdict(analysis))
    ]
    item_rank: dict[str, int] = {}
    for row in rows:
        item_rank.setdefault((row["section"], row["item"]), len(item_rank))
    rows.sort(
        key=lambda row: (
            lv_datasets.S2_SECTIONS.index(row["section"]),
            item_rank[row["section"], row["item"]],
        )
    )
    return pl.DataFrame(rows, schema=list(S2_COLUMNS))


def mapping_tables(names: Iterable[str]) -> dict[str, pl.DataFrame]:
    """The extra mapping tables the datasets' `mapping_tables` hooks publish, by output name."""
    return {
        name: frame
        for dataset in map(lv_datasets.get, names)
        if dataset.mapping_tables
        for name, frame in dataset.mapping_tables().items()
    }


def table_1(summary: pl.DataFrame) -> pl.DataFrame:
    """Manuscript Table 1: cohort characteristics side by side with the Mayo column.

    `summary` is the report stage's Table 1 source (key, section, label, stratum, value,
    n). One output row per `key`; Mayo values come from `MAYO_2003_TABLE_1` and each
    validation cohort fills its own column. A summary key missing from the reference
    table raises, so row specs and reference column must change together.
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

    Any `footnotes` go to `<name>_notes.csv`, one per row.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.csv"]
    atomic.save(frame, paths[0])
    if footnotes is not None:
        paths.append(out_dir / f"{name}_notes.csv")
        notes = pl.DataFrame({"note": list(footnotes)}, schema={"note": pl.String})
        atomic.save(notes, paths[1])
    return paths
