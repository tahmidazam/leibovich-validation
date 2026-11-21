import datetime as dt
from typing import Any

import polars as pl
from lv_datasets import DAYS_PER_YEAR, Context, score_of
from lv_datasets.audit import followup, missing_counts, score_agreement, value_counts

from lv_transorce import raw
from lv_transorce.harmonise import predictors

# Predictor columns whose non-null values a mapping rule must explain; the covered values
# are the raw reader's declared categories.
COVERAGE: dict[str, frozenset[str]] = {
    column: raw.CATEGORIES[column]
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


def run(ctx: Context, horizons: list[float]) -> dict[str, pl.DataFrame]:
    return tables(raw.read_clinical(ctx.data_dir / "clinical.xlsx"), ctx.params, horizons)


def tables(
    clinical: pl.DataFrame, params: dict[str, Any], horizons: list[float]
) -> dict[str, pl.DataFrame]:
    start, end = (dt.date.fromisoformat(str(d)) for d in params["operation_window"])
    lo, hi = params["randomisation_lag_days"]
    outputs: dict[str, pl.DataFrame] = {}
    categorical = [name for name, kind in raw.CLINICAL_SCHEMA.values() if kind == "cat"]
    outputs["transorce_categorical_values"] = value_counts(
        clinical, categorical, COVERAGE, "clinical"
    )
    outputs["transorce_missing"] = missing_counts(
        clinical, [name for name, _ in raw.CLINICAL_SCHEMA.values()], "clinical"
    )

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
        score_of(predictors(*row))
        for row in clinical.select(
            "t_stage",
            "node_status",
            "max_diameter_mm",
            "tumour_size_category",
            "isup_grade",
            "necrosis",
        ).iter_rows()
    ]
    outputs["transorce_score_agreement"] = score_agreement(
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
    times = ((both["dfs_date"] - both["operation_date"]).dt.total_days() / DAYS_PER_YEAR).to_list()
    outputs["transorce_followup"] = followup(
        times, (both["dfs_status"] == 0).to_list(), horizons, "transorce"
    )

    outputs["transorce_counts"] = pl.DataFrame(
        [
            {"measure": "clinical_rows", "value": clinical.height},
            {"measure": "missing_sex", "value": clinical["sex"].null_count()},
            {
                "measure": "missing_operation_date",
                "value": clinical["operation_date"].null_count(),
            },
        ]
    )
    return outputs
