import math
from typing import cast

import polars as pl

from leibovich_validation import leibovich, leibovich_plus, progress

# Series names are fixed so a model keeps one name and one colour in every
# figure and table.
LEIBOVICH = "Leibovich"
LEIBOVICH_PLUS = "Leibovich Plus"
MODELS = (LEIBOVICH, LEIBOVICH_PLUS)

# The two pT3 coefficient bounds of the principal analysis.
BOUNDS = ("pt3a", "pt3b")

_MODEL_SLUGS = {
    LEIBOVICH: "leibovich",
    LEIBOVICH_PLUS: "leibovich_plus",
}


def risk_column(model: str, horizon: float, bound: str = "pt3a") -> str:
    """The prediction column name for one model, horizon and coefficient bound."""
    suffix = f"_{bound}" if bound != "pt3a" else ""
    return f"risk_{_MODEL_SLUGS[model]}_{horizon:g}y{suffix}"


def other_risk_column(horizon: float, bound: str = "pt3a") -> str:
    """The Leibovich Plus other-cause risk column for one horizon and coefficient bound."""
    suffix = f"_{bound}" if bound != "pt3a" else ""
    return f"other_risk_leibovich_plus_{horizon:g}y{suffix}"


def _conditional_leibovich(tumour: leibovich.Tumour, horizon: float, entry: float) -> float:
    """Leibovich metastasis risk by `horizon`, conditional on event-free at `entry`.

    The model ignores other-cause death, so this is 1 - S(horizon) / S(entry).
    """
    if entry == 0:
        return leibovich.risk(tumour, horizon)
    surviving = 1 - leibovich.risk(tumour, entry)
    return 1 - (1 - leibovich.risk(tumour, horizon)) / surviving


def patient_window_risks(
    rates: pl.DataFrame,
    tumour: leibovich.Tumour,
    age: int,
    horizon: float,
    entry: float,
    start_year: int,
    round_to_percent: bool,
) -> tuple[dict[str, float], float]:
    """The two models' metastasis risks and Leibovich Plus's other-cause risk.

    `horizon` is the time from surgery to the window's end, so the window is
    (entry, horizon]. A risk outside [0, 1] raises: Leibovich Plus as published can fall
    over time, and a negative value must not be clipped.
    """
    plus, other = leibovich_plus.published_risks(
        rates, tumour, age, horizon, start_year, round_to_percent, entry=entry
    )
    risks = {
        LEIBOVICH: _conditional_leibovich(tumour, horizon, entry),
        LEIBOVICH_PLUS: plus,
    }
    for model, risk in {**risks, f"{LEIBOVICH_PLUS} other-cause": other}.items():
        if not 0 <= risk <= 1:
            msg = f"{model} risk {risk} over ({entry:g}, {horizon:g}] years is outside [0, 1]"
            raise ValueError(msg)
    return risks, other


def patient_risks(
    rates: pl.DataFrame,
    tumour: leibovich.Tumour,
    age: int,
    horizon: float,
    entry: float,
    start_year: int,
    round_to_percent: bool,
) -> dict[str, float]:
    """The two model versions' metastasis risk for one patient at one horizon."""
    return patient_window_risks(rates, tumour, age, horizon, entry, start_year, round_to_percent)[0]


def predict_frame(
    patients: pl.DataFrame,
    rates_by_sex: dict[str, pl.DataFrame],
    horizons: tuple[float, ...],
    start_year: int,
    round_to_percent: bool,
    show_progress: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Predict every model and bound at every horizon for the patient table.

    `rates_by_sex` maps "male"/"female" to that sex's life-table qx frame. Returns
    (predictions, summary). Predictions hold the outcome columns scoring needs plus one
    risk column per model, horizon and bound; the summary has row counts and risk ranges
    only. A stacked imputed table keeps its `imputation` column. `show_progress` is off
    inside bootstrap resamples, where the resampling bar owns the display.
    """
    risk_rows: list[dict[str, float | None]] = []
    # The flag columns travel with the outcomes so the sensitivity analyses
    # (estimated operation dates, records flagged inconsistent) can
    # filter on them.
    outcome_columns = [
        "patient_id",
        "cohort",
        "sex",
        "arm",
        "age_at_surgery",
        "entry_time",
        "time",
        "event",
        "estimated_operation_date",
        "dfs_inconsistent",
    ]
    if "imputation" in patients.columns:
        outcome_columns.insert(0, "imputation")
    with progress.bar(
        patients.iter_rows(named=True),
        total=patients.height,
        desc="Predicting risks",
        unit="patient",
        disable=None if show_progress else True,
        leave=False,
    ) as bar:
        for row in bar:
            risks: dict[str, float | None] = {}
            missing = (
                row["t_stage"] is None
                or row["grade"] is None
                or row["node_positive"] is None
                or row["size_ge_10cm"] is None
                or row["necrosis"] is None
            )
            if missing:
                for model in MODELS:
                    for horizon in horizons:
                        for bound in BOUNDS:
                            risks[risk_column(model, horizon, bound)] = None
                for horizon in horizons:
                    for bound in BOUNDS:
                        risks[other_risk_column(horizon, bound)] = None
            else:
                rates = rates_by_sex[str(row["sex"])]
                age = math.floor(row["age_at_surgery"])
                offset = float(row["prediction_offset"])
                for bound in BOUNDS:
                    t_stage = (
                        "pT3b" if bound == "pt3b" and row["t_stage"] == "pT3a" else row["t_stage"]
                    )
                    tumour = leibovich.Tumour(
                        t_stage=cast("leibovich.TStage", t_stage),
                        node_positive=row["node_positive"],
                        size_ge_10cm=row["size_ge_10cm"],
                        grade=row["grade"],
                        necrosis=row["necrosis"],
                    )
                    for horizon in horizons:
                        window, other = patient_window_risks(
                            rates,
                            tumour,
                            age,
                            offset + horizon,
                            offset,
                            start_year,
                            round_to_percent,
                        )
                        for model, risk in window.items():
                            risks[risk_column(model, horizon, bound)] = risk
                        risks[other_risk_column(horizon, bound)] = other
            risk_rows.append(risks)

    risk_frame = pl.DataFrame(risk_rows)
    predictions = pl.concat([patients.select(outcome_columns), risk_frame], how="horizontal")

    summary_rows = []
    cohort = patients["cohort"].first()
    for model in MODELS:
        for bound in BOUNDS:
            for horizon in horizons:
                column = predictions[risk_column(model, horizon, bound)]
                predicted = column.drop_nulls()
                summary_rows.append(
                    {
                        "cohort": cohort,
                        "model": model,
                        "bound": bound,
                        "horizon_years": horizon,
                        "n_predicted": predicted.len(),
                        "n_missing_risk": column.null_count(),
                        "mean_risk": predicted.mean(),
                        "min_risk": predicted.min(),
                        "max_risk": predicted.max(),
                    }
                )
    return predictions, pl.DataFrame(summary_rows)
