import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl

from leibovich_validation import (
    bootstrap,
    impute,
    lifetables,
    other_cause,
    params,
    predict,
    progress,
    rbridge,
    score,
)

# The output schema: the score stage's performance columns plus the variant
# tag, so every results/sensitivity file concatenates into Table S3 whatever
# stage wrote it. O/E-only variants carry only the metrics they estimate:
# aj_incidence for the observed rate and oe_ratio per model and bound.
PERFORMANCE_COLUMNS = [
    "cohort",
    "horizon_years",
    "bound",
    "model",
    "metric",
    "estimate",
    "lower",
    "upper",
    "method",
    "n",
    "events",
]
SENSITIVITY_COLUMNS = ["variant", *PERFORMANCE_COLUMNS]

# Display labels for the Table S3 variant column, keyed by the params.yaml
# switch name; dict order fixes the row order of Table S3.
VARIANT_LABELS: dict[str, str] = {
    "pt3b_for_unsplit_pt3": "pT3b coefficient for unsplit pT3",
    "complete_cases": "Complete cases",
    "country_life_table": "Country-matched WHO life table",
    "surgery_clock": "Surgery clock with entry at randomisation",
    "inconsistent_censoring": "Relapsed but disease-free records censored",
}

# The reasons a variant reports the observed/expected ratio only, keyed by
# switch name. Table S3 shows the other measures as not estimable and quotes
# the reason in a footnote.
NOT_ESTIMABLE_REASONS: dict[str, str] = {
    "surgery_clock": (
        "Score() in riskRegression does not support delayed entry, so the surgery-clock "
        "variant reports the Aalen-Johansen incidence and the observed/expected ratio only."
    ),
    "country_life_table": (
        "The country-matched life table changes only the other-cause risk, so this analysis "
        "reports the observed/expected ratios for metastasis and for death without the event."
    ),
}


def _applies(name: str, ctx: RunContext) -> bool:
    """Whether the data support a definition variant."""
    if name == "surgery_clock":
        return bool((ctx.surgery_clock_patients["entry_time"] > 0).any())
    if name == "inconsistent_censoring":
        return bool(ctx.patients["dfs_inconsistent"].any())
    return True


@dataclass(frozen=True)
class RunContext:
    """The shared inputs of one cohort's definition sensitivity run.

    `patients` is on the principal analysis's clock; `surgery_clock_patients` is the same
    cohort timed from surgery with delayed entry (`cohort.surgery_clock`).
    """

    cohort: str
    label: str
    patients: pl.DataFrame
    surgery_clock_patients: pl.DataFrame
    performance_path: Path
    performance_cc_path: Path
    analysis: params.AnalysisParams
    model: params.ModelParams
    rates_by_sex: dict[str, pl.DataFrame]
    horizons: tuple[float, ...]
    n_bins: int
    rug_points: int
    bandwidth: float
    minimum_support: int


def censor_inconsistent(patients: pl.DataFrame) -> pl.DataFrame:
    """Censor the dfs-inconsistent records at their recorded time.

    Records where RELAPSE and DFS_STATUS disagree count as events under the main rule
    and are censored here at the same time.
    """
    return patients.with_columns(
        pl.when(pl.col("dfs_inconsistent"))
        .then(pl.lit("censored"))
        .otherwise(pl.col("event"))
        .alias("event")
    )


def _tagged(name: str, frame: pl.DataFrame) -> pl.DataFrame:
    """Performance rows tagged with the variant, in the shared column order."""
    return frame.with_columns(pl.lit(name).alias("variant")).select(SENSITIVITY_COLUMNS)


def _score_variant_patients(name: str, patients: pl.DataFrame, ctx: RunContext) -> pl.DataFrame:
    """Re-run imputation, prediction and scoring on a variant analysis set."""
    imputed = impute.impute_frame(patients, ctx.analysis.imputations, ctx.analysis.seed)
    predictions, _ = predict.predict_frame(
        imputed,
        ctx.rates_by_sex,
        ctx.horizons,
        ctx.model.start_year,
        ctx.model.round_other_cause_to_percent,
    )
    outputs = score.score_cohort_imputed(
        patients,
        predictions,
        ctx.label,
        ctx.horizons,
        resamples=ctx.analysis.bootstrap_resamples,
        seed=ctx.analysis.seed,
        resample_imputations=ctx.analysis.bootstrap_imputations,
        rates_by_sex=ctx.rates_by_sex,
        start_year=ctx.model.start_year,
        round_to_percent=ctx.model.round_other_cause_to_percent,
        n_bins=ctx.n_bins,
        rug_points=ctx.rug_points,
        min_at_risk=ctx.analysis.min_at_risk,
        bandwidth=ctx.bandwidth,
        minimum_support=ctx.minimum_support,
        age_cut_points=ctx.analysis.age_cut_points,
    )
    return _tagged(name, outputs.performance)


def _copied_rows(name: str, path: Path, bound: str | None = None) -> pl.DataFrame | None:
    """Score-stage rows reused as a variant, optionally kept to one bound.

    With `bound`, model rows for the other bound are dropped but the observed
    Aalen-Johansen row stays, since observed incidence belongs to the analysis set.
    """
    if not path.is_file():
        return None
    frame = pl.read_csv(path)
    if bound is not None:
        frame = frame.filter((pl.col("bound") == bound) | (pl.col("metric") == "aj_incidence"))
    return _tagged(name, frame)


def empty_sensitivity_frame() -> pl.DataFrame:
    """The written schema when no variant produced rows."""
    return pl.DataFrame(
        schema={
            "variant": pl.String,
            "cohort": pl.String,
            "horizon_years": pl.Float64,
            "bound": pl.String,
            "model": pl.String,
            "metric": pl.String,
            "estimate": pl.Float64,
            "lower": pl.Float64,
            "upper": pl.Float64,
            "method": pl.String,
            "n": pl.Int64,
            "events": pl.Int64,
        }
    ).select(SENSITIVITY_COLUMNS)


def _surgery_clock_rows(ctx: RunContext) -> pl.DataFrame:
    """The surgery-clock variant: delayed-entry O/E against unconditional risks from surgery."""
    patients = ctx.surgery_clock_patients

    def predict_risks(frame: pl.DataFrame, show_progress: bool) -> pl.DataFrame:
        predictions, _ = predict.predict_frame(
            frame,
            ctx.rates_by_sex,
            ctx.horizons,
            ctx.model.start_year,
            ctx.model.round_other_cause_to_percent,
            show_progress=show_progress,
        )
        return predictions

    imputed = impute.impute_frame(patients, ctx.analysis.imputations, ctx.analysis.seed)
    return pl.DataFrame(
        oe_rows(
            "surgery_clock",
            patients,
            imputed,
            predict_risks,
            horizons=ctx.horizons,
            resamples=ctx.analysis.bootstrap_resamples,
            seed=ctx.analysis.seed,
            resample_imputations=ctx.analysis.bootstrap_imputations,
        )
    ).select(SENSITIVITY_COLUMNS)


def definition_sensitivity(ctx: RunContext, switches: params.SensitivityParams) -> pl.DataFrame:
    """The enabled definition variants' performance rows for one cohort.

    Each enabled switch the data support contributes rows in builder order. Complete
    cases reuse the score stage's `_cc` output and the pT3b bound reuses the principal
    pt3b rows. Censoring inconsistent records runs the full impute, predict and score
    path; the surgery clock runs the O/E-only path. A variant with no rows is skipped,
    so an all-off run still writes an empty table with the right schema.
    """
    builders: dict[str, Callable[[], pl.DataFrame | None]] = {
        "pt3b_for_unsplit_pt3": lambda: _copied_rows(
            "pt3b_for_unsplit_pt3", ctx.performance_path, bound="pt3b"
        ),
        "inconsistent_censoring": lambda: _score_variant_patients(
            "inconsistent_censoring", censor_inconsistent(ctx.patients), ctx
        ),
        "complete_cases": lambda: _copied_rows("complete_cases", ctx.performance_cc_path),
        "surgery_clock": lambda: _surgery_clock_rows(ctx),
    }
    frames: list[pl.DataFrame] = []
    enabled = [name for name in builders if getattr(switches, name) and _applies(name, ctx)]
    with progress.bar(enabled, desc=f"Variants {ctx.cohort}", unit="variant") as bar:
        for name in bar:
            bar.set_postfix(variant=name, refresh=True)
            rows = builders[name]()
            if rows is not None and not rows.is_empty():
                frames.append(rows)
    if not frames:
        return empty_sensitivity_frame()
    return pl.concat(frames).select(SENSITIVITY_COLUMNS)


def _predict_by_country(
    patients: pl.DataFrame,
    who_rates: dict[str, dict[str, pl.DataFrame]],
    horizons: tuple[float, ...],
    fallback_year: int,
    round_to_percent: bool,
    show_progress: bool,
) -> pl.DataFrame:
    """predict_frame once per country and surgery year, each against that country's WHO table.

    A stacked imputed table also splits on `imputation`. A null `operation_year` is
    replaced by `fallback_year`.
    """
    if patients["country"].null_count():
        msg = "country is missing for some patients; the stratum is required"
        raise ValueError(msg)
    patients = patients.with_columns(
        pl.col("operation_year").fill_null(fallback_year).alias("year")
    )
    group_cols = [c for c in ("imputation", "country", "year") if c in patients.columns]
    groups = patients.partition_by(group_cols, maintain_order=True, as_dict=True)
    parts: list[pl.DataFrame] = []
    with progress.bar(
        groups.items(),
        total=len(groups),
        desc="Predicting risks",
        unit="stratum",
        disable=None if show_progress else True,
        leave=False,
    ) as bar:
        for key, group in bar:
            country = str(key[-2])
            bar.set_postfix(country=country, year=key[-1], refresh=False)
            try:
                rates = who_rates[country]
            except KeyError:
                msg = f"no WHO life table loaded for country {country!r}"
                raise ValueError(msg) from None
            part, _ = predict.predict_frame(
                group.drop("year"),
                rates,
                horizons,
                int(key[-1]),
                round_to_percent,
                show_progress=False,
            )
            parts.append(part)
    if not parts:
        msg = "no patients to predict"
        raise ValueError(msg)
    return pl.concat(parts)


def _oe_measures(
    predictions: pl.DataFrame, horizon: float, aj: rbridge.AalenJohansen
) -> dict[str, float]:
    """The O/E of every model and bound at `horizon` on one predictions frame."""
    out: dict[str, float] = {}
    for model in predict.MODELS:
        for bound in predict.BOUNDS:
            risk = predictions[predict.risk_column(model, horizon, bound)].drop_nulls()
            expected = float(cast(Any, risk.mean())) if risk.len() else math.nan
            out[f"{model}|{bound}"] = rbridge.oe_ratio(aj, expected).estimate
    return out


def _pooled_oe(
    predictions: pl.DataFrame, horizon: float, aj: rbridge.AalenJohansen
) -> dict[str, float]:
    """O/E per completed dataset, pooled by the mean: the MI point estimate."""
    groups = (
        predictions.partition_by("imputation", maintain_order=True)
        if "imputation" in predictions.columns
        else [predictions]
    )
    per_group = [_oe_measures(group, horizon, aj) for group in groups]
    return {key: float(np.mean([measures[key] for measures in per_group])) for key in per_group[0]}


def _pooled_other_oe(predictions: pl.DataFrame, horizon: float, observed: float) -> float:
    """Other-cause O/E per completed dataset, pooled by the mean."""
    column = predict.other_risk_column(horizon, other_cause.BOUND)
    groups = (
        predictions.partition_by("imputation", maintain_order=True)
        if "imputation" in predictions.columns
        else [predictions]
    )
    ratios = [
        other_cause.oe_ratio(observed, float(cast(Any, g[column].drop_nulls().mean()) or 0.0))
        for g in groups
    ]
    return float(np.mean(ratios))


def oe_rows(
    variant: str,
    patients: pl.DataFrame,
    imputed: pl.DataFrame,
    predict_risks: Callable[[pl.DataFrame, bool], pl.DataFrame],
    *,
    horizons: tuple[float, ...],
    resamples: int,
    seed: int,
    resample_imputations: int,
    with_other_cause: bool = False,
) -> list[dict[str, object]]:
    """The O/E-only sensitivity rows of one variant on one cohort.

    `predict_risks(frame, show_progress)` gives the variant's per-patient risks for a
    stacked imputed table. Per horizon the output is the observed Aalen-Johansen row
    (left-truncated at `entry_time`) plus one oe_ratio row per model and bound, in the
    shared performance schema. With `with_other_cause`, each horizon also gets an
    oe_other_cause row for Leibovich Plus on the pT3a bound: observed incidence of death
    without the event over the variant's mean predicted other-cause risk (`other_cause`),
    pooled over imputations and bootstrapped in the same resamples.
    """
    cohort_name = str(patients["cohort"][0])
    predictions = predict_risks(imputed, True)
    outcomes = rbridge.Outcomes(patients["time"], patients["event"], patients["entry_time"])
    aj = {horizon: rbridge.observed_incidence(outcomes, horizon) for horizon in horizons}
    point = {
        f"{horizon:g}y|{name}": value
        for horizon in horizons
        for name, value in _pooled_oe(predictions, horizon, aj[horizon]).items()
    }
    if with_other_cause:
        for horizon in horizons:
            point[f"{horizon:g}y|oe_other"] = _pooled_other_oe(
                predictions, horizon, other_cause.observed_incidence(patients, horizon)
            )

    def statistic(resample: pl.DataFrame, resample_seed: int) -> dict[str, float]:
        """Re-impute one resample, predict under the variant and pool the O/E."""
        completed = impute.resample_completed(resample, resample_imputations, resample_seed)
        resample_predictions = predict_risks(completed, False)
        resample_outcomes = rbridge.Outcomes(
            resample["time"], resample["event"], resample["entry_time"]
        )
        out: dict[str, float] = {}
        for horizon in horizons:
            resample_aj = rbridge.observed_incidence(resample_outcomes, horizon)
            out[f"{horizon:g}y|aj|observed"] = resample_aj.estimate
            for name, value in _pooled_oe(resample_predictions, horizon, resample_aj).items():
                out[f"{horizon:g}y|{name}"] = value
            if with_other_cause:
                out[f"{horizon:g}y|oe_other"] = _pooled_other_oe(
                    resample_predictions,
                    horizon,
                    other_cause.observed_incidence(resample, horizon),
                )
        return out

    intervals = bootstrap.bootstrap_intervals(statistic, patients, resamples=resamples, seed=seed)

    rows: list[dict[str, object]] = []
    for horizon in horizons:
        events = int(((patients["event"] == "metastasis") & (patients["time"] <= horizon)).sum())
        observed = aj[horizon]
        observed_interval = intervals.get(f"{horizon:g}y|aj|observed")
        rows.append(
            {
                "variant": variant,
                "cohort": cohort_name,
                "horizon_years": horizon,
                "bound": "",
                "model": "observed",
                "metric": "aj_incidence",
                "estimate": observed.estimate,
                "lower": None if observed_interval is None else max(0.0, observed_interval.lower),
                "upper": None if observed_interval is None else min(1.0, observed_interval.upper),
                "method": "bootstrap",
                "n": patients.height,
                "events": events,
            }
        )
        if with_other_cause:
            key = f"{horizon:g}y|oe_other"
            interval = intervals.get(key)
            rows.append(
                {
                    "variant": variant,
                    "cohort": cohort_name,
                    "horizon_years": horizon,
                    "bound": other_cause.BOUND,
                    "model": other_cause.MODEL,
                    "metric": "oe_other_cause",
                    "estimate": point.get(key),
                    "lower": None if interval is None else interval.lower,
                    "upper": None if interval is None else interval.upper,
                    "method": "bootstrap",
                    "n": patients.height,
                    "events": other_cause.deaths(patients, horizon),
                }
            )
        for model in predict.MODELS:
            for bound in predict.BOUNDS:
                key = f"{horizon:g}y|{model}|{bound}"
                interval = intervals.get(key)
                rows.append(
                    {
                        "variant": variant,
                        "cohort": cohort_name,
                        "horizon_years": horizon,
                        "bound": bound,
                        "model": model,
                        "metric": "oe_ratio",
                        "estimate": point.get(key),
                        "lower": None if interval is None else interval.lower,
                        "upper": None if interval is None else interval.upper,
                        "method": "bootstrap",
                        "n": patients.height,
                        "events": events,
                    }
                )
    return rows


def life_table_variants(
    patients: pl.DataFrame,
    imputed: pl.DataFrame,
    *,
    switches: params.SensitivityParams,
    data_dir: Path,
    model: params.ModelParams,
    horizons: tuple[float, ...],
    fallback_surgery_year: int,
    resamples: int,
    seed: int,
    resample_imputations: int,
) -> pl.DataFrame:
    """The country-matched WHO life-table rows, if switched on.

    `patients` is the harmonised table (the resampling unit); `imputed` is the stacked
    completed table that point estimates are predicted on. Output is aggregate, in the
    score stage's performance schema plus the variant tag.
    """
    if not switches.country_life_table:
        return empty_sensitivity_frame()
    countries = sorted(str(c) for c in imputed["country"].drop_nulls().unique())
    who_rates = {
        country: lifetables.load_who_rates_by_sex(data_dir, country) for country in countries
    }

    def predict_risks(frame: pl.DataFrame, show_progress: bool) -> pl.DataFrame:
        return _predict_by_country(
            frame,
            who_rates,
            horizons,
            fallback_surgery_year,
            model.round_other_cause_to_percent,
            show_progress,
        )

    rows = oe_rows(
        "country_life_table",
        patients,
        imputed,
        predict_risks,
        horizons=horizons,
        resamples=resamples,
        seed=seed,
        resample_imputations=resample_imputations,
        with_other_cause=True,
    )
    return pl.DataFrame(rows).select(SENSITIVITY_COLUMNS)
