"""The score stage: observed against predicted risk through the R bridge.

Observed cumulative incidence is Aalen-Johansen from survival::survfit
through `rbridge`. Calibration intercept and slope come from GEE on Score()
pseudo-observations, and AUC and Brier score from riskRegression::Score. All
of them take the patient table on the randomisation clock
(`cohort.reorigin_at_randomisation`), so no entry time is left: Score() does
not handle delayed entry.

`score_cohort` scores one predictions frame; it is the complete-case path,
kept for the sensitivity analysis. `score_cohort_imputed` is the main
analysis: every measure is computed inside each completed dataset and pooled
by the mean, and the percentile bootstrap re-runs the imputation inside each
resample, so interval width reflects imputation uncertainty. The paired
statistic returns every measure from one resample so the model contrasts
share the resamples.
"""

import math
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import polars as pl

from leibovich_validation import (
    bootstrap,
    impute,
    other_cause,
    predict,
    rbridge,
    subgroups,
)

# The contrast of the principal analysis: Leibovich Plus against the original
# Leibovich, for calibration in the large, calibration slope and Brier score.
CONTRAST_PAIRS = ((predict.LEIBOVICH_PLUS, predict.LEIBOVICH),)
CONTRAST_METRICS = ("oe", "slope", "brier")

# Risk series are named "model|bound" so one Score() call returns all four
# model versions at once.
NAMES = {f"{model}|{bound}": (model, bound) for bound in predict.BOUNDS for model in predict.MODELS}


@dataclass(frozen=True)
class ScoreOutputs:
    """The aggregate tables the score stage writes; never patient-level."""

    performance: pl.DataFrame
    contrasts: pl.DataFrame
    calibration: pl.DataFrame
    other_cause: pl.DataFrame
    subgroups: pl.DataFrame = field(default_factory=pl.DataFrame)


def risk_set_size(frame: pl.DataFrame, horizon: float) -> int:
    """Patients at risk at `horizon`: entered before it and still followed at it."""
    return frame.filter((pl.col("entry_time") < horizon) & (pl.col("time") >= horizon)).height


def headline_horizon(
    counts: dict[float, int], horizons: tuple[float, ...], min_at_risk: int
) -> float:
    """The cohort's headline horizon under the follow-up fallback.

    The primary horizon leads unless fewer than `min_at_risk` patients remain
    at risk there, when the secondary horizon is promoted.
    """
    primary = horizons[0]
    if len(horizons) > 1 and counts[primary] < min_at_risk:
        return horizons[1]
    return primary


def _fallback_rows(
    cohort: str, frame: pl.DataFrame, horizons: tuple[float, ...], min_at_risk: int
) -> list[dict[str, object]]:
    """The follow-up-fallback rows: one at-risk count per horizon plus the headline.

    The headline_horizon row records which horizon the fallback promotes, so
    the report stages can pick the headline results without re-deriving the
    rule.
    """
    counts = {horizon: risk_set_size(frame, horizon) for horizon in horizons}
    headline = headline_horizon(counts, horizons, min_at_risk)
    rows: list[dict[str, object]] = [
        {
            "cohort": cohort,
            "horizon_years": horizon,
            "bound": "",
            "model": "observed",
            "metric": "at_risk",
            "estimate": float(count),
            "lower": None,
            "upper": None,
            "method": "count",
            "n": count,
            "events": None,
        }
        for horizon, count in counts.items()
    ]
    rows.append(
        {
            "cohort": cohort,
            "horizon_years": headline,
            "bound": "",
            "model": "",
            "metric": "headline_horizon",
            "estimate": headline,
            "lower": None,
            "upper": None,
            "method": "min_at_risk",
            "n": frame.height,
            "events": None,
        }
    )
    return rows


def _score_measures(
    resample: pl.DataFrame,
    names: dict[str, tuple[str, str]],
    risk_columns: list[str],
    horizon: float,
    bandwidth: float,
) -> tuple[dict[str, float], dict[str, rbridge.Calibration]]:
    """Every measure for one patient frame, plus the calibration curves.

    The frame holds one row per patient with the outcome columns and the risk
    columns in `risk_columns`, one per entry of `names`. The returned dict
    holds one value per measure and per "model|bound" series, plus the paired
    model contrasts; the calibrations come back alongside because the imputed
    path needs the first completed dataset's curves.
    """
    out: dict[str, float] = {}
    outcomes = rbridge.Outcomes(resample["time"], resample["event"], resample["entry_time"])
    aj = rbridge.observed_incidence(outcomes, horizon)
    out["aj|observed"] = aj.estimate
    risks = {name: resample[column] for name, column in zip(names, risk_columns, strict=True)}
    for name in names:
        out[f"oe|{name}"] = rbridge.oe_ratio(aj, float(cast(Any, risks[name].mean()))).estimate
    calibrations = rbridge.calibration(outcomes, risks, horizon, bandwidth=bandwidth)
    for name, cal in calibrations.items():
        out[f"slope|{name}"] = cal.slope
        out[f"intercept|{name}"] = cal.intercept
        out[f"ici|{name}"] = cal.ici
    scores = rbridge.performance_scores(outcomes, risks, horizon)
    for record in scores.iter_rows(named=True):
        # Score() returns an undefined measure (an AUC with no controls) as null.
        estimate = record["estimate"]
        out[f"{record['metric']}|{record['model']}"] = math.nan if estimate is None else estimate
    for upper_model, lower_model in CONTRAST_PAIRS:
        contrast = f"{upper_model} - {lower_model}"
        for metric in CONTRAST_METRICS:
            out[f"{metric}_diff|{contrast}"] = (
                out[f"{metric}|{upper_model}|pt3a"] - out[f"{metric}|{lower_model}|pt3a"]
            )
    return out, calibrations


def _bin_frame(curve: pl.DataFrame, n_bins: int) -> pl.DataFrame:
    """Equal-count bins of one model's calibration curve, ordered by predicted risk."""
    curve = curve.sort("predicted")
    return (
        curve.with_columns((pl.int_range(curve.height) * n_bins // curve.height).alias("_bin"))
        .group_by("_bin")
        .agg(
            pl.col("predicted").mean(),
            pl.col("observed").mean().alias("estimate"),
            pl.len().alias("n"),
        )
        .sort("_bin")
    )


def _bin_measures(curves: dict[str, pl.DataFrame], n_bins: int) -> dict[str, float]:
    """The observed value of every calibration bin, keyed "bin|<model|bound>|<k>"."""
    return {
        f"bin|{name}|{record['_bin']}": record["estimate"]
        for name, curve in curves.items()
        if not curve.is_empty()
        for record in _bin_frame(curve, n_bins).iter_rows(named=True)
    }


def _measures_with_curves(
    frame: pl.DataFrame,
    names: dict[str, tuple[str, str]],
    risk_columns: list[str],
    horizon: float,
    bandwidth: float,
    n_bins: int,
    age_cut_points: tuple[int, ...],
) -> tuple[dict[str, float], dict[str, rbridge.Calibration]]:
    """Every measure for one patient frame, plus the calibration curves.

    Besides the model measures this holds the observed incidence, the
    calibration-bin values and the other-cause O/E, so that one resample feeds
    every interval.
    """
    out, cals = _score_measures(frame, names, risk_columns, horizon, bandwidth)
    out.update(_bin_measures({name: cal.curve for name, cal in cals.items()}, n_bins))
    out.update(other_cause.measures(frame, horizon, age_cut_points))
    return out, cals


def _point_estimates(
    resample: pl.DataFrame,
    names: dict[str, tuple[str, str]],
    risk_columns: list[str],
    horizon: float,
    bandwidth: float,
    n_bins: int,
    age_cut_points: tuple[int, ...],
) -> dict[str, float]:
    """Every point estimate for one patient resample, or the full data."""
    return _measures_with_curves(
        resample, names, risk_columns, horizon, bandwidth, n_bins, age_cut_points
    )[0]


def _pooled_measures(dicts: list[dict[str, float]]) -> dict[str, float]:
    """The mean of each measure over completed datasets, the MI point estimate."""
    grouped: dict[str, list[float]] = {}
    for measures in dicts:
        for key, value in measures.items():
            grouped.setdefault(key, []).append(value)
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def _pooled_risks(frames: dict[int, pl.DataFrame], horizon: float) -> dict[str, pl.Series]:
    """Each patient's predicted risk averaged over the completed datasets.

    The frames are row-aligned with the patient table, so the mean is
    positional; nothing joins on patient identifiers.
    """
    out = {}
    for name, (model, bound) in NAMES.items():
        column = predict.risk_column(model, horizon, bound)
        stacked = np.stack([frames[k][column].to_numpy() for k in sorted(frames)])
        out[name] = pl.Series(name, stacked.mean(axis=0))
    return out


def _pooled_curve(
    curve: pl.DataFrame, first_risk: pl.Series, pooled_risk: pl.Series
) -> pl.DataFrame:
    """One patient's pooled predicted risk against its pseudo-observed outcome.

    The pseudo-observations depend on the outcome data only, so they are
    identical in every completed dataset. `curve` holds them sorted by the
    first dataset's predicted risk; `first_risk` is that same risk in patient
    order, which realigns the pseudo-observations to the pooled risks.
    """
    order = np.argsort(first_risk.to_numpy())
    observed = np.empty(len(order), dtype=float)
    observed[order] = curve["observed"].to_numpy()
    return pl.DataFrame({"predicted": pooled_risk, "observed": pl.Series(observed)})


def _pooled_curves(
    frames: dict[int, pl.DataFrame],
    first_cals: dict[str, rbridge.Calibration],
    names: dict[str, tuple[str, str]],
    horizon: float,
) -> tuple[dict[str, pl.Series], dict[str, pl.DataFrame]]:
    """Pooled predicted risks and calibration curves over completed datasets.

    `first_cals` are the calibrations of the lowest-numbered dataset, whose
    pseudo-observations every dataset shares.
    """
    first = frames[min(frames)]
    pooled = _pooled_risks(frames, horizon)
    curves = {
        name: _pooled_curve(
            first_cals[name].curve,
            first[predict.risk_column(names[name][0], horizon, names[name][1])],
            pooled[name],
        )
        for name in names
    }
    return pooled, curves


def _calibration_rows(
    curves: dict[str, pl.DataFrame],
    risks: dict[str, pl.Series],
    cohort_label: str,
    horizon: float,
    n_bins: int,
    rug_points: int,
    cohort_n: int,
    events: int,
    intervals: dict[str, rbridge.BootstrapInterval],
) -> list[dict[str, object]]:
    """Binned observed-against-predicted rows plus a strip of predicted risks.

    Each model's patients are grouped into `n_bins` equal-count bins of
    predicted risk; the observed value is the mean pseudo-observation in the
    bin, with the bootstrap percentile interval in `intervals` (keyed
    "bin|<model|bound>|<k>") clipped to [0, 1]. `rug_points`
    evenly spaced predicted-risk quantiles carry the spread of predictions as
    rows labelled "rug". Everything here is an aggregate; no row is a
    patient.
    """
    rows: list[dict[str, object]] = []
    for name, curve in curves.items():
        model, bound = name.rsplit("|", 1)
        if curve.is_empty():
            continue
        for record in _bin_frame(curve, n_bins).iter_rows(named=True):
            interval = intervals.get(f"bin|{name}|{record['_bin']}")
            rows.append(
                {
                    "kind": "bin",
                    "series": model,
                    "bound": bound,
                    "stratum": cohort_label,
                    "horizon_years": horizon,
                    "predicted": record["predicted"],
                    "estimate": record["estimate"],
                    "lo": None if interval is None else max(0.0, interval.lower),
                    "hi": None if interval is None else min(1.0, interval.upper),
                    "n": record["n"],
                    "events": events,
                    "cohort_n": cohort_n,
                }
            )
        for k in range(rug_points):
            value = risks[name].quantile(k / (rug_points - 1), interpolation="nearest")
            rows.append(
                {
                    "kind": "rug",
                    "series": model,
                    "bound": bound,
                    "stratum": cohort_label,
                    "horizon_years": horizon,
                    "predicted": value,
                    "estimate": None,
                    "lo": None,
                    "hi": None,
                    "n": None,
                    "events": events,
                    "cohort_n": cohort_n,
                }
            )
    return rows


def _smooth_rows(
    curves: dict[str, pl.DataFrame],
    cohort_label: str,
    horizon: float,
    bandwidth: float,
    minimum_support: int,
    cohort_n: int,
    events: int,
) -> list[dict[str, object]]:
    """The smoothed calibration curve of each model as aggregate grid rows.

    Each row is a unique predicted risk with the smoothed pseudo-observed
    incidence there; `n` counts the patients within the bandwidth and the
    grid stops where fewer than `minimum_support` remain.
    """
    rows: list[dict[str, object]] = []
    for name, curve in curves.items():
        model, bound = name.rsplit("|", 1)
        smooth = rbridge.smooth_calibration(curve, bandwidth, minimum_support)
        rows += [
            {
                "kind": "smooth",
                "series": model,
                "bound": bound,
                "stratum": cohort_label,
                "horizon_years": horizon,
                "predicted": record["predicted"],
                "estimate": record["estimate"],
                "lo": None,
                "hi": None,
                "n": int(record["n"]),
                "events": events,
                "cohort_n": cohort_n,
            }
            for record in smooth.grid.iter_rows(named=True)
        ]
    return rows


def _performance_rows(
    cohort: str,
    horizon: float,
    point: dict[str, float],
    intervals: dict[str, rbridge.BootstrapInterval],
    mean_predicted: dict[str, float],
    n: int,
    events: int,
) -> list[dict[str, object]]:
    """One performance row per model, bound and metric, plus the AJ row."""
    rows: list[dict[str, object]] = []
    for name, (model, bound) in NAMES.items():
        metrics = [
            ("oe_ratio", "oe"),
            ("calibration_intercept", "intercept"),
            ("calibration_slope", "slope"),
            ("ici", "ici"),
            ("auc", "auc"),
            ("brier", "brier"),
            ("ipa", "ipa"),
        ]
        for metric, short in metrics:
            key = f"{short}|{name}"
            interval = intervals.get(key)
            rows.append(
                {
                    "cohort": cohort,
                    "horizon_years": horizon,
                    "bound": bound,
                    "model": model,
                    "metric": metric,
                    "estimate": point.get(key),
                    "lower": None if interval is None else interval.lower,
                    "upper": None if interval is None else interval.upper,
                    "method": "bootstrap",
                    "n": n,
                    "events": events,
                }
            )
        rows.append(
            {
                "cohort": cohort,
                "horizon_years": horizon,
                "bound": bound,
                "model": model,
                "metric": "mean_predicted",
                "estimate": mean_predicted[name],
                "lower": None,
                "upper": None,
                "method": "mean",
                "n": n,
                "events": events,
            }
        )
    return rows


def _contrast_rows(
    cohort: str,
    horizon: float,
    point: dict[str, float],
    intervals: dict[str, rbridge.BootstrapInterval],
) -> list[dict[str, object]]:
    """One contrast row per paired model difference and metric."""
    rows: list[dict[str, object]] = []
    for upper_model, lower_model in CONTRAST_PAIRS:
        contrast = f"{upper_model} - {lower_model}"
        for metric in CONTRAST_METRICS:
            key = f"{metric}_diff|{contrast}"
            interval = intervals.get(key)
            rows.append(
                {
                    "cohort": cohort,
                    "horizon_years": horizon,
                    "contrast": contrast,
                    "metric": metric,
                    "estimate": point.get(key),
                    "lower": None if interval is None else interval.lower,
                    "upper": None if interval is None else interval.upper,
                    "method": "bootstrap",
                }
            )
    return rows


def _observed_row(
    cohort: str,
    horizon: float,
    n: int,
    events: int,
    aj: rbridge.AalenJohansen,
    interval: rbridge.BootstrapInterval | None,
) -> dict[str, object]:
    """The observed-incidence performance row shared by both scoring paths.

    The interval is the bootstrap percentile interval of the incidence,
    clipped to [0, 1].
    """
    return {
        "cohort": cohort,
        "horizon_years": horizon,
        "bound": "",
        "model": "observed",
        "metric": "aj_incidence",
        "estimate": aj.estimate,
        "lower": None if interval is None else max(0.0, interval.lower),
        "upper": None if interval is None else min(1.0, interval.upper),
        "method": "bootstrap",
        "n": n,
        "events": events,
    }


def _score_at_horizon(
    predictions: pl.DataFrame,
    names: dict[str, tuple[str, str]],
    cohort: str,
    cohort_label: str,
    horizon: float,
    *,
    resamples: int,
    seed: int,
    n_bins: int,
    rug_points: int,
    bandwidth: float,
    minimum_support: int,
    age_cut_points: tuple[int, ...],
) -> tuple[list[dict[str, object]], ...]:
    """The performance, contrast, calibration and other-cause rows of one horizon."""
    risk_columns = [predict.risk_column(model, horizon, bound) for model, bound in names.values()]
    other_column = predict.other_risk_column(horizon, other_cause.BOUND)
    frame = predictions.drop_nulls(subset=[*risk_columns, other_column])
    n = frame.height
    events = int(((frame["event"] == "metastasis") & (frame["time"] <= horizon)).sum())
    risk_series = {name: frame[column] for name, column in zip(names, risk_columns, strict=True)}

    outcomes = rbridge.Outcomes(frame["time"], frame["event"], frame["entry_time"])
    aj = rbridge.observed_incidence(outcomes, horizon)

    point, cals = _measures_with_curves(
        frame, names, risk_columns, horizon, bandwidth, n_bins, age_cut_points
    )
    intervals = bootstrap.bootstrap_intervals(
        lambda resample, _seed: _point_estimates(
            resample, names, risk_columns, horizon, bandwidth, n_bins, age_cut_points
        ),
        frame,
        resamples=resamples,
        seed=seed,
    )

    perf_rows: list[dict[str, object]] = [
        _observed_row(cohort, horizon, n, events, aj, intervals.get("aj|observed"))
    ]
    perf_rows += _performance_rows(
        cohort,
        horizon,
        point,
        intervals,
        {name: float(cast(Any, risk_series[name].mean())) for name in names},
        n,
        events,
    )
    other_rows = other_cause.rows(
        cohort, horizon, frame, frame[other_column], point, intervals, age_cut_points
    )
    perf_rows.append(
        other_cause.performance_row(
            cohort, horizon, point, intervals, n, other_cause.deaths(frame, horizon)
        )
    )
    contrast_rows = _contrast_rows(cohort, horizon, point, intervals)
    calibration_rows = _calibration_rows(
        {name: cals[name].curve for name in names},
        risk_series,
        cohort_label,
        horizon,
        n_bins,
        rug_points,
        n,
        events,
        intervals,
    )
    calibration_rows += _smooth_rows(
        {name: cals[name].curve for name in names},
        cohort_label,
        horizon,
        bandwidth,
        minimum_support,
        n,
        events,
    )
    return perf_rows, contrast_rows, calibration_rows, other_rows


def _score_imputed_at_horizon(
    patients: pl.DataFrame,
    imp_frames: dict[int, pl.DataFrame],
    names: dict[str, tuple[str, str]],
    intervals: dict[str, rbridge.BootstrapInterval],
    cohort: str,
    cohort_label: str,
    horizon: float,
    *,
    n_bins: int,
    rug_points: int,
    bandwidth: float,
    minimum_support: int,
    age_cut_points: tuple[int, ...],
) -> tuple[list[dict[str, object]], ...]:
    """The pooled performance, contrast, calibration and other-cause rows of one horizon.

    `imp_frames` maps each imputation number to that completed dataset's
    predictions; `intervals` is the full re-imputing bootstrap output, keyed
    "{horizon}y|measure". Each measure is computed inside every completed
    dataset and pooled by the mean. The calibration curve pairs each
    patient's pooled predicted risk with the pseudo-observation from the
    first completed dataset's curve, which shares the same outcome data.
    """
    n = patients.height
    events = int(((patients["event"] == "metastasis") & (patients["time"] <= horizon)).sum())

    outcomes = rbridge.Outcomes(patients["time"], patients["event"], patients["entry_time"])
    aj = rbridge.observed_incidence(outcomes, horizon)

    risk_columns = [predict.risk_column(model, horizon, bound) for model, bound in names.values()]
    measures_per_imp: list[dict[str, float]] = []
    first_key = min(imp_frames)
    first_cals: dict[str, rbridge.Calibration] = {}
    for k, frame in imp_frames.items():
        measures, cals = _measures_with_curves(
            frame, names, risk_columns, horizon, bandwidth, n_bins, age_cut_points
        )
        measures_per_imp.append(measures)
        if k == first_key:
            first_cals = cals

    prefix = f"{horizon:g}y|"
    horizon_intervals = {
        key[len(prefix) :]: interval
        for key, interval in intervals.items()
        if key.startswith(prefix)
    }
    point = _pooled_measures(measures_per_imp)
    pooled_risks, pooled_curves = _pooled_curves(imp_frames, first_cals, names, horizon)
    point.update(_bin_measures(pooled_curves, n_bins))

    perf_rows: list[dict[str, object]] = [
        _observed_row(cohort, horizon, n, events, aj, horizon_intervals.get("aj|observed"))
    ]
    perf_rows += _performance_rows(
        cohort,
        horizon,
        point,
        horizon_intervals,
        {name: float(cast(Any, pooled_risks[name].mean())) for name in names},
        n,
        events,
    )
    other_rows = other_cause.rows(
        cohort,
        horizon,
        patients,
        other_cause.pooled_expected(imp_frames.values(), horizon),
        point,
        horizon_intervals,
        age_cut_points,
    )
    perf_rows.append(
        other_cause.performance_row(
            cohort, horizon, point, horizon_intervals, n, other_cause.deaths(patients, horizon)
        )
    )
    contrast_rows = _contrast_rows(cohort, horizon, point, horizon_intervals)
    calibration_rows = _calibration_rows(
        pooled_curves,
        pooled_risks,
        cohort_label,
        horizon,
        n_bins,
        rug_points,
        n,
        events,
        horizon_intervals,
    )
    calibration_rows += _smooth_rows(
        pooled_curves, cohort_label, horizon, bandwidth, minimum_support, n, events
    )
    return perf_rows, contrast_rows, calibration_rows, other_rows


def score_cohort(
    predictions: pl.DataFrame,
    cohort_label: str,
    horizons: tuple[float, ...],
    resamples: int,
    seed: int,
    n_bins: int,
    rug_points: int,
    min_at_risk: int,
    bandwidth: float,
    minimum_support: int,
    age_cut_points: tuple[int, ...],
) -> ScoreOutputs:
    """Score every model and bound at every horizon for one cohort.

    `predictions` is the interim per-patient frame from `predict.predict_frame`.
    Patients with a missing prediction for a horizon are dropped at that
    horizon before scoring: this is the complete-case path, kept as a
    sensitivity analysis next to the pooled imputation results of
    `score_cohort_imputed`. The performance rows open with the follow-up
    fallback counts and the headline horizon `min_at_risk` promotes. All
    output is aggregate.
    """
    cohort = str(predictions["cohort"][0])
    perf_rows: list[dict[str, object]] = _fallback_rows(cohort, predictions, horizons, min_at_risk)
    contrast_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    other_rows: list[dict[str, object]] = []

    for horizon in horizons:
        perf, contrast, calibration, other = _score_at_horizon(
            predictions,
            NAMES,
            cohort,
            cohort_label,
            horizon,
            resamples=resamples,
            seed=seed,
            n_bins=n_bins,
            rug_points=rug_points,
            bandwidth=bandwidth,
            minimum_support=minimum_support,
            age_cut_points=age_cut_points,
        )
        perf_rows += perf
        contrast_rows += contrast
        calibration_rows += calibration
        other_rows += other

    return ScoreOutputs(
        performance=pl.DataFrame(perf_rows),
        contrasts=pl.DataFrame(contrast_rows),
        calibration=pl.DataFrame(calibration_rows),
        other_cause=pl.DataFrame(other_rows),
    )


def score_cohort_imputed(
    patients: pl.DataFrame,
    predictions: pl.DataFrame,
    cohort_label: str,
    horizons: tuple[float, ...],
    resamples: int,
    seed: int,
    resample_imputations: int,
    rates_by_sex: dict[str, pl.DataFrame],
    start_year: int,
    round_to_percent: bool,
    n_bins: int,
    rug_points: int,
    min_at_risk: int,
    bandwidth: float,
    minimum_support: int,
    age_cut_points: tuple[int, ...],
    leibovich_groups: dict[str, tuple[int, int]] | None = None,
) -> ScoreOutputs:
    """The pooled multiple-imputation scoring for one cohort.

    `patients` is the harmonised table with its missing predictors;
    `predictions` is the predict stage's stacked output, one completed
    dataset per `imputation` value. Each measure is computed inside every
    completed dataset and pooled by the mean. The interval is percentile
    bootstrap over the patients with the imputation re-run inside each
    resample: MICE draws `resample_imputations` completed datasets there,
    predictions and scores follow in each, and the pooled measure enters the
    interval. One bootstrap run covers both horizons, because re-imputing is
    the expensive step. The performance rows open with the follow-up-fallback
    counts and the headline horizon `min_at_risk` promotes. With
    `leibovich_groups` set, the statistic also holds each age and Leibovich
    group's incidence, and `subgroups` carries their percentile intervals. All
    output is aggregate.
    """
    cohort = str(patients["cohort"][0])
    groupings = subgroups.definitions(age_cut_points, leibovich_groups) if leibovich_groups else {}
    imp_frames = {
        int(k): predictions.filter(pl.col("imputation") == k).drop("imputation")
        for k in predictions["imputation"].unique().sort().to_list()
    }
    risk_columns_by_horizon = {
        horizon: [predict.risk_column(model, horizon, bound) for model, bound in NAMES.values()]
        for horizon in horizons
    }

    def statistic(resample: pl.DataFrame, resample_seed: int) -> dict[str, float]:
        """Re-impute one resample, predict and score each completed dataset.

        Calibration bins are formed on the predicted risk pooled over the
        resample's completed datasets, like the reported bins.
        """
        imputed = impute.resample_completed(resample, resample_imputations, resample_seed)
        pooled: list[dict[str, float]] = []
        kept: dict[float, dict[int, pl.DataFrame]] = {horizon: {} for horizon in horizons}
        first_cals: dict[float, dict[str, rbridge.Calibration]] = {}
        for k, (_, completed) in enumerate(imputed.group_by("imputation", maintain_order=True)):
            resample_predictions, _ = predict.predict_frame(
                completed.drop("imputation"),
                rates_by_sex,
                horizons,
                start_year,
                round_to_percent,
                show_progress=False,
            )
            grouped = (
                subgroups.labelled(
                    resample_predictions,
                    completed.drop("imputation"),
                    age_cut_points,
                    leibovich_groups or {},
                )
                if groupings
                else resample_predictions
            )
            for horizon in horizons:
                measures, cals = _measures_with_curves(
                    resample_predictions,
                    NAMES,
                    risk_columns_by_horizon[horizon],
                    horizon,
                    bandwidth,
                    n_bins,
                    age_cut_points,
                )
                measures.update(subgroups.measures(grouped, horizon, groupings))
                pooled.append({f"{horizon:g}y|{key}": v for key, v in measures.items()})
                kept[horizon][k] = resample_predictions
                first_cals.setdefault(horizon, cals)
        out = _pooled_measures(pooled)
        for horizon in horizons:
            _, curves = _pooled_curves(kept[horizon], first_cals[horizon], NAMES, horizon)
            out.update(
                {f"{horizon:g}y|{key}": v for key, v in _bin_measures(curves, n_bins).items()}
            )
        return out

    intervals = bootstrap.bootstrap_intervals(statistic, patients, resamples=resamples, seed=seed)

    perf_rows: list[dict[str, object]] = _fallback_rows(cohort, patients, horizons, min_at_risk)
    contrast_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    other_rows: list[dict[str, object]] = []
    subgroup_rows: list[dict[str, object]] = []
    for horizon in horizons:
        prefix = f"{horizon:g}y|"
        subgroup_rows += subgroups.rows(
            horizon,
            {k[len(prefix) :]: v for k, v in intervals.items() if k.startswith(prefix)},
        )
        perf, contrast, calibration, other = _score_imputed_at_horizon(
            patients,
            imp_frames,
            NAMES,
            intervals,
            cohort,
            cohort_label,
            horizon,
            n_bins=n_bins,
            rug_points=rug_points,
            bandwidth=bandwidth,
            minimum_support=minimum_support,
            age_cut_points=age_cut_points,
        )
        perf_rows += perf
        contrast_rows += contrast
        calibration_rows += calibration
        other_rows += other

    return ScoreOutputs(
        performance=pl.DataFrame(perf_rows),
        contrasts=pl.DataFrame(contrast_rows),
        calibration=pl.DataFrame(calibration_rows),
        other_cause=pl.DataFrame(other_rows),
        subgroups=pl.DataFrame(
            subgroup_rows,
            schema=dict.fromkeys(subgroups.COLUMNS, pl.Utf8)
            | {"horizon_years": pl.Float64, "lower": pl.Float64, "upper": pl.Float64},
        ),
    )
