from collections import defaultdict
from collections.abc import Callable, Iterable
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import lv_datasets
import numpy as np
import polars as pl
from lv_datasets import PREDICTOR_NAMES, SEXES

from leibovich_validation import (
    atomic,
    leibovich,
    leibovich_plus,
    params,
    predict,
    progress,
    rbridge,
)
from leibovich_validation.predict import LEIBOVICH, LEIBOVICH_PLUS

# The metrics of Table 2's main columns.
TABLE2_METRICS = ("oe_ratio", "calibration_slope", "ici", "auc", "brier")

# Other-cause O/E of Leibovich Plus, the extra Table 2 column.
OTHER_CAUSE_METRIC = "oe_other_cause"


def time_grid(step: float, stop: float = leibovich.MAX_YEARS) -> list[float]:
    """Evenly spaced times from 0 to `stop` years."""
    count = round(stop / step)
    return [round(k * step, 6) for k in range(count + 1)]


def model_behaviour(
    rates: pl.DataFrame,
    ages: Iterable[int],
    times: Iterable[float],
    start_year: int,
    round_to_percent: bool,
) -> pl.DataFrame:
    """Predicted metastasis risk over follow-up for each preprint tumour profile and age.

    One row per estimate: `label` is the tumour profile, `stratum` the age at surgery,
    `series` the model version, `time` the years since surgery and `estimate` the
    cumulative incidence. Sex enters only through the life table passed in.
    """
    rows = []
    ages = list(ages)
    times = list(times)
    profiles = leibovich.PREPRINT_PROFILES
    with progress.bar(
        total=len(profiles) * len(ages), desc="Scoring profiles", unit="profile", leave=False
    ) as bar:
        for label, tumour in profiles.items():
            for age in ages:
                bar.set_postfix(profile=label, age=age, refresh=False)
                published = [
                    leibovich_plus.published_risks(
                        rates, tumour, age, t, start_year, round_to_percent
                    )[0]
                    for t in times
                ]
                rows += [
                    (label, age, t, LEIBOVICH_PLUS, risk)
                    for t, risk in zip(times, published, strict=True)
                ]
                rows += [(label, age, t, LEIBOVICH, leibovich.risk(tumour, t)) for t in times]
                bar.update()
    return pl.DataFrame(
        rows, schema=["label", "stratum", "time", "series", "estimate"], orient="row"
    )


def table2_summary(performance: pl.DataFrame) -> pl.DataFrame:
    """Table 2 source rows: one row per performance measure, in figure roles.

    `performance` is score-stage output. The summary keeps the observed Aalen-Johansen
    incidence and the Table 2 metrics, with `label` the metric, `series` the model,
    `stratum` the cohort and `horizon` the years since surgery.
    """
    keep = ("aj_incidence", *TABLE2_METRICS, OTHER_CAUSE_METRIC)
    frame = performance.filter(pl.col("metric").is_in(keep))
    return frame.select(
        pl.col("metric").alias("label"),
        pl.col("model").alias("series"),
        pl.col("cohort").alias("stratum"),
        pl.col("bound"),
        pl.col("horizon_years").alias("horizon"),
        pl.col("estimate"),
        pl.col("lower").alias("lo"),
        pl.col("upper").alias("hi"),
        pl.col("n"),
        pl.col("events"),
        pl.col("method"),
    )


def other_cause_summary(
    other_cause: pl.DataFrame, minimum_group_n: int, descriptive_below_deaths: int
) -> pl.DataFrame:
    """The other-cause O/E rows with a `small` flag and a `status` column.

    A group is `small` with fewer than `minimum_group_n` deaths without the event by the
    horizon, and stays in the table. Rows of a cohort with fewer than
    `descriptive_below_deaths` overall deaths by their longest horizon have status
    "descriptive", the same at every horizon.
    """
    # other_cause imports this module for the age labels.
    from leibovich_validation import other_cause as other_cause_module

    overall = (
        pl.col("events").filter(pl.col("group") == other_cause_module.OVERALL).max().over("cohort")
    )
    return other_cause.with_columns(
        (pl.col("events") < minimum_group_n).alias("small"),
        pl.when(overall < descriptive_below_deaths)
        .then(pl.lit("descriptive"))
        .otherwise(pl.lit(""))
        .alias("status"),
    )


def other_cause_by_age_summary(
    other_cause: pl.DataFrame,
    cohorts: dict[str, str],
    horizon: float,
    minimum_group_n: int,
) -> pl.DataFrame:
    """Observed against predicted other-cause death by age, in `dot` column roles.

    Each age group gives an "observed" row (Aalen-Johansen incidence of death without
    the event, with its clipped bootstrap interval) and a "predicted" row (Leibovich
    Plus's mean other-cause risk), plus an "Overall" row per cohort and horizon.
    `events` counts deaths without the event, and `small` flags groups with fewer than
    `minimum_group_n`. `stratum` is the cohort label from `cohorts`; `cohort_n` is the
    overall row's size.
    """
    # other_cause imports this module for the age labels.
    from leibovich_validation import other_cause as other_cause_module

    frame = other_cause.filter(pl.col("horizon_years") == horizon)
    totals = {
        row["cohort"]: row["n"]
        for row in frame.filter(pl.col("group") == other_cause_module.OVERALL).iter_rows(named=True)
    }
    rows: list[dict[str, Any]] = []
    groups = frame.filter(pl.col("group") != other_cause_module.OVERALL)
    for order, row in enumerate(groups.iter_rows(named=True)):
        common: dict[str, Any] = {
            "grouping": "age",
            "label": row["group"],
            "group_order": order,
            "stratum": cohorts.get(row["cohort"], row["cohort"]),
            "bound": other_cause_module.BOUND,
            "horizon_years": horizon,
            "n": row["n"],
            "events": row["events"],
            "small": row["events"] < minimum_group_n,
            "cohort_n": totals.get(row["cohort"]),
        }
        rows.append(
            {
                **common,
                "kind": "observed",
                "series": "observed",
                "estimate": row["observed"],
                "lo": row["observed_lower"],
                "hi": row["observed_upper"],
                "predicted": None,
                "method": "bootstrap",
            }
        )
        rows.append(
            {
                **common,
                "kind": "predicted",
                "series": row["model"],
                "estimate": None,
                "lo": None,
                "hi": None,
                "predicted": row["mean_predicted"],
                "method": "mean",
            }
        )
    return pl.DataFrame(rows).with_columns(
        [
            pl.col(column).cast(pl.Float64)
            for column in ("estimate", "lo", "hi", "predicted", "n", "events")
        ]
    )


def write_other_cause_by_age(
    score_dir: Path,
    out_dir: Path,
    cohorts: dict[str, str],
    horizon: float,
    minimum_group_n: int,
) -> list[Path]:
    """Write the other-cause-by-age figure table from the score-stage output."""
    files = sorted(score_dir.glob("*_other_cause.csv"))
    if not files:
        return []
    other_cause = pl.concat([pl.read_csv(file) for file in files])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "other_cause_by_age.csv"
    atomic.save(other_cause_by_age_summary(other_cause, cohorts, horizon, minimum_group_n), path)
    return [path]


def calibration_summary(calibration: pl.DataFrame, horizon: float) -> pl.DataFrame:
    """The figure-ready calibration table at one horizon.

    Keeps one horizon of the score stage's binned table and rug rows, so the figure's
    `where` needs only the cohort and coefficient bound.
    """
    return calibration.filter(pl.col("horizon_years") == horizon)


def write_performance_summaries(
    score_dir: Path,
    out_dir: Path,
    horizon: float,
    minimum_group_n: int,
    descriptive_below_deaths: int,
) -> list[Path]:
    """Build the Table 2, calibration and other-cause summaries from score-stage output.

    Reads each cohort's aggregate files under `score_dir`. Missing cohort files are
    skipped, so this can run before every cohort has been scored.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    performance_files = sorted(score_dir.glob("*_performance.csv"))
    if performance_files:
        performance = pl.concat([pl.read_csv(path) for path in performance_files])
        path = out_dir / "table2.csv"
        atomic.save(table2_summary(performance), path)
        written.append(path)
    other_cause_files = sorted(score_dir.glob("*_other_cause.csv"))
    if other_cause_files:
        path = out_dir / "other_cause.csv"
        other_cause = pl.concat([pl.read_csv(file) for file in other_cause_files])
        atomic.save(
            other_cause_summary(other_cause, minimum_group_n, descriptive_below_deaths), path
        )
        written.append(path)
    calibration_files = sorted(score_dir.glob("*_calibration.csv"))
    if calibration_files:
        calibration = pl.concat(
            [pl.read_csv(path) for path in calibration_files], how="diagonal_relaxed"
        )
        path = out_dir / f"calibration_{horizon:g}y.csv"
        atomic.save(calibration_summary(calibration, horizon), path)
        written.append(path)
    return written


def flow_summary(
    flows: dict[str, pl.DataFrame], steps: dict[str, dict[str, tuple[str, str]]]
) -> pl.DataFrame:
    """The participant-flow summary: one row per counted step per cohort.

    `flows` maps a cohort display label to its flow table (columns `step`, `n`) and
    `steps` to its declared (kind, label) per step. `kind` assigns each step to the
    funnel, exclusions box or field-counts box, and `label` is the box text.
    """
    rows = []
    for stratum, frame in flows.items():
        for step, n in frame.iter_rows():
            if step not in steps[stratum]:
                msg = f"{stratum}: flow step {step!r} is not declared by the dataset"
                raise ValueError(msg)
            kind, label = steps[stratum][step]
            rows.append(
                {"stratum": stratum, "step": str(step), "kind": kind, "label": label, "n": int(n)}
            )
    return pl.DataFrame(rows)


def _count_pct(count: int, total: int) -> str:
    """One Table 1 cell: the count with its percentage of the analysis set."""
    return f"{count} ({count / total:.1%})" if total else "0"


def _median_range(values: pl.Series, digits: int = 0) -> str:
    """The median with the observed range, one Table 1 cell."""
    non_null = values.drop_nulls()
    if non_null.is_empty():
        return ""
    return (
        f"{non_null.median():.{digits}f} "
        f"({non_null.min():.{digits}f}\u2013{non_null.max():.{digits}f})"
    )


def _mean_sd(values: pl.Series) -> str:
    """The mean with the standard deviation, one Table 1 cell."""
    non_null = values.drop_nulls()
    if non_null.is_empty():
        return ""
    return f"{non_null.mean():.1f} ({non_null.std():.1f})"


def table1_summary(
    patients: pl.DataFrame,
    stratum: str,
    leibovich_groups: dict[str, tuple[int, int]],
) -> pl.DataFrame:
    """The Table 1 source rows for one cohort, in figure column roles.

    `patients` is the harmonised interim table for the analysis set; every output value
    is an aggregate. `leibovich_groups` maps group names to score bands from
    `params.yaml`. Columns: `key` (stable row id the manuscript table joins on),
    `section`, `label`, `stratum`, `value` (formatted cell) and `n` (count behind it).
    """
    total = patients.height
    rows: list[dict[str, Any]] = []

    def row(key: str, section: str, label: str, value: str, n: int) -> None:
        rows.append(
            {
                "key": key,
                "section": section,
                "label": label,
                "stratum": stratum,
                "value": value,
                "n": n,
            }
        )

    def count(key: str, section: str, label: str, mask: pl.Expr) -> None:
        n = patients.filter(mask).height
        row(key, section, label, _count_pct(n, total), n)

    row("n_patients", "", "Patients, n", str(total), total)
    age = patients["age_at_surgery"]
    row("age_mean_sd", "Age at surgery, years", "mean (SD)", _mean_sd(age), age.len())
    row(
        "age_median_range",
        "Age at surgery, years",
        "median (range)",
        _median_range(age),
        age.len(),
    )
    for sex in SEXES:
        count(f"sex_{sex}", "Sex", sex, pl.col("sex") == sex)
    for stage in lv_datasets.T_POINTS:
        count(f"t_stage_{stage}", "T stage", stage, pl.col("t_stage") == stage)
    count("node_positive", "Nodal status", "pN1 or pN2", pl.col("node_positive"))
    count("node_negative", "Nodal status", "pNx or pN0", pl.col("node_positive").not_())
    count("size_ge_10cm", "Tumour size", "10 cm or greater", pl.col("size_ge_10cm"))
    for grade in lv_datasets.GRADE_POINTS:
        count(f"grade_{grade}", "Nuclear grade", str(grade), pl.col("grade") == grade)
    count("necrosis", "Tumour necrosis", "present", pl.col("necrosis"))
    # Nephrectomy type is not recorded by every dataset; an all-null column
    # shows as an empty cell rather than a zero count, with the missingness
    # reported in the missing-values section.
    recorded_types = patients["nephrectomy_type"].drop_nulls().unique().sort().to_list()
    nephrectomy_types = ["radical", "partial"] + [
        t for t in recorded_types if t not in ("radical", "partial")
    ]
    for nephrectomy in nephrectomy_types:
        n = patients.filter(pl.col("nephrectomy_type") == nephrectomy).height
        value = _count_pct(n, total) if recorded_types else ""
        row(f"nephrectomy_{nephrectomy}", "Nephrectomy", nephrectomy, value, n)
    for name, (lo, hi) in leibovich_groups.items():
        count(
            f"risk_{name}",
            "Leibovich risk group",
            f"{name} (score {lo}\u2013{hi})",
            pl.col("leibovich_score_computed").is_between(lo, hi),
        )
    time = patients["time"]
    row(
        "followup_median_range",
        "Follow-up, years",
        "median (range)",
        _median_range(time, digits=1),
        time.len(),
    )
    n_interest = patients.filter(pl.col("event") == "metastasis").height
    row(
        "events_interest",
        "Events",
        "event of interest, n",
        str(n_interest),
        n_interest,
    )
    n_competing = patients.filter(pl.col("event") == "competing").height
    row(
        "events_competing",
        "Events",
        "death without the event, n",
        str(n_competing),
        n_competing,
    )
    n_censored = patients.filter(pl.col("event") == "censored").height
    row("events_censored", "Events", "censored, n", str(n_censored), n_censored)
    predictor_labels = {
        "t_stage": "T stage",
        "node": "Nodal status",
        "size": "Tumour size",
        "grade": "Nuclear grade",
        "necrosis": "Tumour necrosis",
    }
    for predictor in PREDICTOR_NAMES:
        count(
            f"missing_{predictor}",
            "Missing values",
            predictor_labels[predictor],
            pl.col("missing_predictors").list.contains(predictor),
        )
    count(
        "missing_nephrectomy_type",
        "Missing values",
        "Nephrectomy type",
        pl.col("nephrectomy_type").is_null(),
    )
    count(
        "missing_score",
        "Missing values",
        "Any predictor",
        pl.col("leibovich_score_computed").is_null(),
    )
    return pl.DataFrame(rows)


def write_cohort_summaries(
    interim_dir: Path,
    cohort_dir: Path,
    out_dir: Path,
    cohorts: dict[str, str],
    leibovich_groups: dict[str, tuple[int, int]],
) -> list[Path]:
    """Build the flow and Table 1 summary tables from interim patient tables.

    `cohorts` maps each cohort name to its display label; flow labels come from the
    registry. Reads patient tables under `interim_dir` and flow counts under `cohort_dir`;
    cohorts without files are skipped.
    Everything written is aggregate.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    flows = {
        label: pl.read_csv(cohort_dir / f"{name}_flow.csv")
        for name, label in cohorts.items()
        if (cohort_dir / f"{name}_flow.csv").is_file()
    }
    if flows:
        path = out_dir / "flow.csv"
        steps = {
            label: {
                step: (kind, text.format(**params.load_dataset(name)))
                for step, (kind, text) in lv_datasets.get(name).flow_steps.items()
            }
            for name, label in cohorts.items()
            if label in flows
        }
        atomic.save(flow_summary(flows, steps), path)
        written.append(path)
    patients = {
        label: pl.read_parquet(interim_dir / f"{name}_patients.parquet")
        for name, label in cohorts.items()
        if (interim_dir / f"{name}_patients.parquet").is_file()
    }
    if patients:
        summary = pl.concat(
            [table1_summary(frame, label, leibovich_groups) for label, frame in patients.items()]
        )
        path = out_dir / "table1.csv"
        atomic.save(summary, path)
        written.append(path)
    return written


def age_group_labels(cut_points: Iterable[int]) -> list[str]:
    """Band labels for the age subgroups: "<60", "60-69", "70+" for cuts 60 and 70.

    Bands are left-closed, so a patient aged exactly a cut point joins the upper band.
    Middle labels show the whole years covered, hence the upper bound b - 1.
    """
    points = [int(cut) for cut in sorted(cut_points)]
    if not points:
        msg = "age_cut_points is empty"
        raise ValueError(msg)
    labels: list[str] = [f"<{points[0]}"]
    labels += [f"{a}\u2013{b - 1}" for a, b in pairwise(points)]
    labels.append(f"{points[-1]}+")
    return labels


def completed_leibovich_score() -> pl.Expr:
    """The Leibovich score of one completed predictor set, as a column expression.

    Recomputed in each completed dataset because `leibovich_score_computed` stays null
    wherever the observed predictor set was incomplete.
    """
    return lv_datasets.leibovich_score(*(pl.col(name) for name in lv_datasets.PREDICTOR_COLUMNS))


def group_columns(
    age_cut_points: Iterable[int], leibovich_groups: dict[str, tuple[int, int]]
) -> dict[str, pl.Expr]:
    """One group-label column per subgrouping, keyed by the grouping name."""
    score = completed_leibovich_score()
    leibovich_label = pl.lit(None).cast(pl.String)
    for name, (lo, hi) in leibovich_groups.items():
        leibovich_label = (
            pl.when(score.is_between(lo, hi)).then(pl.lit(name)).otherwise(leibovich_label)
        )
    return {
        "age": pl.col("age_at_surgery")
        .bin_intervals(
            intervals=[float(cut) for cut in age_cut_points],
            labels=age_group_labels(age_cut_points),
        )
        .cast(pl.String)
        .alias("_group_age"),
        "leibovich": leibovich_label.alias("_group_leibovich"),
    }


def subgroup_definitions(
    age_cut_points: Iterable[int], leibovich_groups: dict[str, tuple[int, int]]
) -> dict[str, tuple[str, list[str]]]:
    """Each subgrouping's group-label column and its ordered labels."""
    return {
        "age": ("_group_age", age_group_labels(age_cut_points)),
        "leibovich": ("_group_leibovich", list(leibovich_groups)),
    }


def subgroup_summary(
    imputed: pl.DataFrame,
    predictions: pl.DataFrame,
    stratum: str,
    horizon: float,
    *,
    minimum_group_n: int,
    age_cut_points: Iterable[int],
    leibovich_groups: dict[str, tuple[int, int]],
    intervals: pl.DataFrame,
    incidence: Callable[[rbridge.Outcomes, float], rbridge.AalenJohansen] = (
        rbridge.observed_incidence
    ),
) -> pl.DataFrame:
    """Observed and predicted risk per subgroup at `horizon`, in figure column roles.

    `imputed` is the stacked completed patient table and `predictions` the predict
    stage's stacked output for the same datasets. Within each completed dataset, groups
    are formed on that dataset's predictors and the Aalen-Johansen incidence and each
    model's mean predicted risk are computed, then pooled by the mean. `intervals` is
    the score stage's subgroup table (grouping, label, horizon_years, lower, upper); a
    group without a row there has null bounds. `n` and `events` are means over datasets.
    `small` flags groups whose pooled event count is below `minimum_group_n` (events,
    not patients). A group empty in every dataset keeps a labelled row with null
    estimates. Only the pT3a bound is summarised. `incidence` is injectable so pooling
    can be tested without R.
    """
    risk_columns = {model: predict.risk_column(model, horizon) for model in predict.MODELS}
    frame = imputed.join(
        predictions.select("imputation", "patient_id", *risk_columns.values()),
        on=["imputation", "patient_id"],
        how="left",
        validate="1:1",
    ).with_columns(*group_columns(age_cut_points, leibovich_groups).values())
    cohort_n = predictions["patient_id"].n_unique()
    groupings = subgroup_definitions(age_cut_points, leibovich_groups)
    bounds = {
        (row["grouping"], row["label"]): (row["lower"], row["upper"])
        for row in intervals.filter(pl.col("horizon_years") == horizon).iter_rows(named=True)
    }
    n_imputations = frame["imputation"].n_unique()
    estimates: dict[tuple[str, str], list[float]] = defaultdict(list)
    predicted: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    sizes: dict[tuple[str, str], list[int]] = defaultdict(list)
    events: dict[tuple[str, str], list[int]] = defaultdict(list)
    # Identical membership gives an identical estimate, so the cache keys on
    # the member set: the age groups are computed once per cohort, not once
    # per dataset.
    aj_cache: dict[frozenset[str], rbridge.AalenJohansen] = {}
    with progress.bar(
        total=n_imputations * len(groupings), desc="Pooling subgroups", unit="grouping", leave=False
    ) as bar:
        for imputation, frame_k in frame.group_by("imputation"):
            for grouping, (column, labels) in groupings.items():
                bar.set_postfix(imputation=imputation[0], grouping=grouping, refresh=False)
                for label in labels:
                    subset = frame_k.filter(pl.col(column) == label)
                    key = (grouping, label)
                    sizes[key].append(subset.height)
                    events[key].append(
                        int(((subset["event"] == "metastasis") & (subset["time"] <= horizon)).sum())
                    )
                    if subset.is_empty():
                        continue
                    members = frozenset(subset["patient_id"])
                    if members not in aj_cache:
                        outcomes = rbridge.Outcomes(
                            subset["time"], subset["event"], subset["entry_time"]
                        )
                        aj_cache[members] = incidence(outcomes, horizon)
                    estimates[key].append(aj_cache[members].estimate)
                    for model, risk_column in risk_columns.items():
                        predicted[key][model].append(float(cast(Any, subset[risk_column].mean())))
                bar.update()

    rows: list[dict[str, Any]] = []
    for grouping, (_, labels) in groupings.items():
        for order, label in enumerate(labels):
            key = (grouping, label)
            n = float(np.mean(sizes[key])) if sizes[key] else 0.0
            group_events = float(np.mean(events[key])) if events[key] else 0.0
            common: dict[str, Any] = {
                "grouping": grouping,
                "label": label,
                "group_order": order,
                "stratum": stratum,
                "bound": "pt3a",
                "horizon_years": horizon,
                "n": n,
                "events": group_events,
                "small": group_events < minimum_group_n,
                "cohort_n": cohort_n,
            }
            estimate = float(np.mean(estimates[key])) if estimates[key] else None
            lo, hi = bounds.get(key, (None, None))
            rows.append(
                {
                    **common,
                    "kind": "observed",
                    "series": "observed",
                    "estimate": estimate,
                    "lo": lo,
                    "hi": hi,
                    "predicted": None,
                    "method": "bootstrap",
                }
            )
            for model in predict.MODELS:
                risks = predicted[key][model]
                rows.append(
                    {
                        **common,
                        "kind": "predicted",
                        "series": model,
                        "estimate": None,
                        "lo": None,
                        "hi": None,
                        "predicted": float(np.mean(risks)) if risks else None,
                        "method": "mean",
                    }
                )
    return pl.DataFrame(rows).with_columns(
        [
            pl.col(column).cast(pl.Float64)
            for column in ("estimate", "lo", "hi", "predicted", "n", "events")
        ]
    )


def write_subgroup_summaries(
    interim_dir: Path,
    score_dir: Path,
    out_dir: Path,
    cohorts: dict[str, str],
    horizon: float,
    minimum_group_n: int,
    age_cut_points: Iterable[int],
    leibovich_groups: dict[str, tuple[int, int]],
    incidence: Callable[[rbridge.Outcomes, float], rbridge.AalenJohansen] = (
        rbridge.observed_incidence
    ),
) -> list[Path]:
    """Build the subgroup summary table from interim and score-stage tables.

    Point estimates come from each cohort's interim imputed and prediction tables, and
    intervals from `<name>_subgroups.csv` under `score_dir`. Cohorts missing any of the
    three files are skipped. Everything written is aggregate.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    parts = []
    for name, label in cohorts.items():
        imputed_path = interim_dir / f"{name}_imputed.parquet"
        predictions_path = interim_dir / f"{name}_predictions.parquet"
        intervals_path = score_dir / f"{name}_subgroups.csv"
        if not all(path.is_file() for path in (imputed_path, predictions_path, intervals_path)):
            continue
        parts.append(
            subgroup_summary(
                pl.read_parquet(imputed_path),
                pl.read_parquet(predictions_path),
                label,
                horizon,
                minimum_group_n=minimum_group_n,
                age_cut_points=age_cut_points,
                leibovich_groups=leibovich_groups,
                intervals=pl.read_csv(intervals_path),
                incidence=incidence,
            )
        )
    if parts:
        path = out_dir / "subgroups.csv"
        atomic.save(pl.concat(parts), path)
        written.append(path)
    return written
