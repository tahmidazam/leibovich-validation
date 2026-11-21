import numpy as np
import polars as pl
from lv_datasets import PREDICTOR_NAMES, SchemaError

from leibovich_validation import leibovich, progress, rbridge
from leibovich_validation.rbridge import EstimationError

# The five Leibovich predictors the harmonised table can leave missing. Each
# is null only where its own value is unrecorded, except nodes, which default
# to node-negative in the main analysis.
PREDICTOR_COLUMNS = ("t_stage", "node_positive", "size_ge_10cm", "grade", "necrosis")

# The mice method per imputed predictor. These choices keep imputed values
# inside the observed levels by construction.
MICE_METHODS = {
    "t_stage": "polyreg",  # unordered stages: polytomous logistic regression
    "node_positive": "logreg",
    "size_ge_10cm": "logreg",
    "grade": "polr",  # grades 1-4 are ordered: proportional-odds regression
    "necrosis": "logreg",
}

# Columns allowed on the right-hand side of every imputation equation: the
# predictors themselves, complete patient covariates, and the outcome
# representatives of Bonneville et al., the cause-specific event indicator
# and the Nelson-Aalen cumulative hazard of each cause.
MODEL_COLUMNS = (
    *PREDICTOR_COLUMNS,
    "age_at_surgery",
    "sex",
    "entry_time",
    "event_metastasis",
    "event_competing",
    "na_hazard_metastasis",
    "na_hazard_competing",
)

CAUSES = ("metastasis", "competing")


def nelson_aalen_cumulative_hazard(
    time: pl.Series,
    event: pl.Series,
    cause: str,
    entry: pl.Series | None = None,
) -> pl.Series:
    """Each patient's Nelson-Aalen cumulative hazard of `cause` at their own time.

    Sums d_j / Y_j over event times t_j <= t, where Y_j counts patients at risk just
    before t_j (entry < t_j <= exit), so delayed entry is respected. The values enter
    the imputation model as the cumulative-hazard covariate of `cause`.
    """
    if cause not in rbridge.EVENT_CODES or rbridge.EVENT_CODES[cause] == 0:
        raise ValueError(f"unknown cause {cause!r}; expected one of {CAUSES}")
    times = np.asarray(time.to_list(), dtype=float)
    events = np.asarray(event.to_list(), dtype=object)
    entries = np.zeros(len(times)) if entry is None else np.asarray(entry.to_list(), dtype=float)
    event_times = np.unique(times[events == cause])
    if event_times.size == 0:
        return pl.Series(f"na_hazard_{cause}", np.zeros(len(times)))
    increments = np.array(
        [
            ((events == cause) & (times == t)).sum() / ((entries < t) & (times >= t)).sum()
            for t in event_times
        ]
    )
    cumulative = np.cumsum(increments)
    positions = np.searchsorted(event_times, times, side="right") - 1
    values = np.where(positions >= 0, cumulative[positions.clip(0)], 0.0)
    return pl.Series(f"na_hazard_{cause}", values)


def resample_completed(patients: pl.DataFrame, m: int, resample_seed: int) -> pl.DataFrame:
    """One bootstrap resample's stacked completed datasets.

    The imputation seed derives from the resample's seed, so draws neither repeat the
    point-estimate imputations nor depend on resample order.
    """
    imp_seed = int(np.random.default_rng(resample_seed).integers(0, 2**31 - 1))
    return impute_frame(patients, m, imp_seed, show_progress=False)


def _imputation_frame(patients: pl.DataFrame) -> pl.DataFrame:
    """The patient table plus the outcome columns the imputation model needs."""
    frame = patients.with_columns(
        (pl.col("event") == "metastasis").cast(pl.Int8).alias("event_metastasis"),
        (pl.col("event") == "competing").cast(pl.Int8).alias("event_competing"),
    )
    for cause in CAUSES:
        hazard = nelson_aalen_cumulative_hazard(
            frame["time"], frame["event"], cause, frame["entry_time"]
        )
        frame = frame.with_columns(hazard)
    return frame


def _split_methods(frame: pl.DataFrame) -> tuple[dict[str, str], dict[str, object]]:
    """Which predictors mice must draw, and which can be filled directly.

    A predictor with no missing values is left out. One whose observed values are all
    equal is filled with that level. One with no observed values cannot be imputed.
    """
    methods: dict[str, str] = {}
    fills: dict[str, object] = {}
    for column in PREDICTOR_COLUMNS:
        if frame[column].null_count() == 0:
            continue
        observed = frame[column].drop_nulls().unique()
        if observed.is_empty():
            raise EstimationError(f"cannot impute {column}: no observed values")
        if observed.len() == 1:
            fills[column] = observed[0]
        else:
            methods[column] = MICE_METHODS[column]
    return methods, fills


def _splice(base: pl.DataFrame, completed: pl.DataFrame, imputation: int) -> pl.DataFrame:
    """The patient table with `completed`'s predictor columns written in.

    mice returns rows in input order, so the splice is positional, not a join on
    patient_id. Factor columns arrive as strings and are cast back to the table's dtypes.
    """
    out = base
    for column in PREDICTOR_COLUMNS:
        series = completed[column]
        dtype = base[column].dtype
        if dtype == pl.Boolean:
            out = out.with_columns((series == "TRUE").alias(column))
        else:
            out = out.with_columns(series.cast(dtype).alias(column))
    return out.with_columns(pl.lit(imputation).alias("imputation"))


def _validate_completed(frame: pl.DataFrame) -> None:
    """The schema check on the stacked completed tables; any violation raises."""
    problems = []
    for column in PREDICTOR_COLUMNS:
        if frame[column].null_count():
            problems.append(f"{column} still has missing values after imputation")
    bad_stages = sorted(set(frame["t_stage"].drop_nulls()) - set(leibovich.T_COEF))
    if bad_stages:
        problems.append(f"imputed t_stage outside the allowed stages: {bad_stages}")
    if frame.filter(~pl.col("grade").is_between(1, 4)).height:
        problems.append("imputed grade is outside 1-4")
    if problems:
        raise SchemaError("; ".join(problems))


def impute_frame(
    patients: pl.DataFrame,
    m: int,
    seed: int,
    maxit: int = 5,
    show_progress: bool = True,
) -> pl.DataFrame:
    """The stacked completed patient tables, marked by an `imputation` column.

    Runs `rbridge.mice_impute` once per dataset when `show_progress` is set, else in a
    single call, which is cheaper inside a bootstrap resample. Dataset k drawn alone
    uses `seed + k`. When nothing needs imputing the result is one copy, since pooling
    identical values gives the same value.
    """
    if m < 1:
        raise ValueError(f"m must be at least 1, got {m}")
    work = _imputation_frame(patients)
    methods, fills = _split_methods(work)
    base = patients.with_columns(pl.col(column).fill_null(level) for column, level in fills.items())
    if not methods:
        return base.with_columns(pl.lit(1).alias("imputation")).select(
            "imputation", *patients.columns
        )
    model_frame = work.with_columns(
        pl.col(column).fill_null(level) for column, level in fills.items()
    ).select(list(MODEL_COLUMNS))
    parts: list[pl.DataFrame] = []
    if show_progress:
        with progress.bar(range(1, m + 1), desc="Imputing", unit="dataset") as bar:
            for k in bar:
                bar.set_postfix(imputation=k, seed=seed + k, refresh=True)
                completed = rbridge.mice_impute(
                    model_frame, methods, list(MODEL_COLUMNS), m=1, seed=seed + k, maxit=maxit
                )
                parts.append(_splice(base, completed, k))
    else:
        completed = rbridge.mice_impute(
            model_frame, methods, list(MODEL_COLUMNS), m=m, seed=seed, maxit=maxit
        )
        for k in completed[".imp"].unique().sort().to_list():
            parts.append(_splice(base, completed.filter(pl.col(".imp") == k), int(k)))
    out = pl.concat(parts).select("imputation", *patients.columns)
    _validate_completed(out)
    return out


def missingness_summary(patients: pl.DataFrame) -> pl.DataFrame:
    """Missing values per predictor, plus the count with any value imputed.

    `n_missing` counts patients whose recorded value was absent, from the
    `missing_predictors` list when present, else from nulls. `n_imputed` counts nulls in
    the column, which differs for nodes because unrecorded nodes score as node-negative.
    `any_predictor` counts patients with at least one value imputed. Aggregate counts
    only; this is the source of Table S4's missing-data rows.
    """
    label = str(patients["cohort"][0]) if patients.height else ""
    n = patients.height
    rows = []
    for column, name in zip(PREDICTOR_COLUMNS, PREDICTOR_NAMES, strict=True):
        imputed = patients[column].null_count()
        missing = (
            patients["missing_predictors"].list.contains(name).sum()
            if "missing_predictors" in patients.columns
            else imputed
        )
        rows.append(
            {
                "cohort": label,
                "predictor": column,
                "n_total": n,
                "n_missing": missing,
                "pct_missing": round(100 * missing / n, 1) if n else None,
                "n_imputed": imputed,
            }
        )
    any_imputed = (
        patients.select(
            pl.any_horizontal([pl.col(column).is_null() for column in PREDICTOR_COLUMNS]).sum()
        ).item()
        if n
        else 0
    )
    rows.append(
        {
            "cohort": label,
            "predictor": "any_predictor",
            "n_total": n,
            "n_missing": any_imputed,
            "pct_missing": round(100 * any_imputed / n, 1) if n else None,
            "n_imputed": any_imputed,
        }
    )
    return pl.DataFrame(rows)
