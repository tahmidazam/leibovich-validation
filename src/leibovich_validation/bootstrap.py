"""Percentile bootstrap intervals by patient resampling, with a progress bar."""

import math
import os
import warnings
from collections.abc import Callable, Iterator, Sequence

import joblib
import numpy as np
import polars as pl
from rich.console import Console

from leibovich_validation import progress
from leibovich_validation.rbridge import BootstrapInterval, EstimationError

# Resamples per worker task: small enough for a responsive progress bar, large
# enough that pickling the statistic is negligible next to one resample.
_CHUNK = 5

Outcome = tuple[dict[str, float] | None, EstimationError | None]


def bootstrap_interval(
    statistic: Callable[[pl.DataFrame], float],
    frame: pl.DataFrame,
    *,
    resamples: int,
    seed: int,
    console: Console | None = None,
    workers: int | None = None,
) -> BootstrapInterval:
    """Percentile interval for `statistic` from patient-level resamples.

    `frame` holds one row per patient with whatever columns `statistic` needs.
    Resamples come from a seeded NumPy generator, so the interval is
    reproducible from `seed`. A resample where `statistic` raises
    EstimationError or returns a non-finite value is skipped and counted; if
    every resample fails, the first error is raised. The shared progress bar
    shows resamples done.
    """
    intervals = bootstrap_intervals(
        lambda resample, _seed: {"estimate": float(statistic(resample))},
        frame,
        resamples=resamples,
        seed=seed,
        console=console,
        workers=workers,
    )
    return intervals["estimate"]


def default_workers() -> int:
    """Worker processes from `LP_WORKERS`, 1 when unset.

    The count changes run time only: every resample depends on its own seed
    alone, so the intervals are the same for any number of workers.
    """
    return max(1, int(os.environ.get("LP_WORKERS", "1")))


def _run_chunk(
    statistic: Callable[[pl.DataFrame, int], dict[str, float]],
    frame: pl.DataFrame,
    seeds: Sequence[int],
) -> list[Outcome]:
    """One worker task: the statistic on each seed's resample, errors kept."""
    outcomes: list[Outcome] = []
    for resample_seed in seeds:
        resample = frame.sample(n=frame.height, with_replacement=True, seed=resample_seed)
        try:
            outcomes.append((statistic(resample, resample_seed), None))
        except EstimationError as exc:
            outcomes.append((None, exc))
    return outcomes


def _outcomes(
    statistic: Callable[[pl.DataFrame, int], dict[str, float]],
    frame: pl.DataFrame,
    seeds: list[int],
    workers: int,
    advance: Callable[[int], None],
) -> Iterator[Outcome]:
    """Each resample's outcome in seed order, run in `workers` processes."""
    if workers == 1:
        for resample_seed in seeds:
            yield from _run_chunk(statistic, frame, [resample_seed])
            advance(1)
        return
    # Fresh (loky) processes, not fork: Polars thread pools do not survive a
    # fork. One thread per worker, since the workers already fill the cores.
    os.environ.setdefault("POLARS_MAX_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    chunks = [seeds[i : i + _CHUNK] for i in range(0, len(seeds), _CHUNK)]
    done: dict[int, list[Outcome]] = {}
    with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
        results = joblib.Parallel(n_jobs=workers, return_as="generator_unordered")(
            joblib.delayed(lambda i, chunk: (i, _run_chunk(statistic, frame, chunk)))(i, chunk)
            for i, chunk in enumerate(chunks)
        )
        for i, chunk_outcomes in results:
            done[i] = chunk_outcomes
            advance(len(chunk_outcomes))
    for i in range(len(chunks)):
        yield from done[i]


def bootstrap_intervals(
    statistic: Callable[[pl.DataFrame, int], dict[str, float]],
    frame: pl.DataFrame,
    *,
    resamples: int,
    seed: int,
    console: Console | None = None,
    workers: int | None = None,
) -> dict[str, BootstrapInterval]:
    """Paired percentile intervals for a statistic that returns several values.

    `statistic` takes the resample and the seed that drew it; a statistic with
    its own randomness, such as re-imputation, seeds it from there, so each
    resample's value depends on its seed alone. Every call returns a mapping
    of measure name to estimate, so all measures see the same resample of the
    patients. That pairing is what makes the intervals valid for contrasts
    such as the difference between two models' observed/expected ratios. A
    resample where `statistic` raises EstimationError is skipped for every
    measure; a measure whose value is non-finite is skipped on its own.
    `workers` (default `default_workers()`) runs the resamples in parallel
    processes; the result does not depend on it.
    """
    if frame.height == 0:
        raise EstimationError("empty data: nothing to resample")
    rng = np.random.default_rng(seed)
    seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(resamples)]
    estimates: dict[str, list[float]] = {}
    skipped = 0
    dropped: dict[str, int] = {}
    seen: set[str] = set()
    first_error: EstimationError | None = None
    with progress.bar(console) as bar:
        task = bar.add_task("Bootstrap resamples", total=resamples)
        outcomes = _outcomes(
            statistic,
            frame,
            seeds,
            workers or default_workers(),
            lambda n: bar.advance(task, n),
        )
        for values, error in outcomes:
            if values is None:
                skipped += 1
                first_error = first_error or error
                continue
            for name in seen - values.keys():
                dropped[name] = dropped.get(name, 0) + 1
            for name, value in values.items():
                seen.add(name)
                if math.isfinite(value):
                    estimates.setdefault(name, []).append(float(value))
                else:
                    dropped[name] = dropped.get(name, 0) + 1
    if skipped:
        warnings.warn(
            f"{skipped} of {resamples} resamples failed; the intervals use the survivors",
            stacklevel=2,
        )
    for name, count in dropped.items():
        warnings.warn(
            f"{count} of {resamples} resamples gave a non-finite value for {name!r}",
            stacklevel=2,
        )
    if not estimates:
        raise EstimationError(
            f"all {resamples} bootstrap resamples failed; first error: {first_error}"
        )
    intervals = {}
    for name, values in estimates.items():
        lower, upper = np.quantile(np.asarray(values), [0.025, 0.975])
        intervals[name] = BootstrapInterval(
            float(lower), float(upper), resamples, skipped + dropped.get(name, 0)
        )
    return intervals
