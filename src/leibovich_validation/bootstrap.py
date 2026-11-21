import contextlib
import logging
import math
import os
import time
from collections.abc import Callable, Iterator, Sequence

import joblib
import numpy as np
import polars as pl

from leibovich_validation import progress
from leibovich_validation.rbridge import BootstrapInterval, EstimationError

logger = logging.getLogger(__name__)

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
    workers: int | None = None,
) -> BootstrapInterval:
    """Percentile interval for `statistic` from patient-level resamples.

    `frame` has one row per patient. Resamples use a seeded NumPy generator, so the
    interval is reproducible from `seed`. A resample where `statistic` raises
    EstimationError or returns a non-finite value is skipped and counted; if every
    resample fails, the first error is raised.
    """
    intervals = bootstrap_intervals(
        lambda resample, _seed: {"estimate": float(statistic(resample))},
        frame,
        resamples=resamples,
        seed=seed,
        workers=workers,
    )
    return intervals["estimate"]


def default_workers() -> int:
    """Worker processes from `LV_WORKERS`, 1 when unset.

    Intervals are the same for any count, since each resample depends only on its seed.
    """
    return max(1, int(os.environ.get("LV_WORKERS", "1")))


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
    advance: Callable[[list[Outcome]], None],
) -> Iterator[Outcome]:
    """Each resample's outcome in seed order, run in `workers` processes.

    `advance` sees each finished batch of outcomes, in completion order. An interrupt
    cancels the pending batches and stops the workers.
    """
    if workers == 1:
        for resample_seed in seeds:
            outcomes = _run_chunk(statistic, frame, [resample_seed])
            advance(outcomes)
            yield from outcomes
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
            advance(chunk_outcomes)
    for i in range(len(chunks)):
        yield from done[i]


def bootstrap_intervals(
    statistic: Callable[[pl.DataFrame, int], dict[str, float]],
    frame: pl.DataFrame,
    *,
    resamples: int,
    seed: int,
    workers: int | None = None,
) -> dict[str, BootstrapInterval]:
    """Paired percentile intervals for a statistic that returns several values.

    `statistic` takes the resample and its seed, and returns a mapping of measure name
    to estimate. All measures see the same resample, which keeps intervals valid for
    contrasts such as a difference of O/E ratios. A resample that raises
    EstimationError is skipped for every measure; a non-finite measure is skipped on
    its own. `workers` (default `default_workers()`) does not change the result.
    """
    if frame.height == 0:
        raise EstimationError("empty data: nothing to resample")
    rng = np.random.default_rng(seed)
    seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(resamples)]
    estimates: dict[str, list[float]] = {}
    dropped: dict[str, int] = {}
    seen: set[str] = set()
    first_error: EstimationError | None = None
    workers = workers or default_workers()
    started = time.perf_counter()
    logger.debug("bootstrap: %d resamples, seed %d, %d workers", resamples, seed, workers)
    skipped = 0

    def advance(batch: list[Outcome]) -> None:
        nonlocal skipped
        skipped += sum(values is None for values, _ in batch)
        bar.set_postfix(workers=workers, skipped=skipped, refresh=False)
        bar.update(len(batch))

    with (
        progress.bar(desc="Bootstrap", total=resamples, unit="resample", leave=False) as bar,
        contextlib.closing(_outcomes(statistic, frame, seeds, workers, advance)) as outcomes,
    ):
        for values, error in outcomes:
            if values is None:
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
        logger.warning(
            "%d of %d resamples failed; the intervals use the survivors", skipped, resamples
        )
    for name, count in dropped.items():
        logger.warning("%d of %d resamples gave a non-finite value for %r", count, resamples, name)
    if not estimates:
        raise EstimationError(
            f"all {resamples} bootstrap resamples failed; first error: {first_error}"
        )
    logger.debug("bootstrap: %d resamples in %.1f s", resamples, time.perf_counter() - started)
    intervals = {}
    for name, values in estimates.items():
        lower, upper = np.quantile(np.asarray(values), [0.025, 0.975])
        intervals[name] = BootstrapInterval(
            float(lower), float(upper), resamples, skipped + dropped.get(name, 0)
        )
    return intervals
