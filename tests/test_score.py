"""The score stage: the aggregate outputs.

The calibration binning and paired bootstrap use synthetic tables. One R-backed
smoke test checks the randomisation-clock plumbing end to end, skipped when the R
environment is absent.
"""

from typing import Any, cast

import numpy as np
import polars as pl
import pytest

from leibovich_validation import bootstrap, cohort, predict, rbridge, score, subgroups


def test_bootstrap_intervals_pair_measures_on_the_same_resamples():
    frame = pl.DataFrame({"risk": np.linspace(0.01, 0.9, 50)})

    def statistic(df, _seed):
        mean = float(df["risk"].mean())
        return {"mean": mean, "double": 2 * mean}

    first = bootstrap.bootstrap_intervals(statistic, frame, resamples=30, seed=7)
    again = bootstrap.bootstrap_intervals(statistic, frame, resamples=30, seed=7)
    other = bootstrap.bootstrap_intervals(statistic, frame, resamples=30, seed=8)
    assert first == again
    assert first["mean"] != other["mean"] or first["double"] != other["double"]
    # The paired statistic doubles the mean in every resample, so the
    # intervals scale together.
    assert first["double"].lower == pytest.approx(2 * first["mean"].lower)
    assert first["double"].upper == pytest.approx(2 * first["mean"].upper)


def _seeded_statistic(df, seed):
    # A statistic with its own randomness, like re-imputation, and a resample
    # that cannot be estimated whenever patient 0 is absent.
    if 0 not in df["id"].to_list():
        raise rbridge.EstimationError("no events in this resample")
    noise = float(np.random.default_rng(seed).normal(0, 0.01))
    return {"mean": float(df["risk"].mean()) + noise}


def test_bootstrap_intervals_do_not_depend_on_the_number_of_workers():
    frame = pl.DataFrame({"id": range(8), "risk": np.linspace(0.1, 0.8, 8)})
    serial = bootstrap.bootstrap_intervals(
        _seeded_statistic, frame, resamples=23, seed=3, workers=1
    )
    parallel = bootstrap.bootstrap_intervals(
        _seeded_statistic, frame, resamples=23, seed=3, workers=2
    )
    assert serial == parallel
    assert 0 < serial["mean"].skipped < 23


def test_calibration_rows_bin_and_rug_counts():
    predicted = pl.Series("predicted", np.linspace(0.05, 0.5, 20))
    observed = pl.Series("observed", np.linspace(0.02, 0.6, 20))
    curves = {"model a|pt3a": pl.DataFrame({"predicted": predicted, "observed": observed})}
    rows = score._calibration_rows(
        curves, {"model a|pt3a": predicted}, "TranSORCE", 5.0, 4, 6, 200, 30, {}
    )
    bins = [r for r in rows if r["kind"] == "bin"]
    rug = [r for r in rows if r["kind"] == "rug"]
    assert len(bins) == 4
    assert len(rug) == 6
    assert all(r["series"] == "model a" and r["bound"] == "pt3a" for r in rows)
    assert all(r["cohort_n"] == 200 and r["events"] == 30 for r in rows)
    # Bin means are ordered by predicted risk; rug quantiles span the range.
    predicted_bins = [float(cast(Any, r["predicted"])) for r in bins]
    assert predicted_bins == sorted(predicted_bins)
    assert rug[0]["predicted"] == pytest.approx(predicted.min())
    assert rug[-1]["predicted"] == pytest.approx(predicted.max())
    assert all(r["lo"] is None and r["hi"] is None for r in bins)


def test_smooth_rows_follow_the_smoother_and_stop_at_the_support_floor():
    try:
        checks = rbridge.doctor()
    except Exception as exc:
        pytest.skip(f"R bridge cannot start: {exc}")
    if any(not check.ok for check in checks):
        pytest.skip("R environment not ready")
    # Two patients near 0.1 (pseudo-values 0 and 1) and one each at 0.2 and 0.9.
    curve = pl.DataFrame({"predicted": [0.1, 0.1, 0.2, 0.9], "observed": [0.0, 1.0, 0.5, 1.0]})
    rows = score._smooth_rows({"model a|pt3a": curve}, "TranSORCE", 5.0, 0.05, 2, 200, 30)
    assert len(rows) == 1
    assert rows[0]["kind"] == "smooth"
    assert rows[0]["series"] == "model a" and rows[0]["bound"] == "pt3a"
    assert rows[0]["predicted"] == pytest.approx(0.1)
    assert rows[0]["estimate"] == pytest.approx(0.5)
    assert rows[0]["n"] == 2


def test_min_at_risk_fallback_promotes_the_secondary_horizon():
    """Worked example: a small risk set at 5 years moves the headline to 3.

    Six patients: two exit before 3 years, two between 3 and 5, and two are
    still followed at 5 years; one more entered at 4 years, so it is in the
    5-year risk set but not the 3-year one.
    """
    frame = pl.DataFrame(
        {
            "entry_time": [0.0, 0.0, 0.0, 0.0, 0.0, 4.0],
            "time": [1.0, 2.0, 4.0, 4.5, 6.0, 8.0],
        }
    )
    assert score.risk_set_size(frame, 5.0) == 2
    assert score.risk_set_size(frame, 3.0) == 3
    counts = {5.0: 2, 3.0: 3}
    # Two at risk at 5 years is below the threshold of 3, so the headline
    # moves to the secondary horizon; at the threshold the primary holds.
    assert score.headline_horizon(counts, (5.0, 3.0), min_at_risk=3) == 3.0
    assert score.headline_horizon(counts, (5.0, 3.0), min_at_risk=2) == 5.0


def test_risk_set_counts_patients_on_the_randomisation_clock():
    # Randomised 0.3 years after surgery and last seen at 5.2 years: on the
    # surgery clock the patient is still followed at 5 years, but from
    # randomisation follow-up ends at 4.9 years, so the patient is not in the
    # 5-year risk set. The second patient (0.1 to 6.0) stays in it.
    frame = pl.DataFrame(
        {
            "age_at_surgery": [60.0, 60.0],
            "entry_time": [0.3, 0.1],
            "time": [5.2, 6.0],
        }
    )
    assert score.risk_set_size(frame, 5.0) == 2
    reorigined = cohort.reorigin_at_randomisation(frame)
    assert score.risk_set_size(reorigined, 5.0) == 1
    assert score.risk_set_size(reorigined, 3.0) == 2


def test_fallback_rows_record_counts_and_the_headline():
    frame = pl.DataFrame(
        {
            "entry_time": [0.0, 0.0, 0.0, 0.0, 0.0, 4.0],
            "time": [1.0, 2.0, 4.0, 4.5, 6.0, 8.0],
        }
    )
    rows = score._fallback_rows("transorce", frame, (5.0, 3.0), min_at_risk=3)
    at_risk = {r["horizon_years"]: r for r in rows if r["metric"] == "at_risk"}
    assert at_risk[5.0]["estimate"] == 2.0 and at_risk[5.0]["n"] == 2
    assert at_risk[3.0]["estimate"] == 3.0 and at_risk[3.0]["n"] == 3
    (headline,) = [r for r in rows if r["metric"] == "headline_horizon"]
    assert headline["estimate"] == 3.0
    assert headline["horizon_years"] == 3.0
    assert headline["cohort"] == "transorce"


def _predictions(n: int, seed: int) -> pl.DataFrame:
    """A predictions frame as predict.predict_frame writes it on the randomisation clock."""
    rng = np.random.default_rng(seed)
    exit_times = rng.uniform(0.5, 9.0, n)
    return pl.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(n)],
            "cohort": ["transorce"] * n,
            "sex": ["male"] * n,
            "arm": ["A"] * n,
            "age_at_surgery": rng.uniform(50, 75, n),
            "entry_time": np.zeros(n),
            "time": exit_times,
            "event": rng.choice(
                ["metastasis", "competing", "censored"], n, p=[0.3, 0.1, 0.6]
            ).tolist(),
            "estimated_operation_date": [False] * n,
            **{
                f"other_risk_leibovich_plus_{h}y{suffix}": rng.uniform(0.02, 0.2, n)
                for h in (5, 3)
                for suffix in ("", "_pt3b")
            },
            **{
                f"risk_{slug}_{h}y{suffix}": rng.uniform(0.05, 0.6, n)
                for slug in ("leibovich", "leibovich_plus")
                for h in (5, 3)
                for suffix in ("", "_pt3b")
            },
        }
    )


def test_score_cohort_end_to_end_on_the_randomisation_clock():
    """The full score path: AJ, O/E, calibration, Score and bootstrap."""
    try:
        checks = rbridge.doctor()
    except Exception as exc:
        pytest.skip(f"R bridge cannot start: {exc}")
    if any(not check.ok for check in checks):
        pytest.skip("R environment not ready")
    frame = _predictions(n=200, seed=3)
    outputs = score.score_cohort(
        frame,
        "TranSORCE",
        horizons=(5.0,),
        resamples=5,
        seed=11,
        n_bins=3,
        rug_points=5,
        min_at_risk=50,
        bandwidth=0.05,
        minimum_support=20,
        age_cut_points=(60, 70),
    )
    perf = outputs.performance
    assert {"aj_incidence", "oe_ratio", "calibration_slope", "ici", "auc", "brier"} <= set(
        perf["metric"]
    )
    ici = perf.filter(pl.col("metric") == "ici")
    assert ici.height == 4  # each model and bound
    assert ici["lower"].null_count() == 0
    assert (ici["lower"] <= ici["upper"]).all()
    smooth = outputs.calibration.filter(pl.col("kind") == "smooth")
    assert smooth.height > 0
    assert (smooth["n"] >= 20).all()
    scored = perf.filter(~pl.col("metric").is_in(["at_risk", "headline_horizon"]))
    assert (scored["n"] == 200).all()
    assert set(outputs.contrasts["metric"]) == {"oe", "slope", "brier"}
    bins = outputs.calibration.filter(pl.col("kind") == "bin")
    assert bins.height == 3 * len(predict.MODELS) * len(predict.BOUNDS)  # three bins each
    assert set(outputs.calibration["stratum"]) == {"TranSORCE"}


def test_observed_row_and_bins_clip_percentile_intervals_to_the_unit_interval():
    aj = rbridge.AalenJohansen(0.02, 0.05)
    wide = rbridge.BootstrapInterval(-0.03, 1.2, 100, 0)
    row = score._observed_row("transorce", 5.0, 40, 1, aj, wide)
    assert (row["lower"], row["upper"], row["method"]) == (0.0, 1.0, "bootstrap")
    assert score._observed_row("transorce", 5.0, 40, 1, aj, None)["lower"] is None
    predicted = pl.Series("predicted", np.linspace(0.05, 0.5, 8))
    curves = {"m|pt3a": pl.DataFrame({"predicted": predicted, "observed": predicted})}
    intervals = {f"bin|m|pt3a|{k}": wide for k in range(2)}
    rows = score._calibration_rows(curves, {"m|pt3a": predicted}, "T", 5.0, 2, 3, 8, 1, intervals)
    bins = [r for r in rows if r["kind"] == "bin"]
    assert [(r["lo"], r["hi"]) for r in bins] == [(0.0, 1.0)] * 2


def test_bin_measures_are_the_mean_observed_value_per_equal_count_bin():
    curve = pl.DataFrame({"predicted": [0.4, 0.1, 0.3, 0.2], "observed": [0.9, 0.0, 0.5, 0.1]})
    assert score._bin_measures({"m|pt3a": curve}, 2) == pytest.approx(
        {"bin|m|pt3a|0": 0.05, "bin|m|pt3a|1": 0.7}
    )


def test_intervals_for_incidence_and_bins_lie_in_the_unit_interval_with_few_events(r_available):
    frame = _predictions(n=60, seed=8).with_columns(
        pl.when(pl.col("event") == "metastasis")
        .then(pl.lit("censored"))
        .otherwise(pl.col("event"))
        .alias("event")
    )
    frame = frame.with_columns(
        pl.when(pl.int_range(pl.len()) < 2)
        .then(pl.lit("metastasis"))
        .otherwise(pl.col("event"))
        .alias("event")
    )
    outputs = score.score_cohort(
        frame,
        "TranSORCE",
        horizons=(5.0,),
        resamples=30,
        seed=5,
        n_bins=3,
        rug_points=4,
        min_at_risk=50,
        bandwidth=0.05,
        minimum_support=20,
        age_cut_points=(60, 70),
    )
    observed = outputs.performance.filter(pl.col("metric") == "aj_incidence")
    assert observed["method"].to_list() == ["bootstrap"]
    assert observed["lower"].is_between(0, 1).all()
    assert observed["upper"].is_between(0, 1).all()
    bins = outputs.calibration.filter(pl.col("kind") == "bin")
    assert bins["lo"].is_between(0, 1).all()
    assert bins["hi"].is_between(0, 1).all()
    other = outputs.other_cause
    assert set(other["group"]) == {"Overall", "<60", "60\u201369", "70+"}
    assert other.filter(pl.col("group") == "Overall")["n"].to_list() == [60]


def test_one_resample_statistic_holds_model_incidence_bin_and_other_cause_measures(r_available):
    frame = _predictions(n=120, seed=4)
    columns = [predict.risk_column(model, 5.0, bound) for model, bound in score.NAMES.values()]
    measures = score._point_estimates(frame, score.NAMES, columns, 5.0, 0.05, 3, (60, 70))
    assert {"aj|observed", "oe|Leibovich|pt3a", "bin|Leibovich|pt3a|2"} <= set(measures)
    assert {"aj_other|Overall", "oe_other|Overall", "oe_other|<60"} <= set(measures)


def _grouped_frame() -> pl.DataFrame:
    """Eight patients in two age groups; every event comes before the 5-year horizon.

    The younger group has one event in four patients and the older group has
    three in four, so the Aalen-Johansen incidence is 0.25 and 0.75.
    """
    return pl.DataFrame(
        {
            "_group_age": ["<60"] * 4 + ["70+"] * 4,
            "time": [1.0, 6.0, 6.0, 6.0, 1.0, 2.0, 3.0, 6.0],
            "event": ["metastasis"] + ["censored"] * 3 + ["metastasis"] * 3 + ["censored"],
            "entry_time": [0.0] * 8,
        }
    )


_AGE_GROUPINGS = {"age": ("_group_age", ["<60", "60\u201369", "70+"])}


def test_subgroup_measures_are_the_incidence_of_each_nonempty_group(r_available):
    out = subgroups.measures(_grouped_frame(), 5.0, _AGE_GROUPINGS)
    assert out == pytest.approx({"aj_group|age|<60": 0.25, "aj_group|age|70+": 0.75})


def test_subgroup_measures_give_zero_for_a_group_without_events():
    frame = _grouped_frame().with_columns(pl.lit("censored").alias("event"))
    out = subgroups.measures(frame, 5.0, _AGE_GROUPINGS)
    assert out == {"aj_group|age|<60": 0.0, "aj_group|age|70+": 0.0}


def test_subgroup_labels_follow_the_completed_dataset_not_the_predictions():
    completed = pl.DataFrame(
        {
            "age_at_surgery": [55.0, 65.0],
            "t_stage": ["pT1a", "pT3a"],
            "node_positive": [False, False],
            "size_ge_10cm": [False, False],
            "grade": [1, 4],
            "necrosis": [False, False],
        }
    )
    predictions = pl.DataFrame({"time": [1.0, 2.0]})
    out = subgroups.labelled(predictions, completed, (60, 70), {"low": (0, 2), "high": (3, 11)})
    assert out["_group_age"].to_list() == ["<60", "60\u201369"]
    assert out["_group_leibovich"].to_list() == ["low", "high"]


def test_subgroup_intervals_are_the_bootstrap_percentiles_clipped_to_the_unit_interval(
    r_available,
):
    frame = _grouped_frame()
    intervals = bootstrap.bootstrap_intervals(
        lambda resample, _seed: subgroups.measures(resample, 5.0, _AGE_GROUPINGS),
        frame,
        resamples=40,
        seed=7,
    )
    rng = np.random.default_rng(7)
    expected: dict[str, list[float]] = {"<60": [], "70+": []}
    for _ in range(40):
        resample = frame.sample(
            n=frame.height, with_replacement=True, seed=int(rng.integers(0, 2**31 - 1))
        )
        for label, values in expected.items():
            subset = resample.filter(pl.col("_group_age") == label)
            if subset.height:
                values.append(
                    float(
                        cast(
                            Any,
                            ((subset["event"] == "metastasis") & (subset["time"] <= 5.0)).mean(),
                        )
                    )
                )  # type: ignore[arg-type]
    rows = subgroups.rows(5.0, intervals)
    assert [(r["grouping"], r["label"]) for r in rows] == [("age", "<60"), ("age", "70+")]
    for row, values in zip(rows, expected.values(), strict=True):
        lower, upper = np.quantile(values, [0.025, 0.975])
        assert row["lower"] == pytest.approx(lower)
        assert row["upper"] == pytest.approx(upper)
        assert 0.0 <= cast(float, row["lower"]) <= cast(float, row["upper"]) <= 1.0


def test_subgroup_rows_clip_wide_intervals_and_ignore_other_measures():
    wide = rbridge.BootstrapInterval(-0.1, 1.3, 50, 0)
    rows = subgroups.rows(
        5.0, {"aj_group|age|<60": wide, "aj|observed": wide, "bin|m|pt3a|0": wide}
    )
    assert rows == [
        {"grouping": "age", "label": "<60", "horizon_years": 5.0, "lower": 0.0, "upper": 1.0}
    ]


def test_a_group_empty_in_some_resamples_keeps_an_interval_from_the_rest(r_available):
    frame = _grouped_frame().with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.lit("60\u201369"))
        .otherwise(pl.col("_group_age"))
        .alias("_group_age")
    )
    intervals = bootstrap.bootstrap_intervals(
        lambda resample, _seed: subgroups.measures(resample, 5.0, _AGE_GROUPINGS),
        frame,
        resamples=30,
        seed=2,
    )
    middle = intervals["aj_group|age|60\u201369"]
    assert 0 < middle.skipped < 30
    assert 0.0 <= middle.lower <= middle.upper <= 1.0
