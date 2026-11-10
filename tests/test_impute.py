"""Multiple imputation: hazard covariates, missingness counts and pooling.

The Nelson-Aalen arithmetic, the missingness summary, the deterministic fills
and the pooling are checked on small frames with hand-computed answers, none
of which need R. The mice-backed tests run on synthetic patients with known
missingness and skip when the R environment is absent.
"""

import numpy as np
import polars as pl
import pytest

from leibovich_validation import (
    cohort,
    impute,
    leibovich,
    params,
    predict,
    score,
    tables,
)
from tests.synthetic import vasudev_patient, vasudev_sample, write_vasudev_tables


def _patients(n: int = 4, **columns) -> pl.DataFrame:
    """A harmonised-shaped patient frame; every predictor recorded by default."""
    base = {
        "cohort": ["transorce"] * n,
        "patient_id": [f"p{i}" for i in range(n)],
        "age_at_surgery": [60.0] * n,
        "sex": ["male"] * n,
        "t_stage": ["pT2"] * n,
        "node_positive": [False] * n,
        "size_ge_10cm": [False] * n,
        "grade": [2] * n,
        "necrosis": [False] * n,
        "nephrectomy_type": [None] * n,
        "arm": ["A"] * n,
        "operation_year": [2008] * n,
        "entry_time": [0.0] * n,
        "prediction_offset": [0.0] * n,
        "time": [4.0] * n,
        "event": ["censored"] * n,
        "estimated_operation_date": [False] * n,
        "leibovich_score_computed": [3] * n,
        "leibovich_score_recorded": [3] * n,
        "dfs_inconsistent": [False] * n,
    }
    base.update(columns)
    return pl.DataFrame(base)


def test_nelson_aalen_cumulative_hazard_worked_example():
    # times 1..4, cause events at 1 and 3, competing event at 4.
    time = pl.Series("time", [1.0, 2.0, 3.0, 4.0])
    event = pl.Series("event", ["metastasis", "censored", "metastasis", "competing"])
    hazard = impute.nelson_aalen_cumulative_hazard(time, event, "metastasis")
    # Y(1) = 4, Y(3) = 2: H = 1/4 by time 1-2, 1/4 + 1/2 by time 3-4.
    assert hazard.to_list() == pytest.approx([0.25, 0.25, 0.75, 0.75])
    competing = impute.nelson_aalen_cumulative_hazard(time, event, "competing")
    # Y(4) = 1: the competing hazard appears only at the last time.
    assert competing.to_list() == pytest.approx([0.0, 0.0, 0.0, 1.0])


def test_nelson_aalen_respects_left_truncation():
    # Patient 1 enters at 2.5, so the risk set at time 2 holds only patients 0
    # and 2; at time 4 all three are at risk.
    time = pl.Series("time", [2.0, 4.0, 6.0])
    event = pl.Series("event", ["metastasis", "metastasis", "censored"])
    entry = pl.Series("entry_time", [0.0, 2.5, 0.0])
    hazard = impute.nelson_aalen_cumulative_hazard(time, event, "metastasis", entry)
    # Y(2) = 2 (patient 1 has not entered), Y(4) = 2 (patient 0 failed at 2):
    # H is 1/2 by time 2, then 1/2 + 1/2 from time 4.
    assert hazard.to_list() == pytest.approx([0.5, 1.0, 1.0])


def test_missingness_summary_counts_known_nulls():
    patients = _patients(
        n=6,
        t_stage=[None, None, "pT1a", "pT2", "pT3a", "pT1b"],
        node_positive=[None, None, False, False, True, False],
        size_ge_10cm=[None, None, False, True, False, False],
        grade=[None, None, 1, 2, 3, 4],
        necrosis=[None, None, False, True, False, False],
    )
    summary = impute.missingness_summary(patients)
    rows = {record["predictor"]: record for record in summary.iter_rows(named=True)}
    assert rows["t_stage"]["n_missing"] == 2
    assert rows["t_stage"]["n_total"] == 6
    assert rows["t_stage"]["pct_missing"] == pytest.approx(33.3)
    assert rows["grade"]["n_missing"] == 2
    # Every predictor is missing in the same two rows, so any_predictor is 2.
    assert rows["any_predictor"]["n_missing"] == 2


def test_complete_patients_give_one_completed_copy():
    patients = _patients(n=5)
    out = impute.impute_frame(patients, m=4, seed=1, show_progress=False)
    assert out.height == 5
    assert out["imputation"].to_list() == [1] * 5
    assert out.columns == ["imputation", *patients.columns]


def test_single_observed_level_fills_deterministically():
    # Necrosis is missing in one row and False everywhere else: the only
    # defensible fill is False, with no mice call.
    patients = _patients(n=5, necrosis=[None, False, False, False, False])
    out = impute.impute_frame(patients, m=3, seed=1, show_progress=False)
    assert out.height == 5
    assert out["necrosis"].to_list() == [False] * 5


def test_pooled_measures_averages_each_key():
    pooled = score._pooled_measures(
        [{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}, {"a": 5.0, "b": 6.0}]
    )
    assert pooled["a"] == pytest.approx(3.0)
    assert pooled["b"] == pytest.approx(4.0)


def test_pooled_risks_averages_patient_wise():
    def _frame(risk):
        return pl.DataFrame(
            {
                predict.risk_column(model, 5.0, bound): risk
                for bound in predict.BOUNDS
                for model in predict.MODELS
            }
        )

    frames = {1: _frame([0.1, 0.2]), 2: _frame([0.3, 0.4])}
    pooled = score._pooled_risks(frames, 5.0)
    assert pooled["Leibovich|pt3a"].to_list() == pytest.approx([0.2, 0.3])


def test_pooled_curve_realigns_observed_to_patient_order():
    # The first dataset's risk sorts patients as p1, p0: the curve rows hold
    # the pseudo-observations in that order, and the pooled risk pairs each
    # patient with its own pseudo-observation again.
    curve = pl.DataFrame({"predicted": [0.1, 0.3], "observed": [0.9, 0.4]})
    first_risk = pl.Series("risk", [0.3, 0.1])
    pooled_risk = pl.Series("risk", [0.25, 0.15])
    out = score._pooled_curve(curve, first_risk, pooled_risk)
    assert out.sort("predicted")["predicted"].to_list() == pytest.approx([0.15, 0.25])
    # Patient order: (0.25, 0.4) then (0.15, 0.9).
    assert out["observed"].to_list() == pytest.approx([0.4, 0.9])


def test_predict_consumes_a_stacked_imputed_table():
    rates = pl.DataFrame(
        [(age, year, 0.01) for age in range(111) for year in range(2024, 2044)],
        schema=["age", "year", "qx"],
        orient="row",
    )
    patients = _patients(n=3)
    stacked = pl.concat([patients.with_columns(pl.lit(k).alias("imputation")) for k in (1, 2)])
    predictions, _ = predict.predict_frame(
        stacked, {"male": rates, "female": rates}, (5.0,), 2024, True, show_progress=False
    )
    assert "imputation" in predictions.columns
    assert predictions.height == 6
    assert predictions["risk_leibovich_5y"].null_count() == 0


def _patients_with_block_missingness(n: int, seed: int) -> tuple[pl.DataFrame, np.ndarray]:
    """Synthetic patients with the cohorts' block-missing predictor pattern."""
    rng = np.random.default_rng(seed)
    missing = rng.random(n) < 0.2
    frame = _patients(
        n=n,
        age_at_surgery=rng.uniform(45, 80, n),
        sex=rng.choice(["male", "female"], n).tolist(),
        t_stage=rng.choice(["pT1a", "pT1b", "pT2", "pT3a"], n).tolist(),
        node_positive=(rng.random(n) < 0.1).tolist(),
        size_ge_10cm=(rng.random(n) < 0.15).tolist(),
        grade=rng.choice([1, 2, 3, 4], n).tolist(),
        necrosis=(rng.random(n) < 0.3).tolist(),
        time=rng.uniform(0.5, 8.0, n),
        event=rng.choice(["metastasis", "competing", "censored"], n, p=[0.3, 0.1, 0.6]).tolist(),
    )
    for column in impute.PREDICTOR_COLUMNS:
        frame = frame.with_columns(
            pl.Series(
                column,
                [
                    None if mask else value
                    for value, mask in zip(frame[column], missing, strict=True)
                ],
            )
        )
    return frame, missing


def test_mice_fills_every_missing_predictor(r_available):
    patients, _ = _patients_with_block_missingness(250, seed=3)
    out = impute.impute_frame(patients, m=2, seed=7, show_progress=False)
    assert set(out["imputation"].unique()) == {1, 2}
    for column in impute.PREDICTOR_COLUMNS:
        assert out[column].null_count() == 0


def test_imputed_values_stay_inside_the_allowed_levels(r_available):
    patients, _ = _patients_with_block_missingness(250, seed=3)
    out = impute.impute_frame(patients, m=2, seed=7, show_progress=False)
    assert set(out["t_stage"].unique()) <= set(leibovich.T_COEF)
    assert out["grade"].is_between(1, 4).all()
    assert out["grade"].dtype == pl.Int64
    for column in ("node_positive", "size_ge_10cm", "necrosis"):
        assert out[column].dtype == pl.Boolean
    # The draws can only produce levels that were observed.
    observed = patients["t_stage"].drop_nulls().unique().to_list()
    assert set(out["t_stage"].unique()) <= set(observed)


def test_observed_values_are_not_changed(r_available):
    patients, missing = _patients_with_block_missingness(250, seed=3)
    out = impute.impute_frame(patients, m=2, seed=7, show_progress=False)
    first = out.filter(pl.col("imputation") == 1)
    observed_rows = ~pl.Series(missing)
    for column in impute.PREDICTOR_COLUMNS:
        kept = first.filter(observed_rows)[column].to_list()
        original = patients.filter(observed_rows)[column].to_list()
        assert kept == original


def test_same_seed_gives_the_same_imputations(r_available):
    patients, _ = _patients_with_block_missingness(150, seed=4)
    first = impute.impute_frame(patients, m=2, seed=11, show_progress=False)
    again = impute.impute_frame(patients, m=2, seed=11, show_progress=False)
    assert first.equals(again)


def test_score_pooled_path_end_to_end(r_available):
    """Impute, predict and pool: the full main-analysis chain on synthetic data."""
    patients, _ = _patients_with_block_missingness(300, seed=5)
    imputed = impute.impute_frame(patients, m=2, seed=9, show_progress=False)
    rates = pl.DataFrame(
        [(age, year, 0.02) for age in range(111) for year in range(2024, 2044)],
        schema=["age", "year", "qx"],
        orient="row",
    )
    predictions, _ = predict.predict_frame(
        imputed, {"male": rates, "female": rates}, (5.0,), 2024, True, show_progress=False
    )
    outputs = score.score_cohort_imputed(
        patients,
        predictions,
        "TranSORCE",
        horizons=(5.0,),
        resamples=3,
        seed=13,
        resample_imputations=1,
        rates_by_sex={"male": rates, "female": rates},
        start_year=2024,
        round_to_percent=True,
        n_bins=3,
        rug_points=5,
        min_at_risk=50,
        bandwidth=0.05,
        minimum_support=20,
        age_cut_points=(60, 70),
        leibovich_groups={"low": (0, 2), "intermediate": (3, 5), "high": (6, 11)},
    )
    metrics = set(outputs.performance["metric"])
    assert {"aj_incidence", "oe_ratio", "calibration_slope", "brier"} <= metrics
    assert outputs.contrasts.height == len(score.CONTRAST_PAIRS) * len(score.CONTRAST_METRICS)
    assert set(outputs.other_cause["group"]) == {"Overall", "<60", "60\u201369", "70+"}
    assert (outputs.performance["metric"] == "oe_other_cause").sum() == 1
    groups = outputs.subgroups
    assert {"age", "leibovich"} == set(groups["grouping"])
    assert set(groups["label"]) <= {"<60", "60\u201369", "70+", "low", "intermediate", "high"}
    assert groups["lower"].is_between(0, 1).all()
    assert (groups["lower"] <= groups["upper"]).all()
    bins = outputs.calibration.filter(pl.col("kind") == "bin")
    assert bins["lo"].is_not_null().all()
    bins = outputs.calibration.filter(pl.col("kind") == "bin")
    assert bins.height == 3 * len(predict.MODELS) * len(predict.BOUNDS)


def test_table_s4_reports_missingness_and_the_model():
    missingness = impute.missingness_summary(
        _patients(n=4, grade=[None, 1, 2, 3], t_stage=[None, "pT1a", "pT2", "pT3a"])
    )
    frame = tables.table_s4(missingness, params.load_analysis(), {"transorce": "TranSORCE"})
    items = dict(zip(frame["item"], frame["value"], strict=True))
    assert items["TranSORCE: grade"] == "1 of 4 (25.0%)"
    assert items["TranSORCE: any_predictor"] == "1 of 4 (25.0%)"
    model_rows = frame.filter(pl.col("section") == "Imputation model")
    text = " ".join(str(v) for v in model_rows["value"])
    for needle in ("chained equations", "Nelson-Aalen", "polyreg", "polr", "logreg"):
        assert needle in text


def test_missingness_summary_counts_recorded_gaps_and_imputed_values():
    patients = _patients(
        n=4,
        # Missing nodes score as node-negative, so the column holds False.
        node_positive=[False] * 4,
        size_ge_10cm=[None, False, False, False],
        missing_predictors=[["node", "size"], ["node"], [], ["node"]],
    )
    summary = impute.missingness_summary(patients)
    rows = {record["predictor"]: record for record in summary.iter_rows(named=True)}
    assert (rows["node_positive"]["n_missing"], rows["node_positive"]["n_imputed"]) == (3, 0)
    assert (rows["size_ge_10cm"]["n_missing"], rows["size_ge_10cm"]["n_imputed"]) == (1, 1)
    assert rows["grade"]["n_missing"] == 0
    # any_predictor counts the patients who have a value imputed.
    assert rows["any_predictor"]["n_missing"] == 1


def _vasudev_tables(tmp_path, n: int, seed: int, size_missing: float, necrosis_missing: float):
    """Vasudev cohort tables with the given shares of missing size and necrosis."""
    rng = np.random.default_rng(seed)
    patients, samples = [], []
    for i in range(n):
        relapsed = rng.random() < 0.3
        patients.append(
            vasudev_patient(
                f"p{i}",
                AGE=int(rng.integers(45, 80)),
                RELAPSE="Yes" if relapsed else "No",
                DFS_STATUS="1:Recurred/Progressed" if relapsed else "0:DiseaseFree",
                DFS_MONTHS=int(rng.integers(6, 96)),
                RCC_MONTHS=96,
            )
        )
        samples.append(
            vasudev_sample(
                f"p{i}",
                STAGE_T=str(rng.choice(["TIa", "TIb", "TII", "TIII"])),
                FFPE_GRADE=int(rng.integers(1, 5)),
                NECROSIS=(
                    None
                    if rng.random() < necrosis_missing
                    else str(rng.choice(["Present", "Absent"]))
                ),
                TUMOR_SIZE=None if rng.random() < size_missing else int(rng.integers(20, 140)),
            )
        )
    return write_vasudev_tables(tmp_path, patients, samples)


def test_vasudev_patient_missing_only_size_keeps_the_rest_through_imputation(tmp_path, r_available):
    patient, sample = _vasudev_tables(tmp_path, 200, seed=2, size_missing=0.2, necrosis_missing=0.0)
    harmonised = cohort.harmonise_vasudev(patient, sample, 0.0).patients
    gaps = harmonised["size_ge_10cm"].is_null()
    assert gaps.sum() > 0
    # Only size is null for these patients.
    for column in ("t_stage", "node_positive", "grade", "necrosis"):
        assert harmonised[column].null_count() == 0
    out = impute.impute_frame(harmonised, m=2, seed=5, show_progress=False)
    for k in (1, 2):
        completed = out.filter(pl.col("imputation") == k)
        for column in ("t_stage", "node_positive", "grade", "necrosis"):
            assert completed[column].to_list() == harmonised[column].to_list()
        assert completed["size_ge_10cm"].null_count() == 0
        assert completed["size_ge_10cm"].dtype == pl.Boolean
        kept = completed.filter(~gaps)["size_ge_10cm"].to_list()
        assert kept == harmonised.filter(~gaps)["size_ge_10cm"].to_list()


def test_missing_necrosis_is_imputed_inside_the_allowed_set(tmp_path, r_available):
    patient, sample = _vasudev_tables(
        tmp_path, 200, seed=4, size_missing=0.0, necrosis_missing=0.15
    )
    harmonised = cohort.harmonise_vasudev(patient, sample, 0.0).patients
    gaps = harmonised["necrosis"].is_null()
    assert gaps.sum() > 0
    out = impute.impute_frame(harmonised, m=2, seed=5, show_progress=False)
    assert out["necrosis"].null_count() == 0
    assert out["necrosis"].dtype == pl.Boolean
    kept = out.filter(pl.col("imputation") == 1).filter(~gaps)["necrosis"].to_list()
    assert kept == harmonised.filter(~gaps)["necrosis"].to_list()
