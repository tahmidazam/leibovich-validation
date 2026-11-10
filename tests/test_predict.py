"""Predicted risks: worked examples for the delayed-entry conditional forms.

The life tables are synthetic frames with a constant qx at every age and
calendar year, so the other-cause algebra can be checked by hand.
"""

import polars as pl
import pytest

from leibovich_validation import leibovich, leibovich_plus, predict
from tests.synthetic import constant_rates

START_YEAR = 2024
HIGH = leibovich.PREPRINT_PROFILES["high"]
LOW = leibovich.PREPRINT_PROFILES["low"]


def _patients(**overrides) -> pl.DataFrame:
    row = {
        "patient_id": "p1",
        "cohort": "transorce",
        "sex": "male",
        "arm": "A",
        "age_at_surgery": 60.0,
        "entry_time": 0.0,
        "prediction_offset": 0.0,
        "time": 6.0,
        "event": "censored",
        "estimated_operation_date": False,
        "dfs_inconsistent": False,
        "t_stage": "pT2",
        "node_positive": False,
        "size_ge_10cm": False,
        "grade": 3,
        "necrosis": False,
    }
    row.update(overrides)
    return pl.DataFrame([row])


def _rates_by_sex(qx: float = 0.0) -> dict[str, pl.DataFrame]:
    rates = constant_rates(qx)
    return {"male": rates, "female": rates}


def test_risk_column_names():
    assert predict.risk_column(predict.LEIBOVICH, 5) == "risk_leibovich_5y"
    assert predict.risk_column(predict.LEIBOVICH_PLUS, 3, "pt3b") == "risk_leibovich_plus_3y_pt3b"


def test_zero_entry_matches_the_unconditional_models():
    rates = _rates_by_sex(0.02)["male"]
    for years in (3.0, 5.0):
        risks = predict.patient_risks(rates, HIGH, 60, years, 0.0, START_YEAR, True)
        assert risks[predict.LEIBOVICH] == pytest.approx(leibovich.risk(HIGH, years))
        assert risks[predict.LEIBOVICH_PLUS] == pytest.approx(
            leibovich_plus.published_risks(rates, HIGH, 60, years, START_YEAR, True)[0]
        )


def test_conditional_leibovich_is_the_survival_ratio():
    # F(5 | free at 0.25) = 1 - S(5) / S(0.25).
    entry = 0.25
    expected = 1 - (1 - leibovich.risk(HIGH, 5)) / (1 - leibovich.risk(HIGH, entry))
    rates = _rates_by_sex()["male"]
    risks = predict.patient_risks(rates, HIGH, 60, 5.0, entry, START_YEAR, False)
    assert risks[predict.LEIBOVICH] == pytest.approx(expected)


def test_predict_frame_writes_every_column_and_the_pt3b_bound():
    patients = pl.concat(
        [
            _patients(patient_id="a", t_stage="pT3a", prediction_offset=0.2),
            _patients(patient_id="b", t_stage="pT2", prediction_offset=0.2),
        ]
    )
    predictions, summary = predict.predict_frame(
        patients, _rates_by_sex(0.01), (5.0, 3.0), START_YEAR, True
    )
    expected_columns = {
        predict.risk_column(model, horizon, bound)
        for model in predict.MODELS
        for horizon in (5.0, 3.0)
        for bound in predict.BOUNDS
    }
    assert expected_columns <= set(predictions.columns)
    a = predictions.filter(pl.col("patient_id") == "a").row(0, named=True)
    b = predictions.filter(pl.col("patient_id") == "b").row(0, named=True)
    # The pT3b bound raises the pT3a patient's risk and leaves pT2 unchanged.
    assert a["risk_leibovich_5y_pt3b"] > a["risk_leibovich_5y"]
    assert b["risk_leibovich_5y_pt3b"] == pytest.approx(b["risk_leibovich_5y"])
    # The window is (0.2, 5.2]: the risk is conditional on event-free at the
    # offset, and the 5-year horizon is measured from randomisation.
    pt3a = leibovich.Tumour("pT3a", False, False, 3, False)
    assert a["risk_leibovich_5y"] == pytest.approx(
        1 - (1 - leibovich.risk(pt3a, 5.2)) / (1 - leibovich.risk(pt3a, 0.2))
    )
    assert summary.height == len(predict.MODELS) * len(predict.BOUNDS) * 2


def test_missing_predictor_gives_null_risks(tmp_path):
    patients = _patients(t_stage=None)
    predictions, summary = predict.predict_frame(
        patients, _rates_by_sex(), (5.0,), START_YEAR, True
    )
    assert predictions["risk_leibovich_5y"][0] is None
    assert summary["n_missing_risk"].sum() == summary.height


def test_window_risks_over_offset_to_offset_plus_horizon_for_each_model():
    # Worked example: randomised 0.2 years after surgery, high-risk tumour,
    # age 60 and a constant qx of 0.02 a year. Each model's risk is the
    # metastasis risk over (0.2, 5.2] given alive and free at 0.2, and the
    # window has the same length as the observed 5 years from randomisation.
    qx, offset, tau = 0.02, 0.2, 5.0
    end = offset + tau
    predictions, _ = predict.predict_frame(
        _patients(prediction_offset=offset, t_stage="pT3b", node_positive=True, size_ge_10cm=True),
        _rates_by_sex(qx),
        (tau,),
        START_YEAR,
        False,
    )
    row = predictions.row(0, named=True)

    leibovich_expected = 1 - (1 - leibovich.risk(HIGH, end)) / (1 - leibovich.risk(HIGH, offset))
    assert row["risk_leibovich_5y"] == pytest.approx(leibovich_expected)

    disease = leibovich_expected
    other_end = 1 - (1 - qx) ** end
    other_offset = 1 - (1 - qx) ** offset
    other = 1 - (1 - other_end) / (1 - other_offset)
    assert row["risk_leibovich_plus_5y"] == pytest.approx(leibovich_plus.combine(disease, other)[0])


def test_age_at_surgery_is_unchanged_by_the_offset():
    # The conditional path keeps the age at surgery. The qx rises with age, so moving the
    # age to age + offset would change the other-cause risk and this equality would fail.
    records = [(age, 2024 + k, 0.001 * (age - 40)) for age in range(111) for k in range(0, 40)]
    frame = pl.DataFrame(records, schema=["age", "year", "qx"], orient="row")
    rates = {"male": frame, "female": frame}
    predictions, _ = predict.predict_frame(
        _patients(prediction_offset=2.0), rates, (3.0,), START_YEAR, False
    )
    tumour = leibovich.Tumour("pT2", False, False, 3, False)
    expected = leibovich_plus.published_risks(
        rates["male"], tumour, 60, 5.0, START_YEAR, False, entry=2.0
    )[0]
    assert predictions["risk_leibovich_plus_3y"][0] == pytest.approx(expected)


def test_a_conditional_risk_outside_the_unit_interval_raises(monkeypatch):
    monkeypatch.setattr(leibovich_plus, "published_risks", lambda *args, **kwargs: (-0.01, 0.1))
    rates = _rates_by_sex(0.02)["male"]
    with pytest.raises(ValueError, match="outside"):
        predict.patient_risks(rates, HIGH, 60, 5.0, 0.2, START_YEAR, False)


def test_other_cause_risk_over_the_window_on_a_constant_rate_table():
    # Worked example: constant qx = 0.02, high-risk tumour, no rounding. The
    # other-cause risk over (1, 5] given alive and free at 1 year is
    # 1 - 0.98 ** 4 = 0.07763184 before the Leibovich Plus split, and the stored
    # value is the share of the overall risk 1 - (1 - D)(1 - O) that the
    # split assigns to other-cause death: overall * O / (D + O).
    qx, offset, tau = 0.02, 1.0, 4.0
    predictions, _ = predict.predict_frame(
        _patients(prediction_offset=offset, t_stage="pT3b", node_positive=True, size_ge_10cm=True),
        _rates_by_sex(qx),
        (tau,),
        START_YEAR,
        False,
    )
    row = predictions.row(0, named=True)
    other = 1 - 0.98**4
    assert other == pytest.approx(0.07763184)
    disease = 1 - (1 - leibovich.risk(HIGH, 5.0)) / (1 - leibovich.risk(HIGH, 1.0))
    overall = 1 - (1 - disease) * (1 - other)
    expected = overall * other / (disease + other)
    column = predict.other_risk_column(tau)
    assert column == "other_risk_leibovich_plus_4y"
    assert row[column] == pytest.approx(expected)
    # Metastasis and other-cause shares add up to the overall risk.
    assert row["risk_leibovich_plus_4y"] + row[column] == pytest.approx(overall)


def test_other_cause_risk_with_no_offset_and_the_pt3b_column():
    predictions, _ = predict.predict_frame(
        _patients(t_stage="pT3a", node_positive=True, size_ge_10cm=True),
        _rates_by_sex(0.02),
        (5.0,),
        START_YEAR,
        False,
    )
    row = predictions.row(0, named=True)
    other = 1 - 0.98**5
    for bound, t_stage in (("pt3a", "pT3a"), ("pt3b", "pT3b")):
        tumour = leibovich.Tumour(t_stage, True, True, 3, False)
        disease = leibovich.risk(tumour, 5.0)
        overall = 1 - (1 - disease) * (1 - other)
        assert row[predict.other_risk_column(5.0, bound)] == pytest.approx(
            overall * other / (disease + other)
        )


def test_other_cause_risk_is_null_when_a_predictor_is_missing():
    predictions, _ = predict.predict_frame(
        _patients(grade=None), _rates_by_sex(), (5.0,), START_YEAR, True
    )
    assert predictions[predict.other_risk_column(5.0)][0] is None
