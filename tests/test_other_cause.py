"""Other-cause O/E: groups, hand-worked incidence and the reported rows."""

import polars as pl
import pytest

from leibovich_validation import other_cause, predict, rbridge


def test_age_groups_are_left_closed():
    ages = pl.Series([40.0, 59.9, 60.0, 69.5, 70.0, 88.0])
    labels = other_cause.age_group(ages, (60, 70)).to_list()
    assert labels == ["<60", "<60", "60\u201369", "60\u201369", "70+", "70+"]


def _frame(expected: float) -> pl.DataFrame:
    """Ten patients, none censored before the 5-year horizon.

    Five are aged 55 and five are aged 65. One in each group dies of another
    cause, and each also has one metastasis, so the other-cause incidence is
    1 in 5 per group and 2 in 10 overall.
    """
    column = predict.other_risk_column(5.0)
    return pl.DataFrame(
        {
            "age_at_surgery": [55.0] * 5 + [65.0] * 5,
            "entry_time": [0.0] * 10,
            "time": [1.0, 2.0, 3.0, 4.0, 6.0, 1.5, 2.5, 3.5, 4.5, 6.0],
            "event": [
                "competing",
                "metastasis",
                "censored",
                "censored",
                "censored",
                "competing",
                "metastasis",
                "censored",
                "censored",
                "censored",
            ],
            column: [expected] * 10,
        }
    )


def test_measures_on_a_hand_worked_example(r_available):
    # Aalen-Johansen by hand. Aged under 60: a death without the event at
    # time 1 with 5 at risk gives 1/5 = 0.2, and nothing after it changes the
    # other-cause incidence. Aged 60 to 69: the death at 1.5 with 5 at risk
    # gives 0.2 too. Overall: 10 at risk at time 1 gives 0.1 with survival
    # 0.9; at time 1.5, 9 at risk, adds 0.9 * 1/9 = 0.1, so 0.2.
    out = other_cause.measures(_frame(0.1), 5.0, (60, 70))
    assert out["aj_other|<60"] == pytest.approx(0.2)
    assert out["oe_other|<60"] == pytest.approx(2.0)
    assert out["aj_other|60\u201369"] == pytest.approx(0.2)
    assert out["aj_other|Overall"] == pytest.approx(0.2)
    assert out["oe_other|Overall"] == pytest.approx(2.0)
    assert "aj_other|70+" not in out


def test_rows_report_group_sizes_and_deaths_for_every_group():
    frame = _frame(0.1)
    interval = rbridge.BootstrapInterval(-0.05, 1.3, 10, 0)
    point = {"aj_other|<60": 0.2, "oe_other|<60": 2.0}
    intervals = {"aj_other|<60": interval, "oe_other|<60": interval}
    rows = other_cause.rows(
        "transorce", 5.0, frame, frame[predict.other_risk_column(5.0)], point, intervals, (60, 70)
    )
    by_group = {row["group"]: row for row in rows}
    assert list(by_group) == ["Overall", "<60", "60\u201369", "70+"]
    assert [by_group[g]["n"] for g in by_group] == [10, 5, 5, 0]
    assert [by_group[g]["events"] for g in by_group] == [2, 1, 1, 0]
    assert by_group["<60"]["mean_predicted"] == pytest.approx(0.1)
    assert by_group["70+"]["mean_predicted"] is None
    # The observed interval is held inside [0, 1]; the O/E interval is not.
    assert by_group["<60"]["observed_lower"] == 0.0
    assert by_group["<60"]["observed_upper"] == 1.0
    assert by_group["<60"]["lower"] == -0.05
    assert by_group["<60"]["upper"] == 1.3


def test_pooled_expected_averages_row_aligned_datasets():
    column = predict.other_risk_column(5.0)
    frames = [pl.DataFrame({column: [0.1, 0.3]}), pl.DataFrame({column: [0.3, 0.5]})]
    assert other_cause.pooled_expected(frames, 5.0).to_list() == pytest.approx([0.2, 0.4])


def test_performance_row_carries_the_overall_estimate():
    interval = rbridge.BootstrapInterval(1.5, 2.5, 10, 0)
    row = other_cause.performance_row(
        "transorce", 5.0, {"oe_other|Overall": 2.0}, {"oe_other|Overall": interval}, 438, 14
    )
    assert (row["metric"], row["model"], row["bound"]) == (
        "oe_other_cause",
        "Leibovich Plus",
        "pt3a",
    )
    assert (row["estimate"], row["lower"], row["upper"], row["events"]) == (2.0, 1.5, 2.5, 14)
