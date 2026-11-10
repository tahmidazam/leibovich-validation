import polars as pl
import pytest

from leibovich_validation import leibovich, predict, rbridge, report
from leibovich_validation.leibovich import PREPRINT_PROFILES
from tests.synthetic import constant_rates


def test_time_grid_covers_the_model_domain():
    grid = report.time_grid(0.25)
    assert grid[0] == 0
    assert grid[-1] == leibovich.MAX_YEARS == 10
    assert len(grid) == 41


def test_model_behaviour_has_the_fixed_column_roles():
    table = report.model_behaviour(
        constant_rates(0.05),
        ages=[55, 85],
        times=[0.0, 5.0, 10.0],
        start_year=2024,
        round_to_percent=False,
    )
    assert table.columns == ["label", "stratum", "time", "series", "estimate"]
    assert table.height == len(PREPRINT_PROFILES) * 2 * 3 * 2
    assert table["estimate"].is_between(0, 1).all()
    assert set(table["label"].unique()) == set(PREPRINT_PROFILES)
    assert set(table["series"].unique()) == {report.LEIBOVICH, report.LEIBOVICH_PLUS}
    assert set(table["stratum"].unique()) == {55, 85}


def test_model_behaviour_worked_value_without_competing_deaths():
    # With qx = 0 both models reduce to the unadjusted Leibovich risk.
    table = report.model_behaviour(
        constant_rates(0.0), ages=[60], times=[5.0], start_year=2024, round_to_percent=False
    )
    rows = table.filter((pl.col("label") == "low") & (pl.col("time") == 5))
    assert rows["estimate"].to_list() == pytest.approx(
        [leibovich.risk(PREPRINT_PROFILES["low"], 5)] * 2
    )


def test_model_behaviour_original_series_is_the_leibovich_risk():
    table = report.model_behaviour(
        constant_rates(0.05), ages=[60], times=[5.0], start_year=2024, round_to_percent=False
    )
    row = table.filter((pl.col("label") == "high") & (pl.col("series") == report.LEIBOVICH)).row(
        0, named=True
    )
    assert row["estimate"] == pytest.approx(leibovich.risk(PREPRINT_PROFILES["high"], 5))


def performance_frame() -> pl.DataFrame:
    """A score-stage performance table in miniature."""
    rows = [
        {
            "cohort": "transorce",
            "horizon_years": 5.0,
            "bound": "",
            "model": "observed",
            "metric": "aj_incidence",
            "estimate": 0.20,
            "lower": 0.15,
            "upper": 0.26,
            "method": "aalen-johansen se",
            "n": 438,
            "events": 60,
        }
    ]
    for metric, estimate in (
        ("oe_ratio", 0.9),
        ("calibration_slope", 1.1),
        ("ici", 0.03),
        ("auc", 0.7),
        ("brier", 0.12),
        ("mean_predicted", 0.22),
        ("ipa", 0.05),
    ):
        rows.append(
            {
                "cohort": "transorce",
                "horizon_years": 5.0,
                "bound": "pt3a",
                "model": "Leibovich Plus",
                "metric": metric,
                "estimate": estimate,
                "lower": estimate - 0.05,
                "upper": estimate + 0.05,
                "method": "bootstrap",
                "n": 438,
                "events": 60,
            }
        )
    return pl.DataFrame(rows)


def test_table2_summary_keeps_the_table2_metrics_in_figure_roles():
    summary = report.table2_summary(performance_frame())
    assert set(summary["label"]) == {
        "aj_incidence",
        "oe_ratio",
        "calibration_slope",
        "ici",
        "auc",
        "brier",
    }
    assert summary.columns == [
        "label",
        "series",
        "stratum",
        "bound",
        "horizon",
        "estimate",
        "lo",
        "hi",
        "n",
        "events",
        "method",
    ]
    row = summary.filter(pl.col("label") == "oe_ratio").row(0, named=True)
    assert row["series"] == "Leibovich Plus"
    assert row["stratum"] == "transorce"
    assert row["horizon"] == 5.0
    assert row["lo"] == pytest.approx(0.85)


def test_calibration_summary_filters_one_horizon():
    calibration = pl.DataFrame(
        {
            "kind": ["bin", "bin"],
            "horizon_years": [5.0, 3.0],
            "predicted": [0.1, 0.05],
            "estimate": [0.12, 0.06],
        }
    )
    assert report.calibration_summary(calibration, 5.0)["predicted"].to_list() == [0.1]


def test_write_performance_summaries(tmp_path):
    score_dir = tmp_path / "scores"
    score_dir.mkdir()
    performance_frame().write_csv(score_dir / "transorce_performance.csv")
    written = report.write_performance_summaries(score_dir, tmp_path / "summary", 5.0, 10)
    assert [path.name for path in written] == ["table2.csv"]
    assert pl.read_csv(written[0]).columns[0] == "label"


def test_model_behaviour_csv_round_trip(tmp_path):
    table = report.model_behaviour(
        constant_rates(0.02), ages=[55], times=[0.0, 1.0], start_year=2024, round_to_percent=True
    )
    path = tmp_path / "model_behaviour.csv"
    table.write_csv(path)
    assert pl.read_csv(path).columns == table.columns


def flow_frames() -> dict[str, pl.DataFrame]:
    """Two cohort-stage flow tables in the real step/n shape."""
    return {
        "TranSORCE": pl.DataFrame(
            {
                "step": [
                    "workbook_rows",
                    "eligible_trial_record",
                    "excluded_missing_sex",
                    "estimated_operation_date",
                    "excluded_operation_outside_window",
                    "analysis_set",
                ],
                "n": [100, 100, 2, 5, 3, 95],
            }
        ),
        "Vasudev cohort": pl.DataFrame(
            {
                "step": [
                    "patient_rows",
                    "excluded_overall_stage_iv",
                    "excluded_nonpositive_followup",
                    "analysis_set",
                ],
                "n": [200, 30, 1, 169],
            }
        ),
    }


def test_flow_summary_assigns_kind_label_and_count_per_step():
    summary = report.flow_summary(flow_frames())
    assert summary.columns == ["stratum", "step", "kind", "label", "n"]
    rows = {(row["stratum"], row["step"]): row for row in summary.iter_rows(named=True)}
    assert rows[("TranSORCE", "workbook_rows")]["kind"] == "main"
    assert rows[("TranSORCE", "excluded_missing_sex")]["kind"] == "excluded"
    assert rows[("TranSORCE", "estimated_operation_date")]["kind"] == "note"
    assert rows[("TranSORCE", "estimated_operation_date")]["n"] == 5
    assert rows[("Vasudev cohort", "excluded_nonpositive_followup")]["kind"] == "excluded"
    assert rows[("Vasudev cohort", "analysis_set")]["n"] == 169
    assert all(row["label"] for row in summary.iter_rows(named=True))


def test_flow_summary_rejects_an_unknown_step():
    flows = {"TranSORCE": pl.DataFrame({"step": ["new_step"], "n": [1]})}
    with pytest.raises(ValueError, match="new_step"):
        report.flow_summary(flows)


def patients_frame() -> pl.DataFrame:
    """Four harmonised patients covering every Table 1 row with known answers."""
    return pl.DataFrame(
        {
            "age_at_surgery": [50.0, 60.0, 70.0, 80.0],
            "sex": ["male", "female", "male", "female"],
            "t_stage": ["pT1a", "pT2", "pT3a", None],
            "node_positive": [False, True, False, None],
            "size_ge_10cm": [False, True, False, None],
            "grade": [1, 3, 2, None],
            "necrosis": [False, True, False, None],
            "nephrectomy_type": ["radical", "partial", "radical", None],
            "leibovich_score_computed": [0, 4, 7, None],
            "time": [1.0, 3.0, 5.0, 7.0],
            "event": ["censored", "metastasis", "competing", "censored"],
            "missing_predictors": [
                [],
                [],
                ["node", "necrosis"],
                ["t_stage", "node", "size", "grade", "necrosis"],
            ],
        }
    )


GROUPS = {"low": (0, 2), "intermediate": (3, 5), "high": (6, 11)}


def table1_values(frame: pl.DataFrame) -> dict[str, str]:
    """The key -> value lookup for one cohort's Table 1 summary."""
    return dict(zip(frame["key"], frame["value"], strict=True))


def test_table1_summary_worked_example():
    summary = report.table1_summary(patients_frame(), "Cohort X", GROUPS)
    values = table1_values(summary.filter(pl.col("stratum") == "Cohort X"))
    assert values["n_patients"] == "4"
    assert values["age_mean_sd"] == "65.0 (12.9)"
    assert values["age_median_range"] == "65 (50\u201380)"
    assert values["sex_male"] == "2 (50.0%)"
    assert values["t_stage_pT1a"] == "1 (25.0%)"
    assert values["t_stage_pT3b"] == "0 (0.0%)"
    assert values["node_positive"] == "1 (25.0%)"
    # Two node-negative patients; the fourth has a missing node.
    assert values["node_negative"] == "2 (50.0%)"
    assert values["size_ge_10cm"] == "1 (25.0%)"
    assert values["grade_1"] == "1 (25.0%)"
    assert values["grade_4"] == "0 (0.0%)"
    assert values["necrosis"] == "1 (25.0%)"
    assert values["nephrectomy_radical"] == "2 (50.0%)"
    assert values["nephrectomy_partial"] == "1 (25.0%)"
    assert values["risk_low"] == "1 (25.0%)"
    assert values["risk_intermediate"] == "1 (25.0%)"
    assert values["risk_high"] == "1 (25.0%)"
    assert values["followup_median_range"] == "4.0 (1.0\u20137.0)"
    assert values["events_interest"] == "1"
    assert values["events_competing"] == "1"
    assert values["events_censored"] == "2"
    # Node and necrosis were missing for two patients even though they
    # score as negative and absent.
    assert values["missing_node"] == "2 (50.0%)"
    assert values["missing_necrosis"] == "2 (50.0%)"
    assert values["missing_t_stage"] == "1 (25.0%)"
    assert values["missing_nephrectomy_type"] == "1 (25.0%)"
    assert values["missing_score"] == "1 (25.0%)"


def test_table1_summary_empty_nephrectomy_column_shows_empty_cells():
    patients = patients_frame().with_columns(pl.lit(None).cast(pl.String).alias("nephrectomy_type"))
    values = table1_values(report.table1_summary(patients, "X", GROUPS))
    assert values["nephrectomy_radical"] == ""
    assert values["missing_nephrectomy_type"] == "4 (100.0%)"


def test_write_cohort_summaries(tmp_path):
    interim, cohort_dir, out_dir = tmp_path / "interim", tmp_path / "cohort", tmp_path / "out"
    interim.mkdir()
    cohort_dir.mkdir()
    flow_frames()["TranSORCE"].write_csv(cohort_dir / "transorce_flow.csv")
    patients_frame().write_parquet(interim / "transorce_patients.parquet")
    written = report.write_cohort_summaries(
        interim, cohort_dir, out_dir, {"transorce": "TranSORCE"}, GROUPS
    )
    assert [path.name for path in written] == ["flow.csv", "table1.csv"]
    flow = pl.read_csv(out_dir / "flow.csv")
    assert set(flow["kind"]) == {"main", "excluded", "note"}
    table1 = pl.read_csv(out_dir / "table1.csv")
    assert set(table1["stratum"]) == {"TranSORCE"}
    assert table1.filter(pl.col("key") == "n_patients")["value"][0] == "4"


def test_age_group_labels_follow_the_cut_points():
    assert report.age_group_labels([60, 70]) == ["<60", "60\u201369", "70+"]
    assert report.age_group_labels([70]) == ["<70", "70+"]
    with pytest.raises(ValueError, match="empty"):
        report.age_group_labels([])


def subgroup_frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Six patients over two completed datasets; p4's grade differs between them.

    Scores: p1 and p6 are low (0), p3 and p5 are high (7), p2 is intermediate
    (4) and p4 moves from intermediate (4) to high (6) between datasets. The
    events are p2 at 2 years and p4 at 1 year; the other four are censored
    at 6 years.
    """
    first = pl.DataFrame(
        {
            "imputation": [1] * 6,
            "patient_id": [f"p{i}" for i in range(6)],
            "age_at_surgery": [50.0, 60.0, 70.0, 65.0, 75.0, 55.0],
            "t_stage": ["pT1a", "pT2", "pT3a", "pT2", "pT3a", "pT1a"],
            "node_positive": [False] * 6,
            "size_ge_10cm": [False] * 6,
            "grade": [1, 3, 4, 3, 4, 1],
            "necrosis": [False] * 6,
            "entry_time": [0.0] * 6,
            "time": [6.0, 2.0, 6.0, 1.0, 6.0, 6.0],
            "event": [
                "censored",
                "metastasis",
                "censored",
                "metastasis",
                "censored",
                "censored",
            ],
        }
    )
    second = first.with_columns(
        # Explicit Int64: pl.lit materialises small ints as Int32, which
        # cannot vstack onto `first`'s Int64 imputation column.
        pl.lit(2, dtype=pl.Int64).alias("imputation"),
        pl.Series("grade", [1, 3, 4, 4, 4, 1]),
    )
    risks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    columns = {
        "imputation": [1] * 6 + [2] * 6,
        "patient_id": [f"p{i}" for i in range(6)] * 2,
    }
    for slug in ("leibovich", "leibovich_plus"):
        columns[f"risk_{slug}_5y"] = risks * 2
        columns[f"risk_{slug}_5y_pt3b"] = [risk + 0.01 for risk in risks] * 2
    return pl.concat([first, second]), pl.DataFrame(columns)


def _fake_incidence(outcomes: rbridge.Outcomes, horizon: float) -> rbridge.AalenJohansen:
    """The event proportion at the horizon with a fixed SE: a checkable stand-in."""
    n_events = int(((outcomes.event == "metastasis") & (outcomes.time <= horizon)).sum())
    return rbridge.AalenJohansen(n_events / outcomes.time.len(), 0.1)


def _subgroup_row(summary: pl.DataFrame, grouping: str, label: str, series: str) -> dict:
    """The one row of the summary for a group and series."""
    row = summary.filter(
        (pl.col("grouping") == grouping) & (pl.col("label") == label) & (pl.col("series") == series)
    )
    assert row.height == 1
    return row.row(0, named=True)


def subgroup_intervals() -> pl.DataFrame:
    """Score-stage bootstrap bounds for three groups at the 5-year horizon."""
    return pl.DataFrame(
        {
            "grouping": ["age", "leibovich", "age"],
            "label": ["60\u201369", "high", "<60"],
            "horizon_years": [5.0, 5.0, 3.0],
            "lower": [0.7, 0.0, 0.5],
            "upper": [1.0, 0.4, 0.6],
        }
    )


def subgroup_table() -> pl.DataFrame:
    """The subgroup summary for the six-patient worked example."""
    imputed, predictions = subgroup_frames()
    return report.subgroup_summary(
        imputed,
        predictions,
        "Cohort X",
        5.0,
        minimum_group_n=2,
        age_cut_points=(60, 70),
        leibovich_groups={**GROUPS, "extreme": (8, 11)},
        intervals=subgroup_intervals(),
        incidence=_fake_incidence,
    )


def test_subgroup_summary_has_the_fixed_column_roles():
    summary = subgroup_table()
    assert summary.columns == [
        "grouping",
        "label",
        "group_order",
        "stratum",
        "bound",
        "horizon_years",
        "n",
        "events",
        "small",
        "cohort_n",
        "kind",
        "series",
        "estimate",
        "lo",
        "hi",
        "predicted",
        "method",
    ]
    assert set(summary["series"]) == {"observed", *predict.MODELS}
    assert set(summary["bound"]) == {"pt3a"}
    assert (summary["cohort_n"] == 6).all()


def test_subgroup_summary_pools_group_estimates_over_the_datasets():
    summary = subgroup_table()
    # The 60-69 group holds p2 and p4 in both datasets: two events of two.
    row = _subgroup_row(summary, "age", "60\u201369", "observed")
    assert row["estimate"] == pytest.approx(1.0)
    assert row["n"] == 2
    assert row["events"] == 2
    assert row["small"] is False
    # p4 moves to high in the second dataset, so intermediate pools to
    # n = 1.5 and the high group pools an empty and a one-in-three estimate.
    row = _subgroup_row(summary, "leibovich", "intermediate", "observed")
    assert row["estimate"] == pytest.approx(1.0)
    assert row["n"] == pytest.approx(1.5)
    assert row["events"] == pytest.approx(1.5)
    row = _subgroup_row(summary, "leibovich", "high", "observed")
    assert row["estimate"] == pytest.approx(1 / 6)


def test_subgroup_summary_takes_intervals_from_the_score_stage_bootstrap():
    summary = subgroup_table()
    row = _subgroup_row(summary, "age", "60\u201369", "observed")
    assert (row["lo"], row["hi"], row["method"]) == (0.7, 1.0, "bootstrap")
    row = _subgroup_row(summary, "leibovich", "high", "observed")
    assert (row["lo"], row["hi"]) == (0.0, 0.4)
    # The <60 bounds are for another horizon and the other groups have none.
    for grouping, label in (("age", "<60"), ("leibovich", "intermediate")):
        row = _subgroup_row(summary, grouping, label, "observed")
        assert (row["lo"], row["hi"]) == (None, None)
    predicted = summary.filter(pl.col("kind") == "predicted")
    assert predicted["lo"].null_count() == predicted.height


def test_subgroup_summary_predicted_is_the_pooled_group_mean():
    summary = subgroup_table()
    # intermediate holds p2 and p4 then p2 alone: (0.3 + 0.2) / 2.
    row = _subgroup_row(summary, "leibovich", "intermediate", "Leibovich")
    assert row["predicted"] == pytest.approx(0.25)
    assert row["estimate"] is None
    # high holds p3 and p5 then p3, p4 and p5: (0.4 + 0.4) / 2.
    row = _subgroup_row(summary, "leibovich", "high", "Leibovich")
    assert row["predicted"] == pytest.approx(0.4)


def test_subgroup_summary_empty_group_appears_labelled_with_null_estimates():
    summary = subgroup_table()
    row = _subgroup_row(summary, "leibovich", "extreme", "observed")
    assert row["n"] == 0
    assert row["events"] == 0
    assert row["estimate"] is None
    assert row["small"] is True


def test_subgroup_summary_flags_groups_by_events_not_patients():
    summary = subgroup_table()
    # <60 has two patients but no events: flagged all the same.
    row = _subgroup_row(summary, "age", "<60", "observed")
    assert row["n"] == 2
    assert row["events"] == 0
    assert row["small"] is True


def test_write_subgroup_summaries(tmp_path):
    interim, out_dir = tmp_path / "interim", tmp_path / "out"
    interim.mkdir()
    imputed, predictions = subgroup_frames()
    imputed.write_parquet(interim / "transorce_imputed.parquet")
    predictions.write_parquet(interim / "transorce_predictions.parquet")
    score_dir = tmp_path / "scores"
    score_dir.mkdir()
    subgroup_intervals().write_csv(score_dir / "transorce_subgroups.csv")
    written = report.write_subgroup_summaries(
        interim,
        score_dir,
        out_dir,
        {"transorce": "TranSORCE", "vasudev": "Vasudev cohort"},
        5.0,
        2,
        (60, 70),
        GROUPS,
        incidence=_fake_incidence,
    )
    assert [path.name for path in written] == ["subgroups.csv"]
    summary = pl.read_csv(out_dir / "subgroups.csv")
    # Only the cohort with interim files is summarised: three age labels and
    # three risk bands, one observed and two predicted rows each.
    assert set(summary["stratum"]) == {"TranSORCE"}
    assert summary.height == (3 + 3) * 3
    assert set(summary["grouping"]) == {"age", "leibovich"}


def _other_cause_frame() -> pl.DataFrame:
    rows = [("transorce", "Overall", 14), ("transorce", "70+", 3), ("vasudev", "<60", 12)]
    return pl.DataFrame(
        {
            "cohort": [r[0] for r in rows],
            "horizon_years": [5.0] * 3,
            "group": [r[1] for r in rows],
            "events": [r[2] for r in rows],
        }
    )


def test_other_cause_summary_flags_small_groups_and_labels_transorce_descriptive():
    out = report.other_cause_summary(_other_cause_frame(), 10)
    assert out.height == 3
    assert out["small"].to_list() == [False, True, False]
    assert out["status"].to_list() == ["descriptive", "descriptive", ""]


def test_write_performance_summaries_writes_the_other_cause_summary(tmp_path):
    score_dir = tmp_path / "scores"
    score_dir.mkdir()
    _other_cause_frame().write_csv(score_dir / "transorce_other_cause.csv")
    written = report.write_performance_summaries(score_dir, tmp_path / "summary", 5.0, 10)
    assert [path.name for path in written] == ["other_cause.csv"]
    assert pl.read_csv(written[0])["small"].to_list() == [False, True, False]


def test_other_cause_by_age_summary_puts_each_age_group_in_dot_roles():
    # One cohort: the overall row (n = 100) and two age groups. The 70+ group
    # has 3 deaths without the event, under the threshold of 10.
    frame = pl.DataFrame(
        {
            "cohort": ["vasudev"] * 3 + ["vasudev"],
            "horizon_years": [5.0, 5.0, 5.0, 3.0],
            "group": ["Overall", "<60", "70+", "<60"],
            "model": ["Leibovich Plus"] * 4,
            "n": [100, 60, 40, 60],
            "events": [15, 12, 3, 8],
            "observed": [0.15, 0.2, 0.075, 0.1],
            "observed_lower": [0.1, 0.12, 0.02, 0.05],
            "observed_upper": [0.2, 0.3, 0.15, 0.15],
            "mean_predicted": [0.12, 0.05, 0.22, 0.03],
        }
    )
    out = report.other_cause_by_age_summary(frame, {"vasudev": "Vasudev cohort"}, 5.0, 10)
    assert out["label"].to_list() == ["<60", "<60", "70+", "70+"]
    assert out["kind"].to_list() == ["observed", "predicted"] * 2
    assert out["series"].to_list() == ["observed", "Leibovich Plus"] * 2
    assert out["estimate"].to_list() == [0.2, None, 0.075, None]
    assert out["lo"].to_list() == [0.12, None, 0.02, None]
    assert out["predicted"].to_list() == [None, 0.05, None, 0.22]
    assert out["small"].to_list() == [False, False, True, True]
    assert set(out["stratum"]) == {"Vasudev cohort"}
    assert set(out["cohort_n"]) == {100}
