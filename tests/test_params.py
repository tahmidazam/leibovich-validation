from dataclasses import fields

import pytest

from leibovich_validation import params

PARAMS_YAML = """\
model:
  life_table: lifetables/qx.xlsx
  start_year: 2024
  round_other_cause_to_percent: true
analysis:
  horizon_years: 5
  secondary_horizon_years: 3
  score_agreement_min: 0.95
  seed: 42
  bootstrap_resamples: 2000
  imputations: 20
  bootstrap_imputations: 5
  min_at_risk: 50
  missing_sex: exclude
  vasudev_who_year: 2006
  sensitivity:
    complete_cases: true
    who_country_life_table_vasudev: false
  subgroups:
    age_cut_points: [60, 70]
    leibovich_groups:
      low: [0, 2]
      intermediate: [3, 5]
      high: [6, 11]
reporting:
  minimum_group_n: 10
  minimum_at_risk: 20
  behaviour_ages: [55, 65, 75, 85]
  behaviour_time_step: 0.25
  behaviour_sex: male
  calibration_bins: 5
  calibration_rug_points: 25
  calibration_bandwidth: 0.05
profiles:
  dev:
    bootstrap_resamples: 10
    imputations: 2
    bootstrap_imputations: 1
"""


@pytest.fixture()
def params_path(tmp_path):
    path = tmp_path / "params.yaml"
    path.write_text(PARAMS_YAML)
    return path


def test_load_model_reads_model_settings(params_path):
    assert params.load_model(params_path) == params.ModelParams(
        life_table="lifetables/qx.xlsx", start_year=2024, round_other_cause_to_percent=True
    )


def test_load_analysis_defaults_to_the_full_counts(params_path):
    assert params.load_analysis(path=params_path) == params.AnalysisParams(
        horizon_years=5,
        secondary_horizon_years=3,
        score_agreement_min=0.95,
        seed=42,
        bootstrap_resamples=2000,
        sensitivity_resamples=2000,
        imputations=20,
        bootstrap_imputations=5,
        missing_sex="exclude",
        vasudev_who_year=2006,
        min_at_risk=50,
        age_cut_points=(60, 70),
        sensitivity={"complete_cases": True, "who_country_life_table_vasudev": False},
        leibovich_groups={"low": (0, 2), "intermediate": (3, 5), "high": (6, 11)},
    )


def test_load_analysis_dev_profile_shrinks_only_the_counts(params_path):
    assert params.load_analysis("dev", params_path) == params.AnalysisParams(
        horizon_years=5,
        secondary_horizon_years=3,
        score_agreement_min=0.95,
        seed=42,
        bootstrap_resamples=10,
        sensitivity_resamples=10,
        imputations=2,
        bootstrap_imputations=1,
        missing_sex="exclude",
        vasudev_who_year=2006,
        min_at_risk=50,
        age_cut_points=(60, 70),
        sensitivity={"complete_cases": True, "who_country_life_table_vasudev": False},
        leibovich_groups={"low": (0, 2), "intermediate": (3, 5), "high": (6, 11)},
    )


def test_load_reporting_reads_thresholds_and_the_behaviour_grid(params_path):
    assert params.load_reporting(params_path) == params.ReportingParams(
        minimum_group_n=10,
        minimum_at_risk=20,
        behaviour_ages=(55, 65, 75, 85),
        behaviour_time_step=0.25,
        behaviour_sex="male",
        calibration_bins=5,
        calibration_rug_points=25,
        calibration_bandwidth=0.05,
    )


def test_load_analysis_rejects_an_unknown_profile(params_path):
    with pytest.raises(KeyError, match="unknown analysis profile"):
        params.load_analysis("nope", params_path)


def test_committed_main_profile_keeps_the_full_counts():
    analysis = params.load_analysis()
    assert analysis.bootstrap_resamples == 2000
    assert analysis.imputations == 20
    assert analysis.bootstrap_imputations == 3


def test_load_sensitivity_defaults_absent_switches_to_off(tmp_path):
    path = tmp_path / "params-no-sensitivity.yaml"
    path.write_text("analysis:\n  horizon_years: 5\n")
    # With no sensitivity block in the file, every switch is off.
    switches = params.load_sensitivity(path)
    assert switches == params.SensitivityParams(
        **{f.name: False for f in fields(params.SensitivityParams)}
    )


def test_load_sensitivity_reads_the_switch_block(tmp_path):
    path = tmp_path / "params-sensitivity.yaml"
    path.write_text("analysis:\n  sensitivity:\n    complete_cases: true\n")
    switches = params.load_sensitivity(path)
    assert switches.complete_cases
    assert not switches.transorce_surgery_clock


def test_committed_switches_are_the_five_sensitivity_analyses():
    switches = params.load_sensitivity()
    assert {f.name for f in fields(params.SensitivityParams)} == {
        "pt3b_for_unsplit_pt3",
        "complete_cases",
        "who_country_life_table_vasudev",
        "transorce_surgery_clock",
        "exclude_inconsistent_vasudev",
    }
    assert all(getattr(switches, f.name) is True for f in fields(params.SensitivityParams))
