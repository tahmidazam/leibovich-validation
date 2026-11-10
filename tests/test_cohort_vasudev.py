"""The Vasudev cohort stage on synthetic cBioPortal files.

Each test writes the patient and sample tables in the real cBioPortal format,
reads them through the raw readers and runs `harmonise_vasudev`. There is
one worked example per event rule, eligibility rule, stage rule and anomaly
count: the DFS-field inconsistencies, the stage exclusions
and the predictor mapping.
"""

from pathlib import Path

import pytest

from leibovich_validation import cohort
from leibovich_validation.raw_vasudev import SchemaError
from tests.synthetic import vasudev_patient, vasudev_sample, write_vasudev_tables


def _build(
    tmp_path: Path, patients: list[dict], samples: list[dict], min_agree: float = 0.0
) -> cohort.CohortResult:
    patient, sample = write_vasudev_tables(tmp_path, patients, samples)
    return cohort.harmonise_vasudev(patient, sample, min_agree)


def _with_samples(tmp_path: Path, patients: list[dict], min_agree: float = 0.0):
    samples = [vasudev_sample(row["PATIENT_ID"]) for row in patients]
    return _build(tmp_path, patients, samples, min_agree)


def _by_id(result: cohort.CohortResult) -> dict[str, dict]:
    return {row["patient_id"]: row for row in result.patients.iter_rows(named=True)}


def test_event_rules_one_worked_example_each(tmp_path):
    cases = {
        # RELAPSE records the event at the DFS time.
        "relapse": {"RELAPSE": "Yes", "DFS_STATUS": "1:Recurred/Progressed", "DFS_MONTHS": 36},
        # RCC death without a recorded relapse counts as relapse at death.
        "rcc_death": {
            "OS_STATUS": "1:DECEASED",
            "RCC_RELATED_DEATH": "Yes",
            "RCC_STATUS": "1:DECEASED",
            "RCC_MONTHS": 30,
        },
        # Relapsed but coded disease-free: an event, flagged inconsistent.
        "flagged": {"RELAPSE": "Yes", "DFS_STATUS": "0:DiseaseFree", "DFS_MONTHS": 20},
        # A missing RELAPSE takes its event status from DFS_STATUS.
        "missing_relapse_recurred": {
            "RELAPSE": None,
            "DFS_STATUS": "1:Recurred/Progressed",
            "DFS_MONTHS": 48,
        },
        "missing_relapse_free": {"RELAPSE": None, "DFS_STATUS": "0:DiseaseFree"},
        # Death without relapse of a recorded non-RCC cause is competing.
        "other_death": {
            "OS_STATUS": "1:DECEASED",
            "RCC_RELATED_DEATH": "No",
            "RCC_MONTHS": 48,
        },
        # A death of unknown cause counts as other-cause.
        "unknown_death": {"OS_STATUS": "1:DECEASED", "RCC_STATUS": None},
        # Both event fields missing, alive at last contact: censored.
        "underivable_alive": {"RELAPSE": None, "DFS_STATUS": None},
        # Both event fields missing but RCC death: still a relapse at death.
        "rcc_death_underivable": {
            "RELAPSE": None,
            "DFS_STATUS": None,
            "OS_STATUS": "1:DECEASED",
            "RCC_RELATED_DEATH": "Yes",
            "RCC_STATUS": "1:DECEASED",
            "RCC_MONTHS": 40,
        },
        "alive": {},
    }
    patients = [vasudev_patient(key, **overrides) for key, overrides in cases.items()]
    result = _with_samples(tmp_path, patients)
    table = _by_id(result)
    for key in (
        "relapse",
        "rcc_death",
        "flagged",
        "missing_relapse_recurred",
        "rcc_death_underivable",
    ):
        assert table[key]["event"] == "metastasis", key
    for key in ("other_death", "unknown_death"):
        assert table[key]["event"] == "competing", key
    for key in ("missing_relapse_free", "underivable_alive", "alive"):
        assert table[key]["event"] == "censored", key
    # Events are timed at the DFS time; deaths at the survival time.
    assert table["relapse"]["time"] == pytest.approx(36 / 12)
    assert table["rcc_death"]["time"] == pytest.approx(30 / 12)
    assert table["other_death"]["time"] == pytest.approx(48 / 12)
    assert table["rcc_death_underivable"]["time"] == pytest.approx(40 / 12)
    # Only the inconsistent record is flagged.
    assert table["flagged"]["dfs_inconsistent"] is True
    assert sum(p["dfs_inconsistent"] for p in table.values()) == 1
    flow = dict(result.flow.iter_rows())
    assert flow["missing_relapse_event_status_from_dfs_status"] == 2
    assert flow["relapse_and_dfs_status_both_missing"] == 2
    assert flow["relapsed_but_coded_disease_free"] == 1


def test_dfs_time_is_capped_at_the_survival_time(tmp_path):
    patients = [
        vasudev_patient(
            "event",
            RELAPSE="Yes",
            DFS_STATUS="1:Recurred/Progressed",
            DFS_MONTHS=90,
            RCC_MONTHS=60,
        ),
        vasudev_patient("censored", DFS_MONTHS=90, RCC_MONTHS=60),
        vasudev_patient("ok"),
    ]
    result = _with_samples(tmp_path, patients)
    table = _by_id(result)
    assert table["event"]["event"] == "metastasis"
    assert table["event"]["time"] == pytest.approx(5.0)
    assert table["censored"]["event"] == "censored"
    assert table["censored"]["time"] == pytest.approx(5.0)
    assert table["ok"]["time"] == pytest.approx(6.0)
    assert dict(result.flow.iter_rows())["dfs_time_capped_at_survival_time"] == 2


def test_missing_dfs_time_falls_back_to_the_survival_time(tmp_path):
    patients = [
        vasudev_patient("missing", DFS_MONTHS=None, RCC_MONTHS=72),
        vasudev_patient("ok"),
    ]
    result = _with_samples(tmp_path, patients)
    table = _by_id(result)
    assert table["missing"]["event"] == "censored"
    assert table["missing"]["time"] == pytest.approx(6.0)
    assert dict(result.flow.iter_rows())["missing_dfs_time_used_survival_time"] == 1


def test_eligibility_exclusions_are_counted_in_order(tmp_path):
    patients = [
        vasudev_patient("kept"),
        vasudev_patient("kept_missing_m", STAGE="II"),  # missing STAGE_M stays eligible
        vasudev_patient("no_stage", STAGE=None),
        vasudev_patient("stage_iv", STAGE="IV"),
        vasudev_patient("m1", STAGE="III"),
        vasudev_patient("no_sex", SEX=None),
    ]
    samples = [vasudev_sample(row["PATIENT_ID"]) for row in patients]
    samples[4]["STAGE_M"] = "MI"  # recorded M1 lives on the sample row
    result = _build(tmp_path, patients, samples)
    assert _by_id(result).keys() == {"kept", "kept_missing_m"}
    flow = dict(result.flow.iter_rows())
    assert flow["patient_rows"] == 6
    assert flow["excluded_missing_overall_stage"] == 1
    assert flow["excluded_overall_stage_iv"] == 1
    assert flow["excluded_recorded_m1"] == 1
    assert flow["excluded_missing_sex"] == 1
    assert flow["analysis_set"] == 2


def test_t_stage_and_nodal_mapping(tmp_path):
    patients = [
        vasudev_patient("t1a"),
        vasudev_patient("t3", LEIBOVICH_SCORE=4),
        vasudev_patient("t1_unsplit", LEIBOVICH_SCORE=None),
        vasudev_patient("n1", LEIBOVICH_SCORE=2),
        vasudev_patient("g4", LEIBOVICH_SCORE=4),
    ]
    samples = [
        vasudev_sample("t1a"),
        vasudev_sample("t3", STAGE_T="TIII", STAGE_T_CATEGORY="TIII"),
        vasudev_sample("t1_unsplit", STAGE_T="TI"),
        vasudev_sample("n1", STAGE_N="NI"),
        vasudev_sample("g4", FFPE_GRADE=4, NECROSIS="Present"),
    ]
    result = _build(tmp_path, patients, samples)
    table = _by_id(result)
    assert table["t1a"]["t_stage"] == "pT1a"
    # Unsplit pT3 takes the pT3a coefficient in the main analysis.
    assert table["t3"]["t_stage"] == "pT3a"
    # Unsplit pT1 is missing, so its predictors are missing.
    assert table["t1_unsplit"]["t_stage"] is None
    assert table["t1_unsplit"]["leibovich_score_computed"] is None
    # Recorded NI is node-positive; missing nodal status is node-negative.
    assert table["n1"]["node_positive"] is True
    assert table["t1a"]["node_positive"] is False
    assert table["g4"]["necrosis"] is True
    # t3: pT3a 4 points; n1: node positivity adds 2; g4: grade 3 + necrosis 1.
    assert table["t3"]["leibovich_score_computed"] == 4
    assert table["n1"]["leibovich_score_computed"] == 2
    assert table["g4"]["leibovich_score_computed"] == 4


def test_score_agreement_is_reported_as_counts(tmp_path):
    patients = [
        vasudev_patient("agree"),
        vasudev_patient("disagree", LEIBOVICH_SCORE=5),
        vasudev_patient("unscored", LEIBOVICH_SCORE=None),
    ]
    result = _with_samples(tmp_path, patients)
    agreement = dict(result.score_agreement.iter_rows())
    assert agreement["total_compared"] == 2
    assert agreement["agree"] == 1
    assert agreement["disagrees_unexplained"] == 1


def test_score_agreement_below_the_threshold_stops_the_stage(tmp_path):
    patients = [vasudev_patient("p1", LEIBOVICH_SCORE=5)]
    with pytest.raises(SchemaError, match="stop threshold"):
        _with_samples(tmp_path, patients, min_agree=0.95)


def test_harmonised_table_uses_the_shared_schema(tmp_path):
    result = _with_samples(tmp_path, [vasudev_patient("p1")])
    assert set(cohort.HARMONISED_COLUMNS) <= set(result.patients.columns)
    (row,) = result.patients.iter_rows(named=True)
    assert row["cohort"] == "vasudev"
    assert row["entry_time"] == 0.0
    assert row["event"] == "censored"
    assert row["nephrectomy_type"] == "radical"
    assert row["country"] == "United Kingdom"
    assert row["arm"] is None
    assert row["operation_year"] is None
    assert row["estimated_operation_date"] is False
    assert row["dfs_inconsistent"] is False


def test_months_are_returned_in_years(tmp_path):
    result = _with_samples(tmp_path, [vasudev_patient("p1", DFS_MONTHS=30)])
    assert result.patients["time"][0] == pytest.approx(2.5)


def test_duplicated_patient_ids_stop_the_stage(tmp_path):
    patients = [vasudev_patient("dup"), vasudev_patient("dup")]
    with pytest.raises(SchemaError, match="duplicated"):
        _with_samples(tmp_path, patients)


def test_a_patient_without_a_sample_row_keeps_missing_predictors(tmp_path):
    patients = [vasudev_patient("kept"), vasudev_patient("no_sample")]
    samples = [vasudev_sample("kept")]
    result = _build(tmp_path, patients, samples)
    table = _by_id(result)
    assert table["no_sample"]["t_stage"] is None
    assert table["no_sample"]["leibovich_score_computed"] is None
    flow = dict(result.flow.iter_rows())
    assert flow["patients_without_sample_row"] == 1
    assert flow["analysis_set"] == 2


def test_missing_predictors_lists_every_absent_predictor(tmp_path):
    patients = [
        vasudev_patient("complete"),
        vasudev_patient("unsplit_t1"),
        vasudev_patient("no_node_no_necrosis"),
        vasudev_patient("no_size_no_grade"),
    ]
    samples = [
        vasudev_sample("complete", STAGE_N="NI"),
        # Unsplit "TI" is treated as a missing T stage.
        vasudev_sample("unsplit_t1", STAGE_T="TI", STAGE_N="NI"),
        # Missing node and necrosis score as negative and absent but are
        # still missing recorded values.
        vasudev_sample("no_node_no_necrosis", STAGE_N=None, NECROSIS=None),
        vasudev_sample("no_size_no_grade", STAGE_N="NI", TUMOR_SIZE=None, FFPE_GRADE=None),
    ]
    result = _build(tmp_path, patients, samples)
    table = _by_id(result)
    assert table["complete"]["missing_predictors"] == []
    assert table["unsplit_t1"]["missing_predictors"] == ["t_stage"]
    assert table["no_node_no_necrosis"]["missing_predictors"] == ["node", "necrosis"]
    assert table["no_size_no_grade"]["missing_predictors"] == ["size", "grade"]


def test_a_patient_missing_only_size_keeps_the_other_predictors(tmp_path):
    patients = [vasudev_patient("no_size"), vasudev_patient("anchor")]
    samples = [
        vasudev_sample(
            "no_size",
            STAGE_T="TIII",
            STAGE_N="NI",
            TUMOR_SIZE=None,
            FFPE_GRADE=3,
            NECROSIS="Present",
        ),
        vasudev_sample("anchor"),
    ]
    row = _by_id(_build(tmp_path, patients, samples))["no_size"]
    assert row["size_ge_10cm"] is None
    assert row["t_stage"] == "pT3a"
    assert row["node_positive"] is True
    assert row["grade"] == 3
    assert row["necrosis"] is True
    # The score needs all five predictors, so it stays missing.
    assert row["leibovich_score_computed"] is None
    assert row["missing_predictors"] == ["size"]


def test_missing_necrosis_is_missing_in_the_main_analysis(tmp_path):
    patients = [vasudev_patient("no_necrosis"), vasudev_patient("anchor")]
    samples = [vasudev_sample("no_necrosis", NECROSIS=None), vasudev_sample("anchor")]
    row = _by_id(_build(tmp_path, patients, samples))["no_necrosis"]
    assert row["necrosis"] is None
    assert row["leibovich_score_computed"] is None
    assert row["missing_predictors"] == ["node", "necrosis"]


def test_missing_nodes_stay_node_negative_with_the_other_predictors_kept(tmp_path):
    patients = [vasudev_patient("no_node"), vasudev_patient("anchor")]
    samples = [vasudev_sample("no_node", STAGE_N=None, TUMOR_SIZE=None), vasudev_sample("anchor")]
    row = _by_id(_build(tmp_path, patients, samples))["no_node"]
    assert row["node_positive"] is False
    assert row["t_stage"] == "pT1a"
    assert row["missing_predictors"] == ["node", "size"]


def test_flow_lists_every_exclusion_before_the_analysis_set(tmp_path):
    patients = [
        vasudev_patient("ok"),
        vasudev_patient("stage_iv", STAGE="IV"),
        vasudev_patient("no_followup", DFS_MONTHS=0, RCC_MONTHS=0),
        vasudev_patient("no_sex", SEX=None),
    ]
    result = _with_samples(tmp_path, patients)
    steps = result.flow["step"].to_list()
    n = dict(result.flow.iter_rows())
    excluded = [s for s in steps if s.startswith("excluded_")]
    assert n["excluded_nonpositive_followup"] == 1
    assert max(steps.index(s) for s in excluded) < steps.index("analysis_set")
    assert n["patient_rows"] - sum(n[s] for s in excluded) == n["analysis_set"]
    assert n["analysis_set"] == 1 == len(result.patients)
