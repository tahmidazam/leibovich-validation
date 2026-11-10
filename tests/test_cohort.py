"""The TranSORCE cohort stage on a synthetic spreadsheet in the real format.

Each test builds a small workbook with the ccRCC sheet's real headers, reads it
through the raw reader and runs `harmonise_transorce`. There is one worked
example per cohort rule: missing-value codes and estimated operation dates,
age at surgery, every event and censoring rule, the "Other: Details" mapping,
the date-check exclusions and the score-agreement counts.
"""

import datetime as dt
from pathlib import Path

import polars as pl
import pytest
import yaml

from leibovich_validation import cohort, raw_transorce
from leibovich_validation.raw_vasudev import SchemaError
from tests.synthetic import transorce_patient, write_transorce_workbook

OPERATION_WINDOW = (dt.date(2007, 7, 1), dt.date(2013, 4, 30))
LAG_DAYS = (28, 92)

MAPPING = {
    "Brain": "distant_metastasis",
    "Breast": "second_primary",
    "Scar recurrence, left flank.": "other",
}


def _patient(**overrides) -> dict:
    """A baseline patient row; `identifier` picks the trial and patient ids."""
    identifier = str(overrides.pop("identifier", "1"))
    return transorce_patient(identifier, **overrides)


def _build(tmp_path: Path, rows: list[dict], min_agree: float = 0.0) -> cohort.CohortResult:
    path = tmp_path / "clinical.xlsx"
    write_transorce_workbook(path, rows)
    clinical = raw_transorce.read_clinical(path)
    return cohort.harmonise_transorce(clinical, MAPPING, OPERATION_WINDOW, LAG_DAYS, min_agree)


def _by_id(result: cohort.CohortResult) -> dict[str, dict]:
    return {row["patient_id"]: row for row in result.patients.iter_rows(named=True)}


def test_missing_operation_date_is_estimated_and_flagged(tmp_path):
    rows = [
        _patient(trial_identifier="001", patient_id=1),
        _patient(
            trial_identifier="002",
            patient_id=2,
            operation_date=".a",  # the reader decodes this to missing
            randomisation_date=dt.datetime(2008, 6, 1),
            dfs_date=dt.datetime(2015, 1, 1),
        ),
    ]
    result = _build(tmp_path, rows)
    patients = _by_id(result)
    # The cohort-median lag is 46 days, so the estimate is 2008-06-01 minus 46.
    est = patients["002"]
    assert est["estimated_operation_date"] is True
    assert est["entry_time"] == pytest.approx(46 / 365.25)
    assert est["operation_year"] == 2008
    assert patients["001"]["estimated_operation_date"] is False
    flow = dict(result.flow.iter_rows())
    assert flow["estimated_operation_date"] == 1
    assert flow["analysis_set"] == 2


def test_age_at_surgery_uses_the_randomisation_lag(tmp_path):
    result = _build(tmp_path, [_patient(age_at_randomisation=60.0)])
    (patient,) = result.patients.iter_rows(named=True)
    assert patient["age_at_surgery"] == pytest.approx(60.0 - 46 / 365.25)
    assert patient["entry_time"] == pytest.approx(46 / 365.25)


def test_event_rules_one_worked_example_each(tmp_path):
    cases = {
        # Structured distant site flags count as the event.
        "lung": {"dfs_status": 1, "lung_metastasis": "Yes"},
        "nodal": {"dfs_status": 1, "nodal_metastasis": "Yes"},
        # A distant free-text site counts; second primaries and other sites do not.
        "brain": {"dfs_status": 1, "other_metastasis": "Yes", "other_details": "Brain"},
        "breast": {"dfs_status": 1, "other_metastasis": "Yes", "other_details": "Breast"},
        "scar": {
            "dfs_status": 1,
            "other_metastasis": "Yes",
            "other_details": "Scar recurrence, left flank.",
        },
        # Contralateral kidney is a probable second primary, not the event.
        "contra": {"dfs_status": 1, "contralateral_kidney_metastasis": "Yes"},
        # A local-only recurrence censors the patient at that event.
        "local": {"dfs_status": 1, "local_recurrence": "Renal bed"},
        # A DFS event that is the death itself is the competing event.
        "death_dfs": {
            "dfs_status": 1,
            "dfs_date": dt.datetime(2012, 1, 1),
            "os_status": 1,
            "os_date": dt.datetime(2012, 1, 1),
        },
        # Death without a DFS event is the competing event, at the death date.
        "death": {"os_status": 1, "os_date": dt.datetime(2014, 3, 1)},
        "alive": {},
    }
    rows = [
        _patient(trial_identifier=key, patient_id=i + 1, **overrides)
        for i, (key, overrides) in enumerate(cases.items())
    ]
    result = _build(tmp_path, rows)
    patients = _by_id(result)
    for key in ("lung", "nodal", "brain"):
        assert patients[key]["event"] == "metastasis", key
    for key in ("breast", "scar", "contra", "local", "alive"):
        assert patients[key]["event"] == "censored", key
    for key in ("death_dfs", "death"):
        assert patients[key]["event"] == "competing", key
    # A censored local recurrence keeps its censoring time at the DFS event.
    expected = (dt.date(2015, 1, 1) - dt.date(2008, 1, 15)).days / 365.25
    assert patients["local"]["time"] == pytest.approx(expected)
    # The competing death is timed at death, not at last disease-free contact.
    expected_death = (dt.date(2014, 3, 1) - dt.date(2008, 1, 15)).days / 365.25
    assert patients["death"]["time"] == pytest.approx(expected_death)


def test_unmapped_other_details_string_stops_the_stage(tmp_path):
    row = _patient(dfs_status=1, other_metastasis="Yes", other_details="Never seen before")
    with pytest.raises(SchemaError, match="not in the approved mapping"):
        _build(tmp_path, [row])


def test_date_check_failures_are_excluded_and_counted(tmp_path):
    rows = [
        _patient(trial_identifier="ok", patient_id=1),
        # Operation before the SORCE window.
        _patient(
            trial_identifier="early",
            patient_id=2,
            operation_date=dt.datetime(2006, 1, 10),
            randomisation_date=dt.datetime(2006, 2, 25),
        ),
        # Randomisation lag of 152 days, outside 28-92.
        _patient(
            trial_identifier="late",
            patient_id=3,
            operation_date=dt.datetime(2008, 1, 1),
            randomisation_date=dt.datetime(2008, 6, 1),
        ),
    ]
    result = _build(tmp_path, rows)
    assert _by_id(result).keys() == {"ok"}
    flow = dict(result.flow.iter_rows())
    assert flow["excluded_operation_outside_window"] == 1
    assert flow["excluded_randomisation_lag_outside_range"] == 1
    assert flow["analysis_set"] == 1


def test_missing_sex_is_excluded_and_counted(tmp_path):
    rows = [
        _patient(trial_identifier="ok", patient_id=1),
        _patient(trial_identifier="nosex", patient_id=2, sex=None),
    ]
    result = _build(tmp_path, rows)
    assert _by_id(result).keys() == {"ok"}
    assert dict(result.flow.iter_rows())["excluded_missing_sex"] == 1


def test_missing_predictors_lists_every_absent_predictor(tmp_path):
    rows = [
        _patient(trial_identifier="complete", patient_id=1),
        # Missing node and necrosis score as negative and absent but are
        # still missing recorded values.
        _patient(trial_identifier="node_necrosis", patient_id=2, node_status=None, necrosis=None),
        # Both size fields absent is a missing size; the category alone is
        # enough when the diameter is missing.
        _patient(trial_identifier="no_size", patient_id=3, max_diameter_mm=None),
        _patient(
            trial_identifier="no_size_at_all",
            patient_id=4,
            max_diameter_mm=None,
            tumour_size_category=None,
        ),
        _patient(trial_identifier="no_grade", patient_id=5, isup_grade=None),
    ]
    result = _build(tmp_path, rows)
    table = _by_id(result)
    assert table["complete"]["missing_predictors"] == []
    assert table["node_necrosis"]["missing_predictors"] == ["node", "necrosis"]
    assert table["no_size"]["missing_predictors"] == []
    assert table["no_size_at_all"]["missing_predictors"] == ["size"]
    assert table["no_grade"]["missing_predictors"] == ["grade"]


def test_schema_validation_rejects_bad_tables(tmp_path):
    result = _build(tmp_path, [_patient()])
    bad_event = result.patients.with_columns(pl.lit("relapse").alias("event"))
    with pytest.raises(SchemaError, match="event"):
        cohort.validate_patient_table(bad_event)
    bad_time = result.patients.with_columns((pl.col("time") - 100).alias("entry_time"))
    with pytest.raises(SchemaError, match="entry_time"):
        cohort.validate_patient_table(bad_time)
    with pytest.raises(SchemaError, match="missing columns"):
        cohort.validate_patient_table(result.patients.drop("patient_id"))


def test_score_agreement_is_counted_by_rule(tmp_path):
    # The recorded score counts size by the ">10" category; the measured
    # 95 mm diameter does not reach the threshold, so the category rule
    # explains the disagreement.
    disagreeing = _patient(
        trial_identifier="size",
        patient_id=1,
        t_stage="pT1a",
        isup_grade=1,
        tumour_size_category=">10",
        max_diameter_mm=95.0,
        leibovich_score=1,
    )
    result = _build(tmp_path, [_patient(), disagreeing])
    agreement = dict(result.score_agreement.iter_rows())
    assert agreement["total_compared"] == 2
    assert agreement["agree"] == 1
    assert agreement["disagrees_as_size_threshold_from_recorded_category"] == 1
    assert agreement["disagrees_unexplained"] == 0


def test_score_agreement_below_the_threshold_stops_the_stage(tmp_path):
    disagreeing = _patient(
        t_stage="pT1a",
        isup_grade=1,
        max_diameter_mm=95.0,
        tumour_size_category=">10",
        leibovich_score=1,
    )
    with pytest.raises(SchemaError, match="stop threshold"):
        _build(tmp_path, [disagreeing], min_agree=0.95)


def test_other_details_mapping_loader(tmp_path):
    path = tmp_path / "mapping.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "mapping": {
                    "Brain": "distant_metastasis",
                    "Left Adrenal": {"class": "distant_metastasis", "laterality": True},
                    "Scar": {"class": "other", "flagged": True},
                }
            }
        )
    )
    classes = cohort.load_other_details_mapping(path)
    assert classes == {
        "Brain": "distant_metastasis",
        "Left Adrenal": "distant_metastasis",
        "Scar": "other",
    }


def test_other_details_mapping_loader_rejects_unknown_class(tmp_path):
    path = tmp_path / "mapping.yaml"
    path.write_text(yaml.safe_dump({"mapping": {"Brain": "distant"}}))
    with pytest.raises(SchemaError, match="unknown class"):
        cohort.load_other_details_mapping(path)


def test_real_mapping_file_covers_every_audit_string():
    # The approved mapping covers the audit's distinct strings; loading it
    # exercises the dict and plain-string forms together.
    root = Path(cohort.__file__).resolve().parents[2]
    classes = cohort.load_other_details_mapping(
        root / "docs" / "transorce-other-details-mapping.yaml"
    )
    audit_file = root / "results" / "audit" / "transorce_other_details.csv"
    if audit_file.is_file():
        strings = pl.read_csv(audit_file)["other_details"].to_list()
        assert set(strings) <= set(classes)


def test_reorigin_at_randomisation_moves_time_zero_and_keeps_age_at_surgery():
    # Randomised 0.3 years after surgery, event 5.2 years after surgery: from
    # randomisation the event is at 4.9 years. The old entry is the prediction
    # offset, and age stays the age at surgery.
    patients = pl.DataFrame(
        {"age_at_surgery": [54.7, 61.2], "entry_time": [0.3, 0.0], "time": [5.2, 3.0]}
    )
    out = cohort.reorigin_at_randomisation(patients)
    assert out["time"].to_list() == pytest.approx([4.9, 3.0])
    assert out["entry_time"].to_list() == [0.0, 0.0]
    assert out["prediction_offset"].to_list() == [0.3, 0.0]
    assert out["age_at_surgery"].to_list() == [54.7, 61.2]


def test_reorigin_at_randomisation_rejects_exit_before_entry():
    patients = pl.DataFrame({"age_at_surgery": [60.0], "entry_time": [0.3], "time": [0.3]})
    with pytest.raises(ValueError, match="after entry"):
        cohort.reorigin_at_randomisation(patients)
