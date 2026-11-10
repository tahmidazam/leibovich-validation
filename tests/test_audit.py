import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from leibovich_validation import audit


def _patient_frame() -> pl.DataFrame:
    # Six invented patients, one row each, exercising every event rule.
    return pl.DataFrame(
        {
            "PATIENT_ID": ["K1", "K2", "K3", "K4", "K5", "K6"],
            "COHORT_CATEGORY": ["C1"] * 6,
            "COHORT": ["Discovery Cohort"] * 6,
            "AGE": [60, 70, 55, 66, 80, 50],
            "SEX": ["Male", "Female", "Male", "Male", "Female", "Male"],
            "BMI": [25.0] * 6,
            "BMI_CATEGORIES": ["25-29.9"] * 6,
            "TOBACCO_STATUS": ["Never Smoker"] * 6,
            "COUNTRY": ["Czech Republic"] * 5 + ["Russia"],
            "STAGE": ["I", "III", "III", "III", "II", "IV"],
            "LEIBOVICH_SCORE": [0, 9, 7, 3, 3, None],
            "LEIBOVICH_RISK_GROUP": ["Low", "High", "High", "Intermediate", None, None],
            "RHABDOID_OR_SARCOMATOID": ["Not Present"] * 6,
            "LATERALITY": ["Left Kidney"] * 6,
            "NEPHRECTOMY_TYPE": ["Partial", "Radical", "Radical", "Radical", "Radical", "Radical"],
            "RELAPSE": ["No", "Yes", "Yes", "No", "No", None],
            "OS_STATUS": ["0:LIVING"] * 3 + ["1:DECEASED"] * 2 + ["0:LIVING"],
            "RCC_RELATED_DEATH": [None, None, None, "Yes", "No", None],
            "DFS_STATUS": [
                "0:DiseaseFree",
                "1:Recurred/Progressed",
                "0:DiseaseFree",
                "1:Recurred/Progressed",
                "1:Recurred/Progressed",
                "0:DiseaseFree",
            ],
            "DFS_YEARS": [6.0, 2.5, 8.0, 10.0 / 3, 25.0 / 6, 1.0],
            "RCC_STATUS": ["0:LIVING"] * 3 + ["1:DECEASED", "0:LIVING", "0:LIVING"],
            "RCC_YEARS": [6.0, 5.0, 5.0, 10.0 / 3, 55.0 / 12, 1.0],
        }
    )


def _sample_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "PATIENT_ID": ["K1", "K2", "K3", "K4", "K5", "K6"],
            "SAMPLE_ID": ["K1", "K2", "K3", "K4", "K5", "K6"],
            "TUMOR_SIZE": [40.0, 120.0, 30.0, 60.0, 50.0, None],
            "NECROSIS": ["Absent", "Present", "Absent", "Absent", "Absent", None],
            "FFPE_GRADE": [1, 3, 4, 2, 2, None],
            "STAGE_T": ["TIa", "TIII", "TIb", "TI", "TII", "TIV"],
            "STAGE_T_CATEGORY": ["TI", "TIII", "TI", "TI", "TII", "TIV"],
            "STAGE_N": [None, "NI", "NI", None, None, None],
            "STAGE_M": [None] * 5 + ["MI"],
            "ONCOTREE_CODE": ["CCRCC"] * 6,
            "SOMATIC_STATUS": ["Matched"] * 6,
            "VHL_MUTATION": ["Mutated"] * 6,
            "GENOMIC_GROUP": ["VHL+0"] * 6,
            "TMB_NONSYNONYMOUS": [0.0] * 6,
        }
    )


def _clinical_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "image_tag": [None] * 4,
            "icaird_ref": ["iCAIRD-1"] * 4,
            "trial_identifier": ["003001"] * 4,
            "patient_id": [1, 2, 3, 4],
            "arm": ["A", "B", "C", "A"],
            "age_at_randomisation": [60.0, 65.0, 70.0, 55.0],
            "diagnosis_to_randomisation": [40.0] * 4,
            "randomisation_date": [dt.date(2008, 2, 20)] * 4,
            "operation_date": [
                dt.date(2008, 1, 15),
                None,
                dt.date(2006, 1, 1),
                dt.date(2008, 3, 1),
            ],
            "sex": ["M", "F", "M", None],
            "who_ps": ["0"] * 4,
            "histology": ["Conventional/ Clear Cell"] * 4,
            "t_stage": ["pT3a - 4", "pT1a", "pT3a - 4", None],
            "node_status": ["pNx/ pN0", "pNx/ pN0", "pN1/ pN2", "pNx/ pN0"],
            "tumour_size_category": ["<10", "<10", ">10", "<10"],
            "isup_grade": [3, 1, 2, 4],
            "necrosis": ["Yes", "No", "No", None],
            "leibovich_score": [6, 0, 7, None],
            "leibovich_risk_group": ["0"] * 4,
            "max_diameter_mm": [55.0, 30.0, 110.0, 20.0],
            "dfs_status": [0, 1, 0, 1],
            "dfs_date": [dt.date(2013, 1, 15), dt.date(2009, 1, 15), dt.date(2010, 1, 1), None],
            "os_status": [0, 0, 0, 1],
            "os_date": [
                dt.date(2013, 1, 15),
                dt.date(2009, 1, 15),
                dt.date(2010, 1, 1),
                dt.date(2008, 6, 1),
            ],
            "local_recurrence": ["No"] * 4,
            "contralateral_kidney_metastasis": ["No"] * 4,
            "nodal_metastasis": ["No"] * 4,
            "lung_metastasis": ["No", "Yes", "No", "No"],
            "bone_metastasis": ["No"] * 4,
            "liver_metastasis": ["No"] * 4,
            "other_metastasis": ["No", "No", "No", "Yes"],
            "other_details": [None, None, None, "bowel"],
            "method_of_assessment": ["CT Scan"] * 4,
            "other_method_details": [None] * 4,
        }
    )


def test_reverse_km_median_worked_example():
    times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    censored = [True] * 8
    # Survival of censoring: 7/8, 6/7*7/8=0.75, then 0.625, then 0.5 at t=4.
    assert audit.reverse_km_median(times, censored) == 4.0


def test_vasudev_event_rule_counts():
    outputs = audit.vasudev_outputs(_patient_frame(), _sample_frame(), [3.0, 5.0])
    rules = dict(outputs["vasudev_event_rules"].iter_rows())
    assert rules["relapse_yes"] == 2
    assert rules["relapse_at_dfs_time"] == 1
    assert rules["rcc_death_without_relapse"] == 1
    assert rules["relapsed_but_coded_disease_free"] == 1
    assert rules["dfs_time_later_than_survival"] == 1  # K3: DFS 8.0 > RCC 5.0
    assert rules["deceased"] == 2
    assert rules["rcc_deceased"] == 1
    assert rules["other_cause_deceased"] == 1
    assert rules["overall_stage_iv"] == 1


def test_vasudev_unsplit_stage_counts():
    outputs = audit.vasudev_outputs(_patient_frame(), _sample_frame(), [3.0, 5.0])
    unsplit = dict(outputs["vasudev_unsplit_stage"].select("stage_t", "n").iter_rows())
    assert unsplit == {"TI": 1, "TIII": 1}


def test_vasudev_score_agreement_worked_example():
    outputs = audit.vasudev_outputs(_patient_frame(), _sample_frame(), [3.0, 5.0])
    row = outputs["vasudev_score_agreement"].row(0)
    # K1: pT1a, node-, 40 mm, grade 1, no necrosis -> 0, recorded 0: agree.
    # K2: pT3a, node+, 120 mm, grade 3, necrosis -> 9, recorded 9: agree.
    # K3: pT1b, node+, 30 mm, grade 4 -> 7, recorded 7: agree.
    # K4: unsplit TI -> not computed; K5: computed 3 vs recorded 3 -> agree.
    # K6: nothing recorded. Four compared (K1, K2, K3, K5), all agree.
    assert row == (4, 4, 100.0)


def test_vasudev_followup():
    outputs = audit.vasudev_outputs(_patient_frame(), _sample_frame(), [3.0, 5.0])
    followup = dict(outputs["vasudev_followup"].select("measure", "value").iter_rows())
    # Times 1.0 (cens), 2.5, 10/3, 25/6, 6.0 (cens), 8.0 (cens): the censoring curve
    # reaches 0.5 at 6.0.
    assert followup["reverse_km_median_years"] == 6.0
    assert followup["at_risk_at_3y"] == 4
    assert followup["at_risk_at_5y"] == 2


def test_transorce_date_checks():
    outputs = audit.transorce_outputs(
        _clinical_frame(),
        (dt.date(2007, 7, 1), dt.date(2013, 4, 30)),
        (28, 92),
        [3.0, 5.0],
    )
    checks = {
        r[0]: (r[2], r[3])  # check -> (n_pass, n_fail)
        for r in outputs["transorce_date_checks"].iter_rows()
    }
    # T1 (2008-01-15) and T4 (2008-03-01) are in the window; T3 (2006-01-01) is not;
    # T2 has no operation date.
    assert checks["operation_date_present"] == (3, 1)
    assert checks["operation_in_window"] == (2, 1)
    # Lags among the three with both dates: T1 = 36 days, in range; T3 is over two
    # years and T4 is negative. One pass, two fail.
    assert checks["randomisation_lag_in_range"] == (1, 2)


def test_transorce_score_agreement_worked_example():
    outputs = audit.transorce_outputs(
        _clinical_frame(),
        (dt.date(2007, 7, 1), dt.date(2013, 4, 30)),
        (28, 92),
        [3.0, 5.0],
    )
    row = outputs["transorce_score_agreement"].row(0)
    # T1: pT3a, node-, 55 mm, grade 3, necrosis -> 6, recorded 6: agree.
    # T2: pT1a, node-, <10, grade 1, no necrosis -> 0, recorded 0: agree.
    # T3: pT3a, node+, >10, grade 2, no necrosis -> 7, recorded 7: agree.
    # T4: no T stage -> not computed.
    assert row == (3, 3, 100.0)


def test_transorce_other_details_and_missing_sex():
    outputs = audit.transorce_outputs(
        _clinical_frame(),
        (dt.date(2007, 7, 1), dt.date(2013, 4, 30)),
        (28, 92),
        [3.0, 5.0],
    )
    assert outputs["transorce_other_details"].rows() == [("bowel", 1)]
    counts = dict(outputs["transorce_counts"].iter_rows())
    assert counts["missing_sex"] == 1
    assert counts["missing_operation_date"] == 1


def test_write_outputs_has_no_patient_rows(tmp_path: Path):
    outputs = audit.vasudev_outputs(_patient_frame(), _sample_frame(), [3.0, 5.0])
    outputs.update(
        audit.transorce_outputs(
            _clinical_frame(),
            (dt.date(2007, 7, 1), dt.date(2013, 4, 30)),
            (28, 92),
            [3.0, 5.0],
        )
    )
    written = audit.write_outputs(outputs, tmp_path)
    n_patients = 6
    for path in written:
        frame = pl.read_csv(path)
        assert not ({"patient_id", "PATIENT_ID"} & set(frame.columns))
        assert frame.height != n_patients


def test_assert_aggregate_rejects_patient_rows():
    with pytest.raises(ValueError, match="identifiers"):
        audit.assert_aggregate(pl.DataFrame({"patient_id": ["T1"]}), "bad")
