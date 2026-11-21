import datetime as dt

import polars as pl

from lv_transorce import audit

PARAMS = {"operation_window": ["2007-07-01", "2013-04-30"], "randomisation_lag_days": [28, 92]}


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


def test_transorce_date_checks():
    outputs = audit.tables(_clinical_frame(), PARAMS, [3.0, 5.0])
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
    outputs = audit.tables(_clinical_frame(), PARAMS, [3.0, 5.0])
    row = outputs["transorce_score_agreement"].row(0)
    # T1: pT3a, node-, 55 mm, grade 3, necrosis -> 6, recorded 6: agree.
    # T2: pT1a, node-, <10, grade 1, no necrosis -> 0, recorded 0: agree.
    # T3: pT3a, node+, >10, grade 2, no necrosis -> 7, recorded 7: agree.
    # T4: no T stage -> not computed.
    assert row == (3, 3, 100.0)


def test_transorce_other_details_and_missing_sex():
    outputs = audit.tables(_clinical_frame(), PARAMS, [3.0, 5.0])
    assert outputs["transorce_other_details"].rows() == [("bowel", 1)]
    counts = dict(outputs["transorce_counts"].iter_rows())
    assert counts["missing_sex"] == 1
    assert counts["missing_operation_date"] == 1


