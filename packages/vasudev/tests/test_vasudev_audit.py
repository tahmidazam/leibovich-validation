import polars as pl

from lv_vasudev import audit


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


def test_vasudev_event_rule_counts():
    outputs = audit.tables(_patient_frame(), _sample_frame(), [3.0, 5.0])
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
    outputs = audit.tables(_patient_frame(), _sample_frame(), [3.0, 5.0])
    unsplit = dict(outputs["vasudev_unsplit_stage"].select("stage_t", "n").iter_rows())
    assert unsplit == {"TI": 1, "TIII": 1}


def test_vasudev_score_agreement_worked_example():
    outputs = audit.tables(_patient_frame(), _sample_frame(), [3.0, 5.0])
    row = outputs["vasudev_score_agreement"].row(0)
    # K1: pT1a, node-, 40 mm, grade 1, no necrosis -> 0, recorded 0: agree.
    # K2: pT3a, node+, 120 mm, grade 3, necrosis -> 9, recorded 9: agree.
    # K3: pT1b, node+, 30 mm, grade 4 -> 7, recorded 7: agree.
    # K4: unsplit TI -> not computed; K5: computed 3 vs recorded 3 -> agree.
    # K6: nothing recorded. Four compared (K1, K2, K3, K5), all agree.
    assert row == (4, 4, 100.0)


def test_vasudev_followup():
    outputs = audit.tables(_patient_frame(), _sample_frame(), [3.0, 5.0])
    followup = dict(outputs["vasudev_followup"].select("measure", "value").iter_rows())
    # Times 1.0 (cens), 2.5, 10/3, 25/6, 6.0 (cens), 8.0 (cens): the censoring curve
    # reaches 0.5 at 6.0.
    assert followup["reverse_km_median_years"] == 6.0
    assert followup["at_risk_at_3y"] == 4
    assert followup["at_risk_at_5y"] == 2


