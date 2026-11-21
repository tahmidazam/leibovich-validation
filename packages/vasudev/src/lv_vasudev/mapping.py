from typing import Any

from lv_datasets import S2_SECTIONS, SIZE_THRESHOLD_MM
from lv_datasets.mapping import (
    INCOMPLETE,
    PT3_BOUND,
    arrows,
    quoted,
    row,
    score_rule,
    sex_missing,
)

from lv_vasudev import raw
from lv_vasudev.harmonise import NECROSIS_PRESENT, NODE_POSITIVE, SEX, T_STAGE

PREDICTORS, DESIGN, TIMING, EVENT, SCORE = S2_SECTIONS


def mapping_rows(params: dict[str, Any], analysis: dict[str, Any]) -> list[dict[str, str]]:
    stage_t = raw.SAMPLE_CATEGORIES["STAGE_T"]
    unmapped = sorted(set(stage_t) - set(T_STAGE))
    if unmapped:
        t_missing = (
            f"{' and '.join(unmapped)} count{'s' if len(unmapped) == 1 else ''} "
            "as missing; the incomplete set is imputed"
        )
    else:
        t_missing = "every recorded stage maps; a missing value is imputed"
    patient = raw.PATIENT_CATEGORIES
    sample = raw.SAMPLE_CATEGORIES
    event_raw = "; ".join(f"{name} {quoted(patient[name])}" for name in ("RELAPSE", "DFS_STATUS"))
    death_raw = "; ".join(
        f"{name} {quoted(patient[name])}"
        for name in ("OS_STATUS", "RCC_STATUS", "RCC_RELATED_DEATH")
    )
    return [
        row(
            PREDICTORS,
            "t_stage",
            "STAGE_T",
            quoted(stage_t),
            arrows(T_STAGE) + PT3_BOUND,
            t_missing,
        ),
        row(
            PREDICTORS,
            "node_positive",
            "STAGE_N",
            quoted(sample["STAGE_N"]),
            f'"{NODE_POSITIVE}" counts as node-positive; every other value counts as '
            "node-negative because pNx scores as pN0 ",
            "missing counts as node-negative; the cohort records almost no nodal status",
        ),
        row(
            PREDICTORS,
            "size_ge_10cm",
            "TUMOR_SIZE",
            "millimetres",
            f"a diameter of {SIZE_THRESHOLD_MM} mm or more counts as 10 cm or greater",
            f"missing size {INCOMPLETE}",
        ),
        row(
            PREDICTORS,
            "grade",
            "FFPE_GRADE",
            "integer grade",
            "Fuhrman grade from the FFPE block, used directly",
            f"missing grade {INCOMPLETE}",
        ),
        row(
            PREDICTORS,
            "necrosis",
            "NECROSIS",
            quoted(sample["NECROSIS"]),
            f'"{NECROSIS_PRESENT}" counts as necrosis',
            "missing counts as absent",
        ),
        row(
            DESIGN,
            "patient_id",
            "PATIENT_ID",
            "text",
            "kept as text; joins the patient and sample tables",
            "missing or duplicated identifiers stop the build",
        ),
        row(
            DESIGN,
            "eligibility",
            "STAGE; STAGE_M",
            f"STAGE {quoted(patient['STAGE'])}; STAGE_M {quoted(sample['STAGE_M'])}",
            "overall stage IV, recorded MI and missing overall stage are excluded; "
            "missing STAGE_M at stage I to III stays eligible",
            "missing STAGE is excluded; missing STAGE_M stays eligible",
        ),
        row(DESIGN, "sex", "SEX", quoted(patient["SEX"]), arrows(SEX), sex_missing(analysis)),
        row(
            DESIGN,
            "age_at_surgery",
            "AGE",
            "years",
            "recorded age, used directly",
            "a missing age stops the build",
        ),
        row(
            DESIGN,
            "arm",
            "not recorded",
            "",
            "not a trial cohort; null",
            "always null",
        ),
        row(
            DESIGN,
            "nephrectomy_type",
            "NEPHRECTOMY_TYPE",
            quoted(patient["NEPHRECTOMY_TYPE"]),
            "lowercased; no analysis uses it",
            "stays null",
        ),
        row(
            TIMING,
            "operation_year",
            "not recorded",
            "",
            "null; the cohort records no surgery dates, so the country-matched WHO "
            "life-table sensitivity assigns every patient the mid-point year "
            f"{analysis['fallback_surgery_year']} from params.yaml",
            "always null",
        ),
        row(
            TIMING,
            "entry_time",
            "derived",
            "",
            "zero; surgery is the time origin with no delayed entry",
        ),
        row(
            TIMING,
            "time",
            "DFS_MONTHS; RCC_MONTHS",
            "months",
            "the DFS time in years, capped at the survival time; a missing DFS time "
            "falls back to the survival time",
            "both times missing stops the build",
        ),
        row(
            EVENT,
            "event of interest",
            "RELAPSE; DFS_STATUS; RCC_RELATED_DEATH; RCC_STATUS",
            event_raw,
            "relapse at the DFS time, a wider endpoint than distant metastasis; a "
            "missing RELAPSE takes the status from DFS_STATUS; an RCC death without a "
            "recorded relapse counts as relapse at death; a relapsed but disease-free "
            "record counts as an event and is flagged",
            "RELAPSE and DFS_STATUS both missing leaves the status underivable; the "
            "patient is censored unless a death is recorded",
        ),
        row(
            EVENT,
            "competing event",
            "OS_STATUS; RCC_STATUS; RCC_RELATED_DEATH; RCC_MONTHS",
            death_raw,
            "death without relapse at the survival time; a death of unknown cause "
            "counts as other-cause",
            "a recorded death with no survival time stops the build",
        ),
        row(
            SCORE,
            "leibovich_score_recorded",
            "LEIBOVICH_SCORE",
            "integer",
            score_rule(analysis),
            "missing scores are not compared",
        ),
    ]
