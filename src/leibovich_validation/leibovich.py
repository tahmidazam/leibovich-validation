"""The 2003 Leibovich model for distant metastasis after nephrectomy for clear cell RCC.

Coefficients come from Table 3 and points from Table 4 of Leibovich et al., Cancer 2003
(https://doi.org/10.1002/cncr.11234). The baseline cumulative hazard is not published. Stimpson
et al. (https://doi.org/10.1101/2025.05.22.25328132, Table 1) give unadjusted Leibovich risks at
1, 5 and 10 years for three tumour profiles, and the baseline is recovered from those.
"""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

TStage = Literal["pT1a", "pT1b", "pT2", "pT3a", "pT3b", "pT3c", "pT4"]

T_COEF: dict[str, float] = {
    "pT1a": 0.0,
    "pT1b": 1.243,
    "pT2": 1.648,
    "pT3a": 1.896,
    "pT3b": 2.014,
    "pT3c": 2.014,
    "pT4": 2.014,
}
NODE_POSITIVE_COEF = 0.924
SIZE_GE_10CM_COEF = 0.393
GRADE_COEF: dict[int, float] = {1: 0.0, 2: 0.0, 3: 0.640, 4: 1.313}
NECROSIS_COEF = 0.678

T_POINTS: dict[str, int] = {
    "pT1a": 0,
    "pT1b": 2,
    "pT2": 3,
    "pT3a": 4,
    "pT3b": 4,
    "pT3c": 4,
    "pT4": 4,
}
GRADE_POINTS: dict[int, int] = {1: 0, 2: 0, 3: 1, 4: 3}
NODE_POSITIVE_POINTS = 2
SIZE_GE_10CM_POINTS = 1
NECROSIS_POINTS = 1


@dataclass(frozen=True)
class Tumour:
    t_stage: TStage
    node_positive: bool
    size_ge_10cm: bool
    grade: int
    necrosis: bool


def _check_tumour(tumour: Tumour) -> None:
    if tumour.t_stage not in T_COEF:
        allowed = ", ".join(T_COEF)
        msg = f"unknown T stage {tumour.t_stage!r}; allowed values are {allowed}"
        raise ValueError(msg)
    if tumour.grade not in GRADE_COEF:
        allowed = ", ".join(str(grade) for grade in sorted(GRADE_COEF))
        msg = f"unknown grade {tumour.grade!r}; allowed values are {allowed}"
        raise ValueError(msg)


def linear_predictor(tumour: Tumour) -> float:
    _check_tumour(tumour)
    return (
        T_COEF[tumour.t_stage]
        + NODE_POSITIVE_COEF * tumour.node_positive
        + SIZE_GE_10CM_COEF * tumour.size_ge_10cm
        + GRADE_COEF[tumour.grade]
        + NECROSIS_COEF * tumour.necrosis
    )


def score(tumour: Tumour) -> int:
    _check_tumour(tumour)
    return (
        T_POINTS[tumour.t_stage]
        + NODE_POSITIVE_POINTS * tumour.node_positive
        + SIZE_GE_10CM_POINTS * tumour.size_ge_10cm
        + GRADE_POINTS[tumour.grade]
        + NECROSIS_POINTS * tumour.necrosis
    )


# The three tumour profiles in Stimpson et al. (section 2.3) and their unadjusted Leibovich risks
# (percent) at 1, 5 and 10 years from Table 1.
PREPRINT_PROFILES: dict[str, Tumour] = {
    "low": Tumour("pT1a", node_positive=False, size_ge_10cm=False, grade=1, necrosis=False),
    "intermediate": Tumour("pT2", node_positive=False, size_ge_10cm=False, grade=3, necrosis=False),
    "high": Tumour("pT3b", node_positive=True, size_ge_10cm=True, grade=3, necrosis=False),
}
PREPRINT_LEIBOVICH_RISK: dict[str, dict[int, float]] = {
    "low": {1: 1.17, 5: 2.98, 10: 4.33},
    "intermediate": {1: 10.99, 5: 25.79, 10: 35.39},
    "high": {1: 46.56, 5: 79.91, 10: 90.47},
}


def recover_baseline_cumulative_hazard() -> dict[int, float]:
    """Baseline cumulative hazard at 1, 5 and 10 years.

    Each published risk implies a baseline value, -log(1 - risk) / exp(lp). The three profiles
    disagree only through rounding, so their geometric mean is used.
    """
    baseline = {}
    for years in (1, 5, 10):
        logs = [
            math.log(-math.log(1 - PREPRINT_LEIBOVICH_RISK[name][years] / 100))
            - linear_predictor(tumour)
            for name, tumour in PREPRINT_PROFILES.items()
        ]
        baseline[years] = math.exp(sum(logs) / len(logs))
    return baseline


BASELINE_CUMULATIVE_HAZARD = recover_baseline_cumulative_hazard()

# The recovered baseline is known at these times only. Between knots the cumulative
# hazard is linear, which makes the hazard itself piecewise constant: the year-0-to-1
# average below year 1, then one constant rate over 1 to 5 and another over 5 to 10.
BASELINE_TIMES = (0.0, 1.0, 5.0, 10.0)
MAX_YEARS = BASELINE_TIMES[-1]


def baseline_cumulative_hazard(years: float) -> float:
    """Baseline cumulative hazard at `years` after surgery, piecewise linear between knots."""
    if not 0 <= years <= MAX_YEARS:
        msg = f"time must be inside the model's domain of 0 to {MAX_YEARS:g} years; got {years}"
        raise ValueError(msg)
    values = (0.0, *(BASELINE_CUMULATIVE_HAZARD[int(k)] for k in BASELINE_TIMES[1:]))
    return float(np.interp(years, BASELINE_TIMES, values))


def risk(tumour: Tumour, years: float) -> float:
    """Probability of distant metastasis by `years`, ignoring death from other causes.

    At times other than 1, 5 and 10 years the piecewise-constant baseline hazard applies.
    """
    hazard = baseline_cumulative_hazard(years) * math.exp(linear_predictor(tumour))
    return 1 - math.exp(-hazard)
