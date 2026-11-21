from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from leibovich_validation.config import REPO_ROOT

PARAMS = REPO_ROOT / "params.yaml"


@dataclass(frozen=True)
class ModelParams:
    life_table: str
    start_year: int
    round_other_cause_to_percent: bool


@dataclass(frozen=True)
class AnalysisParams:
    horizon_years: int
    secondary_horizon_years: int
    score_agreement_min: float
    seed: int
    bootstrap_resamples: int
    # Reduced resamples for the sensitivity stages; the principal
    # analysis keeps bootstrap_resamples.
    sensitivity_resamples: int
    imputations: int
    bootstrap_imputations: int
    missing_sex: str
    fallback_surgery_year: int
    # The follow-up fallback threshold: below this many patients at risk at
    # the primary horizon, the secondary horizon becomes the headline.
    min_at_risk: int
    # Age cut points for the age subgroups, from analysis.subgroups.
    age_cut_points: tuple[int, ...]
    # The sensitivity-analysis switches, from analysis.sensitivity.
    sensitivity: dict[str, bool]
    # Leibovich score bands for the risk groups, from analysis.subgroups.
    leibovich_groups: dict[str, tuple[int, int]]


@dataclass(frozen=True)
class ReportingParams:
    minimum_group_n: int
    descriptive_below_deaths: int
    minimum_at_risk: int
    behaviour_ages: tuple[int, ...]
    behaviour_time_step: float
    behaviour_sex: str
    calibration_bins: int
    calibration_rug_points: int
    calibration_bandwidth: float


def _read(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def load_model(path: Path = PARAMS) -> ModelParams:
    raw = _read(path)["model"]
    return ModelParams(
        life_table=str(raw["life_table"]),
        start_year=int(raw["start_year"]),
        round_other_cause_to_percent=bool(raw["round_other_cause_to_percent"]),
    )


def load_dataset(name: str, path: Path = PARAMS) -> dict[str, Any]:
    """The `datasets.<name>` parameter section an adapter reads."""
    return _read(path)["datasets"][name]


def load_analysis(profile: str = "main", path: Path = PARAMS) -> AnalysisParams:
    """The analysis settings, with a named profile overlaid on the main values."""
    raw = _read(path)
    values = dict(raw["analysis"])
    if (
        values.get("interval_method", "bootstrap") != "bootstrap"
        or values.get("interval_level", 0.95) != 0.95
    ):
        msg = "only percentile bootstrap intervals at level 0.95 are implemented"
        raise ValueError(msg)
    if profile != "main":
        try:
            values.update(raw["profiles"][profile])
        except KeyError:
            msg = f"unknown analysis profile: {profile!r}"
            raise KeyError(msg) from None
    return AnalysisParams(
        horizon_years=int(values["horizon_years"]),
        secondary_horizon_years=int(values["secondary_horizon_years"]),
        score_agreement_min=float(values["score_agreement_min"]),
        seed=int(values["seed"]),
        bootstrap_resamples=int(values["bootstrap_resamples"]),
        sensitivity_resamples=int(
            values.get("sensitivity_resamples", values["bootstrap_resamples"])
        ),
        imputations=int(values["imputations"]),
        bootstrap_imputations=int(values["bootstrap_imputations"]),
        missing_sex=str(values["missing_sex"]),
        fallback_surgery_year=int(values["fallback_surgery_year"]),
        min_at_risk=int(values["min_at_risk"]),
        age_cut_points=tuple(int(cut) for cut in values["subgroups"]["age_cut_points"]),
        sensitivity={
            str(name): bool(value) for name, value in (values.get("sensitivity") or {}).items()
        },
        leibovich_groups={
            str(name): (int(band[0]), int(band[1]))
            for name, band in values["subgroups"]["leibovich_groups"].items()
        },
    )


def load_reporting(path: Path = PARAMS) -> ReportingParams:
    """Reporting thresholds and the grid for the model-behaviour summary."""
    raw = _read(path)["reporting"]
    return ReportingParams(
        minimum_group_n=int(raw["minimum_group_n"]),
        descriptive_below_deaths=int(raw["descriptive_below_deaths"]),
        minimum_at_risk=int(raw["minimum_at_risk"]),
        behaviour_ages=tuple(int(age) for age in raw["behaviour_ages"]),
        behaviour_time_step=float(raw["behaviour_time_step"]),
        behaviour_sex=str(raw["behaviour_sex"]),
        calibration_bins=int(raw["calibration_bins"]),
        calibration_rug_points=int(raw["calibration_rug_points"]),
        calibration_bandwidth=float(raw["calibration_bandwidth"]),
    )


@dataclass(frozen=True)
class SensitivityParams:
    """The analysis.sensitivity switches for the five sensitivity analyses.

    The WHO life-table switch drives the life-table stage and the rest drive the
    definition stage. An absent switch is off.
    """

    pt3b_for_unsplit_pt3: bool
    complete_cases: bool
    country_life_table: bool
    surgery_clock: bool
    inconsistent_censoring: bool


def load_sensitivity(path: Path = PARAMS) -> SensitivityParams:
    """The sensitivity-analysis switches; keys the loader knows default to off."""
    raw = _read(path)["analysis"].get("sensitivity") or {}
    return SensitivityParams(
        **{field.name: bool(raw.get(field.name, False)) for field in fields(SensitivityParams)}
    )


@dataclass(frozen=True)
class AuditParams:
    """Settings for the blind audit checks in `audit`."""

    followup_horizons_years: list[float]


def load_audit(path: Path = PARAMS) -> AuditParams:
    raw = _read(path)["audit"]
    return AuditParams(
        followup_horizons_years=[float(h) for h in raw["followup_horizons_years"]],
    )
