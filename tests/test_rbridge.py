"""The R bridge is proven against the STRATOS worked example and synthetic data.

Published reference values come from the rendered Prediction_CSC.md at commit
ba1d9e8e68ed4f031e55dab473ce9084694bdec9 of the STRATOS competing-risks
validation companion repository
(https://github.com/survival-lumc/ValidationCompRisks; van Geloven et al., BMJ 2022,
doi:10.1136/bmj-2021-069249); each assertion names its source section.
"""

import math
import re
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from rich.console import Console

from leibovich_validation import bootstrap, rbridge
from leibovich_validation.config import REPO_ROOT, get_settings


@pytest.fixture(scope="session")
def stratos(r_available):
    """The STRATOS development and validation frames, or a skip."""
    try:
        data_dir = get_settings().data_dir
    except Exception:
        pytest.skip("LP_DATA_DIR is not set")
    paths = [data_dir / "stratos" / name for name in ("rdata.rds", "vdata.rds")]
    if any(not path.is_file() for path in paths):
        pytest.skip("run scripts/download.sh stratos to fetch the example data")
    return rbridge.read_rds(paths[0]), rbridge.read_rds(paths[1])


def _events(status_num: pl.Series) -> pl.Series:
    return pl.Series("event", [rbridge.EVENT_NAMES[int(code)] for code in status_num])


def test_reproduces_stratos_published_measures(stratos):
    rdata, vdata = stratos
    horizon = 5.0
    # CSC on rdata then predictRisk on vdata, as in Prediction_CSC_minimal.R.
    risk = rbridge.predict_csc(
        rdata,
        vdata,
        horizon,
        formula="Hist(time, status_num) ~ age + size + ncat + hr_status",
    )
    assert risk.len() == vdata.height
    outcomes = rbridge.Outcomes(vdata["time"], _events(vdata["status_num"]))

    # Section 1.1 table, "5-year", Validation data: observed incidence 0.10.
    aj = rbridge.observed_incidence(outcomes, horizon)
    assert aj.estimate == pytest.approx(0.10, abs=0.005)

    # Section 2.1.3 table: O/E 0.81 (0.67-0.97).
    oe = rbridge.observed_expected(outcomes, risk, horizon)
    assert oe.estimate == pytest.approx(0.81, abs=0.005)
    assert oe.lower == pytest.approx(0.67, abs=0.005)
    assert oe.upper == pytest.approx(0.97, abs=0.005)

    # Section 2.1.4 table: calibration intercept -0.15 (-0.36, 0.05), slope 1.22
    # (0.84, 1.60). The geese pseudo-observation estimates drift a few hundredths
    # with the package versions (the rendered session used geepack 1.3.12 on R
    # 4.3.3; ours is 1.3.13 on R 4.6.1), so the check is that each estimate falls
    # inside the published interval rather than matching the rounded point value.
    cal = rbridge.calibration(outcomes, risk, horizon, bandwidth=0.05)
    assert -0.36 <= cal.intercept <= 0.05
    assert 0.84 <= cal.slope <= 1.60
    assert cal.curve.columns == ["predicted", "observed"]
    assert cal.curve.height == vdata.height

    # Section 2.1.1.2 table: ICI 0.0308 (nearest-neighbour smoother, bandwidth 0.05).
    assert cal.ici == pytest.approx(0.0308, abs=0.0005)

    # Section 2.2.1 table: AUC 0.71 (0.66-0.77). Section 2.3 table: Brier 0.09
    # (0.07-0.10), IPA 0.06 (0.00-0.08).
    scores = rbridge.performance_scores(outcomes, {"csh_validation": risk}, horizon)

    def row(metric):
        return scores.filter(
            (pl.col("model") == "csh_validation") & (pl.col("metric") == metric)
        ).row(0, named=True)

    auc, brier, ipa = row("auc"), row("brier"), row("ipa")
    assert auc["estimate"] == pytest.approx(0.71, abs=0.005)
    assert auc["lower"] == pytest.approx(0.66, abs=0.005)
    assert auc["upper"] == pytest.approx(0.77, abs=0.005)
    assert brier["estimate"] == pytest.approx(0.09, abs=0.005)
    assert brier["lower"] == pytest.approx(0.07, abs=0.005)
    assert brier["upper"] == pytest.approx(0.10, abs=0.005)
    assert ipa["estimate"] == pytest.approx(0.06, abs=0.005)


def _simulated_cohort(n: int, seed: int) -> pl.DataFrame:
    """A cohort drawn from a known competing-risks model.

    Constant hazards for both causes, with the cause-1 hazard scaled by a
    per-patient log-normal effect and uniform censoring. The true cause-1
    cumulative incidence at horizon h is h1/(h1+h2) * (1 - exp(-(h1+h2) h)),
    so feeding it back as the predicted risk should calibrate perfectly.
    """
    rng = np.random.default_rng(seed)
    h1 = 0.10 * np.exp(rng.normal(0.0, 0.8, n))
    h2 = 0.15
    t1 = rng.exponential(1.0 / h1)
    t2 = rng.exponential(1.0 / h2, n)
    censored_at = rng.uniform(0.5, 8.0, n)
    horizon = 3.0
    risk = h1 / (h1 + h2) * (1 - np.exp(-(h1 + h2) * horizon))
    return pl.DataFrame(
        {
            "time": np.minimum(np.minimum(t1, t2), censored_at),
            "event": [
                rbridge.EVENT_NAMES[int(c)]
                for c in np.where(
                    (t1 <= t2) & (t1 <= censored_at),
                    1,
                    np.where((t2 < t1) & (t2 <= censored_at), 2, 0),
                )
            ],
            "risk": risk,
        }
    )


def _outcomes(frame: pl.DataFrame) -> rbridge.Outcomes:
    return rbridge.Outcomes(frame["time"], frame["event"])


def test_known_model_gives_unit_oe_and_calibration_slope(r_available):
    frame = _simulated_cohort(n=4000, seed=2024)
    oe = rbridge.observed_expected(_outcomes(frame), frame["risk"], 3.0)
    assert 0.9 < oe.estimate < 1.1
    cal = rbridge.calibration(_outcomes(frame), frame["risk"], 3.0, bandwidth=0.05)
    assert 0.85 < cal.slope < 1.15
    assert abs(cal.intercept) < 0.15


def test_ici_is_near_zero_for_a_known_model_and_matches_a_known_offset(r_available):
    frame = _simulated_cohort(n=4000, seed=2024)
    outcomes = _outcomes(frame)
    true_risk = frame["risk"]
    offset = 0.1
    cals = rbridge.calibration(
        outcomes,
        {"true": true_risk, "offset": true_risk + offset},
        3.0,
        bandwidth=0.05,
    )
    assert cals["true"].ici < 0.02
    # The offset copy is wrong by exactly 0.1 everywhere, so its ICI is 0.1
    # up to the smoothing error of the true curve.
    assert cals["offset"].ici == pytest.approx(offset, abs=0.02)


def test_smooth_calibration_matches_a_hand_computed_example(r_available):
    # Within 0.05 of 0.1 lie two patients (pseudo-values 0 and 1, mean 0.5); 0.2
    # and 0.9 stand alone. ICI = (0.4 + 0.4 + 0.3 + 0.1) / 4 = 0.3.
    curve = pl.DataFrame({"predicted": [0.1, 0.1, 0.2, 0.9], "observed": [0.0, 1.0, 0.5, 1.0]})
    full = rbridge.smooth_calibration(curve, 0.05, 0)
    assert full.ici == pytest.approx(0.3)
    assert full.grid["predicted"].to_list() == pytest.approx([0.1, 0.2, 0.9])
    assert full.grid["estimate"].to_list() == pytest.approx([0.5, 0.5, 1.0])
    assert full.grid["n"].to_list() == pytest.approx([2, 1, 1])
    # With two patients required the curve stops at 0.1; the ICI keeps everyone.
    truncated = rbridge.smooth_calibration(curve, 0.05, 2)
    assert truncated.grid["predicted"].to_list() == pytest.approx([0.1])
    assert truncated.ici == pytest.approx(0.3)


def test_smooth_calibration_accepts_risks_equal_to_15_significant_digits(r_available):
    # Risks pooled over imputations can differ only past the 15th significant
    # digit. R's factor() reads them as one level and unique() as two, which
    # breaks prodlim::meanNeighbors. They must smooth as one tied risk.
    near = float(np.nextafter(0.1, 1.0))
    curve = pl.DataFrame({"predicted": [0.1, near, 0.2, 0.9], "observed": [0.0, 1.0, 0.5, 1.0]})
    smoothed = rbridge.smooth_calibration(curve, 0.05, 0)
    assert smoothed.ici == pytest.approx(0.3)
    assert smoothed.grid["predicted"].to_list() == pytest.approx([0.1, 0.2, 0.9])


def test_no_events_before_horizon_returns_zero_incidence(r_available):
    # The metastasis events all occur after the 5-year horizon.
    frame = pl.DataFrame(
        {
            "time": [6.0, 7.0, 8.0, 2.0, 3.0, 6.5],
            "event": [
                "metastasis",
                "metastasis",
                "censored",
                "censored",
                "competing",
                "metastasis",
            ],
            "risk": [0.2, 0.3, 0.1, 0.2, 0.4, 0.3],
        }
    )
    aj = rbridge.observed_incidence(_outcomes(frame), 5.0)
    assert (aj.estimate, aj.se) == (0.0, 0.0)
    assert rbridge.observed_expected(_outcomes(frame), frame["risk"], 5.0) == rbridge.OERatio(
        0.0, 0.0, math.inf
    )
    with pytest.raises(rbridge.EstimationError, match="no 'metastasis' events"):
        rbridge.calibration(_outcomes(frame), frame["risk"], 5.0, bandwidth=0.05)


def test_all_censored_before_horizon(r_available):
    frame = pl.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "event": ["censored"] * 4,
            "risk": [0.1, 0.2, 0.3, 0.4],
        }
    )
    aj = rbridge.observed_incidence(_outcomes(frame), 5.0)
    assert (aj.estimate, aj.se) == (0.0, 0.0)
    oe = rbridge.observed_expected(_outcomes(frame), frame["risk"], 5.0)
    assert oe.estimate == 0.0
    with pytest.raises(rbridge.EstimationError, match="no 'metastasis' events"):
        rbridge.performance_scores(_outcomes(frame), {"m": frame["risk"]}, 5.0)


def test_no_competing_events(r_available):
    # Metastasis and censoring only: the competing state is empty.
    rng = np.random.default_rng(7)
    n = 1500
    h1 = 0.20 * np.exp(rng.normal(0.0, 0.5, n))
    t1 = rng.exponential(1.0 / h1)
    censored_at = rng.uniform(1.0, 6.0, n)
    frame = pl.DataFrame(
        {
            "time": np.minimum(t1, censored_at),
            "event": [rbridge.EVENT_NAMES[int(c)] for c in np.where(t1 <= censored_at, 1, 0)],
            "risk": 1 - np.exp(-h1 * 3.0),
        }
    )
    aj = rbridge.observed_incidence(_outcomes(frame), 3.0)
    assert 0.0 < aj.estimate < 1.0
    oe = rbridge.observed_expected(_outcomes(frame), frame["risk"], 3.0)
    assert math.isfinite(oe.estimate) and oe.estimate > 0
    # Score may or may not accept a cohort with an empty competing state; a
    # finite estimate and a clear EstimationError both satisfy the contract.
    try:
        cal = rbridge.calibration(_outcomes(frame), frame["risk"], 3.0, bandwidth=0.05)
    except rbridge.EstimationError:
        pass
    else:
        assert math.isfinite(cal.slope)


def test_bootstrap_is_reproducible_from_the_seed():
    frame = pl.DataFrame({"risk": np.linspace(0.01, 0.9, 200)})

    def statistic(df):
        return float(df["risk"].mean())

    first = bootstrap.bootstrap_interval(statistic, frame, resamples=50, seed=99)
    again = bootstrap.bootstrap_interval(statistic, frame, resamples=50, seed=99)
    other = bootstrap.bootstrap_interval(statistic, frame, resamples=50, seed=100)
    assert first == again
    assert first != other
    assert first.resamples == 50
    assert first.skipped == 0
    assert first.lower <= first.upper


def test_bootstrap_shows_a_progress_bar():
    frame = pl.DataFrame({"risk": np.linspace(0.01, 0.9, 50)})
    console = Console(record=True, width=60)
    bootstrap.bootstrap_interval(
        lambda df: float(np.asarray(df["risk"]).mean()),
        frame,
        resamples=20,
        seed=1,
        console=console,
    )
    assert "20/20" in console.export_text()


def test_bootstrap_skips_resamples_that_cannot_be_estimated():
    # The statistic fails whenever patient 0 is absent from the resample.
    frame = pl.DataFrame({"id": range(5), "risk": np.linspace(0.1, 0.5, 5)})

    def statistic(df):
        if 0 not in df["id"].to_list():
            raise rbridge.EstimationError("no events in this resample")
        return float(df["risk"].mean())

    result = bootstrap.bootstrap_interval(statistic, frame, resamples=50, seed=1)
    assert result.resamples == 50
    assert 0 < result.skipped < 50


def test_bootstrap_raises_when_every_resample_fails():
    def statistic(df):
        raise rbridge.EstimationError("no events")

    frame = pl.DataFrame({"risk": [0.1, 0.2]})
    with pytest.raises(rbridge.EstimationError, match="resamples failed"):
        bootstrap.bootstrap_interval(statistic, frame, resamples=5, seed=1)


def test_bootstrap_of_oe_through_r(r_available):
    frame = _simulated_cohort(n=300, seed=5)
    result = bootstrap.bootstrap_interval(
        lambda df: rbridge.observed_expected(_outcomes(df), df["risk"], 3.0).estimate,
        frame,
        resamples=25,
        seed=42,
    )
    assert math.isfinite(result.lower) and math.isfinite(result.upper)
    assert result.lower <= result.upper


class _FakeR:
    """Enough of the rpy2 robjects.r facade for doctor() to run without R."""

    def __init__(self, versions, score_error=None):
        self.versions = versions
        self.score_error = score_error

    def __call__(self, code):
        if "R.version.string" in code:
            return ["R version 4.4.1 (test)"]
        if "Score" in code and self.score_error is not None:
            raise RuntimeError(self.score_error)
        return []

    def __getitem__(self, name):
        if name == "packageVersion":

            def package_version(pkg):
                if pkg not in self.versions:
                    raise RuntimeError(f"there is no package called '{pkg}'")
                return [self.versions[pkg]]

            return package_version
        if name == "requireNamespace":
            return lambda pkg, quietly=True: [pkg in self.versions]
        if name == "as.character":
            return lambda value: value
        raise KeyError(name)


class _FakeRO:
    def __init__(self, versions, score_error=None):
        self.r = _FakeR(versions, score_error)
        self.globalenv = {}


def test_doctor_reports_r_missing(monkeypatch):
    def _missing():
        raise rbridge.RNotFoundError("cannot locate R: R_HOME is unset")

    monkeypatch.setattr(rbridge, "_r", _missing)
    checks = rbridge.doctor()
    assert [check.ok for check in checks] == [False]
    assert "install R" in checks[0].fix


def test_doctor_flags_wrong_package_version(monkeypatch):
    versions = dict(rbridge.PINNED_PACKAGES, survival="9.9-9")
    monkeypatch.setattr(rbridge, "_r", lambda: _FakeRO(versions))
    checks = {check.name: check for check in rbridge.doctor()}
    assert checks["R package survival"].ok is False
    assert "r-setup.R" in checks["R package survival"].fix
    assert checks["R package prodlim"].ok is True


def test_doctor_flags_missing_package_and_score_failure(monkeypatch):
    versions = dict(rbridge.PINNED_PACKAGES)
    del versions["mice"]
    monkeypatch.setattr(rbridge, "_r", lambda: _FakeRO(versions, score_error="boom"))
    checks = {check.name: check for check in rbridge.doctor()}
    assert checks["R package mice"].ok is False
    assert "r-setup.R" in checks["R package mice"].fix
    assert checks["riskRegression::Score"].ok is False
    assert "r-setup.R" in checks["riskRegression::Score"].fix


def test_r_setup_pins_match_the_bridge_table():
    script = (REPO_ROOT / "scripts" / "r-setup.R").read_text()
    block = re.search(r"pins\s*<-\s*c\((.*?)\)", script, re.DOTALL)
    assert block is not None
    pins = dict(re.findall(r'(\w+)\s*=\s*"([^"]+)"', block.group(1)))
    assert pins == rbridge.PINNED_PACKAGES


def test_only_rbridge_imports_rpy2():
    src = Path(rbridge.__file__).parent
    offenders = [
        path.name
        for path in src.rglob("*.py")
        if path.name != "rbridge.py" and re.search(r"(import|from)\s+rpy2", path.read_text())
    ]
    assert offenders == []


def test_unknown_event_label_raises():
    with pytest.raises(ValueError, match="unknown event"):
        rbridge.Outcomes(pl.Series("time", [1.0]), pl.Series("event", ["relapse"]))


def _delayed_entry_outcomes() -> tuple[rbridge.Outcomes, pl.Series]:
    frame = pl.DataFrame(
        {
            "time": [5.0, 6.0, 7.0, 4.0],
            "event": ["metastasis", "censored", "competing", "metastasis"],
            "entry": [0.0, 0.2, 0.0, 0.1],
            "risk": [0.2, 0.3, 0.1, 0.4],
        }
    )
    return rbridge.Outcomes(frame["time"], frame["event"], frame["entry"]), frame["risk"]


def test_calibration_refuses_non_zero_entry_times():
    outcomes, risk = _delayed_entry_outcomes()
    with pytest.raises(ValueError, match="entry"):
        rbridge.calibration(outcomes, risk, 3.0, bandwidth=0.05)


def test_performance_scores_refuse_non_zero_entry_times():
    outcomes, risk = _delayed_entry_outcomes()
    with pytest.raises(ValueError, match="entry"):
        rbridge.performance_scores(outcomes, {"m": risk}, 3.0)


def test_all_zero_entry_times_are_not_delayed_entry(r_available):
    frame = _simulated_cohort(n=300, seed=5)
    outcomes = rbridge.Outcomes(frame["time"], frame["event"], pl.Series([0.0] * frame.height))
    scores = rbridge.performance_scores(outcomes, {"m": frame["risk"]}, 3.0)
    assert scores.height > 0


def test_score_ignores_entry_times_on_the_pinned_riskregression(r_available):
    # The reason for the guard above: riskRegression::Score (pinned 2026.03.11)
    # accepts Hist(time, event, entry) but returns the same model AUC and
    # Brier score with and without the entry times, so a left-truncated
    # Score result is truncation-naive. If this test fails after an upgrade,
    # the guard can be reconsidered.
    rng = np.random.default_rng(11)
    n = 3000
    h1 = 0.15 * np.exp(rng.normal(0.0, 0.8, n))
    t1 = rng.exponential(1.0 / h1)
    t2 = rng.exponential(1.0 / 0.1, n)
    exit_time = np.minimum(np.minimum(t1, t2), 8.0)
    status = np.where(exit_time == 8.0, 0, np.where(t1 <= t2, 1, 2))
    entry = rng.uniform(0.0, 3.0, n)
    keep = exit_time > entry  # left truncation: later exits are over-sampled
    risk = h1 / (h1 + 0.1) * (1 - np.exp(-(h1 + 0.1) * 4.0))

    ro = rbridge._r()
    results = {}
    with rbridge._session(ro):
        rbridge._push(ro, "lpv_time", ro.FloatVector(exit_time[keep].tolist()))
        rbridge._push(ro, "lpv_status", ro.IntVector(status[keep].tolist()))
        rbridge._push(ro, "lpv_entry", ro.FloatVector(entry[keep].tolist()))
        rbridge._push(ro, "lpv_risk", ro.FloatVector(risk[keep].tolist()))
        for label, formula in (
            ("plain", "Hist(time, status) ~ 1"),
            ("entry", "Hist(time, status, entry = entry) ~ 1"),
        ):
            rbridge._eval(
                ro,
                f"""
suppressMessages(library(prodlim))
lpv_df <- data.frame(entry = lpv_entry, time = lpv_time, status = lpv_status)
lpv_score <- riskRegression::Score(
  list(m = lpv_risk), formula = {formula}, data = lpv_df, times = 4,
  metrics = c("auc", "brier"), cens.model = "km", cause = 1, conf.int = FALSE
)
lpv_auc <- subset(lpv_score$AUC$score, model == "m")$AUC
lpv_brier <- subset(lpv_score$Brier$score, model == "m")$Brier
""",
            )
            results[label] = (
                float(ro.globalenv["lpv_auc"][0]),
                float(ro.globalenv["lpv_brier"][0]),
            )
    assert results["entry"] == pytest.approx(results["plain"], abs=1e-9)
