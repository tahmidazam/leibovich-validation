"""The only module that talks to R, through rpy2.

Inputs are Polars series and frames of predicted risks, follow-up times and
event types; outputs are Polars frames and plain typed values. The R calls
mirror Prediction_CSC_minimal.R in the companion repository of the STRATOS
competing-risks validation guidance (van Geloven et al., BMJ 2022,
doi:10.1136/bmj-2021-069249), pinned at commit
ba1d9e8e68ed4f031e55dab473ce9084694bdec9
(github.com/survival-lumc/ValidationCompRisks).

Event coding at the boundary: an event series holds "metastasis", "competing"
or "censored". prodlim and riskRegression want the 0/1/2 numeric code with 0
for censoring; survival::survfit wants a factor whose first level is the
censoring state. The conversion happens here and nowhere else.
"""

from __future__ import annotations

import contextlib
import math
import os
import shutil
import subprocess
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast, overload

import polars as pl

# prodlim and riskRegression count causes from 1 with 0 for censoring.
EVENT_CODES: dict[str, int] = {"censored": 0, "metastasis": 1, "competing": 2}
EVENT_NAMES: dict[int, str] = {code: name for name, code in EVENT_CODES.items()}
EVENT_LABELS: tuple[str, ...] = ("censored", "metastasis", "competing")

# CRAN versions pinned by scripts/r-setup.R and checked by `doctor`. A test
# keeps the two tables in sync. Packages that other packages import come
# first, so install_version's upgrade = "never" leaves the pins in place.
PINNED_PACKAGES: dict[str, str] = {
    "survival": "3.8-12",
    "prodlim": "2026.03.11",
    "cmprsk": "2.2-12",
    "geepack": "1.3.13",
    "mice": "3.19.0",
    "pec": "2025.06.24",
    "riskRegression": "2026.03.11",
}

# The pinned packages assume a current R; the known-good STRATOS session used
# R 4.3.3 (Prediction_CSC.md sessionInfo at the pinned commit).
MIN_R_VERSION = "4.0.0"

# qnorm(0.975), the half-width multiplier of the log-scale confidence interval
# for the O/E ratio.
_Z_975 = 1.959963984540054

# Scratch objects in the R global environment carry this prefix.
_TEMP_PREFIX = "lpv_"


class RBridgeError(RuntimeError):
    """The R bridge could not do what was asked."""


class RNotFoundError(RBridgeError):
    """rpy2 could not find or start R."""


class EstimationError(RBridgeError):
    """R could not produce a finite estimate from the data given."""


@dataclass(frozen=True)
class Outcomes:
    """One group's follow-up: exit time, event type, optional delayed entry.

    Validated once at construction, so the estimation functions take a checked
    bundle instead of re-validating the same series at every call. `codes`
    holds the 0/1/2 event coding prodlim and riskRegression expect.
    """

    time: pl.Series
    event: pl.Series
    entry: pl.Series | None = None
    codes: list[int] = field(init=False)

    def __post_init__(self) -> None:
        _check_aligned(self.time, self.event, entry=self.entry)
        object.__setattr__(self, "codes", _event_codes(self.event))


@dataclass(frozen=True)
class AalenJohansen:
    """Observed cumulative incidence at the horizon."""

    estimate: float
    se: float


@dataclass(frozen=True)
class OERatio:
    """Observed over expected risk with a 95% interval on the log scale."""

    estimate: float
    lower: float
    upper: float


@dataclass(frozen=True)
class Calibration:
    """GEE calibration intercept and slope, the ICI, plus the curve points."""

    intercept: float
    slope: float
    curve: pl.DataFrame  # predicted risk against pseudo-observed outcome
    ici: float  # integrated calibration index of the smoothed curve


@dataclass(frozen=True)
class SmoothedCalibration:
    """The smoothed calibration curve and its integrated calibration index."""

    ici: float
    grid: pl.DataFrame  # predicted, estimate, n (patients within the bandwidth)


@dataclass(frozen=True)
class BootstrapInterval:
    """A percentile interval over `resamples` resamples of the patients."""

    lower: float
    upper: float
    resamples: int
    skipped: int


@dataclass(frozen=True)
class Check:
    """One doctor() check: what was found, and what to run when it failed."""

    name: str
    ok: bool
    detail: str
    fix: str


_LABELS_R = ", ".join(f'"{label}"' for label in EVENT_LABELS)


def _aj_script(*, left_truncated: bool) -> str:
    """The survfit() script behind `observed_incidence`.

    Mirrors `summary(survfit(Surv(time, status) ~ 1), times = horizon)` in
    the STRATOS script. `status` must be the factor; the 0/1/2 numeric column
    would give an ordinary two-state survival object. The state column is
    found by name because survfit may drop states that see no transitions.
    With delayed entry the patient is at risk only from `entry`, which Surv
    handles with its (start, stop] form and the id column that form needs;
    a cause with no pstate column is then an error, where the plain path
    reads it as a zero estimate.
    """
    entry = "  entry = lpv_entry,\n" if left_truncated else ""
    identity = "lpv_df$id <- seq_len(nrow(lpv_df))\n" if left_truncated else ""
    surv = "entry, time, status" if left_truncated else "time, status"
    id_arg = ", id = id" if left_truncated else ""
    one_row = "rbind" if left_truncated else "matrix"
    row_args = "" if left_truncated else ", nrow = 1L"
    missing = (
        'if (is.na(lpv_j)) stop(paste("no pstate column for cause", lpv_cause_label))\n'
        "lpv_aj <- c(lpv_pstate[1L, lpv_j], lpv_stderr[1L, lpv_j])"
        if left_truncated
        else (
            "lpv_aj <- if (is.na(lpv_j)) c(0, 0) "
            "else c(lpv_pstate[1L, lpv_j], lpv_stderr[1L, lpv_j])"
        )
    )
    return f"""
lpv_df <- data.frame(
{entry}  time = lpv_time,
  status = factor(lpv_status, levels = c(0L, 1L, 2L), labels = c({_LABELS_R}))
)
{identity}lpv_surv <- survival::survfit(
  survival::Surv({surv}) ~ 1, data = lpv_df{id_arg}
)
lpv_sum <- summary(lpv_surv, times = lpv_horizon, extend = TRUE)
lpv_pstate <- lpv_sum$pstate
lpv_stderr <- lpv_sum$std.err
if (is.null(dim(lpv_pstate))) {{
  lpv_pstate <- {one_row}(lpv_pstate{row_args})
  lpv_stderr <- {one_row}(lpv_stderr{row_args})
}}
lpv_j <- match(lpv_cause_label, colnames(lpv_pstate))
{missing}
"""


def _score_script() -> str:
    """The Score() call in Prediction_CSC_minimal.R at the pinned commit.

    Since riskRegression 2025.09.17, cens.method must include "pseudo" or
    the calibration plotframe comes back without pseudo-observations
    (ValidationCompRisks issue 4). The survival package must be attached:
    Score() evaluates Surv() for the censoring model in the calling session.
    """
    return """
suppressMessages(library(prodlim))
suppressMessages(library(survival))
lpv_df <- data.frame(time = lpv_time, status_num = lpv_status)
lpv_score <- riskRegression::Score(
  lpv_models,
  formula = Hist(time, status_num) ~ 1,
  data = lpv_df,
  times = lpv_horizon,
  metrics = c("auc", "brier"),
  summary = c("ipa"),
  cens.model = "km",
  cens.method = c("ipcw", "pseudo"),
  cause = lpv_cause,
  plots = "calibration",
  conf.int = TRUE
)
"""


# Nearest-neighbour smoother of the pseudo-observations on predicted risk, as
# the STRATOS script does with plotCalibration(pseudo = TRUE, bandwidth = 0.05,
# round = FALSE), which prodlim::meanNeighbors implements: Prediction_CSC.md
# section 2.1.1.2 at commit ba1d9e8e68ed4f031e55dab473ce9084694bdec9 of
# https://github.com/survival-lumc/ValidationCompRisks. The ICI is the mean
# absolute difference between the smoothed curve and predicted risk over all
# patients.
# The grid keeps the unique predicted risks with at least `minimum_support`
# patients within the bandwidth. Risks are rounded to 10 decimals first:
# meanNeighbors labels factor(x) by unique(x), and factor() merges doubles that
# agree to 15 significant digits, as risks pooled over imputations can.
_SMOOTH_R = """
lpv_smooth <- function(risk, pseudovalue, bandwidth, minimum_support) {
  risk <- round(risk, 10)
  nn <- prodlim::meanNeighbors(x = risk, y = pseudovalue, bandwidth = bandwidth)
  fitted <- nn$averageY[match(risk, nn$uniqueX)]
  support <- vapply(nn$uniqueX, function(u) sum(abs(risk - u) <= bandwidth), numeric(1))
  keep <- support >= minimum_support
  list(
    ici = mean(abs(fitted - risk)),
    grid = data.frame(predicted = nn$uniqueX[keep], estimate = nn$averageY[keep],
                      n = support[keep])
  )
}
"""

# GEE on the Score() pseudo-observations, as in the STRATOS script: cloglog
# link, independence working correlation, fixed scale, jackknife errors
# (jack = TRUE). The reported slope is 1 + the cll_pred coefficient. The
# plotframe shape changed across riskRegression versions (it was plotframe,
# then plotFrames[[1]]; the id column became riskRegression_ID), so the code
# normalises it to a named list of per-model frames first.
_CALIBRATION_R = """
lpv_pf <- lpv_score$Calibration$plotframe
if (is.null(lpv_pf)) lpv_pf <- lpv_score$Calibration$plotFrames
if (is.null(lpv_pf)) stop("Score returned no pseudo-observations")
if (is.data.frame(lpv_pf)) {
  if ("model" %in% names(lpv_pf)) {
    # Score() adds a "Null model" row set alongside each model's; keep only the
    # named models.
    lpv_pf <- split(lpv_pf, lpv_pf$model)[lpv_model_names]
  } else {
    if (length(lpv_model_names) > 1L) {
      stop("Score plotframe has no model column for a multi-model call")
    }
    lpv_pf <- stats::setNames(list(lpv_pf), lpv_model_names[1L])
  }
}
if (is.null(names(lpv_pf)) && length(lpv_pf) == length(lpv_model_names)) {
  names(lpv_pf) <- lpv_model_names
}
lpv_int <- numeric(length(lpv_model_names))
lpv_slope <- numeric(length(lpv_model_names))
lpv_ici <- numeric(length(lpv_model_names))
lpv_curves <- data.frame()
for (lpv_i in seq_along(lpv_model_names)) {
  lpv_nm <- lpv_model_names[lpv_i]
  lpv_pseudo <- as.data.frame(lpv_pf[[lpv_nm]])
  lpv_pseudo <- lpv_pseudo[order(lpv_pseudo$risk), ]
  lpv_pseudo$cll_pred <- log(-log(1 - lpv_pseudo$risk))
  lpv_fit_int <- geepack::geese(
    pseudovalue ~ offset(cll_pred), data = lpv_pseudo, id = riskRegression_ID,
    scale.fix = TRUE, family = gaussian, mean.link = "cloglog",
    corstr = "independence", jack = TRUE
  )
  lpv_fit_slope <- geepack::geese(
    pseudovalue ~ offset(cll_pred) + cll_pred, data = lpv_pseudo,
    id = riskRegression_ID, scale.fix = TRUE, family = gaussian,
    mean.link = "cloglog", corstr = "independence", jack = TRUE
  )
  lpv_int[lpv_i] <- unname(lpv_fit_int$beta[1L])
  lpv_slope[lpv_i] <- 1 + unname(lpv_fit_slope$beta["cll_pred"])
  lpv_ici[lpv_i] <- lpv_smooth(lpv_pseudo$risk, lpv_pseudo$pseudovalue,
                               lpv_bandwidth, 0)$ici
  lpv_curves <- rbind(
    lpv_curves,
    data.frame(model = lpv_nm, risk = lpv_pseudo$risk,
               pseudovalue = lpv_pseudo$pseudovalue)
  )
}
"""

# CSC with the default coxph fitter, as sharing_CSC_model.R recommends; the
# training frame stays in the session because predictRisk needs it. String
# covariates become factors with the same levels in train and new, the same
# precaution as the defensive relevel() calls in the STRATOS Rmd.
_CSC_R = """
suppressMessages(library(prodlim))
for (lpv_nm in names(lpv_train)) {
  if ((is.character(lpv_train[[lpv_nm]]) || is.factor(lpv_train[[lpv_nm]])) &&
      lpv_nm %in% names(lpv_new)) {
    lpv_new[[lpv_nm]] <- factor(lpv_new[[lpv_nm]],
                                levels = levels(factor(lpv_train[[lpv_nm]])))
  }
}
lpv_fit <- riskRegression::CSC(stats::as.formula(lpv_formula), data = lpv_train)
lpv_risk <- as.numeric(riskRegression::predictRisk(
  lpv_fit, cause = lpv_cause, newdata = lpv_new, times = lpv_horizon
))
"""

# One chained-equations run. lpv_method_names/lpv_method_values give the mice
# method of each column to impute; lpv_modelcols lists the columns allowed on
# the right-hand side of every equation. String and logical columns become
# factors, and lpv_ordered_cols become ordered factors, so polyreg, logreg and
# polr see the input types they expect and cannot invent new levels. Factor
# columns go back to character for the trip to Python.
_MICE_R = """
suppressMessages(library(mice))
for (lpv_nm in lpv_factor_cols) {
  lpv_df[[lpv_nm]] <- factor(lpv_df[[lpv_nm]], ordered = lpv_nm %in% lpv_ordered_cols)
}
lpv_predmat <- matrix(0L, ncol(lpv_df), ncol(lpv_df),
                      dimnames = list(names(lpv_df), names(lpv_df)))
lpv_predmat[lpv_method_names, lpv_modelcols] <- 1L
diag(lpv_predmat) <- 0L
lpv_methods <- rep("", ncol(lpv_df))
names(lpv_methods) <- names(lpv_df)
lpv_methods[lpv_method_names] <- lpv_method_values
lpv_imp <- mice::mice(lpv_df, m = lpv_m, maxit = lpv_maxit, method = lpv_methods,
                      predictorMatrix = lpv_predmat, seed = lpv_seed,
                      printFlag = FALSE)
lpv_completed <- mice::complete(lpv_imp, action = "long", include = FALSE)
lpv_completed$.id <- NULL
for (lpv_nm in lpv_factor_cols) lpv_completed[[lpv_nm]] <- as.character(lpv_completed[[lpv_nm]])
"""

_DOCTOR_SCORE_R = """
suppressMessages(library(prodlim))
suppressMessages(library(survival))
lpv_toy <- data.frame(
  time = c(0.4, 0.8, 1.2, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5),
  status_num = c(1L, 0L, 2L, 0L, 1L, 0L, 2L, 1L, 0L, 1L)
)
riskRegression::Score(
  list(toy = c(0.2, 0.3, 0.1, 0.4, 0.5, 0.3, 0.2, 0.6, 0.1, 0.4)),
  formula = Hist(time, status_num) ~ 1,
  data = lpv_toy,
  times = 3,
  metrics = "brier",
  cens.model = "km",
  cens.method = c("ipcw", "pseudo"),
  cause = 1,
  conf.int = FALSE
)
"""

_ro: Any = None
_ri: Any = None


def _find_r_home() -> str:
    """R_HOME for rpy2: `R RHOME` first, then the standard install locations."""
    candidates: list[str] = []
    r_bin = shutil.which("R")
    if r_bin is not None:
        try:
            out = subprocess.run(
                [r_bin, "RHOME"], check=True, capture_output=True, text=True, timeout=30
            )
        except OSError, subprocess.SubprocessError:
            pass
        else:
            candidates.append(out.stdout.strip())
    candidates.extend(
        [
            "/Library/Frameworks/R.framework/Resources",  # CRAN macOS installer
            "/opt/homebrew/lib/R",  # Homebrew on Apple Silicon
            "/usr/local/lib/R",  # Homebrew on Intel and most Linux builds
            "/usr/lib/R",
        ]
    )
    for candidate in candidates:
        if candidate and (Path(candidate) / "bin").is_dir():
            return candidate
    raise RNotFoundError(
        "cannot locate R: R_HOME is unset and `R RHOME` failed; "
        f"install R >= {MIN_R_VERSION} (https://cloud.r-project.org) or set R_HOME"
    )


def _r() -> Any:
    """The rpy2 robjects module, with the embedded R session running."""
    global _ro, _ri
    if _ro is not None:
        return _ro
    if not os.environ.get("R_HOME"):
        with contextlib.suppress(RNotFoundError):
            os.environ["R_HOME"] = _find_r_home()  # rpy2 has its own discovery as a fallback
    try:
        import rpy2.rinterface as rinterface
        import rpy2.robjects as robjects

        robjects.r("1L")  # force the embedded session to start now
    except ModuleNotFoundError as exc:
        raise RNotFoundError("rpy2 is not installed; run `uv sync`") from exc
    except Exception as exc:  # rpy2 raises RRuntimeError or OSError when R will not start
        raise RNotFoundError(
            f"rpy2 could not start R: {exc}; install R >= {MIN_R_VERSION} or set R_HOME"
        ) from exc
    _ro, _ri = robjects, rinterface
    return _ro


def _eval(ro: Any, code: str) -> Any:
    try:
        return ro.r(code)
    except Exception as exc:  # rpy2 RRuntimeError and friends
        raise EstimationError(f"R evaluation failed: {exc}") from exc


def _sweep(ro: Any) -> None:
    """Drop every lpv_ scratch object from the R session."""
    with contextlib.suppress(Exception):
        ro.r(f'rm(list = ls(pattern = "^{_TEMP_PREFIX}"), envir = .GlobalEnv)')


@contextlib.contextmanager
def _session(ro: Any) -> Generator[None]:
    """A clean scratch area in the R global environment."""
    _sweep(ro)
    try:
        yield
    finally:
        _sweep(ro)


def _push(ro: Any, name: str, value: Any) -> None:
    ro.globalenv[name] = value


def _check_aligned(time: pl.Series, event: pl.Series, entry: pl.Series | None = None) -> None:
    if len(time) == 0:
        raise EstimationError("empty data: nothing to estimate")
    if len(event) != len(time):
        raise ValueError("time and event series must be the same length")
    if time.null_count():
        raise ValueError("time contains missing values")
    if event.null_count():
        raise ValueError("event contains missing values")
    if entry is not None:
        if len(entry) != len(time):
            raise ValueError("entry and time series must be the same length")
        if entry.null_count():
            raise ValueError("entry contains missing values")
        if (entry < 0).any():
            raise ValueError("entry times cannot be negative")
        # Delayed entry after the exit time is meaningless: the patient would
        # never be at risk.
        if (entry >= time).any():
            raise ValueError("every entry time must be before the exit time")


def _check_risks(outcomes: Outcomes, *risks: pl.Series) -> None:
    """The risk vectors must hold one prediction per patient."""
    if any(len(risk) != len(outcomes.time) for risk in risks):
        raise ValueError("time, event and risk series must be the same length")


def _cause_code(cause: str) -> int:
    try:
        code = EVENT_CODES[cause]
    except KeyError:
        causes = [name for name, value in EVENT_CODES.items() if value > 0]
        raise ValueError(f"unknown cause {cause!r}; expected one of {causes}") from None
    if code == 0:
        raise ValueError("'censored' marks the absence of an event, not a cause")
    return code


def _event_codes(event: pl.Series) -> list[int]:
    unknown = event.filter(event.is_not_null() & ~event.is_in(list(EVENT_CODES)))
    if unknown.len():
        raise ValueError(f"unknown event {unknown[0]!r}; expected one of {sorted(EVENT_CODES)}")
    return event.replace(EVENT_CODES).cast(pl.Int64).to_list()


def _require_event(outcomes: Outcomes, horizon: float, cause: str) -> None:
    n_events = ((outcomes.event == cause) & (outcomes.time <= horizon)).sum()
    if n_events == 0:
        raise EstimationError(
            f"no {cause!r} events at or before the {horizon}-year horizon; "
            "the estimate would be undefined"
        )


def _require_no_delayed_entry(outcomes: Outcomes) -> None:
    """Score() cannot take left-truncated data, so any non-zero entry raises.

    riskRegression::Score (pinned 2026.03.11) accepts entry times in Hist()
    but returns the same model AUC, Brier score and pseudo-observations with
    and without them (tests/test_rbridge.py). Re-origin the patients at the
    start of follow-up first (`cohort.reorigin_at_randomisation`).
    """
    if outcomes.entry is not None and (outcomes.entry != 0).any():
        raise ValueError(
            "Score() ignores entry times, so left-truncated data cannot be scored; "
            "re-origin follow-up so every entry time is zero"
        )


def _risk_vector(risk: pl.Series) -> None:
    if risk.null_count():
        raise ValueError("predicted risk contains missing values")
    lo = float(cast(Any, risk.min()))
    hi = float(cast(Any, risk.max()))
    if lo <= 0 or hi >= 1:
        raise ValueError(
            "predicted risks must lie strictly inside (0, 1) for the cloglog "
            f"calibration transform; found range [{lo}, {hi}]"
        )


def _push_outcomes(ro: Any, outcomes: Outcomes) -> None:
    _push(ro, "lpv_time", ro.FloatVector(outcomes.time.to_list()))
    _push(ro, "lpv_status", ro.IntVector(outcomes.codes))
    if outcomes.entry is not None:
        _push(ro, "lpv_entry", ro.FloatVector(outcomes.entry.to_list()))


def _push_models(ro: Any, risks: dict[str, pl.Series]) -> None:
    _push(
        ro,
        "lpv_models",
        ro.ListVector({name: ro.FloatVector(series.to_list()) for name, series in risks.items()}),
    )
    _push(ro, "lpv_model_names", ro.StrVector(list(risks)))


def _push_frame(ro: Any, name: str, frame: pl.DataFrame) -> None:
    vectors = {}
    for col_name in frame.columns:
        series = frame.get_column(col_name)
        if series.dtype == pl.Boolean:
            vec = ro.BoolVector([_ri.NA_Logical if v is None else bool(v) for v in series])
        elif series.dtype.is_integer():
            vec = ro.IntVector([_ri.NA_Integer if v is None else int(v) for v in series])
        elif series.dtype.is_numeric():
            vec = ro.FloatVector([_ri.NA_Real if v is None else float(v) for v in series])
        else:
            # Strings, categoricals and anything else go over as character.
            vec = ro.StrVector([_ri.NA_Character if v is None else str(v) for v in series])
        vectors[col_name] = vec
    _push(ro, name, ro.r["data.frame"](**vectors, stringsAsFactors=False))


def _is_na(value: Any) -> bool:
    return (
        value is _ri.NA_Integer
        or value is _ri.NA_Character
        or value is _ri.NA_Logical
        or (isinstance(value, float) and math.isnan(value))
    )


def _pull_frame(ro: Any, name: str) -> pl.DataFrame:
    """An R data.frame as a Polars frame, positional to dodge name quoting."""
    columns = []
    for i, col_name in enumerate((str(c) for c in ro.r(f"names({name})")), start=1):
        r_class = str(ro.r(f"class({name}[[{i}L]])")[0])
        values = ro.r(f"{name}[[{i}L]]")
        if r_class == "integer":
            series = pl.Series(col_name, [None if _is_na(v) else int(v) for v in values])
        elif r_class == "logical":
            series = pl.Series(col_name, [None if _is_na(v) else bool(v) for v in values])
        elif r_class == "character":
            series = pl.Series(col_name, [None if _is_na(v) else str(v) for v in values])
        else:
            series = pl.Series(col_name, [None if _is_na(v) else float(v) for v in values])
        columns.append(series)
    return pl.DataFrame(columns)


def _finite_or_raise(value: Any, what: str) -> float:
    out = _finite_or_none(value)
    if out is None:
        raise EstimationError(f"{what} returned no finite value")
    return out


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _numeric_column(ro: Any, table: str, column: str, n: int) -> list[Any]:
    if not bool(ro.r(f'"{column}" %in% names({table})')[0]):
        return [None] * n
    return list(ro.r(f"as.numeric({table}${column})"))


def _pull_metric(
    ro: Any, table: str, value_col: str, metric: str, *, ci: bool = True
) -> list[dict[str, Any]]:
    if not bool(ro.r(f'"{value_col}" %in% names({table})')[0]):
        return []
    models = [str(v) for v in ro.r(f"as.character({table}$model)")]
    n = len(models)
    estimates = _numeric_column(ro, table, value_col, n)
    # The se/lower/upper columns of Brier$score belong to the Brier estimate;
    # IPA carries no interval of its own (the STRATOS IPA intervals were
    # bootstrapped), so IPA rows report the point estimate only.
    ses = _numeric_column(ro, table, "se", n) if ci else [None] * n
    lowers = _numeric_column(ro, table, "lower", n) if ci else [None] * n
    uppers = _numeric_column(ro, table, "upper", n) if ci else [None] * n
    return [
        {
            "model": model,
            "metric": metric,
            "estimate": _finite_or_none(estimate),
            "se": _finite_or_none(se),
            "lower": _finite_or_none(lower),
            "upper": _finite_or_none(upper),
        }
        for model, estimate, se, lower, upper in zip(
            models, estimates, ses, lowers, uppers, strict=True
        )
        if model != "Null model"
    ]


def observed_incidence(
    outcomes: Outcomes,
    horizon: float,
    cause: str = "metastasis",
) -> AalenJohansen:
    """Aalen-Johansen cumulative incidence of `cause` at `horizon`, with its SE.

    When `outcomes` carries entry times the estimate is left-truncated: each
    patient joins the risk set at their entry time. This is the only
    estimator in the bridge that handles delayed entry.
    """
    _cause_code(cause)
    n_cause = ((outcomes.event == cause) & (outcomes.time <= horizon)).sum()
    if n_cause == 0:
        # The cause is never entered before the horizon, so its cumulative
        # incidence is exactly zero; skip the R round trip.
        return AalenJohansen(0.0, 0.0)
    ro = _r()
    with _session(ro):
        _push_outcomes(ro, outcomes)
        _push(ro, "lpv_horizon", float(horizon))
        _push(ro, "lpv_cause_label", cause)
        _eval(ro, _aj_script(left_truncated=outcomes.entry is not None))
        estimate, se = (float(v) for v in ro.globalenv["lpv_aj"])
    return AalenJohansen(estimate, se)


def oe_ratio(aj: AalenJohansen, expected: float) -> OERatio:
    """Observed incidence over mean predicted risk, log-scale interval.

    Shared by `observed_expected` and the score stage, which computes one
    Aalen-Johansen estimate and divides it by each model's mean risk.
    """
    if not math.isfinite(expected) or expected <= 0:
        raise EstimationError(f"mean predicted risk must be positive, got {expected}")
    if aj.estimate == 0:
        # With no observed events the log-scale interval cannot bound the
        # ratio; report the zero estimate rather than a NaN.
        return OERatio(0.0, 0.0, math.inf)
    estimate = aj.estimate / expected
    half = _Z_975 * aj.se / aj.estimate
    return OERatio(
        estimate,
        math.exp(math.log(estimate) - half),
        math.exp(math.log(estimate) + half),
    )


def observed_expected(
    outcomes: Outcomes,
    risk: pl.Series,
    horizon: float,
    cause: str = "metastasis",
) -> OERatio:
    """Observed over expected risk at `horizon`, log-scale interval."""
    _check_risks(outcomes, risk)
    if risk.null_count():
        raise EstimationError("predicted risk contains missing values")
    expected = float(cast(Any, risk.mean()))
    aj = observed_incidence(outcomes, horizon, cause)
    return oe_ratio(aj, expected)


@overload
def calibration(
    outcomes: Outcomes,
    risks: pl.Series,
    horizon: float,
    cause: str = "metastasis",
    *,
    bandwidth: float,
) -> Calibration: ...


@overload
def calibration(
    outcomes: Outcomes,
    risks: dict[str, pl.Series],
    horizon: float,
    cause: str = "metastasis",
    *,
    bandwidth: float,
) -> dict[str, Calibration]: ...


def calibration(
    outcomes: Outcomes,
    risks: pl.Series | dict[str, pl.Series],
    horizon: float,
    cause: str = "metastasis",
    *,
    bandwidth: float,
) -> Calibration | dict[str, Calibration]:
    """Calibration intercept, slope and ICI from Score() pseudo-observations.

    The intercept and slope come from GEE; the ICI from the nearest-neighbour
    smoother of `_SMOOTH_R` at `bandwidth` (in units of predicted risk).

    The curve member holds one row per patient: predicted risk against the
    pseudo-observed outcome, ordered by predicted risk. Pass several named
    risk vectors to get one Calibration back per model. Raises ValueError
    when any entry time is non-zero (see `_require_no_delayed_entry`).
    """
    _require_no_delayed_entry(outcomes)
    risk_map = risks if isinstance(risks, dict) else {"model": risks}
    _check_risks(outcomes, *risk_map.values())
    _cause_code(cause)
    _require_event(outcomes, horizon, cause)
    for series in risk_map.values():
        _risk_vector(series)
    ro = _r()
    curves: dict[str, pl.DataFrame] = {}
    with _session(ro):
        _push_outcomes(ro, outcomes)
        _push_models(ro, risk_map)
        _push(ro, "lpv_horizon", float(horizon))
        _push(ro, "lpv_cause", EVENT_CODES[cause])
        _push(ro, "lpv_bandwidth", float(bandwidth))
        _eval(ro, _score_script())
        _eval(ro, _SMOOTH_R)
        _eval(ro, _CALIBRATION_R)
        icis = [float(v) for v in ro.globalenv["lpv_ici"]]
        intercepts = [float(v) for v in ro.globalenv["lpv_int"]]
        slopes = [float(v) for v in ro.globalenv["lpv_slope"]]
        models = [str(v) for v in _eval(ro, "as.character(lpv_curves$model)")]
        predicted = [float(v) for v in _eval(ro, "as.numeric(lpv_curves$risk)")]
        observed = [float(v) for v in _eval(ro, "as.numeric(lpv_curves$pseudovalue)")]
    for name in risk_map:
        curves[name] = pl.DataFrame(
            {
                "predicted": [p for p, m in zip(predicted, models, strict=True) if m == name],
                "observed": [o for o, m in zip(observed, models, strict=True) if m == name],
            }
        )
    result = {
        name: Calibration(intercept, slope, curves[name], ici)
        for name, intercept, slope, ici in zip(risk_map, intercepts, slopes, icis, strict=True)
    }
    if isinstance(risks, dict):
        return result
    return result["model"]


def smooth_calibration(
    curve: pl.DataFrame, bandwidth: float, minimum_support: int
) -> SmoothedCalibration:
    """Smooth pseudo-observations on predicted risk, with the ICI.

    `curve` has columns predicted and observed, one row per patient. The grid
    stops where fewer than `minimum_support` patients lie within `bandwidth`
    of the predicted risk; the ICI uses every patient.
    """
    _risk_vector(curve["predicted"])
    ro = _r()
    with _session(ro):
        _push(ro, "lpv_risk", ro.FloatVector(curve["predicted"].to_list()))
        _push(ro, "lpv_pseudo", ro.FloatVector(curve["observed"].to_list()))
        _eval(ro, _SMOOTH_R)
        _eval(
            ro,
            f"lpv_sm <- lpv_smooth(lpv_risk, lpv_pseudo, {float(bandwidth)!r}, "
            f"{int(minimum_support)}L)",
        )
        ici = _finite_or_raise(ro.r("lpv_sm$ici")[0], "the ICI")
        _eval(ro, "lpv_grid <- lpv_sm$grid")
        grid = _pull_frame(ro, "lpv_grid")
    return SmoothedCalibration(ici, grid)


_SCORE_SCHEMA = {
    "model": pl.String,
    "metric": pl.String,
    "estimate": pl.Float64,
    "se": pl.Float64,
    "lower": pl.Float64,
    "upper": pl.Float64,
}


def performance_scores(
    outcomes: Outcomes,
    risks: dict[str, pl.Series],
    horizon: float,
    cause: str = "metastasis",
) -> pl.DataFrame:
    """Time-dependent AUC, Brier score and IPA at `horizon` for each named model.

    Returns a tidy frame with columns model, metric ("auc", "brier" or "ipa"),
    estimate, se, lower and upper. Null-model rows are dropped. Raises ValueError
    when any entry time is non-zero (see `_require_no_delayed_entry`).
    """
    _require_no_delayed_entry(outcomes)
    if not risks:
        raise ValueError("at least one named risk vector is required")
    _check_risks(outcomes, *risks.values())
    _cause_code(cause)
    _require_event(outcomes, horizon, cause)
    rows: list[dict[str, Any]] = []
    ro = _r()
    with _session(ro):
        _push_outcomes(ro, outcomes)
        _push_models(ro, risks)
        _push(ro, "lpv_horizon", float(horizon))
        _push(ro, "lpv_cause", EVENT_CODES[cause])
        _eval(ro, _score_script())
        _eval(ro, "lpv_auc_dt <- as.data.frame(lpv_score$AUC$score)")
        _eval(ro, "lpv_brier_dt <- as.data.frame(lpv_score$Brier$score)")
        rows += _pull_metric(ro, "lpv_auc_dt", "AUC", "auc")
        rows += _pull_metric(ro, "lpv_brier_dt", "Brier", "brier")
        rows += _pull_metric(ro, "lpv_brier_dt", "IPA", "ipa", ci=False)
    return pl.DataFrame(rows, schema=_SCORE_SCHEMA, orient="row")


def read_rds(path: Path) -> pl.DataFrame:
    """An R .rds data.frame as a Polars frame; factors become strings."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    ro = _r()
    with _session(ro):
        _push(ro, "lpv_path", str(path))
        _eval(
            ro,
            """
lpv_rdf <- as.data.frame(readRDS(lpv_path))
for (lpv_nm in names(lpv_rdf)) {
  if (is.factor(lpv_rdf[[lpv_nm]])) lpv_rdf[[lpv_nm]] <- as.character(lpv_rdf[[lpv_nm]])
}
""",
        )
        return _pull_frame(ro, "lpv_rdf")


def predict_csc(
    train: pl.DataFrame,
    new: pl.DataFrame,
    horizon: float,
    formula: str,
    cause: str = "metastasis",
) -> pl.Series:
    """Fit riskRegression::CSC on `train` and predict `cause` risk at `horizon`.

    `formula` is the CSC formula, for example
    "Hist(time, status_num) ~ age + size + ncat". Exists to reproduce the
    STRATOS worked example; the project models predict in Python instead.
    """
    ro = _r()
    with _session(ro):
        _push_frame(ro, "lpv_train", train)
        _push_frame(ro, "lpv_new", new)
        _push(ro, "lpv_formula", formula)
        _push(ro, "lpv_horizon", float(horizon))
        _push(ro, "lpv_cause", _cause_code(cause))
        _eval(ro, _CSC_R)
        return pl.Series("risk", [float(v) for v in ro.globalenv["lpv_risk"]])


def mice_impute(
    frame: pl.DataFrame,
    methods: dict[str, str],
    model_columns: list[str],
    m: int,
    seed: int,
    maxit: int = 5,
) -> pl.DataFrame:
    """One `mice::mice` run; the stacked completed frame in long form.

    `frame` holds the columns the imputation model can see. `methods` maps each
    column to impute onto its mice method ("polyreg", "logreg" or "polr" here);
    `model_columns` names the columns allowed as predictors in the equations.
    String and logical columns cross to R as factors, and a column with the "polr"
    method crosses as an ordered factor; both come back as strings, so the
    caller restores the original dtypes. The return frame keeps the input row
    order within each imputation and marks the imputation in a `.imp` column.
    """
    if not methods:
        raise ValueError("methods is empty: nothing for mice to impute")
    unknown = [c for c in (*methods, *model_columns) if c not in frame.columns]
    if unknown:
        raise ValueError(f"unknown columns in methods or model_columns: {unknown}")
    factor_cols = [name for name in frame.columns if frame[name].dtype in (pl.String, pl.Boolean)]
    ordered_cols = [name for name, method in methods.items() if method == "polr"]
    ro = _r()
    with _session(ro):
        _push_frame(ro, "lpv_df", frame)
        _push(ro, "lpv_factor_cols", ro.StrVector([*factor_cols, *ordered_cols]))
        _push(ro, "lpv_ordered_cols", ro.StrVector(ordered_cols))
        _push(ro, "lpv_method_names", ro.StrVector(list(methods)))
        _push(ro, "lpv_method_values", ro.StrVector(list(methods.values())))
        _push(ro, "lpv_modelcols", ro.StrVector(list(model_columns)))
        _push(ro, "lpv_m", int(m))
        _push(ro, "lpv_maxit", int(maxit))
        _push(ro, "lpv_seed", int(seed))
        _eval(ro, _MICE_R)
        return _pull_frame(ro, "lpv_completed")


def _package_version(ro: Any, name: str) -> str | None:
    try:
        return str(ro.r["as.character"](ro.r["packageVersion"](name))[0])
    except Exception:
        return None


def _version_tuple(text: str) -> tuple[int, ...]:
    """R writes 3.8-12 and 3.8.12 for the same version; compare digit runs."""
    return tuple(int(part) for part in text.replace("-", ".").split("."))


def _package_loads(ro: Any, name: str) -> bool:
    try:
        return bool(ro.r["requireNamespace"](name, quietly=True)[0])
    except Exception:
        return False


def doctor() -> list[Check]:
    """Check the R toolchain end to end, with a fix for each failure."""
    checks: list[Check] = []
    try:
        ro = _r()
    except RBridgeError as exc:
        return [
            Check(
                "rpy2 finds R",
                False,
                str(exc),
                f"install R >= {MIN_R_VERSION} (https://cloud.r-project.org), "
                "set R_HOME, or run `uv sync` to install rpy2",
            )
        ]
    try:
        version = str(ro.r("R.version.string")[0])
    except Exception:
        version = "R started but its version string is unavailable"
    checks.append(Check("rpy2 finds R", True, version, ""))
    for name, pinned in PINNED_PACKAGES.items():
        installed = _package_version(ro, name)
        loads = installed is not None and _package_loads(ro, name)
        if installed is None:
            detail = "not installed"
        elif _version_tuple(installed) != _version_tuple(pinned):
            detail = f"installed {installed}, pinned {pinned}"
        elif not loads:
            detail = f"installed {installed} but does not load"
        else:
            checks.append(Check(f"R package {name}", True, installed, ""))
            continue
        checks.append(Check(f"R package {name}", False, detail, "run `Rscript scripts/r-setup.R`"))
    try:
        with _session(ro):
            _eval(ro, _DOCTOR_SCORE_R)
    except EstimationError as exc:
        checks.append(
            Check("riskRegression::Score", False, str(exc), "run `Rscript scripts/r-setup.R`")
        )
    else:
        checks.append(Check("riskRegression::Score", True, "ran on toy data", ""))
    return checks
