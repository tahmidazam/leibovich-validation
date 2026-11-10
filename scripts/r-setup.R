#!/usr/bin/env Rscript
# Install the pinned R package versions the rpy2 bridge needs, into the user
# library. Usage: Rscript scripts/r-setup.R
#
# The pin table mirrors PINNED_PACKAGES in
# src/leibovich_validation/rbridge.py; a test keeps the two in sync.
# Install order matters: packages that the others import (survival, prodlim,
# cmprsk, geepack) go first, so upgrade = "never" leaves the pinned versions
# in place when the heavier packages pull their dependency trees. Pinned
# packages build from the CRAN source archives; their dependencies install as
# binaries where CRAN provides them. The installer itself (remotes) is
# bootstrapped from CRAN at the current version; only the analysis packages
# are pinned.

if (getRversion() < "4.0.0") {
  stop("R >= 4.0.0 is required (the STRATOS reference session used R 4.3.3); found ",
       R.version.string)
}

options(repos = c(CRAN = "https://cloud.r-project.org"))

if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes")
}

pins <- c(
  survival = "3.8-12",
  prodlim = "2026.03.11",
  cmprsk = "2.2-12",
  geepack = "1.3.13",
  mice = "3.19.0",
  pec = "2025.06.24",
  riskRegression = "2026.03.11"
)

# First pass: current CRAN binaries of the pinned packages, which pull the whole
# dependency tree (rms, Hmisc, timereg, mets, data.table, ranger, glmnet, mvtnorm,
# multcomp, Publish, ...) as binaries. install_version then downgrades only the
# pinned packages; their dependencies stay at the current versions.
for (pkg in names(pins)) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message(sprintf("%-15s installing current release plus dependencies", pkg))
    install.packages(pkg)
  }
}

for (pkg in names(pins)) {
  want <- unname(pins[[pkg]])
  have <- tryCatch(packageVersion(pkg), error = function(e) package_version("0"))
  if (have == package_version(want)) {
    message(sprintf("%-15s already at %s", pkg, want))
    next
  }
  message(sprintf("%-15s installing %s", pkg, want))
  remotes::install_version(pkg, want, upgrade = "never", dependencies = FALSE)
}

failed <- names(pins)[!vapply(
  names(pins),
  function(pkg) {
    isTRUE(tryCatch(packageVersion(pkg), error = function(e) package_version("0")) ==
             package_version(pins[[pkg]]))
  },
  logical(1)
)]
if (length(failed) > 0) {
  stop("failed to install: ", paste(failed, collapse = ", "))
}
message("all ", length(pins), " pinned packages installed")
