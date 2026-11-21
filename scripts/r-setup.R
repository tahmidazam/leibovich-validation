#!/usr/bin/env Rscript
# Install the pinned R package versions the rpy2 bridge needs into the user library.
# Usage: Rscript scripts/r-setup.R
#
# The pins mirror PINNED_PACKAGES in src/leibovich_validation/rbridge.py and must match.
# Pinned packages build from CRAN source archives; their dependencies install as binaries
# where CRAN provides them. remotes is installed unpinned.

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

# First pass: install current CRAN binaries, which pull in the dependency tree.
# The second pass then moves only the pinned packages to their pinned versions.
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
