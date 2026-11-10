import matplotlib
import pytest

from leibovich_validation import rbridge

# Figure tests render headless.
matplotlib.use("Agg")


@pytest.fixture(scope="session")
def r_available():
    """Skip every R-backed test when rpy2, R or a pinned package is missing."""
    try:
        checks = rbridge.doctor()
    except Exception as exc:
        pytest.skip(f"R bridge cannot start: {exc}")
    failed = [check for check in checks if not check.ok]
    if failed:
        reasons = "; ".join(f"{check.name} ({check.detail})" for check in failed)
        pytest.skip(f"R environment not ready: {reasons}")
    return True
