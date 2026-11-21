import contextlib
import signal
from collections.abc import Generator


class Interrupted(BaseException):
    """A SIGINT or SIGTERM. Not an Exception, so only the console entry point handles it."""

    def __init__(self, signum: int) -> None:
        self.signum = signum


_pending: int | None = None
_installed = False
_deferring = False


def _raise(signum: int, _frame: object) -> None:
    global _pending
    _pending = signum
    signal.signal(signum, signal.SIG_DFL)  # a second signal ends the process at once
    if not _deferring:
        raise Interrupted(signum)


def install() -> None:
    """Turn SIGINT and SIGTERM into `Interrupted`."""
    global _installed
    _installed = True
    signal.signal(signal.SIGINT, _raise)
    signal.signal(signal.SIGTERM, _raise)


def rearm() -> None:
    """Reinstall the handlers if `install` ran. Embedding R replaces them with R's own."""
    if _installed:
        install()


def check() -> None:
    """Raise `Interrupted` if a signal has arrived."""
    if _pending is not None:
        raise Interrupted(_pending)


@contextlib.contextmanager
def deferred() -> Generator[None]:
    """Hold an arriving signal until the block ends, then raise `Interrupted`.

    For a call into R. rpy2 runs Python's signal handlers inside the call and drops
    what they raise, so the call runs to completion and the stop follows its return.
    """
    global _deferring
    _deferring = True
    try:
        yield
    finally:
        _deferring = False
        check()
