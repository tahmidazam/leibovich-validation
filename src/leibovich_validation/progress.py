import sys

from tqdm import tqdm


def bar(iterable=None, *, disable: bool | None = None, **kwargs) -> tqdm:
    """A tqdm bar on stderr that fits the terminal.

    `disable` is True to silence the bar and None (the default) to hide it when stderr
    is not a terminal. Other arguments go to tqdm: `desc`, `total`, `unit`, `leave`.
    """
    return tqdm(iterable, file=sys.stderr, dynamic_ncols=True, disable=disable, **kwargs)
