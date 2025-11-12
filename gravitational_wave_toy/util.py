import logging
import warnings
from contextlib import contextmanager


@contextmanager
def suppress_warnings_and_logs(logging_ok: bool = True):
    """A helper function to suppress warnings and logs.

    Args:
        logging_ok: Whether to suppress logging. Defaults to True.

    Yields:
        None
    """

    if logging_ok:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    else:
        logging.disable(logging.WARNING)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
        logging.disable(logging.NOTSET)
