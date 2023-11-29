import warnings
import logging
from contextlib import contextmanager

@contextmanager
def suppress_warnings_and_logs(logging_ok: bool = True):
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