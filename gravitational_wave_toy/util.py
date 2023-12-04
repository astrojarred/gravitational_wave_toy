import logging
import warnings
from contextlib import contextmanager

from astropy import units as u
from astropy.cosmology import Planck15
from astropy.cosmology import units as cu


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
