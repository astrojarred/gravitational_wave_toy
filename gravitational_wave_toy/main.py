import logging

from . import GWToyV2
from . import GWPlotter

# activaate logger
logging.basicConfig(level=logging.INFO)


def run():

    GWToyV2.run()


def plot():

    GWPlotter.run()
