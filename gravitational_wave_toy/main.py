import logging

import GWToyV2
import GWPlotter

# activaate logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run():

    GWToyV2.run()


def plot():

    GWPlotter.run()
