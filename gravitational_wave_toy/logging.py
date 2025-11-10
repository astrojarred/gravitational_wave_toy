import logging


class logger:
    def __init__(
        self,
        name: str,
        filename: str = "./gwtoy.log",
        level: int = logging.INFO,
        file_level: int = logging.DEBUG,
    ) -> None:
        self.name = name
        self.filename = filename
        self.file_level = file_level
        self.level = level

        # Create a logger
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.level)

        # Create a file handler
        file_handler = logging.FileHandler(self.filename)
        file_handler.setLevel(self.file_level)

        # Create a stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(self.level)

        # Create a formatter and add it to the handlers
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        # Add the handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def debug(self, message: str) -> None:
        self.logger.debug(message)

    def info(self, message: str) -> None:
        self.logger.info(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def error(self, message: str) -> None:
        self.logger.error(message)

    def critical(self, message: str) -> None:
        self.logger.critical(message)


"""python
# example usage
from gravitational_wave_toy.logging import logger
log = logger(name='test', filename='./gwtoy.log')
log.debug('This is a debug message')
log.info('This is an info message')
log.warning('This is a warning message')
log.error('This is an error message')
log.critical('This is a critical message')
"""
