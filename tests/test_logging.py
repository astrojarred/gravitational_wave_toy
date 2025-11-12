"""Tests for logging module."""

import logging
import tempfile
from pathlib import Path

from gravitational_wave_toy.logging import logger


def test_logger_initialization():
    """Test logger initialization with default parameters."""
    log = logger(name="test_logger")
    assert log.name == "test_logger"
    assert log.filename == "./gwtoy.log"
    assert log.level == logging.INFO
    assert log.file_level == logging.DEBUG
    assert log.logger is not None


def test_logger_custom_filename():
    """Test logger with custom filename."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logfile = Path(tmpdir) / "custom.log"
        log = logger(name="test", filename=str(logfile))
        assert log.filename == str(logfile)
        assert logfile.exists()


def test_logger_custom_levels():
    """Test logger with custom levels."""
    log = logger(name="test", level=logging.WARNING, file_level=logging.ERROR)
    assert log.level == logging.WARNING
    assert log.file_level == logging.ERROR


def test_logger_debug():
    """Test debug logging."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logfile = Path(tmpdir) / "test.log"
        log = logger(name="test", filename=str(logfile), level=logging.DEBUG)
        log.debug("Debug message")
        log.logger.handlers[0].flush()
        assert "Debug message" in logfile.read_text()


def test_logger_info():
    """Test info logging."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logfile = Path(tmpdir) / "test.log"
        log = logger(name="test", filename=str(logfile))
        log.info("Info message")
        log.logger.handlers[0].flush()
        assert "Info message" in logfile.read_text()


def test_logger_warning():
    """Test warning logging."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logfile = Path(tmpdir) / "test.log"
        log = logger(name="test", filename=str(logfile))
        log.warning("Warning message")
        log.logger.handlers[0].flush()
        assert "Warning message" in logfile.read_text()


def test_logger_error():
    """Test error logging."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logfile = Path(tmpdir) / "test.log"
        log = logger(name="test", filename=str(logfile))
        log.error("Error message")
        log.logger.handlers[0].flush()
        assert "Error message" in logfile.read_text()


def test_logger_critical():
    """Test critical logging."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logfile = Path(tmpdir) / "test.log"
        log = logger(name="test", filename=str(logfile))
        log.critical("Critical message")
        log.logger.handlers[0].flush()
        assert "Critical message" in logfile.read_text()


def test_logger_handlers():
    """Test that logger has both file and stream handlers."""
    # Clear any existing handlers to avoid interference from other tests
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    log = logger(name="test_unique_handler")
    handlers = log.logger.handlers
    assert len(handlers) == 2
    handler_types = [type(h).__name__ for h in handlers]
    assert "FileHandler" in handler_types
    assert "StreamHandler" in handler_types

    # Clean up
    log.logger.handlers.clear()


def test_logger_file_level_filtering():
    """Test that file level filtering works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logfile = Path(tmpdir) / "test.log"
        log = logger(name="test", filename=str(logfile), file_level=logging.ERROR)
        log.debug("Debug message")
        log.info("Info message")
        log.warning("Warning message")
        log.error("Error message")
        log.logger.handlers[0].flush()
        content = logfile.read_text()
        assert "Debug message" not in content
        assert "Info message" not in content
        assert "Warning message" not in content
        assert "Error message" in content
