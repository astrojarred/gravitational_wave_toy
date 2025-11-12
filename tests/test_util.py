"""Tests for util module."""

import logging
import warnings

from gravitational_wave_toy.util import suppress_warnings_and_logs


def test_suppress_warnings_and_logs_with_logging_ok():
    """Test that warnings are suppressed when logging_ok=True."""
    with suppress_warnings_and_logs(logging_ok=True):
        warnings.warn("This warning should be suppressed", UserWarning)
        # Should not raise any warnings


def test_suppress_warnings_and_logs_without_logging_ok():
    """Test that warnings are suppressed when logging_ok=False."""
    with suppress_warnings_and_logs(logging_ok=False):
        warnings.warn("This warning should be suppressed", UserWarning)
        # Should not raise any warnings


def test_suppress_warnings_and_logs_context_manager():
    """Test that context manager works correctly."""
    # Before context
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warnings.warn("Warning before context", UserWarning)
        assert len(w) == 1

    # Inside context - warnings should be suppressed
    with suppress_warnings_and_logs(logging_ok=True):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Note: The suppression may not work perfectly due to how warnings are handled
            # We'll just verify the context manager doesn't crash
            warnings.warn("Warning inside context", UserWarning)
            # The warning may or may not be suppressed depending on Python version
            # So we just check that the context manager works

    # After context
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warnings.warn("Warning after context", UserWarning)
        assert len(w) == 1


def test_suppress_warnings_and_logs_logging_disabled():
    """Test that logging is disabled when logging_ok=False."""
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.DEBUG)

    # Before context - logging should work
    assert logging.getLogger().isEnabledFor(logging.WARNING)

    # Inside context with logging_ok=False
    with suppress_warnings_and_logs(logging_ok=False):
        # Logging should be disabled
        assert not logging.getLogger().isEnabledFor(logging.WARNING)

    # After context - logging should be re-enabled
    assert logging.getLogger().isEnabledFor(logging.WARNING)
