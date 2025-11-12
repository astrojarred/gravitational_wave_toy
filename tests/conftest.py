"""Pytest configuration and shared fixtures."""

from pathlib import Path

import pytest


@pytest.fixture
def mock_data_dir():
    """Return the path to the mock data directory."""
    return Path(__file__).parent.parent / "data" / "mock_data"


@pytest.fixture
def mock_csv_path(mock_data_dir):
    """Return the path to the mock CSV file."""
    return mock_data_dir / "GRB_42_mock.csv"


@pytest.fixture
def mock_metadata_path(mock_data_dir):
    """Return the path to the mock metadata CSV file."""
    return mock_data_dir / "GRB_42_mock_metadata.csv"


@pytest.fixture
def mock_fits_path(mock_data_dir):
    """Return the path to the mock FITS file."""
    return mock_data_dir / "GRB_42_mock.fits"


@pytest.fixture
def sample_sensitivity_df():
    """Create a sample sensitivity dataframe for testing."""
    import astropy.units as u
    import pandas as pd

    # Create sample sensitivity curve data
    sensitivity_curves: list = [
        [1e-10, 1e-11, 1e-12, 1e-13] * u.Unit("erg cm-2 s-1"),
        [1e-10, 1e-11, 1e-12, 1e-13] * u.Unit("erg cm-2 s-1"),
        [1e-10, 1e-11, 1e-12, 1e-13] * u.Unit("erg cm-2 s-1"),
    ]
    photon_flux_curves = [
        [1e-9, 1e-10, 1e-11, 1e-12] * u.Unit("cm-2 s-1"),
        [1e-9, 1e-10, 1e-11, 1e-12] * u.Unit("cm-2 s-1"),
        [1e-9, 1e-10, 1e-11, 1e-12] * u.Unit("cm-2 s-1"),
    ]

    df = pd.DataFrame(
        {
            "coinc_event_id": [42, 42, 42],
            "irf_site": ["north", "north", "south"],
            "irf_zenith": [20, 40, 20],
            "irf_ebl": [False, False, True],
            "irf_config": ["alpha", "alpha", "alpha"],
            "irf_duration": [1800, 1800, 1800],
            "sensitivity_curve": sensitivity_curves,
            "photon_flux_curve": photon_flux_curves,
        }
    )
    return df


@pytest.fixture
def sample_extrapolation_df():
    """Create a sample extrapolation dataframe for testing."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "coinc_event_id": [42, 42, 42, 42],
            "obs_delay": [100, 1000, 10000, 100000],
            "obs_time": [10, 100, 1000, 10000],
            "irf_site": ["north", "north", "north", "north"],
            "irf_zenith": [20, 20, 20, 20],
            "long": [0.0, 0.0, 0.0, 0.0],
            "lat": [1.0, 1.0, 1.0, 1.0],
            "eiso": [2e50, 2e50, 2e50, 2e50],
            "dist": [100000.0, 100000.0, 100000.0, 100000.0],
            "theta_view": [5.0, 5.0, 5.0, 5.0],
            "irf_ebl_model": [
                "dominguez11",
                "dominguez11",
                "dominguez11",
                "dominguez11",
            ],
        }
    )
    return df
