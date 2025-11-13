"""Pytest configuration and shared fixtures."""

import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pytest

from gravitational_wave_toy.ctaoirf import IRFHouse


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


def download_irfs(cache_dir: Path) -> Path:
    """Download and extract CTAO IRFs from Zenodo if not already cached.
    
    Args:
        cache_dir: Directory to cache the IRFs.
        
    Returns:
        Path to the extracted IRF directory.
    """
    zenodo_url = "https://zenodo.org/records/5499840/files/cta-prod5-zenodo-fitsonly-v0.1.zip?download=1"
    zip_path = cache_dir / "cta-prod5-zenodo-fitsonly-v0.1.zip"
    irf_dir = cache_dir / "CTA-IRFs"
    
    # Create cache directory if it doesn't exist
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if IRFs are already extracted in the correct location
    if irf_dir.exists() and (irf_dir / "prod5-v0.1").exists():
        # Extract tar.gz files if needed
        _extract_tar_files(irf_dir / "prod5-v0.1" / "fits")
        return irf_dir
    
    # Check if files are already extracted but in wrong location (directly in cache_dir)
    fits_dir = cache_dir / "fits"
    if fits_dir.exists() and not irf_dir.exists():
        # Reorganize existing extracted files
        print("Reorganizing extracted IRF files...")
        irf_dir.mkdir(parents=True, exist_ok=True)
        prod5_dir = irf_dir / "prod5-v0.1"
        prod5_dir.mkdir(parents=True, exist_ok=True)
        
        for item in cache_dir.iterdir():
            if item.name in ["fits", "figures"] and item.is_dir():
                dest = prod5_dir / item.name
                if not dest.exists():
                    shutil.move(str(item), str(dest))
            elif item.name in ["LICENSE", "README.md", "Website.md"] and item.is_file():
                dest = prod5_dir / item.name
                if not dest.exists():
                    shutil.move(str(item), str(dest))
        
        if irf_dir.exists() and (irf_dir / "prod5-v0.1").exists():
            # Extract tar.gz files if needed
            _extract_tar_files(irf_dir / "prod5-v0.1" / "fits")
            return irf_dir
    
    # Download if zip doesn't exist
    if not zip_path.exists():
        try:
            print(f"Downloading IRFs from Zenodo to {zip_path}...")
            urlretrieve(zenodo_url, zip_path)
            print("Download complete.")
        except Exception as e:
            pytest.skip(f"Failed to download IRFs from Zenodo: {e}")
    
    # Extract zip file
    if not irf_dir.exists():
        try:
            print(f"Extracting IRFs to {cache_dir}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(cache_dir)
            print("Extraction complete.")
            
            # The zip extracts directly to cache_dir with fits/, figures/, etc.
            # We need to create the CTA-IRFs/prod5-v0.1 structure
            # Check if fits/ directory exists (indicating direct extraction)
            fits_dir = cache_dir / "fits"
            if fits_dir.exists():
                # Create the expected directory structure
                irf_dir.mkdir(parents=True, exist_ok=True)
                prod5_dir = irf_dir / "prod5-v0.1"
                prod5_dir.mkdir(parents=True, exist_ok=True)
                
                # Move extracted contents to prod5-v0.1
                for item in cache_dir.iterdir():
                    if item.name in ["fits", "figures"] and item.is_dir():
                        dest = prod5_dir / item.name
                        if not dest.exists():
                            shutil.move(str(item), str(dest))
                    elif item.name in ["LICENSE", "README.md", "Website.md"] and item.is_file():
                        dest = prod5_dir / item.name
                        if not dest.exists():
                            shutil.move(str(item), str(dest))
            
            # Handle case where zip extracts to a subdirectory
            if not irf_dir.exists():
                # Look for CTA-IRFs in subdirectories
                for item in cache_dir.iterdir():
                    if item.is_dir():
                        potential_irf_dir = item / "CTA-IRFs"
                        if potential_irf_dir.exists() and (potential_irf_dir / "prod5-v0.1").exists():
                            # Move to expected location
                            shutil.move(str(potential_irf_dir), str(irf_dir))
                            break
                        # Or the subdir itself might be CTA-IRFs
                        if item.name == "CTA-IRFs" and (item / "prod5-v0.1").exists():
                            shutil.move(str(item), str(irf_dir))
                            break
        except Exception as e:
            pytest.skip(f"Failed to extract IRFs: {e}")
    
    # Final check
    if not irf_dir.exists() or not (irf_dir / "prod5-v0.1").exists():
        pytest.skip(f"IRF directory structure not found. Expected {irf_dir}/prod5-v0.1/")
    
    # Extract tar.gz files if they exist
    _extract_tar_files(irf_dir / "prod5-v0.1" / "fits")
    
    return irf_dir


def _extract_tar_files(fits_dir: Path) -> None:
    """Extract tar.gz files in the fits directory to get the actual IRF files.
    
    Args:
        fits_dir: Path to the fits directory containing tar.gz files.
    """
    if not fits_dir.exists():
        return
    
    # Find all tar.gz files
    tar_files = list(fits_dir.glob("*.tar.gz"))
    
    for tar_file in tar_files:
        # Extract to a directory with the same name (without .tar.gz)
        # e.g., CTA-Performance-prod5-v0.1-South-20deg.FITS.tar.gz -> CTA-Performance-prod5-v0.1-South-20deg.FITS/
        # .stem only removes .gz, so we need to remove .tar as well
        dir_name = tar_file.stem  # Removes .gz -> .tar
        if dir_name.endswith(".tar"):
            dir_name = dir_name[:-4]  # Remove .tar
        extract_dir = fits_dir / dir_name
        if not extract_dir.exists() or not any(extract_dir.glob("*.fits.gz")):
            try:
                print(f"Extracting {tar_file.name}...")
                extract_dir.mkdir(parents=True, exist_ok=True)
                with tarfile.open(tar_file, "r:gz") as tar_ref:
                    tar_ref.extractall(extract_dir)
                print(f"Extracted {tar_file.name} to {extract_dir.name}/")
            except Exception as e:
                print(f"Warning: Failed to extract {tar_file.name}: {e}")


@pytest.fixture(scope="session")
def irf_house():
    """Fixture that provides an IRFHouse instance with downloaded IRFs.
    
    Downloads IRFs from Zenodo on first use and caches them for subsequent test runs.
    """
    cache_dir = Path(__file__).parent / ".cache"
    irf_dir = download_irfs(cache_dir)
    
    # Set GAMMAPY_DATA if not already set
    if "GAMMAPY_DATA" not in os.environ:
        data_dir = Path(__file__).parent.parent / "data"
        if data_dir.exists():
            os.environ["GAMMAPY_DATA"] = str(data_dir)
    
    # Create IRFHouse instance
    house = IRFHouse(base_directory=str(irf_dir), check_irfs=False)
    return house
