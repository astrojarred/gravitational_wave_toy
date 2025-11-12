"""Tests for ctaoirf module."""

import pytest

from gravitational_wave_toy.ctaoirf import (
    IRF,
    Azimuth,
    Configuration,
    Duration,
    IRFHouse,
    Site,
    Version,
    Zenith,
)


def test_site_enum():
    """Test Site enum values."""
    assert Site.south.value == "south"
    assert Site.north.value == "north"


def test_configuration_enum():
    """Test Configuration enum values."""
    assert Configuration.alpha.value == "alpha"
    assert Configuration.omega.value == "omega"


def test_azimuth_enum():
    """Test Azimuth enum values."""
    assert Azimuth.south.value == "south"
    assert Azimuth.north.value == "north"
    assert Azimuth.average.value == "average"


def test_zenith_enum():
    """Test Zenith IntEnum values."""
    assert Zenith.z20 == 20
    assert Zenith.z40 == 40
    assert Zenith.z60 == 60
    assert isinstance(Zenith.z20, int)


def test_duration_enum():
    """Test Duration IntEnum values."""
    assert Duration.t1800 == 1800
    assert Duration.t18000 == 18000
    assert Duration.t180000 == 180000
    assert isinstance(Duration.t1800, int)


def test_irf_initialization_with_valid_file(tmp_path):
    """Test IRF initialization with a valid file."""
    # Create a temporary FITS-like file
    irf_file = tmp_path / "test_irf.fits"
    irf_file.write_bytes(b"dummy fits content")

    irf = IRF(
        filepath=str(irf_file),
        configuration=Configuration.alpha,
        site=Site.north,
        duration=1800,
        azimuth=Azimuth.average,
    )

    assert irf.filepath == irf_file
    assert irf.configuration == Configuration.alpha
    assert irf.site == Site.north
    assert irf.duration == 1800
    assert irf.azimuth == Azimuth.average
    assert irf.has_nsb is False


def test_irf_initialization_with_base_directory(tmp_path):
    """Test IRF initialization with base directory."""
    base_dir = tmp_path / "irf_base"
    base_dir.mkdir()
    irf_file = base_dir / "test_irf.fits"
    irf_file.write_bytes(b"dummy fits content")

    irf = IRF(
        base_directory=str(base_dir),
        filepath="test_irf.fits",
        configuration=Configuration.alpha,
        site=Site.south,
        duration=1800,
        azimuth=Azimuth.north,
    )

    assert irf.filepath == irf_file
    assert irf.base_directory == base_dir.resolve()


def test_irf_validation_base_directory_nonexistent():
    """Test IRF validation fails for nonexistent base directory."""
    with pytest.raises(ValueError, match="does not exist"):
        IRF(
            base_directory="/nonexistent/path",
            filepath="test.fits",
            configuration=Configuration.alpha,
            site=Site.north,
            duration=1800,
            azimuth=Azimuth.average,
        )


def test_irf_validation_filepath_nonexistent(tmp_path):
    """Test IRF validation fails for nonexistent filepath."""
    with pytest.raises(ValueError, match="does not exist"):
        IRF(
            filepath=str(tmp_path / "nonexistent.fits"),
            configuration=Configuration.alpha,
            site=Site.north,
            duration=1800,
            azimuth=Azimuth.average,
        )


def test_irf_repr(tmp_path):
    """Test IRF __repr__ method."""
    irf_file = tmp_path / "test_irf.fits"
    irf_file.write_bytes(b"dummy fits content")

    irf = IRF(
        filepath=str(irf_file),
        configuration=Configuration.alpha,
        site=Site.north,
        zenith=Zenith.z20,
        duration=Duration.t1800,
        azimuth=Azimuth.average,
        n_sst=0,
        n_mst=9,
        n_lst=4,
        version=Version.prod5_v0p1,
    )

    repr_str = repr(irf)
    assert "CTAO IRF" in repr_str
    assert "prod5-v0.1" in repr_str
    assert "test_irf.fits" in repr_str
    assert "alpha" in repr_str
    assert "north" in repr_str
    assert "1800s" in repr_str


def test_irf_fspath(tmp_path):
    """Test IRF __fspath__ method."""
    irf_file = tmp_path / "test_irf.fits"
    irf_file.write_bytes(b"dummy fits content")

    irf = IRF(
        filepath=str(irf_file),
        configuration=Configuration.alpha,
        site=Site.north,
        duration=Duration.t1800,
        azimuth=Azimuth.average,
    )

    assert irf.__fspath__() == str(irf_file)


def test_irfhouse_initialization(tmp_path):
    """Test IRFHouse initialization."""
    base_dir = tmp_path / "irf_base"
    base_dir.mkdir()

    irf_house = IRFHouse(base_directory=str(base_dir), check_irfs=False)
    assert irf_house.base_directory == base_dir.resolve()
    assert irf_house.check_irfs is False


def test_irfhouse_validation_base_directory_nonexistent():
    """Test IRFHouse validation fails for nonexistent base directory."""
    with pytest.raises(ValueError, match="does not exist"):
        IRFHouse(base_directory="/nonexistent/path", check_irfs=False)


def test_irfhouse_get_irf_invalid_version(tmp_path):
    """Test IRFHouse.get_irf raises error for invalid version."""
    base_dir = tmp_path / "irf_base"
    base_dir.mkdir()

    irf_house = IRFHouse(base_directory=str(base_dir), check_irfs=False)
    with pytest.raises(ValueError, match="Invalid version"):
        irf_house.get_irf(
            site=Site.north,
            configuration=Configuration.alpha,
            zenith=Zenith.z20,
            duration=Duration.t1800,
            azimuth=Azimuth.average,
            version="fake",  # type: ignore
        )
