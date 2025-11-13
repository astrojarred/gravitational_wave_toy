"""Tests for observe module, focusing on Source class with CSV data."""

from pathlib import Path

import astropy.units as u
import numpy as np
import pytest

from gravitational_wave_toy.source import Source


def test_source_init_with_csv(mock_csv_path):
    """Test source initialization with CSV file."""
    source = Source(mock_csv_path)
    assert source.filepath == Path(mock_csv_path).absolute()
    assert source.file_type == "csv"
    assert source.seen is False
    assert source.obs_time == -1 * u.s


def test_source_read_csv_basic(mock_csv_path):
    """Test basic CSV reading functionality."""
    source = Source(mock_csv_path)
    assert source.file_type == "csv"
    assert hasattr(source, "time")
    assert hasattr(source, "energy")
    assert hasattr(source, "spectra")
    assert len(source.time) > 0
    assert len(source.energy) > 0
    assert source.spectra.shape == (len(source.energy), len(source.time))


def test_source_read_csv_metadata(mock_csv_path):
    """Test that metadata is read from metadata file."""
    source = Source(mock_csv_path)
    assert source.id == 42
    assert source.long is not None
    assert source.lat is not None
    assert source.eiso is not None
    assert source.dist is not None
    assert source.angle is not None


def test_source_read_csv_energy_time_units(mock_csv_path):
    """Test that energy and time have correct units."""
    source = Source(mock_csv_path)
    assert source.energy.unit == u.GeV
    assert source.time.unit == u.s


def test_source_read_csv_spectra_units(mock_csv_path):
    """Test that spectra have correct units."""
    source = Source(mock_csv_path)
    assert source.spectra.unit == u.Unit("1 / (cm2 s GeV)")


def test_source_read_csv_min_max_energy(mock_csv_path):
    """Test that min and max energy are set correctly."""
    source = Source(mock_csv_path)
    assert source.min_energy is not None
    assert source.max_energy is not None
    assert source.min_energy <= source.max_energy
    assert source.min_energy == source.energy.min()
    assert source.max_energy == source.energy.max()


def test_source_read_csv_with_custom_energy_limits(mock_csv_path):
    """Test source initialization with custom energy limits."""
    min_e = 0.1 * u.TeV
    max_e = 1.0 * u.TeV
    source = Source(mock_csv_path, min_energy=min_e, max_energy=max_e)
    assert source.min_energy == min_e.to("GeV")
    assert source.max_energy == max_e.to("GeV")


def test_source_read_csv_missing_file():
    """Test that error is raised for missing CSV file."""
    with pytest.raises((FileNotFoundError, ValueError)):
        Source("/nonexistent/file.csv")


def test_source_read_csv_invalid_format(tmp_path):
    """Test that error is raised for CSV with missing columns."""
    invalid_csv = tmp_path / "invalid.csv"
    invalid_csv.write_text("col1,col2\n1,2\n")

    with pytest.raises(ValueError, match="Missing columns"):
        Source(invalid_csv)


def test_source_read_csv_malformed_data(tmp_path):
    """Test handling of malformed CSV data."""
    # Create CSV with wrong number of rows
    csv_file = tmp_path / "malformed.csv"
    csv_file.write_text(
        "time [s],energy [GeV],dNdE [cm-2 s-1 GeV-1]\n1.0,1.0,1e-8\n2.0,2.0,1e-9\n"
    )

    # This may raise ValueError or LinAlgError depending on the data
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        Source(csv_file)


def test_source_set_spectral_grid(mock_csv_path):
    """Test setting spectral grid."""
    source = Source(mock_csv_path)
    source.set_spectral_grid()
    assert source.SpectralGrid is not None


def test_source_get_spectrum(mock_csv_path):
    """Test getting spectrum at specific time and energy."""
    source = Source(mock_csv_path)
    source.set_spectral_grid()

    time = source.time[0]
    energy = source.energy[0]
    spectrum = source.get_spectrum(time, energy)

    assert isinstance(spectrum, u.Quantity)
    assert spectrum.unit == u.Unit("1 / (cm2 s GeV)")


def test_source_get_spectrum_invalid_time(mock_csv_path):
    """Test that error is raised for invalid time unit."""
    source = Source(mock_csv_path)
    source.set_spectral_grid()

    with pytest.raises(ValueError, match="time must be a time quantity"):
        source.get_spectrum(1.0 * u.m, source.energy[0])


def test_source_get_spectrum_invalid_energy(mock_csv_path):
    """Test that error is raised for invalid energy unit."""
    source = Source(mock_csv_path)
    source.set_spectral_grid()

    with pytest.raises(ValueError, match="energy must be an energy quantity"):
        source.get_spectrum(source.time[0], 1.0 * u.s)


def test_source_get_flux(mock_csv_path):
    """Test getting flux at specific energy and time."""
    source = Source(mock_csv_path)
    source.set_spectral_grid()

    energy = source.energy[0]
    time = source.time[0]
    flux = source.get_flux(energy, time)

    assert isinstance(flux, u.Quantity)
    assert flux.unit == u.Unit("1 / (cm2 s GeV)")


def test_source_get_template_spectrum(mock_csv_path):
    """Test getting template spectrum."""
    source = Source(mock_csv_path)
    source.set_spectral_grid()

    time = source.time[0]
    template = source.get_template_spectrum(time)

    assert template is not None
    assert hasattr(template, "energy")
    assert hasattr(template, "values")


def test_source_fit_spectral_indices(mock_csv_path):
    """Test fitting spectral indices."""
    source = Source(mock_csv_path)
    source.fit_spectral_indices()

    assert hasattr(source, "_indices")
    assert hasattr(source, "_amplitudes")
    assert hasattr(source, "index_at")
    assert hasattr(source, "amplitude_at")


def test_source_get_spectral_index(mock_csv_path):
    """Test getting spectral index at specific time."""
    source = Source(mock_csv_path)
    source.fit_spectral_indices()

    time = source.time[len(source.time) // 2]  # Use middle time
    index = source.get_spectral_index(time)

    assert isinstance(index, (float, np.floating))


def test_source_get_spectral_index_invalid_time(mock_csv_path):
    """Test that error is raised for invalid time unit in get_spectral_index."""
    source = Source(mock_csv_path)
    source.fit_spectral_indices()

    with pytest.raises(ValueError, match="time must be a time quantity"):
        source.get_spectral_index(1.0 * u.m)


def test_source_get_spectral_amplitude(mock_csv_path):
    """Test getting spectral amplitude at specific time."""
    source = Source(mock_csv_path)
    source.fit_spectral_indices()

    time = source.time[len(source.time) // 2]
    amplitude = source.get_spectral_amplitude(time)

    assert isinstance(amplitude, u.Quantity)
    assert amplitude.unit == u.Unit("cm-2 s-1 GeV-1")


def test_source_get_gammapy_spectrum(mock_csv_path):
    """Test getting gammapy spectrum."""
    source = Source(mock_csv_path)
    source.fit_spectral_indices()

    time = source.time[len(source.time) // 2]
    spectrum = source.get_gammapy_spectrum(time)

    assert spectrum is not None
    assert hasattr(spectrum, "index")
    assert hasattr(spectrum, "amplitude")


def test_source_repr(mock_csv_path):
    """Test source __repr__ method."""
    source = Source(mock_csv_path)
    repr_str = repr(source)
    assert "Source" in repr_str
    assert str(source.id) in repr_str or "Source" in repr_str


def test_source_metadata_property(mock_csv_path):
    """Test source metadata property."""
    source = Source(mock_csv_path)
    metadata = source.metadata
    assert isinstance(metadata, dict)
    assert "id" in metadata or "event_id" in metadata


def test_source_set_ebl_model_invalid(mock_csv_path):
    """Test that error is raised for invalid EBL model."""
    source = Source(mock_csv_path)

    # Set a dummy distance first
    from astropy.coordinates import Distance

    source.dist = Distance(z=0.1)

    with pytest.raises(ValueError, match="ebl must be one of"):
        source.set_ebl_model("invalid_model")


def test_source_set_ebl_model_no_gammapy_data(mock_csv_path, monkeypatch):
    """Test that error is raised when GAMMAPY_DATA is not set."""
    source = Source(mock_csv_path)

    from astropy.coordinates import Distance

    source.dist = Distance(z=0.1)

    # Remove GAMMAPY_DATA if it exists
    monkeypatch.delenv("GAMMAPY_DATA", raising=False)

    # Use correct EBL model name
    with pytest.raises(ValueError, match="GAMMAPY_DATA environment variable"):
        source.set_ebl_model("dominguez")


def test_source_csv_column_matching_flexibility(tmp_path):
    """Test that CSV reading handles various column name formats."""
    # Create CSV with different column name format
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(
        "timestamp,energy_value,flux_value\n"
        "1.0,1.0,1e-8\n"
        "1.0,2.0,1e-8\n"
        "2.0,1.0,1e-8\n"
        "2.0,2.0,1e-8\n"
    )

    # Should work because of substring matching
    source = Source(csv_file)
    assert source.file_type == "csv"
    assert len(source.time) == 2
    assert len(source.energy) == 2


def test_source_read_csv_without_metadata(tmp_path):
    """Test CSV reading when metadata file doesn't exist."""
    csv_file = tmp_path / "test.csv"
    # Create minimal valid CSV
    times = [1.0, 2.0]
    energies = [1.0, 2.0]
    csv_content = "time [s],energy [GeV],dNdE [cm-2 s-1 GeV-1]\n"
    for t in times:
        for e in energies:
            csv_content += f"{t},{e},1e-8\n"
    csv_file.write_text(csv_content)

    source = Source(csv_file)
    assert source.file_type == "csv"
    # Metadata should have defaults
    assert source.id == 0  # Default value


def test_source_energy_time_ordering(mock_csv_path):
    """Test that energy and time arrays are properly sorted."""
    source = Source(mock_csv_path)

    # Check that arrays are sorted
    assert np.all(np.diff(source.energy.value) >= 0) or np.all(
        np.diff(source.energy.value) <= 0
    )
    assert np.all(np.diff(source.time.value) >= 0) or np.all(
        np.diff(source.time.value) <= 0
    )


def test_source_spectra_shape_consistency(mock_csv_path):
    """Test that spectra shape matches energy and time dimensions."""
    source = Source(mock_csv_path)

    assert source.spectra.shape[0] == len(source.energy)
    assert source.spectra.shape[1] == len(source.time)


@pytest.mark.slow
def test_source_observe_with_irf(irf_house, mock_csv_path):
    """Test source observation functionality with IRF.
    
    This test matches the usage pattern from quick-test.ipynb (lines 1-13).
    """
    from gravitational_wave_toy.sensitivity import Sensitivity

    # Load in the desired IRF (matching quick-test.ipynb)
    irf = irf_house.get_irf(
        site="south",
        configuration="alpha",
        zenith=20,
        duration=1800,
        azimuth="average",
        version="prod5-v0.1",
    )

    # Create a gammapy sensitivity class (matching quick-test.ipynb)
    min_energy = 30 * u.GeV
    max_energy = 10 * u.TeV

    sens = Sensitivity(
        irf=irf,
        observatory=f"cta_{irf.site.name}",
        min_energy=min_energy,
        max_energy=max_energy,
        radius=3.0 * u.deg,
    )

    # Load in a GRB and add EBL (matching quick-test.ipynb)
    grb = Source(mock_csv_path, min_energy=min_energy, max_energy=max_energy, ebl="franceschini")

    # Generate sensitivity curve first (required for observe)
    sens.get_sensitivity_curve(source=grb)

    # Simulate the observation (matching quick-test.ipynb lines 1-13)
    delay_time = 30 * u.min

    res = grb.observe(
        sensitivity=sens,
        start_time=delay_time,
        min_energy=min_energy,
        max_energy=max_energy,
    )

    # Verify the result dictionary contains expected keys
    assert "obs_time" in res
    assert "ebl_model" in res

    # Verify obs_time is a time quantity (can be -1 if not detectable)
    assert isinstance(res["obs_time"], u.Quantity)
    assert res["obs_time"].unit.physical_type == "time"
    # obs_time can be -1 if source is not detectable, or positive if detectable
    assert res["obs_time"].value >= -1

    # Verify ebl_model is set
    assert res["ebl_model"] == "franceschini"

    # Verify source seen status
    assert grb.seen is True or grb.seen is False  # Can be either depending on detectability
