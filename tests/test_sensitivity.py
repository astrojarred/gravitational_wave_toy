"""Tests for sensitivity module."""

import astropy.units as u
import numpy as np
import pytest

from gravitational_wave_toy.sensitivity import ScaledTemplateModel, Sensitivity


def test_scaled_template_model_initialization():
    """Test ScaledTemplateModel initialization."""
    energy = np.logspace(-1, 2, 10) * u.TeV
    values = np.ones(10) * u.Unit("1 / (TeV s cm2)")

    model = ScaledTemplateModel(energy=energy, values=values, scaling_factor=1e-6)

    assert model.scaling_factor == 1e-6
    assert model.amplitude.value == 1e-6


def test_scaled_template_model_scaling_factor_property():
    """Test ScaledTemplateModel scaling_factor property."""
    energy = np.logspace(-1, 2, 10) * u.TeV
    values = np.ones(10) * u.Unit("1 / (TeV s cm2)")

    model = ScaledTemplateModel(energy=energy, values=values, scaling_factor=1e-6)

    # Test getter
    assert model.scaling_factor == 1e-6

    # Test setter
    model.scaling_factor = 1e-5
    assert model.scaling_factor == 1e-5
    assert model.amplitude.value == 1e-5


def test_scaled_template_model_values_property():
    """Test ScaledTemplateModel values property."""
    energy = np.logspace(-1, 2, 10) * u.TeV
    original_values = np.ones(10) * u.Unit("1 / (TeV s cm2)")

    model = ScaledTemplateModel(
        energy=energy, values=original_values, scaling_factor=2.0
    )

    # Values should be scaled
    scaled_values = model.values
    assert np.allclose(scaled_values.value, original_values.value * 2.0)


def test_scaled_template_model_from_template():
    """Test ScaledTemplateModel.from_template factory method."""
    from gammapy.modeling.models import TemplateSpectralModel

    energy = np.logspace(-1, 2, 10) * u.TeV
    values = np.ones(10) * u.Unit("1 / (TeV s cm2)")

    template = TemplateSpectralModel(energy=energy, values=values)
    scaled = ScaledTemplateModel.from_template(template, scaling_factor=1e-6)

    assert isinstance(scaled, ScaledTemplateModel)
    assert scaled.scaling_factor == 1e-6


def test_scaled_template_model_evaluate():
    """Test ScaledTemplateModel evaluate method."""
    energy = np.logspace(-1, 2, 10) * u.TeV
    values = np.ones(10) * u.Unit("1 / (TeV s cm2)")

    model = ScaledTemplateModel(energy=energy, values=values, scaling_factor=2.0)

    test_energy = 1.0 * u.TeV
    result = model.evaluate(test_energy)

    assert isinstance(result, u.Quantity)
    # Should be interpolated value scaled by factor


def test_sensitivity_initialization_with_curves():
    """Test Sensitivity initialization with sensitivity and photon flux curves."""
    sensitivity_curve = np.logspace(-10, -12, 10) * u.Unit("erg cm-2 s-1")
    photon_flux_curve = np.logspace(-9, -11, 10) * u.Unit("cm-2 s-1")

    sens = Sensitivity(
        observatory="cta_north",
        radius=3.0 * u.deg,
        min_energy=0.02 * u.TeV,
        max_energy=10.0 * u.TeV,
        n_sensitivity_points=10,
        sensitivity_curve=sensitivity_curve,
        photon_flux_curve=photon_flux_curve,
    )

    assert sens.observatory == "cta_north"
    assert sens.radius == 3.0 * u.deg
    assert sens.min_energy == 0.02 * u.TeV
    assert sens.max_energy == 10.0 * u.TeV


def test_sensitivity_initialization_missing_curves():
    """Test that error is raised when curves are missing."""
    with pytest.raises(
        ValueError,
        match="Must provide either irf, sensitivity_curve or photon_flux_curve.",
    ):
        Sensitivity(
            observatory="cta_north",
            radius=3.0 * u.deg,
            min_energy=0.02 * u.TeV,
            max_energy=10.0 * u.TeV,
            n_sensitivity_points=10,
        )


def test_sensitivity_initialization_invalid_energy():
    """Test that error is raised for invalid energy units."""
    sensitivity_curve = np.logspace(-10, -12, 10) * u.Unit("erg cm-2 s-1")
    photon_flux_curve = np.logspace(-9, -11, 10) * u.Unit("cm-2 s-1")

    with pytest.raises(ValueError, match="e_min must be an energy quantity"):
        Sensitivity(
            observatory="cta_north",
            radius=3.0 * u.deg,
            min_energy=1.0 * u.s,  # Wrong unit
            max_energy=10.0 * u.TeV,
            n_sensitivity_points=10,
            sensitivity_curve=sensitivity_curve,
            photon_flux_curve=photon_flux_curve,
        )


def test_sensitivity_get_sensitivity_mode():
    """Test Sensitivity.get method in sensitivity mode."""
    sensitivity_curve = np.logspace(-10, -12, 10) * u.Unit("erg cm-2 s-1")
    photon_flux_curve = np.logspace(-9, -11, 10) * u.Unit("cm-2 s-1")

    sens = Sensitivity(
        observatory="cta_north",
        radius=3.0 * u.deg,
        min_energy=0.02 * u.TeV,
        max_energy=10.0 * u.TeV,
        n_sensitivity_points=10,
        sensitivity_curve=sensitivity_curve,
        photon_flux_curve=photon_flux_curve,
    )

    test_time = 100.0 * u.s
    result = sens.get(test_time, mode="sensitivity")

    assert isinstance(result, u.Quantity)
    assert result.unit == u.Unit("erg cm-2 s-1")


def test_sensitivity_get_photon_flux_mode():
    """Test Sensitivity.get method in photon_flux mode."""
    sensitivity_curve = np.logspace(-10, -12, 10) * u.Unit("erg cm-2 s-1")
    photon_flux_curve = np.logspace(-9, -11, 10) * u.Unit("cm-2 s-1")

    sens = Sensitivity(
        observatory="cta_north",
        radius=3.0 * u.deg,
        min_energy=0.02 * u.TeV,
        max_energy=10.0 * u.TeV,
        n_sensitivity_points=10,
        sensitivity_curve=sensitivity_curve,
        photon_flux_curve=photon_flux_curve,
    )

    test_time = 100.0 * u.s
    result = sens.get(test_time, mode="photon_flux")

    assert isinstance(result, u.Quantity)
    assert result.unit == u.Unit("cm-2 s-1")


def test_sensitivity_get_invalid_mode():
    """Test that error is raised for invalid mode."""
    sensitivity_curve = np.logspace(-10, -12, 10) * u.Unit("erg cm-2 s-1")
    photon_flux_curve = np.logspace(-9, -11, 10) * u.Unit("cm-2 s-1")

    sens = Sensitivity(
        observatory="cta_north",
        radius=3.0 * u.deg,
        min_energy=0.02 * u.TeV,
        max_energy=10.0 * u.TeV,
        n_sensitivity_points=10,
        sensitivity_curve=sensitivity_curve,
        photon_flux_curve=photon_flux_curve,
    )

    with pytest.raises(ValueError, match="mode must be"):
        sens.get(100.0 * u.s, mode="invalid")


def test_sensitivity_get_invalid_time():
    """Test that error is raised for invalid time unit."""
    sensitivity_curve = np.logspace(-10, -12, 10) * u.Unit("erg cm-2 s-1")
    photon_flux_curve = np.logspace(-9, -11, 10) * u.Unit("cm-2 s-1")

    sens = Sensitivity(
        observatory="cta_north",
        radius=3.0 * u.deg,
        min_energy=0.02 * u.TeV,
        max_energy=10.0 * u.TeV,
        n_sensitivity_points=10,
        sensitivity_curve=sensitivity_curve,
        photon_flux_curve=photon_flux_curve,
    )

    with pytest.raises(ValueError, match="t must be a time quantity"):
        sens.get(100.0 * u.m, mode="sensitivity")


def test_sensitivity_get_numeric_time():
    """Test that numeric time is converted to Quantity."""
    sensitivity_curve = np.logspace(-10, -12, 10) * u.Unit("erg cm-2 s-1")
    photon_flux_curve = np.logspace(-9, -11, 10) * u.Unit("cm-2 s-1")

    sens = Sensitivity(
        observatory="cta_north",
        radius=3.0 * u.deg,
        min_energy=0.02 * u.TeV,
        max_energy=10.0 * u.TeV,
        n_sensitivity_points=10,
        sensitivity_curve=sensitivity_curve,
        photon_flux_curve=photon_flux_curve,
    )

    # Should work with numeric time
    result = sens.get(100.0, mode="sensitivity")
    assert isinstance(result, u.Quantity)


def test_sensitivity_get_missing_curve():
    """Test that error is raised when trying to get sensitivity without curve."""
    photon_flux_curve = np.logspace(-9, -11, 10) * u.Unit("cm-2 s-1")

    # Create Sensitivity with only photon_flux_curve, not sensitivity_curve
    sens = Sensitivity(
        observatory="cta_north",
        radius=3.0 * u.deg,
        min_energy=0.02 * u.TeV,
        max_energy=10.0 * u.TeV,
        n_sensitivity_points=10,
        sensitivity_curve=None,  # Explicitly No
        photon_flux_curve=photon_flux_curve,
    )

    with pytest.raises(ValueError, match="Sensitivity curve not yet calculated"):
        sens.get(100.0 * u.s, mode="sensitivity")

    res = sens.get(100.0 * u.s, mode="photon_flux")
    assert isinstance(res, u.Quantity)
    assert res.unit == u.Unit("cm-2 s-1")
    assert np.isfinite(res.value)
    assert res.value > 0
    assert res.value < 1


def test_sensitivity_sensitivity_curve_property():
    """Test Sensitivity sensitivity_curve property."""
    sensitivity_curve = np.logspace(-10, -12, 10) * u.Unit("erg cm-2 s-1")
    photon_flux_curve = np.logspace(-9, -11, 10) * u.Unit("cm-2 s-1")

    sens = Sensitivity(
        observatory="cta_north",
        radius=3.0 * u.deg,
        min_energy=0.02 * u.TeV,
        max_energy=10.0 * u.TeV,
        n_sensitivity_points=10,
        sensitivity_curve=sensitivity_curve,
        photon_flux_curve=photon_flux_curve,
    )

    assert len(sens.sensitivity_curve) == 10
    assert sens.sensitivity_curve[0].unit == u.Unit("erg cm-2 s-1")


def test_sensitivity_photon_flux_curve_property():
    """Test Sensitivity photon_flux_curve property."""
    sensitivity_curve = np.logspace(-10, -12, 10) * u.Unit("erg cm-2 s-1")
    photon_flux_curve = np.logspace(-9, -11, 10) * u.Unit("cm-2 s-1")

    sens = Sensitivity(
        observatory="cta_north",
        radius=3.0 * u.deg,
        min_energy=0.02 * u.TeV,
        max_energy=10.0 * u.TeV,
        n_sensitivity_points=10,
        sensitivity_curve=sensitivity_curve,
        photon_flux_curve=photon_flux_curve,
    )

    assert len(sens.photon_flux_curve) == 10
    assert sens.photon_flux_curve[0].unit == u.Unit("cm-2 s-1")


def test_sensitivity_table_property_empty():
    """Test Sensitivity.table property when empty."""
    sensitivity_curve = np.logspace(-10, -12, 10) * u.Unit("erg cm-2 s-1")
    photon_flux_curve = np.logspace(-9, -11, 10) * u.Unit("cm-2 s-1")

    sens = Sensitivity(
        observatory="cta_north",
        radius=3.0 * u.deg,
        min_energy=0.02 * u.TeV,
        max_energy=10.0 * u.TeV,
        n_sensitivity_points=10,
        sensitivity_curve=sensitivity_curve,
        photon_flux_curve=photon_flux_curve,
    )

    assert sens.table() is None


def test_sensitivity_pandas_property_empty():
    """Test Sensitivity.pandas property when empty."""
    sensitivity_curve = np.logspace(-10, -12, 10) * u.Unit("erg cm-2 s-1")
    photon_flux_curve = np.logspace(-9, -11, 10) * u.Unit("cm-2 s-1")

    sens = Sensitivity(
        observatory="cta_north",
        radius=3.0 * u.deg,
        min_energy=0.02 * u.TeV,
        max_energy=10.0 * u.TeV,
        n_sensitivity_points=10,
        sensitivity_curve=sensitivity_curve,
        photon_flux_curve=photon_flux_curve,
    )

    assert sens.pandas() is None


def test_sensitivity_extrapolation():
    """Test that Sensitivity.get extrapolates beyond curve bounds."""
    sensitivity_curve = np.logspace(-10, -12, 10) * u.Unit("erg cm-2 s-1")
    photon_flux_curve = np.logspace(-9, -11, 10) * u.Unit("cm-2 s-1")

    sens = Sensitivity(
        observatory="cta_north",
        radius=3.0 * u.deg,
        min_energy=0.02 * u.TeV,
        max_energy=10.0 * u.TeV,
        n_sensitivity_points=10,
        sensitivity_curve=sensitivity_curve,
        photon_flux_curve=photon_flux_curve,
    )

    # Test extrapolation beyond max time
    result = sens.get(1e5 * u.s, mode="sensitivity")
    assert isinstance(result, u.Quantity)

    # Test extrapolation below min time
    result = sens.get(1.0 * u.s, mode="sensitivity")
    assert isinstance(result, u.Quantity)


@pytest.mark.slow
def test_sensitivity_get_sensitivity_curve(irf_house, mock_csv_path):
    """Test sensitivity curve generation with IRF and source.
    
    This test matches the usage pattern from quick-test.ipynb (lines 1-2).
    """
    from gravitational_wave_toy.source import Source

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
        n_sensitivity_points=4,
    )

    # Load in a GRB and add EBL (matching quick-test.ipynb)
    grb = Source(mock_csv_path, min_energy=min_energy, max_energy=max_energy, ebl="franceschini")

    # Load the sensitivity curve for the GRB (matching quick-test.ipynb line 1-2)
    sens.get_sensitivity_curve(source=grb)

    # Verify sensitivity_curve is populated
    assert len(sens.sensitivity_curve) > 0
    assert sens.sensitivity_curve.unit == u.Unit("erg cm-2 s-1")
    assert all(sens.sensitivity_curve.value > 0)

    # Verify photon_flux_curve is populated
    assert len(sens.photon_flux_curve) > 0
    assert sens.photon_flux_curve.unit == u.Unit("cm-2 s-1")
    assert all(sens.photon_flux_curve.value > 0)

    # Verify curves have same length as times
    assert len(sens.sensitivity_curve) == len(sens.times)
    assert len(sens.photon_flux_curve) == len(sens.times)
