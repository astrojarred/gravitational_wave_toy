"""Tests for followup module."""
import astropy.units as u
import numpy as np
import pandas as pd
import pytest

from gravitational_wave_toy import followup


def test_get_row_found(sample_sensitivity_df):
    """Test get_row when row is found."""
    row = followup.get_row(
        sens_df=sample_sensitivity_df,
        event_id=42,
        site="north",
        zenith=20,
        ebl=False,
        config="alpha",
        duration=1800,
    )
    
    assert row["coinc_event_id"] == 42
    assert row["irf_site"] == "north"
    assert row["irf_zenith"] == 20


def test_get_row_not_found(sample_sensitivity_df):
    """Test get_row when no row is found."""
    with pytest.raises(ValueError, match="No sensitivity found"):
        followup.get_row(
            sens_df=sample_sensitivity_df,
            event_id=999,
            site="north",
            zenith=20,
            ebl=False,
            config="alpha",
            duration=1800,
        )


def test_get_row_multiple_matches(sample_sensitivity_df):
    """Test get_row when multiple rows match (should use first)."""
    # Add duplicate row
    duplicate_row = sample_sensitivity_df.iloc[0].copy()
    duplicate_df = pd.concat([sample_sensitivity_df, duplicate_row.to_frame().T], ignore_index=True)
    
    # Should not raise error, just use first match
    row = followup.get_row(
        sens_df=duplicate_df,
        event_id=42,
        site="north",
        zenith=20,
        ebl=False,
        config="alpha",
        duration=1800,
    )
    assert row is not None


def test_extrapolate_obs_time_valid(sample_extrapolation_df):
    """Test extrapolate_obs_time with valid delay."""
    result = followup.extrapolate_obs_time(
        event_id=42,
        delay=1000 * u.s,
        extrapolation_df=sample_extrapolation_df,
        filters={"irf_site": "north", "irf_zenith": 20},
    )
    
    assert "obs_time" in result
    assert result["obs_time"] > 0
    assert "error_message" in result


def test_extrapolate_obs_time_below_minimum(sample_extrapolation_df):
    """Test extrapolate_obs_time with delay below minimum."""
    with pytest.raises(ValueError, match="Minimum delay"):
        followup.extrapolate_obs_time(
            event_id=42,
            delay=10 * u.s,  # Below minimum of 100
            extrapolation_df=sample_extrapolation_df,
            filters={"irf_site": "north", "irf_zenith": 20},
        )


def test_extrapolate_obs_time_above_maximum(sample_extrapolation_df, capsys):
    """Test extrapolate_obs_time with delay above maximum (should warn)."""
    result = followup.extrapolate_obs_time(
        event_id=42,
        delay=200000 * u.s,  # Above maximum of 100000
        extrapolation_df=sample_extrapolation_df,
        filters={"irf_site": "north", "irf_zenith": 20},
    )
    
    # Should still return a result (extrapolated)
    assert "obs_time" in result
    # Check that warning was printed
    captured = capsys.readouterr()
    assert "Warning" in captured.out or "warning" in captured.out.lower()


def test_extrapolate_obs_time_other_info(sample_extrapolation_df):
    """Test extrapolate_obs_time with other_info parameter."""
    result = followup.extrapolate_obs_time(
        event_id=42,
        delay=1000 * u.s,
        extrapolation_df=sample_extrapolation_df,
        filters={"irf_site": "north", "irf_zenith": 20},
        other_info=["long", "lat", "eiso"],
    )
    
    assert "long" in result
    assert "lat" in result
    assert "eiso" in result


def test_extrapolate_obs_time_custom_event_id_column(sample_extrapolation_df):
    """Test extrapolate_obs_time with custom event_id_column."""
    # Rename column
    df = sample_extrapolation_df.rename(columns={"coinc_event_id": "event_id"})
    
    result = followup.extrapolate_obs_time(
        event_id=42,
        delay=1000 * u.s,
        extrapolation_df=df,
        event_id_column="event_id",
        filters={"irf_site": "north", "irf_zenith": 20},
    )
    
    assert "obs_time" in result


def test_get_sensitivity_from_sens_df(sample_sensitivity_df):
    """Test get_sensitivity using sens_df."""
    sens = followup.get_sensitivity(
        event_id=42,
        site="north",
        zenith=20,
        sens_df=sample_sensitivity_df,
        ebl=False,
        config="alpha",
        duration=1800,
    )
    
    assert sens is not None
    assert sens.observatory == "cta_north"
    assert len(sens.sensitivity_curve) > 0
    assert len(sens.photon_flux_curve) > 0


def test_get_sensitivity_from_curves():
    """Test get_sensitivity using sensitivity_curve and photon_flux_curve."""
    sensitivity_curve = np.logspace(-10, -12, 10) * u.Unit("erg cm-2 s-1")
    photon_flux_curve = np.logspace(-9, -11, 10) * u.Unit("cm-2 s-1")
    
    sens = followup.get_sensitivity(
        event_id=42,
        site="north",
        zenith=20,
        sensitivity_curve=sensitivity_curve.value.tolist(),
        photon_flux_curve=photon_flux_curve.value.tolist(),
        ebl=False,
        config="alpha",
        duration=1800,
    )
    
    assert sens is not None
    assert sens.observatory == "cta_north"


def test_get_sensitivity_conflicting_inputs(sample_sensitivity_df):
    """Test get_sensitivity raises error when both sens_df and curves are provided."""
    sensitivity_curve = np.logspace(-10, -12, 10) * u.Unit("erg cm-2 s-1")
    
    with pytest.raises(ValueError, match="If sens_df is provided"):
        followup.get_sensitivity(
            event_id=42,
            site="north",
            zenith=20,
            sens_df=sample_sensitivity_df,
            sensitivity_curve=sensitivity_curve.value.tolist(),
            ebl=False,
            config="alpha",
            duration=1800,
        )


def test_get_sensitivity_missing_inputs():
    """Test get_sensitivity raises error when neither sens_df nor curves are provided."""
    with pytest.raises(ValueError, match="Must provide either sens_df"):
        followup.get_sensitivity(
            event_id=42,
            site="north",
            zenith=20,
            ebl=False,
            config="alpha",
            duration=1800,
        )


def test_get_exposure_with_extrapolation_df(sample_extrapolation_df, mock_csv_path):
    """Test get_exposure with extrapolation_df."""
    result = followup.get_exposure(
        event_id=42,
        delay=1000 * u.s,
        site="north",
        zenith=20,
        extrapolation_df=sample_extrapolation_df,
        ebl=None,
    )
    
    assert "obs_time" in result
    assert "start_time" in result
    assert "seen" in result
    assert "id" in result
    assert result["id"] == 42


def test_get_exposure_without_extrapolation_df(mock_csv_path, sample_sensitivity_df):
    """Test get_exposure without extrapolation_df requires grb_filepath."""
    with pytest.raises(ValueError, match="Must provide grb_filepath"):
        followup.get_exposure(
            event_id=42,
            delay=1000 * u.s,
            site="north",
            zenith=20,
            sens_df=sample_sensitivity_df,
        )


def test_get_exposure_invalid_delay_unit(mock_csv_path, sample_extrapolation_df):
    """Test get_exposure raises error for invalid delay unit."""
    with pytest.raises(ValueError, match="delay must be a time quantity"):
        followup.get_exposure(
            event_id=42,
            delay=1000 * u.m,  # Wrong unit
            site="north",
            zenith=20,
            extrapolation_df=sample_extrapolation_df,
        )


def test_get_exposure_invalid_energy_units(mock_csv_path, sample_extrapolation_df):
    """Test get_exposure raises error for invalid energy units."""
    with pytest.raises(ValueError, match="min_energy must be an energy quantity"):
        followup.get_exposure(
            event_id=42,
            delay=1000 * u.s,
            site="north",
            zenith=20,
            extrapolation_df=sample_extrapolation_df,
            min_energy=1.0 * u.s,  # Wrong unit
        )


def test_get_exposure_invalid_radius_unit(mock_csv_path, sample_extrapolation_df):
    """Test get_exposure raises error for invalid radius unit."""
    with pytest.raises(ValueError, match="radius must be an angle quantity"):
        followup.get_exposure(
            event_id=42,
            delay=1000 * u.s,
            site="north",
            zenith=20,
            extrapolation_df=sample_extrapolation_df,
            radius=3.0 * u.s,  # Wrong unit
        )


def test_get_exposure_custom_parameters(mock_csv_path, sample_extrapolation_df):
    """Test get_exposure with custom parameters."""
    result = followup.get_exposure(
        event_id=42,
        delay=1000 * u.s,
        site="north",
        zenith=20,
        extrapolation_df=sample_extrapolation_df,
        radius=5.0 * u.deg,
        min_energy=0.1 * u.TeV,
        max_energy=5.0 * u.TeV,
        target_precision=10 * u.s,
        max_time=6 * u.h,
    )
    
    assert result is not None
    assert "min_energy" in result
    assert "max_energy" in result

