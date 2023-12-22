# Tools to ease followup with tilepy
from pathlib import Path

import pandas as pd
from astropy import units as u

from . import observe, sensitivity


def get_row(
    sens_df: pd.DataFrame,
    event_id: int,
    site: str,
    zenith: int,
    ebl: bool = False,
    software: str = "gammapy",
    config: str = "alpha",
    duration: int = 1800,
):

    # find row with these values
    rows = sens_df[
        (sens_df["coinc_event_id"] == event_id)
        & (sens_df["irf_site"] == site)
        & (sens_df["irf_zenith"] == zenith)
        & (sens_df["irf_ebl"] == ebl)
        & (sens_df["sensitivity_software"] == software)
        & (sens_df["irf_config"] == config)
        & (sens_df["irf_duration"] == duration)
    ]

    if len(rows) < 1:
        raise ValueError("No sensitivity found with these values.")
    if len(rows) > 1:
        print(
            f"Warning: multiple ({len(rows)}) sensitivities found with these values. Will use first row."
        )

    return rows.iloc[0]


def get_sensitivity(
    event_id: int,
    site: str,
    zenith: int,
    sens_df: pd.DataFrame | None = None,
    sens_curve: list | None = None,
    ebl: bool = False,
    software: str = "gammapy",
    config: str = "alpha",
    duration: int = 1800,
    radius: u.Quantity = 3.0 * u.deg,
    min_energy: u.Quantity = 0.03 * u.TeV,
    max_energy: u.Quantity = 10 * u.TeV,
):
    
    if sens_df is None and sens_curve is None:
        raise ValueError("Must provide either sens_df or sens_curve")
    if sens_df:

        row = get_row(
            sens_df=sens_df,
            event_id=event_id,
            site=site,
            zenith=zenith,
            ebl=ebl,
            software=software,
            config=config,
            duration=duration,
        )

        curve = row["sensitivity_curve"]
        
    else:
        curve = sens_curve

    if software == "gammapy":
        sens = sensitivity.SensitivityGammapy(
            observatory=f"cta_{site}",
            radius=radius,
            min_energy=min_energy,
            max_energy=max_energy,
            sensitivity_curve=curve * u.Unit("erg cm-2 s-1"),
        )
    else:
        sens = sensitivity.SensitivityCtools(
            min_energy=min_energy,
            max_energy=max_energy,
            regression=curve,
        )

    return sens


def get_exposure(
    grb_filepath: Path | str,
    event_id: int,
    delay: u.Quantity,
    site: str,
    zenith: int,
    sens_df: pd.DataFrame | None = None,
    sens_curve: list | None = None,
    ebl: str | None = None,
    software: str = "gammapy",
    config: str = "alpha",
    duration: int = 1800,
    radius: u.Quantity = 3.0 * u.deg,
    min_energy: u.Quantity = 0.03 * u.TeV,
    max_energy: u.Quantity = 10 * u.TeV,
    target_precision: u.Quantity = 1 * u.s,
    max_time: u.Quantity = 12 * u.h,
):
    # check delay units
    if delay.unit.physical_type != "time":
        raise ValueError(f"delay must be a time quantity, got {delay}")
    if min_energy.unit.physical_type != "energy":
        raise ValueError(f"min_energy must be an energy quantity, got {min_energy}")
    if max_energy.unit.physical_type != "energy":
        raise ValueError(f"max_energy must be an energy quantity, got {max_energy}")
    if radius.unit.physical_type != "angle":
        raise ValueError(f"radius must be an angle quantity, got {radius}")
    if target_precision.unit.physical_type != "time":
        raise ValueError(
            f"target_precision must be a time quantity, got {target_precision}"
        )
    if max_time.unit.physical_type != "time":
        raise ValueError(f"max_time must be a time quantity, got {max_time}")

    delay = delay.to("s")
    min_energy = min_energy.to("TeV")
    max_energy = max_energy.to("TeV")
    radius = radius.to("deg")
    target_precision = target_precision.to("s")
    max_time = max_time.to("s")

    sens = get_sensitivity(
        event_id=event_id,
        site=site,
        zenith=zenith,
        sens_df=sens_df,
        sens_curve=sens_curve,
        ebl=bool(ebl),
        software=software,
        config=config,
        duration=duration,
        radius=radius,
        min_energy=min_energy,
        max_energy=max_energy,
    )

    grb = observe.GRB(grb_filepath, min_energy, max_energy, ebl=ebl)

    result = grb.observe(
        sens,
        start_time=delay,
        min_energy=min_energy,
        max_energy=max_energy,
        target_precision=target_precision,
        max_time=max_time,
    )

    return result
