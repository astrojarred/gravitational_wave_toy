# Tools to ease followup with tilepy
from pathlib import Path

import pandas as pd
from astropy import units as u
from numpy import log10
from scipy.interpolate import interp1d

from . import observe, sensitivity


def get_row(
    sens_df: pd.DataFrame,
    event_id: int,
    site: str,
    zenith: int,
    ebl: bool = False,
    config: str = "alpha",
    duration: int = 1800,
):

    # find row with these values
    rows = sens_df[
        (sens_df["coinc_event_id"] == event_id)
        & (sens_df["irf_site"] == site)
        & (sens_df["irf_zenith"] == zenith)
        & (sens_df["irf_ebl"] == ebl)
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


def extrapolate_obs_time(
    event_id: int,
    delay: u.Quantity,
    extrapolation_df: pd.DataFrame,
    filters: dict[str, str] = {},
    other_info: list[str] = [],
):
    
    res = {}
    delay = delay.to("s").value
    event_info = extrapolation_df[extrapolation_df["coinc_event_id"] == event_id]
    
    if filters:
        for key, value in filters.items():
            event_info = event_info[event_info[key] == value]
            
    if other_info:
        for key in other_info:
            res[key] = event_info.iloc[0][key]
    
    event_dict = event_info.set_index('obs_delay')['obs_time'].to_dict() 
    
    if delay < min(event_dict.keys()):
        res["error_message"] = f"Minimum delay is {min(event_dict.keys())} seconds for this simulation"
        res["obs_time"] = -1

    # remove negative values
    pos_event_dict = {k: v for k, v in event_dict.items() if v > 0}
        
    # perform log interpolation
    log_event_dict = {log10(k): log10(v) for k, v in pos_event_dict.items()}

    interp = interp1d(list(log_event_dict.keys()), list(log_event_dict.values()), kind="linear")
    
    try:
        res["obs_time"] = 10**interp(log10(delay))
        res["error_message"] = ""
    except ValueError:
        res["obs_time"] = -1
        res["error_message"] = "Extrapolation failed for this simulation"
        
    return res

def get_sensitivity(
    event_id: int,
    site: str,
    zenith: int,
    sens_df: pd.DataFrame | None = None,
    sens_curve: list | None = None,
    ebl: bool = False,
    config: str = "alpha",
    duration: int = 1800,
    radius: u.Quantity = 3.0 * u.deg,
    min_energy: u.Quantity = 0.02 * u.TeV,
    max_energy: u.Quantity = 10 * u.TeV,
):
    
    if sens_df is None and sens_curve is None:
        raise ValueError("Must provide either sens_df or sens_curve")
    if sens_df is not None:

        row = get_row(
            sens_df=sens_df,
            event_id=event_id,
            site=site,
            zenith=zenith,
            ebl=ebl,
            config=config,
            duration=duration,
        )

        curve = row["sensitivity_curve"]
        
    else:
        curve = sens_curve

    sens = sensitivity.Sensitivity(
        observatory=f"cta_{site}",
        radius=radius,
        min_energy=min_energy,
        max_energy=max_energy,
        sensitivity_curve=curve * u.Unit("erg cm-2 s-1"),
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
    extrapolation_df: pd.DataFrame | None = None,
    ebl: str | None = None,
    config: str = "alpha",
    duration: int = 1800,
    radius: u.Quantity = 3.0 * u.deg,
    min_energy: u.Quantity = 0.02 * u.TeV,
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
    
    if extrapolation_df is not None:
        
        obs_info = extrapolate_obs_time(
            event_id=event_id,
            delay=delay,
            extrapolation_df=extrapolation_df,
            filters={"irf_site": site, "irf_zenith": zenith},
            other_info=["long", "lat", "eiso", "dist", "theta_view", "irf_ebl_model"]
        )
        
        obs_time = obs_info["obs_time"]
        if obs_time > 0:
            obs_time = round(obs_time / target_precision.value) * target_precision
        
        # rename key
        obs_info["angle"] = obs_info.pop("theta_view")
        obs_info["ebl_model"] = obs_info.pop("irf_ebl_model")
        
        other_info = {
            "filepath": grb_filepath.absolute(),
            "min_energy": min_energy,
            "max_energy": max_energy,
            "seen": True if obs_time > 0 else False,
            "obs_time": obs_time if obs_time > 0 else -1,
            "start_time": delay,
            "end_time": delay + obs_time if obs_time > 0 else -1,
            "id": event_id,
        }
        
        return {**obs_info, **other_info}

    sens = get_sensitivity(
        event_id=event_id,
        site=site,
        zenith=zenith,
        sens_df=sens_df,
        sens_curve=sens_curve,
        ebl=bool(ebl),
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
