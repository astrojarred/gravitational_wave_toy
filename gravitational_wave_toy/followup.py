import warnings
from pathlib import Path
from typing import Literal

import pandas as pd
from astropy import units as u
from numpy import log10
from scipy.interpolate import interp1d

from . import sensitivity, source


def get_row(
    sens_df: pd.DataFrame,
    event_id: int,
    site: str,
    zenith: int,
    ebl: bool = False,
    config: str = "alpha",
    duration: int = 1800,
    event_id_column: str = "coinc_event_id",
):
    """Retrieve a sensitivity row from the dataframe matching the specified criteria.

    Searches the sensitivity dataframe for a row that matches all of the provided
    parameters. If multiple rows match, returns the first one. Raises an error if
    no matching row is found.

    Args:
        sens_df: DataFrame containing sensitivity data with columns for event ID,
            site, zenith angle, EBL flag, configuration, and duration.
        event_id: The event identifier to search for.
        site: Observatory site name (e.g., "north" or "south").
        zenith: Zenith angle in degrees for the observation.
        ebl: Whether EBL absorption is applied. Defaults to False.
        config: IRF configuration name. Defaults to "alpha".
        duration: Observation duration in seconds. Defaults to 1800.
        event_id_column: Name of the column containing event IDs. Defaults to
            "coinc_event_id".

    Returns:
        pandas.Series: The first matching row from the dataframe.

    Raises:
        ValueError: If no row matches all the specified criteria.
    """
    rows = sens_df[
        (sens_df[event_id_column] == event_id)
        & (sens_df["irf_site"] == site)
        & (sens_df["irf_zenith"] == zenith)
        & (sens_df["irf_ebl"] == ebl)
        & (sens_df["irf_config"] == config)
        & (sens_df["irf_duration"] == duration)
    ]

    if len(rows) < 1:
        raise ValueError("No sensitivity found with these values.")
    if len(rows) > 1:
        # print(
        #     f"Warning: multiple ({len(rows)}) sensitivities found with these values. Will use first row."
        # )
        pass

    return rows.iloc[0]


def extrapolate_obs_time(
    event_id: int,
    delay: u.Quantity,
    extrapolation_df: pd.DataFrame,
    filters: dict[str, str | float | int] = {},
    other_info: list[str] = [],
    event_id_column: str = "coinc_event_id",
):
    """Estimate the required observation time for a given delay using interpolation.

    Uses logarithmic interpolation to estimate the observation time needed to detect
    an event at a specific delay time. The function looks up pre-computed observation
    times from the extrapolation dataframe and interpolates between them. If the delay
    exceeds the maximum value in the dataframe, a warning is issued and the value is
    extrapolated beyond the data range.

    Args:
        event_id: The event identifier to look up.
        delay: Time delay from the event trigger, as an astropy Quantity with time units.
        extrapolation_df: DataFrame containing pre-computed observation times at various
            delays. Must contain columns for event ID, observation delay, and observation
            time, along with any filter columns.
        filters: Dictionary of additional column-value pairs to filter the dataframe.
            Keys should be column names, values should be the values to match.
        other_info: List of column names to include in the returned dictionary.
            These are extracted from the first matching row.
        event_id_column: Name of the column containing event IDs. Defaults to
            "coinc_event_id".

    Returns:
        dict: Dictionary containing:
            - "obs_time": Estimated observation time in seconds, or -1 if the event
                is not detectable or extrapolation fails.
            - "error_message": Empty string if successful, otherwise an error description.
            - Additional keys from other_info if provided.

    Raises:
        ValueError: If the requested delay is below the minimum delay available in
            the dataframe for this event.
    """
    res = {}
    delay = delay.to("s").value
    event_info = extrapolation_df[extrapolation_df[event_id_column] == event_id]

    if filters:
        for key, value in filters.items():
            event_info = event_info[event_info[key] == value]

    if other_info:
        for key in other_info:
            res[key] = event_info.iloc[0][key]

    event_dict = event_info.set_index("obs_delay")["obs_time"].to_dict()

    if delay < min(event_dict.keys()):
        res["error_message"] = (
            f"Minimum delay is {min(event_dict.keys())} seconds for this simulation"
        )
        res["obs_time"] = -1
        raise ValueError(
            f"Minimum delay is {min(event_dict.keys())} seconds for this simulation [{delay}s requested]"
        )
    elif delay > max(event_dict.keys()):
        print(
            f"Warning: delay is greater than maximum delay of {max(event_dict.keys())}s for this simulation [{delay}s requested], value will be extrapolated."
        )

    # remove negative values
    pos_event_dict = {k: v for k, v in event_dict.items() if v > 0}

    if not pos_event_dict:
        res["error_message"] = (
            f"Event is never detectable under the observation conditions {filters}"
        )
        res["obs_time"] = -1
        return res

    pairs = sorted((log10(k), log10(v)) for k, v in pos_event_dict.items())
    xs, ys = zip(*pairs)  # safe since not empty

    # perform log interpolation
    interp = interp1d(xs, ys, kind="linear", bounds_error=True)

    try:
        res["obs_time"] = 10 ** interp(log10(delay))
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
    sensitivity_curve: list[float] | None = None,
    photon_flux_curve: list[float] | None = None,
    ebl: bool = False,
    config: str = "alpha",
    duration: int = 1800,
    radius: u.Quantity = 3.0 * u.deg,
    min_energy: u.Quantity = 0.02 * u.TeV,
    max_energy: u.Quantity = 10 * u.TeV,
    event_id_column: str = "coinc_event_id",
):
    """Create a Sensitivity object for a given event and observation configuration.

    Constructs a Sensitivity instance either by looking up pre-computed sensitivity
    curves from a dataframe or by using directly provided sensitivity and photon flux
    curves. The sensitivity object is configured for the specified CTA observatory
    site, energy range, and observation region.

    Args:
        event_id: The event identifier. Only used when sens_df is provided.
        site: Observatory site name (e.g., "north" or "south").
        zenith: Zenith angle in degrees for the observation.
        sens_df: Optional DataFrame containing pre-computed sensitivity data. If provided,
            sensitivity_curve and photon_flux_curve must be None.
        sensitivity_curve: Optional list of sensitivity values in erg cm⁻² s⁻¹. Must be
            provided along with photon_flux_curve if sens_df is None.
        photon_flux_curve: Optional list of photon flux values in cm⁻² s⁻¹. Must be
            provided along with sensitivity_curve if sens_df is None.
        ebl: Whether EBL absorption is applied. Defaults to False.
        config: IRF configuration name. Defaults to "alpha".
        duration: Observation duration in seconds. Defaults to 1800.
        radius: Angular radius of the observation region. Defaults to 3.0 degrees.
        min_energy: Minimum energy for the sensitivity calculation. Defaults to 0.02 TeV.
        max_energy: Maximum energy for the sensitivity calculation. Defaults to 10 TeV.
        event_id_column: Name of the column containing event IDs in sens_df. Defaults to
            "coinc_event_id".

    Returns:
        Sensitivity: A configured Sensitivity object ready for use in exposure calculations.

    Raises:
        ValueError: If both sens_df and curves are provided, or if neither sens_df nor
            both curves are provided, or if sensitivity_curve is not a list or Quantity.
    """
    if sens_df is not None:
        if sensitivity_curve is not None or photon_flux_curve is not None:
            raise ValueError(
                "If sens_df is provided, sensitivity_curve and photon_flux_curve must both be None."
            )
    else:
        if sensitivity_curve is None or photon_flux_curve is None:
            raise ValueError(
                "Must provide either sens_df or both sensitivity_curve and photon_flux_curve"
            )

    if sens_df is not None:
        row = get_row(
            sens_df=sens_df,
            event_id=event_id,
            site=site,
            zenith=zenith,
            ebl=ebl,
            config=config,
            duration=duration,
            event_id_column=event_id_column,
        )

        sensitivity_curve = row["sensitivity_curve"]
        photon_flux_curve = row["photon_flux_curve"]

    if isinstance(sensitivity_curve, (list, u.Quantity)):
        n_sensitivity_points = len(sensitivity_curve)
    else:
        raise ValueError(
            f"sensitivity_curve must be a list or u.Quantity, got {type(sensitivity_curve)}"
        )

    sens = sensitivity.Sensitivity(
        observatory=f"cta_{site}",
        radius=radius,
        min_energy=min_energy,
        max_energy=max_energy,
        n_sensitivity_points=n_sensitivity_points,
        sensitivity_curve=sensitivity_curve * u.Unit("erg cm-2 s-1"),
        photon_flux_curve=photon_flux_curve * u.Unit("cm-2 s-1"),
    )

    return sens


def get_exposure(
    event_id: int,
    delay: u.Quantity,
    site: str,
    zenith: int,
    grb_filepath: Path | str | None = None,
    sens_df: pd.DataFrame | None = None,
    event_id_column: str = "coinc_event_id",
    sensitivity_curve: list | None = None,
    photon_flux_curve: list | None = None,
    extrapolation_df: pd.DataFrame | Path | str | None = None,
    ebl: str | None = None,
    redshift: float | None = None,
    config: str = "alpha",
    duration: int = 1800,
    radius: u.Quantity = 3.0 * u.deg,
    min_energy: u.Quantity = 0.02 * u.TeV,
    max_energy: u.Quantity = 10 * u.TeV,
    target_precision: u.Quantity = 1 * u.s,
    max_time: u.Quantity = 12 * u.h,
    sensitivity_mode: Literal["sensitivity", "photon_flux"] = "sensitivity",
    n_time_steps: int = 10,
):
    """Calculate exposure information for observing an event with a spectrum evolving in time.

    Determines the required observation time and other exposure parameters for detecting
    an event at a given delay. This function supports two modes of operation:

    1. **Extrapolation mode**: If extrapolation_df is provided, uses pre-computed
       observation times from simulations to quickly estimate the exposure time via
       interpolation. This is faster but requires pre-existing simulation data.

    2. **Direct calculation mode**: If grb_filepath is provided, loads the source
       spectrum and performs a full observation calculation using the Source.observe()
       method. This is more accurate but computationally intensive.

    Args:
        event_id: The event identifier.
        delay: Time delay from the event trigger, as an astropy Quantity with time units.
        site: Observatory site name (e.g., "north" or "south").
        zenith: Zenith angle in degrees for the observation.
        grb_filepath: Path to the GRB source file. Required if extrapolation_df is None.
        sens_df: Optional DataFrame containing pre-computed sensitivity data.
        event_id_column: Name of the column containing event IDs. Defaults to
            "coinc_event_id".
        sensitivity_curve: Optional list of sensitivity values. Must be provided with
            photon_flux_curve if sens_df is None.
        photon_flux_curve: Optional list of photon flux values. Must be provided with
            sensitivity_curve if sens_df is None.
        extrapolation_df: Optional DataFrame or path to parquet file containing
            pre-computed observation times. If provided, uses interpolation mode.
        ebl: Optional EBL model name to apply for absorption. If None, no EBL is applied.
        redshift: Optional redshift value. If provided, overrides the redshift from
            the source file for EBL calculations.
        config: IRF configuration name. Defaults to "alpha".
        duration: Observation duration in seconds for sensitivity lookup. Defaults to 1800.
        radius: Angular radius of the observation region. Defaults to 3.0 degrees.
        min_energy: Minimum energy for the calculation. Defaults to 0.02 TeV.
        max_energy: Maximum energy for the calculation. Defaults to 10 TeV.
        target_precision: Precision for rounding observation times. Defaults to 1 second.
        max_time: Maximum allowed observation time. Defaults to 12 hours.
        sensitivity_mode: Whether to use "sensitivity" or "photon_flux" for detection
            calculations. Defaults to "sensitivity".
        n_time_steps: Number of time steps for the observation calculation. Defaults to 10.

    Returns:
        dict: Dictionary containing exposure information. In extrapolation mode, includes:
            - "obs_time": Observation time in seconds (or -1 if not detectable)
            - "start_time": Start time of observation
            - "end_time": End time of observation (or -1 if not detectable)
            - "seen": Boolean indicating if the event is detectable
            - "id": Event identifier
            - "long", "lat": Source coordinates in radians
            - "eiso": Isotropic equivalent energy in erg
            - "dist": Distance in kpc
            - "angle": Viewing angle in degrees
            - "ebl_model": EBL model name used
            - "min_energy", "max_energy": Energy range
            - "error_message": Error description if applicable

        In direct calculation mode, returns the result from Source.observe().

    Raises:
        ValueError: If unit types are incorrect, if extrapolation_df is None and
            grb_filepath is not provided, or if delay is below the minimum in the
            extrapolation dataframe.
    """
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
        if isinstance(extrapolation_df, (Path, str)):
            extrapolation_df = pd.read_parquet(extrapolation_df)

        obs_info = extrapolate_obs_time(
            event_id=event_id,
            delay=delay,
            extrapolation_df=extrapolation_df,
            filters={"irf_site": site, "irf_zenith": zenith},
            other_info=["long", "lat", "eiso", "dist", "theta_view", "irf_ebl_model"],
            event_id_column=event_id_column,
        )

        obs_time = obs_info["obs_time"]
        if obs_time > 0:
            if obs_time > max_time.value:
                obs_info["error_message"] = (
                    f"Exposure time of {int(obs_time)} s exceeds maximum time"
                )
                obs_time = -1
            else:
                obs_time = round(obs_time / target_precision.value) * target_precision

        # rename key
        obs_info["angle"] = obs_info.pop("theta_view") * u.deg
        obs_info["ebl_model"] = obs_info.pop("irf_ebl_model")

        # add other units
        obs_info["long"] = obs_info["long"] * u.rad
        obs_info["lat"] = obs_info["lat"] * u.rad
        obs_info["eiso"] = obs_info["eiso"] * u.erg
        obs_info["dist"] = obs_info["dist"] * u.kpc

        other_info = {
            "min_energy": min_energy,
            "max_energy": max_energy,
            "seen": True if obs_time > 0 else False,
            "obs_time": obs_time if obs_time > 0 else -1,
            "start_time": delay,
            "end_time": delay + obs_time if obs_time > 0 else -1,
            "id": event_id,
        }

        return {**obs_info, **other_info}

    else:
        if not grb_filepath:
            raise ValueError(
                "Must provide grb_filepath if extrapolation_df is not provided"
            )

    sens = get_sensitivity(
        event_id=event_id,
        site=site,
        zenith=zenith,
        sens_df=sens_df,
        sensitivity_curve=sensitivity_curve,
        photon_flux_curve=photon_flux_curve,
        ebl=bool(ebl),
        config=config,
        duration=duration,
        radius=radius,
        min_energy=min_energy,
        max_energy=max_energy,
        event_id_column=event_id_column,
    )

    grb = source.Source(grb_filepath, min_energy, max_energy, ebl=ebl)

    if redshift is not None:
        grb.set_ebl_model(ebl, z=redshift)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"All-NaN slice encountered")
        result = grb.observe(
            sens,
            start_time=delay,
            min_energy=min_energy,
            max_energy=max_energy,
            target_precision=target_precision,
            max_time=max_time,
            sensitivity_mode=sensitivity_mode,
            n_time_steps=n_time_steps,
        )

    return result
