from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
import scipy
import scipy.stats
from astropy.coordinates import SkyCoord
from gammapy.data import Observation, observatory_locations
from gammapy.datasets import SpectrumDataset, SpectrumDatasetOnOff
from gammapy.estimators import SensitivityEstimator
from gammapy.irf import load_irf_dict_from_file
from gammapy.makers import SpectrumDatasetMaker
from gammapy.maps import MapAxis, RegionGeom
from gammapy.modeling.models import (
    PowerLawSpectralModel,
    SpectralModel,
)
from regions import CircleSkyRegion

from .logging import logger
from .util import suppress_warnings_and_logs

# Set up logging
log = logger(__name__)


class SensitivityCtools:
    def __init__(
        self,
        grbsens_file: str,
        min_energy: u.Quantity,
        max_energy: u.Quantity,
        mode: str = "photon_flux",
    ) -> None:
        if min_energy.unit.physical_type != "energy":
            raise ValueError(f"min_energy must be an energy quantity, got {min_energy}")
        if max_energy.unit.physical_type != "energy":
            raise ValueError(f"max_energy must be an energy quantity, got {max_energy}")

        if mode not in ["photon_flux", "sensitivity"]:
            raise ValueError(f"mode must be 'photon_flux' or 'sensitivity', got {mode}")

        self.mode = mode
        self.output = self.fit_grbsens(grbsens_file)
        self.energy_limits = (min_energy.to("GeV"), max_energy.to("GeV"))

    def parse_grbsens(
        self, grbsens_file, Separator="\t", names=None, orient="list"
    ) -> dict:
        as_dict = pd.read_csv(
            grbsens_file, sep=Separator, comment="#", names=names
        ).to_dict(orient=orient)

        return as_dict

    def open_grbsens(self, grbsens_file) -> dict:
        col_names = [
            "obs_time",
            "crab_flux",
            "photon_flux",
            "energy_flux",
            "sensitivity",
        ]
        sensi_list = self.parse_grbsens(grbsens_file, names=col_names, orient="list")

        return sensi_list

    def fit_grbsens(self, grbsens_file):
        grbsens = self.open_grbsens(grbsens_file)

        result = scipy.stats.linregress(
            np.log10(grbsens["obs_time"]), np.log10(grbsens[self.mode])
        )

        return result

    def get(self, t: u.Quantity):
        if not t.unit.physical_type == "time":
            raise ValueError(f"t must be a time quantity, got {t}")

        t = t.to("s")

        slope, intercept = (
            self.output.slope,
            self.output.intercept,
        )

        unit = (
            u.Unit("cm-2 s-1") if self.mode == "photon_flux" else u.Unit("erg cm-2 s-1")
        )

        return 10 ** (slope * np.log10(t.value) + intercept) * unit


class SensitivityGammapy:
    def __init__(
        self,
        irf: str | Path,
        observatory: str,
        radius: u.Quantity,
        min_energy: u.Quantity,
        max_energy: u.Quantity,
    ) -> None:
        # check that e_min and e_max are energy and convert to GeV
        if min_energy.unit.physical_type != "energy":
            raise ValueError(f"e_min must be an energy quantity, got {min_energy}")
        if max_energy.unit.physical_type != "energy":
            raise ValueError(f"e_max must be an energy quantity, got {max_energy}")
        if radius.unit.physical_type != "angle":
            raise ValueError(f"radius must be an angle quantity, got {radius}")

        self.irf = irf
        self.observatory = observatory
        self.radius = radius.to("deg")
        self.min_energy = min_energy.to("GeV")
        self.max_energy = max_energy.to("GeV")
        self.energy_limits = (min_energy.to("GeV"), max_energy.to("GeV"))
        self._last_table = None

    def get(
        self,
        t: u.Quantity,  # [s]
        index: float,
        amplitude: u.Quantity,  # [GeV-1 cm-2 s-1]
        reference: u.Quantity | None = None,  # [GeV]
        mode="photon_flux",  # "photon_flux" or "senstivity" or "table"
        **kwargs,
    ) -> float:
        if mode not in ["photon_flux", "sensitivity", "table"]:
            raise ValueError(f"mode must be 'photon_flux' or 'sensitivity', got {mode}")

        if t.unit.physical_type != "time":
            raise ValueError(f"t must be a time quantity, got {t}")
        t = t.to("s")

        try:
            amplitude = amplitude.to("GeV-1 cm-2 s-1")
        except u.UnitConversionError:
            raise ValueError(f"amplitude must be a flux quantity, got {amplitude}")
        if reference is None:
            reference = 1 * u.GeV

        t_model = PowerLawSpectralModel(
            index=-index,
            amplitude=amplitude,
            reference=reference,
        )

        with suppress_warnings_and_logs(logging_ok=True):
            sens_table = gamma_sens(
                irf=self.irf,
                observatory=self.observatory,
                duration=t,
                radius=self.radius,
                min_energy=self.min_energy,
                max_energy=self.max_energy,
                model=t_model,
                **kwargs,
            )

        self._last_table = sens_table

        e2dnde = sens_table["e2dnde"].data[0] * sens_table["e2dnde"].unit
        e2dnde = e2dnde.to("erg / (cm2 s)")

        energy = sens_table["energy"].data[0] * sens_table["energy"].unit
        energy = energy.to("GeV")

        if mode == "photon_flux":
            return e2dnde / energy
        elif mode == "table":
            return sens_table
        else:
            return e2dnde


def gamma_sens(
    irf: str | Path,
    observatory: str,
    duration: int,
    radius: float,
    min_energy: float,
    max_energy: float,
    model: SpectralModel | str | None = None,
    source_ra: float = 83.6331,
    source_dec: float = 22.0145,
    sigma: float = 5,
    bins: int = 1,
    offset: float = 0.0,
    acceptance: float = 1,
    acceptance_off: float = 5,
    bkg_syst_fraction: float = 0.05,
):
    """
    Calculate the integral sensitivity for a given set of parameters.

    Parameters
    ----------
    irf : str
        IRF to use
    observatory : str
        Observatory name [see `~gammapy.data.observatory_locations`]
    duration : int
        Observation duration in hours
    radius : float
        On region radius in degrees
    e_min : float
        Minimum energy in TeV
    e_max : float
        Maximum energy in TeV
    sigma : float
        Minimum significance
    bins : int
        Number of energy bins
    offset : float
        Offset in degrees
    acceptance : float
        On region acceptance
    acceptance_off : float
        Off region acceptance
    bkg_syst_fraction : float
        Fraction of background counts above which the number of gamma-rays is

    Returns
    -------
    sensitivity : `~astropy.units.Quantity`
        Integral sensitivity in units of cm^-2 s^-1
    """
    # check that IRF file exists
    irf = Path(irf)
    if not irf.exists():
        raise FileNotFoundError(f"IRF file not found: {irf}")

    # Define energy axis
    energy_axis = MapAxis.from_energy_bounds(
        min_energy, max_energy, bins, unit=u.TeV, name="energy"
    )

    energy_axis_true = MapAxis.from_energy_bounds(
        "0.01 TeV", "100 TeV", nbin=100, name="energy_true"
    )

    # Define region
    pointing = SkyCoord(source_ra, source_dec, unit="deg", frame="icrs")
    offset_pointing = SkyCoord(source_ra, source_dec + offset, unit="deg", frame="icrs")
    region = CircleSkyRegion(center=offset_pointing, radius=radius)
    geom = RegionGeom.create(region, axes=[energy_axis])

    # pointing = SkyCoord(0, 0, unit="deg", frame="icrs")
    # geom = RegionGeom.create(f"icrs;circle(0, {offset}, {radius})", axes=[energy_axis])

    # Define empty dataset
    empty_dataset = SpectrumDataset.create(geom=geom, energy_axis_true=energy_axis_true)

    # define power law dataset
    if model == "powerlaw":
        model = PowerLawSpectralModel(
            index=2.1, amplitude="5.7e-13 cm-2 s-1 TeV-1", reference="1 TeV"
        )

    # Load IRFs
    irfs = load_irf_dict_from_file(irf)

    # Define observation
    location = observatory_locations[observatory]
    # pointing = SkyCoord("0 deg", "0 deg")
    obs: Observation = Observation.create(
        pointing=pointing, irfs=irfs, livetime=duration, location=location
    )

    # Create dataset
    spectrum_maker = SpectrumDatasetMaker(selection=["exposure", "edisp", "background"])
    dataset = spectrum_maker.run(empty_dataset, obs)

    ## Correct for energy dependent region size
    # Define containment
    containment = 0.68

    # correct exposure
    dataset.exposure *= containment

    # Define on region radius
    on_radii = obs.psf.containment_radius(
        energy_true=energy_axis.center, offset=offset*u.deg, fraction=containment
    )

    factor = (1 - np.cos(on_radii)) / (1 - np.cos(geom.region.radius))
    dataset.background *= factor.value.reshape((-1, 1, 1))

    dataset_on_off = SpectrumDatasetOnOff.from_spectrum_dataset(
        dataset=dataset, acceptance=acceptance, acceptance_off=acceptance_off
    )

    ## Calculate sensitivity
    sensitivity_estimator = SensitivityEstimator(
        spectrum=model,
        gamma_min=5,
        n_sigma=sigma,
        bkg_syst_fraction=bkg_syst_fraction,
    )

    sensitivity_table = sensitivity_estimator.run(dataset_on_off)

    return sensitivity_table