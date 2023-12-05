import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import astropy.units as u
import numpy as np
import pandas as pd
import scipy
import scipy.stats
from astropy.coordinates import SkyCoord
from gammapy.data import (
    FixedPointingInfo,
    Observation,
    PointingMode,
    observatory_locations,
)
from gammapy.datasets import SpectrumDataset, SpectrumDatasetOnOff
from gammapy.estimators import SensitivityEstimator
from gammapy.irf import load_irf_dict_from_file
from gammapy.makers import SpectrumDatasetMaker
from gammapy.maps import MapAxis, RegionGeom
from gammapy.modeling.models import (
    EBLAbsorptionNormSpectralModel,
    PowerLawSpectralModel,
    SpectralModel,
)
from gammapy.modeling.models.spectral import EBL_DATA_BUILTIN
from regions import CircleSkyRegion

from .logging import logger

# Set up logging
log = logger(__name__)

# import types
if TYPE_CHECKING:
    from .observe import GRB


class SensitivityCtools:
    def __init__(
        self,
        min_energy: u.Quantity,
        max_energy: u.Quantity,
        grbsens_file: str | Path | None = None,
        mode: str = "photon_flux",
        regression: list | None = None,
    ) -> None:
        if grbsens_file is None and regression is None:
            raise ValueError("Must provide either grbsens_file or regression")

        if min_energy.unit.physical_type != "energy":
            raise ValueError(f"min_energy must be an energy quantity, got {min_energy}")
        if max_energy.unit.physical_type != "energy":
            raise ValueError(f"max_energy must be an energy quantity, got {max_energy}")

        if mode not in ["photon_flux", "sensitivity"]:
            raise ValueError(f"mode must be 'photon_flux' or 'sensitivity', got {mode}")

        self.mode = mode

        if regression is not None:
            if len(regression) != 4:
                raise ValueError("regression must be a list of length 4")
            self.output = {
                "slope": regression[0],
                "intercept": regression[1],
            }
            self.output_sensitivity = {
                "slope": regression[2],
                "intercept": regression[3],
            }
        else:
            self.output, self.output_sensitivity = self.fit_grbsens(grbsens_file)

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
            np.log10(grbsens["obs_time"]), np.log10(grbsens["photon_flux"])
        )

        result_sensitivity = scipy.stats.linregress(
            np.log10(grbsens["obs_time"]), np.log10(grbsens["sensitivity"])
        )

        return result, result_sensitivity

    def get(
        self,
        t: u.Quantity | float | int,
        mode: Literal["photon_flux", "sensitivity"] = None,
    ):
        if not mode:
            mode = self.mode

        if mode not in ["photon_flux", "sensitivity"]:
            raise ValueError(f"mode must be 'photon_flux' or 'sensitivity', got {mode}")

        if isinstance(t, (int, float)):
            t = t * u.s

        if not t.unit.physical_type == "time":
            raise ValueError(f"t must be a time quantity, got {t}")

        t = t.to("s")
        if isinstance(self.output, dict):
            if mode == "sensitivity":
                slope, intercept = (
                    self.output_sensitivity["slope"],
                    self.output_sensitivity["intercept"],
                )
            else:
                slope, intercept = self.output["slope"], self.output["intercept"]
        elif mode == "sensitivity":
            slope, intercept = (
                self.output_sensitivity.slope,
                self.output_sensitivity.intercept,
            )
        else:
            slope, intercept = self.output.slope, self.output.intercept

        unit = u.Unit("cm-2 s-1") if mode == "photon_flux" else u.Unit("erg cm-2 s-1")

        return 10 ** (slope * np.log10(t.value) + intercept) * unit


class SensitivityGammapy:
    def __init__(
        self,
        observatory: str,
        radius: u.Quantity,
        min_energy: u.Quantity,
        max_energy: u.Quantity,
        irf: str | Path | None = None,
        min_time: u.Quantity = 1 * u.s,
        max_time: u.Quantity = 43200 * u.s,
        ebl: str | None = None,
        sensitivity_points: int = 16,
        sensitivity_curve: list | None = None,
    ) -> None:
        # check that e_min and e_max are energy and convert to GeV
        if min_energy.unit.physical_type != "energy":
            raise ValueError(f"e_min must be an energy quantity, got {min_energy}")
        if max_energy.unit.physical_type != "energy":
            raise ValueError(f"e_max must be an energy quantity, got {max_energy}")
        if radius.unit.physical_type != "angle":
            raise ValueError(f"radius must be an angle quantity, got {radius}")
        if min_time.unit.physical_type != "time":
            raise ValueError(f"min_time must be a time quantity, got {min_time}")
        if max_time.unit.physical_type != "time":
            raise ValueError(f"max_time must be a time quantity, got {max_time}")
        if ebl is not None:
            if ebl not in list(EBL_DATA_BUILTIN.keys()):
                raise ValueError(
                    f"ebl must be one of {list(EBL_DATA_BUILTIN.keys())}, got {ebl}"
                )
            # check that environment variable is set
            if not os.environ.get("GAMMAPY_DATA"):
                raise ValueError(
                    "GAMMAPY_DATA environment variable not set. "
                    "Please set it to the path where the EBL data is stored. "
                    "You can copy EBL data from here: https://github.com/astrojarred/gravitational_wave_toy/tree/main/data"
                )

        if not irf and sensitivity_curve is None:
            raise ValueError("Must provide either irf or sensitivity_curve")

        self.irf = irf
        self.observatory = observatory
        self.radius = radius.to("deg")
        self.min_energy = min_energy.to("GeV")
        self.max_energy = max_energy.to("GeV")
        self.min_time = min_time.to("s")
        self.max_time = max_time.to("s")
        self.ebl = ebl
        self.times = (
            np.logspace(
                np.log10(self.min_time.value),
                np.log10(self.max_time.value),
                sensitivity_points,
            )
            * u.s
        )
        self.energy_limits = (min_energy.to("GeV"), max_energy.to("GeV"))
        self._last_table = None

        if sensitivity_curve is not None:
            self._sensitivity_curve = sensitivity_curve
            self._sensitivity_unit = sensitivity_curve[0].unit
            self._sensitivity = scipy.interpolate.interp1d(
                np.log10(self.times.value),
                np.log10(self._sensitivity_curve.value),
                kind="linear",
                fill_value="extrapolate",
            )
        else:
            self._sensitivity_curve = []
            self._sensitivity_unit = None
            self._sensitivity = None

    @property
    def sensitivity_curve(self):
        return self._sensitivity_curve

    def get(self, t: u.Quantity | int | float):
        if isinstance(t, (int, float)):
            t = t * u.s

        if not t.unit.physical_type == "time":
            raise ValueError(f"t must be a time quantity, got {t}")

        if not self._sensitivity:
            raise ValueError("Sensitivity curve not yet calculated for this GRB.")

        t = t.to("s")

        log_t = np.log10(t.value)
        log_sensitivity = self._sensitivity(log_t)

        return 10**log_sensitivity * self._sensitivity_unit

    def get_sensitivity_curve(
        self, grb: "GRB", sensitivity_points: int | None = None, offset: float = 0.0
    ):
        if not sensitivity_points:
            times = self.times
        else:
            times = (
                np.logspace(
                    np.log10(self.min_time.value),
                    np.log10(self.max_time.value),
                    sensitivity_points,
                )
                * u.s
            )
            self.times = times

        sensitivity_curve = []

        for t in times:
            s = self.get_sensitivity_from_model(
                t=t,
                index=grb.get_spectral_index(t),
                amplitude=grb.get_spectral_amplitude(t),
                reference=1 * u.GeV,
                redshift=grb.dist.z,
                mode="sensitivity",
                offset=offset,
            )

            sensitivity_curve.append(s)

        self._sensitivity_unit = sensitivity_curve[0].unit
        self._sensitivity_curve = (
            np.array([s.value for s in sensitivity_curve]) * self._sensitivity_unit
        )

        log_times = np.log10(times.value)
        log_sensitivity_curve = np.log10(self._sensitivity_curve.value)

        # interpolate sensitivity curve
        self._sensitivity = scipy.interpolate.interp1d(
            log_times, log_sensitivity_curve, kind="linear", fill_value="extrapolate"
        )

    def get_sensitivity_from_model(
        self,
        t: u.Quantity,  # [s]
        index: float,
        amplitude: u.Quantity,  # [GeV-1 cm-2 s-1]
        reference: u.Quantity | None = None,  # [GeV]
        redshift: float = 0.0,
        mode="sensitivity",  # "senstivity" or "table"
        **kwargs,
    ) -> float:
        if not self.irf:
            raise ValueError("Must provide irf to calculate sensitivity.")

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

        if self.ebl is not None:
            ebl_model = EBLAbsorptionNormSpectralModel.read_builtin(
                self.ebl,
                redshift=redshift,
            )

            t_model = t_model * ebl_model

        sens_table = self.estimate_sensitivity(
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

        if mode == "table":
            return sens_table
        else:
            return e2dnde

    @staticmethod
    def estimate_sensitivity(
        irf: str | Path,
        observatory: str,
        duration: u.Quantity,
        radius: u.Quantity,
        min_energy: u.Quantity,
        max_energy: u.Quantity,
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

        # check units
        if duration.unit.physical_type != "time":
            raise ValueError(f"duration must be a time quantity, got {duration}")
        else:
            duration = duration.to("s")
        if radius.unit.physical_type != "angle":
            raise ValueError(f"radius must be an angle quantity, got {radius}")
        else:
            radius = radius.to("deg")
        if min_energy.unit.physical_type != "energy":
            raise ValueError(f"min_energy must be an energy quantity, got {min_energy}")
        else:
            min_energy = min_energy.to("TeV")
        if max_energy.unit.physical_type != "energy":
            raise ValueError(f"max_energy must be an energy quantity, got {max_energy}")
        else:
            max_energy = max_energy.to("TeV")

        # Define energy axis
        energy_axis = MapAxis.from_energy_bounds(
            min_energy, max_energy, bins, unit=u.TeV, name="energy"
        )

        energy_axis_true = MapAxis.from_energy_bounds(
            "0.01 TeV", "100 TeV", nbin=100, name="energy_true"
        )

        # Define region
        fixed_icrs = SkyCoord(source_ra, source_dec, unit="deg", frame="icrs")
        pointing = FixedPointingInfo(fixed_icrs=fixed_icrs, mode=PointingMode.POINTING)
        offset_pointing = SkyCoord(
            source_ra, source_dec + offset, unit="deg", frame="icrs"
        )
        region = CircleSkyRegion(center=offset_pointing, radius=radius)
        geom = RegionGeom.create(region, axes=[energy_axis])

        # Define empty dataset
        empty_dataset = SpectrumDataset.create(
            geom=geom, energy_axis_true=energy_axis_true
        )

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
        spectrum_maker = SpectrumDatasetMaker(
            selection=["exposure", "edisp", "background"]
        )
        dataset = spectrum_maker.run(empty_dataset, obs)

        ## Correct for energy dependent region size
        # Define containment
        containment = 0.68

        # correct exposure
        dataset.exposure *= containment

        # Define on region radius
        on_radii = obs.psf.containment_radius(
            energy_true=energy_axis.center, offset=offset * u.deg, fraction=containment
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
