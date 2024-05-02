import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import astropy.units as u
from astropy.io import fits
from astropy.table.table import Table
import numpy as np
import pandas as pd
import scipy
import scipy.stats
from astropy.coordinates import SkyCoord
from gammapy.data import (
    FixedPointingInfo,
    Observation,
    observatory_locations,
)
from gammapy.datasets import SpectrumDataset, SpectrumDatasetOnOff
from gammapy.estimators import SensitivityEstimator
from gammapy.irf import load_irf_dict_from_file
from gammapy.makers import SpectrumDatasetMaker, ReflectedRegionsBackgroundMaker, SafeMaskMaker, WobbleRegionsFinder
from gammapy.maps import MapAxis, RegionGeom
from gammapy.modeling.models import (
    EBLAbsorptionNormSpectralModel,
    PowerLawSpectralModel,
    SpectralModel,
)
from gammapy.modeling.models.spectral import EBL_DATA_BUILTIN
from regions import CircleSkyRegion, PointSkyRegion

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
        min_energy: u.Quantity | None = None,
        max_energy: u.Quantity | None = None,
        irf: str | Path | dict | None = None,
        min_time: u.Quantity = 1 * u.s,
        max_time: u.Quantity = 43200 * u.s,
        ebl: str | None = None,
        sensitivity_points: int = 16,
        sensitivity_curve: list | None = None,
    ) -> None:
        # check that e_min and e_max are energy and convert to GeV
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
        if not irf and (min_energy is None or max_energy is None):
            raise ValueError("Must provide irf or min_energy and max_energy")
        
        # get min and max energy if not provided
        if min_energy is None or max_energy is None:
            if not isinstance(irf, dict):
                with fits.open(irf) as irf_fits:
                    bins_lower = irf_fits[1].data["ENERG_LO"][0]
                    bins_upper = irf_fits[1].data["ENERG_HI"][0]
                if min_energy is None:
                    min_energy = bins_lower[0] * u.TeV
                if max_energy is None:
                    max_energy = bins_upper[-1] * u.TeV
            else:
                raise ValueError("Must provide min_energy and max_energy if irf is not a filepath.")
        
        if min_energy.unit.physical_type != "energy":
            raise ValueError(f"e_min must be an energy quantity, got {min_energy}")
        if max_energy.unit.physical_type != "energy":
            raise ValueError(f"e_max must be an energy quantity, got {max_energy}")

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
        self, grb: "GRB", sensitivity_points: int | None = None, offset: u.Quantity = 0.0 * u.deg, n_energy_bins: int | None = None, **kwargs
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
        
        if not n_energy_bins:
            n_energy_bins = int(np.log10(self.max_energy / self.min_energy) * 5)
        
        for t in times:
            s = self.get_sensitivity_from_model(
                t=t,
                index=grb.get_spectral_index(t),
                amplitude=grb.get_spectral_amplitude(t),
                reference=1 * u.GeV,
                redshift=grb.dist.z,
                mode="sensitivity",
                offset=offset,
                bins=n_energy_bins,
                return_type="energy_flux",
                **kwargs
            )

            sensitivity_curve.append(s)
            
        print(sensitivity_curve)

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
        mode: Literal["sensitivity", "table"] = "sensitivity",
        sensitivity_type: Literal["integral", "differential"] = "integral",
        return_type: Literal["e2dnde", "energy_flux", "photon_flux", "table", "all"] = "energy_flux",
        bins: int | None = 10,
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

        sens_result = self.estimate_sensitivity(
            irf=self.irf,
            observatory=self.observatory,
            duration=t,
            radius=self.radius,
            min_energy=self.min_energy,
            max_energy=self.max_energy,
            model=t_model,
            n_bins=bins,
            sensitivity_type=sensitivity_type,
            return_type=return_type,
            **kwargs,
        )

        return sens_result

    @staticmethod
    def estimate_sensitivity(
        irf: str | Path,
        observatory: str,
        duration: u.Quantity,
        radius: u.Quantity,
        min_energy: u.Quantity,
        max_energy: u.Quantity,
        model: SpectralModel,
        source_ra: float = 83.6331,
        source_dec: float = 22.0145,
        sigma: float = 5,
        n_bins: int | None = None,
        offset: u.Quantity = 0 * u.deg,
        gamma_min: int = 5,
        acceptance: float = 1,
        acceptance_off: float = 5,
        bkg_syst_fraction: float = 0.05,
        sensitivity_type: Literal["differential", "integral"] = "integral",
        return_type: Literal["e2dnde", "energy_flux", "photon_flux", "table", "all"] = "energy_flux",
    ) -> u.Quantity | Table:
        """
        Calculate the integral sensitivity for a given set of parameters.

        Parameters
        ----------
        irf : str or Path
            Path to the IRF file.
        observatory : str
            Name of the observatory.
        duration : `~astropy.units.Quantity`
            Observation duration.
        radius : `~astropy.units.Quantity`
            Radius of the region of interest.
        min_energy : `~astropy.units.Quantity`
            Minimum energy of the energy range.
        max_energy : `~astropy.units.Quantity`
            Maximum energy of the energy range.
        model : `~gammapy.modeling.models.SpectralModel`, str or None, optional
            Spectral model to use for sensitivity estimation.
        source_ra : float, optional
            Right ascension of the source in degrees.
        source_dec : float, optional
            Declination of the source in degrees.
        sigma : float, optional
            Significance level for sensitivity estimation.
        n_bins : int, optional
            Number of energy bins.
        offset : float, optional
            Offset of the source position from the pointing position in degrees.
        gamma_min : int, optional
            Minimum number of gamma-rays per bin
        acceptance : float, optional
            Acceptance for the on-region.
        acceptance_off : float, optional
            Acceptance for the off-region.
        bkg_syst_fraction : float, optional
            Fractional systematic uncertainty on the background estimation.

        Returns
        -------
        sensitivity : `~astropy.units.Quantity`
            Integral sensitivity in units of cm^-2 s^-1
        """
        # check that IRF file exists
        irf = Path(irf)
        
        # Load IRFs
        irfs = load_irf_dict_from_file(irf)

        # check units
        duration = duration.to("s")
        radius = radius.to("deg")
        min_energy = min_energy.to("TeV")
        max_energy = max_energy.to("TeV")
        offset = offset.to("deg")
        source_ra = source_ra * u.deg
        source_dec = source_dec * u.deg
        
        if not n_bins:
            # decide n_bins based on CTA's 5/decade rule
            n_bins = int(np.log10(max_energy / min_energy) * 5)
        
        # check combination of sensitivity type and return type
        if sensitivity_type == "differential":
            return_type = "table"
            
        if sensitivity_type not in ["differential", "integral"]:
            raise ValueError(f"sensitivity_type must be 'differential' or 'integral', got {sensitivity_type}")
        if return_type not in ["e2dnde", "energy_flux", "photon_flux", "table"]:
            raise ValueError(f"return_type must be 'e2dnde', 'energy_flux', 'photon_flux', or 'table', got {return_type}")
        
        # print(f"Calculating {sensitivity_type} sensitivity, returning {return_type}")
        
        energy_axis = MapAxis.from_energy_bounds(min_energy, max_energy, nbin=n_bins)
        energy_axis_true = MapAxis.from_energy_bounds(
            0.01 * u.TeV, 100 * u.TeV, nbin=100, name="energy_true"
        )

        pointing = SkyCoord(ra=source_ra, dec=source_dec)
        pointing_info = FixedPointingInfo(fixed_icrs=pointing)

        source_position = pointing.directional_offset_by(0 * u.deg, offset)
        on_region_radius = 0.1 * u.deg
        on_region = CircleSkyRegion(source_position, radius=on_region_radius)

        geom = RegionGeom.create(on_region, axes=[energy_axis])

        # extract 1D IRFs
        location = observatory_locations[observatory]
        obs = Observation.create(
            pointing=pointing_info, irfs=irfs, livetime=duration, location=location
        )
        
        empty_dataset = SpectrumDataset.create(geom=geom, energy_axis_true=energy_axis_true)
        
        
        # create a bkg model if not included
        if irfs.get("bkg") is not None:
            spectrum_maker = SpectrumDatasetMaker(
                selection=["exposure", "edisp", "background"],
                containment_correction=False,
            )
            dataset = spectrum_maker.run(empty_dataset, obs)
            
            containment = 0.68

            # correct exposure
            dataset.exposure *= containment

            # correct background estimation
            on_radii = obs.psf.containment_radius(
                energy_true=energy_axis.center, offset=offset, fraction=containment
            )
            factor = (1 - np.cos(on_radii)) / (1 - np.cos(geom.region.radius))
            dataset.background *= factor.value.reshape((-1, 1, 1))
            
            dataset_on_off = SpectrumDatasetOnOff.from_spectrum_dataset(
                dataset=dataset, acceptance=acceptance, acceptance_off=acceptance_off
            )
        else:
            spectrum_maker = SpectrumDatasetMaker(
                containment_correction=False, selection=["counts", "exposure", "edisp"]
            )
            # we need a RegionsFinder to find the OFF regions
            # and a BackgroundMaker to fill the array of the OFF counts
            region_finder = WobbleRegionsFinder(n_off_regions=1)
            bkg_maker = ReflectedRegionsBackgroundMaker(region_finder=region_finder)

            dataset = spectrum_maker.run(
                empty_dataset.copy(name=str(irf["obs"].obs_id)), irf["obs"]
            )
            # fill the OFF counts
            dataset_on_off = bkg_maker.run(dataset, irf["obs"])
        
        sensitivity_estimator = SensitivityEstimator(
            spectrum=model, gamma_min=gamma_min, n_sigma=sigma, bkg_syst_fraction=bkg_syst_fraction
        )
        
        if sensitivity_type == "integral" and return_type in ["e2dnde", "table"]:
        
            dataset_on_off_image = dataset_on_off.to_image()
            
            # get the integral flux sensitivity
            sensitivity_table = sensitivity_estimator.run(dataset_on_off_image)
            
            if return_type == "table":
                return sensitivity_table
            
            # get e2dnde
            e2dnde = sensitivity_table["e2dnde"].data[0] * sensitivity_table["e2dnde"].unit
            e2dnde = e2dnde.to("erg / (cm2 s)")
            
            return e2dnde
        
        else:
        
            sensitivity_table = sensitivity_estimator.run(dataset_on_off)
            
            if sensitivity_type == "differential":
                return sensitivity_table
            
            # integrate differential sensitivity
            e2dnde = (np.array(sensitivity_table['e2dnde'].tolist()) * sensitivity_table['e2dnde'].unit).to("erg / (cm2 s)")
            E_ref = (np.array(sensitivity_table['e_ref'].tolist()) * sensitivity_table['e_ref'].unit).to("erg")
            E_min_diff = (np.array(sensitivity_table['e_min'].tolist()) * sensitivity_table['e_min'].unit).to("erg")
            E_max_diff = (np.array(sensitivity_table['e_max'].tolist()) * sensitivity_table['e_max'].unit).to("erg")
            
            # calculate integral sensitivity
            integral_sensitivity = (e2dnde * (1/E_ref) * (E_max_diff - E_min_diff)).sum().to("erg / (cm2 s)")

            if return_type == "energy_flux":
                return integral_sensitivity
            
            photon_flux = (e2dnde * (1/(E_ref**2)) * (E_max_diff - E_min_diff)).sum() * u.Unit("1 / (cm2 s)")
            
            if return_type == "photon_flux":
                return photon_flux
            
            return integral_sensitivity, photon_flux
                        
            