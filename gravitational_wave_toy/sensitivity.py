import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import astropy.units as u
import numpy as np
import pandas as pd
import scipy
import scipy.integrate
import scipy.interpolate
import scipy.stats
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table.table import Table
from gammapy.data import (
    FixedPointingInfo,
    Observation,
    observatory_locations,
)
from gammapy.datasets import SpectrumDataset, SpectrumDatasetOnOff
from gammapy.estimators import SensitivityEstimator, FluxPoints
from gammapy.irf import load_irf_dict_from_file
from gammapy.makers import (
    ReflectedRegionsBackgroundMaker,
    SpectrumDatasetMaker,
    WobbleRegionsFinder,
)
from gammapy.maps import MapAxis, RegionGeom
from gammapy.modeling.models import (
    CompoundSpectralModel,
    EBLAbsorptionNormSpectralModel,
    PowerLawSpectralModel,
    SpectralModel,
    SkyModel,
)
from gammapy.modeling.models.spectral import EBL_DATA_BUILTIN
from gammapy.utils.roots import find_roots
from gammapy.stats import WStatCountsStatistic
from regions import CircleSkyRegion

from .logging import logger

# Set up logging
log = logger(__name__)

# import types
if TYPE_CHECKING:
    from .observe import GRB


class Sensitivity:
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
        photon_flux_curve: list | None = None,
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
                raise ValueError(
                    "Must provide min_energy and max_energy if irf is not a filepath."
                )

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
            ).round()
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
            
        if photon_flux_curve is not None:
            self._photon_flux_curve = photon_flux_curve
            self._photon_flux_unit = photon_flux_curve[0].unit
            self._photon_flux = scipy.interpolate.interp1d(
                np.log10(self.times.value),
                np.log10(self._photon_flux_curve.value),
                kind="linear",
                fill_value="extrapolate"
            )
        else:
            self._photon_flux_curve = []
            self._photon_flux_unit = None
            self._photon_flux = None
            
        self._sensitivity_information = []

    @property
    def sensitivity_curve(self):
        return self._sensitivity_curve
    
    @property
    def photon_flux_curve(self):
        return self._photon_flux_curve
    
    def table(self):
        if not self._sensitivity_information:
            return None
        
        return Table(self._sensitivity_information)
    
    def pandas(self):
        if not self._sensitivity_information:
            return None
        
        return self.table().to_pandas()

    def get(self, t: u.Quantity | int | float, mode: Literal["sensitivity", "photon_flux"] = "sensitivity"):
        
        if mode not in ["sensitivity", "photon_flux"]:
            raise ValueError(f"mode must be 'sensitivity' or 'photon_flux', got {mode}")
        
        if isinstance(t, (int, float)):
            t = t * u.s

        if not t.unit.physical_type == "time":
            raise ValueError(f"t must be a time quantity, got {t}")

        t = t.to("s")

        log_t = np.log10(t.value)
        
        if mode == "sensitivity":
            if not self._sensitivity:
                raise ValueError("Sensitivity curve not yet calculated for this GRB.")
            log_sensitivity = self._sensitivity(log_t)
            return 10**log_sensitivity * self._sensitivity_unit
        elif mode == "photon_flux":
            if not self._photon_flux:
                raise ValueError("Photon flux curve not yet calculated for this GRB.")
            log_photon_flux = self._photon_flux(log_t)
            return 10**log_photon_flux * self._photon_flux_unit

    def get_sensitivity_curve(
        self,
        grb: "GRB",
        sensitivity_points: int | None = None,
        offset: u.Quantity = 0.0 * u.deg,
        n_bins: int | None = None,
        **kwargs,
    ):
        if not sensitivity_points:
            times = self.times
        else:
            times = round((
                np.logspace(
                    np.log10(self.min_time.value),
                    np.log10(self.max_time.value),
                    sensitivity_points,
                ).round()
                * u.s
            ))
            self.times = times

        sensitivity_curve = []
        photon_flux_curve = []

        if not n_bins:
            n_bins = int(np.log10(self.max_energy / self.min_energy) * 5)

        for t in times:
            s = self.get_sensitivity_from_model(
                t=t,
                index=grb.get_spectral_index(t),
                amplitude=grb.get_spectral_amplitude(t),
                reference=1 * u.GeV,
                redshift=grb.dist.z,
                sensitivity_type="integral",
                offset=offset,
                n_bins=n_bins,
                **kwargs,
            )

            self._sensitivity_information.append(s)
            sensitivity_curve.append(s["energy_flux"])
            photon_flux_curve.append(s["photon_flux"])
            

        self._sensitivity_unit = sensitivity_curve[0].unit
        self._sensitivity_curve = (
            np.array([s.value for s in sensitivity_curve]) * self._sensitivity_unit
        )
        self._photon_flux_unit = photon_flux_curve[0].unit
        self._photon_flux_curve = (
            np.array([s.value for s in photon_flux_curve]) * self._photon_flux_unit
        )

        log_times = np.log10(times.value)
        log_sensitivity_curve = np.log10(self._sensitivity_curve.value)
        log_photon_flux_curve = np.log10(self._photon_flux_curve.value)

        # interpolate sensitivity curve
        self._sensitivity = scipy.interpolate.interp1d(
            log_times, log_sensitivity_curve, kind="linear", fill_value="extrapolate"
        )
        self._photon_flux = scipy.interpolate.interp1d(
            log_times, log_photon_flux_curve, kind="linear", fill_value="extrapolate"
        )

    def get_sensitivity_from_model(
        self,
        t: u.Quantity,  # [s]
        index: float,
        amplitude: u.Quantity,  # [GeV-1 cm-2 s-1]
        reference: u.Quantity | None = None,  # [GeV]
        redshift: float = 0.0,
        sensitivity_type: Literal["integral", "differential"] = "integral",
        n_bins: int | None = None,
        **kwargs,
    ) -> float:
        if not self.irf:
            raise ValueError("Must provide irf to calculate sensitivity.")

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

        if sensitivity_type == "integral":
            sens_result = self.estimate_integral_sensitivity(
                irf=self.irf,
                observatory=self.observatory,
                duration=t,
                radius=self.radius,
                min_energy=self.min_energy,
                max_energy=self.max_energy,
                spectral_model=t_model,
                n_bins=n_bins,
                **kwargs,
            )
        else:
            sens_result = self.estimate_differential_sensitivity(
                irf=self.irf,
                observatory=self.observatory,
                duration=t,
                radius=self.radius,
                min_energy=self.min_energy,
                max_energy=self.max_energy,
                model=t_model,
                n_bins=n_bins,
                sensitivity_type=sensitivity_type,
                **kwargs,
            )

        return sens_result
    
    @staticmethod
    def simulate_spectrum(
        irf: str | Path,
        observatory: str,
        duration: u.Quantity,
        radius: u.Quantity,
        min_energy: u.Quantity,
        max_energy: u.Quantity,
        spectral_model: SpectralModel,
        source_ra: u.Quantity = 83.6331 * u.deg,
        source_dec: u.Quantity = 22.0145 * u.deg,
        n_bins: int | None = None,
        offset: u.Quantity = 0 * u.deg,
        acceptance: float = 1,
        acceptance_off: float = 3,
        random_state="random-seed"
    ):
        
        if not n_bins:
            n_bins = int(np.log10(max_energy / min_energy) * 5)
            
        
        energy_axis = MapAxis.from_energy_bounds(min_energy, max_energy, nbin=n_bins)
        energy_axis_true = MapAxis.from_energy_bounds(
            0.01 * u.TeV, 100 * u.TeV, nbin=100, name="energy_true"
        )

        pointing = SkyCoord(ra=source_ra, dec=source_dec)
        pointing_info = FixedPointingInfo(fixed_icrs=pointing)

        source_position = pointing.directional_offset_by(0 * u.deg, offset)
        on_region = CircleSkyRegion(source_position, radius=radius)

        # we set the sky model used in the dataset
        model = SkyModel(spectral_model=spectral_model, name="source")

        # extract 1D IRFs
        irfs = load_irf_dict_from_file(irf)

        # create observation
        location = observatory_locations[observatory]
        obs = Observation.create(
            pointing=pointing_info, irfs=irfs, livetime=duration, location=location
        )
        
        # Make the SpectrumDataset
        geom = RegionGeom.create(region=on_region, axes=[energy_axis])

        dataset_empty = SpectrumDataset.create(
            geom=geom, energy_axis_true=energy_axis_true, name="obs-0"
        )
        maker = SpectrumDatasetMaker(selection=["exposure", "edisp", "background"])

        dataset = maker.run(dataset_empty, obs)

        # Set the model on the dataset, and fake
        dataset.models = model
        dataset.fake(random_state=random_state)

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
        dataset_on_off.fake(npred_background=dataset.npred_background(), random_state=random_state)
        
        return dataset_on_off

    @staticmethod
    def get_ts_difference(norm: float, dataset: SpectrumDatasetOnOff, significance: float = 5):
        

        # dataset.models[0].spectral_model.amplitude.value = norm
        if isinstance(dataset.models[0]._spectral_model, PowerLawSpectralModel):
            dataset.models[0]._spectral_model.amplitude.value = norm
        else:
            dataset.models[0]._spectral_model.model1.amplitude.value = norm
                
        
        # TODO: Check that the inputs here are correct
        n_off = dataset.counts_off.data
        alpha = dataset.alpha.data
        n_pred = dataset.npred_signal().data
        n_on = n_pred + alpha * n_off
        
        # print(n_off.sum(), n_on.sum())
        
        stat = WStatCountsStatistic(
            n_on=n_on, 
            n_off=n_off, 
            alpha=alpha,
        )

        # TODO: how to take into account edisp?
        # i.e. correlation between different bins
        total_sqrt_ts = stat.sqrt_ts.sum()
        # print(norm, total_sqrt_ts, total_sqrt_ts - significance)

        # solve this equation to find normalization
        return total_sqrt_ts - significance

    @staticmethod
    def estimate_integral_sensitivity(  
        irf: str | Path,
        observatory: str,
        duration: u.Quantity,
        radius: u.Quantity,
        min_energy: u.Quantity,
        max_energy: u.Quantity,
        spectral_model: SpectralModel,
        n_iter: int = 10,
        significance: int | float = 5,
        upper_bound_ratio: int | float = 1e6,
        lower_bound_ratio: int | float = 1e-6,
        source_ra: u.Quantity = 83.6331 * u.deg,
        source_dec: u.Quantity = 22.0145 * u.deg,
        n_bins: int | None = None,
        offset: u.Quantity = 0 * u.deg,
        acceptance: int = 1,
        acceptance_off: int = 3,
        random_state="random-seed"
    ):
        """Compute excess matching a given significance.

        This function is the inverse of `significance`.

        Parameters
        ----------
        significance : float
            Significance.

        Returns
        -------
        n_sig : `numpy.ndarray`
            Excess.
        """
        
        roots = np.array([])
        
        for _ in range(n_iter):
        
            dataset = Sensitivity.simulate_spectrum(
                irf=irf,
                observatory=observatory,
                duration=duration,
                radius=radius,
                min_energy=min_energy,
                max_energy=max_energy,
                spectral_model=spectral_model,
                source_ra=source_ra,
                source_dec=source_dec,
                n_bins=n_bins,
                offset=offset,
                acceptance=acceptance,
                acceptance_off=acceptance_off,
                random_state=random_state,
            )
            
            # print(dataset.counts_off.data.sum(), dataset.npred_signal().data.sum())
            
            if isinstance(spectral_model, CompoundSpectralModel):
                original_norm= spectral_model.model1.amplitude.value
            else:
                original_norm = spectral_model.amplitude.value
                
            lower_bound = original_norm * lower_bound_ratio
            upper_bound = original_norm * upper_bound_ratio

            # find upper bounds for secant method as in scipy
            root, _res = find_roots(
                Sensitivity.get_ts_difference,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                points_scale="log",
                args=(dataset, significance),
                method="secant",
            )
            
            roots = np.append(roots, root)
            
        if isinstance(spectral_model, CompoundSpectralModel):
            spectral_model_unit = spectral_model.model1.amplitude.unit
        else:
            spectral_model_unit = spectral_model.amplitude.unit
        
        final_normalization = np.mean(roots) * spectral_model_unit
        final_normalization_err = np.std(roots) * spectral_model_unit
        
        final_spectrum = spectral_model.copy() 
        if isinstance(spectral_model, CompoundSpectralModel):
            final_spectrum.model1.amplitude.value = final_normalization.value
        else:
            final_spectrum.amplitude.value = final_normalization.value
        
        photon_flux = final_spectrum.integral(min_energy, max_energy)
        energy_flux = final_spectrum.energy_flux(min_energy, max_energy)
        
        # calculate errors
        final_spectrum_error = spectral_model.copy()
        if isinstance(final_spectrum_error, CompoundSpectralModel):
            final_spectrum_error.model1.amplitude.value = final_normalization.value + final_normalization_err.value
        else:
            final_spectrum_error.amplitude.value = final_normalization.value + final_normalization_err.value

        photon_flux_error = final_spectrum_error.integral(min_energy, max_energy) - photon_flux
        energy_flux_error = final_spectrum_error.energy_flux(min_energy, max_energy) - energy_flux
        
        return {
            "duration": duration,
            "normalization": final_normalization.to("erg-1 cm-2 s-1"),
            "normalization_err": final_normalization_err.to("erg-1 cm-2 s-1"),
            "photon_flux": photon_flux.to("cm-2 s-1"),
            "photon_flux_err": photon_flux_error.to("cm-2 s-1"),
            "energy_flux": energy_flux.to("erg cm-2 s-1"),
            "energy_flux_err": energy_flux_error.to("erg cm-2 s-1"),
        }
    

    @staticmethod
    def estimate_differential_sensitivity(
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

        energy_axis = MapAxis.from_energy_bounds(min_energy, max_energy, nbin=n_bins)
        energy_axis_true = MapAxis.from_energy_bounds(
            0.01 * u.TeV, 100 * u.TeV, nbin=100, name="energy_true"
        )

        pointing = SkyCoord(ra=source_ra, dec=source_dec)
        pointing_info = FixedPointingInfo(fixed_icrs=pointing)

        source_position = pointing.directional_offset_by(0 * u.deg, offset)
        on_region = CircleSkyRegion(source_position, radius=radius)

        geom = RegionGeom.create(on_region, axes=[energy_axis])

        # extract 1D IRFs
        location = observatory_locations[observatory]
        obs = Observation.create(
            pointing=pointing_info, irfs=irfs, livetime=duration, location=location
        )

        empty_dataset = SpectrumDataset.create(
            geom=geom, energy_axis_true=energy_axis_true
        )

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
            spectrum=model,
            gamma_min=gamma_min,
            n_sigma=sigma,
            bkg_syst_fraction=bkg_syst_fraction,
        )

        sensitivity_table = sensitivity_estimator.run(dataset_on_off)

        return sensitivity_table
        
