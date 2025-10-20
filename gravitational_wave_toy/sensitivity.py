import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import astropy.units as u
import numpy as np
import scipy
import scipy.interpolate
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table.table import Table
from gammapy.data import (
    FixedPointingInfo,
    Observation,
    observatory_locations,
)
from gammapy.datasets import SpectrumDataset, SpectrumDatasetOnOff
from gammapy.estimators import SensitivityEstimator
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
    SkyModel,
    SpectralModel,
    TemplateSpectralModel,
)
from gammapy.modeling.models.spectral import EBL_DATA_BUILTIN
from gammapy.stats import WStatCountsStatistic
from gammapy.utils.roots import find_roots
from regions import CircleSkyRegion

from .logging import logger

# Set up logging
log = logger(__name__)

# import types
if TYPE_CHECKING:
    from .observe import GRB


class ScaledTemplateModel(TemplateSpectralModel):
    """Scaled template spectral model for sensitivity calculations."""

    def __init__(self, scaling_factor: int | float = 1e-6, *args, **kwargs):
        from gammapy.modeling import Parameter
        
        # Create a real amplitude parameter with log scaling (dimensionless)
        self.amplitude = Parameter(
            "amplitude", 
            scaling_factor, 
            unit="1 / (TeV s cm2)",  # Dimensionless scaling factor
            interp="log"
        )
        self.amplitude._is_norm = True
        
        self.scaling_factor = scaling_factor
        self._original_values = None  # Initialize before calling super
        super().__init__(*args, **kwargs)
        # When super().__init__ sets self.values, our setter stores it in _original_values

    @classmethod
    def from_template(
        cls, model: TemplateSpectralModel, scaling_factor: int | float = 1
    ):
        """
        Factory method to create a ScaledTemplateModel from an existing TemplateSpectralModel.
        """
        return cls(
            energy=model.energy, values=model.values, scaling_factor=scaling_factor
        )

    @property
    def values(self):
        return self._original_values * self.scaling_factor

    @values.setter
    def values(self, values: u.Quantity):
        self._original_values = values        


    def evaluate(self, energy):
        """Evaluate the model with scaling applied."""
        # Get the unscaled evaluation from parent class
        unscaled_result = super().evaluate(energy)
        # Apply dimensionless amplitude scaling
        return unscaled_result * self.amplitude.value

    def copy(self):
        """Create a copy of the ScaledTemplateModel with all attributes preserved."""
        # Create a new instance with the same parameters
        return ScaledTemplateModel(
            energy=self.energy,
            values=self._original_values,  # Use original values, not scaled
            scaling_factor=self.amplitude.value,
        )


def _get_model_normalization_info(spectral_model):
    """
    Extract normalization information from different spectral model types.

    Parameters
    ----------
    spectral_model : SpectralModel
        The spectral model to extract normalization from.

    Returns
    -------
    tuple
        (normalization_value, normalization_unit, model_copy_function)
    """
    if isinstance(spectral_model, CompoundSpectralModel):
        # For compound models, check if the first component has amplitude
        if hasattr(spectral_model.model1, "amplitude"):
            norm_value = spectral_model.model1.amplitude.value
            norm_unit = spectral_model.model1.amplitude.unit

            def copy_func(model, new_norm):
                model_copy = model.copy()
                model_copy.model1.amplitude.value = new_norm
                return model_copy

            return norm_value, norm_unit, copy_func
        else:
            # Fallback: use scaling factor if it's a ScaledTemplateModel
            if hasattr(spectral_model.model1, "scaling_factor"):
                norm_value = spectral_model.model1.scaling_factor
                norm_unit = u.dimensionless_unscaled

                def copy_func(model, new_norm):
                    model_copy = model.copy()
                    model_copy.model1.scaling_factor = new_norm
                    return model_copy

                return norm_value, norm_unit, copy_func

    elif hasattr(spectral_model, "amplitude"):
        # Standard models with amplitude parameter
        norm_value = spectral_model.amplitude.value
        norm_unit = spectral_model.amplitude.unit

        def copy_func(model, new_norm):
            model_copy = model.copy()
            model_copy.amplitude.value = new_norm
            return model_copy

        return norm_value, norm_unit, copy_func

    else:
        raise ValueError(f"Unsupported spectral model type: {type(spectral_model)}")


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
        n_sensitivity_points: int = 16,
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
                n_sensitivity_points,
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
                fill_value="extrapolate",
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

    def get(
        self,
        t: u.Quantity | int | float,
        mode: Literal["sensitivity", "photon_flux"] = "sensitivity",
    ):
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
        n_sensitivity_points: int | None = None,
        offset: u.Quantity = 0.0 * u.deg,
        n_bins: int | None = None,
        starting_amplitude: u.Quantity = 1e-12 * u.Unit("TeV-1 cm-2 s-1"),
        reference: u.Quantity = 1 * u.TeV,
        use_model: bool = True,
        **kwargs,
    ):
        if not n_sensitivity_points:
            times = self.times
        else:
            times = (
                np.logspace(
                    np.log10(self.min_time.value),
                    np.log10(self.max_time.value),
                    n_sensitivity_points,
                ).round()
                * u.s
            )
            self.times = times

        sensitivity_curve = []
        photon_flux_curve = []

        if not n_bins:
            n_bins = int(np.log10(self.max_energy / self.min_energy) * 5)

        if grb.file_type == "txt" and use_model:
            # raise a warning that we are using a power law model
            warnings.warn(
                "Using a power law model for sensitivity calculation. If you don't want to fit a power law, set use_model=False.",
                UserWarning,
            )

        for t in times:
            if use_model:
                s = self.get_sensitivity_from_model(
                    t=t,
                    spectral_model=PowerLawSpectralModel(
                        index=-grb.get_spectral_index(t),
                        amplitude=starting_amplitude,
                        reference=reference,
                    ),
                    redshift=grb.dist.z if grb.dist != 0 else 0,
                    sensitivity_type="integral",
                    offset=offset,
                    n_bins=n_bins,
                    **kwargs,
                )
            else:
                s = self.get_sensitivity_from_model(
                    t=t,
                    spectral_model=grb.get_template_spectrum(t),
                    redshift=grb.dist.z if grb.dist != 0 else 0,
                    sensitivity_type="integral",
                    offset=offset,
                    n_bins=n_bins,
                    **kwargs,
                )

            # print(f"Time: {t}")
            # print("Sensitivity:\n", s)
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
        spectral_model: SpectralModel,
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

        # Apply EBL absorption if specified
        if self.ebl is not None and redshift > 0:
            ebl_model = EBLAbsorptionNormSpectralModel.read_builtin(
                self.ebl,
                redshift=redshift,
            )
            t_model = spectral_model * ebl_model
        else:
            t_model = spectral_model

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
        random_state="random-seed",
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
        dataset_on_off.fake(
            npred_background=dataset.npred_background(), random_state=random_state
        )

        return dataset_on_off

    @staticmethod
    def get_ts_difference(
        norm: float, dataset: SpectrumDatasetOnOff, significance: float = 5
    ):
        # Update the spectral model normalization using our helper function
        spectral_model = dataset.models[0]._spectral_model
        _, _, copy_func = _get_model_normalization_info(spectral_model)

        # Create a new model with updated normalization and replace it
        updated_model = copy_func(spectral_model, norm)
        
        # Update the model properly through the Models API
        sky_model = SkyModel(spectral_model=updated_model, name=dataset.models[0].name)
        dataset.models = [sky_model]
        
        # if hasattr(updated_model, "amplitude"):
        #     print(f"    get_ts_difference: norm={norm}, updated_model.amplitude.value={updated_model.amplitude.value}")
        # else:
        #     print(f"    get_ts_difference: norm={norm}, updated_model.amplitude.value={updated_model.model1.amplitude.value}")

        
        # Test: Evaluate the model directly to see if it changes
        # test_energy = 1 * u.TeV
        # test_flux = updated_model(test_energy)
        # print(f"    get_ts_difference: model(1 TeV)={test_flux}")

        # TODO: Check that the inputs here are correct
        n_off = dataset.counts_off.data
        alpha = dataset.alpha.data
        n_pred = dataset.npred_signal().data
        n_on = n_pred + alpha * n_off

        # print(f"    get_ts_difference: n_off={n_off.sum()}, n_on={n_on.sum()}, n_pred={n_pred.sum()}")

        stat = WStatCountsStatistic(
            n_on=n_on,
            n_off=n_off,
            alpha=alpha,
        )

        # TODO: how to take into account edisp?
        # i.e. correlation between different bins
        total_sqrt_ts = stat.sqrt_ts.sum()
        
        # print(f"    get_ts_difference: total_sqrt_ts={total_sqrt_ts}, result={total_sqrt_ts - significance}")
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
        random_state="random-seed",
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

            # Get normalization information using our helper function
            original_norm, norm_unit, copy_func = _get_model_normalization_info(
                spectral_model
            )
            
            # print(f"Iteration {_+1}/{n_iter}: original_norm={original_norm}, norm_unit={norm_unit}")

            lower_bound = original_norm * lower_bound_ratio
            upper_bound = original_norm * upper_bound_ratio
            
            # print(f"  Bounds: [{lower_bound}, {upper_bound}]")

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

        # Use the helper function to get normalization info
        _, norm_unit, copy_func = _get_model_normalization_info(spectral_model)

        final_normalization = np.mean(roots) * norm_unit
        final_normalization_err = np.std(roots) * norm_unit
        
        # print(f"Final roots array: {roots}")
        # print(f"Final mean: {np.mean(roots)}, std: {np.std(roots)}")
        # print(f"Final normalization: {final_normalization}")
        # Create final spectrum with updated normalization
        final_spectrum = copy_func(spectral_model, final_normalization.value)

        photon_flux = final_spectrum.integral(min_energy, max_energy)
        energy_flux = final_spectrum.energy_flux(min_energy, max_energy)

        # calculate errors
        final_spectrum_error = copy_func(
            spectral_model, final_normalization.value + final_normalization_err.value
        )

        photon_flux_error = (
            final_spectrum_error.integral(min_energy, max_energy) - photon_flux
        )
        energy_flux_error = (
            final_spectrum_error.energy_flux(min_energy, max_energy) - energy_flux
        )

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
