from pathlib import Path
from typing import Literal

import astropy.units as u

# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from gammapy.modeling.models import PowerLawSpectralModel
from scipy import integrate
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.optimize import brentq

from .logging import logger
from .sensitivity import SensitivityCtools, SensitivityGammapy

log = logger(__name__)


class GRB:
    def __init__(
        self,
        filepath: str | Path,
        min_energy: u.Quantity | None = None,
        max_energy: u.Quantity | None = None,
    ) -> None:
        if isinstance(min_energy, u.Quantity):
            min_energy = min_energy.to("GeV")
        if isinstance(max_energy, u.Quantity):
            max_energy = max_energy.to("GeV")

        self.filepath = Path(filepath).absolute()
        self.min_energy, self.max_energy = min_energy, max_energy
        self.seen = False
        self.obs_time = -1
        self.start_time = -1
        self.end_time = -1
        self.error_message = ""

        with fits.open(filepath) as hdu_list:
            self.long = hdu_list[0].header["LONG"]
            self.lat = hdu_list[0].header["LAT"]
            self.eiso = hdu_list[0].header["EISO"]
            self.dist = hdu_list[0].header["DISTANCE"]
            self.angle = hdu_list[0].header["ANGLE"]

            datalc = hdu_list[3].data
            datatime = hdu_list[2].data
            dataenergy = hdu_list[1].data

            self.time = datatime.field(0) * u.Unit("s")
            self.energy = dataenergy.field(0) * u.Unit("GeV")

            self.spectra = np.array(
                [datalc.field(i) for i, e in enumerate(self.energy)]
            ) * u.Unit("1 / (cm2 s GeV)")

        # set spectral grid
        self.set_spectral_grid()

        # fit spectral indices
        self.fit_spectral_indices()

        log.debug(f"Loaded event {self.angle}º")

    def __repr__(self):
        return f"<GRB(run={self.run}, id={self.id})>"

    def set_spectral_grid(self):
        self.SpectralGrid = RegularGridInterpolator(
            (np.log10(self.energy.value), np.log10(self.time.value)), self.spectra
        )

    def show_spectral_pattern(self, resolution=100, return_plot=False):
        self.set_spectral_grid()

        loge = np.around(np.log10(self.energy.value), 1)
        logt = np.around(np.log10(self.time.value), 1)

        x = np.around(np.linspace(min(loge), max(loge), resolution + 1), 1)
        y = np.around(np.linspace(min(logt), max(logt), resolution + 1), 1)

        points = []
        for e in x:
            for t in y:
                points.append([e, t])

        plt.xlabel("Log(t)")
        plt.ylabel("Log(E)")
        plt.imshow(
            np.log10(self.SpectralGrid(points)).reshape(resolution + 1, resolution + 1),
            extent=(logt.min(), logt.max(), loge.min(), loge.max()),
            cmap="viridis",
            aspect="auto",
        )
        plt.colorbar(label="spectrum")
        
        if return_plot:
            return plt

    def get_spectrum(
        self, time: u.Quantity, energy: u.Quantity | None = None
    ) -> float | np.ndarray:
        if not time.unit.physical_type == "time":
            raise ValueError(f"time must be a time quantity, got {time}")

        time = time.to("s")

        if energy is None:
            energy = self.energy

        if not energy.unit.physical_type == "energy":
            raise ValueError(f"energy must be an energy quantity, got {energy}")

        energy = energy.to("GeV")

        if (
            isinstance(energy, np.ndarray) or isinstance(energy, list)
        ) and not isinstance(energy, u.Quantity):
            print(energy, type(energy))
            return np.array(
                [
                    self.SpectralGrid((e, np.log10(time.value)))
                    for e in np.log10(energy.value)
                ]
            ) * u.Unit("1 / (cm2 s GeV)")

        return self.SpectralGrid(
            (np.log10(energy.value), np.log10(time.value))
        ) * u.Unit("1 / (cm2 s GeV)")

    def get_flux(self, energy: u.Quantity, time: u.Quantity | None = None):
        if not energy.unit.physical_type == "energy":
            raise ValueError(f"energy must be an energy quantity, got {energy}")

        energy.to("GeV")

        if time is None:
            time = self.time

        if not time.unit.physical_type == "time":
            raise ValueError(f"time must be a time quantity, got {time}")

        time = time.to("s")

        if (isinstance(time, np.ndarray) or isinstance(time, list)) and not isinstance(
            time, u.Quantity
        ):
            return np.array(
                [
                    self.SpectralGrid((np.log10(energy.value), t))
                    for t in np.log10(time.value)
                ]
            ) * u.Unit("1 / (cm2 s GeV)")
        else:
            return self.SpectralGrid(
                (np.log10(energy.value), np.log10(time.value))
            ) * u.Unit("1 / (cm2 s GeV)")

    def get_gammapy_spectrum(self, time: u.Quantity):
        return PowerLawSpectralModel(
            index=self.get_spectral_index(time),
            amplitude=self.get_spectral_amplitude(time).to("cm-2 s-1 GeV-1"),
            reference="1 GeV",
        )

    def fit_spectral_indices(self):
        spectra = self.spectra.T

        indices = []
        amplitudes = []
        times = []
        bad_times = []

        for spectrum, time in zip(spectra, self.time):
            idx = np.isfinite(spectrum) & (spectrum > 0)

            if len(idx[idx] > 3):  # need at least 3 points in the spectrum to fit
                times.append(time)
                fit = np.polyfit(
                    np.log10(self.energy[idx].value), np.log10(spectrum[idx].value), 1
                )
                m = fit[0]
                b = fit[1]
                # print(f"{time:<10.2f} {m:<10.2f} {b:<10.2f}")
                indices.append(m)
                # get amplitudes (flux at E_0 = 1 Gev)
                amplitudes.append(b)  # [ log[ph / (cm2 s GeV)]]
            else:
                bad_times.append(time)

        self._indices = indices
        self._amplitudes = amplitudes
        self._index_times = times
        self._bad_index_times = bad_times

        self.index_at = interp1d(
            np.log10([t.value for t in self._index_times]),
            self._indices,
            fill_value="extrapolate",
        )

        self.amplitude_at = lambda x: 10 ** interp1d(
            np.log10([t.value for t in self._index_times]),
            self._amplitudes,
            fill_value="extrapolate",
        )(x)

    def get_spectral_index(self, time: u.Quantity) -> float:
        if not time.unit.physical_type == "time":
            raise ValueError(f"time must be a time quantity, got {time}")

        time = time.to("s")

        return self.index_at(np.array([np.log10(time.value)]))[0]

    def get_spectral_amplitude(self, time: u.Quantity) -> float:
        if not time.unit.physical_type == "time":
            raise ValueError(f"time must be a time quantity, got {time}")

        time = time.to("s")

        return self.amplitude_at(np.array([np.log10(time.value)]))[0] * u.Unit(
            "cm-2 s-1 GeV-1"
        )

    def show_spectral_evolution(self, resolution=100, return_plot=False):
        self.fit_spectral_indices()

        t = np.linspace(
            np.log10(min(self.time).value),
            np.log10(max(self.time).value),
            resolution + 1,
        )

        plt.plot(t, self.index_at(t))
        plt.xlabel("Log(t) (s)")
        plt.ylabel("Spectral Index")
        
        if return_plot:
            return plt
        
        plt.show()

    def get_integral_spectrum(self, time: u.Quantity, first_energy_bin: u.Quantity):
        if not time.unit.physical_type == "time":
            raise ValueError(f"time must be a time quantity, got {time}")

        if not first_energy_bin.unit.physical_type == "energy":
            raise ValueError(
                f"first_energy_bin must be an energy quantity, got {first_energy_bin}"
            )

        if self.min_energy is None or self.max_energy is None:
            raise ValueError("Please set min and max energy for integral spectrum.")

        spectral_index = self.get_spectral_index(time)
        spectral_index_plus_one = spectral_index + 1

        integral_spectrum = (
            self.get_flux(first_energy_bin, time=time)
            * (first_energy_bin ** (-spectral_index) / spectral_index_plus_one)
            * (
                (self.max_energy**spectral_index_plus_one)
                - (self.min_energy**spectral_index_plus_one)
            )
        )

        return integral_spectrum

    def get_fluence(self, start_time: u.Quantity, stop_time: u.Quantity):
        if not start_time.unit.physical_type == "time":
            raise ValueError(f"start_time must be a time quantity, got {start_time}")
        if not stop_time.unit.physical_type == "time":
            raise ValueError(f"stop_time must be a time quantity, got {stop_time}")

        start_time = start_time.to("s")
        stop_time = stop_time.to("s")

        first_energy_bin = min(self.energy)

        fluence = integrate.quad(
            lambda time: self.get_integral_spectrum(time * u.s, first_energy_bin).value,
            start_time.value,
            stop_time.value,
        )[0] * u.Unit("cm-2")

        log.debug(f"    Fluence: {fluence}")
        return fluence

    def output(self):
        keys_to_drop = [
            "time",
            "energy",
            "spectra",
            "SpectralGrid",
            "rng",
            "power_law_slopes",
            "spectral_indices",
            "_indices",
            "_index_times",
            "_amplitudes",
            "_bad_index_times",
            "index_at",
            "amplitude_at",
        ]

        o = {}

        for k, v in self.__dict__.items():
            # drop unneeded data
            if k not in keys_to_drop:
                # convert numpy numbers
                if isinstance(v, np.integer):
                    o[k] = int(v)
                elif isinstance(v, np.floating):
                    o[k] = float(v)
                elif isinstance(v, u.Quantity):  # check if value is a Quantity object
                    o[k] = v  # convert Quantity to list
                elif isinstance(v, np.ndarray):
                    o[k] = v.tolist()
                else:
                    o[k] = v

        return o

    def check_if_visible(
        self,
        start_time: u.Quantity,
        stop_time: u.Quantity,
        sensitivity: SensitivityCtools | SensitivityGammapy,
        mode: str = "bool",
    ) -> bool | float:
        if mode not in ["bool", "difference"]:
            raise ValueError(f"mode must be 'bool' or 'difference', got {mode}")

        if not start_time.unit.physical_type == "time":
            raise ValueError(f"start_time must be a time quantity, got {start_time}")
        if not stop_time.unit.physical_type == "time":
            raise ValueError(f"stop_time must be a time quantity, got {stop_time}")

        start_time = start_time.to("s")
        stop_time = stop_time.to("s")

        # Interpolation and integration of the flux with time
        average_flux = self.get_fluence(start_time, stop_time) / (
            stop_time - start_time
        )

        # calculate photon flux
        if isinstance(sensitivity, SensitivityCtools):
            photon_flux = sensitivity.get(t=(stop_time - start_time))
        elif isinstance(sensitivity, SensitivityGammapy):
            photon_flux = sensitivity.get(
                t=(stop_time - start_time),
                index=self.get_spectral_index(stop_time),
                amplitude=self.get_spectral_amplitude(stop_time),
            )
        else:
            raise ValueError(f"Unknown sensitivity type: {type(sensitivity)}")

        visible = True if average_flux > photon_flux else False

        print(
            f"{visible} after {stop_time - start_time}: average_flux = {average_flux} |:| photon_flux = {photon_flux}"
        )

        log.debug(
            f"    visible:{visible} avgflux={average_flux}, photon_flux={photon_flux}"
        )

        if mode == "difference":
            difference = (average_flux - photon_flux).value
            sign = np.sign(difference)
            # print(difference, np.log10(abs(difference)), sign)
            return (1 / np.log10(abs(difference))) * sign

        return visible

    def observe(
        self,
        sensitivity: SensitivityCtools | SensitivityGammapy,
        start_time: u.Quantity = 0 * u.s,
        min_energy: u.Quantity = None,
        max_energy: u.Quantity = None,
        max_time: u.Quantity = 12 * u.hour,
        target_precision: u.Quantity = 1 * u.s,
        _max_loops=100,
    ):
        """Modified version to increase timestep along with time size"""

        if not start_time.unit.physical_type == "time":
            raise ValueError(f"start_time must be a time quantity, got {start_time}")

        if not max_time.unit.physical_type == "time":
            raise ValueError(f"max_time must be a time quantity, got {max_time}")

        if not target_precision.unit.physical_type == "time":
            raise ValueError(
                f"target_precision must be a time quantity, got {target_precision}"
            )

        # set energy limits to match the sensitivity
        if not min_energy or not max_energy:
            self.min_energy, self.max_energy = sensitivity.energy_limits

        if not self.min_energy.unit.physical_type == "energy":
            raise ValueError(
                f"min_energy must be an energy quantity, got {self.min_energy}"
            )
        if not self.max_energy.unit.physical_type == "energy":
            raise ValueError(
                f"max_energy must be an energy quantity, got {self.max_energy}"
            )

        self.min_energy = self.min_energy.to("GeV")
        self.max_energy = self.max_energy.to("GeV")

        start_time = start_time.to("s")
        max_time = max_time.to("s")

        try:
            # start the procedure
            self.start_time = start_time
            delay = start_time
            end_time = delay + max_time

            # define a function for Brent's method
            def func(
                x: float,
                start_time: float,
                sensitivity: SensitivityCtools | SensitivityGammapy,
            ):
                return self.check_if_visible(
                    start_time=start_time * u.s,
                    stop_time=x * u.s,
                    sensitivity=sensitivity,
                    mode="difference",
                )

            # find the root of the function
            root, result = brentq(
                func,
                a=start_time.value,
                b=end_time.value,
                args=(start_time.value, sensitivity),
                xtol=target_precision.value,
                full_output=True,
            )
            print(root, result)

            if result.converged:
                self.end_time = round(root, int(1 / target_precision.value)) * u.s
                self.obs_time = self.end_time - delay
                self.seen = True
                log.debug(f"    obs_time={self.obs_time} end_time={self.end_time}")
            else:
                self.seen = False

            # just return dict now
            return self.output()

        except Exception as e:
            raise e

            self.seen = "error"
            self.error_message = str(e)

            return self.output()


class oGRB:
    def __init__(
        self,
        filepath: str | Path,
        min_energy: u.Quantity | None = None,
        max_energy: u.Quantity | None = None,
    ) -> None:
        if isinstance(min_energy, u.Quantity):
            min_energy = min_energy.to("GeV")
        if isinstance(max_energy, u.Quantity):
            max_energy = max_energy.to("GeV")

        self.filepath = Path(filepath).absolute()
        self.min_energy, self.max_energy = min_energy, max_energy
        self.seen = False
        self.obs_time = -1
        self.start_time = -1
        self.end_time = -1
        self.error_message = ""

        with fits.open(filepath) as hdu_list:
            self.long = hdu_list[0].header["LONG"]
            self.lat = hdu_list[0].header["LAT"]
            self.eiso = hdu_list[0].header["EISO"]
            self.dist = hdu_list[0].header["DISTANCE"]
            self.angle = hdu_list[0].header["ANGLE"]

            datalc = hdu_list[3].data
            datatime = hdu_list[2].data
            dataenergy = hdu_list[1].data

            self.time = datatime.field(0) * u.Unit("s")
            self.energy = dataenergy.field(0) * u.Unit("GeV")

            self.spectra = np.array(
                [datalc.field(i) for i, e in enumerate(self.energy)]
            ) * u.Unit("1 / (cm2 s GeV)")

        # set spectral grid
        self.set_spectral_grid()

        # fit spectral indices
        self.fit_spectral_indices()

        log.debug(f"Loaded event {self.angle}º")

    def __repr__(self):
        return f"<GRB(run={self.run}, id={self.id})>"

    def set_spectral_grid(self):
        self.SpectralGrid = RegularGridInterpolator(
            (np.log10(self.energy.value), np.log10(self.time.value)), self.spectra
        )

    def show_spectral_pattern(self, resolution=100):
        self.set_spectral_grid()

        loge = np.around(np.log10(self.energy.value), 1)
        logt = np.around(np.log10(self.time.value), 1)

        x = np.around(np.linspace(min(loge), max(loge), resolution + 1), 1)
        y = np.around(np.linspace(min(logt), max(logt), resolution + 1), 1)

        points = []
        for e in x:
            for t in y:
                points.append([e, t])

        plt.xlabel("Log(t)")
        plt.ylabel("Log(E)")
        plt.imshow(
            np.log10(self.SpectralGrid(points)).reshape(resolution + 1, resolution + 1),
            extent=(logt.min(), logt.max(), loge.min(), loge.max()),
            cmap="viridis",
            aspect="auto",
        )
        plt.colorbar(label="spectrum")

    def get_spectrum(
        self, time: u.Quantity, energy: u.Quantity | None = None
    ) -> float | np.ndarray:
        if not time.unit.physical_type == "time":
            raise ValueError(f"time must be a time quantity, got {time}")

        time = time.to("s")

        if energy is None:
            energy = self.energy

        if not energy.unit.physical_type == "energy":
            raise ValueError(f"energy must be an energy quantity, got {energy}")

        energy = energy.to("GeV")

        if (
            isinstance(energy, np.ndarray) or isinstance(energy, list)
        ) and not isinstance(energy, u.Quantity):
            print(energy, type(energy))
            return np.array(
                [
                    self.SpectralGrid((e, np.log10(time.value)))
                    for e in np.log10(energy.value)
                ]
            ) * u.Unit("1 / (cm2 s GeV)")

        return self.SpectralGrid(
            (np.log10(energy.value), np.log10(time.value))
        ) * u.Unit("1 / (cm2 s GeV)")

    def get_flux(self, energy: u.Quantity, time: u.Quantity | None = None):
        if not energy.unit.physical_type == "energy":
            raise ValueError(f"energy must be an energy quantity, got {energy}")

        energy.to("GeV")

        if time is None:
            time = self.time

        if not time.unit.physical_type == "time":
            raise ValueError(f"time must be a time quantity, got {time}")

        time = time.to("s")

        if (isinstance(time, np.ndarray) or isinstance(time, list)) and not isinstance(
            time, u.Quantity
        ):
            return np.array(
                [
                    self.SpectralGrid((np.log10(energy.value), t))
                    for t in np.log10(time.value)
                ]
            ) * u.Unit("1 / (cm2 s GeV)")
        else:
            return self.SpectralGrid(
                (np.log10(energy.value), np.log10(time.value))
            ) * u.Unit("1 / (cm2 s GeV)")

    def get_gammapy_spectrum(self, time: u.Quantity):
        return PowerLawSpectralModel(
            index=self.get_spectral_index(time),
            amplitude=self.get_spectral_amplitude(time).to("cm-2 s-1 GeV-1"),
            reference="1 GeV",
        )

    def fit_spectral_indices(self):
        spectra = self.spectra.T

        indices = []
        amplitudes = []
        times = []
        bad_times = []

        for spectrum, time in zip(spectra, self.time):
            idx = np.isfinite(spectrum) & (spectrum > 0)

            if len(idx[idx] > 3):  # need at least 3 points in the spectrum to fit
                times.append(time)
                fit = np.polyfit(
                    np.log10(self.energy[idx].value), np.log10(spectrum[idx].value), 1
                )
                m = fit[0]
                b = fit[1]
                # print(f"{time:<10.2f} {m:<10.2f} {b:<10.2f}")
                indices.append(m)
                # get amplitudes (flux at E_0 = 1 Gev)
                amplitudes.append(b)  # [ log[ph / (cm2 s GeV)]]
            else:
                bad_times.append(time)

        self._indices = indices
        self._amplitudes = amplitudes
        self._index_times = times
        self._bad_index_times = bad_times

        self.index_at = interp1d(
            np.log10([t.value for t in self._index_times]),
            self._indices,
            fill_value="extrapolate",
        )

        self.amplitude_at = lambda x: 10 ** interp1d(
            np.log10([t.value for t in self._index_times]),
            self._amplitudes,
            fill_value="extrapolate",
        )(x)

    def get_spectral_index(self, time: u.Quantity) -> float:
        if not time.unit.physical_type == "time":
            raise ValueError(f"time must be a time quantity, got {time}")

        time = time.to("s")

        return self.index_at(np.array([np.log10(time.value)]))[0]

    def get_spectral_amplitude(self, time: u.Quantity) -> float:
        if not time.unit.physical_type == "time":
            raise ValueError(f"time must be a time quantity, got {time}")

        time = time.to("s")

        return self.amplitude_at(np.array([np.log10(time.value)]))[0] * u.Unit(
            "cm-2 s-1 GeV-1"
        )

    def show_spectral_evolution(self, resolution=100):
        self.fit_spectral_indices()

        t = np.linspace(
            np.log10(min(self.time).value),
            np.log10(max(self.time).value),
            resolution + 1,
        )

        plt.plot(t, self.index_at(t))
        plt.xlabel("Log(t) (s)")
        plt.ylabel("Spectral Index")

        plt.show()

    def get_integral_spectrum(self, time: u.Quantity, first_energy_bin: u.Quantity, mode: Literal["gammapy", "ctools"] = "gammapy"):
        if not time.unit.physical_type == "time":
            raise ValueError(f"time must be a time quantity, got {time}")

        if not first_energy_bin.unit.physical_type == "energy":
            raise ValueError(
                f"first_energy_bin must be an energy quantity, got {first_energy_bin}"
            )

        if self.min_energy is None or self.max_energy is None:
            raise ValueError("Please set min and max energy for integral spectrum.")

        spectral_index = self.get_spectral_index(time)
        amount_to_add = 1 if mode == "ctools" else 2
        spectral_index_plus = spectral_index + amount_to_add

        integral_spectrum = (
            self.get_flux(first_energy_bin, time=time)
            * (first_energy_bin ** (-spectral_index) / spectral_index_plus)
            * (
                (self.max_energy**spectral_index_plus)
                - (self.min_energy**spectral_index_plus)
            )
        )

        return integral_spectrum

    def get_fluence(self, start_time: u.Quantity, stop_time: u.Quantity, mode: Literal["gammapy", "ctools"] = "gammapy"):
        if not start_time.unit.physical_type == "time":
            raise ValueError(f"start_time must be a time quantity, got {start_time}")
        if not stop_time.unit.physical_type == "time":
            raise ValueError(f"stop_time must be a time quantity, got {stop_time}")

        start_time = start_time.to("s")
        stop_time = stop_time.to("s")

        first_energy_bin = min(self.energy)

        fluence = integrate.quad(
            lambda time: self.get_integral_spectrum(time * u.s, first_energy_bin, mode=mode).value,
            start_time.value,
            stop_time.value,
        )[0] * u.Unit("cm-2")

        log.debug(f"    Fluence: {fluence}")
        return fluence

    def output(self):
        keys_to_drop = [
            "time",
            "energy",
            "spectra",
            "SpectralGrid",
            "rng",
            "power_law_slopes",
            "spectral_indices",
            "_indices",
            "_index_times",
            "_amplitudes",
            "_bad_index_times",
            "index_at",
            "amplitude_at",
        ]

        o = {}

        for k, v in self.__dict__.items():
            # drop unneeded data
            if k not in keys_to_drop:
                # convert numpy numbers
                if isinstance(v, np.integer):
                    o[k] = int(v)
                elif isinstance(v, np.floating):
                    o[k] = float(v)
                elif isinstance(v, u.Quantity):  # check if value is a Quantity object
                    o[k] = v  # convert Quantity to list
                elif isinstance(v, np.ndarray):
                    o[k] = v.tolist()
                else:
                    o[k] = v

        return o

    def check_if_visible(
        self, sensitivity: SensitivityGammapy | SensitivityCtools, start_time, stop_time
    ):
        # Interpolation and integration of the flux with time
        mode = "gammapy" if isinstance(sensitivity, SensitivityGammapy) else "ctools"
        average_flux = self.get_fluence(start_time, stop_time, mode=mode) / (
            stop_time - start_time
        )

        # calculate photon flux
        photon_flux = sensitivity.get(t=(stop_time - start_time))

        visible = True if average_flux > photon_flux else False

        log.debug(
            f"    visible:{visible} avgflux={average_flux}, photon_flux={photon_flux}"
        )

        return visible

    def observe(
        self,
        sensitivity: SensitivityGammapy | SensitivityCtools,
        start_time: float = 0,
        min_energy=None,
        max_energy=None,
        max_time=None,
        target_precision=1,
        _max_loops=10000,
    ):
        """Modified version to increase timestep along with time size"""

        try:
            # set energy limits to match the sensitivity
            if not min_energy or not max_energy:
                self.min_energy, self.max_energy = sensitivity.energy_limits

            # start the procedure
            self.start_time = start_time
            delay = start_time

            # set default max time
            if max_time is None:
                max_time = 43200  # 12h after starting observations

            # check maximum time
            visible = self.check_if_visible(sensitivity, delay, max_time + delay)

            # not visible even after maximum observation time
            if not visible:
                return self.output()

            loop_number = 0
            precision = int(10 ** int(np.floor(np.log10(max_time + delay))))
            observation_time = precision
            previous_observation_time = precision

            # find the inflection point
            while loop_number < _max_loops:
                loop_number += 1
                visible = self.check_if_visible(
                    sensitivity, delay, delay + observation_time
                )

                if visible:
                    # if desired precision is reached, return results and break!
                    if np.log10(precision) == np.log10(target_precision):
                        round_precision = int(-np.log10(precision))
                        end_time = delay + round(observation_time, round_precision)
                        self.end_time = round(end_time, round_precision)
                        self.obs_time = round(observation_time, round_precision)
                        self.seen = True
                        log.debug(
                            f"    obs_time={observation_time} end_time={end_time}"
                        )
                        break

                    elif observation_time == precision:
                        # reduce precision
                        precision = 10 ** (int(np.log10(precision)) - 1)
                        observation_time = precision
                        log.debug(f"    Updating precision to {precision}")

                    else:  # reduce precision but add more time
                        precision = 10 ** (int(np.log10(precision)) - 1)
                        observation_time = previous_observation_time + precision
                        log.debug(
                            f"    Going back to {previous_observation_time} and adding more time {precision}s"
                        )

                else:
                    previous_observation_time = observation_time
                    observation_time += precision
                    # update DT and loop again

            # just return dict now
            return self.output()

        except Exception as e:
            print(e)

            self.seen = "error"
            self.error_message = str(e)

            return self.output()


class bGRB:
    def __init__(
        self,
        filepath: str | Path,
        min_energy: u.Quantity | None = None,
        max_energy: u.Quantity | None = None,
    ) -> None:
        if isinstance(min_energy, u.Quantity):
            min_energy = min_energy.to("GeV")
        if isinstance(max_energy, u.Quantity):
            max_energy = max_energy.to("GeV")

        self.filepath = Path(filepath).absolute()
        self.min_energy, self.max_energy = min_energy, max_energy
        self.seen = False
        self.obs_time = -1
        self.start_time = -1
        self.end_time = -1
        self.error_message = ""

        with fits.open(filepath) as hdu_list:
            self.long = hdu_list[0].header["LONG"]
            self.lat = hdu_list[0].header["LAT"]
            self.eiso = hdu_list[0].header["EISO"]
            self.dist = hdu_list[0].header["DISTANCE"]
            self.angle = hdu_list[0].header["ANGLE"]

            datalc = hdu_list[3].data
            datatime = hdu_list[2].data
            dataenergy = hdu_list[1].data

            self.time = datatime.field(0) * u.Unit("s")
            self.energy = dataenergy.field(0) * u.Unit("GeV")

            self.spectra = np.array(
                [datalc.field(i) for i, e in enumerate(self.energy)]
            ) * u.Unit("1 / (cm2 s GeV)")

        # set spectral grid
        self.set_spectral_grid()

        # fit spectral indices
        self.fit_spectral_indices()

        log.debug(f"Loaded event {self.angle}º")

    def __repr__(self):
        return f"<GRB(run={self.run}, id={self.id})>"

    def set_spectral_grid(self):
        self.SpectralGrid = RegularGridInterpolator(
            (np.log10(self.energy.value), np.log10(self.time.value)), self.spectra
        )

    def show_spectral_pattern(self, resolution=100):
        self.set_spectral_grid()

        loge = np.around(np.log10(self.energy.value), 1)
        logt = np.around(np.log10(self.time.value), 1)

        x = np.around(np.linspace(min(loge), max(loge), resolution + 1), 1)
        y = np.around(np.linspace(min(logt), max(logt), resolution + 1), 1)

        points = []
        for e in x:
            for t in y:
                points.append([e, t])

        plt.xlabel("Log(t)")
        plt.ylabel("Log(E)")
        plt.imshow(
            np.log10(self.SpectralGrid(points)).reshape(resolution + 1, resolution + 1),
            extent=(logt.min(), logt.max(), loge.min(), loge.max()),
            cmap="viridis",
            aspect="auto",
        )
        plt.colorbar(label="spectrum")

    def get_spectrum(
        self, time: u.Quantity, energy: u.Quantity | None = None
    ) -> float | np.ndarray:
        if not time.unit.physical_type == "time":
            raise ValueError(f"time must be a time quantity, got {time}")

        time = time.to("s")

        if energy is None:
            energy = self.energy

        if not energy.unit.physical_type == "energy":
            raise ValueError(f"energy must be an energy quantity, got {energy}")

        energy = energy.to("GeV")

        if (
            isinstance(energy, np.ndarray) or isinstance(energy, list)
        ) and not isinstance(energy, u.Quantity):
            print(energy, type(energy))
            return np.array(
                [
                    self.SpectralGrid((e, np.log10(time.value)))
                    for e in np.log10(energy.value)
                ]
            ) * u.Unit("1 / (cm2 s GeV)")

        return self.SpectralGrid(
            (np.log10(energy.value), np.log10(time.value))
        ) * u.Unit("1 / (cm2 s GeV)")

    def get_flux(self, energy: u.Quantity, time: u.Quantity | None = None):
        if not energy.unit.physical_type == "energy":
            raise ValueError(f"energy must be an energy quantity, got {energy}")

        energy.to("GeV")

        if time is None:
            time = self.time

        if not time.unit.physical_type == "time":
            raise ValueError(f"time must be a time quantity, got {time}")

        time = time.to("s")

        if (isinstance(time, np.ndarray) or isinstance(time, list)) and not isinstance(
            time, u.Quantity
        ):
            return np.array(
                [
                    self.SpectralGrid((np.log10(energy.value), t))
                    for t in np.log10(time.value)
                ]
            ) * u.Unit("1 / (cm2 s GeV)")
        else:
            return self.SpectralGrid(
                (np.log10(energy.value), np.log10(time.value))
            ) * u.Unit("1 / (cm2 s GeV)")

    def get_gammapy_spectrum(self, time: u.Quantity):
        return PowerLawSpectralModel(
            index=self.get_spectral_index(time),
            amplitude=self.get_spectral_amplitude(time).to("cm-2 s-1 GeV-1"),
            reference="1 GeV",
        )

    def fit_spectral_indices(self):
        spectra = self.spectra.T

        indices = []
        amplitudes = []
        times = []
        bad_times = []

        for spectrum, time in zip(spectra, self.time):
            idx = np.isfinite(spectrum) & (spectrum > 0)

            if len(idx[idx] > 3):  # need at least 3 points in the spectrum to fit
                times.append(time)
                fit = np.polyfit(
                    np.log10(self.energy[idx].value), np.log10(spectrum[idx].value), 1
                )
                m = fit[0]
                b = fit[1]
                # print(f"{time:<10.2f} {m:<10.2f} {b:<10.2f}")
                indices.append(m)
                # get amplitudes (flux at E_0 = 1 Gev)
                amplitudes.append(b)  # [ log[ph / (cm2 s GeV)]]
            else:
                bad_times.append(time)

        self._indices = indices
        self._amplitudes = amplitudes
        self._index_times = times
        self._bad_index_times = bad_times

        self.index_at = interp1d(
            np.log10([t.value for t in self._index_times]),
            self._indices,
            fill_value="extrapolate",
        )

        self.amplitude_at = lambda x: 10 ** interp1d(
            np.log10([t.value for t in self._index_times]),
            self._amplitudes,
            fill_value="extrapolate",
        )(x)

    def get_spectral_index(self, time: u.Quantity) -> float:
        if not time.unit.physical_type == "time":
            raise ValueError(f"time must be a time quantity, got {time}")

        time = time.to("s")

        return self.index_at(np.array([np.log10(time.value)]))[0]

    def get_spectral_amplitude(self, time: u.Quantity) -> float:
        if not time.unit.physical_type == "time":
            raise ValueError(f"time must be a time quantity, got {time}")

        time = time.to("s")

        return self.amplitude_at(np.array([np.log10(time.value)]))[0] * u.Unit(
            "cm-2 s-1 GeV-1"
        )

    def show_spectral_evolution(self, resolution=100):
        self.fit_spectral_indices()

        t = np.linspace(
            np.log10(min(self.time).value),
            np.log10(max(self.time).value),
            resolution + 1,
        )

        plt.plot(t, self.index_at(t))
        plt.xlabel("Log(t) (s)")
        plt.ylabel("Spectral Index")

        plt.show()

    def get_integral_spectrum(self, time: u.Quantity, first_energy_bin: u.Quantity):
        if not time.unit.physical_type == "time":
            raise ValueError(f"time must be a time quantity, got {time}")

        if not first_energy_bin.unit.physical_type == "energy":
            raise ValueError(
                f"first_energy_bin must be an energy quantity, got {first_energy_bin}"
            )

        if self.min_energy is None or self.max_energy is None:
            raise ValueError("Please set min and max energy for integral spectrum.")

        spectral_index = self.get_spectral_index(time)
        spectral_index_plus_one = spectral_index + 1

        integral_spectrum = (
            self.get_flux(first_energy_bin, time=time)
            * (first_energy_bin ** (-spectral_index) / spectral_index_plus_one)
            * (
                (self.max_energy**spectral_index_plus_one)
                - (self.min_energy**spectral_index_plus_one)
            )
        )

        return integral_spectrum

    def get_fluence(self, start_time: u.Quantity, stop_time: u.Quantity):
        if not start_time.unit.physical_type == "time":
            raise ValueError(f"start_time must be a time quantity, got {start_time}")
        if not stop_time.unit.physical_type == "time":
            raise ValueError(f"stop_time must be a time quantity, got {stop_time}")

        start_time = start_time.to("s")
        stop_time = stop_time.to("s")

        first_energy_bin = min(self.energy)

        fluence = integrate.quad(
            lambda time: self.get_integral_spectrum(time * u.s, first_energy_bin).value,
            start_time.value,
            stop_time.value,
        )[0] * u.Unit("cm-2")

        log.debug(f"    Fluence: {fluence}")
        return fluence

    def output(self):
        keys_to_drop = [
            "time",
            "energy",
            "spectra",
            "SpectralGrid",
            "rng",
            "power_law_slopes",
            "spectral_indices",
            "_indices",
            "_index_times",
            "_amplitudes",
            "_bad_index_times",
            "index_at",
            "amplitude_at",
        ]

        o = {}

        for k, v in self.__dict__.items():
            # drop unneeded data
            if k not in keys_to_drop:
                # convert numpy numbers
                if isinstance(v, np.integer):
                    o[k] = int(v)
                elif isinstance(v, np.floating):
                    o[k] = float(v)
                elif isinstance(v, u.Quantity):  # check if value is a Quantity object
                    o[k] = v  # convert Quantity to list
                elif isinstance(v, np.ndarray):
                    o[k] = v.tolist()
                else:
                    o[k] = v

        return o

    def check_if_visible(
        self,
        stop_time: u.Quantity,
        start_time: u.Quantity,
        sensitivity: SensitivityCtools | SensitivityGammapy,
        mode: str = "bool",
    ) -> bool | float:
        if mode not in ["bool", "difference"]:
            raise ValueError(f"mode must be 'bool' or 'difference', got {mode}")

        if not start_time.unit.physical_type == "time":
            raise ValueError(f"start_time must be a time quantity, got {start_time}")
        if not stop_time.unit.physical_type == "time":
            raise ValueError(f"stop_time must be a time quantity, got {stop_time}")

        start_time = start_time.to("s")
        stop_time = stop_time.to("s")

        # Interpolation and integration of the flux with time
        average_flux = self.get_fluence(start_time, stop_time) / (
            stop_time - start_time
        )

        # calculate photon flux
        if isinstance(sensitivity, SensitivityCtools):
            photon_flux = sensitivity.get(t=(stop_time - start_time))
        elif isinstance(sensitivity, SensitivityGammapy):
            photon_flux = sensitivity.get(
                t=(stop_time - start_time),
                index=self.get_spectral_index(stop_time),
                amplitude=self.get_spectral_amplitude(stop_time),
            )
        else:
            raise ValueError(f"Unknown sensitivity type: {type(sensitivity)}")

        visible = True if average_flux > photon_flux else False

        print(
            f"{visible} after {stop_time - start_time}: average_flux = {average_flux} |:| photon_flux = {photon_flux}"
        )

        log.debug(
            f"    visible:{visible} avgflux={average_flux}, photon_flux={photon_flux}"
        )

        if mode == "difference":
            return average_flux - photon_flux

        return visible

    def _bisect_find_zeros(
        self,
        sensitivity: SensitivityCtools | SensitivityGammapy,
        start_time: u.Quantity,
        stop_time: u.Quantity,
        target_precision: u.Quantity,
        max_iter: int = 100,
        current_iter: int = 0,
        last_guess: u.Quantity = None,
    ):
        midpoint = (start_time + stop_time) / 2
        diff = self.check_if_visible(
            start_time, midpoint, sensitivity, mode="difference"
        )
        seen = diff > 0

        current_iter += 1
        if current_iter > max_iter:
            return midpoint, seen

        if seen and last_guess is not None:
            guess_distance = np.abs(midpoint - last_guess).to(target_precision.unit)
            if guess_distance < target_precision:
                return midpoint, seen

        f_start = self.check_if_visible(
            start_time, start_time, sensitivity, mode="difference"
        )
        f_mid = self.check_if_visible(
            start_time, midpoint, sensitivity, mode="difference"
        )

        if np.sign(f_start) != np.sign(f_mid):
            # Midpoint is an improvement towards the root on left side
            return self._bisect_find_zeros(
                sensitivity,
                start_time,
                midpoint,
                target_precision,
                max_iter,
                current_iter,
                midpoint,
            )
        else:
            # Midpoint is an improvement towards the root on right side
            return self._bisect_find_zeros(
                sensitivity,
                midpoint,
                stop_time,
                target_precision,
                max_iter,
                current_iter,
                midpoint,
            )

    def observe(
        self,
        sensitivity: SensitivityCtools | SensitivityGammapy,
        start_time: u.Quantity = 0 * u.s,
        min_energy: u.Quantity = None,
        max_energy: u.Quantity = None,
        max_time: u.Quantity = 12 * u.hour,
        target_precision: u.Quantity = 1 * u.s,
        _max_loops=100,
    ):
        """Modified version to increase timestep along with time size"""

        if not start_time.unit.physical_type == "time":
            raise ValueError(f"start_time must be a time quantity, got {start_time}")

        if not max_time.unit.physical_type == "time":
            raise ValueError(f"max_time must be a time quantity, got {max_time}")

        if not target_precision.unit.physical_type == "time":
            raise ValueError(
                f"target_precision must be a time quantity, got {target_precision}"
            )

        # set energy limits to match the sensitivity
        if not min_energy or not max_energy:
            self.min_energy, self.max_energy = sensitivity.energy_limits

        if not self.min_energy.unit.physical_type == "energy":
            raise ValueError(
                f"min_energy must be an energy quantity, got {self.min_energy}"
            )
        if not self.max_energy.unit.physical_type == "energy":
            raise ValueError(
                f"max_energy must be an energy quantity, got {self.max_energy}"
            )

        self.min_energy = self.min_energy.to("GeV")
        self.max_energy = self.max_energy.to("GeV")

        start_time = start_time.to("s")
        max_time = max_time.to("s")

        self._num_iters = 0
        self._last_guess = max_time + start_time

        try:
            # start the procedure
            self.start_time = start_time
            delay = start_time

            # check maximum time
            print(f"Checking if visible from {delay} to {max_time + delay}")
            visible = self.check_if_visible(delay, max_time + delay, sensitivity)

            # not visible even after maximum observation time
            if not visible:
                return self.output()

            end_time, seen = self._bisect_find_zeros(
                sensitivity, start_time, max_time + start_time, target_precision
            )

            if seen:
                self.end_time = end_time
                self.obs_time = end_time - start_time
                self.seen = True
            else:
                self.end_time = -1
                self.obs_time = -1
                self.seen = False

            # just return dict now
            return self.output()

        except Exception as e:
            raise e
            print(e)

            self.seen = "error"
            self.error_message = str(e)

            return self.output()
