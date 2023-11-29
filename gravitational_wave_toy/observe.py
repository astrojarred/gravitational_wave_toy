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
        try:
            self.id = int(self.filepath.stem.split("_")[1])
        except ValueError:
            self.id = 0

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
        return f"<GRB(id={self.id})>"

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
        plt.colorbar(label="Log spectrum [ph / (cm2 s GeV)]")
        
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

        unit = u.Unit("cm-2") if mode == "ctools" else u.Unit("GeV cm-2")
        fluence = integrate.quad(
            lambda time: self.get_integral_spectrum(time * u.s, first_energy_bin, mode=mode).value,
            start_time.value,
            stop_time.value,
        )[0] * unit

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
            "_num_iters",
            "_last_guess"
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
        mode: Literal["bool", "difference"] = "bool",
    ) -> bool:
        
        if not start_time.unit.physical_type == "time":
            raise ValueError(f"start_time must be a time quantity, got {start_time}")
        if not stop_time.unit.physical_type == "time":
            raise ValueError(f"stop_time must be a time quantity, got {stop_time}")

        start_time = start_time.to("s")
        stop_time = stop_time.to("s")
        
        # Interpolation and integration of the flux with time
        sens_type = "gammapy" if isinstance(sensitivity, SensitivityGammapy) else "ctools"
        # CTOOLS: 1 / (cm2)   || GAMMAPY: GeV / (cm2)
        fluence = self.get_fluence(start_time, stop_time, mode=sens_type)
        
        # CTOOLS: 1 / (cm2 s)   || GAMMAPY: GeV / (cm2 s)
        average_flux = fluence /  (stop_time - start_time)
        
        if sens_type == "ctools":
            # calculate photon flux [ 1 / (cm2 s) ]
            photon_flux = sensitivity.get(t=(stop_time - start_time))
            visible = average_flux > photon_flux
            difference = average_flux - photon_flux
            
            log.debug(
                f"CTOOLS:    visible:{visible} avgflux={average_flux}, photon_flux={photon_flux}"
            )
            
        else:  # gammapy
            e2dnde = sensitivity.get(
                t=(stop_time - start_time),
            ).to("GeV / (cm2 s)")
            
            visible = average_flux > e2dnde 
            difference = average_flux - e2dnde
            
            log.debug(
                f"GAMMAPY:    visible:{visible} avgflux={average_flux}, sensitivity={sensitivity}"
            )
            

        if mode == "bool":
            return visible
        else:
            return difference

    def _bisect_find_zeros(
        self,
        sensitivity: SensitivityCtools | SensitivityGammapy,
        start_time: u.Quantity,
        stop_time: u.Quantity,
        target_precision: u.Quantity,
        max_iter: int = 100,
        current_iter: int = 0,
        lowest_possible: u.Quantity | None = None,
        highest_possible: u.Quantity | None = None,
    ):
                
        if lowest_possible is None:
            lowest_possible = start_time
        if highest_possible is None:
            highest_possible = stop_time
            
        midpoint = (lowest_possible + highest_possible) / 2
        
        if current_iter >= max_iter:
            return midpoint, False
        
        seen = self.check_if_visible(
            start_time, 
            midpoint, 
            sensitivity,
        )
        
        # print in table format
        log.debug(f"{current_iter:<10} {start_time:<10.2f} {stop_time:<10.2f} {midpoint:<10.2f} {seen}")
        
        if seen:
            
            if highest_possible - midpoint < target_precision:
                if midpoint - start_time < target_precision:
                    res = target_precision+start_time
                else:
                    res = midpoint
                # round res to target precision
                res = round((res / target_precision).value) * target_precision
                return res, True
            
            return self._bisect_find_zeros(
                sensitivity,
                start_time=start_time,
                stop_time=stop_time,
                target_precision=target_precision,
                max_iter=max_iter,
                current_iter=current_iter + 1,
                lowest_possible=lowest_possible,
                highest_possible=midpoint,
            )
        else:
            return self._bisect_find_zeros(
                sensitivity,
                start_time=start_time,
                stop_time=stop_time,
                target_precision=target_precision,
                max_iter=max_iter,
                current_iter=current_iter + 1,
                lowest_possible=midpoint,
                highest_possible=highest_possible,
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
        if min_energy is None or max_energy is None:
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
            print(e)

            self.seen = "error"
            self.error_message = str(e)

            return self.output()
