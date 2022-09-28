#!/usr/bin/env python
# Script name: gwobserve.py
# Author(s): B. Patricelli (barbara.patricelli@pi.infn.it),
#            J. Green (jarred.green@inaf.it),
#            A. Stamerra (antonio.stamerra.it)
# Based on: ObservingTimes.py --
#   Version: 5.0 (August 2020)
#   Author(s): B. Patricelli (barbara.patricelli@pi.infn.it)

# imports
import warnings

import numpy as np
import pandas as pd
import scipy
import scipy.stats
from astropy.io import fits
from matplotlib import pyplot as plt
from scipy import integrate
from scipy.interpolate import RegularGridInterpolator, interp1d


warnings.filterwarnings("ignore")  # surpress warnings when using on the command line
warnings.simplefilter("ignore", np.RankWarning)

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# classes
class Sensitivity:
    def __init__(self, grbsens_file: str, min_energy: float, max_energy: float) -> None:

        self.output = self.fit_grbsens(grbsens_file)
        self.energy_limits = (min_energy, max_energy)

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

        return result

    def get(self, t):

        slope, intercept = (
            self.output.slope,
            self.output.intercept,
        )

        return 10 ** (slope * np.log10(t) + intercept)


class GRB:
    def __init__(
        self,
        filepath: str,
        min_energy: str = None,
        max_energy: str = None,
    ) -> None:

        self.filepath = filepath
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

            self.time = datatime.field(0)
            self.energy = dataenergy.field(0)

            self.spectra = np.array(
                [datalc.field(i) for i, e in enumerate(self.energy)]
            )

        # set spectral grid
        self.set_spectral_grid()

        # fit spectral indices
        self.fit_spectral_indices()

        logging.debug(f"Loaded event {self.angle}º")

    def __repr__(self):
        return f"<GRB(run={self.run}, id={self.id})>"

    def set_spectral_grid(self):

        self.SpectralGrid = RegularGridInterpolator(
            (np.log10(self.energy), np.log10(self.time)), self.spectra
        )

    def show_spectral_pattern(self, resolution=100):

        self.set_spectral_grid()

        loge = np.around(np.log10(self.energy), 1)
        logt = np.around(np.log10(self.time), 1)

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

    def get_spectrum(self, time, energy=None):

        if not energy:
            energy = self.energy

        if hasattr(energy, "__len__"):
            return np.array(
                [self.SpectralGrid((e, np.log10(time))) for e in np.log10(energy)]
            )
        else:
            return self.SpectralGrid((np.log10(energy), np.log10(time)))

    def get_flux(self, energy, time=None):

        if not time:
            time = self.time

        if hasattr(time, "__len__"):
            return np.array(
                [self.SpectralGrid((np.log10(energy), t)) for t in np.log10(time)]
            )
        else:
            return self.SpectralGrid((np.log10(energy), np.log10(time)))

    def fit_spectral_indices(self):

        spectra = self.spectra.T

        indices = []
        times = []
        bad_times = []

        for spectrum, time in zip(spectra, self.time):

            idx = np.isfinite(spectrum) & (spectrum > 0)

            if len(idx[idx] > 3):  # need at least 3 points in the spectrum to fit

                times.append(time)
                indices.append(
                    np.polyfit(np.log10(self.energy[idx]), np.log10(spectrum[idx]), 1)[
                        0
                    ]
                )
            else:
                bad_times.append(time)

        self._indices = indices
        self._index_times = times
        self._bad_index_times = bad_times

        self.index_at = interp1d(
            np.log10(self._index_times),
            self._indices,
            fill_value="extrapolate",
        )

    def get_spectral_index(self, time):

        return self.index_at(np.array([np.log10(time)]))[0]

    def show_spectral_evolution(self, resolution=100):

        self.fit_spectral_indices()

        t = np.linspace(
            np.log10(min(self.time)), np.log10(max(self.time)), resolution + 1
        )

        plt.plot(t, self.index_at(t))
        plt.xlabel("Log(t) (s)")
        plt.ylabel("Spectral Index")

        plt.show()

    def get_integral_spectrum(self, time, first_energy_bin):

        if not self.min_energy or not self.max_energy:
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

    def get_fluence(self, start_time, stop_time):

        first_energy_bin = min(self.energy)

        fluence = integrate.quad(
            lambda time: self.get_integral_spectrum(time, first_energy_bin),
            start_time,
            stop_time,
        )[0]

        logging.debug(f"    Fluence: {fluence}")
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
            "_bad_index_times",
            "index_at",
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
                elif isinstance(v, np.ndarray):
                    o[k] = v.tolist()
                else:
                    o[k] = v

        return o

    def check_if_visible(self, sensitivity: Sensitivity, start_time, stop_time):

        # Interpolation and integration of the flux with time
        average_flux = self.get_fluence(start_time, stop_time) / (
            stop_time - start_time
        )

        # calculate photon flux
        photon_flux = sensitivity.get(t=(stop_time - start_time))

        visible = True if average_flux > photon_flux else False

        logging.debug(
            f"    visible:{visible} avgflux={average_flux}, photon_flux={photon_flux}"
        )

        return visible

    def observe(
        self,
        sensitivity: Sensitivity,
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
                        logging.debug(
                            f"    obs_time={observation_time} end_time={end_time}"
                        )
                        break

                    elif observation_time == precision:
                        # reduce precision
                        precision = 10 ** (int(np.log10(precision)) - 1)
                        observation_time = precision
                        logging.debug(f"    Updating precision to {precision}")

                    else:  # reduce precision but add more time
                        precision = 10 ** (int(np.log10(precision)) - 1)
                        observation_time = previous_observation_time + precision
                        logging.debug(
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


def observe_grb(
    grb_file_path,
    sensitivity: Sensitivity,
    start_time: float = 0,
    max_time=None,
    target_precision=1,
    _max_loops=10000,
):
    """Modified version to increase timestep along with time size"""

    grb = GRB(grb_file_path)

    return grb.observe(
        sensitivity,
        start_time,
        max_time=max_time,
        target_precision=target_precision,
        _max_loops=_max_loops,
    )
