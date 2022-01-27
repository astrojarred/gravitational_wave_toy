#!/usr/bin/env python
# Script name: GWToyV2.py
# Author(s): B. Patricelli (barbara.patricelli@pi.infn.it),
#            J. Green (jarred.green@inaf.it),
#            A. Stamerra (antonio.stamerra.it)
# Based on: ObservingTimes.py --
#   Version: 5.0 (August 2020)
#   Author(s): B. Patricelli (barbara.patricelli@pi.infn.it)

# imports
import warnings

warnings.filterwarnings("ignore")  # surpress warnings when using on the command line

import glob
import logging
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore", np.RankWarning)

import pandas as pd
import ray
import scipy
import scipy.stats
import yaml
from astropy.io import fits
from scipy import integrate
from scipy.interpolate import interp1d, RectBivariateSpline
from tqdm.auto import tqdm

# activaate logger
logging.basicConfig(level=logging.INFO)

# classes
class Sensitivities:
    def __init__(self, grbsens_files: dict, energy_limits: dict) -> None:

        self.output = {}
        self.energy_limits = energy_limits

        # make interpolations of the sensitivity curves
        for site, zeniths in grbsens_files.items():
            self.output[site] = {}
            for zenith, file in zeniths.items():
                self.output[site][zenith] = self.fit_grbsens(file)

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

    def get(self, t, site, zenith):

        slope, intercept = (
            self.output[site][zenith].slope,
            self.output[site][zenith].intercept,
        )

        return 10 ** (slope * np.log10(t) + intercept)

    def get_energy_limits(self, site, zenith):

        return (
            self.energy_limits[site.lower()][int(zenith)]["min"],
            self.energy_limits[site.lower()][int(zenith)]["max"],
        )


class GRB:
    def __init__(
        self,
        filepath: str,
        random_seed=0,
        zeniths=[20, 40, 60],
        sites=["south", "north"],
        energy_limits=[30, 10000],
    ) -> None:

        self.zenith = 0
        self.site = "south"
        self.min_energy, self.max_energy = energy_limits
        self.seen = False
        self.obs_time = -1
        self.start_time = -1
        self.end_time = -1

        with fits.open(filepath) as hdu_list:

            self.run = hdu_list[0].header["RUN"]
            self.id = hdu_list[0].header["MERGERID"]
            self.ra = hdu_list[0].header["RA"]
            self.dec = hdu_list[0].header["DEC"]
            self.eiso = hdu_list[0].header["EISO"]
            self.z = hdu_list[0].header["REDSHIFT"]
            self.angle = hdu_list[0].header["ANGLE"]

            datalc = hdu_list[3].data
            datatime = hdu_list[2].data
            dataenergy = hdu_list[1].data

            self.time = datatime.field(0)
            self.energy = dataenergy.field(0)

            self.spectra = np.nan_to_num(
                np.array([datalc.field(i) for i, e in enumerate(self.energy)])
            )

        # get interpolation
        self.spectrum = RectBivariateSpline(self.energy, self.time, self.spectra)

        self.power_law_slopes = np.array(
            [self.fit_spectral_index(time) for time in self.time]
        )
        slopes_idx = np.isfinite(self.power_law_slopes)
        self.spectral_indices = interp1d(
            self.time[slopes_idx],
            self.power_law_slopes[slopes_idx],
            fill_value="extrapolate",
        )

        # set site and zenith
        self.rng = np.random.default_rng(
            int(self.id) * 10000 + int(self.run) + random_seed
        )

        self.site = self.rng.choice(sites)
        self.zenith = self.rng.choice(zeniths)

        logging.debug(
            f"Got GRB run{self.run}_ID{self.id}, {self.site}, z{self.zenith}, {self.angle}º"
        )

    def get_spectrum(self, time, energy=None):

        if not energy:
            energy = self.energy

        if hasattr(energy, "__len__"):
            return np.array([i[0] for i in (lambda e: self.spectrum(e, time))(energy)])
        else:
            return (lambda e: self.spectrum(e, time))(energy)[0][0]

    def get_flux(self, energy, time=None):

        if not time:
            time = self.time

        if hasattr(time, "__len__"):
            return (lambda t: self.spectrum(energy, t))(time)[0]
        else:
            return (lambda t: self.spectrum(energy, t))(time)[0][0]

    def get_spectral_index(self, time):

        return self.spectral_indices(np.array([time]))[0]

    def power_law(self, energy, spectral_index=-2.1, energy_0=None, normalization=1):

        if not energy_0:
            energy_0 = min(self.energy)

        return normalization * (energy / energy_0) ** (spectral_index)

    def get_integral_spectrum(self, time, min_energy, max_energy):

        integral_spectrum = integrate.quad(
            lambda energy: self.power_law(
                energy,
                spectral_index=self.get_spectral_index(time),
                energy_0=min(self.energy),
                normalization=self.get_flux(energy=min(self.energy), time=time),
            ),
            min_energy,
            max_energy,
        )[0]

        return integral_spectrum

    def get_fluence(self, start_time, stop_time, min_energy=None, max_energy=None):

        if not min_energy:
            min_energy = self.min_energy
        if not max_energy:
            max_energy = self.max_energy

        fluence = integrate.quad(
            lambda time: self.get_integral_spectrum(time, min_energy, max_energy),
            start_time,
            stop_time,
        )[0]

        logging.debug(f"    Fluence: {fluence}")
        return fluence

    def get_fast_integral_spectrum(self, time, first_energy_bin):

        spectral_index = self.get_spectral_index(time)
        spectral_index_plus_one = spectral_index + 1

        integral_spectrum = (
            self.get_flux(first_energy_bin, time=time)
            * (first_energy_bin ** (-spectral_index) / spectral_index_plus_one)
            * (
                (self.max_energy ** spectral_index_plus_one)
                - (self.min_energy ** spectral_index_plus_one)
            )
        )

        return integral_spectrum

    def get_fast_fluence(self, start_time, stop_time):

        first_energy_bin = min(self.energy)

        fluence = integrate.quad(
            lambda time: self.get_fast_integral_spectrum(time, first_energy_bin),
            start_time,
            stop_time,
        )[0]

        logging.debug(f"    Fluence: {fluence}")
        return fluence

    def fit_spectral_index(self, time, cut=0):

        spectrum = self.get_spectrum(time)
        energy = self.energy

        if cut > 0:
            spectrum = spectrum[:-cut]
            energy = energy[:-cut]

        idx = np.isfinite(spectrum) & (spectrum > 0)

        try:
            spectral_index = np.polyfit(
                np.log10(energy[idx]), np.log10(spectrum[idx]), 1
            )[0]
        except TypeError or ValueError:
            logging.debug(f"Spectral index fitting issue at t={time}")
            spectral_index = np.nan

        return spectral_index

    def output(self):

        keys_to_drop = [
            "time",
            "energy",
            "spectra",
            "spectrum",
            "rng",
            "power_law_slopes",
            "spectral_indices",
        ]
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in keys_to_drop
        }


def _to_iterator(obj_ids):
    """To decide when to update tqdm bar
    See: https://github.com/ray-project/ray/issues/5554#issuecomment-558397627"""

    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        try:
            yield ray.get(done[0])
        except RuntimeError:
            yield 1


def check_if_visible(grb: GRB, sensitivity: Sensitivities, start_time, stop_time):

    # Interpolation and integration of the flux with time
    average_flux = grb.get_fast_fluence(start_time, stop_time) / (
        stop_time - start_time
    )

    # calculate photon flux
    photon_flux = sensitivity.get(
        t=(stop_time - start_time), site=grb.site, zenith=grb.zenith
    )

    visible = True if average_flux > photon_flux else False

    logging.debug(
        f"    visible:{visible} avgflux={average_flux}, photon_flux={photon_flux}"
    )

    return visible


def observe_grb(
    grb_file_path,
    sensitivity: Sensitivities,
    log_directory=None,
    start_time: float = 0,
    max_time=None,
    max_angle=360,
    zeniths=[20, 40, 60],
    sites=["south", "north"],
    energy_limits=[30, 10000],
    random_seed=1,
    target_precision=1,
    read=True,
):
    """Modified version to increase timestep along with time size"""

    # load GRB data
    grb = GRB(
        grb_file_path,
        random_seed=random_seed,
        zeniths=zeniths,
        sites=sites,
        energy_limits=energy_limits,
    )

    # check for angle
    if grb.angle > max_angle:
        return None

    # check for file already existing
    log_filename = f"{log_directory}/run{grb.run}_ID{grb.id}_{start_time}_{grb.site}_z{grb.zenith}.csv"

    if read:
        if Path(log_filename).exists():
            logging.debug(f"Output already exists: {log_filename}")
            return pd.read_csv(log_filename, index_col=0)

    # get energy limits
    grb.min_energy, grb.max_energy = sensitivity.get_energy_limits(grb.site, grb.zenith)

    # start the procedure
    grb.start_time = start_time
    delay = start_time

    # set default max time
    if max_time is None:
        max_time = 43200  # 12h after starting observations

    # check maximum time
    logging.debug(f"Checking if visible is observed for maximum time")
    visible = check_if_visible(grb, sensitivity, delay, max_time + delay)

    # not visible even after maximum observation time
    if not visible:
        logging.debug(f"GRB not visible after {max_time+delay}s with {delay}s delay")
        df = pd.DataFrame(grb.output(), index=[f"{grb.id}_{grb.run}"])
        df.to_csv(log_filename)
        return df

    loop_number = 0
    precision = int(10 ** int(np.floor(np.log10(max_time + delay))))
    observation_time = precision
    previous_observation_time = precision

    # find the inflection point
    while loop_number < 10000:

        loop_number += 1
        logging.debug(
            f"Starting new loop #{loop_number}; observation_time {observation_time}, precision {precision}"
        )

        visible = check_if_visible(grb, sensitivity, delay, delay + observation_time)

        if visible:

            logging.debug(
                f"    GRB Visible at obs_time={observation_time} end_time={delay + observation_time}"
            )

            # if desired precision is reached, return results and break!
            if np.log10(precision) == np.log10(target_precision):
                round_precision = int(-np.log10(precision))
                end_time = delay + round(observation_time, round_precision)
                grb.end_time = round(end_time, round_precision)
                grb.obs_time = round(observation_time, round_precision)
                grb.seen = True
                logging.debug(f"    obs_time={observation_time} end_time={end_time}")
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

    df = pd.DataFrame(grb.output(), index=[f"{grb.id}_{grb.run}"])
    df.to_csv(log_filename)

    return log_filename


def run():

    logging.info("Welcome to GWToy for CTA, for use with catalogue v1.")

    # load in the settings
    with open("./gw_settings.yaml") as file:
        logging.info("Settings file found!")
        parsed_yaml_file = yaml.load(file, Loader=yaml.FullLoader)

    catalog_directory = parsed_yaml_file["catalog"]
    file_list = glob.glob(f"{catalog_directory}/*.fits")
    log_directory = parsed_yaml_file.get("log_directory")
    n_grbs = parsed_yaml_file.get("grbs_to_analyze")

    n_cores = parsed_yaml_file.get("ncores")
    grbsens_files = parsed_yaml_file["grbsens_files"]
    maximum_angle = parsed_yaml_file.get("maximum_angle")
    first_index = parsed_yaml_file.get("first_index")
    last_index = parsed_yaml_file.get("last_index")
    output_filename = parsed_yaml_file["output_filename"]
    zeniths = parsed_yaml_file["zeniths"]
    sites = parsed_yaml_file["sites"]
    time_delays = parsed_yaml_file["time_delays"]
    precision = parsed_yaml_file.get("precision")
    random_seed = parsed_yaml_file.get("random_seed")
    energy_limits = parsed_yaml_file["energy_limits"]

    # parse inputs
    if not n_cores:
        n_cores = 1

    # create the log directory if needed
    if not log_directory:
        log_directory = "./gw_toy_logs"
    Path(log_directory).mkdir(parents=True, exist_ok=True)

    if not precision:
        precision = 2

    if not maximum_angle:
        maximum_angle = 360

    # determine which grbs to analyze
    if n_grbs:
        first_index, last_index = 0, n_grbs
    else:
        if not first_index:
            first_index = 0
        if not last_index:
            last_index = len(file_list)

        n_grbs = last_index - first_index

    logging.info(
        f"Settings:\n"
        f"  - {n_cores} cores\n"
        f"  - output filename: {output_filename}\n"
        f"  - zenith angles: {zeniths}\n"
        f"  - time delays (s): {time_delays}\n"
        f"  - decimal precision: {precision}\n"
        f"  - # mergers to analyze: {len(range(first_index, last_index))}\n"
    )

    # generate look-up dictionary of fits of the sensitivities
    sensitivity = Sensitivities(grbsens_files, energy_limits)

    # initialize ray and create remote solver
    logging.info("Starting ray:")
    ray.init(num_cpus=n_cores, log_to_driver=False, logging_level=logging.FATAL)
    observe_grb_remote = ray.remote(observe_grb)

    total_runs = n_grbs * len(time_delays)

    logging.info(f"Running {total_runs} observations")
    # set up each observation
    grb_object_ids = [
        observe_grb_remote.remote(
            grb_file_path,
            sensitivity,
            log_directory=log_directory,
            start_time=delay,
            zeniths=zeniths,
            sites=sites,
            random_seed=random_seed,
            target_precision=precision,
        )
        for grb_file_path in file_list[first_index:last_index]
        for delay in sorted(time_delays, reverse=True)
    ]

    for _ in tqdm(_to_iterator(grb_object_ids), total=total_runs):
        pass

    # run the observations
    logging.info("Done observing!\nCollecting csv filenames.")
    csvs = []
    for obj_id in tqdm(grb_object_ids, total=total_runs):
        if not isinstance(ray.get(obj_id), type(None)):
            csvs.append(ray.get(obj_id))

    logging.info("Done. Shutting down Ray.")
    ray.shutdown()

    logging.info("Creating the combined output")

    # create the final pandas dataframe and write to a csv
    dfs = []
    for filename in tqdm(csvs, total=total_runs):
        try:
            df = pd.read_csv(filename, index_col=0)
            dfs.append(df)
        except pd.errors.EmptyDataError:
            pass

    final_table = pd.concat(dfs, axis=0)

    logging.info("Saving files. ")
    final_table.to_csv(output_filename, index=False)
    logging.info(f"Saved csv: {output_filename}")
    pickle_filename = output_filename.split(".")[0] + ".pkl"
    final_table.to_pickle(pickle_filename)
    logging.info(f"Saved pandas dataframe: {pickle_filename}")

    logging.info("All done!")


if __name__ == "__main__":

    run()
