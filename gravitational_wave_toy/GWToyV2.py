#!/usr/bin/env python
# Script name: GWToyV2.py
# Author(s): B. Patricelli (barbara.patricelli@pi.infn.it),
#            J. Green (jarred.green@inaf.it),
#            A. Stamerra (antonio.stamerra.it)
# Based on: ObservingTimes.py --
#   Version: 5.0 (August 2020)
#   Author(s): B. Patricelli (barbara.patricelli@pi.infn.it)

# imports
import glob
import logging

import numpy as np
import pandas as pd
import ray
import scipy
import scipy.stats
import yaml
from astropy.io import fits
from scipy import integrate
from scipy.interpolate import RectBivariateSpline
from tqdm.auto import tqdm

# activaate logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

        # set site and zenith
        self.rng = np.random.default_rng(
            int(self.id) * 10000 + int(self.run) + random_seed
        )

        self.site = self.rng.choice(sites)
        self.zenith = self.rng.choice(zeniths)

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

        logger.debug(f"    Fluence: {fluence}")
        return fluence

    def get_spectral_index(self, time):

        spectrum = self.get_spectrum(time)

        idx = np.isfinite(spectrum) & (spectrum > 0)

        return np.polyfit(np.log10(self.energy[idx]), np.log10(spectrum[idx]), 1)[0]

    def output(self):

        keys_to_drop = ["time", "energy", "spectra", "spectrum", "rng"]
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


def observe_grb(
    grb_file_path,
    sensitivity: Sensitivities,
    start_time: float = 0,
    max_time=None,
    zeniths=[20, 40, 60],
    sites=["south", "north"],
    energy_limits=[30, 10000],
    random_seed=1,
    precision=1,
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

    # get energy limits
    grb.min_energy, grb.max_energy = sensitivity.get_energy_limits(grb.site, grb.zenith)

    # set default max time
    if max_time is None:
        max_time = 86400 + start_time  # 24h after starting observations
    else:
        max_time += start_time

    # start the procedure
    dt = 1
    grb.start_time = start_time

    n = 1  # keeps track of loops carried out at current scale
    previous_dt = -1
    original_tstart = start_time
    t = start_time + n
    previous_t = t
    autostep = True

    while t < max_time:  # second loop from 1 to max integration time

        logger.debug(
            f"NEW LOOP; t={t:.2f}, dt={dt:.2f}, previous_t={previous_t:.2f}, previous_dt={previous_dt:.2f} n={n:.2f}"
        )

        if autostep:

            dt = 10 ** int(np.floor(np.log10(t)))

            logger.debug(f"    AUTOSTEP; t={t} dt={dt}")
            if dt != previous_dt:  # if changing scale, reset n
                n = 1
                logger.debug(f"    AUTOSTEP; resetting n")

        t = start_time + n * dt  # tstart = 210, + loop number
        obst = t - original_tstart  # how much actual observing time has gone by
        logger.debug(
            f"    Updating t: t: {t:.2f}, obs_t: {obst:.2f} start_time: {start_time:.2f}, n: {n:.2f}, dt: {dt:.2f}"
        )

        # Interpolation and integration of the flux with time
        average_flux = grb.get_fluence(original_tstart, t) / obst

        # calculate photon flux
        photon_flux = sensitivity.get(t=obst, site=grb.site, zenith=grb.zenith)

        logger.debug(
            f"    t={t:.2f}, dt={dt:.2f}, avgflux={average_flux}, photon_flux={photon_flux}"
        )

        if average_flux > photon_flux:  # if it is visible:
            logger.debug(
                f"\nClose solution, t={round(t, precision)}, avgflux={average_flux}, photon_flux={photon_flux}"
            )

            if dt > (10 ** (-1 * precision)):

                # if the desired precision is not reached, go deeper
                # change scale of dt and go back a step
                autostep = False
                start_time = previous_t
                dt = dt / 10
                n = 0

            else:
                # desired prevision is reached! Give solution

                tend = original_tstart + round(obst, precision)
                grb.end_time = round(tend, precision)
                grb.obs_time = round(obst, precision)
                grb.seen = True
                # print(f"dt={dt}, tend={tend}, obst={round(obst,precision)}")

                break

        else:
            # if not seen yet, keep going

            previous_dt = dt
            previous_t = t
            n = n + 0.1
            logger.debug(f"    Updating n: {n:.2f}")

    return pd.DataFrame(grb.output(), index=[f"{grb.id}_{grb.run}"])


def run():

    logger.info("Welcome to GWToy for CTA, for use with catalogue v1.")

    # load in the settings
    with open("./gw_settings.yaml") as file:
        logger.info("Settings file found!")
        parsed_yaml_file = yaml.load(file, Loader=yaml.FullLoader)

    catalog_directory = parsed_yaml_file["catalog"]
    file_list = glob.glob(catalog_directory)
    n_grbs = parsed_yaml_file["grbs_to_analyze"]

    n_cores = parsed_yaml_file.get("ncores")
    grbsens_files = parsed_yaml_file["grbsens_files"]
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

    if not precision:
        precision = 2

    # determine which grbs to analyze
    if n_grbs:
        first_index, last_index = 0, n_grbs - 1
    else:
        if not first_index:
            first_index = 0
        if not last_index:
            last_index = len(file_list)

    logger.info(
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
    logger.info("Starting ray:")
    ray.init(num_cpus=n_cores)
    observe_grb_remote = ray.remote(observe_grb)

    total_runs = len(range(first_index, last_index)) * len(zeniths) * len(time_delays)

    logger.info(f"Running {total_runs} observations")
    # set up each observation
    grb_object_ids = [
        observe_grb_remote.remote(
            grb_file_path,
            sensitivity,
            start_time=delay,
            zeniths=zeniths,
            sites=sites,
            random_seed=random_seed,
            precision=precision,
        )
        for grb_file_path in file_list[first_index:last_index]
        for delay in sorted(time_delays, reverse=True)
    ]

    for _ in tqdm(_to_iterator(grb_object_ids), total=total_runs):
        pass

    # run the observations
    grb_dfs = []
    for obj_id in grb_object_ids:
        grb_dfs.append(ray.get(obj_id))

    logger.info("Done observing!\nCreating file output.")

    # create the final pandas dataframe and write to a csv
    final_table = pd.concat(grb_dfs)
    final_table.to_csv(output_filename, index=False)
    logger.info(f"Saved csv: {output_filename}")
    pickle_filename = output_filename.split(".")[0] + ".pkl"
    final_table.to_pickle(pickle_filename)
    logger.info(f"Saved pandas dataframe: {pickle_filename}")

    ray.shutdown()

    logger.info("All done!")


if __name__ == "__main__":

    run()
