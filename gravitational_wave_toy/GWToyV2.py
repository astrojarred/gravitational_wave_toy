#!/usr/bin/env python
# Script name: GWToyV2.py
# Author(s): B. Patricelli (barbara.patricelli@pi.infn.it),
#            J. Green (jarred.green@inaf.it),
#            A. Stamerra (antonio.stamerra.it)
# Based on: ObservingTimes.py --
#   Version: 5.0 (August 2020)
#   Author(s): B. Patricelli (barbara.patricelli@pi.infn.it)


# Imports
import os
import subprocess
import yaml

from scipy.interpolate import interp1d, RectBivariateSpline
from scipy import integrate
import scipy.stats
import scipy

import ray
from tqdm.auto import tqdm

from astropy.io import fits
import numpy as np
import pandas as pd

# definitions


def run_bash(command: str, show: bool = False):
    split_command = command.split(" ")
    output = subprocess.Popen(split_command, stdout=subprocess.PIPE).communicate()
    if show:
        print(output[0].decode("utf-8"))
    else:
        return output[0].decode("utf-8")


def ParseGrbsens(catFileName, Separator="\t", names=None, orient="list"):
    as_dict = pd.read_csv(catFileName, sep=Separator, comment="#", names=names).to_dict(
        orient=orient
    )
    return as_dict


def open_grbsens(filename: str):
    col_names = ["obs_time", "crab_flux", "photon_flux", "energy_flux", "sensitivity"]
    sensi_list = ParseGrbsens(filename, names=col_names, orient="list")

    return sensi_list


def ParseBNS(filename, sep=" ", names=None, orient="index"):
    as_dict = pd.read_csv(
        filename, sep=sep, quotechar='"', comment="#", names=names
    ).to_dict(orient=orient)
    return as_dict


# open grbsens files and create interpolation class:
def get_interpolation(filename: str):
    sensi_list = open_grbsens(filename)
    # interpolation, x-> obstime, y-> photon_flux
    interpolation = interp1d(sensi_list["obs_time"], sensi_list["photon_flux"])

    return interpolation


def get_interpolation_dict(file_dict: dict):
    for direction, zeniths in file_dict.items():
        for zenith, file in zeniths.items():
            file_dict[direction][zenith] = get_interpolation(
                file_dict[direction][zenith]
            )

    return file_dict


def interpolate_grbsens(x: float, direction: str, zenith: int, interpolations: dict):

    try:
        return interpolations[direction.lower()][zenith](x)
    except KeyError:
        raise AttributeError(f"No {direction} z{zenith} file found!")


def fit_grbsens(filepath: str):
    grbsens = open_grbsens(filepath)

    result = scipy.stats.linregress(
        np.log10(grbsens["obs_time"]), np.log10(grbsens["photon_flux"])
    )

    return result


def get_fit_dict(file_dict: dict):
    output = {}

    for direction, zeniths in file_dict.items():
        output[direction] = {}
        for zenith, file in zeniths.items():
            # get fit and write results
            result = fit_grbsens(file)
            output[direction][zenith] = result

    return output


def fit(t: float, fit_dict: dict, site: str, zenith: int):
    if zenith not in [20, 40, 60]:
        raise AttributeError("Zenith must be 20, 40, or 60.")

    if site.lower() not in ["south", "north"]:
        raise AttributeError("Site must be `south` or `north`.")

    result = fit_dict[site.lower()][zenith]

    # catch really low numbers
    if t < 0.1:
        t = 0.1

    return 10 ** (result.slope * np.log10(t) + result.intercept)


def fit_compact(t: float, slope: float, intercept: float):
    return 10 ** (slope * np.log10(t) + intercept)


def get_energy_limits(zenith: int):
    if zenith not in [20, 40, 60]:
        raise AttributeError("Zenith must be 20, 40, or 60.")
    if zenith == 20:
        lower, upper = 30, 10000
    elif zenith == 40:
        lower, upper = 40, 10000
    else:
        lower, upper = 110, 10000

    return lower, upper


def bns_stats(df: pd.DataFrame = None, input_file: str = None):
    """Print statistics about the data runs"""

    if df is None and input_file is None:
        raise AttributeError("Please proide either a dataframe or an input file.")

    if df:
        n_runs = len(df)
        df_20 = df[df["alt"] == 20]
        df_40 = df[df["alt"] == 40]
        df_60 = df[df["alt"] == 60]

        for data in [df_20, df_40, df_60]:
            print("")


def find_files(catalog_directory: str, ext: str = "fits"):
    fits_files = [
        os.path.abspath(f"{catalog_directory}/{f}")
        for f in os.listdir(os.path.abspath(catalog_directory))
        if f.endswith(f".{ext}")
    ]
    return fits_files


class GRB:
    def __init__(self, filepath: str):

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

            self.lc = datalc
            self.time = datatime.field(0)
            self.energy = dataenergy.field(0)
            self.min_energy = min(self.energy)
            self.max_energy = max(self.energy)
            self.spec = datalc[0]

            self.spectra = np.nan_to_num(
                np.array([datalc.field(i) for i, e in enumerate(self.energy)])
            )

            # get interpolation
            self.spectrum = RectBivariateSpline(self.energy, self.time, self.spectra)

    def get_spectrum(self, time, energy=None):

        if not energy:
            energy = self.energy

        if hasattr(energy, "__len__"):
            return [i[0] for i in (lambda e: self.spectrum(e, time))(energy)]
        else:
            return (lambda e: self.spectrum(e, time))(energy)[0][0]

    def get_lightcurve(self, energy, time=None):

        if not time:
            time = self.time

        if hasattr(time, "__len__"):
            return (lambda t: self.spectrum(energy, t))(time)[0]
        else:
            return (lambda t: self.spectrum(energy, t))(time)[0][0]

    try:

        grb = {}

        with fits.open(filepath) as hdu_list:

            grb["run"] = hdu_list[0].header["RUN"]
            grb["id"] = hdu_list[0].header["MERGERID"]
            grb["ra"] = hdu_list[0].header["RA"]
            grb["dec"] = hdu_list[0].header["DEC"]
            grb["eiso"] = hdu_list[0].header["EISO"]
            grb["z"] = hdu_list[0].header["REDSHIFT"]
            grb["angle"] = hdu_list[0].header["ANGLE"]

            datalc = hdu_list[3].data
            datatime = hdu_list[2].data
            dataenergy = hdu_list[1].data

            grb["lc"] = datalc.field(0)
            grb["time"] = datatime.field(0)
            grb["energy"] = dataenergy.field(0)
            grb["spec"] = datalc[0]

            return grb

    except FileNotFoundError:
        print(f"Input V1 GRB {filepath} not found.")


# Spectrum
def spectrum(x):
    # This is correct for on-axis GRBs, but not for off-axis
    # TODO: we'll have to implement some changes as not all GRBs were modelled with the same spectrum
    return (x / 1) ** (-2.1)


def get_integral_spectra(zeniths: list):
    output = {}

    for z in zeniths:
        output[z] = {}

        lower, upper = get_energy_limits(z)

        intl, errl = integrate.quad(lambda x: spectrum(x), lower, upper)

        output[z]["integral"] = intl
        output[z]["error"] = errl

    return output


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
    bns_index: int,
    bns_dict: dict,
    fit_dict: dict,
    integral_dict: dict,
    tstart: float = 0,
    max_time=None,
    precision=1,
    observatory=None,
    zenith=None,
):
    """Modified version to increase timestep along with time size"""

    # get run and merger ID

    run = bns_dict[bns_index]["run"]
    merger_id = bns_dict[bns_index]["MergerID"]
    if "merger" in merger_id.lower():
        merger_id = merger_id[6:]

    InputGRB = f"./GammaCatalogV1.0/{run}_{merger_id}.fits"

    # open grb file
    grb = open_v1_fits(InputGRB)

    # add other bns information:
    #    - later these can be replaces with functions to calculate them
    grb["zenith"] = bns_dict[bns_index]["Mean Altitude"]
    grb["observatory"] = bns_dict[bns_index]["Observatory"]
    if observatory:
        # force observatory if given as input
        grb["observatory"] = observatory.lower()
    if zenith:
        # force zenith if given as input
        grb["zenith"] = zenith

    z = grb["zenith"]
    site = grb["observatory"].lower()

    # set the grb to not be seen by default
    grb["seen"] = False
    grb["tend"] = -1
    grb["obstime"] = -1

    # Integral of the spectra

    integral_spectrum = integral_dict[z]["integral"]  # GeV

    sens_slope = fit_dict[site][z].slope
    sens_intercept = fit_dict[site][z].intercept

    # set default max time
    if max_time is None:
        max_time = 86400 + tstart  # 24h after starting observations
    else:
        max_time += tstart

    # start the procedure
    dt = 1
    grb["tstart"] = tstart

    # Interpolation of the flux with time
    flux = interp1d(grb["time"], grb["lc"], fill_value="extrapolate")

    n = 1  # keeps track of loops carried out at current scale
    previous_dt = -1
    original_tstart = tstart
    t = tstart + n
    previous_t = t
    autostep = True

    while t < max_time:  # second loop from 1 to max integration time

        if autostep:

            dt = 10 ** int(np.floor(np.log10(t)))

            if dt != previous_dt:  # if changing scale, reset n
                n = 1

        t = tstart + n * dt  # tstart = 210, + loop number
        obst = t - original_tstart  # how much actual observing time has gone by

        fluence = integrate.quad(flux, original_tstart, t)
        avg_flux = fluence[0] * integral_spectrum / obst

        # calculate photon flux
        photon_flux = fit_compact(obst, sens_slope, sens_intercept)

        # print(f"t={t:.2f}, dt={dt}, avgflux={avg_flux}, photon_flux={photon_flux}")

        if avg_flux > photon_flux:  # if it is visible:
            # print(f"\nClose solution, t={round(t, precision)}, avgflux={avg_flux}, photon_flux={photon_flux}")

            if dt > (10 ** (-1 * precision)):

                # if the desired precision is not reached, go deeper
                # change scale of dt and go back a step
                autostep = False
                tstart = previous_t
                dt = dt / 10
                n = 0

            else:
                # desired prevision is reached! Give solution

                tend = original_tstart + round(obst, precision)
                grb["tend"] = round(tend, precision)
                grb["obstime"] = round(obst, precision)
                grb["seen"] = True
                # print(f"dt={dt}, tend={tend}, obst={round(obst,precision)}")

                break

        else:
            # if not seen yet, keep going

            previous_dt = dt
            previous_t = t
            n = n + 0.1

    # print(f"Seen: {new_row['seen']}")
    del grb["lc"], grb["time"], grb["energy"], grb["spec"]
    return pd.DataFrame(grb, index=[bns_index])


if __name__ == "__main__":

    print("Welcome to GWToy for CTA, for use with catalogue v1.")

    # load in the settings
    with open("./gw_settings.yaml") as file:
        print("Settings file found!")
        parsed_yaml_file = yaml.load(file, Loader=yaml.FullLoader)

    n_cores = parsed_yaml_file["ncores"]
    files = parsed_yaml_file["grbsens_files"]
    bns_dict = ParseBNS(parsed_yaml_file["bns_file"])
    first_index = parsed_yaml_file["first_index"]
    last_index = parsed_yaml_file["last_index"]
    n_mergers = len(bns_dict)
    output_filename = parsed_yaml_file["output_filename"]
    zeniths = parsed_yaml_file["zeniths"]
    time_delays = parsed_yaml_file["time_delays"]
    precision = parsed_yaml_file["precision"]

    if not first_index:
        first_index = 0
    if not last_index:
        last_index = n_mergers

    print(
        f"Settings:\n"
        f"  - {n_cores} cores\n"
        f"  - output filename: {output_filename}\n"
        f"  - zenith angles: {zeniths}\n"
        f"  - time delays (s): {time_delays}\n"
        f"  - decimal precision: {precision}\n"
        f"  - # mergers to analyze: {len(range(first_index, last_index))}\n"
    )

    # generate look-up dictionary of fits of the sensitivities
    fit_dict = get_fit_dict(files)

    # generate integrated spectrum dict
    spectral_dict = get_integral_spectra(zeniths=zeniths)

    # initialize ray and create remote solver
    print("Starting ray:")
    ray.init(num_cpus=n_cores)
    observe_grb_remote = ray.remote(observe_grb)

    total_runs = len(range(first_index, last_index)) * len(zeniths) * len(time_delays)

    print(f"Running {total_runs} observations")
    # set up each observation
    grb_object_ids = [
        observe_grb_remote.remote(
            bns_index,
            bns_dict,
            fit_dict,
            spectral_dict,
            tstart=delay,
            zenith=z,
            precision=precision,
        )
        for bns_index in range(first_index, last_index)
        for z in zeniths
        for delay in sorted(time_delays, reverse=True)
    ]

    for _ in tqdm(_to_iterator(grb_object_ids), total=total_runs):
        pass

    # run the observations
    grb_dfs = []
    for obj_id in grb_object_ids:
        grb_dfs.append(ray.get(obj_id))

    print("Done observing!\nCreating file output.")

    # create the final pandas dataframe and write to a csv
    final_table = pd.concat(grb_dfs)
    final_table.to_csv(output_filename, index=False)
    print(f"Saved csv: {output_filename}")
    pickle_filename = output_filename.split(".")[0] + ".pkl"
    final_table.to_pickle(pickle_filename)
    print(f"Saved pandas dataframe: {pickle_filename}")

    ray.shutdown()

    print("All done!")
