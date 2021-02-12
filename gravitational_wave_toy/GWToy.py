#!/usr/bin/env python
# Script name: GWToy.py
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

from scipy.interpolate import interp1d
from scipy import integrate
import scipy.stats
import scipy

import ray
from tqdm.auto import tqdm

from astropy.io import fits
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

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


def interpolate_grbsens(x: float, direction: str, zenith: int, interpoations: dict):

    try:
        return interpolations[direction.lower()][zenith](x)
    except KeyError:
        raise AttributeError(f"No {direction} z{zenith} file found!")


def fit_grbsens(filepath: str):
    grbsens = open_grbsens(filepath)

    result = scipy.stats.linregress(np.log10(grbsens["obs_time"]), np.log10(grbsens["photon_flux"]))

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


def open_v1_fits(filepath: str):
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


def observe_grb(bns_index: int, bns_dict: dict, fit_dict: dict, integral_dict: dict, tstart: float = 0,
                     max_time=None, precision=1, observatory=None, zenith=None):
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

            if dt > 10 ** (-1 * precision):

                # if the desired precision is not reached, go deeper
                # change scale of dt and go back a step
                autostep = False
                tstart = previous_t
                dt = dt / 10
                n = 1

            else:
                # desired prevision is reached! Give solution

                tend = original_tstart + obst
                grb["tend"] = tend
                grb["obstime"] = obst
                grb["seen"] = True

                # print(f"\n\nSeen! Obs time: {obst}, End time: {tend}")

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

    usg = "\033[1;31m%prog [ options] inputFile\033[1;m \n"

    desc = "\033[34mThis is a sample script\033[0m \n"

    parser = OptionParser(description=desc, usage=usg)

    parser.add_option(
        "-i",
        "--InputBNS",
        default="BNS-GW-Time_onAxis5deg-final.txt",
        help="File with the BNS mergers",
    )
    parser.add_option(
        "-d", "--deltat", type="float", default=1.0, help="time bin width (s)"
    )
    parser.add_option(
        "-t", "--inttime", type="int", default=10210, help="maximum integration time"
    )

    parser.add_option(
        "-l", "--lowerenergy", type=float, default=20, help="lower energy limit (GeV)"
    )
    parser.add_option(
        "-m",
        "--higherenergy",
        type=float,
        default=10000,
        help="higher energy limit (GeV)",
    )

    parser.add_option("-u", "--initialN", type=int, default=0, help="initial BNS")
    parser.add_option("-v", "--finalN", type=int, default=1, help="final BNS")

    (options, args) = parser.parse_args()
    if len(args) != 0:
        parser.error("incorrect number of arguments. Use -h for help")

    InputData = options.InputBNS
    dt = options.deltat
    inttime = options.inttime
    x1 = options.lowerenergy
    x2 = options.higherenergy

    ini = options.initialN
    fin = options.finalN

    #####################################################

    print("************************************")
    print("** " + ScriptName)
    print("************************************")

    print("** Options:")

    # List parameters and args
    print("\n**Input parameters:")
    for key, val in parser.values.__dict__.items():
        print(key, ":", val)
        if val is None:
            print(
                "\nERROR!",
                key,
                "is a required option! Exit.\nType ",
                ScriptName + ".py -h for help\n",
            )
            sys.exit(1)

    #######################################################################

    #########################################
    # Creating the output directories with the obs time
    os.system("mkdir -p ObsTimes_toy")
    #########################################

    #########################################
    # Create the 1D interpolation classes
    files = {
        "north": {
            20: "grbsens-5.0sigma_t1s-t16384s_irf-North_z20_0.5h.txt",
            40: "grbsens-5.0sigma_t1s-t16384s_irf-North_z40_0.5h.txt",
            60: "grbsens-5.0sigma_t1s-t16384s_irf-North_z60_0.5h.txt",
        },
        "south": {
            20: "grbsens-5.0sigma_t1s-t16384s_irf-South_z20_0.5h.txt",
            40: "grbsens-5.0sigma_t1s-t16384s_irf-South_z40_0.5h.txt",
            60: "grbsens-5.0sigma_t1s-t16384s_irf-South_z60_0.5h.txt",
        },
    }

    interpolations = get_interpolation_dict(files)

    # Reading the file with the selection of BNS mergers

    bns = ParseBNS(InputData)

    # create output file
    outfile = f"ObsTimes_toy/obstime.csv"

    output_cols = [
        "run",
        "MergerID",
        "alt",
        "observatory",
        "tstart",
        "tend",
        "obstime",
        "seen",
    ]
    output = pd.DataFrame(columns=output_cols)

    for j in range(ini, fin):

        run = bns[j]["run"]
        merger_id = bns[j]["MergerID"]
        zenith = bns[j]["Mean Altitude"]
        site = bns[j]["Observatory"]

        if "merger" in merger_id.lower():
            merger_id = merger_id[6:]

        new_row = {
            "run": run,
            "MergerID": merger_id,
            "alt": zenith,
            "observatory": site,
            "tstart": 0,
            "tend": 0,
            "obstime": 0,
            "seen": False,
        }

        #######################################################
        # Defining which sensitivity to use
        #######################################################

        ####### -> -> ->
        ####### Here we have the new fs function, and I added the case zenith=60 deg

        print(f"#{j}| BNS: {merger_id}, Cite: {site}, Zentih: {zenith}, ", end="")

        #######################################################
        # Associating a GRB to the BNS mergers
        #######################################################

        InputGRB = f"GammaCatalogV1.0/{run}_{merger_id}.fits"

        try:
            hdu_list = fits.open(InputGRB)
            # print(f"Found file {InputGRB}")
            #    hdu_list.info()
        except FileNotFoundError:
            print(f"Input GRB {InputGRB} not found.")
            continue

        datalc = hdu_list[3].data
        datatime = hdu_list[2].data
        dataenergy = hdu_list[1].data

        lc = datalc.field(0)
        time = datatime.field(0)
        energy = dataenergy.field(0)
        spec = datalc[0]
        #    Norm=spec[0]

        # open grb file
        grb = open_v1_fits(InputGRB)

        # Integral of the spectrum

        lower, upper = get_energy_limits(zenith)

        intl, errl = integrate.quad(lambda x: spectrum(x), lower, upper)  # GeV

        # Interpolation of the flux with time
        flux = interp1d(grb["time"], grb["lc"])

        #################################################
        # Starting the procedure
        #################################################

        # defining the starting time of observation

        tslew = 30  # pointing time (s)
        # talert = 180  # latency for the GW alert (s)
        # get random starting delay
        talert = random.randint(120, 600)  # between 2 and 10 mins

        tstart = talert + tslew  # (s)
        print(f"Delay: {tstart}, ", end="")

        dt = 1

        new_row["tstart"] = tstart

        # loop over first pointint only
        # for n in range(1, inttime):  # loop from 1 to max integration time

        for m in range(1, inttime):  # second loop from 1 to max integration time

            t = tstart + m * dt  # tstart = 210, + loop number
            obst = t - tstart  # how much actual observing time has gone by

            fluencen, errorn = integrate.quad(lambda x: flux(x), tstart, t)
            averagefluxn = fluencen * intl / obst  # ph/cm2/s

            photon_flux = interpolate_grbsens(obst, site, zenith, interpolations)

            if averagefluxn > photon_flux:  # if it is visible:

                if tstart + obst < inttime:  # if starting time < max time, write
                    tend = tstart + obst
                    new_row["tend"] = tend
                    new_row["obstime"] = obst
                    new_row["seen"] = True

                tstart = tstart + obst

                break

        print(f"Seen: {new_row['seen']}")
        output = output.append(new_row, ignore_index=True)

        # if tstart >= inttime:
        #     break

    # print and save final results
    print("\nFinal results:")
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(output)

    print(f"\nSaving file to {outfile}")
    output.to_csv(outfile, sep="\t", index=False)
