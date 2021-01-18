#!/usr/bin/env python
# Script name: GWToy.py
# Author(s): B. Patricelli (barbara.patricelli@pi.infn.it),
#            J. Green (jarred.green@inaf.it),
#            A. Stamerra (antonio.stamerra.it)
# Based on: ObservingTimes.py --
#   Version: 5.0 (August 2020)
#   Author(s): B. Patricelli (barbara.patricelli@pi.infn.it)


# Imports
import os, sys
from optparse import OptionParser
from astropy.io import fits

import random

from scipy.interpolate import interp1d
from scipy import integrate


import numpy as np
import pandas as pd

# definitions


def ParseGrbsens(catFileName, Separator="\t", names=None, orient="list"):
    as_dict = pd.read_csv(catFileName, sep=Separator, comment="#", names=names).to_dict(
        orient=orient
    )
    return as_dict


def ParseBNS(filename, sep=" ", names=None, orient="index"):
    as_dict = pd.read_csv(
        filename, sep=sep, quotechar='"', comment="#", names=names
    ).to_dict(orient=orient)
    return as_dict


# open grbsens files and create interpolation class:
def get_interpolation(filename: str):
    col_names = ["obs_time", "crab_flux", "photon_flux", "energy_flux", "sensitivity"]
    sensi_list = ParseGrbsens(filename, names=col_names, orient="list")

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


def bns_stats(df=None, input_file=None):
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


#######################################################

# Get Script name
ScriptName = os.path.split(sys.argv[0])[1].split(".")[0]

#######################################################
# Main
#######################################################

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

        # Spectrum
        def spectrum(x):
            return (x / 1) ** (-2.1)

        # Integral of the spectrum

        lower, upper = get_energy_limits(zenith)

        intl, errl = integrate.quad(lambda x: spectrum(x), lower, upper)  # GeV

        # Interpolation of the flux with time
        flux = interp1d(time, lc)

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
