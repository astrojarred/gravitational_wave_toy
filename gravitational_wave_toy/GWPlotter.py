#!/usr/bin/env python
# Script name: GWPlotter.py
# Plot output csv files produced by GWToy.py
# Author(s): B. Patricelli (barbara.patricelli@pi.infn.it),
#            J. Green (jarred.green@inaf.it),
#            A. Stamerra (antonio.stamerra.it)
# Based on: ObservingTimes.py --
#   Version: 5.0 (August 2020)
#   Author(s): B. Patricelli (barbara.patricelli@pi.infn.it)

import yaml

import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm


def open_gw_file(filepath, filetype=None):

    if filetype is None:
        filetype = os.path.splitext(filepath)[-1]

    if filetype not in [".csv", ".pkl"]:
        raise AttributeError("Input file must either be a `csv` or pandas `pkl` file.")

    if filetype == ".csv":
        data = pd.read_csv(filepath)
    else:
        data = pd.read_pickle(filepath)

    return data


def analyze(input_data, site=None, zenith=None, obs_times=None):
    data = input_data

    if site:
        data = data[data["observatory"] == site.lower().capitalize()]

    if zenith:
        data = data[data["zenith"] == zenith]

    delays = np.unique(data["tstart"])

    if obs_times is None:
        obs_times = [2 ** x for x in range(0, 16)]

    results = {}
    results_df = pd.DataFrame(columns=["delay", "obs_time", "seen", "total"])
    for delay in delays:
        results[delay] = {}
        for obs_time in obs_times:
            new = {}
            results[delay][obs_time] = {}
            current_data = data[data["tstart"] == delay]
            results[delay][obs_time]["seen"] = len(current_data[(current_data["obstime"] <= obs_time ) & (current_data["obstime"] > 0)])
            results[delay][obs_time]["total"] = len(current_data)
            results[delay][obs_time]["percent"] = results[delay][obs_time]["seen"] / results[delay][obs_time]["total"]
            new["delay"] = delay
            new["obs_time"] = obs_time
            new["seen"] = results[delay][obs_time]["seen"]
            new["total"] = results[delay][obs_time]["total"]
            results_df = results_df.append(new, ignore_index=True)

    results_df["percent"] = results_df["seen"] / results_df["total"]
    return results_df


def convert_time(seconds: float):
    if seconds < 60:
        return f"{int(seconds):.0f}s"
    if seconds < 3600:
        return f"{int(seconds) / 60:.0f}m"
    else:
        return f"{seconds / 3600:.1f}h"

    return


def plot_toy(data, output_dir, site=None, zenith=None, obs_times=None, filetype="png"):
    sns.set_theme()

    df = analyze(data, site=site, zenith=zenith, obs_times=obs_times)

    df["percent"] = df["percent"] * 100

    pivot = df.pivot("delay", "obs_time", "percent").astype(float)

    f, ax = plt.subplots(figsize=(9, 9))
    heatmap = sns.heatmap(pivot, annot=True, fmt=".0f", linewidths=.5, ax=ax, cmap="viridis")

    if not site:
        site = "Both sites"
    else:
        site = f"CTA {site.capitalize()}"

    if not zenith:
        zenith = "all zeniths"
    else:
        zenith = f"z{zenith}"

    plt.title(f"GW/GRB Detectability for {site}, {zenith}")

    fig = heatmap.get_figure()
    fig.savefig(f"{output_dir}/GW_{site}_{zenith}.{filetype}")
    print(f"Saved plot {output_dir}/GW_{site}_{zenith}.{filetype}")


if __name__ == "__main__":

    # load in the settings
    with open("./plot_settings.yaml") as file:
        print("Plot settings file found!")
        parsed_yaml_file = yaml.load(file, Loader=yaml.FullLoader)

    input_file = parsed_yaml_file["input_file"]
    output_dir = parsed_yaml_file["output_directory"]
    sites = parsed_yaml_file["sites"]
    zeniths = parsed_yaml_file["zeniths"]
    obs_times = parsed_yaml_file["observation_times"]

    print(f"Making output directory {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    data = open_gw_file(input_file)
    print(f"Successfully loaded input file: {input_file}")

    n_plots = len(sites)*len(zeniths)

    loading_bar = tqdm(total=n_plots)

    for site in sites:
        for zenith in zeniths:
            plot_toy(data, output_dir, site=site, zenith=zenith, obs_times=obs_times)
            loading_bar.update(1)
