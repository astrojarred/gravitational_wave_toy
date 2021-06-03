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

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm


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
        data = data[data["site"] == site.lower()]

    if zenith:
        data = data[data["zenith"] == zenith]

    delays = np.unique(data["start_time"])

    if obs_times is None:
        obs_times = [2 ** x for x in range(0, 16)]

    results = {}
    results_df = pd.DataFrame(columns=["delay", "obs_time", "seen", "total"])
    for delay in delays:
        results[delay] = {}
        for obs_time in obs_times:
            new = {}
            results[delay][obs_time] = {}
            current_data = data[data["start_time"] == delay]
            results[delay][obs_time]["seen"] = len(
                current_data[
                    (current_data["obs_time"] <= obs_time)
                    & (current_data["obs_time"] > 0)
                ]
            )
            results[delay][obs_time]["total"] = len(current_data)
            results[delay][obs_time]["percent"] = (
                results[delay][obs_time]["seen"] / results[delay][obs_time]["total"]
            )
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


def plot_toy(
    data,
    output_dir,
    annotate=False,
    site=None,
    zenith=None,
    obs_times=None,
    x_tick_labels="auto",
    y_tick_labels="auto",
    min_value=None,
    max_value=None,
    color_scheme="viridis",
    color_scale=None,
    as_percent=False,
    filetype="png",
    subtitle=None,
    show_only=False,
):
    sns.set_theme()

    if str(zenith).lower() == "all":
        zenith = None
    if site.lower() == "all":
        site = None

    df = analyze(data, site=site, zenith=zenith, obs_times=obs_times)
    df.rename(columns={"obs_time": "exposure time"}, inplace=True)

    if as_percent:
        df["percent"] = df["percent"] * 100

    pivot = df.pivot("exposure time", "delay", "percent").astype(float)

    f, ax = plt.subplots(figsize=(9, 9))

    cbar_kws = {"label": "Percentage of GRBs detected", "orientation": "vertical"}

    if color_scale == "log":
        from matplotlib.colors import LogNorm

        color_scale = LogNorm(vmin=min_value, vmax=max_value)

    if annotate:
        heatmap = sns.heatmap(
            pivot,
            annot=True,
            fmt=".0f",
            linewidths=0.5,
            ax=ax,
            cmap=color_scheme,
            vmin=min_value,
            vmax=max_value,
            xticklabels=x_tick_labels,
            yticklabels=y_tick_labels,
            cbar_kws=cbar_kws,
        )
    else:
        heatmap = sns.heatmap(
            pivot,
            annot=False,
            ax=ax,
            cmap=color_scheme,
            vmin=min_value,
            vmax=max_value,
            xticklabels=x_tick_labels,
            yticklabels=y_tick_labels,
            cbar_kws=cbar_kws,
        )

    heatmap.invert_yaxis()

    if not site:
        site = "Both sites"
    else:
        site = f"CTA {site.capitalize()}"

    if not zenith:
        zenith = "all zeniths"
    else:
        zenith = f"z{zenith}"

    if subtitle:
        plt.title(
            f"GRB Detectability for {site}, {zenith}: {subtitle} (n={len(np.unique(data.index))})"
        )
    else:
        plt.title(f"GRB Detectability for {site}, {zenith}")

    fig = heatmap.get_figure()

    if not show_only:
        output_file = f"{output_dir}/GW_{site.replace(' ','_')}_{zenith.replace(' ','_')}.{filetype}"
        fig.savefig(output_file)
        # print(f"Saved plot {output_file}")


def run():

    # set matplotlib backend
    matplotlib.use("Agg")

    # load in the settings
    with open("./plot_settings.yaml") as file:
        print("Plot settings file found!")
        parsed_yaml_file = yaml.load(file, Loader=yaml.FullLoader)

    input_file = parsed_yaml_file["input_file"]
    output_dir = parsed_yaml_file["output_directory"]
    sites = parsed_yaml_file["sites"]
    zeniths = parsed_yaml_file["zeniths"]
    obs_times = parsed_yaml_file["observation_times"]
    annotate = parsed_yaml_file["show_percents"]
    log_scale = parsed_yaml_file["log_scale"]
    color_scheme = parsed_yaml_file["color_scheme"]
    x_tick_labels = parsed_yaml_file.get("x_tick_labels")
    y_tick_labels = parsed_yaml_file.get("y_tick_labels")
    min_value = parsed_yaml_file["min_value"]
    max_value = parsed_yaml_file["max_value"]

    print(f"Making output directory {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    if not x_tick_labels:
        x_tick_labels = "auto"
    if not y_tick_labels:
        y_tick_labels = "auto"

    data = open_gw_file(input_file)
    print(f"Successfully loaded input file: {input_file}")

    n_plots = len(sites) * len(zeniths)

    loading_bar = tqdm(total=n_plots)

    print("Min max", min_value, max_value)

    for site in sites:
        for zenith in zeniths:
            loading_bar.set_description(
                desc=f"Plotting zenith={zenith} and site={site}"
            )
            plot_toy(
                data,
                output_dir,
                site=site,
                zenith=zenith,
                obs_times=obs_times,
                min_value=min_value,
                max_value=max_value,
                x_tick_labels=x_tick_labels,
                y_tick_labels=y_tick_labels,
                color_scheme=color_scheme,
                annotate=annotate,
            )
            loading_bar.update(1)


if __name__ == "__main__":

    run()
