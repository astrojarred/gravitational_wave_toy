from pathlib import Path

import seaborn as sns
import numpy as np
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt

class GWData:
    """
    A class for reading and filtering gravitational wave data stored in Parquet or CSV format.
    """

    def __init__(self, input_file: str):
        """
        Constructor for the GWData class.

        Args:
            input_file (str): The path to the input file.
        """

        # Store the absolute path to the input file.
        self._input_file = Path(input_file).absolute()

        # Determine the file type from the file extension.
        self._file_type = self._input_file.suffix

        # Load the data from the file, using Dask for parallel processing.
        if self._file_type == ".parquet":
            self._data = dd.read_parquet(self._input_file)
        elif self._file_type == ".csv":
            self._data = dd.read_csv(self._input_file, n_partitions=24)
        else:
            raise ValueError("File type not supported, please use .parquet or .csv")

        # Set the initial data to the full data set.
        self._current_data = self._data

        self._obs_times = self._default_obs_times

        self._results = pd.DataFrame(
            columns=["delay", "obs_time", "n_seen", "total", "percent_seen"]
        )

    @property
    def df(self) -> dd.DataFrame:
        """
        Property to access the current (filtered) data frame.

        Returns:
            data (dd.DataFrame): The current (filtered) data frame.
        """
        return self._current_data

    @property
    def observation_times(self) -> list:
        """
        Property to access the observation times.

        Returns:
            obs_times (list): The observation times.
        """
        return self._obs_times

    @property
    def results(self) -> pd.DataFrame:
        """
        Property to access the results.

        Returns:
            results (pd.DataFrame): The results.
        """
        if len(self._results) == 0:
            self._calculate_results()
        return self._results

    def __len__(self) -> int:
        """
        Get the length of the current data frame.

        Returns:
            length (int): The length of the current data frame.
        """
        return len(self._current_data)

    def __repr__(self) -> str:
        """
        Get a string representation of the GWData object.

        Returns:
            A string representation of the GWData object.
        """
        return f"GWData({self._input_file})"
        
    @property
    def _default_delays(self):
        return [round(i) for i in np.logspace(1, np.log10(7 * 24 * 3600), 50)]
    
    @property
    def _default_obs_times(self):
        return np.logspace(1, np.log10(1 * 3600 + 0.1), 50, dtype=int)

    def _calculate_results(self):
        data = self._current_data.compute()

        # Filter out rows with "obs_time" <= 0
        # Group the data by "start_time" and "obs_time"
        groups = data[data["obs_time"] > 0].groupby(["delay", "obs_time"])

        # Calculate the number of rows that satisfy the condition for each group
        seen = groups["obs_time"].count()

        # Calculate the total number of rows for each "start_time" group
        total = data.groupby("delay")["obs_time"].count()

        # Create a DataFrame with unique pairs of "delay" and "obs_time"
        pairs = (
            pd.MultiIndex.from_product(
                [data["delay"].unique(), self._obs_times], names=["delay", "obs_time"]
            )
            .to_frame()
            .reset_index(drop=True)
        )

        # Calculate the "n_seen" and "total" values for each pair
        pairs[["n_seen", "total"]] = pairs.apply(
            lambda row: (
                seen.where(
                    (seen.index.get_level_values("delay") == row.delay)
                    & (seen.index.get_level_values("obs_time") <= row.obs_time)
                )
                .dropna()
                .sum(),
                total.get(row.delay, 0),
            ),
            result_type="expand",
            axis=1,
        )

        # set n_seen and total to integer values
        pairs["n_seen"] = pairs["n_seen"].astype(int)
        pairs["total"] = pairs["total"].astype(int)
        pairs["percent_seen"] = pairs["n_seen"] / pairs["total"]

        self._results = pairs

    def set_filters(self, *args) -> None:
        self._current_data = self._data
        self._results = pd.DataFrame(
            columns=["delay", "obs_time", "n_seen", "total", "percent_seen"]
        )

        for a in args:
            if not isinstance(a, tuple):
                raise TypeError("Filters must be passed as tuples")
            column, op, value = a
            if op not in ["==", "=", "<", ">", "<=", ">=", "in", "not in", "notin"]:
                raise ValueError(
                    "Filter operation must be one of ==, =, <, >, <=, >=, in, not in"
                )
            if op == "in":
                # Filter using isin for list values
                self._current_data = self._current_data[self._current_data[column].isin(value)]
            elif op == "not in" or op == "notin":
                self._current_data = self._current_data[~self._current_data[column].isin(value)]
            elif op == "==" or op == "=":
                # Filter using == for non-list values
                self._current_data = self._current_data[self._current_data[column] == value]
            elif op == "<":
                self._current_data = self._current_data[self._current_data[column] < value]
            elif op == ">":
                self._current_data = self._current_data[self._current_data[column] > value]
            elif op == "<=":
                self._current_data = self._current_data[self._current_data[column] <= value]
            elif op == ">=":
                self._current_data = self._current_data[self._current_data[column] >= value]


    def set_observation_times(self, obs_times: list) -> None:
        """
        Set the observation times.

        Args:
            obs_times (list): The observation times.
        """
        self._obs_times = obs_times

    def reset(self) -> None:
        """
        Reset the current data frame to the full data set.
        """
        self._current_data = self._data
        self._results = pd.DataFrame(
            columns=["delay", "obs_time", "n_seen", "total", "percent_seen"]
        )
        
    @staticmethod
    def _convert_time(seconds: float):
        if seconds < 60:
            return f"{int(seconds):.0f}s"
        if seconds < 3600:
            return f"{int(seconds) / 60:.0f}m"
        if seconds < 86400:
            return f"{int(seconds) / 3600:.0f}h"
        else:
            return f"{int(seconds) / 86400:.0f}d"

    def plot(
        self,
        output_file=None,
        annotate=False,
        x_tick_labels=None,
        y_tick_labels=None,
        min_value=None,
        max_value=None,
        color_scheme="mako",
        color_scale=None,
        as_percent=False,
        filetype="png",
        title=None,
        subtitle=None,
        n_labels=10,
        filename_suffix="",
        show_only=False,
    ) -> None:
        
        sns.set()

        df = self.results
        df.rename(columns={"obs_time": "exposure time"}, inplace=True)

        sns.set_theme()

        if as_percent:
            df["percent_seen"] = df["percent_seen"] * 100

        pivot = df.pivot("exposure time", "delay", "percent_seen").astype(float)

        f, ax = plt.subplots(figsize=(9, 9))

        cbar_kws = {"label": "Percentage of GRBs detected", "orientation": "vertical"}

        if color_scale == "log":
            from matplotlib.colors import LogNorm

            color_scale = LogNorm(vmin=min_value, vmax=max_value)

        if not x_tick_labels:
            x_delays = np.sort(self._results.delay.unique())
            label_delays = x_delays[::int(len(x_delays) / n_labels)]
            x_tick_pos = np.arange(len(x_delays))[::int(len(x_delays) / n_labels)]
            if x_delays[-1] != label_delays[-1]:
                label_delays = np.append(label_delays, x_delays[-1])
                x_tick_pos = np.append(x_tick_pos, len(x_delays) - 1)
            # x_tick_labels = [self._convert_time(x) if x in label_delays else "" for x in x_delays]
            x_tick_labels = [self._convert_time(x) for x in label_delays]
        if not y_tick_labels:
            label_obs_times = self.observation_times[::int(len(self.observation_times) / n_labels)]
            y_tick_pos = np.arange(len(self.observation_times))[::int(len(self.observation_times) / n_labels)]
            if self.observation_times[-1] != label_obs_times[-1]:
                label_obs_times = np.append(label_obs_times, self.observation_times[-1])
                y_tick_pos = np.append(y_tick_pos, len(self.observation_times) - 1)
            # y_tick_labels = [self._convert_time(x) if x in label_obs_times else "" for x in self.observation_times]
            y_tick_labels = [self._convert_time(x) for x in label_obs_times]

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
                # norm=color_scale,
                square=True,
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
                # norm=color_scale,
                square=True,
            )

        heatmap.invert_yaxis()
        # heatmap.set_facecolor("#1C1C1C")

        sites = self.df["site"].unique().compute()
        if len(sites) > 1:
            site = "CTA N + S"
        else:
            site = f"CTA {sites[0].capitalize()}"

        zeniths = self.df["zeniths"].unique().compute()
        zenith = f"z{zeniths[0]}"
        if len(zeniths) > 1:
            for z in zeniths[1:]:
                zenith += f"/z{z}"

        if title:
            plt.title(title)
        if subtitle:
            plt.title(
                f"GRB Detectability for {site}, {zenith}: {subtitle} (n={self._results.groupby('delay').total.first().iloc[0]})"
            )
        else:
            plt.title(
                f"GRB Detectability for {site}, {zenith} (n={self._results.groupby('delay').total.first().iloc[0]})"
            )
            
        plt.xlabel("$t_{0}$", fontsize=16)
        plt.ylabel("$t_{\mathrm{exp}}$", fontsize=16)
        # Set tick positions and labels
        ax.set_xticks(x_tick_pos, x_tick_labels, rotation=45, fontsize=12)
        #ax.set_xticklabels(x_tick_labels, rotation=45, fontsize=12)
        ax.set_yticks(y_tick_pos, y_tick_labels, fontsize=12)
        #ax.set_yticklabels(y_tick_labels, fontsize=12)
        plt.tick_params(axis="both", length=5, color="black", direction="out", bottom=True, left=True)


        fig = heatmap.get_figure()

        if not show_only:
            # output_file = f"{output_dir}/GW_{site.replace(' ','_')}_{zenith.replace(' ','_')}{filename_suffix}.{filetype}"
            fig.savefig(output_file + f".{filetype}")
            # print(f"Saved plot {output_file}")
        else:
            plt.show()

        return ax
