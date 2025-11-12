from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm


class GWData:
    """
    A class for reading and filtering gravitational wave data stored in Parquet or CSV format.

    Attributes:
        df (pd.DataFrame): The current (filtered) data frame.
        observation_times (np.ndarray): The observation times.
        results (pd.DataFrame): The results.

    Methods:
        set_filters(self, *args) -> None: Set filters on the current data frame.
        set_observation_times(self, obs_times: np.ndarray) -> None: Set the observation times.
        reset(self) -> None: Reset the current data frame to the full data set.
        plot(self, *args, **kwargs) -> None | plt.Axes: Plot the data.

    Example:
        >>> gw_data = GWData("data.parquet")
        >>> gw_data.set_filters(("z", "<", 4))
        >>> gw_data.plot()
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
            self._data = pd.read_parquet(self._input_file)
        elif self._file_type == ".csv":
            self._data = pd.read_csv(self._input_file)
        else:
            raise ValueError("File type not supported, please use .parquet or .csv")

        # Set the initial data to the full data set.
        self._current_data = self._data

        self._obs_times = self._default_obs_times

        self._results = pd.DataFrame(
            columns=["delay", "obs_time", "n_seen", "total", "percent_seen"]
        )

    @property
    def df(self) -> pd.DataFrame:
        """
        Property to access the current (filtered) data frame.

        Returns:
            data (pd.DataFrame): The current (filtered) data frame.
        """
        return self._current_data

    @property
    def observation_times(self) -> np.ndarray:
        """
        Property to access the observation times.

        Returns:
            obs_times (np.ndarray): The observation times.
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
    def _default_delays(self) -> list[int]:
        return [round(i) for i in np.logspace(1, np.log10(7 * 24 * 3600), 50)]

    @property
    def _default_obs_times(self) -> np.ndarray:
        return np.logspace(1, np.log10(1 * 3600 + 0.1), 50, dtype=int)

    def _calculate_results(self) -> None:
        """
        Helper method to calculate the results percentages of detected sources and save into a DataFrame. The results are stored dirctly in the _results attribute.
        """
        data = self._current_data

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
        """
        Set filters on the current data frame.

        Args:
            *args: The filters to set.
                Each filter must be a tuple of the form (column, operator, value).
                The operator must be one of ==, =, <, >, <=, >=, in, not in, notin.
                The value must be a single value or a list of values.

        Returns:
            None: The current data frame is updated in place.

        Example:
            Filter on the redshift column:
            >>> gw_data.set_filters(("z", "<", 4))
        """

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
                self._current_data = self._current_data[
                    self._current_data[column].isin(value)
                ]
            elif op == "not in" or op == "notin":
                self._current_data = self._current_data[
                    ~self._current_data[column].isin(value)
                ]
            elif op == "==" or op == "=":
                # Filter using == for non-list values
                self._current_data = self._current_data[
                    self._current_data[column] == value
                ]
            elif op == "<":
                self._current_data = self._current_data[
                    self._current_data[column] < value
                ]
            elif op == ">":
                self._current_data = self._current_data[
                    self._current_data[column] > value
                ]
            elif op == "<=":
                self._current_data = self._current_data[
                    self._current_data[column] <= value
                ]
            elif op == ">=":
                self._current_data = self._current_data[
                    self._current_data[column] >= value
                ]

    def set_observation_times(self, obs_times: np.ndarray) -> None:
        """
        Set the observation times.

        Args:
            obs_times (np.ndarray): The observation times.
        """
        self._obs_times = np.array(obs_times)

    def reset(self) -> None:
        """
        Reset the current data frame to the full data set.
        """
        self._current_data = self._data
        self._results = pd.DataFrame(
            columns=["delay", "obs_time", "n_seen", "total", "percent_seen"]
        )

    @staticmethod
    def _convert_time(seconds: float) -> str:
        """
        Helper method to convert a time in seconds to a string representation of the time.

        Args:
            seconds (float): The time in seconds.

        Returns:
            str: A string representation of the time.
        """
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.0f}m"
        elif seconds < 86400:
            return f"{seconds / 3600:.0f}h"
        else:
            return f"{seconds / 86400:.0f}d"

    def plot(
        self,
        ax: plt.Axes | None = None,
        output_file: str | None = None,
        annotate: bool = False,
        x_tick_labels: np.ndarray | list[str] | None = None,
        y_tick_labels: np.ndarray | list[str] | None = None,
        min_value: float | None = None,
        max_value: float | None = None,
        color_scheme: str = "mako",
        color_scale: str | None = None,
        as_percent: bool = False,
        title: str | None = None,
        subtitle: str | None = None,
        n_labels: int = 10,
        square: bool = True,
        return_ax: bool = True,
    ) -> None | plt.Axes:
        """
        Generates a heatmap of the data stored in the instance of the class.

        Args:
            ax (plt.Axes): The axis object to plot on.
            output_file (str | None): The path to the output file.
            annotate (bool): Whether or not to annotate the heatmap.
            x_tick_labels (list[str] | None): The labels for the x-axis ticks.
            y_tick_labels (list[str] | None): The labels for the y-axis ticks.
            min_value (float | None): The minimum value for the color scale.
            max_value (float | None): The maximum value for the color scale.
            color_scheme (str): The name of the color scheme to use for the heatmap.
            color_scale (str | None): The type of color scale to use for the heatmap.
            as_percent (bool): Whether or not to display the results as percentages.
            title (str | None): The title for the plot.
            subtitle (str | None): The subtitle for the plot.
            n_labels (int): The number of labels to display on the axes.
            square (bool): If True, set the Axes aspect to "equal" so each cell will be square-shaped.
            return_ax (bool): Whether or not to return the axis object.

        Returns:
            Either `matplotlib.axes._axes.Axes` or None if `return_ax` is False.
        """
        # Get the results dataframe from the class.
        df = self.results
        df.rename(columns={"obs_time": "exposure time"}, inplace=True)

        # Set the plot style using seaborn.
        sns.set_theme()

        # Convert the results to percentages, if requested.
        if as_percent:
            df["percent_seen"] = df["percent_seen"] * 100

        # Pivot the data so that it can be plotted as a heatmap.
        pivot = df.pivot("exposure time", "delay", "percent_seen").astype(float)

        # Create a new figure and axis.
        if ax is None:
            _, ax = plt.subplots(figsize=(9, 9))

        # Set the colorbar options.
        cbar_kws = {"label": "Percentage of sources detected", "orientation": "vertical"}

        # Set the color scale if logarithmic scale is selected.
        if color_scale == "log":
            norm = LogNorm(vmin=min_value, vmax=max_value)
        else:
            norm = None

        # Set the x-axis tick labels.
        if not x_tick_labels:
            x_delays = np.sort(self._results.delay.unique())
            label_delays = x_delays[:: int(len(x_delays) / n_labels)]
            x_tick_pos = np.arange(len(x_delays))[:: int(len(x_delays) / n_labels)]
            if x_delays[-1] != label_delays[-1]:
                label_delays = np.append(label_delays, x_delays[-1])
                x_tick_pos = np.append(x_tick_pos, len(x_delays) - 1)
            x_tick_labels = [self._convert_time(x) for x in label_delays]

        # Set the y-axis tick labels.
        if not y_tick_labels:
            label_obs_times = self.observation_times[
                :: int(len(self.observation_times) / n_labels)
            ]
            y_tick_pos = np.arange(len(self.observation_times))[
                :: int(len(self.observation_times) / n_labels)
            ]
            if self.observation_times[-1] != label_obs_times[-1]:
                label_obs_times = np.append(label_obs_times, self.observation_times[-1])
                y_tick_pos = np.append(y_tick_pos, len(self.observation_times) - 1)
            y_tick_labels = [self._convert_time(x) for x in label_obs_times]

        # Create the heatmap, with or without annotations.
        heatmap = sns.heatmap(
            pivot,
            annot=True if annotate else None,
            fmt=".0f" if annotate else ".2g",
            linewidths=0.5 if annotate else 0,
            ax=ax,
            cmap=color_scheme,
            vmin=min_value,
            vmax=max_value,
            xticklabels=x_tick_labels,
            yticklabels=y_tick_labels,
            cbar_kws=cbar_kws,
            norm=norm,
            square=square,
        )

        # Invert the y-axis so that the plot is oriented correctly.
        heatmap.invert_yaxis()

        # Set the title and axis labels.
        sites = self.df["site"].unique()
        if len(sites) > 1:
            site = "CTAO N + S"
        else:
            site = f"CTAO {sites[0].capitalize()}"

        zeniths = self.df["zeniths"].unique()
        zenith = f"z{zeniths[0]}"
        if len(zeniths) > 1:
            for z in zeniths[1:]:
                zenith += f"/z{z}"

        if subtitle:
            ax.set_title(
                f"Source Detectability for {site}, {zenith}: {subtitle} (n={self._results.groupby('delay').total.first().iloc[0]})"
            )
        elif title:
            ax.set_title(title)
        else:
            ax.set_title(
                f"Source Detectability for {site}, {zenith} (n={self._results.groupby('delay').total.first().iloc[0]})"
            )

        # Set the tick positions and labels for the x and y axes.
        ax.set_xlabel("$t_{0}$", fontsize=16)
        ax.set_ylabel(r"$t_{\mathrm{exp}}$", fontsize=16)
        ax.set_xticks(x_tick_pos, x_tick_labels, rotation=45, fontsize=12)
        ax.set_yticks(y_tick_pos, y_tick_labels, fontsize=12)

        # Set the tick parameters for both axes.
        ax.tick_params(
            axis="both",
            length=5,
            color="black",
            direction="out",
            bottom=True,
            left=True,
        )

        # Get the figure and save it, or show it if requested.
        fig = heatmap.get_figure()

        print("ax, output_file, filetype", ax, output_file)
        if output_file:
            print("saving")
            fig.savefig(output_file, bbox_inches="tight", pad_inches=0)

        if not return_ax:
            plt.show()
            return None

        return ax
