"""Source class for loading and analyzing time-energy spectra of astrophysical events.

This module provides functionality for reading spectral data from various file formats
(FITS, CSV, text), interpolating spectra, fitting spectral indices, and determining
source visibility and significance for gamma-ray observations.
"""

import math
import os
import re
import warnings
from pathlib import Path
from typing import Literal

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import Distance
from astropy.io import fits
from gammapy.modeling.models import (
    EBLAbsorptionNormSpectralModel,
    PowerLawSpectralModel,
)
from gammapy.modeling.models.spectral import EBL_DATA_BUILTIN
from gammapy.utils.roots import find_roots
from scipy import integrate
from scipy.interpolate import RegularGridInterpolator, interp1d

from .logging import logger
from .sensitivity import ScaledTemplateModel, Sensitivity

log = logger(__name__)


class Source:
    """Class for loading and analyzing time-energy spectra of astrophysical events.

    The Source class handles reading spectral data from various file formats (FITS, CSV, text),
    interpolating spectra in time and energy, fitting power-law spectral indices, and determining
    source visibility and detection significance for gamma-ray observations. It supports EBL
    absorption modeling and integration with sensitivity curves for observability calculations.

    Args:
        filepath: Path to the source file or directory. Supports:
            - FITS files (.fits, .fit, .fits.gz, .fit.gz)
            - CSV files (.csv) with optional metadata files
            - Directories containing text files with spectral data
        min_energy: Minimum energy for spectral integration, typically bounded by the IRF.
            If None, uses the minimum energy from the data.
        max_energy: Maximum energy for spectral integration, typically bounded by the IRF.
            If None, uses the maximum energy from the data.
        ebl: Name of the EBL absorption model to apply (e.g., "franceschini", "dominguez").
            If None and a distance/redshift is available, no EBL model is applied.

    Attributes:
        time: Array of time values (astropy Quantity, units of seconds).
        energy: Array of energy values (astropy Quantity, units of GeV).
        spectra: 2D array of differential flux values (shape: [n_energy, n_time],
            units of cm⁻² s⁻¹ GeV⁻¹).
        id: Source identifier.
        long: Longitude (astropy Quantity, units of radians).
        lat: Latitude (astropy Quantity, units of radians).
        eiso: Isotropic equivalent energy (astropy Quantity, units of erg).
        dist: Distance to the source (astropy Distance object).
        angle: Viewing angle (astropy Quantity, units of degrees).
        fluence: Fluence value (astropy Quantity, units of cm⁻²).
        seen: Visibility status (True if detectable, False if not, "error" if calculation failed).
        obs_time: Required observation time to detect the source (astropy Quantity).
        end_time: Time at which detection occurs (astropy Quantity).
    """

    def __init__(
        self,
        filepath: str | Path,
        min_energy: u.Quantity | None = None,
        max_energy: u.Quantity | None = None,
        ebl: str | None = None,
        times: list[u.Quantity] | None = None,
    ) -> None:
        if isinstance(min_energy, u.Quantity):
            min_energy = min_energy.to("GeV")
        if isinstance(max_energy, u.Quantity):
            max_energy = max_energy.to("GeV")

        self.filepath = Path(filepath).absolute()
        self.min_energy, self.max_energy = min_energy, max_energy
        self.seen: bool | Literal["error"] = False
        self.obs_time = -1 * u.s
        self.start_time = -1 * u.s
        self.end_time = -1 * u.s
        self.error_message: str | None = None
        self.dist = None
        self.file_type: Literal["fits", "txt", "csv", None] = None
        self.id = 0

        self.input_times: list[u.Quantity] | None = None
        if times is not None:
            self.input_times = times

        # initialize metadata
        self.eiso = None
        self.fluence = None
        self.dist = None
        self.angle = None
        self.long = None
        self.lat = None
        self.fluence = None

        # choose reader based on file extension or directory contents
        if self.filepath.is_dir():
            # For directories, check for txt files first, then fits files
            txt_files = list(self.filepath.glob("*.txt"))
            fits_files = list(self.filepath.glob("*.fits")) + list(
                self.filepath.glob("*.fit")
            )

            if txt_files:
                self.file_type = "txt"
                self.read_txt()
            elif fits_files:
                self.file_type = "fits"
                self.read_fits()
            else:
                raise ValueError(
                    f"No supported files (.txt or .fits) found in directory {self.filepath}"
                )
        else:
            # For single files, use original logic
            name_lower = self.filepath.name.lower()
            if name_lower.endswith((".fits", ".fit", ".fits.gz", ".fit.gz")):
                self.file_type = "fits"
                self.read_fits()
            elif name_lower.endswith(".csv"):
                self.file_type = "csv"
                self.read_csv()
            elif name_lower.endswith(".txt"):
                self.file_type = "txt"
                self.read_txt()
            else:
                raise ValueError(f"Unsupported file format for {self.filepath}")

        if self.input_times is not None:
            # check for time-compatible units:
            if not all(isinstance(t, u.Quantity) and t.unit.physical_type == "time" for t in self.input_times):
                raise ValueError(f"Input times must have time units.")
            self.time = u.Quantity(self.input_times).to("s")

        # set spectral grid
        self.SpectralGrid = None
        self.set_spectral_grid()

        # fit spectral indices
        self.fit_spectral_indices()

        # set EBL model (and optionally update distance via redshift)
        if self.dist is not None and not self.dist == 0:
            self.set_ebl_model(ebl)
        else:
            self.ebl = None
            self.ebl_model = None

        log.debug(f"Loaded event {self.id}º")

    def __repr__(self):
        """Return a string representation of the Source instance."""
        return f"<Source(id={self.id})>"

    @property
    def metadata(self) -> dict:
        """Return a dictionary of the source metadata.

        Returns:
            dict: A dictionary of the source metadata which includes the event id, longitude, latitude, eiso, distance, and jet opening angle.
        """
        return {
            "id": self.id,
            "longitude": self.long,
            "latitude": self.lat,
            "eiso": self.eiso,
            "distance": self.dist,
            "angle": self.angle,
            "fluence": self.fluence,
        }

    def read_fits(self) -> None:
        """Read spectral data and metadata from a FITS file.

        Extracts time-energy spectra from the FITS file structure, where HDU[1] contains
        energy data, HDU[2] contains time data, and HDU[3] contains the light curve
        spectra. Metadata fields (longitude, latitude, EISO, distance, angle, fluence)
        are read from the primary header if present, with missing fields logged as info.

        Sets the following attributes:
            - self.time: 1D array of time values (astropy Quantity, units of seconds)
            - self.energy: 1D array of energy values (astropy Quantity, units of GeV)
            - self.spectra: 2D array of flux values (shape: [n_energy, n_time],
              units of cm⁻² s⁻¹ GeV⁻¹)
            - Metadata attributes (long, lat, eiso, dist, angle, fluence) if present
        """
        with fits.open(self.filepath) as hdu_list:
            # Try/catch each header field; set to default if missing
            try:
                self.long = hdu_list[0].header["LONG"] * u.Unit("rad")
            except KeyError:
                log.info("Longitude (LONG) not found in FITS header")
            try:
                self.lat = hdu_list[0].header["LAT"] * u.Unit("rad")
            except KeyError:
                log.info("Latitude (LAT) not found in FITS header")
            try:
                self.eiso = hdu_list[0].header["EISO"] * u.Unit("erg")
            except KeyError:
                log.info(
                    "Isotropic equivalent energy of prompt emission (EISO) not found in FITS header"
                )
            try:
                self.dist = Distance(hdu_list[0].header["DISTANCE"], unit="kpc")
            except KeyError:
                log.info("Distance (DISTANCE) not found in FITS header")
            try:
                self.angle = hdu_list[0].header["ANGLE"] * u.Unit("deg")
            except KeyError:
                log.info("Viewing angle (ANGLE) not found in FITS header")
            try:
                self.fluence = hdu_list[0].header["FLUENCE"] * u.Unit("1 / cm2")
            except KeyError:
                log.info("Fluence (FLUENCE) not found in FITS header")

            datalc = hdu_list[3].data
            datatime = hdu_list[2].data
            dataenergy = hdu_list[1].data

            self.time = datatime.field(0) * u.Unit("s")
            self.energy = dataenergy.field(0) * u.Unit("GeV")

            self.spectra = np.array(
                [datalc.field(i) for i, e in enumerate(self.energy)]
            ) * u.Unit("1 / (cm2 s GeV)")

    def read_txt(self) -> None:
        """Read spectral data from a directory of text files.

        Expects a directory containing spectral files named with the pattern
        '{basename}_tobs=NN.txt', where basename matches the directory name and NN
        is a time index. Each file should contain two columns: energy (GeV) and
        differential flux (cm⁻² s⁻¹ GeV⁻¹).

        Files are sorted by their time index and combined into a time-energy grid.
        The energy grid is taken from the first file, assuming all files share the
        same energy bins.

        Sets the following attributes:
            - self.time: 1D array of time values (astropy Quantity, units of seconds)
            - self.energy: 1D array of energy values (astropy Quantity, units of GeV)
            - self.spectra: 2D array of flux values (shape: [n_energy, n_time],
              units of cm⁻² s⁻¹ GeV⁻¹)
            - self.min_energy, self.max_energy: if not already set

        Raises:
            FileNotFoundError: If no matching spectral files are found in the directory.
        """
        # expect a directory containing source spectral files like source001_tobs=00.txt, source001_tobs=01.txt, etc.
        dir_path = self.filepath

        # Extract base name from directory name (e.g., "source001" from "/path/to/source001/")
        base = dir_path.name

        # find spectral files in directory
        candidates = list(dir_path.glob(f"{base}_tobs=*.txt"))
        if len(candidates) == 0:
            raise FileNotFoundError(
                f"No spectral files matching {base}_tobs=*.txt found in {dir_path}"
            )

        def extract_index(p: Path) -> int:
            m = re.search(r"_tobs=(\d+)(?:_|\.|$)", p.name)
            return int(m.group(1)) if m else -1

        candidates.sort(key=extract_index)

        spectra_columns = []  # list of flux arrays per time

        for p in candidates:
            arr = np.loadtxt(p)
            energy = arr[:, 0] * u.GeV
            dNdE = arr[:, 1] * u.Unit("1 / (cm2 s GeV)")

            spectra_columns.append(dNdE)

        # set energy grid
        self.energy = energy

        # build spectra with shape (n_energy, n_time)
        spectra_stack = u.Quantity(spectra_columns)  # (n_time, n_energy)
        self.spectra = spectra_stack.T  # (n_energy, n_time)

        if not isinstance(self.min_energy, u.Quantity):
            self.min_energy = self.energy.min()
        if not isinstance(self.max_energy, u.Quantity):
            self.max_energy = self.energy.max()

    def read_csv(self) -> None:
        """
        Read source time-energy spectra from a CSV file and optional metadata file.

        Expected CSV format:
            - Columns: 'time', 'energy', 'flux' (case-insensitive, substring matching allowed).
                - 'time': Time in seconds (s).
                - 'energy': Energy in giga-electronvolts (GeV).
                - 'flux': Differential flux in units of cm⁻² s⁻¹ GeV⁻¹.
            - Each row represents a single (time, energy, flux) measurement.
            - The file may use column names with or without units, e.g., "time [s]", "energy [GeV]", "flux [cm-2 s-1 GeV-1]".

        Data structure requirements:
            - The method builds a time-energy grid from the CSV, with arrays:
                - self.time: 1D array of time values (astropy Quantity, units of seconds).
                - self.energy: 1D array of energy values (astropy Quantity, units of GeV).
                - self.spectra: 2D array of flux values (shape: [n_energy, n_time], units of cm⁻² s⁻¹ GeV⁻¹).
            - The CSV may contain repeated time or energy values; the method will organize the data into a grid.

        Optional metadata file:
            - If a file named '{base}_metadata.csv' exists in the same directory, it will be read for additional parameters.
            - Expected format: CSV file with columns 'parameter', 'value', and 'units', e.g.:
                parameter,value,units
                id,42.0,
                longitude,0.0,rad
                latitude,1.0,rad
                eiso,2e+50,erg
                distance,100000.0,kpc
                angle,5.0,deg
                fluence,2.3e-5,1 / cm2
            - Suggested units for metadata fields:
                - id: int
                - longitude: rad
                - latitude: rad
                - eiso: erg
                - distance: kpc
                - angle: deg
                - fluence: 1 / cm2
            - If any metadata field is missing, default (dummy) values are kept as None.

        Attributes set by this method:
            - self.time, self.energy, self.spectra
            - self.id, self.long, self.lat, self.eiso, self.dist, self.angle, self.fluence
            - self.min_energy, self.max_energy (if not already set)

        Raises:
            - FileNotFoundError: If the CSV file does not exist.
            - ValueError: If required columns are missing or data cannot be parsed.
        """

        csv_path = self.filepath

        # Read CSV data
        df = pd.read_csv(csv_path)

        # Extract columns (handle both with and without brackets in column names)
        # Uses substring matching for flexibility (e.g., "time [s]", "timestamp", "energy [GeV]", etc.)
        time_col = None
        energy_col = None
        flux_col = None

        # Improved column matching logic: Try exact match first, then fall back to substring match (case-insensitive).
        # This avoids accidental matches like "time_energy" for "time".
        lower_cols = {col.lower(): col for col in df.columns}
        time_col = energy_col = flux_col = None

        # Try to match exactly (ignoring case and possible brackets/units)
        def find_exact_or_substring(names, options):
            # Try exact match first
            for opt in options:
                for name in names:
                    if name.lower().strip() == opt.lower():
                        return names[name]
            # Then substring match
            for opt in options:
                for name in names:
                    if opt.lower() in name.lower():
                        return names[name]
            return None

        # Common variants (possibly with units)
        time_col = find_exact_or_substring(lower_cols, ["time", "time [s]"])
        energy_col = find_exact_or_substring(lower_cols, ["energy", "energy [gev]"])
        flux_col = find_exact_or_substring(
            lower_cols,
            ["flux", "flux [cm-2 s-1 gev-1]", "dNdE", "dNdE [cm-2 s-1 gev-1]"],
        )

        # Documented: If multiple columns match by substring, the first match in column order is used.

        if time_col is None or energy_col is None or flux_col is None:
            missing = []
            if time_col is None:
                missing.append("time")
            if energy_col is None:
                missing.append("energy")
            if flux_col is None:
                missing.append("flux")
            raise ValueError(
                f"CSV file must contain columns for time, energy, and flux. "
                f"Missing columns: {', '.join(missing)}. "
                f"Found columns: {list(df.columns)}. "
                f"Expected column names should contain 'time', 'energy', and 'flux' (case-insensitive)."
            )

        time_values = df[time_col].values * u.s
        energy_values = df[energy_col].values * u.GeV

        # Get unique sorted values
        unique_times = np.unique(time_values.value)
        unique_energies = np.unique(energy_values.value)

        # Reshape flux array to (n_energy, n_time)
        # Data is structured as: for each time, all energies are listed
        n_time = len(unique_times)
        n_energy = len(unique_energies)

        # Verify data structure
        if len(df) != n_time * n_energy:
            raise ValueError(
                f"Data length ({len(df)}) does not match expected "
                f"n_time * n_energy ({n_time} * {n_energy} = {n_time * n_energy})"
            )

        # Sort data by time first, then by energy to ensure correct ordering
        df_sorted = df.sort_values(by=[time_col, energy_col])
        flux_sorted = df_sorted[flux_col].values * u.Unit("1 / (cm2 s GeV)")

        # Reshape flux values: (n_time, n_energy) then transpose to (n_energy, n_time)
        spectra_reshaped = flux_sorted.value.reshape(n_time, n_energy).T
        self.spectra = spectra_reshaped * u.Unit("1 / (cm2 s GeV)")

        self.time = unique_times * u.s
        self.energy = unique_energies * u.GeV

        # Read metadata file if it exists
        metadata_path = csv_path.parent / f"{csv_path.stem}_metadata.csv"

        if metadata_path.exists():
            try:
                metadata_df = pd.read_csv(metadata_path)

                metadata_dict = {}
                if (
                    "parameter" in metadata_df.columns
                    and "value" in metadata_df.columns
                ):
                    for row in metadata_df.itertuples(index=False):
                        # Convert to string and strip, handling NaN/float cases
                        param = (
                            str(row.parameter).strip()
                            if pd.notna(row.parameter)
                            else ""
                        )
                        value = row.value

                        # Handle Units column - convert to string, handle NaN/empty
                        if "units" in metadata_df.columns:
                            unit_val = row.units

                        if pd.notna(unit_val):
                            unit_str = str(unit_val).strip()
                        else:
                            unit_str = ""

                        if pd.notna(value) and param:
                            metadata_dict[param] = {"value": value, "unit": unit_str}

                # Parse metadata with defaults using a mapping dictionary
                # Format: 'metadata_key': ('attribute_name', 'default_unit', 'parser_func')
                metadata_mapping = {
                    "id": ("id", None, lambda v, unit_str: int(float(v))),
                    "longitude": (
                        "long",
                        "rad",
                        lambda v, unit_str: float(v) * u.Unit(unit_str or "rad"),
                    ),
                    "latitude": (
                        "lat",
                        "rad",
                        lambda v, unit_str: float(v) * u.Unit(unit_str or "rad"),
                    ),
                    "eiso": (
                        "eiso",
                        "erg",
                        lambda v, unit_str: float(v) * u.Unit(unit_str or "erg"),
                    ),
                    "distance": (
                        "dist",
                        "kpc",
                        lambda v, unit_str: Distance(float(v), unit=unit_str or "kpc"),
                    ),
                    "angle": (
                        "angle",
                        "deg",
                        lambda v, unit_str: float(v) * u.Unit(unit_str or "deg"),
                    ),
                    "fluence": (
                        "fluence",
                        "1 / cm2",
                        lambda v, unit_str: float(v) * u.Unit(unit_str or "1 / cm2"),
                    ),
                }

                for metadata_key, (
                    attr_name,
                    default_unit,
                    parser_func,
                ) in metadata_mapping.items():
                    if metadata_key in metadata_dict:
                        try:
                            value = metadata_dict[metadata_key]["value"]
                            unit = metadata_dict[metadata_key]["unit"] or default_unit
                            setattr(self, attr_name, parser_func(value, unit))
                            log.debug(f"Set {attr_name} to {getattr(self, attr_name)}")
                        except (
                            ValueError,
                            TypeError,
                            u.UnitConversionError,
                        ) as field_exc:
                            # Convert bytes to strings for safe formatting
                            value_str = (
                                value.decode("utf-8")
                                if isinstance(value, bytes)
                                else str(value)
                            )
                            unit_str = (
                                unit.decode("utf-8")
                                if isinstance(unit, bytes)
                                else str(unit)
                            )
                            log.warning(
                                f"Could not parse metadata field '{metadata_key}' (value={value_str}, unit={unit_str}) in {metadata_path}: {type(field_exc).__name__} {field_exc}. Using default value for '{attr_name}'."
                            )
            except Exception as e:
                log.warning(
                    f"Could not parse metadata file {metadata_path}: {type(e).__name__} {e}. Using defaults."
                )
        else:
            log.warning(f"No metadata file found at {metadata_path}")

        # Set energy limits if not already set
        if not isinstance(self.min_energy, u.Quantity):
            self.min_energy = self.energy.min()
        if not isinstance(self.max_energy, u.Quantity):
            self.max_energy = self.energy.max()

    def set_ebl_model(self, ebl: str | None, z: float | None = None) -> bool:
        """Set or update the EBL absorption model and optionally the source redshift.

        Configures the extragalactic background light (EBL) absorption model for the source.
        If a redshift is provided, the source distance is updated accordingly. The EBL model
        is used to account for gamma-ray absorption due to pair production with EBL photons.

        Args:
            ebl: Name of the EBL model to use. Must be one of the built-in models available
                in Gammapy (e.g., "franceschini", "dominguez", etc.). If None, no EBL model
                is set. Requires GAMMAPY_DATA environment variable to be set.
            z: Optional redshift value. If provided and different from the current redshift,
                the source distance is updated using the default cosmology.

        Returns:
            True if the distance (redshift) was changed, False otherwise.

        Raises:
            ValueError: If the EBL model name is not recognized, or if GAMMAPY_DATA
                environment variable is not set when attempting to use an EBL model.
        """
        distance_changed = False

        # Determine current redshift if available
        current_z_val = None
        try:
            if self.dist is not None and self.dist.z is not None:
                current_z_val = float(self.dist.z.value)
        except (AttributeError, TypeError, ValueError):
            current_z_val = None

        # Update distance if a new redshift is supplied
        if z is not None:
            if (current_z_val is None) or (not np.isclose(z, current_z_val)):
                # Suppress the astropy cosmology optimizer warning
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=".*fval is not bracketed.*",
                        category=RuntimeWarning,
                    )
                    self.dist = Distance(z=z)
                distance_changed = True

        # Configure EBL model
        if ebl is not None and self.dist is not None:
            if ebl not in list(EBL_DATA_BUILTIN.keys()):
                raise ValueError(
                    f"ebl must be one of {list(EBL_DATA_BUILTIN.keys())}, got {ebl}"
                )
            if not os.environ.get("GAMMAPY_DATA"):
                raise ValueError(
                    "GAMMAPY_DATA environment variable not set. "
                    "Please set it to the path where the EBL data is stored. "
                    "You can copy EBL data from here: https://github.com/astrojarred/gravitational_wave_toy/tree/main/data"
                )

            self.ebl = EBLAbsorptionNormSpectralModel.read_builtin(
                ebl, redshift=self.dist.z.value
            )
            self.ebl_model = ebl
        else:
            self.ebl = None
            self.ebl_model = None

        return distance_changed

    def set_spectral_grid(self):
        """Create an interpolator for the time-energy spectral grid.

        Builds a RegularGridInterpolator on log-space coordinates (log energy, log time)
        to enable efficient interpolation of flux values at arbitrary time and energy
        points. The interpolator uses log-space to better handle the wide dynamic range
        typical of gamma-ray spectra.

        The interpolator is stored in self.SpectralGrid and is used by get_spectrum()
        and get_flux() methods. This method is idempotent and will not recreate the
        grid if it already exists.
        """
        if self.SpectralGrid is not None:
            return

        try:
            self.SpectralGrid = RegularGridInterpolator(
                (np.log10(self.energy.value), np.log10(self.time.value)),
                self.spectra,
                bounds_error=False,
                fill_value=None,
            )
        except Exception as e:
            log.error(f"Energy: {np.log10(self.energy.value)}")
            log.error(f"Time: {np.log10(self.time.value)}")
            raise e

    def show_spectral_pattern(
        self,
        resolution=100,
        return_plot=False,
        cutoff_flux=1e-20 * u.Unit("1 / (cm2 s GeV)"),
    ):
        """Display a 2D visualization of the spectral pattern as a function of time and energy.

        Creates a heatmap showing the log of the differential flux across the time-energy
        plane. Values below the cutoff flux are set to the cutoff to improve visualization
        of the spectral structure.

        Args:
            resolution: Number of grid points along each axis for the visualization.
            return_plot: If True, return the matplotlib figure instead of displaying it.
            cutoff_flux: Minimum flux value to display; values below this are set to the cutoff.

        Returns:
            matplotlib.pyplot if return_plot is True, otherwise None.
        """
        self.set_spectral_grid()

        loge = np.log10(self.energy.value)
        logt = np.log10(self.time.value)

        x = np.linspace(loge.min(), loge.max(), resolution + 1)[::-1]
        y = np.linspace(logt.min(), logt.max(), resolution + 1)

        points = []
        for e in x:
            for t in y:
                points.append([e, t])

        spectrum = self.SpectralGrid(points)
        # set everything below the cutoff energy to cutoff_energy
        cutoff_flux = cutoff_flux.to("1 / (cm2 s GeV)")
        spectrum[spectrum < cutoff_flux.value] = cutoff_flux.value

        plt.xlabel("Log(t [s])")
        plt.ylabel("Log(E [GeV])")
        plt.imshow(
            np.log10(spectrum).reshape(resolution + 1, resolution + 1),
            extent=(logt.min(), logt.max(), loge.min(), loge.max()),
            cmap="viridis",
            aspect="auto",
        )
        plt.colorbar(label="Log spectrum [ph / (cm2 s GeV)]")

        if return_plot:
            return plt

    def get_spectrum(
        self, time: u.Quantity, energy: u.Quantity | None = None
    ) -> float | np.ndarray:
        """Get the differential flux spectrum at a given time.

        Interpolates the spectral grid to return flux values at the specified time.
        If energy is not provided, returns the spectrum across the full energy grid.

        Args:
            time: Time at which to evaluate the spectrum. Must have time units.
            energy: Energy or array of energies at which to evaluate. If None, uses
                the full energy grid. Must have energy units.

        Returns:
            Differential flux (u.Quantity with units cm⁻² s⁻¹ GeV⁻¹). Returns a scalar
            for a single energy point, or an array for multiple energies.

        Raises:
            ValueError: If time or energy units are incorrect, or if spectral grid
                has not been initialized.
        """
        if not time.unit.physical_type == "time":
            raise ValueError(f"time must be a time quantity, got {time}")

        if self.SpectralGrid is None:
            raise ValueError(
                "Spectral grid not set. Please call `set_spectral_grid()` first."
            )

        time = time.to("s")

        if energy is None:
            energy = self.energy

        if not energy.unit.physical_type == "energy":
            raise ValueError(f"energy must be an energy quantity, got {energy}")

        energy = energy.to("GeV")

        if (
            isinstance(energy, np.ndarray) or isinstance(energy, list)
        ) and not isinstance(energy, u.Quantity):
            return np.array(
                [
                    self.SpectralGrid((e, np.log10(time.value)))
                    for e in np.log10(energy.value)
                ]
            ) * u.Unit("1 / (cm2 s GeV)")

        return self.SpectralGrid(
            (np.log10(energy.value), np.log10(time.value))
        ) * u.Unit("1 / (cm2 s GeV)")

    def get_flux(self, energy: u.Quantity, time: u.Quantity | None = None):
        """Get the differential flux at a given energy.

        Interpolates the spectral grid to return flux values at the specified energy.
        If time is not provided, returns the flux across the full time grid.

        Args:
            energy: Energy at which to evaluate the flux. Must have energy units.
            time: Time or array of times at which to evaluate. If None, uses the
                full time grid. Must have time units.

        Returns:
            Differential flux (u.Quantity with units cm⁻² s⁻¹ GeV⁻¹). Returns a scalar
            for a single time point, or an array for multiple times.

        Raises:
            ValueError: If energy or time units are incorrect, or if spectral grid
                has not been initialized.
        """
        if not energy.unit.physical_type == "energy":
            raise ValueError(f"energy must be an energy quantity, got {energy}")

        if self.SpectralGrid is None:
            raise ValueError(
                "Spectral grid not set. Please call `set_spectral_grid()` first."
            )

        energy = energy.to("GeV")

        if time is None:
            time = self.time

        if not time.unit.physical_type == "time":
            raise ValueError(f"time must be a time quantity, got {time}")

        time = time.to("s")

        if (isinstance(time, np.ndarray) or isinstance(time, list)) and not isinstance(
            time, u.Quantity
        ):
            return np.array(
                [
                    self.SpectralGrid((np.log10(energy.value), t))
                    for t in np.log10(time.value)
                ]
            ) * u.Unit("1 / (cm2 s GeV)")
        else:
            return self.SpectralGrid(
                (np.log10(energy.value), np.log10(time.value))
            ) * u.Unit("1 / (cm2 s GeV)")

    def get_gammapy_spectrum(
        self,
        time: u.Quantity,
        amplitude: u.Quantity | None = None,
        reference: u.Quantity = 1 * u.TeV,
    ):
        """Create a Gammapy PowerLawSpectralModel representing the spectrum at a given time.

        Fits a power law to the spectrum at the specified time and returns a Gammapy
        spectral model. The spectral index is determined from the fitted indices, and
        the amplitude is either calculated at the reference energy or provided directly.

        Args:
            time: Time at which to evaluate the spectrum. Must have time units.
            amplitude: Optional amplitude at the reference energy. If None, the amplitude
                is calculated from the flux at the reference energy.
            reference: Reference energy for the power law model. Defaults to 1 TeV.

        Returns:
            PowerLawSpectralModel instance representing the spectrum at the given time.
        """
        return PowerLawSpectralModel(
            index=-self.get_spectral_index(time),
            amplitude=self.get_flux(energy=reference, time=time).to("cm-2 s-1 TeV-1")
            if amplitude is None
            else amplitude,
            reference=reference,
        )

    def get_template_spectrum(self, time: u.Quantity, scaling_factor: int | float = 1):
        """Create a template spectral model from the spectrum at a given time.

        Extracts the full energy spectrum at the specified time and wraps it in a
        ScaledTemplateModel, which can be used for likelihood fitting with an
        adjustable normalization.

        Args:
            time: Time at which to extract the spectrum. Must have time units.
            scaling_factor: Initial scaling factor for the template model. Defaults to 1.

        Returns:
            ScaledTemplateModel instance containing the spectrum at the given time.
        """
        dNdE = self.get_spectrum(time)
        return ScaledTemplateModel(
            energy=self.energy, values=dNdE, scaling_factor=scaling_factor
        )

    def fit_spectral_indices(self):
        """Fit power-law spectral indices for each time bin.

        Performs a linear fit in log-log space (log flux vs log energy) for each time
        bin to determine the spectral index. Only time bins with at least 3 valid
        (finite and positive) flux points are fitted. The results are stored and used
        to create interpolation functions for the spectral index and amplitude as
        functions of time.

        Sets the following attributes:
            - self._indices: List of fitted spectral indices (slopes)
            - self._amplitudes: List of fitted amplitudes (log flux at 1 GeV)
            - self._index_times: List of times for which fits were successful
            - self._bad_index_times: List of times that could not be fitted
            - self.index_at: Interpolation function for spectral index vs log(time)
            - self.amplitude_at: Interpolation function for amplitude vs log(time)
        """
        spectra = self.spectra.T

        indices = []
        amplitudes = []
        times = []
        bad_times = []

        for spectrum, time in zip(spectra, self.time):
            idx = np.isfinite(spectrum) & (spectrum > 0)

            if len(idx[idx] > 3):  # need at least 3 points in the spectrum to fit
                times.append(time)
                fit = np.polyfit(
                    np.log10(self.energy[idx].value), np.log10(spectrum[idx].value), 1
                )
                m = fit[0]
                b = fit[1]
                # print(f"{time:<10.2f} {m:<10.2f} {b:<10.2f}")
                indices.append(m)
                # get amplitudes (flux at E_0 = 1 Gev)
                amplitudes.append(b)  # [ log[ph / (cm2 s GeV)]]
            else:
                bad_times.append(time)

        self._indices = indices
        self._amplitudes = amplitudes
        self._index_times = times
        self._bad_index_times = bad_times

        self.index_at = interp1d(
            np.log10([t.value for t in self._index_times]),
            self._indices,
            fill_value="extrapolate",
        )

        self.amplitude_at = lambda x: 10 ** interp1d(
            np.log10([t.value for t in self._index_times]),
            self._amplitudes,
            fill_value="extrapolate",
        )(x)

    def get_spectral_index(self, time: u.Quantity) -> float:
        """Get the power-law spectral index at a given time.

        Uses the fitted spectral indices and interpolation to return the spectral index
        (power-law slope) at the specified time. The index is defined such that
        flux ∝ E^index, so typical values are negative.

        Args:
            time: Time at which to evaluate the spectral index. Must have time units.

        Returns:
            Spectral index as a float (dimensionless).

        Raises:
            ValueError: If time units are incorrect.
        """
        if not time.unit.physical_type == "time":
            raise ValueError(f"time must be a time quantity, got {time}")

        time = time.to("s")

        return self.index_at(np.array([np.log10(time.value)]))[0]

    def get_spectral_amplitude(self, time: u.Quantity) -> u.Quantity:
        """Get the spectral amplitude (flux at 1 GeV) at a given time.

        Uses the fitted spectral amplitudes and interpolation to return the differential
        flux at 1 GeV for the specified time. This represents the normalization of
        the power-law spectrum.

        Args:
            time: Time at which to evaluate the amplitude. Must have time units.

        Returns:
            Differential flux at 1 GeV (u.Quantity with units cm⁻² s⁻¹ GeV⁻¹).

        Raises:
            ValueError: If time units are incorrect.
        """
        if not time.unit.physical_type == "time":
            raise ValueError(f"time must be a time quantity, got {time}")

        time = time.to("s")

        return self.amplitude_at(np.array([np.log10(time.value)]))[0] * u.Unit(
            "cm-2 s-1 GeV-1"
        )

    def show_spectral_evolution(self, resolution=100, return_plot=False):
        """Plot the evolution of the spectral index over time.

        Creates a plot showing how the power-law spectral index changes as a function
        of time. The spectral indices are first fitted if not already done.

        Args:
            resolution: Number of time points to evaluate for the plot.
            return_plot: If True, return the matplotlib figure instead of displaying it.

        Returns:
            matplotlib.pyplot if return_plot is True, otherwise None.
        """
        self.fit_spectral_indices()

        t = np.linspace(
            np.log10(min(self.time).value),
            np.log10(max(self.time).value),
            resolution + 1,
        )

        plt.plot(t, self.index_at(t))
        plt.xlabel("Log(t) (s)")
        plt.ylabel("Spectral Index")

        if return_plot:
            return plt

        plt.show()

    def get_integral_spectrum(
        self,
        time: u.Quantity,
        first_energy_bin: u.Quantity,
        mode: Literal["sensitivity", "photon_flux"] = "sensitivity",
        fit_powerlaw: bool = True,
    ):
        """Calculate the integral flux or energy flux over the energy range.

        Integrates the spectrum over the energy range defined by min_energy and max_energy.
        The spectrum can be represented either as a power-law model (fit_powerlaw=True) or
        as a template spectrum (fit_powerlaw=False). EBL absorption is applied if an EBL
        model is set.

        Args:
            time: Time at which to evaluate the spectrum. Must have time units.
            first_energy_bin: First energy bin (used for compatibility, not directly
                in calculation). Must have energy units.
            mode: Integration mode. "photon_flux" returns the integral flux (photons),
                "sensitivity" returns the energy flux (GeV).
            fit_powerlaw: If True, use a power-law model; if False, use the template spectrum.

        Returns:
            Integral flux (u.Quantity with units cm⁻² s⁻¹) for photon_flux mode, or
            energy flux (u.Quantity with units GeV cm⁻² s⁻¹) for sensitivity mode.

        Raises:
            ValueError: If time or energy units are incorrect, or if min/max energy
                are not set.
        """
        if not time.unit.physical_type == "time":
            raise ValueError(f"time must be a time quantity, got {time}")

        if not first_energy_bin.unit.physical_type == "energy":
            raise ValueError(
                f"first_energy_bin must be an energy quantity, got {first_energy_bin}"
            )

        if self.min_energy is None or self.max_energy is None:
            raise ValueError("Please set min and max energy for integral spectrum.")

        # spectral_index = self.get_spectral_index(time)
        # amount_to_add = 1 if mode == "ctools" else 2
        # spectral_index_plus = spectral_index + amount_to_add

        if fit_powerlaw:
            model = self.get_gammapy_spectrum(time)
        else:
            model = self.get_template_spectrum(time)

        if self.ebl is not None:
            model = model * self.ebl

        if mode == "photon_flux":
            integral_spectrum = model.integral(
                energy_min=self.min_energy, energy_max=self.max_energy
            ).to("cm-2 s-1")
        else:
            integral_spectrum = model.energy_flux(
                energy_min=self.min_energy, energy_max=self.max_energy
            ).to("GeV cm-2 s-1")

        return integral_spectrum

    def get_fluence(
        self,
        start_time: u.Quantity,
        stop_time: u.Quantity,
        mode: Literal["sensitivity", "photon_flux"] = "sensitivity",
    ):
        """Calculate the fluence (time-integrated flux) over a time interval.

        Integrates the integral spectrum over time from start_time to stop_time.
        The fluence represents the total energy or photons received per unit area
        over the observation period.

        Args:
            start_time: Start of the integration interval. Must have time units.
            stop_time: End of the integration interval. Must have time units.
            mode: Integration mode. "photon_flux" returns photon fluence,
                "sensitivity" returns energy fluence.

        Returns:
            Fluence (u.Quantity with units cm⁻² for photon_flux mode, or
            GeV cm⁻² for sensitivity mode).

        Raises:
            ValueError: If time units are incorrect.
        """
        if not start_time.unit.physical_type == "time":
            raise ValueError(f"start_time must be a time quantity, got {start_time}")
        if not stop_time.unit.physical_type == "time":
            raise ValueError(f"stop_time must be a time quantity, got {stop_time}")

        start_time = start_time.to("s")
        stop_time = stop_time.to("s")

        first_energy_bin = min(self.energy)

        unit = u.Unit("cm-2") if mode == "photon_flux" else u.Unit("GeV cm-2")
        fluence = (
            integrate.quad(
                lambda time: self.get_integral_spectrum(
                    time * u.s, first_energy_bin, mode=mode
                ).value,
                start_time.value,
                stop_time.value,
            )[0]
            * unit
        )

        log.debug(f"    Fluence: {fluence}")
        return fluence

    def output(self):
        """Generate a dictionary representation of the source for serialization.

        Creates a dictionary containing all source attributes except for large data
        arrays and internal interpolation objects. This is useful for saving results
        or converting to JSON. Numpy arrays are converted to lists, and numpy numeric
        types are converted to native Python types.

        Returns:
            Dictionary containing source metadata and observation results, excluding
            large data structures like time, energy, spectra, and interpolation grids.
        """
        keys_to_drop = [
            "time",
            "energy",
            "spectra",
            "SpectralGrid",
            "rng",
            "power_law_slopes",
            "spectral_indices",
            "ebl",
            "_indices",
            "_index_times",
            "_amplitudes",
            "_bad_index_times",
            "index_at",
            "amplitude_at",
            "_num_iters",
            "_last_guess",
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
                elif isinstance(v, u.Quantity):  # check if value is a Quantity object
                    o[k] = v  # convert Quantity to list
                elif isinstance(v, np.ndarray):
                    o[k] = v.tolist()
                else:
                    o[k] = v

        return o

    def visibility_function(
        self,
        stop_time: float,
        start_time: u.Quantity,
        sensitivity: Sensitivity,
        sensitivity_mode: Literal["sensitivity", "photon_flux"] = "sensitivity",
    ) -> float:
        """Calculate the log ratio of average flux to sensitivity.

        Computes the logarithm of (average_flux / sensitivity) for a given exposure
        window. A positive value indicates the source is detectable above the sensitivity
        threshold. The average flux is calculated from the fluence over the exposure time.

        Args:
            stop_time: End time of the exposure window in seconds (as float).
            start_time: Start time of the exposure window. Must have time units.
            sensitivity: Sensitivity object providing the sensitivity curve.
            sensitivity_mode: Mode for sensitivity calculation ("sensitivity" or "photon_flux").

        Returns:
            Log10 ratio of average flux to sensitivity (dimensionless float).
            Positive values indicate detectability.
        """
        # start time = delay
        # stop_time (t) = delay + exposure time

        stop_time = stop_time * u.s

        fluence = self.get_fluence(start_time, stop_time, mode=sensitivity_mode)

        average_flux = fluence / (stop_time - start_time)

        sens = sensitivity.get(
            t=(stop_time - start_time),
            mode=sensitivity_mode,
        ).to("GeV / (cm2 s)" if sensitivity_mode == "sensitivity" else "1 / (cm2 s)")

        # exposure_time = stop_time - start_time

        # print(f"{'++' if average_flux > sens else '--'}, Exp time: {exposure_time}, Average flux: {average_flux}, Sensitivity: {sens}")
        return np.log10(average_flux.value) - np.log10(sens.value)

    def is_visible(
        self,
        start_time: u.Quantity,
        stop_time: u.Quantity,
        sensitivity: Sensitivity,
        sensitivity_mode: Literal["sensitivity", "photon_flux"] = "sensitivity",
    ) -> bool:
        """Determine if the source is detectable above the sensitivity threshold.

        Checks whether the average flux over the specified time window exceeds the
        sensitivity for that exposure time. This is a binary check based on the
        visibility function.

        Args:
            start_time: Start of the observation window. Must have time units.
            stop_time: End of the observation window. Must have time units.
            sensitivity: Sensitivity object providing the sensitivity curve.
            sensitivity_mode: Mode for sensitivity calculation ("sensitivity" or "photon_flux").

        Returns:
            True if the source is detectable (average flux > sensitivity), False otherwise.
        """
        # print(f'Start time: {start_time}, Stop time: {stop_time}')
        return (
            self.visibility_function(
                stop_time.to_value(u.s),
                start_time.to(u.s),
                sensitivity,
                sensitivity_mode,
            )
            > 0
        )

    def observe(
        self,
        sensitivity: Sensitivity,
        start_time: u.Quantity = 0 * u.s,
        min_energy: u.Quantity = None,
        max_energy: u.Quantity = None,
        max_time: u.Quantity = 4 * u.hour,
        target_precision: u.Quantity = 1 * u.s,
        n_time_steps: int = 10,
        xtol: float = 1e-5,
        rtol: float = 1e-5,
        sensitivity_mode: Literal["sensitivity", "photon_flux"] = "sensitivity",
        **kwargs,
    ):
        """Determine when the source becomes detectable and calculate observation parameters.

        Finds the earliest time at which the source becomes visible above the sensitivity
        threshold, starting from start_time and searching up to max_time. Uses root-finding
        to locate when the visibility function crosses zero. If the source is immediately
        visible, returns the minimum observation time (target_precision).

        Sets observation results in the source object:
            - self.seen: True if detectable, False if not, "error" if calculation failed
            - self.obs_time: Required observation time to detect the source
            - self.end_time: Time at which detection occurs
            - self.error_message: Error message if calculation failed

        Args:
            sensitivity: Sensitivity object providing the sensitivity curve.
            start_time: Time to start searching for detectability. Defaults to 0 s.
            min_energy: Minimum energy for integration. If None, uses sensitivity limits.
            max_energy: Maximum energy for integration. If None, uses sensitivity limits.
            max_time: Maximum time to search for detectability. Defaults to 4 hours.
            target_precision: Time precision for the observation time. Defaults to 1 s.
            n_time_steps: Number of time steps for root-finding grid.
            xtol: Absolute tolerance for root-finding.
            rtol: Relative tolerance for root-finding.
            sensitivity_mode: Mode for sensitivity calculation ("sensitivity" or "photon_flux").
            **kwargs: Additional arguments passed to the root-finding function.

        Returns:
            Dictionary representation of the source (via output() method) containing
            observation results.

        Raises:
            ValueError: If time or energy units are incorrect, or if min/max energy
                are not set.
        """
        if not start_time.unit.physical_type == "time":
            raise ValueError(f"start_time must be a time quantity, got {start_time}")

        if not max_time.unit.physical_type == "time":
            raise ValueError(f"max_time must be a time quantity, got {max_time}")

        if not target_precision.unit.physical_type == "time":
            raise ValueError(
                f"target_precision must be a time quantity, got {target_precision}"
            )

        # set energy limits to match the sensitivity
        if min_energy is None or max_energy is None:
            self.min_energy, self.max_energy = sensitivity.energy_limits

        if self.min_energy is None or self.max_energy is None:
            raise ValueError("Please set min and max energy for observe function.")

        if not self.min_energy.unit.physical_type == "energy":
            raise ValueError(
                f"min_energy must be an energy quantity, got {self.min_energy}"
            )
        if not self.max_energy.unit.physical_type == "energy":
            raise ValueError(
                f"max_energy must be an energy quantity, got {self.max_energy}"
            )

        self.min_energy = self.min_energy.to("GeV")
        self.max_energy = self.max_energy.to("GeV")

        start_time = start_time.to("s")
        max_time = max_time.to("s")

        # check if immediately visible
        if self.is_visible(
            start_time, target_precision + start_time, sensitivity, sensitivity_mode
        ):
            self.end_time = target_precision + start_time
            self.obs_time = target_precision
            self.seen = True
            return self.output()

        try:
            res = find_roots(
                self.visibility_function,
                lower_bound=start_time.to_value(u.s) + target_precision.to_value(u.s),
                upper_bound=start_time.to_value(u.s) + max_time.to_value(u.s),
                points_scale="log",
                args=(start_time, sensitivity, sensitivity_mode),
                nbin=n_time_steps,
                xtol=xtol,
                rtol=rtol,
                method="brentq",
                **kwargs,
            )

            first_root = np.nanmin(res[0])

            if math.isnan(first_root):
                self.end_time = -1 * u.s
                self.obs_time = -1 * u.s
                self.seen = False
                return self.output()

            end_time = round((first_root / target_precision).value) * target_precision

            self.end_time = end_time
            self.obs_time = end_time - start_time
            self.seen = True

            return self.output()

        except Exception as e:
            print(e)

            self.seen = "error"
            self.error_message = str(e)

            return self.output()

    def get_significance(
        self,
        start_time: u.Quantity,
        stop_time: u.Quantity,
        sensitivity: Sensitivity,
        sensitivity_mode: Literal["sensitivity", "photon_flux"] = "sensitivity",
    ):
        """Calculate the detection significance for a given observation window.

        Computes the significance in units of sigma, assuming the sensitivity curve
        represents a 5-sigma detection threshold. The significance scales with the
        square root of time for a given flux level.

        Args:
            start_time: Start of the observation window. Must have time units.
            stop_time: End of the observation window. Must have time units.
            sensitivity: Sensitivity object providing the sensitivity curve.
            sensitivity_mode: Mode for sensitivity calculation ("sensitivity" or "photon_flux").

        Returns:
            Detection significance in units of sigma (dimensionless float or array).

        Raises:
            ValueError: If time units are incorrect.
        """
        if not start_time.unit.physical_type == "time":
            raise ValueError(f"start_time must be a time quantity, got {start_time}")
        if not stop_time.unit.physical_type == "time":
            raise ValueError(f"stop_time must be a time quantity, got {stop_time}")

        start_time = start_time.to("s")
        stop_time = stop_time.to("s")

        fluence = self.get_fluence(start_time, stop_time, mode=sensitivity_mode)
        average_flux = fluence / (stop_time - start_time)
        sens = sensitivity.get(
            t=(stop_time - start_time),
            mode=sensitivity_mode,
        ).to("GeV / (cm2 s)" if sensitivity_mode == "sensitivity" else "1 / (cm2 s)")

        # sens represents the 5sigma sensitivity curve
        # and significance scales with sqrt(t) for a given flux
        sig = 5 * (average_flux / sens)

        return sig

    def get_significance_evolution(
        self,
        sensitivity: Sensitivity,
        start_time: u.Quantity,
        max_time: u.Quantity = 12 * u.hour,
        min_energy: u.Quantity = None,
        max_energy: u.Quantity = None,
        n_time_steps: int = 50,
        sensitivity_mode: Literal["sensitivity", "photon_flux"] = "sensitivity",
    ):
        """Calculate the evolution of detection significance over time.

        Computes the significance as a function of observation end time, starting
        from start_time and evaluating at logarithmically spaced points up to max_time.
        This shows how the significance improves with longer observations.

        Args:
            sensitivity: Sensitivity object providing the sensitivity curve.
            start_time: Start time for all observations. Must have time units.
            max_time: Maximum end time to evaluate. Defaults to 12 hours.
            min_energy: Minimum energy for integration. If None, uses sensitivity limits.
            max_energy: Maximum energy for integration. If None, uses sensitivity limits.
            n_time_steps: Number of time points to evaluate. Defaults to 50.
            sensitivity_mode: Mode for sensitivity calculation ("sensitivity" or "photon_flux").

        Returns:
            Tuple of (end_times, significances) where:
                - end_times: Array of observation end times in seconds
                - significances: Array of corresponding significance values in sigma

        Raises:
            ValueError: If time or energy units are incorrect, or if min/max energy
                are not set.
        """
        if not start_time.unit.physical_type == "time":
            raise ValueError(f"start_time must be a time quantity, got {start_time}")

        if not max_time.unit.physical_type == "time":
            raise ValueError(f"max_time must be a time quantity, got {max_time}")

        # set energy limits to match the sensitivity
        if min_energy is None or max_energy is None:
            self.min_energy, self.max_energy = sensitivity.energy_limits

        if self.min_energy is None or self.max_energy is None:
            raise ValueError(
                "Please set min and max energy for significance evolution function."
            )

        if not self.min_energy.unit.physical_type == "energy":
            raise ValueError(
                f"min_energy must be an energy quantity, got {self.min_energy}"
            )
        if not self.max_energy.unit.physical_type == "energy":
            raise ValueError(
                f"max_energy must be an energy quantity, got {self.max_energy}"
            )

        self.min_energy = self.min_energy.to("GeV")
        self.max_energy = self.max_energy.to("GeV")

        start_time = start_time.to("s")
        max_time = max_time.to("s")

        end_times = np.logspace(
            np.log10(start_time.value),
            np.log10(max_time.value),
            n_time_steps,
        )

        # calculate significance
        sig = np.array(
            [
                self.get_significance(
                    start_time,
                    end_time * u.s,
                    sensitivity,
                    sensitivity_mode=sensitivity_mode,
                )
                for end_time in end_times
            ]
        )

        return end_times, sig
