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


class GRB:
    def __init__(
        self,
        filepath: str | Path,
        min_energy: u.Quantity | None = None,
        max_energy: u.Quantity | None = None,
        ebl: str | None = None,
    ) -> None:
        if isinstance(min_energy, u.Quantity):
            min_energy = min_energy.to("GeV")
        if isinstance(max_energy, u.Quantity):
            max_energy = max_energy.to("GeV")

        self.filepath = Path(filepath).absolute()
        self.min_energy, self.max_energy = min_energy, max_energy
        self.seen = False
        self.obs_time = -1 * u.s
        self.start_time = -1 * u.s
        self.end_time = -1 * u.s
        self.error_message = ""
        self.dist = None
        self.file_type: Literal["fits", "txt", None] = None

        try:
            # Extract GRB ID from directory name (e.g., "GRB001" -> 1)
            if self.filepath.is_dir():
                # For directory path, extract from directory name
                dir_name = self.filepath.name
                if dir_name.startswith("GRB"):
                    self.id = int(
                        dir_name[3:]
                    )  # Remove "GRB" prefix and convert to int
                else:
                    self.id = 0
            else:
                # For file path, use original logic
                self.id = int(self.filepath.stem.split("_")[1])
        except (ValueError, IndexError):
            self.id = 0

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
        return f"<GRB(id={self.id})>"
    
    @property
    def metadata(self) -> dict:
        return {
            "id": self.id,
            "long": self.long,
            "lat": self.lat,
            "eiso": self.eiso,
            "dist": self.dist,
            "angle": self.angle,
        }

    def read_fits(self) -> None:
        with fits.open(self.filepath) as hdu_list:
            self.long = hdu_list[0].header["LONG"] * u.Unit("rad")
            self.lat = hdu_list[0].header["LAT"] * u.Unit("rad")
            self.eiso = hdu_list[0].header["EISO"] * u.Unit("erg")
            self.dist = Distance(hdu_list[0].header["DISTANCE"], unit="kpc")
            self.angle = hdu_list[0].header["ANGLE"] * u.Unit("deg")

            datalc = hdu_list[3].data
            datatime = hdu_list[2].data
            dataenergy = hdu_list[1].data

            self.time = datatime.field(0) * u.Unit("s")
            self.energy = dataenergy.field(0) * u.Unit("GeV")

            self.spectra = np.array(
                [datalc.field(i) for i, e in enumerate(self.energy)]
            ) * u.Unit("1 / (cm2 s GeV)")

    def read_txt(self) -> None:
        # expect a directory containing GRB spectral files like GRB001_tobs=00.txt, GRB001_tobs=01.txt, etc.
        dir_path = self.filepath

        # Extract base name from directory name (e.g., "GRB001" from "/path/to/GRB001/")
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
        time_indices = []

        for p in candidates:
            arr = np.loadtxt(p)
            energy = arr[:, 0] * u.GeV
            dNdE = arr[:, 1] * u.Unit("1 / (cm2 s GeV)")

            spectra_columns.append(dNdE)
            time_indices.append(extract_index(p))

        # set energy grid
        self.energy = energy

        # Create time array from indices (assuming indices represent time steps)
        self.time = u.Quantity(time_indices) * u.s

        # build spectra with shape (n_energy, n_time)
        spectra_stack = u.Quantity(spectra_columns)  # (n_time, n_energy)
        self.spectra = spectra_stack.T  # (n_energy, n_time)

        # Set default metadata since we don't have lightcurve data
        self.eiso = 0 * u.erg  # Default Eiso
        self.fluence = 0 * u.Unit("1 / cm2")  # Default fluence
        self.dist = None  # default to None, will be set by redshift if provided
        self.angle = 0 * u.deg
        self.long = 0 * u.rad
        self.lat = 0 * u.rad

        if not isinstance(self.min_energy, u.Quantity):
            self.min_energy = self.energy.min()
        if not isinstance(self.max_energy, u.Quantity):
            self.max_energy = self.energy.max()
            
    def read_csv(self) -> None:
        # expect a single CSV file with columns: time [s], energy [GeV], flux [cm-2 s-1 GeV-1]
        # and optionally a metadata file named {base}_metadata.txt in the same directory
        
        csv_path = self.filepath
        
        # Read CSV data
        df = pd.read_csv(csv_path)
        
        # Extract columns (handle both with and without brackets in column names)
        # Uses substring matching for flexibility (e.g., "time [s]", "timestamp", "energy [GeV]", etc.)
        time_col = None
        energy_col = None
        flux_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'time' in col_lower:
                time_col = col
            elif 'energy' in col_lower:
                energy_col = col
            elif 'flux' in col_lower:
                flux_col = col
        
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
        metadata_path = csv_path.parent / f"{csv_path.stem}_metadata.txt"
        
        # Default values
        self.eiso = 0 * u.erg
        self.fluence = 0 * u.Unit("1 / cm2")
        self.dist = None
        self.angle = 0 * u.deg
        self.long = 0 * u.rad
        self.lat = 0 * u.rad
        
        if metadata_path.exists():
            try:
                metadata_df = pd.read_csv(metadata_path)
                
                metadata_dict = {}
                if 'Parameter' in metadata_df.columns and 'Value' in metadata_df.columns:
                    for _, row in metadata_df.iterrows():
                        # Convert to string and strip, handling NaN/float cases
                        param = str(row['Parameter']).strip() if pd.notna(row['Parameter']) else ''
                        value = row['Value']
                        
                        # Handle Units column - convert to string, handle NaN/empty
                        if 'Units' in metadata_df.columns:
                            unit_val = row['Units']
                            if pd.notna(unit_val):
                                unit_str = str(unit_val).strip()
                            else:
                                unit_str = ''
                        else:
                            unit_str = ''
                        
                        if pd.notna(value) and param:
                            metadata_dict[param.lower()] = {
                                'value': value,
                                'unit': unit_str
                            }
                
                # Parse metadata with defaults using a mapping dictionary
                # Format: 'metadata_key': ('attribute_name', 'default_unit', 'parser_func')
                metadata_mapping = {
                    'event_id': ('id', None, lambda v, unit_str: int(float(v))),
                    'longitude': ('long', 'rad', lambda v, unit_str: float(v) * u.Unit(unit_str or 'rad')),
                    'latitude': ('lat', 'rad', lambda v, unit_str: float(v) * u.Unit(unit_str or 'rad')),
                    'eiso': ('eiso', 'erg', lambda v, unit_str: float(v) * u.Unit(unit_str or 'erg')),
                    'distance': ('dist', 'kpc', lambda v, unit_str: Distance(float(v), unit=unit_str or 'kpc')),
                    'angle': ('angle', 'deg', lambda v, unit_str: float(v) * u.Unit(unit_str or 'deg')),
                }
                
                for metadata_key, (attr_name, default_unit, parser_func) in metadata_mapping.items():
                    if metadata_key in metadata_dict:
                        try:
                            value = metadata_dict[metadata_key]['value']
                            unit = metadata_dict[metadata_key]['unit'] or default_unit
                            setattr(self, attr_name, parser_func(value, unit))
                        except (ValueError, TypeError, u.UnitConversionError):
                            pass  # Keep default value
                        
            except Exception as e:
                log.warning(f"Could not parse metadata file {metadata_path}: {type(e).__name__} {e}. Using defaults.")
        
        # Set energy limits if not already set
        if not isinstance(self.min_energy, u.Quantity):
            self.min_energy = self.energy.min()
        if not isinstance(self.max_energy, u.Quantity):
            self.max_energy = self.energy.max()

    def set_ebl_model(self, ebl: str | None, z: float | None = None) -> bool:
        """Set or update the EBL absorption model and optionally the source redshift.

        Returns True if the distance (redshift) was changed.
        """
        distance_changed = False

        # Determine current redshift if available
        current_z_val = None
        try:
            current_z_val = float(self.dist.z.value)
        except (AttributeError, TypeError):
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
        if not time.unit.physical_type == "time":
            raise ValueError(f"time must be a time quantity, got {time}")

        time = time.to("s")

        if energy is None:
            energy = self.energy

        if not energy.unit.physical_type == "energy":
            raise ValueError(f"energy must be an energy quantity, got {energy}")

        energy = energy.to("GeV")

        if (
            isinstance(energy, np.ndarray) or isinstance(energy, list)
        ) and not isinstance(energy, u.Quantity):
            print(energy, type(energy))
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
        if not energy.unit.physical_type == "energy":
            raise ValueError(f"energy must be an energy quantity, got {energy}")

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
        return PowerLawSpectralModel(
            index=-self.get_spectral_index(time),
            amplitude=self.get_flux(energy=reference, time=time).to("cm-2 s-1 TeV-1")
            if amplitude is None
            else amplitude,
            reference=reference,
        )

    def get_template_spectrum(self, time: u.Quantity, scaling_factor: int | float = 1):
        dNdE = self.get_spectrum(time)
        return ScaledTemplateModel(
            energy=self.energy, values=dNdE, scaling_factor=scaling_factor
        )

    def fit_spectral_indices(self):
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
        if not time.unit.physical_type == "time":
            raise ValueError(f"time must be a time quantity, got {time}")

        time = time.to("s")

        return self.index_at(np.array([np.log10(time.value)]))[0]

    def get_spectral_amplitude(self, time: u.Quantity) -> u.Quantity:
        if not time.unit.physical_type == "time":
            raise ValueError(f"time must be a time quantity, got {time}")

        time = time.to("s")

        return self.amplitude_at(np.array([np.log10(time.value)]))[0] * u.Unit(
            "cm-2 s-1 GeV-1"
        )

    def show_spectral_evolution(self, resolution=100, return_plot=False):
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
        use_model: bool = True,
    ):
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

        if use_model:
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
        if not start_time.unit.physical_type == "time":
            raise ValueError(f"start_time must be a time quantity, got {start_time}")

        if not max_time.unit.physical_type == "time":
            raise ValueError(f"max_time must be a time quantity, got {max_time}")

        # set energy limits to match the sensitivity
        if min_energy is None or max_energy is None:
            self.min_energy, self.max_energy = sensitivity.energy_limits

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
