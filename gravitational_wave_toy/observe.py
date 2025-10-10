import math
import os
import re
from pathlib import Path
from typing import Literal

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
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
from .sensitivity import Sensitivity

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
        try:
            self.id = int(self.filepath.stem.split("_")[1])
        except ValueError:
            self.id = 0

        # choose reader based on file extension
        name_lower = self.filepath.name.lower()
        if name_lower.endswith((".fits", ".fit", ".fits.gz", ".fit.gz")):
            self.read_fits()
        elif name_lower.endswith(".txt"):
            self.read_txt()
        else:
            raise ValueError(f"Unsupported file format for {self.filepath}")

        # set spectral grid
        self.SpectralGrid = None
        self.set_spectral_grid()

        # fit spectral indices
        self.fit_spectral_indices()

        # set EBL model
        if ebl is not None:
            if ebl not in list(EBL_DATA_BUILTIN.keys()):
                raise ValueError(
                    f"ebl must be one of {list(EBL_DATA_BUILTIN.keys())}, got {ebl}"
                )
            # check that environment variable is set
            if not os.environ.get("GAMMAPY_DATA"):
                raise ValueError(
                    "GAMMAPY_DATA environment variable not set. "
                    "Please set it to the path where the EBL data is stored. "
                    "You can copy EBL data from here: https://github.com/astrojarred/gravitational_wave_toy/tree/main/data"
                )

            self.ebl = EBLAbsorptionNormSpectralModel.read_builtin(
                ebl,
                redshift=self.dist.z,
            )
            self.ebl_model = ebl
        else:
            self.ebl = None
            self.ebl_model = None

        log.debug(f"Loaded event {self.angle}º")

    def __repr__(self):
        return f"<GRB(id={self.id})>"

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
        # expect a lightcurve file like GRB001_lc.txt
        lc_path = self.filepath

        # read lightcurve columns
        # header: t_obs[s], flux_int[ph/cm2/s], photon_index, z, Eiso, Fluence, flux_ebl[ph/cm2/s]
        lc_data = np.loadtxt(lc_path)

        times_s = lc_data[:, 0] * u.s
        photon_index = lc_data[:, 2]
        z = float(lc_data[0, 3])
        eiso = float(lc_data[0, 4]) * u.erg
        fluence_val = float(lc_data[0, 5]) * u.Unit("1 / cm2")

        # store metadata
        self.time = times_s
        self.photon_index = photon_index
        self.eiso = eiso
        self.fluence = fluence_val
        self.dist = Distance(z=z)
        # angles are not provided in txt, default to 0
        self.angle = 0 * u.deg
        self.long = 0 * u.rad
        self.lat = 0 * u.rad

        # find associated spectral files in same directory
        dir_path = lc_path.parent
        stem = lc_path.name
        base = stem.split("_lc.txt")[0]

        candidates = list(dir_path.glob(f"{base}_tobs=*.txt"))
        if len(candidates) == 0:
            raise FileNotFoundError(
                f"No spectral files matching {base}_tobs=*.txt found in {dir_path}"
            )

        def extract_index(p: Path) -> int:
            m = re.search(r"_tobs=(\d+)\.txt$", p.name)
            return int(m.group(1)) if m else -1

        candidates.sort(key=extract_index)

        energy_grid_GeV = None
        spectra_columns = []  # list of flux arrays per time
        selected_times = []

        for p in candidates:
            arr = np.loadtxt(p)
            energy_eV = arr[:, 0] * u.eV

            # detect header to determine flux units
            header = ""
            try:
                with p.open("r") as fh:
                    for line in fh:
                        if line.strip().startswith("#"):
                            header = line.strip()
                            break
            except Exception:
                header = ""

            if "erg/cm2/s/eV" in header:
                energy_GeV = energy_eV.to("GeV")
                flux_energy_density = arr[:, 1] * (u.erg / (u.cm**2 * u.s * u.eV))
                photon_energy = energy_eV.to(u.erg)
                flux_per_eV = (flux_energy_density / photon_energy).to(
                    "1 / (cm2 s eV)"
                )
                flux_per_GeV = flux_per_eV.to("1 / (cm2 s GeV)")
            else:
                # assume photon flux already provided per eV
                energy_GeV = energy_eV.to("GeV")
                flux_per_eV = arr[:, 1] * u.Unit("1 / (cm2 s eV)")
                flux_per_GeV = flux_per_eV.to("1 / (cm2 s GeV)")

            if energy_grid_GeV is None:
                energy_grid_GeV = energy_GeV
            else:
                # if grids differ, interpolate onto the first grid
                if not np.allclose(energy_GeV.value, energy_grid_GeV.value, rtol=0, atol=0):
                    interp = interp1d(
                        energy_GeV.value,
                        flux_per_GeV.value,
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                    flux_per_GeV = interp(energy_grid_GeV.value) * u.Unit(
                        "1 / (cm2 s GeV)"
                    )

            spectra_columns.append(flux_per_GeV)

            idx = extract_index(p)
            if 0 <= idx < len(self.time):
                selected_times.append(self.time[idx])

        # set energy grid
        self.energy = energy_grid_GeV

        # align times with loaded spectra; if none matched by index, fall back to first N
        if len(selected_times) == len(spectra_columns):
            self.time = u.Quantity(selected_times)
        else:
            self.time = self.time[: len(spectra_columns)]

        # build spectra with shape (n_energy, n_time)
        spectra_stack = u.Quantity(spectra_columns)  # (n_time, n_energy)
        self.spectra = spectra_stack.T  # (n_energy, n_time)
        
        if not isinstance(self.min_energy, u.Quantity):
            self.min_energy = self.energy.min()
        if not isinstance(self.max_energy, u.Quantity):
            self.max_energy = self.energy.max()

    def set_spectral_grid(self):
        if self.SpectralGrid is not None:
            return
        
        self.SpectralGrid = RegularGridInterpolator(
            (np.log10(self.energy.value), np.log10(self.time.value)), self.spectra
        )

    def show_spectral_pattern(self, resolution=100, return_plot=False, cutoff_flux=1e-15 * u.Unit("1 / (cm2 s GeV)")):
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

        spectral_index = self.get_spectral_index(time)
        amount_to_add = 1 if mode == "ctools" else 2
        spectral_index_plus = spectral_index + amount_to_add

        if not use_model:
            integral_spectrum = (
                self.get_flux(first_energy_bin, time=time)
                * (first_energy_bin ** (-spectral_index) / spectral_index_plus)
                * (
                    (self.max_energy**spectral_index_plus)
                    - (self.min_energy**spectral_index_plus)
                )
            )
        else:
            model = self.get_gammapy_spectrum(time)

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
