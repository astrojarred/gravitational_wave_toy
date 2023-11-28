from pathlib import Path

import astropy.units as u

# %matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from gammapy.modeling.models import PowerLawSpectralModel
from tqdm import tqdm

from gravitational_wave_toy.ctairf import IRFHouse
from gravitational_wave_toy.sensitivity import SensitivityGammapy
from gravitational_wave_toy.util import suppress_warnings_and_logs
from gravitational_wave_toy.logging import logger

sns.set()
    
log = logger(__name__)

def compare_all_IRFs():
    
    log.info("Starting comparison of all IRFs")
    
    house = IRFHouse(base_directory="/Users/jarred/Documents/Work/CTA-IRFs")

    for site in ["south", "north"]:
        for configuration in ["alpha", "omega"]:
            log.info(f"Processing IRF: {site} {configuration}")
            colors = sns.color_palette("husl", 3)
            plt.figure(figsize=(10, 8))
            for i, zenith in enumerate([20, 40, 60]):
                irf = house.get_irf(
                    site=site,
                    configuration=configuration,
                    zenith=zenith,
                    duration=1800,
                    azimuth="average",
                    version="prod5-v0.1" if configuration == "alpha" else "prod3b-v2",
                )

                grbsens_dir = Path(
                    f"/Users/jarred/Documents/Work/gravitational_wave_toy/CTA_sensitivity/grbsens_output_v3_Sep_2022/{configuration}_configuration"
                )
                file = (
                    grbsens_dir
                    / f"grbsens-5.0sigma_t1s-t16384s_irf-{irf.site.name.capitalize()}_z{irf.zenith}_0.5h.txt"
                )
                cols = [
                    "duration",
                    "crab_flux",
                    "photon_flux",
                    "energy_flux",
                    "sensitivity",
                ]

                ctools_curve = pd.read_csv(file, sep="\t", comment="#", names=cols)

                times = ctools_curve.duration.to_numpy()
                tables = {}
                gammapy_res = {}

                model = PowerLawSpectralModel(
                    index=2.1, amplitude="5.7e-13 cm-2 s-1 TeV-1", reference="1 TeV"
                )
                
                pbar = tqdm(times, desc=f"Processing IRF: {irf.version} {irf.site} {irf.zenith}deg")

                for duration in pbar:
                    with suppress_warnings_and_logs():
                        s = SensitivityGammapy.gamma_sens(
                            irf=irf.filepath,
                            observatory=f"cta_{irf.site.name}",
                            duration=duration * u.s,
                            model=model,
                            radius=3.0 * u.deg,
                            min_energy=0.03 * u.TeV,
                            max_energy=10 * u.TeV,
                            sigma=5,
                            bins=1,
                            offset=0.0,
                        )

                    tables[duration] = s
                    gammapy_res[duration] = s["e2dnde"][0]
                    
                    # update tqdm description
                    pbar.set_description(f"Processing IRF: {irf.version} {irf.site} {irf.zenith}deg {duration}s")

                plt.loglog(ctools_curve["duration"], ctools_curve["sensitivity"], label=f"z{zenith}: ctools", marker="x", color=colors[i])
                plt.loglog(tables.keys(), gammapy_res.values(), label=f"z{zenith}: gammapy", marker="o", color=colors[i])
                plt.legend(prop={'family': 'monospace', 'size': 18})
                plt.xlabel("Log Duration [s]", fontsize=20)
                plt.ylabel("Log Sensitivity [erg cm^-2 s^-1]", fontsize=20)
                plt.title(f"CTA-{site[0].upper()} {configuration}: Ctools vs Gammapy Sens.", fontsize=20)
                # add text in bottom left corner of plot with irf version
            plt.text(0.05, 0.05, f"IRF: {irf.version} {configuration} config", transform=plt.gca().transAxes, fontsize=14)
            plt.show()
