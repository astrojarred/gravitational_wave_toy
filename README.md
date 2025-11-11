<p align="center">
  <a href="" rel="noopener">
 <img style="width: 200px; height: 200px; max-width: 100%;" src="images/project_logo.png" src="images/project_logo.png" alt="GW Toy logo"
 ></a>
</p>

<h3 align="center">Gravitational Wave Toy</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![GitHub Issues](https://img.shields.io/github/issues/astrojarred/gravitational_wave_toy.svg)](https://github.com/astrojarred/gravitational_wave_toy/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/astrojarred/gravitational_wave_toy.svg)](https://github.com/astrojarred/gravitational_wave_toy/pulls)
![GitHub License](https://img.shields.io/github/license/astrojarred/gravitational_wave_toy)


</div>

---

<p align="center"> The purpose of this is to simualte the possibility of detecting very-high-energy electromagnetic counterparts to gravitational wave events events. The package can also create heatmaps for observations of simulated gravitational wave events to determine under which circumstances the event could be detectable by a gamma-ray observatory.
</p>

## 📝 Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Authors](#️authors)
- [Instructions](#instructions)
  - [GW Observations](#instructions-gw-obs)
    - [Example Simulation](#example-simulation)
    - [Example followup calculation](#example-followup)
  - [Reading Results](#instructions-reading)
  - [Plotting Heatmaps](#instructions-plotting)

## 🧐 About<a name = "about"></a>

An input GRB model and an instrument sensitivity curve specific to the desired observational conditions are provided as input. Given a delay from the onset of the event, the code uses a simple optimization algorithm to determine at which point in time (if at all) the source would be detectable by an instrument with the given sensitivity.

The purpose of this is to simualte the possibility of detecting very-high-energy electromagnetic counterparts to gravitational wave events events. The package can also create heatmaps for observations of simulated gravitational wave events to determine under which circumstances the event could be detectable by a gamma-ray observatory.

For more information, check out our [ICRC proceedings from 2023](https://arxiv.org/abs/2310.07413).

## 🏁 Getting Started<a name = "getting_started"></a>

You need a gravitational wave event catalog. If you don't have this please contact the maintainers.

In addition, you need a python installation and the packages outlines in `pyproject.toml`. We recommend using `poetry` or `conda` to manage your python environment.

Note: dask is only necessary to read in the output data with the `plot` class.

## ✍️ Authors<a name = "Maintainers"></a>

- [Jarred Green](https://github.com/astrojarred) (jgreen at mpp.mpg.de)
- Barbara Patricelli (barbara.patricelli at pi.infn.it)
- Antonio Stamerra (antonio.stamerra at inaf.it)

## 🧑‍🏫 Instructions<a name = "instructions"></a>

### GW Observations<a name = "instructions-gw-obs"></a>

#### Methods
This code simulates observations of simulated gravitational wave events to determine under which circumstances the event could be detectable by a gamma-ray observatory. An input GRB model and an instrument sensitivity curve specific to the desired observational conditions are provided as input. Given a delay from the onset of the event, the code uses a simple optimization algorithm to determine at which point in time (if at all) the source would be detectable by an instrument with the given sensitivity.

#### Inputs

- Instrument sensitivity curves
  - This can be calculated using [gammapy](https://gammapy.org/) (recommended) or [ctools](http://cta.irap.omp.eu/ctools/index.html), via `grbsens`. See the next section for details.
- IRFs. As of v3.0.0 the most recent Alpha configuration IRFs are prod5-v0.1 and can be found on [Zenodo](https://zenodo.org/records/5499840#.YUya5WYzbUI).
  - Place files downloaded from Zenodo in a folder with the name of the production (e.g. `prod5-v0.1`) and place all of these folders together in a parent folder (e.g. `CTA-IRFs`).
  ```
  # sample filetree
  CTA-IRFs      (this is the root directory to use with the IRFHouse class)
  ├── prod5-v0.1
  │   ├── fits
  │   │   ├── CTA-Performance-prod5-v0.1-North-20deg.FITS.tar.gz
  │   │   ├── CTA-Performance-prod5-v0.1-South-20deg.FITS.tar.gz
  │   │   ├── etc...
  │   prod3b-v2
  │   etc...
  ```
- GW event models (currently compatible with O5 simulations).
- EBL files (optional). For CTA these can be downloaded directly with `gammapy` using the `gammapy download datasets` command.
  - These are also automatically [included in the package](/data/ebl/)
  - The environment variable `GAMMAPY_DATA` must be set to the location of the parent folder of `ebl` for this to work.

#### Sensitivity Curves

- Since v3.0, this package can now use gammapy to calculate sensitivity curves. This is the recommended method.
  - Curves will be calculated on an individual basis for each event.
- The old method of using `grbsens` is still supported, but is still supported to compare various methods. An instrument sensitivity file from `grbsens`.

#### Output

- a dictionary object containing detection information and parameters of the event itself (extracted from the model)

#### Example Simulation<a name ="example-simulation"></a>

```python
from astropy import units as u

from gravitational_wave_toy.ctairf import IRFHouse
from gravitational_wave_toy.observe import GRB
from gravitational_wave_toy.sensitivity import SensitivityCtools, SensitivityGammapy

# create a home for all your IRFs
house = IRFHouse(base_directory="./CTA-IRFs")

# load in the desired IRF
irf = house.get_irf(
    site="south",
    configuration="alpha",
    zenith=20,
    duration=1800,
    azimuth="average",
    version="prod5-v0.1",
)

# create a gammapy sensitivity class
min_energy = 30 * u.GeV
max_energy = 10 * u.TeV

sens = SensitivityGammapy(
    irf=irf,
    observatory=f"cta_{irf.site.name}",
    min_energy=min_energy,
    max_energy=max_energy,
    radius=3.0 * u.deg,
)

# load in a GRB and add EBL
grb_filepath = "/path/to/your/grb/cat05_1234.fits".
grb = GRB(file, min_energy=min_energy, max_energy=max_energy, ebl="franceschini")

# load the sensitivity curve for the GRB
sens.get_sensitivity_curve(grb=grb)

# simulate the observation
delay_time = 30 * u.min

res = grb.observe(
    sensitivity=sens,
    start_time=delay_time,
    min_energy=min_energy,
    max_energy=max_energy,
)

print(f"Observation time at delay={delay_time} is {res_ebl['obs_time']} with EBL={res_ebl['ebl_model']}")
# Obs time at delay=1800.0 s is 1292.0 s with EBL=franceschini
```

### Example followup calculation<a name ="example-followup"></a>

```python
import astropy.units as u
import pandas as pd
from gravitational_wave_toy import followup

lookup_talbe = "./O5_gammapy_observations_v4.parquet"

# optional, but it's recommended to load the DataFrame first save time
# otherwise you can directly pass the filepath to the get_exposure method
lookup_df = pd.read_parquet(lookup_talbe)

event_id = 1
delay = 10 * u.s
site = "north"
zenith = 60
ebl = "franceschini"


followup.get_exposure(
    event_id=event_id,
    delay=delay,
    site=site,
    zenith=zenith,
    extrapolation_df=lookup_df,
    ebl=ebl,
)

# returns, e.g.
# {
#     'long': <Quantity 2.623 rad>,
#     'lat': <Quantity 0.186 rad>,
#     'eiso': <Quantity 2.67e+50 erg>,
#     'dist': <Quantity 466000. kpc>,
#     'obs_time': <Quantity 169. s>,
#     'error_message': '',
#     'angle': <Quantity 24.521 deg>,
#     'ebl_model': 'franceschini',
#     'min_energy': <Quantity 0.02 TeV>,
#     'max_energy': <Quantity 10. TeV>,
#     'seen': True,
#     'start_time': <Quantity 10. s>,
#     'end_time': <Quantity 179. s>,
#     'id': 4
# }
```

### Reading Results<a name = "instructions-reading"></a>

Note: The most recent simulations for CTA for the O5 observing run are [available on the CTA XWiki](https://cta.cloud.xwiki.com/xwiki/wiki/phys/view/Transients%20WG/Chasing%20the%20counterpart%20of%20GW%20alerts%20with%20CTA/O5%20Observation%20times%20with%20gw-toy%20package/) for CTA members.

It's very easy to read in the results of the `gwobserve` method when stored in csv or parquet formats. We recommend using `dask` (for very large datasets) or `pandas`. Dask has the same DataFrame functionality but is optimized for large datasets.

```python
import dask.dataframe as dd

df = dd.read_parquet("/path/to/your/file.parquet")

# filter if you'd like
alpha_config = df[df["config"] == "alpha"]

# convert to pandas
pandas_df = alpha_config.compute()
```

Note: You can also use the `gwplot.GWData` introduced below to access a very easy API for filtering and plotting the data.

### Plotting Heatmaps<a name = "instructions-plotting"></a>

#### Methods
This code creates heatmaps from the results of the `gwobserve` method (stored as csv or parquet files) which shows the ideal exposures observation times for different instruments, sensitivities, and subsets of GW events.

#### Inputs

- An output file containing many observations from `gwobserve` in csv or parquet format
- Optional: how you would like to filter the data before creating the plots

#### Output

- heatmaps of the results (either interactive or exported directly as an image)

#### Example

```python
# import the data
from gravitational_wave_toy import gwplot
gws = gwplot.GWData("/path/to/data/file.parquet")  # or CSV

gws.df  # view the underlying dask dataframe

gws.set_filters(
   ("config", "==", "alpha"),
   ("site", "==", "south"),
   ("ebl", "==", True),
)

# optionally set a list of observation/exposure times to use on the x-axis of the heatmap
# default is 50 log-spaced bins between 10s and 1hr

# gws.set_observation_times([10, 100, 1000, 3600])

ax = gws.plot(
   output_file="heatmap.png",
   title="CTA South, alpha configuration, with EBL",
   min_value=0,
   max_value=1,
)
```

Other important options to `gws.plot` include:

- `intput_file` (str): The path to the output file.
- `annotate` (bool): Whether or not to annotate the heatmap.
- `x_tick_labels` (list): The labels for the x-axis ticks.
- `y_tick_labels` (list): The labels for the y-axis ticks.
- `min_value` (float): The minimum value for the color scale.
- `max_value` (float): The maximum value for the color scale.
- `color_scheme` (str): The name of the color scheme to use for the heatmap.
- `color_scale` (str): The type of color scale to use for the heatmap.
- `as_percent` (bool): Whether or not to display the results as percentages.
- `filetype` (str): The type of file to save the plot as.
- `title` (str): The title for the plot.
- `subtitle` (str): The subtitle for the plot.
- `n_labels` (int): The number of labels to display on the axes.
- `show_only` (bool): Whether or not to show the plot instead of saving it.
- `return_ax` (bool): Whether or not to return the axis object.


---
Logo credit: [smalllikeart](https://www.flaticon.com/authors/smalllikeart)

