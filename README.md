# Gravitational Wave Toy

Author(s): 
- B. Patricelli (barbara.patricelli at pi.infn.it),
- J. Green (jgreen at mpp.mpg.de),
- A. Stamerra (antonio.stamerra at inaf.it)

The purpose of this is to simualte the possibility of detecting very-high-energy electromagnetic counterparts to gravitational wave events events.
      
## Requirements
Set up a python environment using `conda` or `poetry` which includes the following packages:
- python (3.8+)
- numpy
- pandas
- scipy
- astropy
- matplotlib

## Instructions
### GW Observations
**Methods:**
This code simulates observations of simulated gravitational wave events to determine under which circumstances the event could be detectable by a gamma-ray observatory. An input GRB model and an instrument sensitivity curve specific to the desired observational conditions are provided as input. Given a delay from the onset of the event, the code uses a simple optimization algorithm to determine at which point in time (if at all) the source would be detectable by an instrument with the given sensitivity.


**Note:** Details regarding the mathematics are provided in `math.md`

**Inputs:**
   - An instrument sensitivity file from `grbsens`
     - Newest version is v3 and can be found [here](CTA_sensitivity/grbsens_output_v3_Sep_2022/)
   - GW event models (currently compatible with O5 simulations)

**Output**
   - a dictionary object containing detection information and parameters of the event itself (extracted from the model)

**Example:**
```python
from gravitational_wave_toy import gwobserve

# import the sensitivity
# interpolation of the sensitivity curve will be done here
sens = gwobserve.Sensitivity(
    "/path/to/sensitivity/file/grbsens-5.0sigma_t1s-t16384s_irf.txt",
    min_energy=0.3,     # generally determined by the energy range of the IRFs
    max_energy=10000,
)

# import a GRB file
# interpolation of the spectra are performed here
grb = GW.GRB("../GammaCatalog_O5/cat05_runID.fits")

# simulate observations
res = grb.observe(
    sensitivity=sens, 
    start_time=600,     # starting delay in seconds
    target_precision=0.1  # numerical precision 
    )

# output
print(res)
> {
>   'filepath': '../GammaCatalog_O5/catO5_100.fits',
>   'min_energy': 0.3,
>   'max_energy': 10000,
>   'seen': True,
>   'obs_time': 15,
>   'start_time': 3600,
>   'end_time': 3615,
>   'error_message': '',
>   'long': 6.1,
>   'lat': -10.5,
>   'eiso': 4.37e+48,
>   'dist': 124000.0,
>   'angle': 34.412
> }



```

### GW Toy: Create the detectability matrix
**Note:** This is not working on the main branch at the moment
1. Edit all parameters in `gw_settings.yaml`
   - You will need the output sensitivities of `grbsens`
2. Run `python GWToy.py`
3. The output will be saved both as a csv and as a pandas pickle

### GWPlotter: Create your heatmaps
**Note:** This is not working on the main branch at the moment
1. Edit the parameters in `plot_settings.yaml`
2. Run `python GWPlotter.py`
3. The plots will be saved as `png` files in the directory specified in the settings.
