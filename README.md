# Gravitational Wave Toy

Author(s): 
- B. Patricelli (barbara.patricelli at pi.infn.it),
- J. Green (jarred.green at inaf.it),
- A. Stamerra (antonio.stamerra at inaf.it)

The purpose of this is to simualte the possibility of detecting VHE EM counterparts to GW wave events.
      
## Requirements
Set up a python environment using `conda` or `poetry` which includes the following packages:
- python v**3.7.9**
- scipy
- numpy
- matplotlib
- seaborn
- astropy
- pandas
- ray
- tqdm

## Instructions
### GW Toy: Create the detectability matrix
**Note:** This version of the code works with v1 of the BNS merger catalogue
1. Edit all parameters in `gw_settings.yaml`
   - You will need the output sensitivities of `grbsens`
2. Run `python GWToy.py`
3. The output will be saved both as a csv and as a pandas pickle

### GWPlotter: Create your heatmaps
1. Edit the parameters in `plot_settings.yaml`
2. Run `python GWPlotter.py`
3. The plots will be saved as `png` files in the directory specified in the settings.
