# Version 2 of grbsens output for CTA
- produced with grbsens package v0.2
    - Using master branch after [commit #955b13b](https://github.com/astrojarred/grbsens/commit/955b13bce5cd905972e19ee190cb71d356d7a2c7)
- produced on 25 May 2021
- produced by Jarred Green (jarred.green@inaf.it)

### Special Notes

- The only difference between v1 and this version, is that the spectral index of the power law model was changed to -2.2 (from -2.1)
- The 4s duration bin was replaced with 5s. This is due to large unexplained fluctuations in sensitivity caused by ctools.

## Simulation Parameters
- Sensitivity type:
    - Integral
- Spectra:
    - Power law with $\gamma = -2.2$
    - **This is the only difference between v1 and v2**
- 6 IRFs from *prod3b_v2*:
    - North_z20_0.5h
    - North_z40_0.5h
    - North_z60_0.5h
    - South_z20_0.5h
    - South_z40_0.5h
    - South_z60_0.5h
- Energy Limits (different for each IRF zenith angle). The minima were selected to avoid cutoff effects for each specific IRF (see previous email from Iftach from 30 January 2020, text below). The maximum of 10TeV was selected to match the maximum energy of L. Nava's GRB catalogues.
    - z20:  30**GeV** -> 10**TeV**
    - z40:  40**GeV** -> 10**TeV**
    - z60: 110**GeV** -> 10**TeV**
- Energy bins:
    - 1 bin spanning entire energy range
- Time bins to simulate:
    - 15 time bins from 0**s** -> 16384**s**
    - in powers of 2 ($2^0$ -> $2^{14}$)
- Significance ("`sigma`"):
    - $5\sigma$ (post-trial)
- Radius ("`rad`"):
    - $3^{\circ}$
- Offset:
    - 1.0º
- Pixel size for binned analysis ("`binsz`"):
    - 0.2
