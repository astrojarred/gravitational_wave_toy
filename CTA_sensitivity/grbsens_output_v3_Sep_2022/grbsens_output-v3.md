# Version 3 of grbsens output for CTA

- produced with grbsens package v0.3
  - Using branch `add-EBL` after [commit #ce231e45](https://github.com/astrojarred/grbsens/commit/ce231e454bdc028e6a02e72564707e8ee26875aa)
- produced September 2022
- produced by Jarred Green (jgreen@mpp.mpg.de)

### Special Notes

- The main difference between v2 and this version is that we now add various array configurations, starting with alpha, and later omega
- 

## Simulation Parameters

- Sensitivity type:
  - Integral
- Spectra:
  - Power law with $\gamma = -2.1$
- 6 IRFs from _prod5-v0.1_ (alpha config):
  - North_z20_0.5h
  - North_z40_0.5h
  - North_z60_0.5h
  - South_z20_0.5h
  - South_z40_0.5h
  - South_z60_0.5h
- More configs to come
- With and without EBL:
  - Franceschini 2008
- Energy Limits (different for each IRF zenith angle). The minima were selected to avoid cutoff effects for each specific IRF (see previous email from Iftach from 30 January 2020, text below). The maximum of 10TeV was selected to match the maximum energy of L. Nava's GRB catalogues.
  - z20: 30**GeV** -> 10**TeV**
  - z40: 40**GeV** -> 10**TeV**
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
