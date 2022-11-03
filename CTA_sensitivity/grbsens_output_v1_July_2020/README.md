# Version 1 of grbsens output for CTA
- produced with grbsens package v0.1
    - Using master branch after [commit #955b13b](https://github.com/astrojarred/grbsens/commit/955b13bce5cd905972e19ee190cb71d356d7a2c7)
- produced on 6 July 2020
- produced by Jarred Green (jarred.green@inaf.it)

### Special Notes

- For only the `North_z60_0.5h` IRF, at durations around 8s-16s, there were some strong fluctuations in the sensitivity. For this reason, we replaced the 8s with 10s and 16s with 18s in this one csv file. These issues will be investigated with further iterations of these outputs.

## Simulation Parameters
- Sensitivity type:
    - Integral
- Spectra:
    - Power law with $\gamma = -2.1$
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
- Pixel size for binned analysis ("`binsz`"):
    - 0.2



## Note about Energy ranges

Directly from an email by Iftach, 30 January 2020

**Energy threshold:**

- we should generate events from the minimum allowed value for a given IRF, but then we should only use for the analysis events above a slightly higher threshold, in order to avoid cutoff effects. I’ll be happy to explain why this is important.

- The IRFs are defined with these minimum energy thresholds:
  - 20deg → E_gen ≥ 12.6 GeV
  - 40deg → E_gen ≥ 26.4 GeV
  - 60deg → E_gen ≥ 105.3 GeV

- My suggestion: we should generate events from these E_gen energies, but the analysis (ie low edge for potential discovered source) should be:

  - 20deg → E_grb ≥ 30 GeV

  - 40deg → E_grb ≥ 40 GeV

  - 60deg → E_grb ≥ 110 GeV

- This corresponds roughly to [E_gen + deltaE], which is the expected energy resolution at low energies (deltaE/E (68%) = 0.25)

- Note also that the CTA performance plots have a cutoff of E>20 GeV since we can’t trust the sims below this range. In terms of requirements, we actually only have a guarantee above 30 GeV, again, due to the large uncertainties at these energies. (Once we have real data, I’m sure we can reduce the energy threshold, but this is not part of the formal requirements.)
- We can therefore not use <30GeV energies for spectra/detection in any case for a consortium publication, unless we go through dedicated performance studies, which are not the focus of the GRB paper…