# Sensitivity Calculation with gammapy

Integral and differential sensitivities are now calculated in this package using gammapy's `SensitivityEstimator` function. Details on the function can be found in the documentation [here](https://docs.gammapy.org/1.2/api/gammapy.estimators.SensitivityEstimator.html#gammapy.estimators.SensitivityEstimator). In addition, a brief tutorial on how to use this function (and how it's used in this package) can be found [here](https://docs.gammapy.org/1.2/tutorials/analysis-1d/cta_sensitivity.html).

## Sensitivity Estimator

This document outlines what exactly is happening when we run the sensitivity calculation in this package. The `SensitivityEstimator` function is used to calculate the sensitivity of a given instrument to a given source. The function takes in the following parameters (defaults used in this package are in brackets):

- `spectrum`: the spectral model of the source
- `n_sigma [5]`: the significance level at which to calculate the sensitivity
- `gamma_min [5]`: minimum number of gamma-rays required to detect the source in each energy bin
- `bkg_syst_fraction [0.01]`: Fraction of background counts above which the number of gamma-rays must be in each bin

The sensitivity estimator performs the following calculation:

$$ E^2 \frac{dN}{dE} = \frac{n_\text{excess}}{n_\text{predicted}} S(E) E^2 $$

If one cancels the $E^2$ term on both sides, one can see that sensitivity is calculated by scaling the model at a given energy by the ratio of the number of excess counts to the number of predicted counts. Note that is this a vectorized calculation, so the sensitivity is calculated at each energy bin.

Now we will walk through how each of the terms in this equation are calculated.

### Spectral Model

The spectral model of the source is passed in as a parameter to the sensitivity estimator. When evaluated by the sensitivity estimator, units of $\frac{1}{\text{GeV} \cdot \text{cm}^2 \cdot \text{s}}$ are returned.

### Excess Counts

The number of excess counts is calculated using the WStat statistic. More information on how this is [derived](https://docs.gammapy.org/1.2/user-guide/stats/wstat_derivation.html?highlight=wstat) and implemented in gammapy can be found [here](https://docs.gammapy.org/1.2/user-guide/stats/index.html?highlight=wstat#wstat-counts-statistic).

For our purposes, the number of excess counts takes the following inputs:

$$ n_\text{excess}(\sigma_0, n_\text{off}, \alpha) $$

and uses the WStat statistic to find the precise number of excess counts that lead to a detection at the given significance level. This is performed by numerically solving the following equation:

$$ \sigma_0^2 = W(\sigma=0, n_\text{off}, n_\text{on}, \alpha ) - W(\sigma=\sigma_0, n_\text{off}, n_\text{on}, \alpha) $$

where,

- $W(\sigma)$ is the WStat statistic evaluated at a given significance level
- $n_\text{off}$ is the number of background counts estimated from our spectrum
- $n_\text{on}$ is the number of on counts, calculated by defaults as $n_\text{on} = n_\text{off} \cdot \alpha$
- $\alpha$ is the ratio of the on region to the off region, by default set to 0.2

In this package, we're interested in the case where $\sigma_0 = 5$ by default.  In code, this is implemented as:

```python
from gammapy.stats import WStatCountsStatistic

dataset = # see tutorial, contains off counts calculated bin-wise from the input spectrum

n_off = dataset.counts_off.data

# number of on counts in each bin
# alpha is the exposure ratio between on and off regions
# here default is 0.2 in every energy bin

n_on = dataset.alpha.data * n_off

# class for calculating Poisson distributed variables with unknown background
# n_on: int: Measured counts in on region.
# n_off: int: Measured counts in off region.
# alpha: float:  Acceptance ratio of on and off measurements.
# mu_sig: float: Expected signal counts in on region. default is 0
# note: n_bkg = alpha + n_off

stat = WStatCountsStatistic(
    n_on=n_on, n_off=n_off, alpha=dataset.alpha.data
)

# compute excess matching a given significance level
# does this by finding roots of a function:
    # stat0 is the statistic for the null hypothesis:        wstat(n_on=n_sig + n_bkg, n_off=n_off, alpha=alpha, mu_sig=0)
    # stat1 is the statistic for the alternative hypothesis: wstat(n_on=n_sig + n_bkg, n_off=n_off, alpha=alpha, mu_sig=n_sig)
    # sign(n_sig) * sqrt(stat0 - stat1) - significance = 0
    # this is basically the equivalent of stat0 = stat1 + significance^2
# excess counts is the number of photons necessary in each bin to reach the significance level

excess_counts = stat.n_sig_matching_significance(n_sigma)
is_gamma_limited = excess_counts < gamma_min
excess_counts[is_gamma_limited] = gamma_min
bkg_syst_limited = (
    excess_counts < bkg_syst_fraction * dataset.background.data
)
excess_counts[bkg_syst_limited] = (
    bkg_syst_fraction * dataset.background.data[bkg_syst_limited]
)
excess = Map.from_geom(geom=dataset._geom, data=excess_counts)

return excess
```

### Predicted Counts

The number of predicted counts is calculated by convolving the spectral model of the source with the IRFs. As an outline of how this is calculated in gammapy:

$$
\begin{align*}
n_\text{pred} [\text{counts}] & =  \text{total flux in energy bin}[ \frac{1}{\text{cm}^2 \cdot \text{s}}] \\
& \times \text{IRF exposure in energy bin}[\text{cm}^2 \cdot \text{s}] \\
& \times \text{IRF energy dispersion}[\text{unitless}]
\end{align*}
$$

or: 

$$ n_\text{pred}(S(E),  t_\text{livetime}, \text{IRF}) = \int_{E_{\text{min}}}^{E_{\text{max}}} S(E) \cdot [ A_\text{eff} * t_\text{livetime} ] \cdot \text{edisp}$$

In code this is calculated automatically in gammapy as follows:

```python
npred = dataset.npred_signal()
```

since the dataset already contains information about the given spectrum, its binning, the IRFs used for the background model, and the observation time, this calculation is relatively straightforward.

## Bringing it all together

Let's repeat the first equation:

$$ E^2 \frac{dN}{dE} = \frac{n_\text{excess}}{n_\text{predicted}} S(E) E^2 $$

or including the :

$$ E^2 \frac{dN}{dE} = \frac{n_\text{excess}(\sigma_0, S(E), n_\text{off}(IRF, t_\text{livetime}), \alpha)}{n_\text{pred}(S(E),  t_\text{livetime}, \text{IRF})} \cdot S(E) \cdot E^2$$