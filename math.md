### Determining the visitibility of a simulated GRB After a certain oberservational period

The goal is to determine the average flux after time $t$ and compare it to the sensitivity of the instrument. The flux and spectral index at any point in time are interpolated from the simulated GRB.

$\text{observed:bool}(t) = F_{\text{avg}}(t) \gt \text{sensitivity}_\text{CTA}(t) $

1. Average flux

$F_{\text{avg}}(t) = \frac{S(t)}{t_{\text{obs}}} = \frac{S(t)}{t - t_{\text{start}} } $
   - where $S(t)$ is the fluence
   -  $t_{\text{obs}}$ is the total observing time starting from $t_{\text{start}}$

2. The fluence is calculated like so,

$S(t) = \int_{t_{\text{start}}}^t F(t) dt $

3. where the integral flux of the power law is

$F(t) = \int_{E_{\text{min}}}^{E_{\text{max}}} F_0(t) \left(\frac{E}{E_0} \right)^{\gamma(t)} dE = F_0(t) \frac{E_0^{-\gamma(t)}}{\gamma(t) + 1} \left( E_{\text{max}}^{\gamma(t) + 1} - E_{\text{min}}^{\gamma(t) + 1} \right)  $
   - where, $F_0(t)$ is the flux in the lowest energy bin of the GRB,
   - $E_0$ is the energy of the lowest energy bin,
   - $\gamma(t)$ is the spectral index at time $t$, calculated by fitting, for each time bin, the simulated spectra to a power law

