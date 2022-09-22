### Determining the visitibility of a simulated GRB After a certain oberservational period

The goal is to determine the average flux after time $t$ and compare it to the sensitivity of the instrument.

$\text{observed:bool}(t) = F_{\text{avg}}(t) \gt \text{sensitivity}_\text{CTA}(t) $

1. Average flux

$F_{\text{avg}}(t) = \frac{S(t)}{t_{\text{obs}}} = \frac{S(t)}{t - t_0 } $
   - where $S(t)$ is the fluence and $t_{\text{obs}}$ is the total observing time.


2. The fluence is calculated like so,

$S(t) = \int_{t_0}^t F_0(t) S_{\text{int}}(t) dt $
   - where $F_0(t)$ is the flux interpolated from the simulated catalog GRB in the lowest energy bin,
   - and where $S_{\text{int}}(t)$ is the integral flux of a power law spectrum
   - this is solved numerically

3. where the integral spectrum of the power law is

$S_{\text{int}}(t) = \int_{E_{\text{min}}}^{E_{\text{max}}} \left(\frac{E}{E_0} \right)^{\gamma(t)} dE = \frac{E_0^{-\gamma(t)}}{\gamma(t) + 1} \left( E_{\text{max}}^{\gamma(t) + 1} - E_{\text{min}}^{\gamma(t) + 1} \right)  $
   - where the spectral index $\gamma(t)$ is interpolated from the spectral index evolution of the model
   - analytical solution used directly in the fluence equation

