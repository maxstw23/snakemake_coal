# Pion $v_2$ Shape Correction — Experiments (2026-05-01)

## Problem

Bayesian NCQ fit (`quark_v2_bayes.py`) simultaneously fits $\pi^\pm$, $K^\pm$, $p$, $\bar{p}$ $v_2(p_T)$ to 18 shared quark sigmoid parameters. K, p, $\bar{p}$ fit well ($\chi^2$/ndf $\sim$ 2–7) but pions show severe shape mismatch ($\chi^2$/ndf $\sim$ 97–152). The model over-predicts at low $p_T$, under-predicts at mid $p_T$, fits at high $p_T$ — an S-curve residual.

**Root cause**: cross-species tension. Quark sigmoids $v_2^{\rm prod}$ and $v_2^{\rm tr}$ must serve pions at $p_T/2$ and baryons at $p_T/3$ simultaneously. The data demands different shapes at different scalings.

## Attempts

### 1. $\rho \to \pi\pi$ feed-down (Taylor expansion, 1st order)

Added $f_\rho \cdot [1 - D_{\rm eff}(p_T)]$ multiplicative factor + $f_\rho \cdot \Delta p_{T,\rm eff} \cdot v_2'$ derivative term. Replaced $\varepsilon_0^\pi$ with $f_\rho$ (net 0 extra params).

**Result**: Failed. $f_\rho \to 0$ regardless of prior. At low $p_T$ where correction is needed, $D_{\rm eff} \approx 0$ and $\Delta p_{T,\rm eff} \cdot v_2' \sim 10^{-4}$ — three orders of magnitude too small.

### 2. Narrower prior $\text{Beta}(7,35)$

Same Taylor expansion, narrower prior peaking at $f_\rho = 0.15$.

**Result**: Failed. $f_\rho \to 0.003$, parameters went haywire ($\bar{f} = 0.08$, $\delta_f = 2.6$, $\pi^+$ $\chi^2$/ndf = 1935).

### 3. Taylor $f_\rho$ + $\varepsilon_0^\pi$ together

Kept both $\varepsilon_0^\pi$ (flat shift) and $f_\rho$ (shape correction).

**Result**: No improvement. $f_\rho \to 0.007$, $\varepsilon_0^\pi$ absorbed everything, parameters identical to baseline.

### 4. Full transfer matrix $f_\rho$ + $\varepsilon_0^\pi$

Used full $T_{v_2}$ and $T_{\rm yield}$ matrices (no Taylor truncation): $v_2^{\rm obs} = (1-f_\rho)v_2^{\rm NCQ} + f_\rho v_2^{\rm decay} + \varepsilon_0^\pi$.

**Result**: No improvement. $f_\rho \to 0.011$. Same as attempt 3. $v_2^{\rm decay}$ and $v_2^{\rm NCQ}$ are too similar (built from same sigmoid parameters) — the model can't USE $f_\rho$ to change the pion prediction.

### 5. Full matrix with Uniform prior $\text{Beta}(1,1)$, no $\varepsilon_0^\pi$

Removed $\varepsilon_0^\pi$ entirely, used uniform prior on $f_\rho$.

**Result**: Failed. $f_\rho \to 0.000$, parameters went haywire ($\bar{f} = 0.08$, $\delta_f = 2.6$). The full matrix correction is fundamentally ineffective.

### 6. Free $n_q$ for pions

Allowed $n_q$ (number of constituent quarks) to float for pions only. NCQ scaling $p_T/n_q$ instead of $p_T/2$.

**Result**: $n_q \to 2.01$ — data strongly prefers $n_q = 2$. No improvement.

### 7. Phenomenological S-curve correction ✅

Replaced $\rho$ feed-down with a difference-of-two-exponentials:

$$\text{corr}(p_T) = A_{\rm fast} \cdot e^{-p_T/\tau_{\rm fast}} - B_{\rm slow} \cdot e^{-p_T/\tau_{\rm slow}}$$

4 new parameters: $A_{\rm fast}$, $\tau_{\rm fast}$, $B_{\rm slow}$, $\tau_{\rm slow}$ (all HalfNormal). Added to pion prediction alongside $\varepsilon_0^\pi$. Total: 22 params.

**Result**: **Dramatic success**.

| Species | Before | After | Improvement |
|---------|--------|-------|-------------|
| $\pi^+$ | 151.6 | **13.2** | 11.5× |
| $\pi^-$ | 96.8 | **8.8** | 11× |
| $K^+$ | 2.4 | **1.8** | 1.3× |
| $K^-$ | 3.1 | **2.4** | 1.3× |
| $p$ | 6.7 | **1.5** | 4.5× |
| $\bar{p}$ | 3.8 | **1.9** | 2× |

**Key insight**: The S-curve didn't just fix pions — by releasing cross-species tension, it improved ALL species. The quark sigmoids relaxed to their proper values.

**Fitted correction**:
$$\text{corr}(p_T) = 0.095 \cdot e^{-p_T/0.69} - 0.114 \cdot e^{-p_T/0.29}$$

At $p_T = 0$: −0.019 (suppression), crosses zero at ~0.3 GeV, broad positive lobe at mid $p_T$, decays to zero above ~1.5 GeV.

**Updated physics parameters**:
- $f_u = 0.263 \pm 0.054$, $f_d = 0.345 \pm 0.070$
- $f_d/f_u = 1.315 \pm 0.067$ (Au prediction: 1.141)
- $\bar{f} = 0.302 \pm 0.061$ (transported fraction)
- $\delta_f = 0.393 \pm 0.082$ (isospin asymmetry)
- $\varepsilon_0^\pi = -0.004 \pm 0.004$ (flat offset now consistent with zero)

## Active State of `quark_v2_bayes.py`

Currently configured with: S-curve correction (attempt 7). 22 parameters total.

The $\rho$ feed-down code (Taylor expansion, full matrix, `_load_transfer_cache`, `_compute_deff_dpteff`) is still in the file but dormant — not wired into `build_model()`.

## Next Steps (not yet tried)

- **Drop pions from inference**: Fit only K⁺, K⁻, p, $\bar{p}$. Use quark parameters to predict $\pi^\pm$ as a posterior predictive check. Quantifies pion-specific NCQ deviation as a physics result.
- **Try on other energies**: 7.7, 11.5, 14.6, 27 GeV. The S-curve correction may be energy-dependent.
- **Interpret the S-curve**: The fitted exponentials likely represent competing hadronic effects (rescattering vs. feed-down). A physical interpretation would strengthen the model.
