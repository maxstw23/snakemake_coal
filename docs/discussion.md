# Discussion: Resonance Decay, B/A Ratio, and Validation

This document summarizes discussions on the interplay between resonance feeddown, the two ratio methods, and the validation strategy for the coalescence test.

## Two ratio methods

### Simple ratio (default)

$$R = \frac{v_2^{\pi^-} - \frac{2}{3}v_2^{\bar{p}}}{v_2^{\pi^+} - \frac{2}{3}v_2^{\bar{p}}}$$

Uses integrated $v_2$ over $p_T/n_q \in [0.08, 0.6]$ GeV/c. The antiproton baseline $\frac{2}{3}v_2^{\bar{p}}$ appears identically in numerator and denominator.

### B/A ratio (alternative)

$$\frac{B}{A} = \frac{\Delta v_2(p) + \frac{2}{1-f}\Delta v_2(\pi)}{\Delta v_2(p) - \frac{1}{1-f}\Delta v_2(\pi)}$$

where $\Delta v_2(\pi) = v_2^{\pi^-} - v_2^{\pi^+}$ and $\Delta v_2(p) = v_2^p - v_2^{\bar{p}}$. At $f = 0$, this reduces to:

$$\frac{\Delta v_2(p) + 2\Delta v_2(\pi)}{\Delta v_2(p) - \Delta v_2(\pi)}$$

Both methods are implemented in the pipeline: the simple ratio in YAML files via `plot_v2_new.py`, and the B/A ratio in `plot_alternative_ratio()` (pT-differential) and `plot_alternative_ratio_integrated()` (integrated v2, matching the simple ratio's inputs).

## Observed values

- **Simple ratio**: ~1.14, consistent with the Glauber $N_d^{tr}/N_u^{tr} = 315/276$ prediction.
- **B/A ratio**: ~1.4, significantly above the Glauber prediction.
- **Integrated B/A ratio**: consistent with the pT-differential version (similar central values, slightly smaller errorbars due to wider pT range).

## Effect of resonance fraction $f > 0$ on each method

### Simple ratio

When $f > 0$, the measured pion $v_2$ picks up a resonance decay component $f \cdot v_{2,s}$ that is common to $\pi^+$ and $\pi^-$. This adds a common positive quantity to both numerator and denominator of $R$. For a ratio > 1, adding the same value to both num and denom pushes it toward 1:

$$R(f > 0) < R(f = 0)$$

Since $R(f=0) \approx 1.14$ already matches the Glauber band, turning on $f > 0$ would push $R$ *below* the prediction. This implies either $f$ is small, or there is a compensating effect (see below).

### B/A ratio

With $f > 0$, the proton $v_2$ also receives resonance feeddown (e.g., $\Delta(1232) \to N\pi$, $N^* \to N\pi$). At BES energies, baryon transport creates a large proton/antiproton asymmetry in resonance yields, so $\Delta v_2(p)$ is inflated beyond pure coalescence. The B/A ratio, which isolates $\Delta v_2(p)$ directly, is sensitive to this contamination. Including $f > 0$ in the formula makes the ratio even larger, worsening the discrepancy.

## Why the two methods disagree

The simple ratio and B/A ratio are algebraically equivalent only under pure coalescence with $f = 0$. Their disagreement reveals that non-coalescence contributions to the proton sector (mean-field potential, hadronic rescattering, baryon junction) are significant at BES energies.

The simple ratio is less sensitive to these effects because $\frac{2}{3}v_2^{\bar{p}}$ appears identically in both numerator and denominator, providing partial cancellation of contamination. The B/A ratio instead depends on $\Delta v_2(p) = v_2^p - v_2^{\bar{p}}$, which is directly sensitive to the proton/antiproton asymmetry where non-coalescence effects are largest.

The discrepancy itself is informative: it is evidence that the proton sector carries large non-coalescence contributions, consistent with expectations at BES energies.

## Resonance feeddown to pions

### Magnitude of $f$

Thermal model and transport code estimates (UrQMD, AMPT) suggest that 50--70% of measured pions at BES energies originate from resonance decay, primarily $\rho \to \pi\pi$. So $f \sim 0.5$--$0.7$ is realistic.

### Why a large $f$ does not destroy the signal

Despite a large fraction of pions coming from resonance decay, significant $\pi^+/\pi^-$ $v_2$ splitting is observed. This is because:

1. **Charged rho channels preserve isospin**: While $\rho^0 \to \pi^+\pi^-$ is isospin-symmetric and dilutes the signal, $\rho^+ \to \pi^+\pi^0$ and $\rho^- \to \pi^-\pi^0$ are not symmetric. At finite $\mu_B$, the yields of $\rho^+$ and $\rho^-$ differ (reflecting the underlying $u/d$ asymmetry), so the charged $\rho$ channels propagate the coalescence signal through the resonance decay.

2. **Not all resonance decay is dilutive**: Only the flavor-symmetric channels ($\rho^0$, $\omega$, etc.) wash out the isospin information. The fraction of truly dilutive decay may be smaller than the total $f$.

3. **Consistency check**: The fact that the simple ratio agrees with the Glauber prediction at $f = 0$ suggests that the *effective* dilution is small — either because charged rho channels compensate, or because the direct pion splitting is somewhat larger than naive coalescence and the dilution brings it to the right value.

An open question is the quantitative breakdown of $\rho^0$ vs $\rho^\pm$ contributions to pion feeddown at BES energies.

## Compatibility with proton $v_1$ analysis

The claim that $\Delta v_2(p)$ is contaminated by non-coalescence effects does not contradict using proton-antiproton $\Delta v_1$ slope as an EM field probe:

- **Different observables, different sensitivities**: Mean-field potentials primarily affect in-plane/out-of-plane dynamics ($v_2$), while the EM effect on $v_1$ is a rapidity-odd directed flow signal.
- **v1 slope vs v2**: The slope $dv_1/dy$ has different sensitivity to baryon transport than $v_2$.
- **Internal cross-check**: Comparing proton $v_1$ slope against $\Lambda$ $v_1$ slope (different quark content, different resonance feeddown) provides its own consistency test.

The resolution is that proton flow is not *generally* unreliable; rather, the B/A ratio formula specifically amplifies non-coalescence contamination by isolating $\Delta v_2(p)$ in a way that prevents cancellation. The simple coalescence ratio and the $v_1$ analysis use proton flow in different combinations that are less sensitive to the same contamination.

## Validation strategy for the simple ratio

1. **TPC vs EPD consistency**: Two independent event planes (TPC 2nd-order, EPD 2nd-order participant plane) give consistent results, ruling out detector-specific biases. (Note: 1st-order EPD spectator plane was less consistent.)

2. **Efficiency insensitivity**: $v_2$ values are stable with and without efficiency correction.

3. **$f \approx 0$ consistency**: The measured ratio already sits on the Glauber band at $f = 0$, so the resonance feeddown correction is small or self-compensating.

4. **B/A discrepancy as positive evidence**: The disagreement with the B/A ratio is not a contradiction but is consistent with known non-coalescence effects in the proton sector that the simple ratio avoids by construction.

5. **Centrality dependence**: The ratio follows the expected trend from Glauber, though errorbars are too large to resolve the detailed shape (e.g., neutron skin enhancement in peripheral collisions).

6. **Energy dependence gap**: No data between 27 and 200 GeV to demonstrate the smooth trend toward $R \to 1$ as $\mu_B \to 0$. At 200 GeV, $\Delta v_2(\pi) \approx 0$ because the net quark density at midrapidity vanishes, so the coalescence signal disappears. The isobar ratio test at 200 GeV is statistics-limited (ratio of small differences).

---

## Science/Nature Publication Strategy

*Discussed 2026-03-27. Notes on narrative framing and what needs to land.*

### Core message

The result bridges two physics regimes: a static, ground-state nuclear property ($N_d^{tr}/N_u^{tr}$ set by nuclear stoichiometry) and a macroscopic collective observable (elliptic flow) from the most violent collisions humans create. The headline claim is that **isospin information survives QGP formation** — surviving thermalization, collective expansion, hadronization, resonance decays, and kinetic freeze-out — and is readable in the final-state pion flow. This is a statement about information survival in a strongly-coupled quantum system, which has appeal beyond heavy-ion physics.

One-sentence pitch: *We show that the quark flavor composition of atomic nuclei — a ground-state property — is preserved through quark-gluon plasma formation and can be read out from the collective flow of final-state pions, providing a new connection between nuclear structure and QCD matter under extreme conditions.*

### Narrative angles (ranked by impact potential)

1. **"Memory of the initial state survives the QGP."** The QGP is among the most thermalized, strongly-coupled systems in nature. The fact that it does not erase the Au isospin asymmetry is the conceptual surprise. This framing resonates with broader questions about thermalization and information in many-body systems.

2. **Coalescence as quark-level nuclear tomography.** The method reads out $N_d/N_u$ without DIS or electron scattering — just pion flow in heavy-ion collisions. Methodological novelty: using collective flow as a flavor-resolving probe of nuclear structure.

3. **Neutron skin via flow observables.** If a clear centrality trend emerges (peripheral collisions more neutron-rich due to neutron skin geometry), the result would connect to the hot topic of neutron skins (PREX-II made Nature Physics). A centrality trend consistent with a Skyrme-HF density profile would be a strong addition. *Currently no clear centrality trend in data — this angle is not yet available.*

4. **Energy dependence as a baryon transport map.** The BES-II energy scan gives the ratio at 7 energies. The signal should strengthen at lower $\sqrt{s}$ (more transported quarks, higher $\mu_B$) and vanish at high energy ($\mu_B \to 0$, no transported quarks). This energy evolution directly maps baryon stopping — an open QCD question.

### What would strengthen the case for a top journal

1. **Neutron skin story**: needs a clear centrality trend. Currently absent. If statistics improve or if centrality-differential systematics can be controlled, comparing to Glauber+neutron-skin models (e.g., SAMi-J, NL3* parametrizations) would be the single biggest upgrade.

2. **Isobar cross-check (Ru/Zr)**: the predictions $\Delta v_2^{\pi^-}/\Delta v_2^{\pi^+} = -1$ and $(v_2^{\pi^-} - v_2^{\pi^+})_{Ru}/(v_2^{\pi^-} - v_2^{\pi^+})_{Zr} = 1/2$ are clean and resonance-free. Even a consistency check (not a discovery) across systems elevates the paper from "one measurement at 1.14" to "a validated predictive framework across multiple nuclear species."

3. **Resonance fraction correction (B/A method)**: if the corrected result still lands on 1.14 with a reliable estimate of $f$, the robustness argument is much stronger. Currently the B/A ratio gives ~1.4 and the two methods' disagreement is attributed to non-coalescence proton effects — this needs to be clearly addressed in the paper.

4. **One-figure summary**: a single figure showing $R$ vs centrality (or energy), with the Glauber 1.14 prediction and model band, should tell the whole story immediately to a non-specialist.

### Current status

- Simple ratio: ~1.14, consistent with Glauber prediction.
- No clear centrality trend currently visible in the data (errorbars too large to resolve neutron skin shape).
- Energy dependence trend (increasing $R$ toward lower $\sqrt{s}$) is the more accessible quantitative statement.
- The neutron skin angle is the highest-impact but requires better statistical control of the centrality dependence.
