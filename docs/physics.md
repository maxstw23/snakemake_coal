# Physics Background: Coalescence Test via Pion Elliptic Flow

## Overview

This analysis tests the **quark coalescence model** in heavy-ion collisions by measuring whether the isospin asymmetry of Au nuclei is preserved in the elliptic flow ($v_2$) of final-state pions. It uses BES-II Au+Au data at $\sqrt{s_{NN}} = 7.7$--$27$ GeV collected by the STAR detector at RHIC.

## Core Idea

In the coalescence picture, a hadron's collective flow is the sum of its constituent quarks' flows:

$$v_2^{hadron}(p_T) \approx \sum_{i=1}^{n_q} v_2^{q_i}(p_T / n_q)$$

Quarks fall into two categories:
- **Transported quarks** ($q_{tr}$): originating from the initial Au nuclei, present from the collision onset, acquire stronger flow.
- **Produced quarks** ($q_{pr}$): pair-produced from vacuum later in the evolution, acquire weaker flow.

## The Coalescence Ratio

Gold is neutron-rich ($N/Z \approx 1.5$), so there are more transported $d$ quarks than $u$ quarks. Since $\pi^-(d\bar{u})$ draws from the larger $d$ reservoir and $\pi^+(u\bar{d})$ from the smaller $u$ reservoir, the $\pi^-$ should carry slightly more flow.

Using the antiproton ($\bar{p} = \bar{u}\bar{u}\bar{d}$, all produced quarks) as a baseline for produced-quark flow:

$$\frac{v_2(\pi^-) - \frac{2}{3}v_2(\bar{p})}{v_2(\pi^+) - \frac{2}{3}v_2(\bar{p})} = \frac{N_d^{tr}}{N_u^{tr}}$$

The prediction from Au stoichiometry is:

$$\frac{N_d^{tr}}{N_u^{tr}} = \frac{2N + Z}{N + 2Z} = \frac{315}{276} \approx 1.14$$

A measurement consistent with 1.14 validates that coalescence preserves initial-state isospin information.

## Centrality Dependence

The effect is expected to vary with centrality due to:
- **Neutron skin**: Neutrons are distributed at larger radii than protons in Au, so the overlap region in peripheral collisions is more neutron-rich, enhancing the $d/u$ asymmetry.
- **Nuclear deformation**: Orientation of deformed nuclei changes the local $d/u$ ratio.

## Key Experimental Details

| Parameter | Value |
|-----------|-------|
| Collision system | Au+Au |
| Energies | 7.7, 9.2, 11.5, 14.6, 17.3, 19.6, 27 GeV |
| Particle species | $\pi^\pm$, $K^\pm$, $p/\bar{p}$ |
| Rapidity cut | $\|\eta\| < 0.6$ |
| Event planes | TPC (2nd harmonic, all energies), EPD (2nd harmonic, $\geq 14.6$ GeV) |
| Centrality bins | 9 bins (0-80%) |
| pT/nq integration range | 0.08--0.6 GeV/c |

## Systematic Uncertainty Strategy

Two "regular" variations (sys_tag_1, sys_tag_2) plus optional "special" variations are run through the full analysis chain:

| # | Source | Default | Variation |
|---|--------|---------|-----------|
| 1 | $\|V_z\|$ | < 145 (70) cm | < 70 (35) cm |
| 2 | $N_{fit}$ | >= 15 | >= 20 |

Significance is assessed via the Barlow test. Total systematic = quadrature sum of significant deviations / $\sqrt{3}$.
