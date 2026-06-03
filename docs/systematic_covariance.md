# Cross-energy systematic covariance for the energy-dependence GOF

Companion to [stats_review.md](stats_review.md) **M16** and the χ²/ndf in
`generate_paper_plots.py::plot_energy_dep` (energy_dep.pdf). How to account for the
correlation of systematic uncertainties across beam energies when testing the 10–40%
ratio $R(\sqrt{s})$ against the prediction $c = 315/276$.

## The goodness-of-fit object

$$\chi^2 = \boldsymbol{\Delta}^{\mathsf T}\,V^{-1}\,\boldsymbol{\Delta},
\qquad \Delta_e = R_e - c, \qquad e = 1\ldots N_{\rm energy},$$

with ndf $= N$ (fixed reference, no free parameters — see M15). The current code uses only
the diagonal of $V$. Split the covariance into statistical and systematic parts:

$$V = V_{\rm stat} + V_{\rm sys}.$$

- **$V_{\rm stat}$ is diagonal**: different beam energies are independent datasets, so there is
  no cross-energy statistical correlation,
  $$(V_{\rm stat})_{ee'} = \sigma_{{\rm stat},e}^2\,\delta_{ee'}.$$
- **$V_{\rm sys}$ has off-diagonal terms**: the *same* analysis-cut variations (the `sys_tag`s)
  are applied at every energy, so a cut that biases the ratio coherently across energies
  induces correlation. Populating these off-diagonals is the whole problem.

## Three models for $V_{\rm sys}$

### (1) Diagonal (ρ = 0) — current plot
$$(V_{\rm sys})_{ee'} = \sigma_{{\rm sys},e}^2\,\delta_{ee'}.$$
Assumes each energy's systematic is independent. Wrong if the sources push coherently.

### (2) Fully correlated (ρ = 1)
$$(V_{\rm sys})_{ee'} = \sigma_{{\rm sys},e}\,\sigma_{{\rm sys},e'}.$$
Rank-1; assumes a single common nuisance scales all energies together, and uses only the
magnitude $\sigma_{{\rm sys},e} = \sqrt{\sum_s \delta_{s,e}^2}$ — the sign pattern is already
lost. Over-correlates.

### (3) Per-source decomposition — the correct one
Keep each systematic **source** $s$ separate. Source $s$ produces a **signed shift vector**
across energies, $\boldsymbol{\delta}_s = (\delta_{s,1},\ldots,\delta_{s,N})$. Treat each source
as a single nuisance parameter, **100 % correlated across all energies it affects** (it is the
same cut everywhere), with **different sources independent**:

$$\boxed{\,V_{\rm sys} = \sum_{\rm sources\ s} \mathbf{s}_s\,\mathbf{s}_s^{\mathsf T}\,},
\qquad
(\mathbf{s}_s)_e = \operatorname{sign}(\delta_{s,e})\,
\sqrt{\tfrac{1}{3}\,\max\!\big(\delta_{s,e}^2 - \delta_{{\rm err},e}^2,\ 0\big)}.$$

Each rank-1 term $\mathbf{s}_s\mathbf{s}_s^{\mathsf T}$ has

$$\text{diagonal } (\mathbf{s}_s)_e^2 = \tfrac{1}{3}\max(\delta_{s,e}^2-\delta_{{\rm err},e}^2,0),
\qquad
\text{off-diagonal } (\mathbf{s}_s)_e (\mathbf{s}_s)_{e'}.$$

- The diagonal is **identical** to what `combine_sys.py` already builds: Barlow subtraction
  $\delta^2-\delta_{\rm err}^2$ then the uniform $/3$ (the $(\sqrt3)^2$ of C2). So
  $\sum_s(\mathbf{s}_s)_e^2 = \sigma_{{\rm sys},e}^2$ — the per-source matrix reproduces the
  current diagonal exactly and only **adds** the off-diagonals.
- The off-diagonal sign is the discarded information: **positive** if source $s$ moves $e$ and
  $e'$ the same way, **negative** if opposite. This is exactly `delta_signed`
  (`combine_sys.py:36,60`), which is currently thrown away.

This is the PDG / HEPData standard ("one nuisance per source, fully correlated across the
points it affects"). It self-tunes: a source that flips sign across energies contributes
negative off-diagonals and behaves uncorrelated; a source that pushes coherently contributes
positive off-diagonals and behaves correlated. No hand-chosen $\rho$.

## Equivalent nuisance-parameter (profiled) form

$V_{\rm sys} = \sum_s \mathbf{s}_s\mathbf{s}_s^{\mathsf T}$ is the marginalisation of

$$\chi^2 = \min_{\{\alpha_s\}}\ \sum_e
\frac{\Big(\Delta_e - \sum_s \alpha_s\,(\mathbf{s}_s)_e\Big)^2}{\sigma_{{\rm stat},e}^2}
\;+\; \sum_s \alpha_s^2,$$

i.e. each source may shift the points along its signed pattern $(\mathbf{s}_s)_e$ by a pull
$\alpha_s$, paying a unit-Gaussian penalty $\alpha_s^2$. Interpretation:

- a **coherent offset** (all $(\mathbf{s}_s)_e$ same sign) can absorb a common bias →
  *lowers* $\chi^2$ if the data sit uniformly off $c$;
- it **cannot** absorb a trend or point-to-point scatter → those remain limited by
  $\sigma_{\rm stat}$.

This is why the bracket on the 10–40% TPC data is wide and counter-intuitive:
$$\chi^2/\mathrm{ndf}\big|_{\rho=0} = 3.10, \qquad \chi^2/\mathrm{ndf}\big|_{\rho=1} = 5.00.$$
The high-tension energies (7.7, 19.6 GeV) have tiny $\sigma_{\rm sys}$, so no nuisance can move
them, while the diagonal treatment lets the large-$\sigma_{\rm sys}$ energies
(9.2, 14.6, 17.3 GeV) self-down-weight. The per-source matrix gives the physically correct
answer in between, fixed by the real sign patterns rather than an assumed $\rho$.

## Source grouping when the variation differs by energy

The method does **not** require the cut value to be identical across energies — $(\mathbf{s}_s)_e$
already carries energy-dependent magnitude and sign. It requires only that the shifts be driven
by **one common nuisance**. So if a `sys_tag` is defined differently in different energy groups,
the question is whether those groups share a nuisance.

Concrete example — the **vz (z-vertex) cut**:
- 19.6, 27 GeV: default $|v_z|<70$ cm, varied to $|v_z|<35$ cm;
- 7.7, 9.2, 11.5, 14.6, 17.3 GeV: default $|v_z|<145$ cm, varied to $|v_z|<70$ cm.

Different acceptance regimes ⇒ do **not** assume the $70$-cm-regime and $145$-cm-regime share a
nuisance. **Split into two regime-grouped sources**, $A=\{19.6,27\}$ and $B=\{7.7,9.2,11.5,14.6,
17.3\}$, with $(\mathbf{s}_{\mathrm{vz}_A})_e=0$ for $e\notin A$ and $(\mathbf{s}_{\mathrm{vz}_B})_e=0$
for $e\notin B$:

$$V_{\rm sys} = \mathbf{s}_{\mathrm{vz}_A}\mathbf{s}_{\mathrm{vz}_A}^{\mathsf T}
             + \mathbf{s}_{\mathrm{vz}_B}\mathbf{s}_{\mathrm{vz}_B}^{\mathsf T}
             + \sum_{\text{other }s}\mathbf{s}_s\mathbf{s}_s^{\mathsf T}.$$

This gives a **block structure**: vz correlated within $A$, within $B$, and exactly zero across
$A$–$B$. The diagonal is unchanged ($\sum_s(\mathbf{s}_s)_e^2 = \sigma_{{\rm sys},e}^2$), so only
the off-diagonal coupling between the two groups is removed. Neither ρ=0 nor global ρ=1 can
express this; per-source does, purely through how sources are grouped.

**General rule:** audit every `sys_tag` — any whose variation definition changes by energy group
must be entered as one source *per group*, not one source across all energies.

## Implementation (done, 2026-06-01)

1. **`combine_sys.py`** writes `sys_sources_{cb}` = the signed contributions
   $(\mathbf{s}_s)_e$ (one value per `sys_tag`, per combined bin) to the output yaml,
   alongside the quadrature-summed `yerr_sys`.
2. **`plot_energy_dep`** (`build_sys_covariance`) assembles
   $V = \mathrm{diag}(\sigma_{{\rm stat},e}^2) + \sum_s \mathbf{s}_s\mathbf{s}_s^{\mathsf T}$
   with `SYS_REGIME_SPLITS`, and computes
   $\chi^2 = \boldsymbol{\Delta}^{\mathsf T} V^{-1}\boldsymbol{\Delta}$ (falls back to a
   diagonal $V_{\rm sys}$ if `sys_sources` is absent in an old yaml).

## Worked example (TPC, 10–40%)

The two active sources, as signed contribution vectors over the 7 energies
($s_{s,e}$, from `sys_sources_1040`):

```
            7.7      9.2      11.5     14.6     17.3     19.6     27
Vz  (tag1): -0.00287 -0.03683 -0.01051 -0.02009 -0.03092  0.0      0.0
Nfit(tag2):  0.00098 -0.00962  0.00099  0.00396  0.00191  0.00024 -0.00729
```

Vz is **negative at every lower energy** (tightening $|V_z|$ coherently lowers $R$) and
**zero at 19.6 & 27** (failed the Barlow gate there).

**Diagonal entry** $V_{\rm sys}[e,e]=\sum_s s_{s,e}^2 = \sigma_{{\rm sys},e}^2$. At 9.2:
$$(-0.03683)^2 + (-0.00962)^2 = 0.0014488,\quad \sqrt{\;}=0.0381 = \texttt{yerr\_sys}_{9.2},$$
so the build reproduces the existing per-energy systematic exactly.

**Off-diagonal (covariance) entry** $V_{\rm sys}[e,e']=\sum_s s_{s,e}\,s_{s,e'}$:
$$\mathrm{Cov}(9.2,14.6)=\underbrace{(-0.03683)(-0.02009)}_{\text{Vz}=+0.000740}
+\underbrace{(-0.00962)(0.00396)}_{\text{Nfit}=-0.000038}=+0.000702,$$
giving $\rho = 0.000702/(0.0381\times0.0205)\approx 0.9$ — Vz dominates and is positive
(both shifts negative), so 9.2 and 14.6 are strongly correlated.
$$\mathrm{Cov}(19.6,14.6)=\underbrace{(0)(-0.02009)}_{\text{Vz}=0}
+\underbrace{(0.00024)(0.00396)}_{\text{Nfit}\approx10^{-6}}\approx 10^{-6}\;(\rho\approx0.02),$$
i.e. 19.6 is essentially decoupled because Vz vanished there.

**Resulting correlation matrix:**
```
         7.7    9.2    11.5   14.6   17.3   19.6   27
 7.7    1.00   0.24   0.27   0.29   0.28   0.01  -0.09
 9.2    0.24   1.00   0.87   0.88   0.93  -0.02   0.23
 11.5   0.27   0.87   1.00   0.92   0.93   0.01  -0.08
 14.6   0.29   0.88   0.92   1.00   0.97   0.02  -0.17
 17.3   0.28   0.93   0.93   0.97   1.00   0.01  -0.06
 19.6   0.01  -0.02   0.01   0.02   0.01   1.00  -0.07
 27    -0.09   0.23  -0.08  -0.17  -0.06  -0.07   1.00
```
Read-off: {9.2, 11.5, 14.6, 17.3} form a tightly-correlated block ($\rho\approx0.87$–0.97)
tied by the coherent Vz source; 7.7 couples weakly (small Vz shift); 19.6 and 27 float nearly
free (Vz zero there). This block is exactly why the covariance χ² rose — those four energies
are forced to agree to ~stat precision but their central values scatter. Reproduce:
`scripts/diagnostics/explain_cov.py`.

## Caveats to report alongside the number

- Only **2 sources** ⇒ $V_{\rm sys}$ is rank $\le 2$; the off-diagonal structure is real but
  thin, and the per-energy $\sigma_{\rm sys}$ is itself noisy (C3; the EPD $\chi^2/\mathrm{ndf}
  = 0.13 \ll 1$ flags over-estimated EPD systematics).
- "100 % correlated across energies per source" is the standard default but is an assumption;
  a genuinely energy-dependent cut effect not captured by $(\mathbf{s}_s)_e$ is a residual.
