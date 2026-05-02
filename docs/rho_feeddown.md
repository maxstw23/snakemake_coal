# ρ Feed-Down Correction for Pion v₂

## 1. The Problem: Systematic Shape Tension in Pion v₂

The current Bayesian NCQ model predicts pion v₂ via the two-quark composition:

$$v_2^{\pi^+}(p_T) = v_2^u(p_T/2) + v_2^{\bar{u}}(p_T/2) + \varepsilon_0$$
$$v_2^{\pi^-}(p_T) = v_2^{\bar{u}}(p_T/2) + v_2^d(p_T/2) + \varepsilon_0$$

where $\varepsilon_0$ is a flat (pT-independent) isospin-symmetric hadronic offset. A full-range χ²
check against 19.6 GeV data (centrality 10–40%) reveals a severe **S-curve** residual pattern:

| $p_T$ range | Behaviour | Pull magnitude |
|-------------|-----------|----------------|
| 0.19–0.28 GeV | Model **overshoots** data | −54σ to −15σ |
| 0.34–0.78 GeV | Model **undershoots** data | +8σ to +17σ |
| 0.84–1.94 GeV | Good agreement | ~1–3σ |

Overall: $\chi^2/N \approx 152$ for $\pi^+$ and $97$ for $\pi^-$ (36 bins each).
A flat offset $\varepsilon_0$ cannot correct a **shape** mismatch — it shifts the entire curve uniformly.

The asymmetry between $\pi^-$ and $\pi^+$ (piminus worse in the 0.34–0.78 GeV range) further
suggests a charge-dependent effect beyond what a shared $\varepsilon_0$ can absorb.

---

## 2. Physical Origin: ρ→ππ Feed-Down Dilution

A significant fraction of observed pions in Au+Au collisions are **decay daughters of ρ
resonances** ($\rho^0 \to \pi^+\pi^-$, $\rho^\pm \to \pi^\pm\pi^0$). The two-body decay
kinematics randomise the daughter azimuthal angle relative to the mother, diluting v₂.

### 2.1 Kinematic Dilution Factor

For a mother of transverse momentum $p_{T,m}$, the mean v₂ weight transferred to a daughter
via the decay angle $\Delta\phi = \phi_{\rm daughter} - \phi_{\rm mother}$ is:

$$D(p_{T,m}) = \langle \cos 2\Delta\phi \rangle = \frac{1}{2}\int_0^\pi
\frac{p_\parallel^2 - p_\perp^2}{p_\parallel^2 + p_\perp^2}\,\sin\alpha\,d\alpha$$

where $\alpha$ is the polar decay angle in the ρ rest frame, and the parallel/perpendicular
components in the lab frame are:

$$p_\parallel = \gamma_T(p^*\cos\alpha + \beta_T E^*), \qquad
p_\perp = p^*\sin\alpha$$

with the boost parameters and decay kinematics:

$$p^* = \frac{\sqrt{M_\rho^2 - 4m_\pi^2}}{2} \approx 0.358\ \text{GeV}, \qquad
E^* = \frac{M_\rho}{2} \approx 0.388\ \text{GeV}$$

$$\beta_T = \frac{p_{T,m}}{m_{T,m}}, \qquad \gamma_T = \frac{m_{T,m}}{M_\rho},
\qquad m_{T,m} = \sqrt{p_{T,m}^2 + M_\rho^2}$$

**Physical limits:**

- $D(p_{T,m} \to 0) = 0$: The decay kick is isotropic; v₂ is fully diluted.
- $D(p_{T,m} \to \infty) = 1$: The boost strongly collimates daughters; v₂ is fully preserved.
- Crossover scale: $p_{T,m} \sim 2p^* = \sqrt{M_\rho^2 - 4m_\pi^2} \approx 0.72$ GeV.

This dilution factor — rising from 0 to 1 over the range 0–1.5 GeV — produces exactly the
observed S-curve: at low $p_T$ the real pion data is suppressed below the undiluted NCQ
prediction, while at intermediate $p_T$ the model, having no dilution correction, compensates
by being pulled upward, yielding a systematic undershoot.

### 2.2 Monte Carlo Transfer Matrix

In practice, the dilution is computed numerically via a Monte Carlo transfer matrix that
accounts for the full phase-space integration:

$$T_{\rm yield}[d, m] = \Pr(\text{daughter in bin } d \mid \text{mother in bin } m)$$
$$T_{v_2}[d, m] = \langle \cos 2\Delta\phi \rangle_{d \leftarrow m}$$

The observed v₂ of the **decay component** at daughter bin $d$ is then:

$$v_2^{\rm decay}[d] = \frac{\sum_m T_{v_2}[d,m]\, f_m[m]\, v_2^{\rm mother}[m]}{\sum_m T_{\rm yield}[d,m]\, f_m[m]}$$

where $f_m[m]$ is the mother (ρ) $p_T$ spectrum, modelled by a blast-wave distribution.

The ResFlow package (`~/Research/ResFlow`) implements this calculation in PyTorch with
$N_{\rm MC} = 5\times 10^6$ events, providing `T_yield` and `T_v2` as precomputed NumPy arrays.

---

## 3. Mixture Model for Observed Pion v₂

The observed pion v₂ is a mixture of a **primordial** component (pions produced directly,
following the NCQ formula) and a **feed-down** component (daughters of ρ decay):

$$v_2^{\rm obs}(p_T) = (1 - f_\rho)\, v_2^{\rm NCQ}(p_T) + f_\rho\, v_2^{\rm decay}(p_T)$$

### 3.1 NCQ Constraint on the Mother

Under the NCQ picture, the ρ resonance itself satisfies the same two-quark composition
(since $n_q = 2$, $\rho^+ = u\bar{d}$):

$$v_2^\rho(p_T) \approx v_2^{\rm NCQ,\,meson}(p_T)$$

That is, **the mother ρ and the primordial pion share the same underlying quark v₂ function**.
This means the mixture simplifies to:

$$v_2^{\rm obs}(p_T) = v_2^{\rm NCQ}(p_T)\cdot\bigl[(1 - f_\rho) + f_\rho\, D_{\rm eff}(p_T)\bigr]$$

where $D_{\rm eff}(p_T)$ is the effective pT-dependent dilution factor (folded over the ρ
spectrum and transfer matrix). The shape modification is fully determined by known kinematics
once $f_\rho$ is specified.

### 3.2 Charge Asymmetry

The ρ feed-down enters differently for $\pi^+$ and $\pi^-$:

- $\rho^0 \to \pi^+\pi^-$: contributes symmetrically to both
- $\rho^+ \to \pi^+\pi^0$: contributes only to $\pi^+$
- $\rho^- \to \pi^-\pi^0$: contributes only to $\pi^-$

In a neutron-rich system (Au+Au, $N > Z$), the $\rho^+/\rho^-$ ratio deviates from unity,
potentially generating an additional isospin asymmetry beyond what the flat $\varepsilon_0$
captures. At first approximation, if $f_\rho^+$ and $f_\rho^-$ are allowed to differ, this
would naturally explain why $\pi^-$ residuals are larger than $\pi^+$ in the intermediate
$p_T$ range.

---

## 4. Planned Implementation in `quark_v2_bayes.py`

### 4.1 Precomputation (one-time, outside the PyMC model)

Before sampling, build the transfer matrices from ResFlow:

```python
from resflow import build_transfer_matrices

pt_edges = np.arange(0, 3.01, 0.05)  # bin edges for transfer matrix
T_yield, T_v2, pt_m_centers = build_transfer_matrices(
    pt_edges, M_parent=0.7755, m_daughter=0.13957,
    T_kin=0.100, beta_avg=0.4, N_mc=5_000_000
)
```

Convert to PyTensor constants for use inside the model:

```python
T_yield_pt = pt.as_tensor_variable(T_yield.astype('float64'))
T_v2_pt    = pt.as_tensor_variable(T_v2.astype('float64'))
```

### 4.2 Model Changes

**New parameters:**

| Parameter | Prior | Meaning |
|-----------|-------|---------|
| `f_rho` | $\mathrm{Beta}(2, 10)$ ≈ 10–20% | ρ feed-down fraction |

The Beta(2,10) prior centres around $f_\rho \approx 0.17$ with most mass below 0.3,
consistent with thermal model estimates (~10–20%) and the ResFlow result of f ≈ 0.13 at 27 GeV.

**Replace** the flat `eps0_pi` with the feed-down forward model:

```python
# Blast-wave ρ spectrum at mother pT bin centres (precomputed, fixed)
spec_rho = blastwave_dndpt(pt_m_centers, mass=M_RHO, T_kin=T_kin, beta_avg=beta_avg)
spec_rho_t = pt.as_tensor_variable(spec_rho / spec_rho.sum())

# Feed-down fraction
f_rho = pm.Beta('f_rho', alpha=2, beta=10)

def v2_decay_pion(p_daughter_arr):
    """v2 of pions from rho decay, evaluated at daughter pT grid."""
    # Mother v2 at mother pT centres (same NCQ formula evaluated at pt_m_centers)
    v2_rho_m = qv2(pt_m_centers / 2, 'u') + qv2(pt_m_centers / 2, 'a')
    weighted  = T_v2_pt * spec_rho_t[None, :] * v2_rho_m[None, :]
    norm      = T_yield_pt * spec_rho_t[None, :]
    # Interpolate from transfer matrix bins to data pT grid
    ...
    return v2_decay_at_daughter_bins

def ncq_pred_pion(sp, p_arr):
    v2_ncq   = qv2(p_arr/2, 'u') + qv2(p_arr/2, 'a')   # piplus example
    v2_decay = v2_decay_pion(p_arr)
    return (1 - f_rho) * v2_ncq + f_rho * v2_decay
```

**Remove** `eps0_pi` from the model. The ρ feed-down correction now provides the
pT-dependent shape correction that the flat offset could not.

### 4.3 Parameter Count

| Before | After |
|--------|-------|
| `eps0_pi` ~ N(0, 0.05) [1 param] | `f_rho` ~ Beta(2,10) [1 param] |
| Total: 18 free | Total: 18 free |

Net change: zero additional parameters, but with a **physically motivated pT shape** instead of
a flat offset.

### 4.4 Expected Improvement

The transfer matrix dilution factor $D_{\rm eff}(p_T)$ rises from ~0 at $p_T \sim 0$ to ~1
above $p_T \sim 1.5$ GeV, which should:

- Suppress the model prediction at low $p_T$ (fixing the overshoot)
- Reshape the intermediate range (fixing the undershoot at 0.4–0.8 GeV)
- Leave the high-$p_T$ region ($> 0.9$ GeV) essentially unchanged (since $D \approx 1$ there)

---

## 5. Open Questions

1. **Charge-specific feed-down:** Should $f_\rho^+$ and $f_\rho^-$ be separate parameters
   to account for the $\rho^+/\rho^-$ asymmetry in neutron-rich matter? This would add 1
   parameter but could explain why $\pi^-$ residuals are systematically larger.

2. **Blast-wave spectrum sensitivity:** The transfer matrix depends on the assumed ρ $p_T$
   spectrum (blast-wave parameters $T_{\rm kin}$, $\beta_{\rm avg}$). How sensitive is
   $f_\rho$ to this assumption? A systematic study varying $T_{\rm kin}$ and $\beta_{\rm avg}$
   would quantify this.

3. **Energy dependence of $f_\rho$:** The ρ/π ratio is energy-dependent. The prior on
   $f_\rho$ could be made energy-dependent using measured ρ yields from STAR.

4. **Interaction with isospin asymmetry:** The new $f_\rho$ and the existing `delta_f`
   (isospin asymmetry in quark transport) both affect the $\pi^-$–$\pi^+$ difference. Their
   interplay in the posterior needs careful monitoring.

---

## 6. First-Order Taylor Expansion of the Transfer Matrix

### 6.1 Motivation

The full transfer-matrix forward model requires a matrix–vector multiply inside PyMC
(PyTensor), which is feasible but introduces a potential degeneracy: since v₂_decay is
computed from the same NCQ formula as v₂_NCQ, the model can redistribute probability
between $f_\rho$ and the quark-level parameters $(a, b, c, d, \nu)$ to achieve the same
likelihood. A first-order Taylor expansion collapses the matrix into two precomputed
scalar functions of $p_T$, making the correction analytically transparent and easier to
identify.

### 6.2 Derivation

Under NCQ, the ρ mother satisfies

$$v_2^\rho(p_{T,m}) = v_2^{\rm NCQ}(p_{T,m})$$

so the decay component at daughter bin $d$ is

$$v_2^{\rm decay}[d] = \frac{\displaystyle\sum_m T_{v_2}[d,m]\; s_m\; v_2^{\rm NCQ}(p_{T,m})}{\displaystyle\sum_m T_{\rm yield}[d,m]\; s_m}$$

where $s_m$ is the blast-wave ρ spectrum weight in mother bin $m$.

Expand $v_2^{\rm NCQ}(p_{T,m})$ around the daughter bin center $p_{T,d}$:

$$v_2^{\rm NCQ}(p_{T,m}) \approx v_2^{\rm NCQ}(p_{T,d}) + v_2^{\rm NCQ}{}^\prime(p_{T,d})\cdot(p_{T,m} - p_{T,d})$$

Substituting and collecting terms:

$$v_2^{\rm decay}[d] \;\approx\; D_{\rm eff}(p_{T,d})\cdot v_2^{\rm NCQ}(p_{T,d}) \;+\; \Delta p_{T,\rm eff}(p_{T,d})\cdot v_2^{\rm NCQ}{}^\prime(p_{T,d})$$

where the two **precomputed scalar arrays** are

$$\boxed{D_{\rm eff}(p_{T,d}) = \frac{\displaystyle\sum_m T_{v_2}[d,m]\; s_m}{\displaystyle\sum_m T_{\rm yield}[d,m]\; s_m}}$$

$$\boxed{\Delta p_{T,\rm eff}(p_{T,d}) = \frac{\displaystyle\sum_m T_{v_2}[d,m]\; s_m\;(p_{T,m} - p_{T,d})}{\displaystyle\sum_m T_{\rm yield}[d,m]\; s_m}}$$

Inserting into the mixture model $v_2^{\rm obs} = (1-f_\rho)\,v_2^{\rm NCQ} + f_\rho\,v_2^{\rm decay}$:

$$\boxed{v_2^{\rm obs}(p_T) \;=\; v_2^{\rm NCQ}(p_T)\cdot\bigl[1 - f_\rho\bigl(1 - D_{\rm eff}(p_T)\bigr)\bigr] \;+\; f_\rho\,\Delta p_{T,\rm eff}(p_T)\cdot v_2^{\rm NCQ}{}^\prime(p_T)}$$

### 6.3 Physical Interpretation of Each Term

| Term | Behaviour | Physical meaning |
|------|-----------|-----------------|
| $v_2^{\rm NCQ}\cdot[1 - f_\rho(1-D_{\rm eff})]$ | Low $p_T$: $D_{\rm eff}\to 0$, factor $\to (1-f_\rho)$. High $p_T$: $D_{\rm eff}\to 1$, factor $\to 1$. | Dilution: decay daughters at low $p_T$ inherit almost no $v_2$ from the mother. |
| $f_\rho\,\Delta p_{T,\rm eff}\cdot v_2^{\rm NCQ}{}^\prime$ | Large at low $p_T$ (mothers far above daughters); small at high $p_T$. | pT-shift: the T_{v2}-weighted mother distribution is shifted to higher $p_T$ than the daughter, so the mother contributes a v₂ value from a steeper part of the curve. |

$D_{\rm eff}$ runs from $\approx 0$ at $p_T = 0$ to $\approx 1$ above $p_T \sim 1.5$ GeV, matching the physical crossover scale $2p^* \approx 0.72$ GeV. $\Delta p_{T,\rm eff}$ is positive (mothers always above daughters in $T_{v_2}$-weighted sense) and largest at low $p_T$.

### 6.4 Why This Breaks the Degeneracy

In the full matrix approach, $f_\rho$ multiplies a vector $v_2^{\rm decay}[\cdot]$ that is itself computed from the NCQ parameters — so the model can trade $f_\rho$ against $(a, b, c)$ and obtain equal likelihood. In the first-order expansion:

- The zeroth-order correction $[1 - f_\rho(1-D_{\rm eff}(p_T))]$ changes the **shape** of $v_2^{\rm NCQ}$ in a pT-dependent way (from $1-f_\rho$ at low $p_T$ to $1$ at high $p_T$). This shape change cannot be mimicked by rescaling $a$ alone.
- The first-order correction $f_\rho\,\Delta p_{T,\rm eff}\cdot v_2^{\rm NCQ}{}'$ adds a term proportional to the derivative — a different shape than $v_2^{\rm NCQ}$ itself.

Together, $f_\rho$ is identified by the **curvature change** it induces relative to the pure NCQ shape, not just by an amplitude shift.

### 6.5 Zeroth-Order Approximation (Pure Dilution)

Dropping the derivative term gives the simplest implementable version:

$$v_2^{\rm obs}(p_T) \;\approx\; v_2^{\rm NCQ}(p_T)\cdot\underbrace{\bigl[1 - f_\rho\bigl(1 - D_{\rm eff}(p_T)\bigr)\bigr]}_{\text{precomputed, one free param } f_\rho}$$

This is a single pT-dependent attenuation factor multiplied onto the NCQ prediction, with $D_{\rm eff}(p_T)$ as a fixed array. Implementation inside `build_model()`:

```python
# Precomputed outside PyMC (from transfer matrix):
#   D_eff    : array (n_data_bins,)  — effective dilution at each data pT
#   dpt_eff  : array (n_data_bins,)  — pT-shift moment at each data pT

D_eff_t   = pt.as_tensor_variable(D_eff.astype('float64'))
dpt_eff_t = pt.as_tensor_variable(dpt_eff.astype('float64'))

# Inside build_model():
f_rho = pm.Beta('f_rho', alpha=2, beta=10)

def ncq_pred_pion(sp, p_arr):
    v2_ncq = qv2(p_arr/2, 'u') + qv2(p_arr/2, 'a')   # piplus example
    # Zeroth-order correction:
    correction = 1.0 - f_rho * (1.0 - D_eff_t)
    return v2_ncq * correction
    # First-order correction (add if zeroth-order insufficient):
    # v2_ncq_prime = ...  # finite-difference or analytic derivative
    # return v2_ncq * correction + f_rho * dpt_eff_t * v2_ncq_prime
```

### 6.6 Validity of the Approximation

The Taylor expansion is accurate when $v_2^{\rm NCQ}$ varies slowly over the spread of the mother $p_T$ distribution (i.e., over the width of $T_{v_2}[d,:]$ as a function of $m$). This spread is roughly $\sigma \sim 0.3$–$0.5$ GeV. Since the Richards sigmoid has a characteristic scale $b \sim 0.5$–$1.0$ GeV, the approximation is expected to be good to $\mathcal{O}(\sigma^2/b^2) \sim 10$–$25\%$ of the correction. The first-order term captures the dominant part of this systematic.

A numerical check (comparing the Taylor-expanded prediction against the full matrix result for a typical NCQ curve) can be done by evaluating both on a dense $p_T$ grid before committing to the approximation.

---

## 7. Transfer Matrix — Explicit Definition

### 7.1 Binning

Discretize $p_T$ into $N$ bins with edges:

$$\{p_0, p_1, p_2, \ldots, p_N\}$$

and bin centres:

$$p_{T,i} = \frac{p_i + p_{i+1}}{2}, \quad i = 0, 1, \ldots, N-1$$

Current implementation: $N = 50$, range $[0, 2.5]$ GeV, bin width $0.05$ GeV.

### 7.2 Monte Carlo Simulation

Simulate $N_{\rm MC} = 5\times 10^6$ independent $\rho \to \pi^+\pi^-$ decays. For each event $k$:

1. Sample mother $\rho$ momentum $p_{T}^{(m,k)}$ from blast-wave spectrum.
2. Decay in $\rho$ rest frame: daughters have momentum
   $$p^* = \frac{\sqrt{M_\rho^2 - 4m_\pi^2}}{2} \approx 0.358\ \text{GeV}$$
   in opposite directions with decay angle $\alpha$ drawn uniformly.
3. Boost to lab frame:
   $$\beta_T = \frac{p_T^{(m)}}{\sqrt{(p_T^{(m)})^2 + M_\rho^2}}, \qquad \gamma_T = \frac{1}{\sqrt{1 - \beta_T^2}}$$
   Each daughter acquires lab momentum $p_T^{(d,k)}$ and azimuth $\phi^{(d,k)}$.
4. Record per daughter: mother bin $m$, daughter bin $d$, and $\cos(2\Delta\phi^{(k)})$ where $\Delta\phi^{(k)} = \phi^{(d,k)} - \phi^{(m,k)}$.

Only $\Delta\phi$ matters — the mother's absolute $\phi$ is irrelevant because $v_2$ is measured relative to the event plane. The uniform draw of $\alpha$ fully determines $\Delta\phi$ after boosting.

### 7.3 Matrix Element $T_{\rm yield}[d, m]$

**Definition:**

$$T_{\rm yield}[d, m] = \Pr\bigl(\text{daughter in bin } d \;\big|\; \text{mother in bin } m\bigr)$$

**Construction from Monte Carlo:**

Let $C[m \to d]$ be the count of daughters landing in bin $d$ whose mother was in bin $m$. Then:

$$T_{\rm yield}[d, m] = \frac{C[m \to d]}{\sum_{d'} C[m \to d']}$$

**Properties:**
- $T_{\rm yield}[d, m] \geq 0$ for all $d, m$
- $\sum_d T_{\rm yield}[d, m] = 1$ (normalized per-daughter; each mother produces 2 daughters, so raw counts sum to $2 \cdot C[m]$, but we divide by $2 \cdot C[m]$)

**Physical interpretation by column $m$ (i.e. $T_{\rm yield}[\,:\,, m]$ for fixed mother $p_T$):**
- Mother at rest ($p_T^{(m)} = 0$): isotropic decay, daughters uniformly distributed up to $p^* \approx 0.358$ GeV. The column is broad, peaking near $p^*$.
- Boosted mother (large $p_T^{(m)}$): one daughter is collimated forward (inherits $p_T \approx p_T^{(m)}$), the other backward (gets low $p_T$). The column has two peaks: near $0$ and near $p_T^{(m)}$.

### 7.4 Matrix Element $T_{v_2}[d, m]$

**Definition:**

$$T_{v_2}[d, m] = \bigl\langle \cos(2\Delta\phi) \bigr\rangle_{\substack{\text{daughter in } d \\ \text{mother in } m}}$$

The average is over all MC events where the mother is in bin $m$ and the daughter lands in bin $d$.

**Construction from Monte Carlo:**

Let $S_{c2}[m \to d] = \sum_{k \in (m \to d)} \cos(2\Delta\phi^{(k)})$ be the accumulated $\cos 2\Delta\phi$. Then:

$$T_{v_2}[d, m] = \frac{S_{c2}[m \to d]}{C[m \to d]}$$

**Properties:**
- $-1 \leq T_{v_2}[d, m] \leq 1$
- $T_{v_2}[d, m] = 0$: decay fully randomizes azimuth (daughter $v_2$ gets zero contribution from mother $v_2$)
- $T_{v_2}[d, m] = 1$: daughter perfectly inherits mother's azimuth ($v_2^{\rm daughter} = v_2^{\rm mother}$)
- In practice $T_{v_2}[d, m] \geq 0$ always, because two-body decay preferentially emits daughters near the mother's direction

**Physical interpretation by regime:**
- **Low $p_T$ daughter from low $p_T$ mother**: decay kick ($p^* = 0.358$ GeV) dominates over boost. Daughters spray isotropically. $\langle\cos 2\Delta\phi\rangle \to 0$. Thus $T_{v_2} \to 0$.
- **High $p_T$ daughter from high $p_T$ mother**: boost collimates daughters. $\Delta\phi$ is small. $\cos 2\Delta\phi \to 1$. Thus $T_{v_2} \to 1$.
- **Intermediate**: $T_{v_2}$ rises smoothly from $0$ to $1$ over $p_T \sim 0\text{--}1.5$ GeV.

### 7.5 From Matrices to $v_2^{\rm decay}$

The observed $v_2$ of decay pions in daughter bin $d$ is a weighted average over all mother bins $m$:

$$v_2^{\rm decay}[d] = \frac{\displaystyle\sum_m T_{v_2}[d,m] \cdot s[m] \cdot v_2^\rho[m]}{\displaystyle\sum_m T_{\rm yield}[d,m] \cdot s[m]}$$

where:
- $s[m] = \frac{dN}{dp_T}\bigr|_{p_T = p_{T,m}}$ is the blast-wave $\rho$ spectrum weight in mother bin $m$
- $T_{\rm yield}[d,m] \cdot s[m]$ is the number of daughters from bin $m$ that land in bin $d$, weighted by how many mothers exist at bin $m$
- $T_{v_2}[d,m] \cdot s[m]$ is the same, further weighted by the $v_2$ transfer efficiency

**Full matrix form** (as stored by `build_rho_precomputed()`):

$$\boxed{A[d, m] = T_{v_2}[d, m] \cdot s[m]} \qquad \text{shape } (N, N)$$

$$\boxed{{\rm norm}[d] = \sum_{m=0}^{N-1} T_{\rm yield}[d, m] \cdot s[m]} \qquad \text{shape } (N,)$$

$$\boxed{v_2^{\rm decay}[d] = \frac{\displaystyle\sum_{m=0}^{N-1} A[d, m] \cdot v_2^\rho[m]}{{\rm norm}[d]}}$$

The mother $v_2^\rho[m]$ is computed from the **same** NCQ formula evaluated at mother $p_T$:

$$v_2^\rho[m] = v_2^u\!\left(\frac{p_{T,m}}{2}\right) + v_2^{\bar{u}}\!\left(\frac{p_{T,m}}{2}\right)$$

No new free parameters — the $\rho$'s $v_2$ reuses the quark-level $v_2^q$ functions already in the model.

### 7.6 Observed Pion as a Mixture

The measured pion $v_2$ in daughter bin $d$ is:

$$\boxed{v_2^{\rm obs}[d] = (1 - f_\rho) \cdot v_2^{\rm NCQ}[d] \;+\; f_\rho \cdot v_2^{\rm decay}[d]}$$

where:
- $v_2^{\rm NCQ}[d] = v_2^u(p_{T,d}/2) + v_2^{\bar{u}}(p_{T,d}/2)$ is the primary (direct) pion prediction
- $f_\rho \sim {\rm Beta}(2, 10)$ is the fraction of observed pions originating from $\rho$ decays

Both $v_2^{\rm NCQ}$ and $v_2^{\rm decay}$ are functions of the same quark $v_2$ parameters. The only new degree of freedom is $f_\rho$. The $p_T$-dependent shape correction is encoded in the precomputed matrices — not from extra fitting flexibility.

### 7.7 Why This Fixes the Shape Mismatch

**Low $p_T$ (0.18–0.34 GeV, residuals −15σ to −54σ):**
$T_{v_2}[d,m]$ is small ($\sim 0$–$0.3$) because these daughters come from low-$p_T$ mothers where the decay kick dominates. Hence $v_2^{\rm decay}[d] \ll v_2^{\rm NCQ}[d]$, and:
$$v_2^{\rm obs}[d] = (1 - f_\rho) \cdot v_2^{\rm NCQ}[d] + f_\rho \cdot (\text{diluted}) \;<\; v_2^{\rm NCQ}[d]$$
The model prediction **drops**, fixing the overshoot.

**Mid $p_T$ (0.4–0.8 GeV, residuals +8σ to +17σ):**
$T_{v_2}[d,m]$ is intermediate ($\sim 0.3$–$0.7$). Without feed-down, the model compensated for the low-$p_T$ overshoot by distorting the NCQ parameters, creating an undershoot here. With $f_\rho$ providing the correct low-$p_T$ suppression, the NCQ parameters relax to their proper values — fixing the undershoot.

**High $p_T$ (>0.9 GeV, already good fit):**
$T_{v_2}[d,m] \to 1$ (daughters inherit mother $v_2$). Hence $v_2^{\rm decay}[d] \approx v_2^{\rm NCQ}[d]$, and:
$$v_2^{\rm obs}[d] \approx v_2^{\rm NCQ}[d] \quad \text{regardless of } f_\rho$$
No change to the already-good high-$p_T$ prediction.
