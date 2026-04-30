# Bayesian Quark v2 Extraction

Extracts latent quark-level $v_2$ functions from a simultaneous Bayesian fit to
$\pi^\pm$, $K^\pm$, $p$, $\bar{p}$ $v_2(p_T)$ data using the NCQ coalescence model.

**Script**: `scripts/quark_v2_bayes.py`

---

## Physics Model

### NCQ predictions

Each hadron's $v_2$ is the sum of its constituent quark $v_2$ values (NCQ):

| Hadron | NCQ prediction |
|--------|---------------|
| $\pi^+$ (u$\bar{d}$) | $v_2^u(p_T/2) + v_2^{\bar{q}}(p_T/2) + \varepsilon_0^{\pi}$ |
| $\pi^-$ ($\bar{u}$d) | $v_2^{\bar{q}}(p_T/2) + v_2^d(p_T/2) + \varepsilon_0^{\pi}$ |
| $K^+$ (u$\bar{s}$)  | $v_2^u(p_T/2) + v_2^s(p_T/2)$ |
| $K^-$ ($\bar{u}$s)  | $v_2^{\bar{q}}(p_T/2) + v_2^s(p_T/2)$ |
| $p$ (uud)            | $2\,v_2^u(p_T/3) + v_2^d(p_T/3)$ |
| $\bar{p}$ ($\bar{u}\bar{u}\bar{d}$) | $3\,v_2^{\bar{q}}(p_T/3)$ |

### Mixture model for transported quarks

$v_2^u$ and $v_2^d$ are decomposed into transported and produced components:

$$v_2^u(p) = f\cdot v_2^{\rm tr}(p) + (1-f)\cdot v_2^{\rm prod}(p)$$

$$v_2^d(p) = f\cdot v_2^{\rm tr}(p) + (1-f)\cdot v_2^{\rm prod}(p) \qquad \text{(isospin symmetry: same } f\text{)}$$

$$v_2^{\bar{q}}(p) = v_2^{\rm prod}(p) \qquad \text{(identity: all produced)}$$

- **$v_2^{\rm prod}$**: produced (anti)quarks, pinned by $\bar{p}$ data
- **$v_2^{\rm tr}$**: transported quarks, independent Richards sigmoid
- **$f \in (0,1)$**: transported fraction, shared for u and d

### Richards sigmoid parametrization

All quark $v_2$ functions use the generalized Richards sigmoid (5 parameters per flavor):

$$v_2^q(p) = \frac{a}{\left(1 + e^{-(p-b)/c}\right)^{1/\nu}} - d$$

$\nu = 1$ recovers the standard logistic; $\nu < 1$ gives a slower approach to saturation.

### Pion hadronic correction

Pions are pseudo-Goldstone bosons — hadronic rescattering and $\rho \to \pi\pi$ feed-down
boost their $v_2$ above the NCQ expectation by an isospin-blind offset, **shared by
$\pi^+$ and $\pi^-$**:

$$\varepsilon_0^{\pi} \sim \mathcal{N}(0,\, 0.05) \qquad \text{(single parameter, applied to both charges)}$$

Because the offset is identical for both charges, it cancels exactly in
$v_2^{\pi^-} - v_2^{\pi^+}$, leaving the isospin signal $f\cdot(v_2^{\rm tr}-v_2^{\rm prod})$
intact.

**Key outputs**:
- $f \cdot (v_2^{\rm tr} - v_2^{\rm prod})$: transported quark contribution
- $v_2^{\rm tr}$ vs $v_2^{\rm prod}$: is $v_2^{\rm tr}$ meaningfully above $v_2^{\rm prod}$?
- Posterior of $f$ vs thermal model prior $\tanh(\mu_B / 3T_{\rm ch})$

---

## Bayesian Inference

### Parameters

$$\theta = \{\,\underbrace{a_{\rm tr}, b_{\rm tr}, c_{\rm tr}, d_{\rm tr}, \nu_{\rm tr}}_{v_2^{\rm tr}},\;
\underbrace{a_a, b_a, c_a, d_a, \nu_a}_{v_2^{\rm prod}},\;
\underbrace{a_s, b_s, c_s, d_s, \nu_s}_{v_2^s},\;
f,\;
\varepsilon_0^{\pi}\,\} \quad (17\text{ parameters})$$

### Likelihood

Each data point is modelled as a Gaussian observation:

$$v_2^{{\rm sp},\,{\rm obs}}(p_{T,i}) \sim \mathcal{N}\!\left(\mu_{\rm sp}(p_{T,i};\,\theta),\;\sigma_i^{\rm sp}\right)$$

where $\sigma_i^{\rm sp}$ is the measured errorbar and $\mu_{\rm sp}$ is the NCQ prediction. The full likelihood factorizes over species and bins:

$$p(\text{data}\mid\theta) = \prod_{\rm sp}\prod_i \mathcal{N}\!\left(v_2^{{\rm sp},\,{\rm obs}}(p_{T,i})\;\Big|\;\mu_{\rm sp}(p_{T,i};\,\theta),\;\sigma_i^{\rm sp}\right)$$

Bins with smaller $\sigma_i$ impose tighter constraints — pion data dominates the NCQ shape; baryon data anchors $v_2^{\rm prod}$ and $f \cdot v_2^{\rm tr}$.

### Priors

| Parameter | Prior | Motivation |
|-----------|-------|------------|
| $a_q$ | HalfNormal(0.15) | $v_2$ amplitude positive, $\mathcal{O}(0.1)$ |
| $b_q$ | Normal(0.3, 0.3) | inflection near 300 MeV/$c$ |
| $c_q$ | HalfNormal(0.2) | width scale positive |
| $d_q$ | Normal(0, 0.02) | small baseline offset |
| $\nu_q$ | LogNormal(0, 0.8) | positive, $\mathcal{O}(1)$ |
| $\text{logit}(f)$ | $\mathcal{N}\!\left(\text{logit}(\tanh(\mu_B/3T_{\rm ch})),\;0.5\right)$ | thermal model prior (arXiv:1701.07065) |
| $\varepsilon_0^{\pi}$ | $\mathcal{N}(0, 0.05)$ | shared by $\pi^+$ and $\pi^-$ (isospin-blind); K, p, $\bar{p}$ carry no offset |

### Posterior

By Bayes' theorem:

$$p(\theta \mid \text{data}) \;\propto\; p(\text{data}\mid\theta)\cdot p(\theta)$$

The posterior is a 17-dimensional probability distribution — a "landscape" where
height represents how consistent a parameter combination is with both the data and
our prior beliefs. We want to draw random points from this landscape, with denser
sampling in regions of higher probability.

This cannot be computed analytically for two reasons:
1. **No closed form**: the product of Gaussians (likelihood) times non-conjugate priors
   (LogNormal for $\nu$, HalfNormal for $a$, etc.) does not simplify to a known
   distribution.
2. **Normalizing constant is intractable**: computing $\int p(\text{data}\mid\theta)\,p(\theta)\,d\theta$
   over 17 dimensions is a $\mathcal{O}(N^{17})$ numerical problem — infeasible for
   any reasonable grid size $N$.

We instead use **Markov Chain Monte Carlo (MCMC)**: construct a random walk through
parameter space that, by design, visits regions proportional to their posterior
probability. After enough steps, the chain's empirical distribution converges to the
true posterior.

---

### How NUTS works

**Step 1 — The landscape.** Define the potential energy as

$$U(\theta) = -\log p(\theta \mid \text{data}) = -\log p(\text{data}\mid\theta) - \log p(\theta)$$

High posterior probability = low $U$. We want to sample from the "valleys" of this landscape.

**Step 2 — Introduce momentum (Hamiltonian Monte Carlo).** HMC augments the parameter
vector $\theta$ with an auxiliary momentum vector $\rho$ of the same dimension, drawn
fresh each iteration from $\mathcal{N}(0, I)$. The total energy is:

$$H(\theta, \rho) = U(\theta) + \frac{1}{2}\|\rho\|^2$$

Think of a ball ($\theta$) rolling on the potential energy surface $U$, with $\rho$
as its velocity. The ball follows **Hamiltonian dynamics**: it rolls downhill (toward
high-probability regions) and gains speed, then rolls uphill and slows, conserving
total energy $H$ throughout. This means it explores the landscape efficiently without
getting stuck.

Numerically, dynamics are integrated using the **leapfrog algorithm**:

$$\rho_{t+\epsilon/2} = \rho_t - \frac{\epsilon}{2}\nabla_\theta U(\theta_t)$$
$$\theta_{t+\epsilon} = \theta_t + \epsilon\,\rho_{t+\epsilon/2}$$
$$\rho_{t+\epsilon} = \rho_{t+\epsilon/2} - \frac{\epsilon}{2}\nabla_\theta U(\theta_{t+\epsilon})$$

Here $\epsilon$ is the step size and $\nabla_\theta U$ is the gradient of the log-posterior
with respect to all 17 parameters — computed automatically by PyMC via autodifferentiation.
After $L$ leapfrog steps, the proposed $(\theta', \rho')$ is accepted with probability
$\min(1,\, e^{-H(\theta',\rho') + H(\theta,\rho)})$. Because $H$ is conserved, acceptance
is nearly always 1 when the integrator is accurate.

**Step 3 — NUTS: automatic trajectory length.** HMC requires choosing $L$ (number of
leapfrog steps). Too few: slow exploration. Too many: the ball doubles back on itself,
wasting computation. NUTS solves this by **doubling the trajectory** (forward or backward
in time) until it detects the ball starting to turn back — the "No-U-Turn" criterion:

$$(\theta' - \theta) \cdot \rho' < 0 \quad \text{(stop: trajectory is turning back)}$$

NUTS also adapts $\epsilon$ during the tuning phase (first 1000 steps, discarded) to
hit the target acceptance rate of 0.95.

**Step 4 — The chain.** Each accepted $\theta'$ becomes the next state. After tuning,
the chain runs for 2000 draws × 4 independent chains. The result is a set of
$S = 8000$ samples $\{\theta^{(s)}\}$ that collectively approximate the posterior.

**Why 4 chains?** Running multiple chains from different starting points lets us
detect non-convergence: if all chains mix to the same distribution, $\hat{R} \approx 1$.
Large $\hat{R}$ means chains are exploring different regions — the posterior is not
yet characterized.

Configuration: 4 chains × 2000 draws + 1000 tuning steps, target acceptance = 0.95.

### From samples to quark $v_2(p_T)$ curves

NUTS returns $S$ posterior samples $\{\theta^{(s)}\}_{s=1}^S$. Each sample is a full
set of 17 parameter values. The quark $v_2$ functions are then obtained by a
deterministic **pushforward**: for each sample, evaluate the Richards sigmoid at a
fine $p_T$ grid:

$$v_2^{q,(s)}(p) = \frac{a_q^{(s)}}{\left(1 + e^{-(p - b_q^{(s)})/c_q^{(s)}}\right)^{1/\nu_q^{(s)}}} - d_q^{(s)}$$

This produces an ensemble of $S$ curves $\{v_2^{q,(s)}(p)\}$. The posterior
distribution of the quark $v_2$ function at any given $p$ is then approximated by
the histogram of these $S$ values. Crucially, the curves are **smooth and correlated
across $p_T$** because they share the same underlying parameters — the posterior
does not treat each $p_T$ bin independently.

For the mixture flavors u and d, the pushforward is:

$$v_2^{u,(s)}(p) = f^{(s)} \cdot v_2^{{\rm tr},(s)}(p) + (1 - f^{(s)}) \cdot v_2^{{\rm prod},(s)}(p)$$

where $f^{(s)}$, $v_2^{{\rm tr},(s)}$, $v_2^{{\rm prod},(s)}$ all come from the **same**
posterior draw — preserving the correlations between $f$ and the two component functions.

**What makes a "good" posterior for $v_2^q(p_T)$?**

- **Narrow band**: parameters are well-determined by the data; the quark $v_2$ function
  is precisely constrained at those $p_T$ values
- **Band covers data**: posterior predictive check passes — model is well-specified
- **Smooth credible intervals**: no oscillations or kinks, which would indicate
  sampler pathologies (check R̂ and divergences)
- **f posterior shifted from prior**: the data is informative about the transported
  fraction beyond what the thermal model alone provides

### Posterior predictive check

The posterior predictive integrates out parameter uncertainty:

$$p(\hat{v}_2^{\rm sp} \mid \text{data}) = \int p(\hat{v}_2^{\rm sp}\mid\theta)\,p(\theta\mid\text{data})\,d\theta \;\approx\; \frac{1}{S}\sum_{s=1}^S \mu_{\rm sp}(p_T;\,\theta^{(s)})$$

The shaded bands show the 16th–84th (dark) and 2.5th–97.5th (light) percentiles
across samples.

> **Important**: posterior predictive width reflects **parameter uncertainty**, not
> model adequacy. A narrow band that misses the data signals model misspecification —
> the parameters are well-determined but the model form is wrong.

---

## Data

Input CSVs produced by the main Snakemake pipeline:

```
result/sys_tag_0/energy_{E}/v2_noeff_corrected_cen{1-9}.csv   # raw v2(pT), 500 bins × 0.01 GeV
result/sys_tag_0/energy_{E}/v2_noeff_corrected_res.csv        # EPD resolution per centrality
```

Columns used: `{particle}_v2_EPD`, `{particle}_v2_err_EPD` for
`particle` ∈ {piplus, piminus, kplus, kminus, proton, antiproton}.

**Resolution correction**:

$$v_2^{\rm corr}(p_T) = v_2^{\rm raw}(p_T) \;/\; R_{\rm EPD}$$

where $R_{\rm EPD}$ is the per-centrality event-plane resolution scalar.

---

## Configuration

```python
ENERGY        = '19p6GeV'     # energy tag matching result/ directory
CEN_BINS      = [5, 6, 7]    # centrality bins 5/6/7 → 10-40%
EP            = 'EPD'         # event plane method

MESON_PT_LO   = 0.16         # GeV  → pT/nQ = 0.08 GeV
MESON_PT_HI   = 2.00         # GeV
BARYON_PT_LO  = 0.24         # GeV  → pT/nQ = 0.08 GeV
BARYON_PT_HI  = 3.00         # GeV

REBIN         = 5             # rebin factor → 0.05 GeV bins
```

Centrality bin → percentage mapping:

| Bins | Centrality |
|------|-----------|
| 8, 9 | 0–10% |
| 5, 6, 7 | 10–40% |
| 1, 2, 3, 4 | 40–80% |

---

## Running

```bash
conda activate lambda_v1
cd /mnt/d/Research/snakemake_coal
python scripts/quark_v2_bayes.py
```

Sampling takes ~30 minutes on a 16-core machine (4 chains × 2000 draws).

---

## Output

All output goes to `plots/{ENERGY}/`:

| File | Contents |
|------|----------|
| `quark_v2_functions.pdf` | 5 panels: $v_2^u$, $v_2^d$, $v_2^{\rm prod}$, $v_2^{\rm tr}$, $v_2^s$ vs $p_T/n_q$ with 68%/95% posterior bands |
| `quark_v2_comparison.pdf` | All 5 quark $v_2$ curves overlaid |
| `transported_signal.pdf` | $v_2^{\rm tr}$ vs $v_2^{\rm prod}$; $f\cdot(v_2^{\rm tr}-v_2^{\rm prod})$; posterior of $f$ |
| `posterior_predictive.pdf` | Observed hadron $v_2$ vs posterior predictive (model check) |
| `trace.nc` | Full ArviZ posterior trace (netCDF), reloadable for further analysis |

---

## Convergence Checks

| Statistic | Target |
|-----------|--------|
| $\hat{R}$ | $< 1.01$ |
| ESS (bulk) | $> 400$ |
| Divergences | 0 |

To reload a saved trace:

```python
import arviz as az
trace = az.from_netcdf('plots/19p6GeV/trace.nc')
```

---

## Extending to Other Energies

Available energies: `7p7GeV`, `11p5GeV`, `14p6GeV`, `19p6GeV`, `27GeV`.

The transported fraction $f$ and the signal $f\cdot(v_2^{\rm tr} - v_2^{\rm prod})$
are expected to grow toward lower energies as baryon stopping increases (larger $\mu_B$).
