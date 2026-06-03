# Statistical treatment of the coalescence ratio $R$

Companion to [stats_review.md](stats_review.md) item **C1**. This note collects the
equations for handling the near-zero-denominator problem in the coalescence ratio.

## The observable and the problem

Per centrality, the measured coalescence ratio is

$$R = \frac{N}{D} = \frac{v_2^{\pi^-} - \tfrac23 v_2^{\bar p}}{v_2^{\pi^+} - \tfrac23 v_2^{\bar p}}
\;\;\overset{\text{model}}{=}\;\; \frac{N_d^{\,tr}}{N_u^{\,tr}} = \frac{2N+Z}{N+2Z}
\;\approx\; \frac{315}{276} \approx 1.14 \quad(\text{Au}).$$

The denominator $D = v_2^{\pi^+} - \tfrac23 v_2^{\bar p}$ is a difference of two similar small
numbers, so it is often statistically consistent with zero. When $D \to 0$ the ratio has a
**pole**, and $R$ is no longer approximately Gaussian — it becomes Cauchy-like.

Variances-only errors (cross-species covariance neglected by the PID-exclusivity argument,
see stats_review.md C1):

$$\sigma_D = \sqrt{\sigma^2_{v_2^{\pi^+}} + \tfrac49\,\sigma^2_{v_2^{\bar p}}}, \qquad
\sigma_N = \sqrt{\sigma^2_{v_2^{\pi^-}} + \tfrac49\,\sigma^2_{v_2^{\bar p}}}.$$

The numerator and denominator share $v_2^{\bar p}$, so they are correlated even with
variances-only inputs:

$$\sigma_{ND} = \mathrm{Cov}(N, D) = \tfrac49\,\sigma^2_{v_2^{\bar p}}.$$

## Why the naive error fails (and what is safe)

A ratio of Gaussians has **no finite mean or variance** (Cauchy-like). The `uncertainties`
package computes a first-order (delta-method) estimate of *the mean and standard deviation* —
quantities that do not exist here — which is why it can return e.g. $R = 3.13 \pm 18.8$.

- **Mean ± std**: never use it for $R$. It is non-reproducible noise when $D$ is small.
- **Median + percentiles**: robust **only** when the denominator rarely changes sign, i.e.
  $D/\sigma_D \gtrsim 2\text{–}3$. When $D/\sigma_D \approx 0$ (~50% sign flips) even the median
  is unstable and more toys do not help.

### The diagnostic / gate

$$\frac{D}{\sigma_D} = \frac{1}{\sigma_D/|D|} = \text{significance of the denominator}.$$

| $D/\sigma_D$ | regime | action |
|---|---|---|
| $\gtrsim 3$ | $R$ ≈ Gaussian | linear error fine; this is where the physics lives |
| $\sim 1\text{–}3$ | skewed / heavy-tailed | use Fieller or toy median + asymmetric interval |
| $\lesssim 1$ | pole inside error cloud | flag "denominator $\sim 0$, $R$ unconstrained"; do not quote |

---

## Option 1 — Fieller's theorem (exact CI, analytic, no pole)

Stop dividing. The hypothesis "$R = r$" is equivalent to the **linear** statement

$$g(r) \equiv N - r\,D = 0.$$

For fixed $r$, $g(r)$ is a linear combination of Gaussians, hence Gaussian:

$$\widehat g(r) = \hat N - r\hat D, \qquad
\mathrm{Var}\!\left[g(r)\right] = \sigma_N^2 - 2r\,\sigma_{ND} + r^2\sigma_D^2.$$

The $(1-\alpha)$ confidence interval for $R$ is the set of $r$ with $\widehat g$ consistent
with zero, $\;\widehat g(r)^2 \le t^2\,\mathrm{Var}[g(r)]\;$ with $t = z_{1-\alpha/2}$
($t = 1$ for 68%). This is a **quadratic in $r$**:

$$\boxed{\;(\hat D^2 - t^2\sigma_D^2)\,r^2
\;-\; 2(\hat N\hat D - t^2\sigma_{ND})\,r
\;+\; (\hat N^2 - t^2\sigma_N^2) \;\le\; 0\;}$$

Let the discriminant define the roots

$$r_{\pm} = \frac{(\hat N\hat D - t^2\sigma_{ND}) \pm
\sqrt{(\hat N\hat D - t^2\sigma_{ND})^2 - (\hat D^2 - t^2\sigma_D^2)(\hat N^2 - t^2\sigma_N^2)}}
{\hat D^2 - t^2\sigma_D^2}.$$

Behaviour, controlled entirely by the leading coefficient:

- **$\hat D^2 - t^2\sigma_D^2 > 0 \iff \hat D/\sigma_D > t$** → bounded interval $[r_-, r_+]$
  (asymmetric, matches the toy percentiles; reduces to $\hat N/\hat D \pm$ linear-$\sigma$ when
  $D/\sigma_D \gg 1$).
- **$\hat D/\sigma_D \le t$** → denominator not significant → the solution set is the
  *exterior* of the roots: an **unbounded interval** (or all reals). Fieller correctly reports
  "unconstrained."

**Key point:** Fieller's condition for a bounded interval, $\hat D/\sigma_D > t$, is exactly
the empirical $D/\sigma_D$ gate above — derived rigorously. Fieller is the analytic twin of the
toy-MC: same interval where the toy median is stable, honest unboundedness where it is not.

---

## Option 2 — Global slope fit — REJECTED (physics: neutron skin)

> **Do not use a global, centrality-independent slope fit.** It was proposed on the assumption
> that $R = (2N+Z)/(N+2Z)$ is a fixed property of the Au nucleus. That assumption is **wrong**:
> the **neutron skin** means peripheral collisions preferentially sample the neutron-rich
> nuclear surface, so the effective participant $N/Z$ — and hence $R$ — **increases toward
> peripheral**. The centrality dependence of $R$ is therefore a *signal* (a probe of the
> neutron skin), not noise to be averaged out. A single-slope fit across all centralities would
> destroy exactly the trend we want to measure.
>
> Consequence: treat each centrality (or merged bin) **independently** with Fieller (Option 1)
> or toy-MC, then study $R$ vs centrality. A slope fit is at most valid *within* a narrow
> centrality range where $R$ is approximately constant — not globally — and even then the
> merge-then-ratio + Fieller route is simpler and sufficient for the reported bins.

---

## Option 3 — Equivalent linear signal $\Delta v_2^\pi$ (already plotted)

Since

$$R - 1 = \frac{N - D}{D} = \frac{v_2^{\pi^-} - v_2^{\pi^+}}{D} = \frac{\Delta v_2^\pi}{D},$$

the physics signal is simply the pion splitting $\Delta v_2^\pi = v_2^{\pi^-} - v_2^{\pi^+} \ne 0$,
a clean pole-free Gaussian observable (`delta_pion.pdf` already exists). The prediction
$R = 315/276$ is the linear statement

$$\Delta v_2^\pi = \frac{39}{276}\,D = \frac{39}{276}\bigl(v_2^{\pi^+} - \tfrac23 v_2^{\bar p}\bigr),$$

which is the same line as the Option-2 slope fit (of $\Delta v_2^\pi$ vs $D$).

---

## Recommendation

- **Fieller** for per-centrality and per-merged-bin confidence intervals (drop-in, closed-form,
  exact; honest unbounded intervals in the tail). This is the headline tool.
- Study **$R$ vs centrality** as the physics observable — the peripheral rise is the
  neutron-skin signal. Do **not** impose a global centrality-independent slope (see Option 2).
- Toy-MC remains a useful independent cross-check; report median + 16/84 only where
  $D/\sigma_D \gtrsim 2\text{–}3$.
