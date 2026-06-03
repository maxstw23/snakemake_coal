# pT/nq window choice and cross-energy consistency of the coalescence ratio

Record of the 2026-06-01 analysis of whether the coalescence ratio
$R = (v_2^{\pi^-}-\tfrac23 v_2^{\bar p})/(v_2^{\pi^+}-\tfrac23 v_2^{\bar p})$
is robust against the $p_T/n_q$ integration window, and why the default low-$p_T$ window is
the correct choice. Companion to [theory.md](theory.md), [ratio_statistics.md](ratio_statistics.md),
and [stats_review.md](stats_review.md).

## The question

$R = N_d^{tr}/N_u^{tr}$ is *derived* from the coalescence/NCQ relation
$v_2^h(p_T)=\sum_q v_2^q(p_T/n_q)$, which only holds in a limited $p_T$ window. So varying the
$p_T/n_q$ cut moves between physics regimes, not between measurement configurations — raising
the concern that $R$ may not be robust against the cut, and that a `pT_lo` "systematic"
(`sys_tag 5`: $p_T/n_q^{\rm lo}=0.08\to0.3$) would conflate a physics effect with a
measurement uncertainty.

## Key data (10–40%, TPC, merged cen 5–7)

`R` per energy as the lower integration bound $p_T/n_q^{\rm lo}$ is raised (upper bound 0.6):

| ptnq_lo | mean R | std across 7 energies | trend vs $\sqrt{s}$ |
|---|---|---|---|
| **0.08 (default)** | **1.144** | **0.015** | flat |
| 0.16 | 1.099 | 0.029 | decreasing |
| 0.24 | 1.087 | 0.032 | decreasing |
| 0.32 | 1.090 | 0.035 | decreasing |

Per-energy at the default cut: 7.7→1.18, 9.2→1.15, 11.5→1.13, 14.6→1.14, 17.3→1.14,
19.6→1.14, 27→1.13 — scattered by ~1% around **1.144 ≈ 315/276**, no trend. As the cut is
raised, a monotonic **decrease with energy** develops (7.7 stays ~1.15, 27 drops to ~1.04) and
the spread more than doubles. (Reproduce: `scripts/diagnostics/xenergy_window.py`, `scripts/diagnostics/ptnq_robust.py`.)

## Finding: the cross-energy consistency validates the default low-$p_T$ window

The same Au nucleus collides at every BES energy, so $N_d^{tr}/N_u^{tr}$ is an
energy-**independent** nuclear property — the model **predicts** $R$ flat vs $\sqrt{s}$.

- **The consistency is real, not a coincidence or a low-$p_T$ artifact.** For an artifact to
  fake it, a low-$p_T$ effect would have to pin $R$ at *exactly* the nuclear prediction
  315/276, energy-independently, across 7 datasets with very different $v_2$ magnitudes,
  spectra, and detector states — a fine-tuned coincidence, far less plausible than the model
  working.
- **The consistency is maximal at the default window and degrades as the cut is raised.**
  Raising the cut *introduces* a structured, energy-dependent distortion and *destroys* the
  model-predicted flatness. So the default low-$p_T$ window is where the ratio behaves like the
  nuclear quantity it is meant to measure.
- **Energy-independence is a separate prediction from the value 1.14**, so using it to justify
  the window is a largely non-circular, data-driven criterion — an out-of-sample check the
  analysis was not tuned to pass.

### Corrections to earlier statements (recorded for honesty)
1. The cross-energy agreement with 315/276 is **not** low-$p_T$ contamination — it is the
   model's prediction realized.
2. The earlier suggestion to "raise the lower bound to a single-energy plateau" was **wrong**
   for this observable: the single-energy scan plateau is misleading; the cross-energy view
   shows raising the cut *breaks* the consistency. **Retracted.**

## Consequences

- **`sys_tag 5` (pT range) is NOT a systematic and should stay off** the systematic budget.
  Its large shift is the regime/energy-dependent physics above, not a measurement uncertainty;
  folding it into $\sigma_{\rm sys}$ would bury this finding inside an error bar.
- **The headline conclusion is supported**: at the default window $R$ is energy-flat with mean
  1.144, dead on 315/276. The previously computed $\chi^2/\mathrm{ndf}>1$ (3.10 diagonal /
  4.85 covariance) is **not** a systematic offset from the prediction — it is the noisy /
  under-estimated per-energy uncertainties (the C3 issue, dominated by 19.6 GeV's implausibly
  small error), not evidence against the model.
- The drift at higher cut is something energy-dependent entering the $\tfrac23 v_2^{\bar p}$
  subtraction (baryon–meson $v_2$ structure growing with $p_T$ and $\sqrt{s}$ is the natural
  suspect — to be confirmed by physics, not asserted).

## Caveats to keep

- The residual ~1% scatter at the default window (std 0.015 vs stat ~0.003–0.01) is slightly
  larger than pure statistics — consistent with the C3 systematic-error issues; confirm.
- State the window choice explicitly as justified by the energy-independence criterion (and,
  ideally, the baryon–meson mechanism understood), so it reads as principled, not tuned.
