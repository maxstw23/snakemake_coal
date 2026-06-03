# Statistical Integrity Review — Findings & Tracker

Scan date: 2026-05-31. Reviewer: Claude (with maxstw23).
Scope: full pipeline — efficiency, C++ v2 extraction, `simple_profile.py`, `fit_v2.py`,
`plot_v2_new.py`, `combine_sys.py`, `quark_v2_bayes.py`, `rho_coal_ratio.py`.

Status legend: `OPEN` (not yet discussed) · `DISCUSSING` · `ACCEPTED` (real, to fix) ·
`WONTFIX` (judged acceptable, with reason) · `FIXED`.

---

## CRITICAL — affects headline numbers

### C1. Ratio error propagation breaks near a zero denominator — `IMPLEMENTED`
Files: `plot_v2_new.py:231` (per-cen), `:237` (merged), `:737` (pT scan); `rho_coal_ratio.py:455`.
`uncertainties` does first-order linear propagation. The denominator
D = v2_π+ − ⅔v2_p̄ is small & noisy by construction, so the ratio is Cauchy-like, not
Gaussian → symmetric σ underestimates the uncertainty and the point estimate is biased;
meaningless when D is consistent with 0.
Note done right: resolution cancels in the ratio (shared ufloat, `plot_v2_new.py:104–111`).

DECIDED fix: parametric toy-MC (≡ the parametric bootstrap already used in
rho_coal_ratio.py:133). Sample THREE independent Gaussians, one per species
(v2_π+, v2_π−, v2_p̄), each N(value, stat_err²); form N and D from the SAME draws so the
shared-v2_p̄ term is handled algebraically; report median + 16/84 percentiles. Full FC/Neyman
deemed overkill (no physical boundary here, just a non-Gaussian transform); percentile
interval = Neyman/fiducial for a monotonic transform of Gaussians, so it has coverage.

COVARIANCE decision (2026-05-31): neglect cross-species covariance, use per-species
variances only. Justification: PID exclusivity (dE/dx bands, TOF m² cuts, opposite charge)
makes track-counting shot noise independent across species — the dominant channel is zero
by construction. Residual channels NOT removed: shared EP Ψ fluctuation + same-event
momentum-conservation non-flow; judged 2nd-order (EPD large-η gap; tends to cancel in a
ratio) and accepted as an explicit approximation. Empirical fallback if challenged:
sub-sample method (M groups → Var(R) with all correlations baked in), needs per-event
output from the C++ extraction. NOTE: shared v2_p̄ is NOT this covariance — it's algebraic
and is still preserved by the single-draw-into-both-N-and-D scheme above.

DIAGNOSTIC DONE (2026-05-31, scripts/diagnostics/dsigma_diagnostic.py): D/σ_D computed for all
energy/cen/EP, with linear-vs-toyMC comparison.
- Mid-central 10–50% (where the 1.14 physics test lives): D/σ_D ≳ 5–90, linear == toy to the
  digit. HEADLINE RESULT NOT AT RISK from C1.
- Peripheral cen1–2 (and a few 0–5% EPD at low E): D/σ_D ≲ 1 → linear prints nonsense, e.g.
  19.6 TPC 70–80% R=3.13±18.8 (toy 0.96 [0.11,1.79]); 27 EPD 70–80% R=4.27±29.7 (toy 0.98).
  Both central value AND error are wrong there; toy gives sane bounded asymmetric intervals.
- BADRES low-E peripheral EPD bins already masked out of analysis → no action.
- Merged headline likely safer: ratio is formed AFTER counts-weighted v2 merge
  (plot_v2_new.py:237), which raises D/σ_D vs the noisy cen1/2 inputs. Confirm merged 40–80%.

REVISED SCOPE: C1 is localized to peripheral error bars, not the physics. Fix = toy-MC
(median+16/84) for the per-cen ratio scatter; flag bins with D/σ_D≲1 as "denominator ~0,
R unconstrained" rather than quoting them. Keep linear elsewhere (it agrees).

MECHANISM CONFIRMED (scripts/diagnostics/pole_mechanism.py): the pole does NOT disappear in the toy.
- Ratio of Gaussians is Cauchy-like → NO finite mean/variance. The linear `uncertainties`
  result IS a delta-method estimate of that (nonexistent) mean±std → that's why it returns
  e.g. 3.13±18.8. Toy mean reproduces the garbage (std ~547, non-reproducible). Never use
  mean±std for the ratio.
- MEDIAN+percentiles are robust ONLY when sign-flips are rare. At D/σ_D=3.9, 0% sign flips,
  median reproducible to the digit. At D/σ_D≈0, ~50% of toys flip the denominator sign,
  density at the median →0, and the median itself is unstable (0.96→0.04 on reseed, even at
  1e6 toys). More toys do NOT help (density vanishes, not MC noise).
GATE (concrete): use sign-flip fraction (or D/σ_D) as the trigger.
  D/σ_D ≳ 2–3  → toy median + 16/84 (meaningful, asymmetric). Covers all physics bins.
  D/σ_D ≲ 1   → flag "denominator ~0, R unconstrained", do not quote. Toy's role here is
               purely diagnostic (sign-flip fraction is the cleanest trigger).

IMPLEMENTED 2026-06-02 (plot_v2_new.py): ratio_toymc() samples (N,D) from their exact 2×2
uncertainties-covariance (shared p̄ + resolution handled; res cancels in R), reports
median + 16/84 + signflip fraction. Wired into the per-cen scatter (coal_combined.pdf,
coal_{EP}.pdf) as ASYMMETRIC error bars; bins with signflip>0.16 print an "unconstrained"
warning. ADDITIVE: linear y/yerr kept (combine_sys unaffected); toy stored as new yaml keys
(y_toymc, yerr_toymc_lo/hi, signflip_frac). Merged headline (y_1040 etc.) left LINEAR (correct,
D/σ_D≫1). Validated 19.6: mid-central toy==linear; peripheral asymmetric (70% +0.113/−0.085).
FULL PROPAGATION DONE 2026-06-02: combine_sys.py now emits asymmetric per-cen
yerr_stat_lo/hi (from the default's toy) and yerr_lo/hi = √(stat_side²+sys²) (sys symmetric);
generate_paper_plots.py plot_ratio plots y_toymc + ASYMMETRIC stat bars + symmetric sys band.
Verified: peripheral 19.6 TPC bin y=1.254 stat +0.113/−0.085 sys=0.043 tot +0.121/−0.095.
MERGED bins (010/1040/4080) kept SYMMETRIC by design (D/σ_D≫1; toy==linear) → the energy_dep
χ²/ndf (which uses y_1040) is UNCHANGED (3.10 diag / 4.85 cov). This is also correct statistics:
asymmetric errors in a χ² are ill-defined, so χ² inputs must stay symmetric — and here they
legitimately are. Minor residual: combine_sys still computes the sys DELTA wrt linear y (not
y_toymc); negligible for well-determined bins, exact once all sys_tags carry toy fields.

### C2. `/3` in systematic combination — `WONTFIX` (not a bug; reviewer misread)
File: `combine_sys.py:43–44, 67–68`. RESOLVED 2026-06-01 (maxstw23): the `/3` is (√3)² =
the variance of a UNIFORM distribution of half-width δ (PDG/GUM Type-B). It applies PER
SOURCE: σ_sys² = Σ_i (δ_i² − δ_err,i²)/3, summed in quadrature. NOT a source-count divisor.
Coherent chain: observed shift δ → Barlow-subtract stat part (h²=δ²−δ_err²) → uniform
half-width h → variance h²/3 → quadrature → add to stat².
RESIDUAL (minor, documentation only):
- √3 vs √12 is a prior choice. /3 treats δ as the HALF-width of a symmetric ±δ box (assumes
  bias could go either way, you saw one side). /12 treats δ as the FULL range (the span
  default→variation actually probed). For a single one-directional cut variation /12 is the
  more literal reading; /3 is the more conservative (2× larger). Either is defensible — but
  ADD A CODE COMMENT + state the convention in the analysis note (currently reads as a magic
  number).

### C3. Systematic-estimation weak link (binding issue) — `DISCUSSING`
PULL TEST 2026-06-03 (scripts/diagnostics/pull_test.py): per-pT v2 2nd-difference scatter /
expected-from-errors = 0.83–1.12 (median ~0.95) at 19.6 GeV across π±/p̄ × cen5–7 × TPC/EPD →
per-pT STAT errors are HONEST, so the integrated stat errors (incl 19.6's small 0.0029) are
REAL, not under-estimated. CORRECTION: 19.6's stat error is fine; it's 19.6's SYSTEMATIC
(0.0002, both sources failed Barlow) that's implausibly small — a C3 symptom. IMPLICATION: the
7 energies scatter by ~1% (std 0.015) > their stat (~0.005) → with stat validated, the χ²>1 /
covariance-4.85 is a SYSTEMATIC/physics issue (under-estimated 2-source budget OR a real ~1%
energy variation), NOT a stat artifact. → C3 is genuinely the binding issue; PID+east/west
will resolve which. Original Barlow-gate points below still stand:
File: `combine_sys.py:37, 39–42`. `significance = (delta_err < delta)`; only passers added.
UPDATE 2026-06-01: the gate ALSO guards h²=δ²−δ_err²>0 (avoids sqrt of negative), so it's more
sensible than first credited — flooring non-significant sources at 0 is the standard Barlow
choice. RESIDUAL concern (real but mild): only ever ADDING positive contributions, never
letting a downward stat fluctuation reduce σ_sys, gives a mild UPWARD bias. Defensible
conservatism; document it. Separately: `delta_err = sqrt(|σ_sys²−σ_def²|)` uses abs(), which
masks the σ_sys<σ_def case that would flag a broken subset/correlation assumption — worth a
warning print rather than silent abs().

---

## SIGNIFICANT

### S4. Resolution uncertainty dropped in 2 of 3 consumers — `ACCEPTED` (fix later, not now)
DECISION 2026-06-01 (maxstw23): real, should be corrected — add res as a nuisance parameter
(per centrality/EP, Gaussian prior from res_err) in quark_v2_bayes.py; propagate res_err in
rho_coal_ratio.py. Deferred, not implementing now.
Files: `quark_v2_bayes.py:209,214–215`; `rho_coal_ratio.py:379–382,450`. Both divide by
`res_val` nominal only; `{ep}_res_err` unused. Resolution is a fully-correlated
normalization across all pT/species in a centrality → Bayesian likelihood is overconfident.
`plot_v2_new.py` does propagate it → inconsistent treatment. Fix: nuisance parameter per
centrality/EP with Gaussian prior from res_err.

### S5. Pion "S-curve" correction — `DEMOTED` (revisit for π⁺/π⁻ expansion)
File: `quark_v2_bayes.py:516–527, 533–535`. UPDATE 2026-06-02: current focus is
v2^tr − v2^prod (NOT the π⁺/π⁻ isospin diff). The S-curve is CHARGE-BLIND and in this model
v2^π+ − v2^π- = v2^u − v2^d = 0 by construction, so it cannot touch an isospin signal (none in
the model). It only adjusts the common pion level / absorbs ρ feed-down — its intended job.
Minor for current focus. REVISIT when f_u≠f_d (the π⁺/π⁻ expansion) is added — then it sits on
the signal channel and the injection/prior-predictive checks become necessary.

### S6. `f`–(v2^tr−v2^prod) DEGENERACY; transported excess is prior-driven — `DISCUSSING`
File: `quark_v2_bayes.py:484, 498–504`. SHARPENED 2026-06-02: not just a tight prior — a genuine
degeneracy. v2^tr always enters as f·v2^tr, and v2^prod is pinned by antiparticles, so every
particle−antiparticle splitting measures only the PRODUCT f·(v2^tr−v2^prod)
(p−p̄=3f·Δ, Λ−Λ̄=2f·Δ, K+−K−=f·Δ). f and Δ=(v2^tr−v2^prod) are NOT separately identifiable from
data; the split is set by the thermal f prior (+ weak Richards amplitude priors). So the
HEADLINE object v2^tr−v2^prod ≈ (measured product)/(prior f) — central value AND band inherit
the f-prior. Must quantify + state. Interpolated T_ch/μ_B (`:188–190`) not propagated into σ.
DIAGNOSTIC DONE (scripts/diagnostics/bayes_prior_scan.py, 19.6 GeV, σ=0.5 vs 2.0):
- product f·Δ = 0.0044 [0.0043,0.0046] BOTH priors → prior-stable to ±3% (the data-driven qty).
- f: 0.295→0.272; Δ: 0.0150→0.0163, Δ band grows ~30% (0.0061→0.0080) → decomposition is
  prior-dependent, as predicted.
- NUANCE: even at σ=2.0 (~flat logit-f prior) f stayed tight ([0.21,0.34]) → a SECOND implicit
  prior, the Richards amplitude a_tr~HalfNormal(0.15), also breaks the f–Δ degeneracy. So this
  single-prior scan UNDERSTATES the true prior dependence of Δ.
RECOMMENDATION: (1) quote f·(v2^tr−v2^prod) as the measured result (±3%, prior-stable);
(2) report v2^tr−v2^prod & f only "given the GCE f prior", with a prior-sensitivity band that
varies BOTH the f prior AND a_tr; current transported_signal.pdf band understates the
uncertainty; (3) genuinely measuring Δ needs an independent handle on f.

### S7. Signal-fraction uncertainty not propagated into Λ v2 — `OUT OF SCOPE`
DECISION 2026-06-02: Λ v2 is not currently used → ignore. Mechanism still valid if revived:
v2-vs-mass fit uses bx=b/(b+s) FIXED; mass-fit signal-fraction error (`fit_v2.py:425/437`) not
folded into v2 error → Λ v2 errors underestimated in low-S/B bins. Fix then: refit with bx±σ_bx
or joint mass+v2 likelihood. See [[feedback_lambda_scope]].

### S8. "Fit until valid" randomized refitting — `DISCUSSING` (narrowed; likely minor)
File: `fit_v2.py:911–972`. Up to 500 random-seed retries until Minuit valid/accurate, fixed_z
after 250.
FINDINGS 2026-06-01:
- 0 dropped bins at every energy (scripts/diagnostics/dropped_bins.py) → non-random-missingness worry MOOT.
- v2 fit (fit_v2.py:485) is LINEAR in (v2,p0,p1) with bx(M) fixed from the mass fit, soft_l1
  convex → UNIQUE minimum. v2-level random restarts only clear Minuit numerical flags; they
  do NOT select a v2 value. So those retries are harmless.
- Seed-dependence can enter ONLY via the MASS fit: (1) double-Gaussian z-degeneracy (which
  optimum → which bx(M)); (2) single-Gaussian fallback (fixed_z=1 after 250 retries) → hard
  bins fit with a DIFFERENT signal model than neighbors → possibly shifted v2.
- Lambda is a cross-check, not the headline (feedback_lambda_scope) → lower priority.
NARROWED QUESTION: do fixed_z/high-retry bins have v2 shifted vs easy bins?
DIAGNOSTICS: cheap = log (refit_count, fixed_z_used) per bin + correlate with v2; decisive =
seed-stability rerun (3–5 global seeds, compare per-bin v2) — heavy.
FIX DIRECTION (later): keep best-cost valid fit among a FIXED N starts (not first-valid →
deterministic global optimum); seed starts from neighboring (pt,cen) bins (smooth spectrum) →
largely removes need for single-Gaussian fallback.

### S9. Efficiency-correction uncertainty not propagated — `BOUNDED, negligible for ratio`
NOT in the original scan; surfaced at wrap-up. TPC efficiency (TPC_eff.py fit → Efficiency.cpp,
applied via ExtendedTProfile weight 1/TOF_eff/TPC_eff, correct_eff=1) applied without propagating
its fit uncertainty into v2. Note TPC_eff.py saves only CENTRAL params (:109–113), not errors.
CHEAP BOUND 2026-06-02 (full eff correction's effect on R): R(eff) vs R(noeff), merged 10–40%,
moves only −0.5%…+0.3% across all 7 energies → the entire correction nearly cancels in the ratio
(charge-blind, as expected). Its UNCERTAINTY is a fraction of that → NEGLIGIBLE vs stat (~0.3%).
So full propagation NOT worth it for the ratio. CAVEAT: this bound is for the RATIO only;
eff-corrected per-species v2 or the Bayesian quark functions would need the real treatment
[(a) save fit errors in TPC_eff.py + bootstrap, or (b) eff variation as a sys_tag → covariance].

---

## MODERATE / NOTED

### M9. BW "χ²" in ρ inversion is not a χ²; BW params fixed w/o error — `OUT OF SCOPE`
DECISION 2026-06-02: ρ-from-pion-pairs (rho_coal_ratio.py) analysis not in use a.t.m. → ignore.
If revived: `:96–100` denominator = normalized model value not variance (Poisson broken by
/max); `:337–341` T,β,n fixed, no error propagated.

### M10. Bootstrap N=100 for a std; silent nan drops — `OUT OF SCOPE`
DECISION 2026-06-02: same ρ analysis (rho_coal_ratio.py), not in use → ignore. If revived: ~7%
uncertainty on the std; bump to ≥500, report failure rate.

### M11. `SimpleProfile.errors()` low-count fallback is ad hoc — `LOW PRIORITY`
File: `simple_profile.py:58–72`. count<5 → 2·sqrt(|⟨y²⟩−⟨y⟩²|) (global spread ×2).
Heuristic factor 2 + global substitution. NOTE 2026-06-02: feeds Λ v2 (out of scope) and the
pT-differential δv2 plots (π/p/K). The HEADLINE integrated ratio uses the C++ TProfile, NOT
SimpleProfile → unaffected. So only secondary plots; low priority unless a δv2 plot is a final.

### M12. Counts-weighting for rebin/merge — `WONTFIX` (reviewer misframe)
Files: `plot_v2_new.py:129–131` (merge), `:681–687` (pT scan).
RESOLVED 2026-06-01 (maxstw23): counts-weighting is CORRECT for rebinning, not suboptimal.
Merging pT (or cen) bins estimates the yield-weighted integrated v2 = Σ v2_i N_i / Σ N_i
(N_i = dN/dpT); counts ARE the yield, so this is the definition. Inverse-variance would be
WRONG (biased) — neighboring bins measure different v2(pT), not the same quantity. Error
formula σ=√(Σ(w_i ε_i)²)/Σw_i matches the estimator under bin independence. Only residual =
inter-bin shared-EP correlation, which is the SAME approximation already accepted in C1
(neglect shared-Ψ covariance). Consistent → no action.

### M13. Merging distinct centralities in Bayes fit — `ACCEPTED, low urgency`
File: `quark_v2_bayes.py:207–219`. Inverse-variance-averages cen 5/6/7 (different physics)
instead of counts-weighting (the physical yield-weighted average used everywhere else).
IMPACT CHECKED 2026-06-02 (scripts/diagnostics/m13_impact.py, 19.6 GeV): per-species integrated v2 differs
~1.4–2.3% (iv BELOW cw, coherently across all species); per-bin up to ~1.8σ. BUT the shift is
common and CANCELS in particle−antiparticle: p−p̄ moves only 0.15% (K+−K− ~2.4%). Since the
transported signal f·Δ lives in those differences, physics impact is sub-percent on the
dominant p−p̄ channel → does NOT move the result. FIX: switch load_merged_v2 to counts-weighting
for consistency (right average, matches ratio pipeline); low urgency (won't change f·Δ).
SCOPE: M13 is ONLY the Bayesian code (quark_v2_bayes.py). The MAIN pion-ratio headline
(0-10/10-40/40-80, plot_v2_new.py:129) is COUNTS-weighted → CORRECT (M12). One secondary
inverse-variance merge in the main analysis, merge_helper (plot_v2_new.py:654), only combines
adjacent noisy cen bins for the per-cen SCATTER display + per-cen systematics, NOT the headline;
same minor concern, could be counts-weighted for consistency.

### M14. soft_l1 robust loss + Hesse errors — `OUT OF SCOPE` (Λ fit)
File: `fit_v2.py:502`. Lives in the Λ v2 fit → not used (2026-06-02). If revived: under robust
loss Hesse cov ≠ χ² error; minos mitigates; state it.

### M15. `χ²/N` not reduced χ² (21 shared params) — `ACCEPTED` (trivial relabel)
File: `quark_v2_bayes.py:792–794`. dof = N points, not N−p. Agreed 2026-06-02. Fix: relabel/
footnote the per-panel value as χ²/N (not reduced χ²), or report one global χ²/(N_tot−21).

### M16. No look-elsewhere / multiple-comparison framing — `DISCUSSING`
Ratio compared vs several lines across 7 energies × 9 cen × 2 EP.
DECISION DISCUSSION 2026-06-01 (BH vs Bonferroni):
- If correcting per-bin: use FWER (Holm–Bonferroni), NOT FDR/BH. Reasons: (1) HEP convention
  is conservative re false positives (FDR is a screening tool); (2) tests are CORRELATED
  (shared systematics, overlapping merged/per-cen bins, common π/p̄ across EP) → plain BH
  assumes independence/PRDS, would need Benjamini–Yekutieli (×Σ1/i ≈ ln m, ~5× for m~100);
  Bonferroni/Holm valid under ANY dependence. Holm > plain Bonferroni (uniformly more powerful,
  same guarantee).
- BUT per-bin NHST is likely the WRONG FRAME: (a) arguing CONSISTENCY with 315/276 can't use
  rejection tests (failing to reject ≠ consistency; correction perversely inflates apparent
  consistency) → use global χ²/GOF or equivalence (TOST); (b) prediction is centrality-dependent
  (neutron skin) so "vs 1.14" tests the wrong null in most bins.
RECOMMENDATION: primary = single GLOBAL χ² of R(cen,energy) vs centrality-dependent model
(correlations in covariance) → one p-value, multiplicity vanishes. Reserve Holm–Bonferroni for
secondary per-bin "this bin deviates" flags, with PRE-REGISTERED family size m (currently m is
ill-defined = researcher DOF).

PARTIALLY IMPLEMENTED 2026-06-01: upgraded the χ²/ndf in generate_paper_plots.py
`plot_energy_dep` (energy_dep.pdf, the PAPER plot — NOT the Bayes one; that edit was reverted).
It already had a stat-only χ² vs 315/276; now uses FULL uncertainty sqrt(stat²+sys²) (reads
yerr_stat_1040 + yerr_sys_1040 from plots/final/), ndf=N (fixed ref, no params, M15), and adds
the TPC excl-7.7 line. On-plot annotations + stdout: TPC 21.7/7=3.10, TPC excl-7.7 10.3/6=1.72,
EPD 0.5/4=0.13.
NOTE: EPD χ²/ndf=0.13 ≪1 → EPD errors look OVER-estimated (large/noisy sys) — links to C3.
CROSS-ENERGY COVARIANCE (still open): bracket ρ=0 (uncorrelated)=3.10 vs ρ=1 (fully
correlated)=5.00 — wide, so correlation MATTERS (and ρ=1 is WORSE, not better — high-tension
energies 7.7/19.6 have tiny sys, common nuisance can't absorb them). The on-plot number uses
the diagonal (ρ=0) treatment. PROPER fix = per-source covariance V_sys = Σ_s δ_s δ_sᵀ (each
source 100% correlated across energies, sources independent) — needs combine_sys.py to EMIT the
signed per-source per-energy deltas (it computes delta_signed at :36/:60 but discards it).
IMPLEMENTED 2026-06-01: (1) combine_sys.py now emits sys_sources_{cb} = signed per-source
contributions s_s=sign(δ)√((δ²−δ_err²)/3) per combined bin (Σ s_s²=yerr_sys², verified exact).
(2) generate_paper_plots.py: build_sys_covariance() assembles V_sys=Σ_s s_s s_sᵀ with
SYS_REGIME_SPLITS (sys_tag 1=|Vz| split into {19.6,27} vs rest; sys_tag 2=Nfit one source);
plot_energy_dep uses V=diag(stat²)+V_sys, falls back to diagonal if sys_sources absent.
RESULT (covariance vs diagonal): TPC 4.85 vs 3.10, TPC excl-7.7 3.63 vs 1.72, EPD 0.33 vs 0.13.
=> covariance makes tension WORSE (correlated sys can't absorb the trend; diagonal was
artificially deflating χ² via self-down-weighting). On-plot annotation now uses covariance;
both printed to stdout. Methodology + regime-split rule: docs/systematic_covariance.md.
NOTE: finals regenerate on next snakemake run (combine_sys.py is a rule input → cascades).
Per-cen (non-combined) sys_sources not emitted yet — trivial extension if the per-cen ratio
plots ever need covariance. Still open: EPD χ²<1 (over-estimated sys, C3).

---

## Done correctly (kept for balance)
- Invariant-mass fits use ExtendedBinnedNLL — correct Poisson binned likelihood (`fit_v2.py:325`).
- Resolution correlation cancels in the main ratio (shared ufloat, `plot_v2_new.py:104–111`).
- `theory.md` flags the f≠0 bias of the simple ratio and gives the unbiased B/A form.
- Invalid-resolution bin masking (res − 2σ ≤ 0, `plot_v2_new.py:157`).
- Parametric bootstrap through the nonlinear ρ inversion is the right propagation idea.
