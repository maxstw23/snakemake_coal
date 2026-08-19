# Pion Coalescence Test in BES-II Au+Au Collisions

A Snakemake pipeline that tests the quark coalescence model by measuring the isospin asymmetry of
transported quarks through pion elliptic flow ($v_2$) in STAR BES-II Au+Au data at
$\sqrt{s_{NN}} = 7.7$–$27$ GeV.

Gold nuclei are neutron-rich ($N/Z \approx 1.5$), so more $d$ than $u$ quarks are transported from
the initial state. In the coalescence picture $\pi^-(d\bar{u})$ should then carry slightly more flow
than $\pi^+(u\bar{d})$. Using the antiproton as a produced-quark baseline:

$$R \equiv \frac{v_2(\pi^-) - \tfrac{2}{3}v_2(\bar{p})}{v_2(\pi^+) - \tfrac{2}{3}v_2(\bar{p})} = \frac{N_d^{tr}}{N_u^{tr}} \approx 1.14$$

This repository contains **only the downstream stage**: it starts from the per-energy `result*.root`
histogram files produced by the upstream STAR analysis maker (`kfparticle_unified`) and ends at
`plots/final/report.pdf`. It does not read STAR picoDSTs.

**The input data ships with this repository.** `data/` holds trimmed `result*.root` files containing
exactly the histograms the pipeline reads, so a fresh clone reproduces every result with no external
data. You only need to go back to the upstream maker if you change what the pipeline reads — see §2.

---

## 1. Requirements

| Component | Why | Notes |
|---|---|---|
| Conda / mamba (miniforge) | Python environment | `environment.yaml` builds env **`lambda_v1`** |
| Docker | CERN ROOT | The flow-extraction rules run `rootproject/root:latest` in a container — no local ROOT install needed |
| ~20 GB disk | Intermediate + output files | `data/` adds 356 MB; the rest is `result/`, `plots/` and `temp/` |
| Local ROOT 6 (**optional**) | `compile_efficiency`, `compile_user_class`, `*_special_sys` rules call a bare `root` | See §5.1 for the Docker-only workaround |
| [ResFlow](https://github.com/) checkout (**optional**) | resonance-decay inversion | Only for `rho_coal_ratio` and the ρ feed-down correction inside `quark_v2_bayes`; expected at `~/Research/ResFlow` |

Docker must be able to bind-mount the repository directory (`-v "$(pwd)":/work`). On WSL2 this means
the repo path has to be visible to Docker Desktop; on native Linux any path works.

```bash
conda env create -f environment.yaml     # creates env "lambda_v1"
conda activate lambda_v1
docker pull rootproject/root:latest
```

`pymc` / `arviz` are only needed for the Bayesian NCQ rules (§7); drop those two lines from
`environment.yaml` if you do not want them.

---

## 2. Input data

### 2.0 What is in the repository, and when you need more

`data/` is **tracked and already populated** with 31 trimmed `result*.root` files (356 MB total,
largest 13.9 MB). Clone and run — nothing to download, nothing to request.

The trimmed files are not a lossy summary of the measurement: they are a *subset of objects*, copied
bit-for-bit. `scripts/slim_result_files.py` keeps every histogram any rule opens and drops the rest
(PID QA maps, Ω/Ξ/φ correlation histograms, recentering maps — ~97% of each raw file by volume, none
of it read here), then rewrites with LZMA. The kept objects are byte-identical: same bin contents,
errors, `fBinEntries`, `fSumw2`, `fEntries` and class names. Running the full pipeline on trimmed vs.
raw inputs produces **1100 of 1120 result CSVs byte-identical, with zero differences outside the
Λ fits** (which are non-deterministic by construction — `--max_refit 500` random restarts).

**You need to regenerate raw files only if you change what is read.** Adding a new histogram, a new
species, a new binning, or a new systematic tag means the object simply is not in the trimmed files.
In that case:

1. Re-run the upstream maker **`kfparticle_unified`** to produce a fresh, untrimmed
   `result<N>_<energy>.root`.
2. Drop it into `data/` under the layout below. The pipeline reads raw and trimmed files identically —
   the trim changes nothing about how the files are opened, so an untrimmed file is a drop-in.
3. Optionally re-trim before committing:
   ```bash
   python3 scripts/slim_result_files.py --out data_slim        # writes a mirrored, trimmed tree
   python3 scripts/slim_result_files.py --out data_slim --check # verify against the source
   ```
   Update the keep-list at the top of that script first if your new histograms are not matched by it,
   or they will be dropped.

Raw untrimmed productions are **not** in git (1.34 GB). Keep them outside the repo; `backup/` is
git-ignored and is the conventional spot.

### 2.1 Layout

The Snakefile discovers files by glob, and for each `(production, energy)` pair it picks the file with
the **highest leading integer** in the filename (`result103_27GeV.root` beats `result9_27GeV.root`).

```
data/
├── result<N>_<energy>.root                       # REQUIRED: default production      [in repo]
├── sys_tag_1/result<N>_sys_tag_1_<energy>.root   # optional: |Vz| < 35 cm            [in repo]
├── sys_tag_2/result<N>_sys_tag_2_<energy>.root   # optional: nHitsFit >= 20          [in repo]
├── sys_tag_4/result<N>_sys_tag_4_<energy>.root   # optional: PID nSigma 2 -> 3       [in repo]
├── part_tag_1/result<N>_part_tag_1_<energy>.root # optional: swapped-plane           [in repo, 14p6GeV only]
└── embedding/<energy>/cen{1..9}.<particle>.root  # git-ignored, normally NOT needed (see §2.4)
```

`<energy>` ∈ `7p7GeV 9p2GeV 11p5GeV 14p6GeV 17p3GeV 19p6GeV 27GeV` (whatever you list in
`config.yaml`), `<particle>` ∈ `piplus piminus kplus kminus proton antiproton`.

Systematic productions are **discovered per energy**: a tag that only ships 3 of the 7 energies is
fine, the systematic band is simply built from the tags that exist for that energy
(`sys_tags_for()` in the Snakefile). With no `sys_tag_*` directories at all the pipeline still runs
and the reported systematic uncertainty is zero.

### 2.2 Histogram contract — `result*.root`

| Object | Name | Type |
|---|---|---|
| $v_2$ vs $p_T$, per centrality | `h{particle}_{EP}_v2_pt_{cen}` | `TProfile` |
| …EPD, harmonic-resolved | `h{particle}_{EP}_v2_pt_{cen}_1st`, `..._2nd` | `TProfile` |
| $v_2$ vs $(y, p_T)$ (2D productions) | `h{particle}_{EP}_v2_y_pt_{cen}` | `TProfile2D` |
| TPC event-plane resolution | `hTPCEP_ew_cos` | `TProfile` |
| EPD resolution, 1st / 2nd order | `hEPDEP_ew_cos_1`, `hEPDEP_ew_cos_2` | `TProfile` |
| TOF matching, all / matched | `hgpTeta_{cen}`, `hgpTeta_TOF_{cen}` | `TH2D` ($p_T$ vs $\eta$) |
| $\Lambda$ invariant mass | `hLambda{,bar}M_pt_{i}_cen_{j}` | `TH1` |
| $\Lambda$ $v_2$ vs $M_{inv}$ | `hLambda{,bar}_{EP}_v2_pt_{i}_cen_{j}` | `TProfile` |

with `{particle}` ∈ `piplus piminus kplus kminus proton antiproton`, `{EP}` ∈ `TPC EPD`,
`{cen}` ∈ 1–9 (1 = most peripheral … 9 = most central).

The Snakefile auto-detects the 2D case at parse time by probing for
`hpiplus_EPD_v2_y_pt_1`; if present, `coal_preprocess.cpp` projects out the rapidity slice
$|y| <$ `eta_cut` before the flow extraction, otherwise the 1D profiles are used directly.

If the `_1st`/`_2nd` suffixed profiles are absent, the code falls back to the unsuffixed
`h{particle}_{EP}_v2_pt_{cen}`, and the alternative-plane analyses (§7) will not be meaningful.

### 2.3 Which upstream production carries which event plane

The Snakefile builds four event-plane "legs" from two productions:

| Leg | Output tree | Production | `EPD_method` |
|---|---|---|---|
| 2nd-order participant (**default**) | `result/`, `plots/` | `data/result*` | `2nd` |
| 1st-order spectator | `result/spectator_1st/` | `data/result*` | `1st` |
| 1st-order participant | `result/participant_1st/` | `data/part_tag_1/` | `1st` |
| 2nd-order spectator | `result/spectator_2nd/` | `data/part_tag_1/` | `2nd` |

In the default production `"1st"` = spectator and `"2nd"` = participant; in the `part_tag_1`
production the two are **swapped**. If you have no `data/part_tag_1/`, the two cross-planes and the
`plane_matrix` target simply drop out of the DAG. See
[docs/spectator_plane_ratio.md](docs/spectator_plane_ratio.md).

### 2.4 Embedding files

Needed by `TPC_eff` to fit the TPC tracking efficiency. Each file must contain `hPt`, `hPtMc`,
`hEta`, `hSelPtEta`, `hSelPtEtaMc`, `hnSigmaPt`.

**Normally you do not need these.** `scripts/{energy}/Efficiency.cpp` — the generated efficiency
lookup table — is committed for **all seven energies**, so `TPC_eff` never fires and the 134 MB
embedding tree is not required to build any default target.

You need embedding data only to *refit* the efficiency: delete `scripts/{energy}/Efficiency.cpp` to
force a refit, or add an energy that has no committed table. Then you need the full
`data/embedding/{energy}/` set — all six species × 9 centrality bins (54 files).

(`config.yaml` ships with `correct_eff: 1` and `generate_paper_plots` always consumes the
efficiency-corrected CSVs, so the efficiency path itself is always active — it is just satisfied by
the committed tables rather than by a refit.)

---

## 3. Quick start

```bash
conda activate lambda_v1

snakemake -n                 # dry run: shows what would be built
snakemake --cores all        # full pipeline
```

Nothing else is required — no `--forcerun`, no manual `touch`. Snakemake rebuilds exactly what your
data changes imply. Expect a few hours for a cold run of all 7 energies × 4 productions.

Useful partial targets:

```bash
# one energy, flow extraction only
snakemake --cores all result/sys_tag_0/energy_27GeV/v2_eff_corrected.csv

# one energy, coalescence-ratio plots
snakemake --cores all plots/sys_tag_0/energy_27GeV/coal_combined.pdf

# the headline report only
snakemake --cores all plots/final/report.pdf

# DAG / rule graph (needs graphviz)
bash create_dag.sh
```

---

## 4. What comes out

```
result/sys_tag_{0,1,2,4}/energy_{energy}/
  v2_eff_corrected.csv          integrated v2 per particle / EP / centrality
  v2_eff_corrected_res.csv      event-plane resolution per centrality
  v2_eff_corrected_cen{1..9}.csv  v2 vs pT per centrality; 500 rows = the first 500 bins
                                  of the source TProfile (0.01 GeV/c wide), no pT column
  fit_{Lambda,Lambdabar}_v2_{TPC,EPD}.csv

plots/sys_tag_{0,1,2,4}/energy_{energy}/
  coal_{TPC,EPD}.pdf/.yaml      R vs centrality, one event plane
  coal_combined.pdf             TPC + EPD overlay
  delta_pion.pdf/.yaml          v2(pi-) - v2(pi+) vs centrality
  {pi,p,k}_delta_v2_{EP}.yaml   per-species Delta-v2 (consumed by the paper plots)
  ratio_pt_scan.pdf             R vs the pT/nq integration window

plots/final/
  energy_{energy}/coal_{TPC,EPD}.yaml   stat + sys + total uncertainty (after combine_sys)
  report.pdf                            <-- the headline deliverable
  ratio.pdf, energy_dep.pdf, pion_v2.pdf, ... (+ .eps/.svg for the paper)
```

`plots/<plane>/…` mirrors this tree for each alternative event plane (§2.2).

`result/` and `plots/` are git-ignored: they are regenerated, never committed.

---

## 5. Known gotchas

### 5.1 No local ROOT installation

`compile_efficiency` and `compile_user_class` invoke a bare `root` (unlike the flow-extraction rules,
which use Docker). Their outputs — `scripts/ExtendedTProfile_cpp.so` and
`scripts/{energy}/Efficiency_cpp.so` — **are committed to git**, so on a fresh clone they normally
count as up to date and the rules never fire. If Snakemake does decide to rebuild them and you have
no local ROOT, build them by hand in the container:

```bash
docker run --rm -v "$(pwd)/scripts":/work -w /work rootproject/root:latest \
    root -b -q -l -e '.L ExtendedTProfile.cpp+'

docker run --rm -v "$(pwd)/scripts/27GeV":/work -w /work rootproject/root:latest \
    root -b -q -l -e '.L Efficiency.cpp+'
```

then re-run Snakemake. (The `.so` is only a declared input; the flow-extraction rules recompile the
sources inside their own container anyway.)

### 5.2 `plots/final/report.pdf` requires isobar inputs

`generate_paper_plots` includes a RuRu/ZrZr isobar panel and therefore hard-requires

```
plots/sys_tag_0/energy_isobar_{Ru,Zr}/pi_delta_v2_{TPC,EPD}.yaml
```

These come from isobar productions that are not part of the Au+Au energy scan. If you don't have
them, either

* add `isobar_Ru, isobar_Zr` to `config.yaml: energies` and supply
  `data/result*_isobar_{Ru,Zr}.root` (the yamls are then written by `plot_v2`), **or**
* drop the panel: remove the `delta_v2_isobar=` input from `rule generate_paper_plots` (and
  `rule generate_paper_plots_plane`) in the `Snakefile`, remove `--input_delta_v2_isobar` from their
  shell commands, and comment out the `plot_isobar_test(...)` call in
  `scripts/generate_paper_plots.py:1226`.

Every other target builds from the Au+Au data alone.

### 5.3 Parallel PyMC jobs race on the ArviZ cache

Running several `quark_v2_bayes` jobs concurrently can crash at import time on a shared ArviZ cache
file. Pre-create it once:

```bash
mkdir -p ~/.cache/arviz && touch ~/.cache/arviz/daily_warning
```

### 5.4 Docker file ownership

Container-written files (the CSVs under `result/`) are owned by the container user. On Linux hosts
add `--user "$(id -u):$(id -g)"` to the `docker run` lines in the `Snakefile` if that causes
permission problems on later runs.

### 5.5 Adding an energy

`scripts/{energy}/Efficiency.cpp` is committed for all seven energies currently in `config.yaml`. Add
a new energy and `TPC_eff` will run for it, requiring the full `data/embedding/{energy}/` set (54
files). Commit the generated `Efficiency.cpp`/`.h` afterwards so nobody else needs the embedding
sample.

---

## 6. Configuration

All analysis parameters live in `config.yaml`:

| Key | Meaning |
|---|---|
| `energies` | Which energies enter the DAG |
| `EPD_method` | EPD harmonic for the default leg (`2nd` = participant plane) |
| `eta_cut` | Rapidity cut applied to particles (0.6) |
| `use_mT` | Use $m_T - m_0$ instead of $p_T$ (0 = off) |
| `correct_eff` | 1 = the TPC-efficiency-corrected CSVs are the primary result |
| `pt_fit_lo/hi` | $p_T$ range used when fitting the TPC efficiency |
| `ptnq_lo/hi` | $p_T/n_q$ window for the integrated $v_2$ (0.08–0.6 GeV/c) |
| `plotting.yrange_*` | Per-energy y-axis limits for the ratio plots |

The `ptnq_lo = 0.08` lower bound is deliberate and validated across energies — see
[docs/ptnq_window_choice.md](docs/ptnq_window_choice.md) before changing it.

Systematic variations are **not** in `config.yaml`; they are the `data/sys_tag_*` productions
(§2), combined by `scripts/combine_sys.py` with a Barlow test and a $1/\sqrt{3}$ (uniform-prior)
factor.

---

## 7. Analyses outside the default `rule all`

Build these explicitly when you want them:

| Target | What it does | Extra requirements |
|---|---|---|
| `snakemake plots/{energy}/trace.nc` | Bayesian NCQ quark-$v_2$ fit (10–40%) | `pymc`, `arviz` |
| `snakemake plots/{energy}/cen_{010,4080}/trace.nc` | same for 0–10% / 40–80% | `pymc`, `arviz` |
| `snakemake plots/final/energy_dep_bayes.pdf` | posterior $R$ vs $\sqrt{s_{NN}}$ | all `trace.nc` |
| `snakemake plots/sys_tag_0/energy_19p6GeV/rho_ratio.pdf` | ρ→ππ feed-down / coalescence cross-check | ResFlow at `~/Research/ResFlow` |
| `snakemake plots/sys_tag_{tag}/energy_isobar/eq1.pdf` | Ru vs Zr isobar comparison | isobar productions |
| `snakemake plots/spectator_check/compare_epd_planes.pdf` | 1st- vs 2nd-order EPD cross-check | hard-coded path in `SPECTATOR_DATA_DIR` — edit it |
| `snakemake plots/coal_report.pdf` | concatenated summary PDF | — |

`scripts/diagnostics/` holds standalone robustness scripts (prior scans, pull tests, pT-window
scans) that are not wired into the DAG; run them directly with `python`.

`mwe_example/` is a self-contained, data-free reproduction of the three headline figures from
exported arrays — useful for checking plot styling without the full pipeline.

---

## 8. Repository layout

```
Snakefile              pipeline definition (all rules)
config.yaml            analysis parameters
environment.yaml       conda environment (env name: lambda_v1)
create_dag.sh          DAG / rulegraph rendering
scripts/
  v2_eff_correction.cpp, v2_no_eff_correction.cpp   flow extraction (ROOT)
  coal_preprocess.cpp, draw_TOF_eff_2.cpp           2D->1D projection, TOF efficiency
  ExtendedTProfile.{cpp,h}                          TProfile subclass with manual Sumw2
  TPC_eff.py                                        efficiency fit -> Efficiency.cpp
  fit_v2.py                                         Lambda v2 from invariant-mass fits
  plot_v2_new.py                                    coalescence ratios + per-energy plots
  combine_sys.py                                    systematic combination (Barlow)
  generate_paper_plots.py                           final report
  quark_v2_bayes.py, rho_coal_ratio.py              Bayesian NCQ fit, rho feed-down
  plane_matrix.py, attribute_plane.py               event-plane decomposition
  slim_result_files.py                              trim result*.root to the pipeline's keep-list
  {energy}/                                         generated + compiled efficiency code
data/    trimmed result*.root inputs (TRACKED); data/embedding/ git-ignored
backup/  git-ignored; raw untrimmed productions live here
result/  CSV outputs (git-ignored)
plots/   PDF/YAML outputs (git-ignored)
logs/    per-rule stdout/stderr
temp/    transient per-rule working dirs
```

Each rule writes `logs/<...>/<rule>.log` and `.err` — check these first when a rule fails, since
the shell commands redirect all output there.

---

## 9. Documentation

- [docs/physics.md](docs/physics.md) — physics motivation, coalescence model, key equations
- [docs/theory.md](docs/theory.md) — resonance-fraction correction, B/A ratio, isobar predictions
- [docs/pipeline.md](docs/pipeline.md) — rule-by-rule description of the DAG
- [docs/scripts.md](docs/scripts.md) — per-script reference
- [docs/data_io.md](docs/data_io.md) — data formats and directory layout
- [docs/ptnq_window_choice.md](docs/ptnq_window_choice.md) — why the $p_T/n_q$ window is what it is
- [docs/systematic_covariance.md](docs/systematic_covariance.md) — correlated-systematics treatment
- [docs/ratio_statistics.md](docs/ratio_statistics.md) — bias/coverage of the ratio estimator
- [docs/stats_review.md](docs/stats_review.md) — statistical-integrity audit and its findings
- [docs/quark_v2_bayes.md](docs/quark_v2_bayes.md) — Bayesian NCQ model
- [docs/rho_feeddown.md](docs/rho_feeddown.md) — ρ→ππ feed-down analysis
- [docs/spectator_plane_ratio.md](docs/spectator_plane_ratio.md) — spectator vs participant plane
