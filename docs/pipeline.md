# Snakemake Pipeline Architecture

## DAG Overview

The pipeline has the following dependency structure:

```
data/*.root ─┐
             ├─> TPC_eff ──> compile_efficiency ──┐
             │                                     ├─> v2_eff_correction ──> plot_v2 ──┐
             ├─> compile_user_class ───────────────┤                                   │
             │                                     ├─> v2_no_eff_correction ──> plot_v2 ┤
             │                                     │                                    ├─> combine_sys ──> generate_paper_plots
             │                                     └─> v2_no_eff_correction_special_sys  │                           │
             │                                                 └──> plot_v2_special_sys ─┘                           │
             └─> fit_lambda ──────────────────────────────────> plot_v2 ─────────────────┘                           │
                                                                                                                     v
                                                               plot_all ──> generate_report        plots/final/report.pdf
```

## Wildcards

The pipeline is parametrized by three wildcards:

| Wildcard | Values | Purpose |
|----------|--------|---------|
| `{energy}` | 7p7GeV, 9p2GeV, 11p5GeV, 14p6GeV, 17p3GeV, 19p6GeV, 27GeV | Collision energy |
| `{sys_tag}` | 0 (default), 1, 2 | Systematic variation index |
| `{EP_method}` | TPC, EPD | Event plane detector |

Additional wildcards for Lambda fitting: `{particle}` (Lambda, Lambdabar), `{EP}` (TPC, EPD).

## Rule-by-Rule Description

### Stage 1: Detector Calibration

#### `TPC_eff`
- **Script**: `scripts/TPC_eff.py`
- **Input**: Embedding ROOT files (`data/embedding/{energy}/cen{1-9}.{particle}.root`)
- **Output**: `scripts/{energy}/Efficiency.cpp`, `scripts/{energy}/Efficiency.h`
- **What it does**: Fits the TPC tracking efficiency as a function of $p_T$ and $\eta$ for each particle species and centrality bin using parametric functions. The fit is performed with iminuit. Output is a C++ source file containing hardcoded efficiency lookup tables.
- **Parameters**: `pt_fit_lo`, `pt_fit_hi` (from config), `eta_cut`

#### `compile_efficiency`
- **Input**: `scripts/{energy}/Efficiency.cpp/.h`
- **Output**: `scripts/{energy}/Efficiency_cpp.so`
- **What it does**: Compiles the generated Efficiency code into a ROOT shared library using ACLiC (`.L Efficiency.cpp+`).

#### `compile_user_class`
- **Input**: `scripts/ExtendedTProfile.cpp/.h`
- **Output**: `scripts/ExtendedTProfile_cpp.so`
- **What it does**: Compiles the ExtendedTProfile ROOT class (a TProfile subclass with manual `SetSumw2()` support).

### Stage 2: Flow Extraction

#### `v2_eff_correction`
- **Script**: `scripts/v2_eff_correction.cpp` (run via ROOT)
- **Input**: Data ROOT file, compiled Efficiency and ExtendedTProfile libraries
- **Output**: `result/sys_tag_{sys_tag}/energy_{energy}/v2_eff_corrected.csv`, `..._res.csv`
- **What it does**:
  1. Preprocesses 2D histograms to 1D (if the data contains 2D hists)
  2. Computes TOF efficiency from embedding
  3. Calculates $v_2$ for $\pi^\pm$, $K^\pm$, $p/\bar{p}$ with TPC efficiency weighting
  4. Outputs CSV with $v_2$ values per centrality bin + resolution CSV
- **Execution**: Copies scripts to `scripts/{energy}/` directory and runs ROOT in batch mode

#### `v2_no_eff_correction`
- **Script**: `scripts/v2_no_eff_correction.cpp`
- **Input**: Same as above minus Efficiency library
- **Output**: `result/sys_tag_{sys_tag}/energy_{energy}/v2_noeff_corrected.csv`, `..._res.csv`
- **What it does**: Same flow extraction but without efficiency weighting. This is the primary analysis path (efficiency correction is for cross-check only).
- **Execution**: Copies scripts to `temp/sys_tag_{sys_tag}/energy_{energy}/` and runs ROOT

#### `v2_no_eff_correction_special_sys`
- **Script**: Same as `v2_no_eff_correction`
- **Output**: `result/special_sys_tag_{sys_tag}/energy_{energy}/v2_noeff_corrected.csv`
- **What it does**: Runs the flow extraction with special systematic parameter variations (e.g., different nHitsFit cuts passed as extra argument). Uses default data files (sys_tag_0).

#### `fit_lambda`
- **Script**: `scripts/fit_v2.py`
- **Input**: Raw ROOT data
- **Output**: `result/sys_tag_{sys_tag}/energy_{energy}/fit_{particle}_v2_{EP}.csv`
- **What it does**: Extracts $\Lambda$/$\bar{\Lambda}$ $v_2$ from invariant mass fits. The $v_2$ vs $M_{inv}$ method uses a double-Gaussian signal + polynomial background to separate signal $v_2$ from combinatorial background. Fits are done in bins of $p_T$ and rapidity.

### Stage 3: Plotting and Systematic Combination

#### `plot_v2`
- **Script**: `scripts/plot_v2_new.py`
- **Input**: v2 CSV files (with and without efficiency), Lambda fit CSVs, resolution CSV
- **Output**: Multiple PDFs and YAMLs per energy:
  - `coal_TPC.pdf/.yaml` - Coalescence ratio with TPC event plane
  - `coal_EPD.pdf/.yaml` - Coalescence ratio with EPD event plane
  - `coal_combined.pdf` - Combined TPC+EPD plot
  - `delta_pion.pdf/.yaml` - Pion $\Delta v_2$ plot
  - `coal_lambda_{TPC,EPD}.pdf` - Lambda comparison plots
  - `lambda_delta_v2_{TPC,EPD}.yaml` - Lambda $\Delta v_2$ data
- **What it does**: Corrects $v_2$ by event plane resolution, merges centrality bins where needed, computes coalescence ratios, and generates publication-style plots.

#### `plot_v2_special_sys`
- Same as `plot_v2` but for special systematic variations.

#### `combine_sys`
- **Script**: `scripts/combine_sys.py`
- **Input**: YAML configs from sys_tag_0 (default) + sys_tag_1, sys_tag_2 (variations)
- **Output**: `plots/final/energy_{energy}/coal_{EP_method}.yaml`
- **What it does**: Computes systematic uncertainty by taking quadrature sum of deviations from default, applies Barlow test, divides by $\sqrt{3}$, and combines with statistical uncertainty.

#### `plot_all`
- **Script**: `scripts/plot_all.py`
- **Input**: All default v2 CSVs across energies
- **Output**: `plots/coal_all.pdf`, `plots/coal_peri.pdf`
- **What it does**: Creates energy-dependent overlay of coalescence ratios for centrality 10-40% and 40-80%.

#### `plot_isobar`
- **Script**: `scripts/plot_v2_isobar.py`
- **Input**: Ru and Zr v2 CSVs
- **Output**: `plots/sys_tag_{sys_tag}/energy_isobar/eq1.pdf`, `eq2.pdf`
- **What it does**: Compares coalescence ratios between Ru and Zr isobars. Currently not in the default `rule all` target.

### Stage 4: Final Output

#### `generate_paper_plots`
- **Script**: `scripts/generate_paper_plots.py`
- **Input**: All resolution CSVs, combined YAML configs, delta_v2 YAMLs, v2 CSVs
- **Output**: `plots/final/report.pdf`
- **What it does**: Assembles all results into publication-quality figures with consistent formatting.

#### `generate_report`
- **Input**: All `coal_combined.pdf` + `coal_all.pdf` + `coal_peri.pdf`
- **Output**: `plots/coal_report.pdf`
- **What it does**: Concatenates PDFs using PyMuPDF (`fitz`).

## Default Targets (`rule all`)

```python
rule all:
    input: 'plots/final/report.pdf',
           expand('plots/sys_tag_0/energy_{energy}/delta_pion.pdf', energy=energies),
           expand('plots/sys_tag_0/energy_{energy}/coal_combined.pdf', energy=energies),
           expand('result/sys_tag_0/energy_{energy}/v2_noeff_corrected.csv', energy=energies)
```

## Running the Pipeline

```bash
# Full pipeline
snakemake --cores <N>

# Specific energy
snakemake result/sys_tag_0/energy_27GeV/v2_noeff_corrected.csv

# Dry run
snakemake -n

# Generate DAG visualization
bash create_dag.sh
```
