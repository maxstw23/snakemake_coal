# Data Formats and I/O

## Input Data

### Raw Analysis ROOT Files
- **Location**: `data/result*_{energy}.root` (default), `data/sys_tag_{1,2}/result*_{energy}.root` (systematic variations)
- **Content**: 2D or 1D histograms of $v_2$ vs kinematic variables for each particle species, centrality, and event plane method
- **Naming convention**: Histograms follow pattern `h{particle}_{EP}_v2_{var}_{centrality}` where:
  - `{particle}`: piplus, piminus, kplus, kminus, proton, antiproton
  - `{EP}`: TPC, EPD
  - `{var}`: pt (1D), y_pt (2D)
  - `{centrality}`: 1-9

### Embedding Files
- **Location**: `data/embedding/{energy}/cen{1-9}.{particle}.root`
- **Content**: Matched/total track distributions from Monte Carlo embedding for efficiency calculation
- **Particles**: piplus, piminus, proton, antiproton, kplus, kminus
- **9 centrality bins** per energy per particle

## Intermediate Data

### Preprocessed ROOT Files
- **Created by**: `coal_preprocess.cpp` (called internally)
- **Location**: Temporary, deleted after use
- **Content**: 1D projections of 2D histograms with rapidity cut applied

### TOF Efficiency
- **Created by**: `draw_TOF_eff_2.cpp`
- **File**: `TOFEfficiency.root` (temporary)
- **Content**: TEfficiency objects for TOF acceptance x efficiency

### Compiled Libraries
- **Files**: `scripts/{energy}/Efficiency_cpp.so`, `scripts/ExtendedTProfile_cpp.so`
- **Created by**: ROOT ACLiC compiler via `compile_efficiency` and `compile_user_class` rules

## Output Data

### CSV Files (`result/`)

#### `v2_noeff_corrected.csv` / `v2_eff_corrected.csv`
Per-centrality v2 values. Columns include:
- Centrality bin index
- $v_2$ and error for each particle ($\pi^+$, $\pi^-$, $K^+$, $K^-$, $p$, $\bar{p}$)
- Both TPC and EPD event plane results

#### `v2_*_res.csv`
Event plane resolution per centrality bin for TPC and EPD methods.

#### `fit_{particle}_v2_{EP}.csv`
Lambda/Lambdabar $v_2$ extracted from invariant mass fits, per centrality and rapidity bin.

### YAML Files (`plots/`)

#### `coal_TPC.yaml` / `coal_EPD.yaml`
Coalescence ratio data with statistical uncertainties, structured for downstream combination.

#### `delta_pion.yaml`
Pion $\Delta v_2$ data per centrality.

#### `lambda_delta_v2_{TPC,EPD}.yaml`
Lambda $\Delta v_2$ data per centrality.

#### `plots/final/energy_{energy}/coal_{EP_method}.yaml`
Combined results with statistical, systematic, and total uncertainties.

### PDF Plots (`plots/`)

| File | Content |
|------|---------|
| `coal_TPC.pdf` | Coalescence ratio $N_d^{tr}/N_u^{tr}$ vs centrality (TPC EP) |
| `coal_EPD.pdf` | Coalescence ratio vs centrality (EPD EP) |
| `coal_combined.pdf` | TPC + EPD overlay |
| `delta_pion.pdf` | $v_2(\pi^-) - v_2(\pi^+)$ vs centrality |
| `coal_lambda_TPC.pdf` | Lambda $v_2$ comparison (TPC EP) |
| `coal_lambda_EPD.pdf` | Lambda $v_2$ comparison (EPD EP) |
| `coal_all.pdf` | Energy-scan overlay (10-40% centrality) |
| `coal_peri.pdf` | Energy-scan overlay (40-80% centrality) |
| `plots/final/report.pdf` | Complete publication-quality report |
| `plots/coal_report.pdf` | Concatenated summary PDF |

## Directory Structure

```
snakemake_coal/
  config.yaml              # Analysis parameters
  Snakefile                 # Workflow definition
  environment.yaml         # Conda environment spec
  data/                    # Input ROOT files (not tracked in git)
    embedding/{energy}/    # Efficiency embedding files
    sys_tag_{1,2}/         # Systematic variation data
  scripts/                 # All analysis code
    {energy}/              # Per-energy compiled efficiency
  result/                  # CSV output
    sys_tag_{0,1,2}/energy_{energy}/
    special_sys_tag_{N}/energy_{energy}/
  plots/                   # PDF and YAML output
    sys_tag_{0,1,2}/energy_{energy}/
    special_sys_tag_{N}/energy_{energy}/
    final/                 # Combined results with systematics
    QA/                    # Quality assurance plots from TPC_eff
  temp/                    # Working directories (transient)
  logs/                    # Snakemake execution logs
  tables/                  # LaTeX tables
```
