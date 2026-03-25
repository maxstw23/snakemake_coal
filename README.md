# Pion Coalescence Test in BES-II Au+Au Collisions

A Snakemake pipeline that tests the quark coalescence model by measuring the isospin asymmetry of transported quarks through pion elliptic flow ($v_2$) in STAR BES-II Au+Au data at $\sqrt{s_{NN}} = 7.7$--$27$ GeV.

## Physics

Gold nuclei are neutron-rich ($N/Z \approx 1.5$), so more $d$ quarks than $u$ quarks are transported from the initial state. In the coalescence picture, $\pi^-(d\bar{u})$ should carry slightly more flow than $\pi^+(u\bar{d})$. Using the antiproton as a produced-quark baseline:

$$\frac{v_2(\pi^-) - \frac{2}{3}v_2(\bar{p})}{v_2(\pi^+) - \frac{2}{3}v_2(\bar{p})} = \frac{N_d^{tr}}{N_u^{tr}} \approx 1.14$$

A measurement consistent with 1.14 validates that coalescence preserves initial-state isospin information.

## Pipeline

```
data/*.root
  ├─> TPC_eff ──> compile_efficiency ──> v2_eff_correction ──> plot_v2 ──┐
  ├─> compile_user_class ──────────────> v2_no_eff_correction ──> plot_v2 ┤
  └─> fit_lambda ──────────────────────────────────────> plot_v2 ─────────┘
                                                                          │
                                                    combine_sys ──> generate_paper_plots
                                                                          │
                                                                          v
                                                              plots/final/report.pdf
```

Key stages:
1. **Detector calibration** -- TPC efficiency from embedding, compiled as ROOT shared libraries
2. **Flow extraction** -- $v_2$ for $\pi^\pm$, $K^\pm$, $p/\bar{p}$ via C++ ROOT macros; $\Lambda/\bar{\Lambda}$ via invariant mass fits
3. **Coalescence ratio** -- Compute $N_d^{tr}/N_u^{tr}$ per centrality bin, combine systematic uncertainties (Barlow test)
4. **Paper plots** -- Publication-quality figures assembled into `plots/final/report.pdf`

## Prerequisites

- Python 3 with numpy, matplotlib, uproot, iminuit, PyMuPDF (`fitz`)
- ROOT 6 with ACLiC (for compiling C++ macros)
- Snakemake
- Conda environment: `coal` (via miniforge3)

## Usage

```bash
# Activate environment
conda activate coal

# Full pipeline
snakemake --cores <N>

# Dry run
snakemake -n

# Specific energy
snakemake result/sys_tag_0/energy_27GeV/v2_noeff_corrected.csv

# DAG visualization
bash create_dag.sh
```

## Configuration

All analysis parameters are in `config.yaml`:
- **Energies**: 7.7, 9.2, 11.5, 14.6, 17.3, 19.6, 27 GeV
- **Event planes**: TPC and EPD (2nd harmonic)
- **Rapidity cut**: $|\eta| < 0.6$
- **$p_T/n_q$ integration**: 0.08--0.6 GeV/c
- **Systematics**: 2 regular variations (vertex $z$ cut, $N_\text{fit}$ cut)

## Directory Structure

```
├── Snakefile              # Pipeline definition
├── config.yaml            # Analysis parameters
├── scripts/               # C++ ROOT macros and Python scripts
│   ├── v2_*correction.cpp # Flow extraction
│   ├── TPC_eff.py         # Efficiency fitting
│   ├── fit_v2.py          # Lambda invariant mass fits
│   ├── plot_v2_new.py     # Coalescence ratio plots
│   ├── combine_sys.py     # Systematic uncertainty combination
│   └── generate_paper_plots.py
├── data/                  # Input ROOT files (not tracked)
├── result/                # Intermediate CSVs
├── plots/                 # Output plots and YAMLs
└── docs/                  # Detailed documentation
```

## Documentation

- [docs/physics.md](docs/physics.md) -- Physics motivation and key equations
- [docs/pipeline.md](docs/pipeline.md) -- Snakemake DAG and rule descriptions
- [docs/scripts.md](docs/scripts.md) -- Script reference
- [docs/data_io.md](docs/data_io.md) -- Data formats and I/O specifications
- [docs/theory.md](docs/theory.md) -- Extended coalescence derivations
