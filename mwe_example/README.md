# Coalescence Ratio Plots — Minimum Working Example

This MWE generates the three key figures from a pion elliptic flow
coalescence analysis in BES-II Au+Au collisions:

1. **pion_v2.pdf** — `v_2/n_q` vs centrality for π⁺, π⁻, \bar{p}
   (8-panel, all 7 energies, TPC/EPD event planes)
2. **ratio.pdf** — Coalescence ratio
   $(v_2^{\pi^-}-\frac{2}{3}v_2^{\bar{p}})/(v_2^{\pi^+}-\frac{2}{3}v_2^{\bar{p}})$
   vs centrality
3. **energy_dep.pdf** — Same ratio, 10–40% centrality, vs
   $\sqrt{s_{\text{NN}}}$

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run
python generate_plots.py

# 3. Find outputs in output/
ls output/
#   pion_v2.pdf  ratio.pdf  energy_dep.pdf  ratio.txt  energy_dep.txt
```

## Inputs

| Directory | Contents | Source |
|-----------|----------|--------|
| `inputs/v2/{energy}/v2.csv` | Raw v2 per centrality (9 bins × 30 cols) | STAR BES-II ROOT → C++ v2 extraction |
| `inputs/v2/{energy}/res.csv`  | Event-plane resolution per centrality | same |
| `inputs/ratio/{energy}_{EP}.json` | Combined coalescence ratio with stat+sys errors | `combine_sys.py` from full pipeline |

Energies: 7.7, 9.2, 11.5, 14.6, 17.3, 19.6, 27 GeV.

## Dependencies

- numpy, matplotlib, pandas, uncertainties
- Tested with Python 3.10+

## Full pipeline reference

See https://github.com/maxstw23/snakemake_coal for the complete
Snakemake pipeline (ROOT C++ extraction, Lambda v2 fits, systematic
uncertainty combination, and paper-quality figures).
