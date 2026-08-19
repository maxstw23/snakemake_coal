#!/usr/bin/env python3
"""
Generate transported_excess_vs_energy.pdf — the average transported quark v2
excess Δv2 = ⟨v2^tr − v2^prod⟩ over pT/n_q ∈ [0.2, 0.8] GeV/c, vs sqrt(s_NN).

For each energy with a saved trace.nc, evaluate v2^tr(p) and v2^prod(p) on a
fine grid for every posterior sample, take the per-sample mean of the difference
over the window, then summarize as median + 68% credible interval.
"""

import argparse
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import arviz as az
from pathlib import Path

ENERGIES = ['7p7GeV', '9p2GeV', '11p5GeV', '14p6GeV', '17p3GeV', '19p6GeV', '27GeV']

ENERGY_TO_SQRTS = {
    '7p7GeV': 7.7, '9p2GeV': 9.2, '11p5GeV': 11.5, '14p6GeV': 14.6,
    '17p3GeV': 17.3, '19p6GeV': 19.6, '27GeV': 27.0,
}

# (label, subdir-suffix, color, marker, x-offset on log axis)
# subdir is appended to plots/{energy}/, empty string ⇒ trace.nc lives directly there.
CENTRALITIES = [
    ('0-10%',  'cen_010',  'C0', 's',  0.97),
    ('10-40%', '',         'crimson', 'o', 1.00),
    ('40-80%', 'cen_4080', 'C2', '^',  1.03),
]

PTNQ_LO = 0.20
PTNQ_HI = 0.80
N_GRID  = 400


def _richards_np(p_grid, a, b, c, d, nu):
    return (a[:, None] / (1 + np.exp(-(p_grid[None, :] - b[:, None]) / c[:, None]))
            ** (1.0 / nu[:, None]) - d[:, None])


def _flat(post, name):
    return post[name].values.flatten()


def excess_samples(trace_path, p_grid):
    """Return (S,) array of per-sample mean v2^tr − v2^prod over p_grid."""
    trace = az.from_netcdf(trace_path)
    post = trace.posterior
    tr = _richards_np(p_grid,
                      _flat(post, 'a_tr'), _flat(post, 'b_tr'),
                      _flat(post, 'c_tr'), _flat(post, 'd_tr'), _flat(post, 'nu_tr'))
    pr = _richards_np(p_grid,
                      _flat(post, 'a_a'), _flat(post, 'b_a'),
                      _flat(post, 'c_a'), _flat(post, 'd_a'), _flat(post, 'nu_a'))
    return (tr - pr).mean(axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace-dir', default='plots')
    parser.add_argument('--out', default='plots/final/transported_excess_vs_energy.pdf')
    parser.add_argument('--csv-out', default='plots/final/transported_excess_vs_energy.csv')
    args = parser.parse_args()

    p_grid = np.linspace(PTNQ_LO, PTNQ_HI, N_GRID)

    sqrts_vals, meds, los, his = [], [], [], []
    rows = []
    for energy in ENERGIES:
        trace_path = Path(args.trace_dir) / energy / 'trace.nc'
        if not trace_path.exists():
            print(f'  {energy:8s}: trace not found, skipping')
            continue
        s = excess_samples(trace_path, p_grid)
        med = float(np.median(s))
        lo  = float(np.percentile(s, 16))
        hi  = float(np.percentile(s, 84))
        sqrts_vals.append(ENERGY_TO_SQRTS[energy])
        meds.append(med); los.append(lo); his.append(hi)
        rows.append((energy, ENERGY_TO_SQRTS[energy], med, lo, hi))
        print(f'  {energy:8s}: <Δv2> = {med:+.4f}  [{lo:+.4f}, {hi:+.4f}]')

    if not sqrts_vals:
        print('No traces found. Run quark_v2_bayes first.')
        return

    sqrts_vals = np.array(sqrts_vals)
    meds = np.array(meds); los = np.array(los); his = np.array(his)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.errorbar(sqrts_vals, meds,
                yerr=[meds - los, his - meds],
                fmt='o', color='crimson', ms=8, capsize=4, lw=1.5,
                label='Bayes posterior (68% CI)')
    ax.axhline(0, color='k', lw=0.6, ls='--', alpha=0.6)

    ax.set_xscale('log')
    ax.set_xlabel(r'$\sqrt{s_{\rm NN}}$ (GeV)', fontsize=14)
    ax.set_ylabel(r'$\langle v_2^{\rm tr} - v_2^{\rm prod}\rangle$', fontsize=14)
    ax.set_title(rf'Au+Au, 10–40%  |  $p_T/n_q \in [{PTNQ_LO:.2f},\,{PTNQ_HI:.2f}]$ GeV/$c$',
                 fontsize=12)
    ax.set_xticks(sqrts_vals)
    ax.set_xticklabels([f'{s:g}' for s in sqrts_vals])
    ax.minorticks_off()
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11, frameon=False)

    fig.tight_layout()
    fig.savefig(args.out)
    plt.close(fig)
    print(f'\nSaved {args.out}')

    if args.csv_out:
        os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
        with open(args.csv_out, 'w') as f:
            f.write('energy,sqrts_GeV,delta_v2_med,delta_v2_lo68,delta_v2_hi68\n')
            for e, s, m, lo, hi in rows:
                f.write(f'{e},{s},{m:.6f},{lo:.6f},{hi:.6f}\n')
        print(f'Saved {args.csv_out}')


if __name__ == '__main__':
    main()
