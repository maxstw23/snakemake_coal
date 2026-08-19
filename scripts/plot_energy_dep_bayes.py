#!/usr/bin/env python3
"""
Generate energy_dep_bayes.pdf — d/u ratio from posterior quark v2, vs sqrt(s_NN).

    R = ∫(v2^d(p) - v2^prod(p)) dp / ∫(v2^u(p) - v2^prod(p)) dp

integrated over pT/nq ∈ [0.08, 0.60] GeV. For each posterior sample we evaluate
the quark v2 functions on a fine grid, subtract v2^prod, average, then take the ratio.
The coalescence ratio from the pipeline is overlaid for comparison.
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
import yaml

ENERGIES = ['7p7GeV', '9p2GeV', '11p5GeV', '14p6GeV', '17p3GeV', '19p6GeV', '27GeV']

ENERGY_TO_SQRTS = {
    '7p7GeV': 7.7, '9p2GeV': 9.2, '11p5GeV': 11.5, '14p6GeV': 14.6,
    '17p3GeV': 17.3, '19p6GeV': 19.6, '27GeV': 27.0,
}

PTNQ_LO = 0.00
PTNQ_HI = 1.00


def _richards_np(p_grid, a, b, c, d, nu):
    return (a[:, None] / (1 + np.exp(-(p_grid[None, :] - b[:, None]) / c[:, None]))
            ** (1.0 / nu[:, None]) - d[:, None])


def _flat(post, name):
    return post[name].values.flatten()


def eval_quark_v2_posterior(trace, flavor, p_grid):
    post = trace.posterior
    if flavor == 'a':
        return _richards_np(p_grid,
                            _flat(post, 'a_a'), _flat(post, 'b_a'),
                            _flat(post, 'c_a'), _flat(post, 'd_a'), _flat(post, 'nu_a'))
    elif flavor == 'tr':
        return _richards_np(p_grid,
                            _flat(post, 'a_tr'), _flat(post, 'b_tr'),
                            _flat(post, 'c_tr'), _flat(post, 'd_tr'), _flat(post, 'nu_tr'))
    elif flavor in ('u', 'd'):
        fq = _flat(post, 'f')[:, None]
        prod = eval_quark_v2_posterior(trace, 'a', p_grid)
        tr = eval_quark_v2_posterior(trace, 'tr', p_grid)
        return fq * tr + (1.0 - fq) * prod
    else:
        raise ValueError(f'Unknown flavor: {flavor}')


def load_bayes_ratio(energy, trace_dir):
    """Load trace and compute integrated d/u ratio over pT/nq range."""
    trace_path = Path(trace_dir) / energy / 'trace.nc'
    if not trace_path.exists():
        return None
    trace = az.from_netcdf(trace_path)

    p_grid = np.linspace(PTNQ_LO, PTNQ_HI, 400)

    v2_u = eval_quark_v2_posterior(trace, 'u', p_grid)    # (S, N)
    v2_d = eval_quark_v2_posterior(trace, 'd', p_grid)    # (S, N)
    v2_a = eval_quark_v2_posterior(trace, 'a', p_grid)    # (S, N)

    num = np.mean(v2_d - v2_a, axis=1)   # (S,)
    den = np.mean(v2_u - v2_a, axis=1)   # (S,)
    valid = np.abs(den) > 1e-6
    return num[valid] / den[valid]


def load_coal_ratio(energy, coal_dir, cen_range='10-40%'):
    key_map = {'0-10%': 'y_010', '10-40%': 'y_1040', '40-80%': 'y_4080'}
    err_map = {'0-10%': 'yerr_010', '10-40%': 'yerr_1040', '40-80%': 'yerr_4080'}
    ykey = key_map[cen_range]
    yerr_key = err_map[cen_range]

    for ep in ['TPC']:
        yaml_path = Path(coal_dir) / 'sys_tag_0' / f'energy_{energy}' / f'coal_{ep}.yaml'
        if yaml_path.exists():
            with open(yaml_path) as f:
                d = yaml.load(f, Loader=yaml.CLoader)
            y = d.get(ykey, -999)
            if y != -999 and not (isinstance(y, float) and np.isnan(y)):
                e = d.get(yerr_key, 0)
                return y, e
    return np.nan, np.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace-dir', default='plots')
    parser.add_argument('--coal-dir', default='plots')
    parser.add_argument('--out', default='plots/final/energy_dep_bayes.pdf')
    parser.add_argument('--cen-range', default='10-40%')
    args = parser.parse_args()

    sqrts_vals = []
    ratio_meds, ratio_lows, ratio_highs = [], [], []
    coal_vals, coal_errs = [], []

    for energy in ENERGIES:
        ratios = load_bayes_ratio(energy, args.trace_dir)
        if ratios is None:
            continue
        sqrts_vals.append(ENERGY_TO_SQRTS[energy])
        ratio_meds.append(np.median(ratios))
        ratio_lows.append(np.percentile(ratios, 16))
        ratio_highs.append(np.percentile(ratios, 84))

        cy, ce = load_coal_ratio(energy, args.coal_dir, args.cen_range)
        coal_vals.append(cy)
        coal_errs.append(ce)

    if len(sqrts_vals) == 0:
        print('No traces found. Run quark_v2_bayes first.')
        return

    sqrts_vals = np.array(sqrts_vals)
    ratio_meds = np.array(ratio_meds)
    ratio_lows = np.array(ratio_lows)
    ratio_highs = np.array(ratio_highs)
    coal_vals = np.array(coal_vals)
    coal_errs = np.array(coal_errs)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.fill_between(sqrts_vals, ratio_lows, ratio_highs,
                    color='C4', alpha=0.25, label='Bayes (68% CI)')
    ax.plot(sqrts_vals, ratio_meds, 'o-', color='C4', lw=2, ms=8)

    valid_coal = np.isfinite(coal_vals) & (coal_errs > 0)
    if valid_coal.any():
        ax.errorbar(sqrts_vals[valid_coal], coal_vals[valid_coal],
                    yerr=coal_errs[valid_coal], fmt='s', color='C0',
                    ms=8, capsize=3, lw=1.5,
                    label=f'Coalescence ({args.cen_range})')

    ax.axhline(315/276, color='C3', ls='--', lw=1.5, label='315/276')
    ax.axhline(1, color='black', ls='--', lw=1.0, alpha=0.5)

    ax.set_xlabel(r'$\sqrt{s_{\rm NN}}$ (GeV)', fontsize=15)
    ax.set_ylabel(r'$N_d^{\rm tr} / N_u^{\rm tr}$', fontsize=15)
    ax.set_title(rf'Au+Au, {args.cen_range}  |  $p_T/n_q \in [{PTNQ_LO:.2f},\,{PTNQ_HI:.2f}]$ GeV', fontsize=13)
    ax.set_xlim(5, 30)
    ax.set_ylim(0.95, 1.55)
    ax.legend(fontsize=12, frameon=False)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out)
    plt.close(fig)
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
