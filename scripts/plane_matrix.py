#!/usr/bin/env python3
"""2x2 plane matrix at a single energy: {1st,2nd} order x {participant,spectator}.

The default-vs-spectator_1st comparison (attribute_plane.py) changes two things at
once -- the harmonic order (2nd -> 1st) AND the plane type (participant -> spectator).
The part_tag_1 production swaps the plane assignment of the event-plane histograms
(its "1st" histograms are the PARTICIPANT plane, its "2nd" the SPECTATOR plane),
supplying the two missing cells of the matrix:

                 participant                       spectator
    1st order    part_tag_1  (participant_1st)     default file (spectator_1st)
    2nd order    default file (sys_tag_0)          part_tag_1  (spectator_2nd)

With all four cells we can hold the order fixed and vary only the plane type (the
physics of interest -- the spectator plane couples to the initial-state geometry /
neutron skin), or hold the plane type fixed and vary only the order (a harmonic
cross-check that the shift is not just a 1st-vs-2nd artefact).

For each cell we compute the resolution-invariant coalescence ratio

    R = (v2_pi- - 2/3 v2_pbar) / (v2_pi+ - 2/3 v2_pbar) = 1 + eps/D,
    eps = (v2_pi- - v2_pi+)/<v2_pi>,  rho = v2_pbar/<v2_pi>,  D = 1 - eps/2 - 2/3 rho

per centrality bin (same algebra as attribute_plane.py; the resolution cancels per
bin so no merge/resolution-correction is needed for R). Raw pi+/pi-/pbar v2 are also
overlaid, resolution-corrected (v2_EPD / EPD_res) so the four planes are physically
comparable. Outputs under <out>/:
    plane_matrix.pdf      R vs centrality (4 planes) + pure order/plane-type effects
    plane_matrix_v2.pdf   raw resolution-corrected pi+/pi-/pbar v2 vs centrality
    plane_matrix.yaml     per-bin R/eps/rho table for the 4 planes
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
from uncertainties import ufloat

# reuse the exact masking/algebra conventions of the default-vs-spectator study
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attribute_plane import (XCEN, STATIC_MASK, D_of, R_of, eps_rho,
                             bad_resolution, load)

# The four matrix cells: key, label, order, plane-type, result subdir (None=default tree).
PLANES = [
    ('def', '2nd participant (default)', '2nd', 'participant', None),
    ('s1',  '1st spectator',             '1st', 'spectator',   'spectator_1st'),
    ('p1',  '1st participant',           '1st', 'participant', 'participant_1st'),
    ('s2',  '2nd spectator',             '2nd', 'spectator',   'spectator_2nd'),
]
PLANE_KEYS = [p[0] for p in PLANES]
STYLE = {  # color, marker
    'def': ('#1f77b4', 'o'),
    's1':  ('#d62728', 's'),
    'p1':  ('#2ca02c', '^'),
    's2':  ('#9467bd', 'D'),
}


def collect(result_dir, sys_tag, energy):
    """Per-plane, per-bin eps/rho/D/R, each plane keeping ONLY its own good bins.

    A bin is kept whenever it is VALID -- i.e. it passes the static mask AND has a
    resolution distinguishable from zero (bad_resolution). The D-significance is no
    longer a hard cut: bins with D consistent with 0 (D.n<=0 or D.n/D.s<2) are kept
    but flagged weak=True, so they are still plotted (hollow / translucent) rather than
    silently dropped. This is what surfaces the 1st-order PARTICIPANT plane: its raw
    hEPDEP_ew_cos_1 is consistent with zero only in the peripheral/central bins
    (~75, 15, 7.5%) and small-but-significant (~0.003-0.006) in the mid-central bins,
    but ~30-60x weaker than the 1st-order SPECTATOR plane (~0.05-0.24), so every kept
    participant bin is weak. Returns:
        per_plane[key]  = {bin(1-9): dict(cen, eps, rho, D, R, weak)}
        res_pass[key]   = [valid bins (pass static mask + resolution)]
        data[key]       = (v2 df, res df)        # raw, for the v2 overlay
    """
    data = {key: load(result_dir, sub, sys_tag, energy) for key, *_, sub in PLANES}
    per_plane, res_pass = {}, {}
    for key in PLANE_KEYS:
        df, res = data[key]
        good, rok = {}, []
        for b in range(9):
            if b + 1 in STATIC_MASK.get(energy, []):
                continue
            if bad_resolution(res, b):           # resolution consistent with 0 -> invalid
                continue
            rok.append(b + 1)
            eps, rho = eps_rho(df, b)
            D = D_of(eps, rho)
            weak = D.n <= 0 or D.n / D.s < 2      # D consistent with 0 -> R imprecise
            good[b + 1] = dict(cen=XCEN[b], eps=eps, rho=rho, D=D,
                               R=R_of(eps, rho), weak=weak)
        per_plane[key] = good
        res_pass[key] = rok
    return per_plane, res_pass, data


def _errbar(ax, x, uvals, color, marker, label, weak=False):
    """Plot points; weak=True renders hollow + translucent (kept but de-emphasised)."""
    if not x:
        return
    y = np.array([u.n for u in uvals])
    e = np.array([u.s for u in uvals])
    kw = dict(color=color, marker=marker, ms=6, lw=1.4, capsize=2, label=label)
    if weak:
        kw.update(mfc='none', alpha=0.45, lw=1.0, ls='none')
    ax.errorbar(x, y, yerr=e, **kw)


def _diff(per_plane, key_a, key_b):
    """(cen list, R_a - R_b list) over bins WELL-constrained (non-weak) on BOTH planes."""
    common = sorted(b for b in set(per_plane[key_a]) & set(per_plane[key_b])
                    if not per_plane[key_a][b]['weak'] and not per_plane[key_b][b]['weak'])
    cen = [per_plane[key_a][b]['cen'] for b in common]
    d = [per_plane[key_a][b]['R'] - per_plane[key_b][b]['R'] for b in common]
    return cen, d


def plot_R(per_plane, energy, out):
    """R overlay (all planes, own good bins) + bridge decomposition of the shift."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    # Panel 1: R vs centrality. Every valid bin is drawn; strong bins filled, weak
    # bins (D ~ 0, e.g. the whole 1st-participant plane) hollow + translucent.
    ax = axes[0]
    strong_R = []                                # to anchor the y-scale on good points
    for key, label, *_ in PLANES:
        c, m = STYLE[key]
        bins = sorted(per_plane[key])
        strong = [b for b in bins if not per_plane[key][b]['weak']]
        weak = [b for b in bins if per_plane[key][b]['weak']]
        _errbar(ax, [per_plane[key][b]['cen'] for b in strong],
                [per_plane[key][b]['R'] for b in strong], c, m, label)
        _errbar(ax, [per_plane[key][b]['cen'] for b in weak],
                [per_plane[key][b]['R'] for b in weak], c, m,
                None if strong else f'{label} (weak)', weak=True)
        strong_R += [per_plane[key][b]['R'].n for b in strong]
    ax.axhline(1.0, color='gray', ls=':', lw=1)
    if strong_R:                                 # keep the readable physics in frame
        lo, hi = min(strong_R), max(strong_R)
        pad = max(0.05, 0.25 * (hi - lo))
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel('centrality (%)'); ax.set_ylabel('R')
    ax.set_title(f'{energy}: coalescence ratio R'); ax.legend(fontsize=8)
    ax.invert_xaxis()

    # Panel 2: decompose the conflated shift using the 2nd-order spectator (s2) bridge:
    #   R(1st spec) - R(2nd part) = [R(1st spec)-R(2nd spec)] + [R(2nd spec)-R(2nd part)]
    #          total (conflated)        pure ORDER (spectator)     pure PLANE-TYPE (2nd)
    ax = axes[1]
    _errbar(ax, *_diff(per_plane, 's1', 'def'), '#7f7f7f', 'o',
            'total: 1st spec - 2nd part (conflated)')
    _errbar(ax, *_diff(per_plane, 's1', 's2'), '#9467bd', 'D',
            'order: 1st spec - 2nd spec')
    _errbar(ax, *_diff(per_plane, 's2', 'def'), '#1f77b4', 's',
            'plane-type: 2nd spec - 2nd part')
    ax.axhline(0.0, color='gray', ls=':', lw=1)
    ax.set_xlabel('centrality (%)'); ax.set_ylabel(r'$\Delta R$')
    ax.set_title('decomposition of the plane shift'); ax.legend(fontsize=8)
    ax.invert_xaxis()

    fig.tight_layout()
    fig.savefig(f'{out}/plane_matrix.pdf')
    fig.savefig(f'{out}/plane_matrix.png', dpi=140)
    plt.close(fig)


def plot_v2(per_plane, data, energy, out):
    """Resolution-corrected raw pi+/pi-/pbar v2 vs centrality, planes overlaid.

    Each plane uses its own valid bins; weak bins (D ~ 0, the whole 1st-participant
    plane) are drawn hollow/translucent. The 1st-participant points carry large errors
    because dividing by its tiny resolution amplifies the noise."""
    species = [('piplus', r'$\pi^+$'), ('piminus', r'$\pi^-$'),
               ('antiproton', r'$\bar p$')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)

    def _v2(df, res, b1, sp):
        b = b1 - 1
        r = ufloat(res['EPD_res'][b], res['EPD_res_err'][b])
        return ufloat(df[f'{sp}_v2_EPD'][b], df[f'{sp}_v2_err_EPD'][b]) / r

    for ax, (sp, splab) in zip(axes, species):
        strong_v2 = []
        for key, label, *_ in PLANES:
            df, res = data[key]
            c, m = STYLE[key]
            bins = sorted(per_plane[key])
            strong = [b for b in bins if not per_plane[key][b]['weak']]
            weak = [b for b in bins if per_plane[key][b]['weak']]
            _errbar(ax, [per_plane[key][b]['cen'] for b in strong],
                    [_v2(df, res, b, sp) for b in strong], c, m, label)
            _errbar(ax, [per_plane[key][b]['cen'] for b in weak],
                    [_v2(df, res, b, sp) for b in weak], c, m,
                    None if strong else f'{label} (weak)', weak=True)
            strong_v2 += [_v2(df, res, b, sp).n for b in strong]
        if strong_v2:                            # keep well-resolved planes in frame
            lo, hi = min(strong_v2 + [0.0]), max(strong_v2)
            pad = max(0.01, 0.25 * (hi - lo))
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel('centrality (%)'); ax.set_title(f'{splab} $v_2$ (EPD, res-corrected)')
        ax.invert_xaxis()
    axes[0].set_ylabel(r'$v_2$'); axes[0].legend(fontsize=8)
    fig.suptitle(f'{energy}: raw species $v_2$ across the plane matrix')
    fig.tight_layout()
    fig.savefig(f'{out}/plane_matrix_v2.pdf')
    fig.savefig(f'{out}/plane_matrix_v2.png', dpi=140)
    plt.close(fig)


def write_yaml(per_plane, energy, out):
    def u2(u):
        return [float(u.n), float(u.s)]
    payload = {'energy': energy,
               'planes': {k: lbl for k, lbl, *_ in PLANES},
               'good_bins': {k: sorted(per_plane[k]) for k in PLANE_KEYS},
               'R': {}}
    for key in PLANE_KEYS:
        payload['R'][key] = {b: {'cen': per_plane[key][b]['cen'],
                                 'R': u2(per_plane[key][b]['R']),
                                 'eps': u2(per_plane[key][b]['eps']),
                                 'rho': u2(per_plane[key][b]['rho']),
                                 'weak': bool(per_plane[key][b]['weak'])}
                             for b in sorted(per_plane[key])}
    with open(f'{out}/plane_matrix.yaml', 'w') as fh:
        yaml.safe_dump(payload, fh, sort_keys=False, default_flow_style=None)


def print_summary(per_plane, res_pass, energy):
    print(f"\n=== plane matrix, {energy} ===")
    for key, label, *_ in PLANES:
        strong = [b for b in sorted(per_plane[key]) if not per_plane[key][b]['weak']]
        weak = [b for b in sorted(per_plane[key]) if per_plane[key][b]['weak']]
        print(f"  {label:28s}: {len(strong)} strong {strong} + {len(weak)} weak {weak}")
    if not any(not per_plane['p1'][b]['weak'] for b in per_plane['p1']):
        print(f"  NOTE: every valid 1st-participant bin is WEAK (plotted hollow). Its "
              f"resolution is NOT zero everywhere -- hEPDEP_ew_cos_1 passes the res cut "
              f"in bins {res_pass['p1']} (small, ~0.003-0.006) but is ~30-60x weaker than "
              f"the 1st-order spectator plane, so D is consistent with 0 in every bin. "
              f"The decomposition still uses only well-constrained (non-weak) bins.")
    # report the decomposition averaged over common bins
    for label, a, b in [('total  (1st spec - 2nd part)', 's1', 'def'),
                        ('order  (1st spec - 2nd spec)', 's1', 's2'),
                        ('plane  (2nd spec - 2nd part)', 's2', 'def')]:
        _, d = _diff(per_plane, a, b)
        if d:
            mean = sum(x.n for x in d) / len(d)
            print(f"  <dR> {label} = {mean:+.3f}  over {len(d)} bins")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result-dir', default='result')
    ap.add_argument('--sys-tag', default='0')
    ap.add_argument('--energy', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    per_plane, res_pass, data = collect(args.result_dir, args.sys_tag, args.energy)
    os.makedirs(args.out, exist_ok=True)
    if not any(per_plane.values()):
        raise SystemExit(f"no good bins for {args.energy} -- nothing to plot")
    print_summary(per_plane, res_pass, args.energy)
    plot_R(per_plane, args.energy, args.out)
    plot_v2(per_plane, data, args.energy, args.out)
    write_yaml(per_plane, args.energy, args.out)
    print(f"\nwrote {args.out}/plane_matrix.{{pdf,png}}, plane_matrix_v2.{{pdf,png}}, plane_matrix.yaml")


if __name__ == '__main__':
    main()
