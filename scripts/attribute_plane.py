#!/usr/bin/env python3
"""Spectator (1st-order) vs participant (2nd-order) plane: ratio attribution.

Side study (see docs/spectator_plane_ratio.md). The coalescence ratio

    R = (v2_pi- - 2/3 v2_pbar) / (v2_pi+ - 2/3 v2_pbar)

increases when computed with the 1st-order spectator plane instead of the default
2nd-order participant plane. R is invariant under any common rescaling of the three
v2's (so the EP resolution cancels), and collapses to two scale-invariant knobs:

    eps = (v2_pi- - v2_pi+) / <v2_pi>        (isospin splitting)   -> "cause 1"
    rho =  v2_pbar          / <v2_pi>        (antiproton)          -> "cause 2"
    R   = 1 + eps / D,   D = 1 - eps/2 - (2/3) rho

This script computes eps, rho, R per centrality bin for both planes (resolution
cancels per bin, so no resolution-correction/merge is needed), attributes Delta R to
the two causes, and writes two QA figures:

  - <out>/ratio_attribution.{pdf,png}   raw R, rho, eps vs centrality (def vs spec)
  - <out>/plane_double_ratio.{pdf,png}  double ratios (effect of the plane switch),
                                         using the exact identity
                                         (R_spec-1)/(R_def-1) = (eps_spec/eps_def)
                                                                * (D_def/D_spec)

Masked energies/centralities are excluded to match plot_v2_new.py:
  - 7.7 / 9.2 / 11.5 GeV are dropped entirely (default-EPD merged ratios fully masked);
  - statically-masked bins (17.3->1, 19.6->1,8, 27->1) are dropped;
  - any bin whose denominator D is consistent with zero (D/sigma_D < 2) on either
    plane is dropped (there R is Cauchy-like and meaningless; cf. ratio_statistics.md).

Usage:
    python scripts/attribute_plane.py
    python scripts/attribute_plane.py --result-dir result --plane spectator_1st \
        --sys-tag 0 --out plots/spectator_1st
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from uncertainties import ufloat

# bin1..bin9 centrality (%) midpoints, matching plot_v2_new.py
XCEN = [75., 65., 55., 45., 35., 25., 15., 7.5, 2.5]

# Energies kept (7.7/9.2/11.5 GeV excluded: default-EPD merged ratios fully masked).
ENERGIES = ['14p6GeV', '17p3GeV', '19p6GeV', '27GeV']
ELABEL = {'14p6GeV': '14.6 GeV', '17p3GeV': '17.3 GeV',
          '19p6GeV': '19.6 GeV', '27GeV': '27 GeV'}

# 1-indexed default-EPD static bin masks (subset of plot_v2_new.py masked_bins).
STATIC_MASK = {'14p6GeV': [], '17p3GeV': [1], '19p6GeV': [1, 8], '27GeV': [1]}

# Coarse centrality classes (final-measurement merge, cf. plot_v2_new.py
# cen_bins_correspondence), 1-indexed bins, with a representative midpoint (%).
CLASSES = {'0-10%': ([8, 9], 5.), '10-40%': ([5, 6, 7], 25.), '40-80%': ([1, 2, 3, 4], 60.)}


def D_of(eps, rho):
    return 1 - eps / 2 - (2. / 3.) * rho


def R_of(eps, rho):
    return 1 + eps / D_of(eps, rho)


def load(result_dir, plane, sys_tag, energy):
    """Return (v2 dataframe, resolution dataframe) for one plane/energy.

    plane=None -> default 2nd-order tree; otherwise the alt-plane subtree.
    """
    base = (f'{result_dir}/sys_tag_{sys_tag}' if plane is None
            else f'{result_dir}/{plane}/sys_tag_{sys_tag}')
    v2 = pd.read_csv(f'{base}/energy_{energy}/v2_eff_corrected.csv')
    res = pd.read_csv(f'{base}/energy_{energy}/v2_eff_corrected_res.csv')
    return v2, res


def eps_rho(df, b):
    """eps, rho (ufloat) for centrality-bin index b, from the EPD columns."""
    pip = ufloat(df['piplus_v2_EPD'][b], df['piplus_v2_err_EPD'][b])
    pim = ufloat(df['piminus_v2_EPD'][b], df['piminus_v2_err_EPD'][b])
    pbar = ufloat(df['antiproton_v2_EPD'][b], df['antiproton_v2_err_EPD'][b])
    avg = (pip + pim) / 2
    return (pim - pip) / avg, pbar / avg


def bad_resolution(res, b):
    """True if the EPD resolution is consistent with zero (mask the bin)."""
    r = res['EPD_res'][b] - 2 * res['EPD_res_err'][b]
    return (not np.isfinite(r)) or r <= 0


def collect(result_dir, plane, sys_tag):
    """Per-(energy, bin) records for unmasked, well-constrained bins."""
    records = {}
    for E in ENERGIES:
        dfd, resd = load(result_dir, None, sys_tag, E)
        dfs, ress = load(result_dir, plane, sys_tag, E)
        rows = []
        for b in range(9):
            if b + 1 in STATIC_MASK[E]:
                continue
            if bad_resolution(resd, b) or bad_resolution(ress, b):
                continue
            ed, rd = eps_rho(dfd, b)
            es, rs = eps_rho(dfs, b)
            Dd, Ds = D_of(ed, rd), D_of(es, rs)
            if Dd.n <= 0 or Ds.n <= 0 or Dd.n / Dd.s < 2 or Ds.n / Ds.s < 2:
                continue
            Rd, Rs = R_of(ed, rd), R_of(es, rs)
            c1 = R_of(es, rd).n - Rd.n   # isospin-only shift
            c2 = R_of(ed, rs).n - Rd.n   # antiproton-only shift
            rows.append(dict(cen=XCEN[b], eps_d=ed, eps_s=es, rho_d=rd, rho_s=rs,
                             D_d=Dd, D_s=Ds, R_d=Rd, R_s=Rs, c1=c1, c2=c2))
        records[E] = rows
    return records


def merged_v2(df, res, bins, sp):
    """Count-weighted mean of resolution-corrected v2_EPD over `bins` (1-indexed).

    Unlike the per-bin case the resolution does NOT cancel across a merge, so each
    bin is resolution-corrected first (sharing one resolution ufloat per bin, so the
    correlation between species is tracked). Mirrors plot_v2_new.py.
    """
    terms, wsum = [], 0.0
    for b1 in bins:
        b = b1 - 1
        r = ufloat(res['EPD_res'][b], res['EPD_res_err'][b])
        v = ufloat(df[f'{sp}_v2_EPD'][b], df[f'{sp}_v2_err_EPD'][b]) / r
        w = float(df[f'{sp}_counts'][b])
        terms.append(v * w); wsum += w
    # start from the first term (not int 0) to avoid a std_dev==0 accumulator
    out = terms[0]
    for t in terms[1:]:
        out = out + t
    return out / wsum


def collect_merged(result_dir, plane, sys_tag):
    """Per-(energy, centrality-class) records, resolution-corrected and merged.

    A class uses only its non-statically-masked, good-resolution constituent bins.
    """
    records = {}
    for E in ENERGIES:
        dfd, resd = load(result_dir, None, sys_tag, E)
        dfs, ress = load(result_dir, plane, sys_tag, E)
        rows = []
        for label, (allbins, cen) in CLASSES.items():
            bins = [b for b in allbins if b not in STATIC_MASK[E]
                    and not bad_resolution(resd, b - 1) and not bad_resolution(ress, b - 1)]
            if not bins:
                continue
            def er(df, res):
                pip = merged_v2(df, res, bins, 'piplus')
                pim = merged_v2(df, res, bins, 'piminus')
                pbar = merged_v2(df, res, bins, 'antiproton')
                avg = (pip + pim) / 2
                return (pim - pip) / avg, pbar / avg
            ed, rd = er(dfd, resd); es, rs = er(dfs, ress)
            Dd, Ds = D_of(ed, rd), D_of(es, rs)
            Rd, Rs = R_of(ed, rd), R_of(es, rs)
            rows.append(dict(cen=cen, label=label, eps_d=ed, eps_s=es, rho_d=rd, rho_s=rs,
                             D_d=Dd, D_s=Ds, R_d=Rd, R_s=Rs,
                             c1=R_of(es, rd).n - Rd.n, c2=R_of(ed, rs).n - Rd.n))
        records[E] = rows
    return records


def print_table(records):
    print('Per-bin attribution (unmasked, D/sigma_D > 2 on both planes).')
    print('c1 = isospin-only R shift, c2 = antiproton-only R shift.\n')
    hdr = (f"{'E':7}{'cen%':>6} | {'eps_def':>15}{'eps_spc':>15} | "
           f"{'rho_def':>13}{'rho_spc':>13} | {'R_def':>6}{'R_spc':>6} | "
           f"{'c1':>7}{'c2':>7} who")
    print(hdr)
    n_iso = n_pbar = 0
    for E, rows in records.items():
        for r in rows:
            who = 'ISO' if abs(r['c1']) > abs(r['c2']) else 'PBAR'
            n_iso += who == 'ISO'
            n_pbar += who == 'PBAR'
            print(f"{E:7}{r['cen']:6.1f} | "
                  f"{r['eps_d'].n:8.4f}+-{r['eps_d'].s:5.4f}"
                  f"{r['eps_s'].n:8.4f}+-{r['eps_s'].s:5.4f} | "
                  f"{r['rho_d'].n:7.3f}+-{r['rho_d'].s:4.3f}"
                  f"{r['rho_s'].n:7.3f}+-{r['rho_s'].s:4.3f} | "
                  f"{r['R_d'].n:6.3f}{r['R_s'].n:6.3f} | "
                  f"{r['c1']:7.3f}{r['c2']:7.3f} {who}")
        print()
    print(f"Tally (central-value |c1| vs |c2|):  ISO = {n_iso}, PBAR = {n_pbar}")
    print("NB: the ISO count is inflated by low-significance eps swings where "
          "eps_def ~ 0; significance-weighted the antiproton (rho) dominates.")


def _wmean(vals):
    """Inverse-variance weighted mean of ufloats -> (mean, err)."""
    n = np.array([v.n for v in vals]); s = np.array([v.s for v in vals])
    w = 1 / s ** 2
    return np.sum(w * n) / np.sum(w), np.sqrt(1 / np.sum(w))


def _null_chi2(vals, null):
    n = np.array([v.n for v in vals]); s = np.array([v.s for v in vals])
    return np.sum(((n - null) / s) ** 2), len(vals)


def print_consistency(records):
    """Test the plane shift against the null (no effect), per energy and pooled.

    The per-bin / per-energy double ratios are mostly consistent with unity at
    current statistics; only the pooled antiproton shift is significant. Errors
    assume the two planes are independent (they share events), and within-energy
    centrality bins are not independent -- so the pooled significance is an
    optimistic upper bound. Interpret accordingly.
    """
    print("\nConsistency with the null (no plane effect):")
    print(f"{'E':8} {'<d_rho>':>17}{'sig':>5} | {'<d_eps>':>17}{'sig':>5} | "
          f"{'chi2/N: Dratio':>15}{'(R-1)ratio':>12}  (vs 1)")
    allr = []; alle = []; allD = []; allRr = []
    rowsets = list(records.items()) + [('ALL', [r for rs in records.values() for r in rs])]
    for E, rows in rowsets:
        if not rows:
            continue
        drho = [r['rho_s'] - r['rho_d'] for r in rows]
        deps = [r['eps_s'] - r['eps_d'] for r in rows]
        Drat = [r['D_d'] / r['D_s'] for r in rows]
        Rrat = [(r['R_s'] - 1) / (r['R_d'] - 1) for r in rows]
        mr, er = _wmean(drho); me, ee = _wmean(deps)
        cD, nD = _null_chi2(Drat, 1.0); cR, nR = _null_chi2(Rrat, 1.0)
        if E == 'ALL':
            print('-' * 78)
        print(f"{E:8} {mr:+8.4f}+-{er:6.4f}{mr/er:5.1f} | "
              f"{me:+8.4f}+-{ee:6.4f}{me/ee:5.1f} | "
              f"{cD/nD:6.1f} ({nD:2d}) {cR/nR:6.1f} ({nR:2d})")
    print("errors assume independent planes/bins -> pooled significance is optimistic.")


def _xaxis(ax, records):
    """Label the x-axis; use class names if these are merged records."""
    merged = any(r.get('label') for rs in records.values() for r in rs)
    if merged:
        ax.set_xticks([c for _, c in CLASSES.values()])
        ax.set_xticklabels([k for k in CLASSES], fontsize=8)
        ax.set_xlabel('Centrality class')
    else:
        ax.set_xlabel('Centrality (%)')


def plot_raw(records, out, suffix=''):
    fig, ax = plt.subplots(3, 4, figsize=(17, 9), sharex=True)
    style = {'def': dict(marker='o', mfc='white', color='C0', label='2nd (participant)'),
             'spc': dict(marker='s', color='C3', label='1st (spectator)')}
    for j, (E, rows) in enumerate(records.items()):
        x = [r['cen'] for r in rows]
        for key, kd, ks in (('R', 'R_d', 'R_s'), ('rho', 'rho_d', 'rho_s'),
                            ('eps', 'eps_d', 'eps_s')):
            i = {'R': 0, 'rho': 1, 'eps': 2}[key]
            for tag, col in (('def', kd), ('spc', ks)):
                ax[i, j].errorbar(x, [r[col].n for r in rows],
                                  [r[col].s for r in rows], ls='none', **style[tag])
        ax[0, j].set_title(ELABEL[E])
        ax[0, j].axhline(315 / 276, ls='--', c='gray', lw=.8)
        ax[1, j].axhline(1.0, ls=':', c='gray', lw=.8)
        ax[2, j].axhline(0.0, ls=':', c='gray', lw=.8)
        _xaxis(ax[2, j], records)
    ax[0, 0].set_ylabel(r'$R=\frac{v_2^{\pi^-}-\frac{2}{3} v_2^{\bar p}}'
                        r'{v_2^{\pi^+}-\frac{2}{3} v_2^{\bar p}}$', fontsize=13)
    ax[1, 0].set_ylabel(r'$\rho = v_2^{\bar p}/\langle v_2^\pi\rangle$', fontsize=13)
    ax[2, 0].set_ylabel(r'$\varepsilon=(v_2^{\pi^-}\!-\!v_2^{\pi^+})/\langle v_2^\pi\rangle$',
                        fontsize=13)
    ax[0, 0].legend(fontsize=9, loc='best')
    fig.suptitle('Spectator (1st) vs participant (2nd) plane: raw R, rho, epsilon'
                 + (' (merged centrality)' if suffix else ''), fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out}/ratio_attribution{suffix}.{ext}', dpi=110)
    plt.close(fig)


def plot_double_ratio(records, out, suffix=''):
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.6), sharey=True)
    for j, (E, rows) in enumerate(records.items()):
        xD = [r['cen'] for r in rows]
        D = [r['D_d'] / r['D_s'] for r in rows]                 # antiproton/denom factor
        Rr = [(r['R_s'] - 1) / (r['R_d'] - 1) for r in rows]    # total
        # isospin factor only where eps_def is well-measured (else the ratio blows up)
        xe = [r['cen'] for r in rows if r['eps_d'].n / r['eps_d'].s > 2]
        ep = [r['eps_s'] / r['eps_d'] for r in rows if r['eps_d'].n / r['eps_d'].s > 2]
        ax[j].errorbar(np.array(xD) - 0.8, [d.n for d in D], [d.s for d in D],
                       ls='none', marker='s', color='C3',
                       label=r'$D_{2nd}/D_{1st}$  (antiproton / denom.)')
        ax[j].errorbar(np.array(xe) + 0.8, [e.n for e in ep], [e.s for e in ep],
                       ls='none', marker='o', mfc='white', color='C0',
                       label=r'$\varepsilon_{1st}/\varepsilon_{2nd}$  (isospin)')
        ax[j].errorbar(xD, [r.n for r in Rr], [r.s for r in Rr], ls='none',
                       marker='^', color='0.3', alpha=.6,
                       label=r'$(R_{1st}\!-\!1)/(R_{2nd}\!-\!1)$  (total)')
        ax[j].axhline(1.0, ls='--', c='gray', lw=.9)
        ax[j].set_title(ELABEL[E])
        _xaxis(ax[j], records)
    ax[0].set_ylabel('Double ratio  (1st-order spectator / 2nd-order)')
    ax[0].set_ylim(0.4, 2.2)
    ax[0].legend(fontsize=8.5, loc='upper left')
    fig.suptitle('Effect of switching to the 1st-order spectator plane'
                 + (' (merged centrality)' if suffix else '')
                 + ':  total R shift (gray) = isospin factor (blue) x '
                 'denominator/antiproton factor (red)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out}/plane_double_ratio{suffix}.{ext}', dpi=115)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--result-dir', default='result')
    ap.add_argument('--plane', default='spectator_1st',
                    help='alt-plane subtree name under result/')
    ap.add_argument('--sys-tag', default='0')
    ap.add_argument('--out', default='plots/spectator_1st')
    ap.add_argument('--merged', action='store_true',
                    help='merge into 0-10/10-40/40-80%% classes (resolution-corrected)')
    args = ap.parse_args()

    if args.merged:
        records = collect_merged(args.result_dir, args.plane, args.sys_tag)
        suffix = '_merged'
    else:
        records = collect(args.result_dir, args.plane, args.sys_tag)
        suffix = ''
    print_table(records)
    print_consistency(records)
    os.makedirs(args.out, exist_ok=True)
    plot_raw(records, args.out, suffix)
    plot_double_ratio(records, args.out, suffix)
    print(f"\nwrote {args.out}/ratio_attribution{suffix}.{{pdf,png}} and "
          f"{args.out}/plane_double_ratio{suffix}.{{pdf,png}}")


if __name__ == '__main__':
    main()
