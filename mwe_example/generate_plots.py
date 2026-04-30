#!/usr/bin/env python3
"""
Minimum Working Example: Generate the three key coalescence plots.

Usage:
    python generate_plots.py [--inputs INPUTS_DIR] [--output OUTPUT_DIR]

Dependencies: numpy, matplotlib, uncertainties, pandas
"""

import argparse
import json
import os
import shutil

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import numpy as np
import pandas as pd
from uncertainties import unumpy, ufloat


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']


def eb(ax, x, y, yerr, color, marker='o', ms=8, capsize=2, mfc=None, zorder=2, lw=1.5, min_stub=0.25):
    mfc_actual = mfc if mfc is not None else color
    ax.plot(x, y, marker=marker, color=color, mfc=mfc_actual, mec=color,
            ms=ms, ls='none', zorder=zorder + 1)
    trans = ax.transData
    y0_px, y1_px = trans.transform([0, ax.get_ylim()[0]])[1], trans.transform([0, ax.get_ylim()[1]])[1]
    px_per_data = abs(y1_px - y0_px) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    r = (ms / 2) * (ax.figure.dpi / 72.0) / px_per_data
    for xi, yi, ei in zip(np.asarray(x), np.asarray(y), np.asarray(yerr)):
        if np.isnan(yi) or np.isnan(ei) or ei <= 0:
            continue
        if ei > (1 + min_stub) * r:
            ax.plot([xi, xi], [yi + r, yi + ei], color=color, lw=lw, zorder=zorder, solid_capstyle='butt')
            ax.plot([xi, xi], [yi - ei, yi - r], color=color, lw=lw, zorder=zorder, solid_capstyle='butt')
            if capsize > 0:
                ax.plot([xi], [yi + ei], marker='_', color=color, ms=capsize * 3, lw=lw, zorder=zorder)
                ax.plot([xi], [yi - ei], marker='_', color=color, ms=capsize * 3, lw=lw, zorder=zorder)


marker_styles = {
    'TPC':  {'marker': '*', 'color': 'C0', 'ms': 10, 'ls': 'none', 'capsize': 2},
    'EPD':  {'marker': 'o', 'color': 'C1', 'ms': 8,  'ls': 'none', 'capsize': 2},
}

# -- Glauber model band (fixed, from theoretical calculation)
GL_CENTRALITY = np.array([2.5, 7.5, 15, 25, 35, 45, 55, 65, 75])
GL_RATIO_DU   = np.array([1.13304, 1.13351, 1.13777, 1.14464, 1.15486,
                          1.17016, 1.19128, 1.22017, 1.26102])
GL_RATIO_ERR  = np.array([1.56853e-08, 4.95918e-08, 9.90949e-08, 1.74393e-07,
                          3.19487e-07, 5.92654e-07, 1.24839e-06, 3.09313e-06,
                          5.11625e-06])
GL_UPPER = GL_RATIO_DU + GL_RATIO_ERR
GL_LOWER = GL_RATIO_DU - GL_RATIO_ERR

ENERGIES = ['7p7GeV', '9p2GeV', '11p5GeV', '14p6GeV', '17p3GeV', '19p6GeV', '27GeV']


def energy_label(e):
    return e.replace('p', '.')


def energy_as_float(e):
    return float(energy_label(e).replace('GeV', ''))


# ============================================================
#  Plot 1: Pion v2 / n_q
# ============================================================
def plot_pion_v2(v2_dir, output_dir):
    files = []
    resfiles = []
    for e in ENERGIES:
        files.append(os.path.join(v2_dir, e, 'v2.csv'))
        resfiles.append(os.path.join(v2_dir, e, 'res.csv'))

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(ncols=4, nrows=2, hspace=0.0, wspace=0.0)
    axs = gs.subplots(sharex='col', sharey='row').flatten()

    for i, (f, fres) in enumerate(zip(files, resfiles)):
        df = pd.read_csv(f)
        df_res = pd.read_csv(fres)
        en_str = energy_label(ENERGIES[i])
        for EP in ['TPC', 'EPD']:
            if EP == 'EPD' and energy_as_float(ENERGIES[i]) < 14.6:
                continue
            res = unumpy.uarray(df_res[f'{EP}_res'].values, df_res[f'{EP}_res_err'].values)
            res_mask = unumpy.nominal_values(res) >= 0.05
            if ENERGIES[i] == '7p7GeV' and EP == 'TPC':
                res_mask[0] = False
            pip_v2 = unumpy.uarray(df[f'piplus_v2_{EP}'].values, df[f'piplus_v2_err_{EP}'].values) / res / 2
            pim_v2 = unumpy.uarray(df[f'piminus_v2_{EP}'].values, df[f'piminus_v2_err_{EP}'].values) / res / 2
            ap_v2  = unumpy.uarray(df[f'antiproton_v2_{EP}'].values, df[f'antiproton_v2_err_{EP}'].values) / res / 3
            x = np.array([75., 65., 55., 45., 35., 25., 15., 7.5, 2.5])
            x = x[res_mask]
            pip_v2 = pip_v2[res_mask]
            pim_v2 = pim_v2[res_mask]
            ap_v2  = ap_v2[res_mask]
            shift_EP = 0.5 if EP == 'TPC' else -0.5
            shift_par = 1.0
            mfc = None if EP == 'TPC' else 'none'
            z = 4 if EP == 'TPC' else 2
            eb(axs[i], x + shift_par + shift_EP, unumpy.nominal_values(pip_v2), unumpy.std_devs(pip_v2),
               color='C0', marker='o', mfc=mfc, zorder=z)
            eb(axs[i], x - shift_par + shift_EP, unumpy.nominal_values(pim_v2), unumpy.std_devs(pim_v2),
               color='C3', marker='^', mfc=mfc, zorder=z)
            eb(axs[i], x + shift_EP, unumpy.nominal_values(ap_v2), unumpy.std_devs(ap_v2),
               color='C2', marker='d', mfc=mfc, zorder=z)
            axs[i].annotate(r'$\sqrt{s_{\text{NN}}}=$' + en_str, xy=(0.15, 0.9),
                            fontsize=15, xycoords='axes fraction', ha='left')
            axs[i].set_xlim(-5, 85)
            axs[i].set_ylim(-0.005, 0.0349)
            lb, rb = axs[i].get_xlim()

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel(r'$\text{Centrality (%)}$', fontsize=15)
    plt.ylabel(r'$v_2/n_q$', fontsize=15, labelpad=20)
    h_auau = axs[7].plot([], [], ' ')[0]
    h_pip_tpc = axs[7].errorbar([], [], yerr=[], fmt='o', capsize=2, ms=8, color='C0')
    h_pip_epd = axs[7].errorbar([], [], yerr=[], fmt='o', capsize=2, ms=8, color='C0', mfc='white')
    h_pim_tpc = axs[7].errorbar([], [], yerr=[], fmt='^', capsize=2, ms=8, color='C3')
    h_pim_epd = axs[7].errorbar([], [], yerr=[], fmt='^', capsize=2, ms=8, color='C3', mfc='white')
    h_pbar_tpc = axs[7].errorbar([], [], yerr=[], fmt='d', capsize=2, ms=8, color='C2')
    h_pbar_epd = axs[7].errorbar([], [], yerr=[], fmt='d', capsize=2, ms=8, color='C2', mfc='white')
    axs[7].tick_params(axis='x', which='both', length=0)
    axs[7].legend(
        [h_auau, (h_pip_tpc, h_pip_epd), (h_pim_tpc, h_pim_epd), (h_pbar_tpc, h_pbar_epd)],
        ['Au+Au', r'$\pi^+$ (TPC/EPD)', r'$\pi^-$ (TPC/EPD)', r'$\bar{p}$ (TPC/EPD)'],
        handler_map={tuple: HandlerTuple(ndivide=None)},
        fontsize=15, frameon=False, loc='center'
    )
    plt.savefig(os.path.join(output_dir, 'pion_v2.pdf'))
    plt.close()
    print('  -> pion_v2.pdf')


# ============================================================
#  Plot 2: Coalescence ratio vs centrality
# ============================================================
def plot_ratio(ratio_dir, output_dir):
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(ncols=4, nrows=2, hspace=0.0, wspace=0.0)
    axs = gs.subplots(sharex='col', sharey='row').flatten()

    for i, energy in enumerate(ENERGIES):
        en_str = energy_label(energy)
        for EP in ['TPC', 'EPD']:
            fpath = os.path.join(ratio_dir, f'{energy}_{EP}.json')
            if not os.path.exists(fpath):
                continue
            with open(fpath, 'r') as f:
                data = json.load(f)
            x = np.array(data['x'])
            if len(x) == 0:
                continue
            ratio = unumpy.uarray(data['y'], data['yerr_stat'])
            err_sys = np.array(data['yerr_sys'])
            shift = 1. if EP == 'TPC' else -1.
            axs[i].errorbar(x + shift, unumpy.nominal_values(ratio), unumpy.std_devs(ratio),
                            **marker_styles[EP])
            for j in range(len(x)):
                axs[i].fill_between(np.array([x[j] - 0.5, x[j] + 0.5]) + shift,
                                    unumpy.nominal_values(ratio)[j] - err_sys[j],
                                    unumpy.nominal_values(ratio)[j] + err_sys[j],
                                    color=marker_styles[EP]['color'], alpha=0.3)
            axs[i].annotate(r'$\sqrt{s_{\text{NN}}}=$' + en_str, xy=(0.15, 0.9),
                            fontsize=15, xycoords='axes fraction', ha='left')
            axs[i].set_xlim(-5, 85)
            lb, rb = axs[i].get_xlim()
            axs[i].set_ylim(0.55, 1.649)
            axs[i].hlines(315 / 276, lb, rb, color='C3', label='315/276', linestyle='--')
            axs[i].hlines(1, lb, rb, color='black', linestyle='--')

        axs[i].fill_between(GL_CENTRALITY, GL_LOWER, GL_UPPER, color='C2', alpha=0.8, label='Glauber')

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel(r'$\text{Centrality (%)}$', fontsize=15)
    plt.ylabel(r'$\frac{v_2^{\pi^-}-\frac{2}{3}v_2^{\bar{p}}}{v_2^{\pi^+}-\frac{2}{3}v_2^{\bar{p}}}$', fontsize=18)
    axs[7].plot([], [], ' ', label='Au+Au')
    axs[7].errorbar([], [], yerr=[], label='TPC Event plane', **marker_styles['TPC'])
    axs[7].errorbar([], [], yerr=[], label='EPD Event plane', **marker_styles['EPD'])
    axs[7].fill_between([], [], [], color='C2', alpha=0.8, label='Glauber d/u')
    axs[7].hlines(999, 0, 85, color='C3', linestyle='--', label='315/276')
    axs[7].legend(fontsize=15, frameon=False, loc='center')

    plt.savefig(os.path.join(output_dir, 'ratio.pdf'))
    # Also write numeric output
    with open(os.path.join(output_dir, 'ratio.txt'), 'w') as out:
        for i, energy in enumerate(ENERGIES):
            en_str = energy_label(energy)
            out.write(f'AuAu, {en_str}\n')
            for EP in ['TPC', 'EPD']:
                out.write(f'{EP}\n')
                fpath = os.path.join(ratio_dir, f'{energy}_{EP}.json')
                if not os.path.exists(fpath):
                    continue
                with open(fpath, 'r') as f:
                    data = json.load(f)
                x = data['x']
                ratio = np.array(data['y'])
                yerr = np.array(data['yerr'])
                for j in range(len(x)):
                    out.write(f'{x[j]:.2f} {ratio[j]:.4f}+-{yerr[j]:.4f}\n')
            out.write('\n')
    plt.close()
    print('  -> ratio.pdf')


# ============================================================
#  Plot 3: Energy dependence (10-40%)
# ============================================================
def plot_energy_dep(ratio_dir, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    ratio_1040 = {'TPC': np.zeros(len(ENERGIES)) * ufloat(0, 0),
                  'EPD': np.zeros(len(ENERGIES)) * ufloat(0, 0)}
    err_sys_1040 = {'TPC': np.zeros(len(ENERGIES)),
                    'EPD': np.zeros(len(ENERGIES))}

    for i, energy in enumerate(ENERGIES):
        for EP in ['TPC', 'EPD']:
            fpath = os.path.join(ratio_dir, f'{energy}_{EP}.json')
            if not os.path.exists(fpath):
                continue
            with open(fpath, 'r') as f:
                data = json.load(f)
            ratio_1040[EP][i] = ufloat(data['y_1040'], data['yerr_stat_1040'])
            err_sys_1040[EP][i] = data['yerr_sys_1040']

    energy_str = [energy_label(e) for e in ENERGIES]
    e_vals = np.array([energy_as_float(e) for e in ENERGIES])
    shift = 0.2

    ax.errorbar(e_vals + shift,
                unumpy.nominal_values(ratio_1040['TPC']),
                unumpy.std_devs(ratio_1040['TPC']),
                fmt='o', label='TPC', capsize=2, ms=8)
    for j in range(len(e_vals)):
        ax.fill_between(np.array([e_vals[j] - 0.5, e_vals[j] + 0.5]) + shift,
                        unumpy.nominal_values(ratio_1040['TPC'])[j] - err_sys_1040['TPC'][j],
                        unumpy.nominal_values(ratio_1040['TPC'])[j] + err_sys_1040['TPC'][j],
                        color='C0', alpha=0.3)

    epd_mask = e_vals >= 14.6
    ax.errorbar((e_vals - shift)[epd_mask],
                unumpy.nominal_values(ratio_1040['EPD'])[epd_mask],
                unumpy.std_devs(ratio_1040['EPD'])[epd_mask],
                fmt='o', label='EPD', capsize=2, ms=8)
    for j in np.where(epd_mask)[0]:
        ax.fill_between(np.array([e_vals[j] - 0.5, e_vals[j] + 0.5]) - shift,
                        unumpy.nominal_values(ratio_1040['EPD'])[j] - err_sys_1040['EPD'][j],
                        unumpy.nominal_values(ratio_1040['EPD'])[j] + err_sys_1040['EPD'][j],
                        color='C1', alpha=0.3)

    expectation = 315 / 276
    chi2_tpc = np.sum((unumpy.nominal_values(ratio_1040['TPC']) - expectation)**2
                       / unumpy.std_devs(ratio_1040['TPC'])**2)
    ndf_tpc = len(ratio_1040['TPC'])
    chi2_epd = np.sum((unumpy.nominal_values(ratio_1040['EPD'])[epd_mask] - expectation)**2
                       / unumpy.std_devs(ratio_1040['EPD'])[epd_mask]**2)
    ndf_epd = int(np.sum(epd_mask))
    ax.annotate(fr'TPC $\chi^2$/ndf = {chi2_tpc:.1f}/{ndf_tpc}',
                xy=(0.05, 0.82), fontsize=13, xycoords='axes fraction')
    ax.annotate(fr'EPD $\chi^2$/ndf = {chi2_epd:.1f}/{ndf_epd}',
                xy=(0.05, 0.74), fontsize=13, xycoords='axes fraction')
    ax.annotate(r'AuAu, 10-40%', xy=(0.05, 0.9), fontsize=15,
                xycoords='axes fraction', ha='left')
    ax.set_xlabel(r'$\sqrt{s_{\text{NN}}}$ (GeV)', fontsize=15)
    ax.set_ylabel(r'$\frac{v_2^{\pi^-}-\frac{2}{3}v_2^{\bar{p}}}{v_2^{\pi^+}-\frac{2}{3}v_2^{\bar{p}}}$', fontsize=15)

    with open(os.path.join(output_dir, 'energy_dep.txt'), 'w') as f:
        for i, e in enumerate(energy_str):
            f.write(f'{e} {unumpy.nominal_values(ratio_1040["TPC"])[i]:.4f} '
                    f'{unumpy.std_devs(ratio_1040["TPC"])[i]:.4f} '
                    f'{unumpy.nominal_values(ratio_1040["EPD"])[i]:.4f} '
                    f'{unumpy.std_devs(ratio_1040["EPD"])[i]:.4f}\n')

    ax.set_xticks(e_vals, labels=e_vals)
    ax.set_xlim(5, 30)
    ax.set_ylim(1.0, 1.3)
    lb, rb = ax.get_xlim()
    ax.hlines(315 / 276, lb, rb, color='C3', linestyle='--', label='315/276')
    ax.legend(fontsize=15)
    plt.savefig(os.path.join(output_dir, 'energy_dep.pdf'))
    plt.close()
    print('  -> energy_dep.pdf')


# ============================================================
#  Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Generate the three key coalescence ratio plots')
    parser.add_argument('--inputs', default='inputs',
                        help='Input data directory (default: inputs)')
    parser.add_argument('--output', default='output',
                        help='Output directory for plots (default: output)')
    args = parser.parse_args()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    v2_dir = os.path.join(args.inputs, 'v2')
    ratio_dir = os.path.join(args.inputs, 'ratio')

    print('Generating pion_v2.pdf ...')
    plot_pion_v2(v2_dir, output_dir)

    print('Generating ratio.pdf ...')
    plot_ratio(ratio_dir, output_dir)

    print('Generating energy_dep.pdf ...')
    plot_energy_dep(ratio_dir, output_dir)

    print(f'\nAll plots saved to {output_dir}/')


if __name__ == '__main__':
    main()
