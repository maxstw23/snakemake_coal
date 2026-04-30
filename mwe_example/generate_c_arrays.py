#!/usr/bin/env python3
"""
Generate C-style array declarations from MWE data for ROOT scripts.

Output: pion_v2_data.txt, ratio_data.txt, energy_dep_data.txt
"""

import json
import math
import os

import numpy as np
import pandas as pd

INPUTS = os.path.join(os.path.dirname(__file__), 'inputs')
OUTPUT = os.path.join(os.path.dirname(__file__), 'output_c_arrays')
ENERGIES = ['7p7GeV', '9p2GeV', '11p5GeV', '14p6GeV', '17p3GeV', '19p6GeV', '27GeV']
CENTRALITY = [75., 65., 55., 45., 35., 25., 15., 7.5, 2.5]


def energy_label(e):
    return e.replace('p', '.')


def nq(particle):
    return 2 if particle in ('piplus', 'piminus') else 3


def propagate_v2nq(v2, v2_err, res, res_err, nq):
    """v2/nq = v2 / res / nq with error propagation."""
    val = np.full_like(v2, np.nan, dtype=float)
    err = np.full_like(v2, np.nan, dtype=float)
    for i in range(len(v2)):
        if np.isnan(v2[i]) or np.isnan(res[i]) or res[i] <= 0 or nq <= 0:
            continue
        v = v2[i] / res[i] / nq
        rel2 = (v2_err[i] / v2[i])**2 + (res_err[i] / res[i])**2
        val[i] = v
        err[i] = abs(v) * math.sqrt(rel2)
    return val, err


def format_array(arr, fmt='%.6e'):
    parts = ', '.join(fmt % v if not np.isnan(v) else 'NAN' for v in arr)
    return '{' + parts + '}'


# ============================================================
#  1. pion_v2_data.txt — v2/nq for all energies, EP, particles
# ============================================================
def gen_pion_v2():
    lines = []
    lines.append('// ============================================================')
    lines.append('// pion_v2.pdf — v2/n_q vs centrality')
    lines.append('// ============================================================')
    lines.append('')

    for energy in ENERGIES:
        en = energy_label(energy)
        lines.append(f'// sqrt(s) = {en} {"=" * 50}')

        v2_path = os.path.join(INPUTS, 'v2', energy, 'v2.csv')
        res_path = os.path.join(INPUTS, 'v2', energy, 'res.csv')
        df = pd.read_csv(v2_path)
        df_res = pd.read_csv(res_path)

        for EP in ['TPC', 'EPD']:
            epd_threshold = 14.6
            epd_energy = float(en.replace('GeV', ''))
            if EP == 'EPD' and epd_energy < epd_threshold:
                lines.append(f'')
                lines.append(f'// --- {EP} (no data) ---')
                lines.append(f'int n_{energy}_{EP} = 0;')
                continue

            res = df_res[f'{EP}_res'].values
            res_err = df_res[f'{EP}_res_err'].values

            # Build mask of valid bins
            mask = res >= 0.05
            if energy == '7p7GeV' and EP == 'TPC':
                mask[0] = False
            mask &= ~np.isnan(res)

            # Apply mask to centrality
            x = np.array(CENTRALITY)[mask]
            n = len(x)

            lines.append(f'')
            lines.append(f'// --- {EP} ({n} points) ---')
            lines.append(f'int n_{energy}_{EP} = {n};')
            lines.append(f'double centrality_{energy}_{EP}[{n}] = {format_array(x, "%.1f")};')

            for particle in ['piplus', 'piminus', 'antiproton']:
                col_v2 = f'{particle}_v2_{EP}'
                col_err = f'{particle}_v2_err_{EP}'
                v2 = df[col_v2].values
                v2_err = df[col_err].values

                val, err = propagate_v2nq(v2, v2_err, res, res_err, nq(particle))

                # Apply mask
                val = val[mask]
                err = err[mask]

                arr_val = format_array(val)
                arr_err = format_array(err)
                lines.append(f'double {particle}_v2nq_{energy}_{EP}[{n}] = {arr_val};')
                lines.append(f'double {particle}_v2nq_err_{energy}_{EP}[{n}] = {arr_err};')

        lines.append('')

    os.makedirs(OUTPUT, exist_ok=True)
    with open(os.path.join(OUTPUT, 'pion_v2_data.txt'), 'w') as f:
        f.write('\n'.join(lines))
    print('  -> pion_v2_data.txt')


# ============================================================
#  2. ratio_data.txt — coalescence ratio vs centrality
# ============================================================
def gen_ratio():
    lines = []
    lines.append('// ============================================================')
    lines.append('// ratio.pdf — Coalescence ratio vs centrality')
    lines.append('//   R = (v2_pim - 2/3 v2_pbar) / (v2_pip - 2/3 v2_pbar)')
    lines.append('// ============================================================')
    lines.append('')

    for energy in ENERGIES:
        en = energy_label(energy)
        lines.append(f'// sqrt(s) = {en} {"=" * 50}')

        for EP in ['TPC', 'EPD']:
            fpath = os.path.join(INPUTS, 'ratio', f'{energy}_{EP}.json')
            if not os.path.exists(fpath):
                lines.append(f'// {EP}: no file')
                continue

            with open(fpath) as f:
                data = json.load(f)

            x = np.array(data.get('x', []))
            y = np.array(data.get('y', []))
            yerr = np.array(data.get('yerr', []))

            if len(x) == 0:
                lines.append(f'')
                lines.append(f'// --- {EP} (no data) ---')
                lines.append(f'double ratio_x_{energy}_{EP}[1] = {{0}};')
                lines.append(f'double ratio_y_{energy}_{EP}[1] = {{0}};')
                lines.append(f'double ratio_yerr_{energy}_{EP}[1] = {{0}};')
                lines.append(f'double ratio_yerr_stat_{energy}_{EP}[1] = {{0}};')
                lines.append(f'double ratio_yerr_sys_{energy}_{EP}[1] = {{0}};')
                lines.append(f'int ratio_n_{energy}_{EP} = 0;')
                continue

            yerr_stat = np.array(data.get('yerr_stat', []))
            yerr_sys = np.array(data.get('yerr_sys', []))

            n = len(x)
            lines.append(f'')
            lines.append(f'// --- {EP} ({n} points) ---')
            lines.append(f'int ratio_n_{energy}_{EP} = {n};')
            lines.append(f'double ratio_x_{energy}_{EP}[{n}] = {format_array(x, "%.1f")};')
            lines.append(f'double ratio_y_{energy}_{EP}[{n}] = {format_array(y, "%.6e")};')
            lines.append(f'double ratio_yerr_{energy}_{EP}[{n}] = {format_array(yerr, "%.6e")};')
            lines.append(f'double ratio_yerr_stat_{energy}_{EP}[{n}] = {format_array(yerr_stat, "%.6e")};')
            lines.append(f'double ratio_yerr_sys_{energy}_{EP}[{n}] = {format_array(yerr_sys, "%.6e")};')

        lines.append('')

    with open(os.path.join(OUTPUT, 'ratio_data.txt'), 'w') as f:
        f.write('\n'.join(lines))
    print('  -> ratio_data.txt')


# ============================================================
#  3. energy_dep_data.txt — 10-40% ratio vs sqrt(s_NN)
# ============================================================
def gen_energy_dep():
    lines = []
    lines.append('// ============================================================')
    lines.append('// energy_dep.pdf — 10-40% centrality ratio vs sqrt(s_NN)')
    lines.append('// ============================================================')
    lines.append('')

    n = len(ENERGIES)
    e_vals = np.array([float(energy_label(e).replace('GeV', '')) for e in ENERGIES])

    lines.append(f'int n_energies = {n};')
    lines.append(f'double sqrt_s_NN[{n}] = {format_array(e_vals, "%.1f")};')
    lines.append('')

    for EP in ['TPC', 'EPD']:
        y_vals = np.full(n, np.nan)
        yerr_vals = np.full(n, np.nan)
        yerr_stat_vals = np.full(n, np.nan)
        yerr_sys_vals = np.full(n, np.nan)

        for i, energy in enumerate(ENERGIES):
            fpath = os.path.join(INPUTS, 'ratio', f'{energy}_{EP}.json')
            if not os.path.exists(fpath):
                continue
            with open(fpath) as f:
                data = json.load(f)
            y_1040 = data.get('y_1040', np.nan)
            if y_1040 == -999.0 or (isinstance(y_1040, float) and math.isnan(y_1040)):
                continue
            y_vals[i] = y_1040
            yerr_vals[i] = data.get('yerr_1040', np.nan)
            yerr_stat_vals[i] = data.get('yerr_stat_1040', np.nan)
            yerr_sys_vals[i] = data.get('yerr_sys_1040', np.nan)

        lines.append(f'// --- {EP} ---')
        lines.append(f'double ratio_1040_{EP}[{n}] = {format_array(y_vals, "%.6e")};')
        lines.append(f'double ratio_1040_{EP}_err[{n}] = {format_array(yerr_vals, "%.6e")};')
        lines.append(f'double ratio_1040_{EP}_err_stat[{n}] = {format_array(yerr_stat_vals, "%.6e")};')
        lines.append(f'double ratio_1040_{EP}_err_sys[{n}] = {format_array(yerr_sys_vals, "%.6e")};')
        lines.append('')

    # Also include the Glauber band and 315/276 expectation
    gl_centrality = np.array([2.5, 7.5, 15, 25, 35, 45, 55, 65, 75])
    gl_ratio = np.array([1.13304, 1.13351, 1.13777, 1.14464, 1.15486,
                         1.17016, 1.19128, 1.22017, 1.26102])
    gl_err = np.array([1.56853e-08, 4.95918e-08, 9.90949e-08, 1.74393e-07,
                       3.19487e-07, 5.92654e-07, 1.24839e-06, 3.09313e-06,
                       5.11625e-06])
    lines.append('// Glauber model band (for ratio.pdf)')
    n_gl = len(gl_centrality)
    lines.append(f'int glauber_n = {n_gl};')
    lines.append(f'double glauber_centrality[{n_gl}] = {format_array(gl_centrality, "%.1f")};')
    lines.append(f'double glauber_ratio[{n_gl}] = {format_array(gl_ratio, "%.6e")};')
    lines.append(f'double glauber_ratio_err[{n_gl}] = {format_array(gl_err, "%.6e")};')
    lines.append('')
    lines.append('// 315/276 expectation line')
    lines.append('double expectation_315_276 = 315.0 / 276.0;')

    with open(os.path.join(OUTPUT, 'energy_dep_data.txt'), 'w') as f:
        f.write('\n'.join(lines))
    print('  -> energy_dep_data.txt')


if __name__ == '__main__':
    gen_pion_v2()
    gen_ratio()
    gen_energy_dep()
    print(f'\nAll files saved to {OUTPUT}/')
