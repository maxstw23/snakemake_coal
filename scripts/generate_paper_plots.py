from importlib.resources import files
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from matplotlib.legend_handler import HandlerTuple
from uncertainties import unumpy, ufloat, core
import numpy as np
# import pickle
import os 
import argparse
import yaml
import copy
import pandas as pd
from measurement import Measurement
import shutil
from data_point import DataPoint


# print(matplotlib.font_manager.get_font_names())
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
# plt.rcParams['font.weight'] = 'bold'

def eb(ax, x, y, yerr, color, marker='o', ms=8, capsize=2, mfc=None, zorder=2, lw=1.5, min_stub=0.25):
    """Errorbar with lines clipped to the marker edge so they never cross through it."""
    mfc_actual = mfc if mfc is not None else color
    ax.plot(x, y, marker=marker, color=color, mfc=mfc_actual, mec=color,
            ms=ms, ls='none', zorder=zorder + 1)
    # Marker radius in data coordinates via axis transform
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
    'TPC':
    {
        'marker': '*',
        'color': 'C0',
        'ms': 10,
        'ls': 'none',
        'capsize': 2
    },
    'EPD':
    {
        'marker': 'o',
        'color': 'C1',
        'ms': 8,
        'ls': 'none',
        'capsize': 2
    }
}
plot_config = {
    'Lambda':
    {
        'marker': '*',
        'color': 'C3',
        'label': r'$\Lambda^0-\bar{\Lambda}^0$',
        'markersize': 10,
        'zorder': 1,
        'ls': 'none',
        'capsize': 2,
        'alpha': 0.8
    },
    'combo':
    {
        'marker': 's',
        'color': 'C0',
        'label': r'$(p-\bar{p})-(K^+-K^-)$',
        'markersize': 8,
        'ls': 'none',
        'capsize': 2,
        'alpha': 0.8
    },
    'combo2':
    {
        'marker': 'o',
        'color': 'black',
        'label': r'$p-\bar{p}$',
        'markersize': 8,
        'markerfacecolor': 'white',
        'ls': 'none',
        'capsize': 2,
        'alpha': 0.8
    }
}

def find_files(input_files, key):
    for f in input_files:
        if key in f:
            return f
    return None


def show_figure(fig):
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)


def plot_res(dict_input, figs, paper_plots_path):
    ### resolution
    files = dict_input['res']
    fig_res = plt.figure(figsize=(8 ,12))
    gs_res = fig_res.add_gridspec(2, 1, hspace=0.0)
    ax_res = gs_res.subplots(sharex='col', sharey='row')
    ax_res = ax_res.flatten()

    # filter out isobars and sort by energy
    _energy_key = lambda f: float(f.split('/')[-2].split('_')[1].replace('p', '.').replace('GeV', ''))
    res_files = sorted([f for f in files if not f.split('/')[-2].split('_')[1].startswith('isobar')], key=_energy_key)

    # energies that have EPD data in the ratio plot
    epd_energies = {'14.6GeV', '17.3GeV', '19.6GeV', '27GeV'}

    markers = ['o', 's', 'D', '^', 'v', 'p', '*']
    colors = [f'C{i}' for i in range(len(res_files))]

    epd_files = [(i, f) for i, f in enumerate(res_files)
                 if f.split('/')[-2].split('_')[1].replace('p', '.') in epd_energies]

    shift_range = 3
    n_tpc = len(res_files)
    n_epd = len(epd_files)
    x_base = np.array([75, 65, 55, 45, 35, 25, 15, 7.5, 2.5])

    for i, f in enumerate(res_files):
        df = pd.read_csv(f)
        energy = f.split('/')[-2].split('_')[1].replace('p', '.')
        TPC_res = unumpy.uarray(df['TPC_res'].values, df['TPC_res_err'].values)
        x = x_base - 0.5 * shift_range + i / (n_tpc - 1) * shift_range
        ax_res[0].errorbar(x, unumpy.nominal_values(TPC_res), unumpy.std_devs(TPC_res), label=energy,
                           fmt=markers[i], capsize=2, ms=8, color=colors[i])
        ax_res[0].plot(x, unumpy.nominal_values(TPC_res), '--', color=colors[i], alpha=0.4)

    for j, (orig_i, f) in enumerate(epd_files):
        df = pd.read_csv(f)
        energy = f.split('/')[-2].split('_')[1].replace('p', '.')
        EPD_res = unumpy.uarray(df['EPD_res'].values, df['EPD_res_err'].values)
        EPD_mask = unumpy.nominal_values(EPD_res) > 0
        x = x_base - 0.5 * shift_range + j / max(n_epd - 1, 1) * shift_range
        ax_res[1].errorbar(x[EPD_mask], unumpy.nominal_values(EPD_res[EPD_mask]), unumpy.std_devs(EPD_res[EPD_mask]),
                           label=energy, fmt=markers[orig_i], capsize=2, ms=8, color=colors[orig_i])
        ax_res[1].plot(x[EPD_mask], unumpy.nominal_values(EPD_res[EPD_mask]), '--', color=colors[orig_i], alpha=0.4)

    fig_res.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel(r'$\text{Centrality (%)}$', fontsize=15)
    plt.ylabel(r'$\text{Res}(\Psi_{EP})$', fontsize=15, labelpad=20)
    ax_res[0].legend(fontsize=12, frameon=False)
    ax_res[1].legend(fontsize=12, frameon=False)
    ax_res[0].annotate('(a) TPC', xy=(0.03, 0.94), xycoords='axes fraction', fontsize=15)
    ax_res[1].annotate('(b) EPD', xy=(0.03, 0.94), xycoords='axes fraction', fontsize=15)
    plt.figure(fig_res.number)
    plt.savefig(paper_plots_path + '/resolution.pdf')
    plt.savefig(paper_plots_path + '/resolution.eps', format='eps')
    plt.savefig(paper_plots_path + '/resolution.svg', format='svg')
    figs.append(fig_res)
    plt.close()
    return figs

def plot_pion_v2(dict_input, figs, paper_plot_path):
    # remove isobar files
    files = [f for f in dict_input['v2'] if not f.split('/')[-2].split('_')[1].startswith('isobar')]
    resfiles = [f for f in dict_input['res'] if not f.split('/')[-2].split('_')[1].startswith('isobar')]

    fig_coal = plt.figure(figsize=(16, 8))
    gs_coal = fig_coal.add_gridspec(ncols=4, nrows=2, hspace=0.0, wspace=0.0)
    ax_coal = gs_coal.subplots(sharex='col', sharey='row')
    ax_coal = ax_coal.flatten()
    for i, (f, fres) in enumerate(zip(files, resfiles)):
        df = pd.read_csv(f)
        df_res = pd.read_csv(fres)
        energy = f.split('/')[-2].split('_')[1].replace('p', '.')
        for EP in ['TPC', 'EPD']:
            if EP == 'EPD' and float(energy.replace('GeV', '')) < 14.6:
                continue
            resolution = unumpy.uarray(df_res[f'{EP}_res'].values, df_res[f'{EP}_res_err'].values)
            res_mask = unumpy.nominal_values(resolution) >= 0.05
            if energy == '7.7GeV' and EP == 'TPC':
                res_mask[0] = False  # 70-80% bin has poor statistics at 7.7 GeV
            piplus_v2 = unumpy.uarray(df[f'piplus_v2_{EP}'].values, df[f'piplus_v2_err_{EP}'].values) / resolution / 2
            piminus_v2 = unumpy.uarray(df[f'piminus_v2_{EP}'].values, df[f'piminus_v2_err_{EP}'].values) / resolution / 2
            antiproton_v2 = unumpy.uarray(df[f'antiproton_v2_{EP}'].values, df[f'antiproton_v2_err_{EP}'].values) / resolution / 3
            x = np.array([75., 65., 55., 45., 35., 25., 15., 7.5, 2.5])

            x = x[res_mask]
            piplus_v2 = piplus_v2[res_mask]
            piminus_v2 = piminus_v2[res_mask]
            antiproton_v2 = antiproton_v2[res_mask]
            shift_EP = 0.5 if EP == 'TPC' else -0.5
            shift_par = 1.0
            mfc = None if EP == 'TPC' else 'none'
            z = 4 if EP == 'TPC' else 2
            eb(ax_coal[i], x+shift_par+shift_EP, unumpy.nominal_values(piplus_v2), unumpy.std_devs(piplus_v2),
               color='C0', marker='o', mfc=mfc, zorder=z)
            eb(ax_coal[i], x-shift_par+shift_EP, unumpy.nominal_values(piminus_v2), unumpy.std_devs(piminus_v2),
               color='C3', marker='^', mfc=mfc, zorder=z)
            eb(ax_coal[i], x+shift_EP, unumpy.nominal_values(antiproton_v2), unumpy.std_devs(antiproton_v2),
               color='C2', marker='d', mfc=mfc, zorder=z)
            ax_coal[i].annotate(r'$\sqrt{s_{\text{NN}}}=$' + energy, xy=(0.15, 0.9), fontsize=15, xycoords='axes fraction', horizontalalignment='left')
            ax_coal[i].set_xlim(-5, 85)
            ax_coal[i].set_ylim(-0.005, 0.0349)
            lb, rb = ax_coal[i].get_xlim()
    fig_coal.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel(r'$\text{Centrality (%)}$', fontsize=15)
    plt.ylabel(r'$v_2/n_q$', fontsize=15, labelpad=20)
    h_auau = ax_coal[7].plot([], [], ' ')[0]
    h_piplus_tpc = ax_coal[7].errorbar([], [], yerr=[], fmt='o', capsize=2, ms=8, color='C0')
    h_piplus_epd = ax_coal[7].errorbar([], [], yerr=[], fmt='o', capsize=2, ms=8, color='C0', mfc='white')
    h_piminus_tpc = ax_coal[7].errorbar([], [], yerr=[], fmt='^', capsize=2, ms=8, color='C3')
    h_piminus_epd = ax_coal[7].errorbar([], [], yerr=[], fmt='^', capsize=2, ms=8, color='C3', mfc='white')
    h_pbar_tpc = ax_coal[7].errorbar([], [], yerr=[], fmt='d', capsize=2, ms=8, color='C2')
    h_pbar_epd = ax_coal[7].errorbar([], [], yerr=[], fmt='d', capsize=2, ms=8, color='C2', mfc='white')
    ax_coal[7].tick_params(axis='x', which='both', length=0)
    ax_coal[7].legend(
        [h_auau, (h_piplus_tpc, h_piplus_epd), (h_piminus_tpc, h_piminus_epd), (h_pbar_tpc, h_pbar_epd)],
        ['Au+Au', r'$\pi^+$ (TPC/EPD)', r'$\pi^-$ (TPC/EPD)', r'$\bar{p}$ (TPC/EPD)'],
        handler_map={tuple: HandlerTuple(ndivide=None)},
        fontsize=15, frameon=False, loc='center'
    )

    plt.figure(fig_coal.number)
    plt.savefig(paper_plot_path + '/pion_v2.pdf')
    plt.savefig(paper_plot_path + '/pion_v2.eps', format='eps')
    plt.savefig(paper_plot_path + '/pion_v2.svg', format='svg')
    figs.append(fig_coal)
    plt.close()
    return figs


def plot_ratio(dict_input, figs, paper_plot_path):
    merged_bins = {
        '7.7GeV': {'EPD': [7,8,9]},
         '9.2GeV': {'EPD': [1,2,3]},
        '11.5GeV': {'EPD': [1,2]},
        '14.6GeV': {'TPC': [1,2], 'EPD': [1,2]},
        '17.3GeV': {'TPC': [1,2], 'EPD': [1,2,3]},
        '19.6GeV': {'TPC': [1,2], 'EPD': [1,2]},
        '27GeV': {'TPC': [1,2,3], 'EPD': [1,2]},
    }
    # after that, still some points are out of range, manually mask them
    # note, these are the bins after merging
    # masked_bins = {'9.2GeV': {'EPD': [1]}}
    masked_bins = {}

    files = {}
    _energy_key = lambda f: float(f.split('/')[-2].split('_')[1].replace('p', '.').replace('GeV', ''))
    for EP in ['TPC', 'EPD']:
        files[EP] = sorted(
            [f for f in dict_input['ratio'] if f.split('/')[-1].split('_')[1].startswith(EP) and not f.split('/')[-2].split('_')[1].startswith('isobar')],
            key=_energy_key
        )

    fig_coal = plt.figure(figsize=(16, 8))
    gs_coal = fig_coal.add_gridspec(ncols=4, nrows=2, hspace=0.0, wspace=0.0)
    ax_coal = gs_coal.subplots(sharex='col', sharey='row')
    ax_coal = ax_coal.flatten()

    for i, f in enumerate(files['TPC']):
        energy = f.split('/')[-2].split('_')[1].replace('p', '.')
        for EP in ['TPC', 'EPD']:
            with open(files[EP][i], 'r') as f:
                data_dict = yaml.load(f, Loader=yaml.CLoader)
            x = np.array(data_dict['x'])
            ratio = unumpy.uarray(data_dict['y'], data_dict['yerr_stat'])
            err_sys = np.array(data_dict['yerr_sys'])

            shift = 1. if EP == 'TPC' else -1.
            ax_coal[i].errorbar(x+shift, unumpy.nominal_values(ratio), unumpy.std_devs(ratio), **marker_styles[EP])
            for j in range(len(x)):
                ax_coal[i].fill_between(np.array([x[j]-0.5, x[j]+0.5])+shift, 
                                        unumpy.nominal_values(ratio)[j]-err_sys[j], unumpy.nominal_values(ratio)[j]+err_sys[j], color=marker_styles[EP]['color'], alpha=0.3)
            ax_coal[i].annotate(r'$\sqrt{s_{\text{NN}}}=$' + energy, xy=(0.15, 0.9), fontsize=15, xycoords='axes fraction', horizontalalignment='left')
            ax_coal[i].set_xlim(-5, 85)
            lb, rb = ax_coal[i].get_xlim()
            ax_coal[i].set_ylim(0.55, 1.649)
            ax_coal[i].hlines(315 / 276, lb, rb, color='C3', label='315/276', linestyle='--')
            ax_coal[i].hlines(1, lb, rb, color='black', linestyle='--')
            # ax_coal[i].set_xlim(lb, rb)
        # no Glauber for now
        cen_glauber = np.array([2.5,7.5,15,25,35,45,55,65,75])
        ratio_du = np.array([1.13304,1.13351,1.13777,1.14464,1.15486,1.17016,1.19128,1.22017,1.26102])
        ratio_du_err = np.array([1.56853e-08,4.95918e-08,9.90949e-08,1.74393e-07,3.19487e-07,5.92654e-07,1.24839e-06,3.09313e-06,5.11625e-06])
        ratio_upper = ratio_du + ratio_du_err
        ratio_lower = ratio_du - ratio_du_err
        ax_coal[i].fill_between(cen_glauber, ratio_lower, ratio_upper, color='C2', alpha=0.8, label='Glauber')
    fig_coal.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel(r'$\text{Centrality (%)}$', fontsize=15)
    plt.ylabel(r'$\frac{v_2^{\pi^-}-\frac{2}{3}v_2^{\bar{p}}}{v_2^{\pi^+}-\frac{2}{3}v_2^{\bar{p}}}$', fontsize=18)
    # plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.12)
    ax_coal[7].plot([], [], ' ', label='Au+Au')
    ax_coal[7].errorbar([], [], yerr=[], label='TPC Event plane', **marker_styles['TPC'])
    ax_coal[7].errorbar([], [], yerr=[], label='EPD Event plane', **marker_styles['EPD'])
    # ax_coal[7].annotate(r'$\bf{STAR}\;\it{Preliminary}$', xy=(0.15, 0.8), xycoords='axes fraction', fontsize=20)
    ax_coal[7].fill_between([], [], [], color='C2', alpha=0.8, label='Glauber d/u')
    ax_coal[7].hlines(999, lb, rb, color='C3', linestyle='--', label='315/276')
    # ax_coal[7].tick_params(axis='x', which='both', length=0)
    ax_coal[7].legend(fontsize=15, frameon=False, loc='center')

    # print result to txt
    with open(paper_plot_path + '/ratio.txt', 'w') as file:
        for i, f in enumerate(files['TPC']):
            energy = f.split('/')[-2].split('_')[1].replace('p', '.')
            file.write(f'AuAu, {energy}\n')
            for EP in ['TPC', 'EPD']:
                file.write(f'{EP}\n')
                with open(files[EP][i], 'r') as f:
                    data_dict = yaml.load(f, Loader=yaml.CLoader)
                x = np.array(data_dict['x'])
                ratio = unumpy.uarray(data_dict['y'], data_dict['yerr'])
                for j in range(len(x)):
                    file.write(f'{x[j]:.2f} {unumpy.nominal_values(ratio)[j]:.4f}+-{unumpy.std_devs(ratio)[j]:.4f}\n')
            file.write('\n')
    plt.figure(fig_coal.number)
    ratio_pdf = paper_plot_path + '/ratio.pdf'
    if os.path.exists(ratio_pdf):
        shutil.copyfile(ratio_pdf, paper_plot_path + '/ratio_old.pdf')
    plt.savefig(ratio_pdf)
    plt.savefig(paper_plot_path + '/ratio.eps', format='eps')
    plt.savefig(paper_plot_path + '/ratio.svg', format='svg', transparent = True, bbox_inches = 'tight', pad_inches = 0)
    figs.append(fig_coal)
    plt.close()
    return figs


def plot_alternative_ratio(dict_input, figs, paper_plot_path):
    figure = plt.figure(figsize=(16, 8))
    gs = figure.add_gridspec(ncols=4, nrows=2, hspace=0.0, wspace=0.0)
    ax = gs.subplots(sharex='col', sharey='row')
    ax = ax.flatten()
    for EP in ['TPC', 'EPD']:
        files_proton = dict_input['delta_v2']['p'][EP]
        files_pion = dict_input['delta_v2']['pi'][EP]

        for i, (file_proton, file_pion) in enumerate(zip(files_proton, files_pion)):
            energy = file_proton.split('/')[-2].split('_')[1].replace('p', '.')
            with open(file_proton, 'r') as f:
                data_dict_proton = yaml.load(f, Loader=yaml.CLoader)
            with open(file_pion, 'r') as f:
                data_dict_pion = yaml.load(f, Loader=yaml.CLoader)
            x = np.array(data_dict_proton['split']['x'])
            delta_p = unumpy.uarray(data_dict_proton['split']['y'], data_dict_proton['split']['y_err'])
            delta_pi = unumpy.uarray(data_dict_pion['split']['y'], data_dict_pion['split']['y_err'])

            ratio = np.divide(delta_p - 2. * delta_pi, delta_p + delta_pi, where=unumpy.nominal_values(delta_p - delta_pi)!=0, out=np.zeros_like(delta_p))
            mask = unumpy.nominal_values(ratio) != 0
            x = x[mask]
            ratio = ratio[mask]

            shift = 1. if EP == 'TPC' else -1.
            ax[i].errorbar(x+shift, unumpy.nominal_values(ratio), unumpy.std_devs(ratio), **marker_styles[EP])
            ax[i].annotate(r'AuAu, $\sqrt{s_{\text{NN}}}=$' + energy, xy=(0.85, 0.9), fontsize=15, xycoords='axes fraction', horizontalalignment='right')
            ax[i].set_xlim(-5, 85)
            lb, rb = ax[i].get_xlim()
            ax[i].set_ylim(0.509, 1.759)
            ax[i].hlines(315 / 276, lb, rb, color='C3', label='315/276', linestyle='--')
            ax[i].hlines(1, lb, rb, color='black', label='1', linestyle='--')
            # no Glauber for now
            cen_glauber = np.array([2.5,7.5,15,25,35,45,55,65,75])
            ratio_du = np.array([1.13304,1.13351,1.13777,1.14464,1.15486,1.17016,1.19128,1.22017,1.26102])
            ratio_du_err = np.array([1.56853e-08,4.95918e-08,9.90949e-08,1.74393e-07,3.19487e-07,5.92654e-07,1.24839e-06,3.09313e-06,5.11625e-06])
            ratio_upper = ratio_du + ratio_du_err
            ratio_lower = ratio_du - ratio_du_err
            ax[i].fill_between(cen_glauber, ratio_lower, ratio_upper, color='C2', alpha=0.8, label='Glauber')


    figure.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel(r'$\text{Centrality (%)}$', fontsize=15)
    plt.ylabel(r'$\frac{\Delta v_2(p)+2\Delta v_2(\pi)}{\Delta v_2(p)-\Delta v_2(\pi)}$', fontsize=18)
    ax[7].errorbar([], [], yerr=[], label='TPC Event plane', **marker_styles['TPC'])
    ax[7].errorbar([], [], yerr=[], label='EPD Event plane', **marker_styles['EPD'])
    ax[7].fill_between([], [], [], color='C2', alpha=0.8, label='Glauber d/u')
    ax[7].hlines(999, lb, rb, color='C3', linestyle='--', label='315/276')
    ax[7].hlines(999, lb, rb, color='black', linestyle='--', label='1')
    ax[7].legend(fontsize=15, frameon=False, loc='center')
    plt.figure(figure.number)
    plt.savefig(paper_plot_path + '/alternative_ratio.pdf')
    plt.savefig(paper_plot_path + '/alternative_ratio.eps', format='eps')
    plt.savefig(paper_plot_path + '/alternative_ratio.svg', format='svg', transparent = True, bbox_inches = 'tight', pad_inches = 0)
    figs.append(figure)
    plt.close()
    return figs

def plot_alternative_ratio_integrated(dict_input, figs, paper_plot_path):
    merged_bins = {
        '7.7GeV': {'EPD': [7,8,9]},
         '9.2GeV': {'EPD': [1,2,3]},
        '11.5GeV': {'EPD': [1,2]},
        '14.6GeV': {'TPC': [1,2], 'EPD': [1,2]},
        '17.3GeV': {'TPC': [1,2], 'EPD': [1,2,3]},
        '19.6GeV': {'TPC': [1,2], 'EPD': [1,2]},
        '27GeV': {'TPC': [1,2,3], 'EPD': [1,2]},
    }
    masked_bins = {}

    files = [f for f in dict_input['v2'] if not f.split('/')[-2].split('_')[1].startswith('isobar')]
    resfiles = [f for f in dict_input['res'] if not f.split('/')[-2].split('_')[1].startswith('isobar')]

    figure = plt.figure(figsize=(16, 8))
    gs = figure.add_gridspec(ncols=4, nrows=2, hspace=0.0, wspace=0.0)
    ax = gs.subplots(sharex='col', sharey='row')
    ax = ax.flatten()

    for i, (f, fres) in enumerate(zip(files, resfiles)):
        df = pd.read_csv(f)
        df_res = pd.read_csv(fres)
        energy = f.split('/')[-2].split('_')[1].replace('p', '.')
        for EP in ['TPC', 'EPD']:
            resolution = unumpy.uarray(df_res[f'{EP}_res'].values, df_res[f'{EP}_res_err'].values)
            piplus_v2 = unumpy.uarray(df[f'piplus_v2_{EP}'].values, df[f'piplus_v2_err_{EP}'].values) / resolution
            piminus_v2 = unumpy.uarray(df[f'piminus_v2_{EP}'].values, df[f'piminus_v2_err_{EP}'].values) / resolution
            proton_v2 = unumpy.uarray(df[f'proton_v2_{EP}'].values, df[f'proton_v2_err_{EP}'].values) / resolution
            antiproton_v2 = unumpy.uarray(df[f'antiproton_v2_{EP}'].values, df[f'antiproton_v2_err_{EP}'].values) / resolution
            x = np.array([75., 65., 55., 45., 35., 25., 15., 7.5, 2.5])
            x = unumpy.uarray(x, np.ones_like(x))

            if energy in merged_bins:
                if EP in merged_bins[energy]:
                    bins = merged_bins[energy][EP]
                    dict_v2 = {'x': x, 'pip': piplus_v2, 'pim': piminus_v2, 'p': proton_v2, 'ap': antiproton_v2}
                    dict_v2 = merge_helper(dict_v2, [bins])
                    x = dict_v2['x']
                    piplus_v2 = dict_v2['pip']
                    piminus_v2 = dict_v2['pim']
                    proton_v2 = dict_v2['p']
                    antiproton_v2 = dict_v2['ap']

            if energy in masked_bins:
                if EP in masked_bins[energy]:
                    bins = masked_bins[energy][EP]
                    for b in bins:
                        x[b-1] = ufloat(0, 0)
                        piplus_v2[b-1] = ufloat(0, 0)
                        piminus_v2[b-1] = ufloat(0, 0)
                        proton_v2[b-1] = ufloat(0, 0)
                        antiproton_v2[b-1] = ufloat(0, 0)
                    mask = unumpy.nominal_values(x) != 0
                    x = x[mask]
                    piplus_v2 = piplus_v2[mask]
                    piminus_v2 = piminus_v2[mask]
                    proton_v2 = proton_v2[mask]
                    antiproton_v2 = antiproton_v2[mask]

            x = unumpy.nominal_values(x)
            delta_p = proton_v2 - antiproton_v2
            delta_pi = piplus_v2 - piminus_v2

            ratio = np.divide(delta_p - 2. * delta_pi, delta_p + delta_pi, where=unumpy.nominal_values(delta_p + delta_pi)!=0, out=np.zeros_like(delta_p))
            mask = unumpy.nominal_values(ratio) != 0
            x_masked = x[mask]
            ratio = ratio[mask]

            shift = 1. if EP == 'TPC' else -1.
            ax[i].errorbar(x_masked+shift, unumpy.nominal_values(ratio), unumpy.std_devs(ratio), **marker_styles[EP])
            ax[i].annotate(r'AuAu, $\sqrt{s_{\text{NN}}}=$' + energy, xy=(0.85, 0.9), fontsize=15, xycoords='axes fraction', horizontalalignment='right')
            ax[i].set_xlim(-5, 85)
            lb, rb = ax[i].get_xlim()
            ax[i].set_ylim(0.509, 1.759)
            ax[i].hlines(315 / 276, lb, rb, color='C3', label='315/276', linestyle='--')
            ax[i].hlines(1, lb, rb, color='black', label='1', linestyle='--')
        cen_glauber = np.array([2.5,7.5,15,25,35,45,55,65,75])
        ratio_du = np.array([1.13304,1.13351,1.13777,1.14464,1.15486,1.17016,1.19128,1.22017,1.26102])
        ratio_du_err = np.array([1.56853e-08,4.95918e-08,9.90949e-08,1.74393e-07,3.19487e-07,5.92654e-07,1.24839e-06,3.09313e-06,5.11625e-06])
        ratio_upper = ratio_du + ratio_du_err
        ratio_lower = ratio_du - ratio_du_err
        ax[i].fill_between(cen_glauber, ratio_lower, ratio_upper, color='C2', alpha=0.8, label='Glauber')

    figure.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel(r'$\text{Centrality (%)}$', fontsize=15)
    plt.ylabel(r'$\frac{\Delta v_2(p)+2\Delta v_2(\pi)}{\Delta v_2(p)-\Delta v_2(\pi)}$', fontsize=18)
    ax[7].errorbar([], [], yerr=[], label='TPC Event plane', **marker_styles['TPC'])
    ax[7].errorbar([], [], yerr=[], label='EPD Event plane', **marker_styles['EPD'])
    ax[7].fill_between([], [], [], color='C2', alpha=0.8, label='Glauber d/u')
    ax[7].hlines(999, lb, rb, color='C3', linestyle='--', label='315/276')
    ax[7].hlines(999, lb, rb, color='black', linestyle='--', label='1')
    ax[7].legend(fontsize=15, frameon=False, loc='center')
    plt.figure(figure.number)
    plt.savefig(paper_plot_path + '/alternative_ratio_integrated.pdf')
    plt.savefig(paper_plot_path + '/alternative_ratio_integrated.eps', format='eps')
    plt.savefig(paper_plot_path + '/alternative_ratio_integrated.svg', format='svg', transparent = True, bbox_inches = 'tight', pad_inches = 0)
    figs.append(figure)
    plt.close()
    return figs

def plot_ratio_one_energy(dict_input, figs, paper_plot_path):
    merged_bins = {
        '7.7GeV': {'EPD': [7,8,9]},
         '9.2GeV': {'EPD': [1,2,3]},
        '11.5GeV': {'EPD': [1,2]},
        '14.6GeV': {'TPC': [1,2], 'EPD': [1,2]},
        '17.3GeV': {'TPC': [1,2], 'EPD': [1,2,3]},
        '19.6GeV': {'TPC': [1,2], 'EPD': [1,2]},
        '27GeV': {'TPC': [1,2,3], 'EPD': [1,2]},
    }

    # after that, still some points are out of range, manually mask them
    # note, these are the bins after merging
    # masked_bins = {'9.2GeV': {'EPD': [1]}}
    masked_bins = {}
    
    # remove isobar files
    files = [f for f in dict_input['v2'] if f.split('/')[-2].split('_')[1].startswith('19p6GeV')]
    resfiles = [f for f in dict_input['res'] if f.split('/')[-2].split('_')[1].startswith('19p6GeV')]

    fig_coal = plt.figure(figsize=(6.4, 4.8))
    gs_coal = fig_coal.add_gridspec(ncols=1, nrows=1, hspace=0.0, wspace=0.0)
    ax_coal = gs_coal.subplots(sharex='col', sharey='row')
    ax_coal = [ax_coal]
    lb, rb = ax_coal[0].get_xlim()
    for i, (f, fres) in enumerate(zip(files, resfiles)):
        df = pd.read_csv(f)
        df_res = pd.read_csv(fres)
        energy = f.split('/')[-2].split('_')[1].replace('p', '.')
        for EP in ['TPC', 'EPD']:
            resolution = unumpy.uarray(df_res[f'{EP}_res'].values, df_res[f'{EP}_res_err'].values)
            piplus_v2 = unumpy.uarray(df[f'piplus_v2_{EP}'].values, df[f'piplus_v2_err_{EP}'].values) / resolution
            piminus_v2 = unumpy.uarray(df[f'piminus_v2_{EP}'].values, df[f'piminus_v2_err_{EP}'].values) / resolution
            antiproton_v2 = unumpy.uarray(df[f'antiproton_v2_{EP}'].values, df[f'antiproton_v2_err_{EP}'].values) / resolution
            ### TEST ###
            # antiproton_pip_v2 = unumpy.uarray(df[f'antiproton_pip_v2_{EP}'].values, df[f'antiproton_pip_v2_err_{EP}'].values) / resolution
            # antiproton_pim_v2 = unumpy.uarray(df[f'antiproton_pim_v2_{EP}'].values, df[f'antiproton_pim_v2_err_{EP}'].values) / resolution
            x = np.array([75., 65., 55., 45., 35., 25., 15., 7.5, 2.5])  
            x = unumpy.uarray(x, np.ones_like(x))

            if energy in merged_bins:
                if EP in merged_bins[energy]:
                    bins = merged_bins[energy][EP]
                    dict_v2 = {'x': x, 'pip': piplus_v2, 'pim': piminus_v2, 'ap': antiproton_v2}
                    ### TEST ###
                    # dict_v2 = {'x': x, 'pip': piplus_v2, 'pim': piminus_v2, 'ap': antiproton_v2, 'ap_pip': antiproton_pip_v2, 'ap_pim': antiproton_pim_v2}
                    dict_v2 = merge_helper(dict_v2, [bins])
                    x = dict_v2['x']
                    piplus_v2 = dict_v2['pip']
                    piminus_v2 = dict_v2['pim']
                    antiproton_v2 = dict_v2['ap']
                    ### TEST ###
                    # antiproton_pip_v2 = dict_v2['ap_pip']
                    # antiproton_pim_v2 = dict_v2['ap_pim']

            # now deal with masked bins
            if energy in masked_bins:
                if EP in masked_bins[energy]:
                    bins = masked_bins[energy][EP]
                    for b in bins:
                        x[b-1] = ufloat(0, 0)
                        piplus_v2[b-1] = ufloat(0, 0)
                        piminus_v2[b-1] = ufloat(0, 0)
                        antiproton_v2[b-1] = ufloat(0, 0)
                    x = x[unumpy.nominal_values(x) != 0]
                    piplus_v2 = piplus_v2[unumpy.nominal_values(piplus_v2) != 0]
                    piminus_v2 = piminus_v2[unumpy.nominal_values(piminus_v2) != 0]
                    antiproton_v2 = antiproton_v2[unumpy.nominal_values(antiproton_v2) != 0]
                    ### TEST ###
                    # antiproton_pip_v2 = antiproton_pip_v2[unumpy.nominal_values(antiproton_pip_v2) != 0]
                    # antiproton_pim_v2 = antiproton_pim_v2[unumpy.nominal_values(antiproton_pim_v2) != 0]

            x = unumpy.nominal_values(x)
            ratio = (piminus_v2 - piplus_v2) / (piplus_v2 - antiproton_v2 * 2. / 3.) + 1
            if energy == '14.6GeV':
                print(f'14.6 GeV {EP} ratio:')
                print(ratio)
            ### TEST ###
            # ratio = (piminus_v2 - antiproton_pim_v2 * 2. / 3.) / (piplus_v2 - antiproton_pim_v2 * 2. / 3.)
            shift = 1. if EP == 'TPC' else -1.
            ax_coal[i].errorbar(x+shift, unumpy.nominal_values(ratio), unumpy.std_devs(ratio), **marker_styles[EP])
            ax_coal[i].annotate(r'$\sqrt{s_{\text{NN}}}=$' + energy, xy=(0.15, 0.9), fontsize=15, xycoords='axes fraction', horizontalalignment='left')
            ax_coal[i].set_xlim(-5, 85)
            lb, rb = ax_coal[i].get_xlim()
            ax_coal[i].set_ylim(0.5, 1.5)
            ax_coal[i].hlines(315 / 276, lb, rb, color='C3', linestyle='--')
            ax_coal[i].hlines(1, lb, rb, color='black', linestyle='--')
            # ax_coal[i].set_xlim(lb, rb)
        # no Glauber for now
        cen_glauber = np.array([2.5,7.5,15,25,35,45,55,65,75])
        ratio_du = np.array([1.13304,1.13351,1.13777,1.14464,1.15486,1.17016,1.19128,1.22017,1.26102])
        ratio_du_err = np.array([1.56853e-08,4.95918e-08,9.90949e-08,1.74393e-07,3.19487e-07,5.92654e-07,1.24839e-06,3.09313e-06,5.11625e-06])
        ratio_upper = ratio_du + ratio_du_err
        ratio_lower = ratio_du - ratio_du_err
        ax_coal[i].fill_between(cen_glauber, ratio_lower, ratio_upper, color='C2', alpha=0.8)
    fig_coal.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel(r'$\text{Centrality (%)}$', fontsize=15)
    plt.ylabel(r'$\frac{v_2^{\pi^-}-\frac{2}{3}v_2^{\bar{p}}}{v_2^{\pi^+}-\frac{2}{3}v_2^{\bar{p}}}$', fontsize=18)
    # plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.12)
    ax_coal[0].errorbar([], [], yerr=[], label='TPC Event plane', **marker_styles['TPC'])
    ax_coal[0].errorbar([], [], yerr=[], label='EPD Event plane', **marker_styles['EPD'])
    ax_coal[0].fill_between([], [], [], color='C2', alpha=0.8, label='Glauber')
    ax_coal[0].hlines(999, lb, rb, color='C3', linestyle='--', label='315/276')
    ax_coal[0].hlines(999, lb, rb, color='black', linestyle='--', label='1')
    ax_coal[0].tick_params(axis='x', which='both', length=0)
    ax_coal[0].legend(fontsize=15, frameon=False, loc='lower center')   

    plt.figure(fig_coal.number)
    plt.text(0.01, 1.02, '(b)', transform=ax_coal[0].transAxes, fontsize=15)
    plt.tight_layout()
    plt.savefig(paper_plot_path + '/ratio_one_energy.pdf')
    plt.savefig(paper_plot_path + '/ratio_one_energy.eps', format='eps')
    plt.savefig(paper_plot_path + '/ratio_one_energy.svg', format='svg')
    figs.append(fig_coal)
    plt.close()
    return figs


def plot_energy_dep(dict_input, figs, paper_plots_path):
    # remove isobar files
    files = {}
    _energy_key = lambda f: float(f.split('/')[-2].split('_')[1].replace('p', '.').replace('GeV', ''))
    for EP in ['TPC', 'EPD']:
        files[EP] = sorted(
            [f for f in dict_input['ratio'] if f.split('/')[-1].split('_')[1].startswith(EP) and not f.split('/')[-2].split('_')[1].startswith('isobar')],
            key=_energy_key
        )
    fig_dep, ax_dep = plt.subplots(figsize=(8, 6))

    ratio_1040 = {}
    ratio_1040['TPC'] = np.zeros(len(files['TPC'])) * ufloat(0, 0)
    ratio_1040['EPD'] = np.zeros(len(files['EPD'])) * ufloat(0, 0)
    err_sys_1040 = {}
    err_sys_1040['TPC'] = np.zeros(len(files['TPC']))
    err_sys_1040['EPD'] = np.zeros(len(files['EPD']))

    for i, f in enumerate(files['TPC']):
        energy = f.split('/')[-2].split('_')[1].replace('p', '.')
        for EP in ['TPC', 'EPD']:
            with open(files[EP][i], 'r') as f:
                data_dict = yaml.load(f, Loader=yaml.CLoader)
            ratio_1040[EP][i] = ufloat(data_dict['y_1040'], data_dict['yerr_stat_1040'])
            err_sys_1040[EP][i] = data_dict['yerr_sys_1040']


    energy_str = [f.split('/')[-2].split('_')[1].replace('p', '.') for f in files['TPC']]
    energy_float = np.array([float(e.replace('GeV', '')) for e in energy_str])
    # we want to shift the markers to avoid overlap
    # but we are plotting the x-axis in log scale, so we need to shift the markers in log scale
    shift = 0.2 # multiply or divide by 1.5
    ax_dep.errorbar(energy_float + shift, unumpy.nominal_values(ratio_1040['TPC']), unumpy.std_devs(ratio_1040['TPC']), fmt='o', label='TPC', capsize=2, ms=8)
    for j in range(len(energy_float)):
        ax_dep.fill_between(np.array([energy_float[j]-0.5, energy_float[j]+0.5]) + shift, 
                            unumpy.nominal_values(ratio_1040['TPC'])[j]-err_sys_1040['TPC'][j], unumpy.nominal_values(ratio_1040['TPC'])[j]+err_sys_1040['TPC'][j], color='C0', alpha=0.3)
    # for EPD, show only 14.6 GeV and above
    ax_dep.errorbar((energy_float - shift)[energy_float >= 14.6], unumpy.nominal_values(ratio_1040['EPD'])[energy_float >= 14.6], unumpy.std_devs(ratio_1040['EPD'])[energy_float >= 14.6], fmt='o', label='EPD', capsize=2, ms=8)
    for j in range(len(energy_float)):
        if energy_float[j] >= 14.6:
            ax_dep.fill_between(np.array([energy_float[j]-0.5, energy_float[j]+0.5]) - shift, 
                                unumpy.nominal_values(ratio_1040['EPD'])[j]-err_sys_1040['EPD'][j], unumpy.nominal_values(ratio_1040['EPD'])[j]+err_sys_1040['EPD'][j], color='C1', alpha=0.3)
    # chi2/ndf comparing to expectation 315/276
    expectation = 315 / 276
    for EP in ['TPC', 'EPD']:
        mask = np.ones(len(energy_float), dtype=bool) if EP == 'TPC' else (energy_float >= 14.6)
        vals = unumpy.nominal_values(ratio_1040[EP])[mask]
        errs = unumpy.std_devs(ratio_1040[EP])[mask]
        chi2 = np.sum((vals - expectation)**2 / errs**2)
        ndf = len(vals)
        print(f'energy_dep chi2/ndf ({EP} vs 315/276): {chi2:.2f}/{ndf} = {chi2/ndf:.2f}')

    chi2_tpc = np.sum((unumpy.nominal_values(ratio_1040['TPC']) - expectation)**2 / unumpy.std_devs(ratio_1040['TPC'])**2)
    ndf_tpc = len(ratio_1040['TPC'])
    mask_epd = energy_float >= 14.6
    chi2_epd = np.sum((unumpy.nominal_values(ratio_1040['EPD'])[mask_epd] - expectation)**2 / unumpy.std_devs(ratio_1040['EPD'])[mask_epd]**2)
    ndf_epd = int(np.sum(mask_epd))
    ax_dep.annotate(fr'TPC $\chi^2$/ndf = {chi2_tpc:.1f}/{ndf_tpc}', xy=(0.05, 0.82), fontsize=13, xycoords='axes fraction')
    ax_dep.annotate(fr'EPD $\chi^2$/ndf = {chi2_epd:.1f}/{ndf_epd}', xy=(0.05, 0.74), fontsize=13, xycoords='axes fraction')

    ax_dep.annotate(r'AuAu, 10-40%', xy=(0.05, 0.9), fontsize=15, xycoords='axes fraction', horizontalalignment='left')
    ax_dep.set_xlabel(r'$\sqrt{s_{\text{NN}}}$ (GeV)', fontsize=15)
    ax_dep.set_ylabel(r'$\frac{v_2^{\pi^-}-\frac{2}{3}v_2^{\bar{p}}}{v_2^{\pi^+}-\frac{2}{3}v_2^{\bar{p}}}$', fontsize=15)

    # print results to txt
    with open(paper_plots_path + '/energy_dep.txt', 'w') as f:
        for i, energy in enumerate(energy_str):
            f.write(f'{energy} {unumpy.nominal_values(ratio_1040["TPC"])[i]:.4f} {unumpy.std_devs(ratio_1040["TPC"])[i]:.4f} {unumpy.nominal_values(ratio_1040["EPD"])[i]:.4f} {unumpy.std_devs(ratio_1040["EPD"])[i]:.4f}\n')
    
    # ax_dep.set_xscale('log')
    ax_dep.set_xticks(energy_float, labels=energy_float)
    ax_dep.set_xlim(5, 30)
    ax_dep.set_ylim(1.0, 1.3)
    lb, rb = ax_dep.get_xlim()
    ax_dep.hlines(315 / 276, lb, rb, color='C3', linestyle='--', label='315/276')
    # ax_dep.hlines(1, lb, rb, color='C4', linestyle='--', label='1')
    ax_dep.legend(fontsize=15)
    # make a copy of the plot from the previous iteration
    energy_dep_pdf = paper_plots_path + '/energy_dep.pdf'
    if os.path.exists(energy_dep_pdf):
        shutil.copyfile(energy_dep_pdf, paper_plots_path + '/energy_dep_old.pdf')
    plt.savefig(energy_dep_pdf)
    plt.savefig(paper_plots_path + '/energy_dep.eps', format='eps')
    plt.savefig(paper_plots_path + '/energy_dep.svg', format='svg')
    figs.append(fig_dep)
    plt.close()
    return figs

def plot_energy_dep_lambda(dict_input, figs, paper_plots_path):
    resfiles = [f for f in dict_input['res'] if not f.split('/')[-2].split('_')[1].startswith('isobar')]

    with open(dict_input['delta_v2']['lambda']['TPC'][0], 'r') as f:
        data_dict = yaml.load(f, Loader=yaml.CLoader)
        x_label = data_dict['merged']['x_label']
    
    mask = {'EPD': [11.5, 14.6]}
    
    for EP in ['TPC', 'EPD']:
        for is_horizontal in [False, True]:
            fs = 20 if is_horizontal else 15
            fig_dep = plt.figure(figsize=(8, 12)) if not is_horizontal else plt.figure(figsize=(20, 8))
            gs_dep = fig_dep.add_gridspec(ncols=1, nrows=3, hspace=0, wspace=0) if not is_horizontal else fig_dep.add_gridspec(ncols=3, nrows=1, hspace=0, wspace=0)
            ax_dep = gs_dep.subplots(sharex='col', sharey='row')
            ax_dep = ax_dep.flatten()

            files_lambda = dict_input['delta_v2']['lambda'][EP]
            files_proton = dict_input['delta_v2']['p'][EP]
            files_kaon = dict_input['delta_v2']['k'][EP]

            for i, cen_group in enumerate(x_label[:3]):
                delta_lambda = np.zeros(len(files_lambda)) * ufloat(0, 0)
                delta_pmk = np.zeros(len(files_proton)) * ufloat(0, 0)
                delta_p = np.zeros(len(files_kaon)) * ufloat(0, 0)
                for j, (flambda, fproton, fkaon) in enumerate(zip(files_lambda, files_proton, files_kaon)):
                    with open(flambda, 'r') as f1, open(fproton, 'r') as f2, open(fkaon, 'r') as f3:
                        data_dict_lambda = yaml.load(f1, Loader=yaml.CLoader)
                        data_dict_proton = yaml.load(f2, Loader=yaml.CLoader)
                        data_dict_kaon = yaml.load(f3, Loader=yaml.CLoader)

                        delta_lambda[j] = ufloat(data_dict_lambda['merged']['y'][i], data_dict_lambda['merged']['y_err'][i])
                        delta_p[j] = ufloat(data_dict_proton['merged']['y'][i], data_dict_proton['merged']['y_err'][i])
                        delta_pmk[j] = delta_p[j] - ufloat(data_dict_kaon['merged']['y'][i], data_dict_kaon['merged']['y_err'][i])

                # masking
                energy_str = [f.split('/')[-2].split('_')[1].replace('p', '.') for f in resfiles]
                energy_float = np.array([float(e.replace('GeV', '')) for e in energy_str])
                current_mask = np.where(np.array([energy_float[energy_index] not in mask.get(EP, []) for energy_index in range(len(files_lambda))]))[0]
                print(f'Energy mask for {cen_group} {EP}: {energy_float[current_mask]}')
                delta_lambda = delta_lambda[current_mask]
                delta_pmk = delta_pmk[current_mask]
                delta_p = delta_p[current_mask]
                energy_float = energy_float[current_mask]

                datapoints_lambda = DataPoint(unumpy.nominal_values(delta_lambda), unumpy.std_devs(delta_lambda))
                datapoints_pmk = DataPoint(unumpy.nominal_values(delta_pmk), unumpy.std_devs(delta_pmk))
                datapoints_p = DataPoint(unumpy.nominal_values(delta_p), unumpy.std_devs(delta_p))
                chi2ndf_2 = calculate_chi2_per_ndf(datapoints_lambda - datapoints_p, DataPoint(np.zeros(len(datapoints_lambda))), nparams=0)
                chi2ndf_1 = calculate_chi2_per_ndf(datapoints_lambda - datapoints_pmk, DataPoint(np.zeros(len(datapoints_lambda))), nparams=0)
                
                shift = 0.2 # multiply or divide by 1.5
                if x_label[i] == '40-80%':
                    ax_dep[i].errorbar(energy_float[1:] + shift, unumpy.nominal_values(delta_lambda[1:]), unumpy.std_devs(delta_lambda[1:]), **plot_config['Lambda'])
                else:
                    ax_dep[i].errorbar(energy_float + shift, unumpy.nominal_values(delta_lambda), unumpy.std_devs(delta_lambda), **plot_config['Lambda'])
                ax_dep[i].errorbar(energy_float, unumpy.nominal_values(delta_pmk), unumpy.std_devs(delta_pmk), **plot_config['combo'])
                ax_dep[i].errorbar(energy_float - shift, unumpy.nominal_values(delta_p), unumpy.std_devs(delta_p), **plot_config['combo2'])
                
                # ax_dep[i].set_xlabel(r'$\sqrt{s_{\text{NN}}}$ (GeV)', fontsize=15)
                # ax_dep[i].set_ylabel(r'$\Delta v_2$', fontsize=15)
                ax_dep[i].annotate(cen_group, xy=(0.15, 0.9), fontsize=fs, xycoords='axes fraction', horizontalalignment='left')
                ax_dep[i].annotate(fr'$\chi^2$/ndf (p) = {chi2ndf_2:.2f}', xy=(0.4, 0.15), xycoords='axes fraction', fontsize=fs)
                ax_dep[i].annotate(fr'$\chi^2$/ndf (p-K) = {chi2ndf_1:.2f}', xy=(0.4, 0.05), xycoords='axes fraction', fontsize=fs)
                ax_dep[i].set_xticks(energy_float, labels=energy_float)
                ax_dep[i].tick_params(labelsize=fs-5)
                ax_dep[i].set_xlim(5, 30)
                ax_dep[i].set_ylim(-0.0149, 0.0499)
                if EP == 'EPD':
                    ax_dep[i].set_ylim(-0.0249, 0.0699)
                lb, rb = ax_dep[i].get_xlim()
                ax_dep[i].hlines(0, lb, rb, color='black', linestyle='--')
            
            fig_dep.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.grid(False)
            plt.xlabel(r'$\sqrt{s_{\text{NN}}}$ (GeV)', fontsize=fs, labelpad=15)
            plt.ylabel(r'$\Delta v_2$', fontsize=fs, labelpad=30)

            ax_dep[1].legend(fontsize=fs, frameon=False, loc='upper right')

            plt.tight_layout()
            hori_string = '_horizontal' if is_horizontal else '_vertical'
            plt.savefig(paper_plots_path + f'/energy_dep_lambda_{EP}{hori_string}.pdf')
            plt.savefig(paper_plots_path + f'/energy_dep_lambda_{EP}{hori_string}.eps', format='eps')
            plt.savefig(paper_plots_path + f'/energy_dep_lambda_{EP}{hori_string}.svg', format='svg')
            figs.append(fig_dep)
            plt.close()
    return figs


def plot_isobar_test(dict_input, figs, paper_plots_path):
    for EP in ['TPC', 'EPD']:
        fig_isobar, ax_isobar = plt.subplots(figsize=(8, 6))
        file_ZrZr = dict_input['delta_v2_isobar']['pi']['Zr'][EP][0]
        file_RuRu = dict_input['delta_v2_isobar']['pi']['Ru'][EP][0]

        with open(file_ZrZr, 'r') as fZr, open(file_RuRu, 'r') as fRu:
            data_dict_Zr = yaml.load(fZr, Loader=yaml.CLoader)
            data_dict_Ru = yaml.load(fRu, Loader=yaml.CLoader)

            delta_v2_Zr = unumpy.uarray(data_dict_Zr['split']['y'], data_dict_Zr['split']['y_err'])
            delta_v2_Ru = unumpy.uarray(data_dict_Ru['split']['y'], data_dict_Ru['split']['y_err'])
            x = np.array(data_dict_Zr['split']['x'])
            
            ratio = delta_v2_Ru / delta_v2_Zr # should be around 0.5
            ax_isobar.errorbar(x, unumpy.nominal_values(ratio), unumpy.std_devs(ratio), fmt='o', capsize=2, ms=8)
            ax_isobar.annotate(r'RuRu/ZrZr, $\sqrt{s_{\text{NN}}}=200$ GeV', xy=(0.05, 0.9), fontsize=15, xycoords='axes fraction', horizontalalignment='left')
            ax_isobar.set_xlabel(r'$\text{Centrality (%)}$', fontsize=15)
            ax_isobar.set_ylabel(r'$\frac{\Delta v_2^{\pi}{RuRu}}{\Delta v_2^{\pi}{ZrZr}}$', fontsize=15)
            ax_isobar.set_xlim(-5, 85)
            ax_isobar.set_ylim(0, 1)
            lb, rb = ax_isobar.get_xlim()
            ax_isobar.hlines(0.5, lb, rb, color='C3', linestyle='--', label='0.5')
            ax_isobar.legend(fontsize=15)
        plt.tight_layout()
        plt.savefig(paper_plots_path + f'/isobar_test_{EP}.pdf')
        plt.savefig(paper_plots_path + f'/isobar_test_{EP}.eps', format='eps')
        plt.savefig(paper_plots_path + f'/isobar_test_{EP}.svg', format='svg')
        figs.append(fig_isobar)
        plt.close()
    return figs

def plot_efficiency_comparison(dict_input, figs, paper_plots_path):
    # Do this for all energies, compare v2 with and without efficiency correction
    files = [f for f in dict_input['v2'] if not f.split('/')[-2].split('_')[1].startswith('isobar')]
    resfiles = [f for f in dict_input['res'] if not f.split('/')[-2].split('_')[1].startswith('isobar')]
    fig_eff = plt.figure(figsize=(16, 8))
    gs_eff = fig_eff.add_gridspec(ncols=4, nrows=2, hspace=0.0, wspace=0.0)
    ax_eff = gs_eff.subplots(sharex='col', sharey='row')
    ax_eff = ax_eff.flatten()
    for i, (f, fres) in enumerate(zip(files, resfiles)):
        df = pd.read_csv(f)
        df_res = pd.read_csv(fres)
        energy = f.split('/')[-2].split('_')[1].replace('p', '.')
        if energy == '9.2GeV' or energy == '11.5GeV' or energy == '17.3GeV':
            continue
        file_eff = find_files(dict_input['v2_eff'], energy.replace('.', 'p'))
        df_eff = pd.read_csv(file_eff)

        EP = 'EPD'
        resolution = unumpy.uarray(df_res[f'{EP}_res'].values, df_res[f'{EP}_res_err'].values)
        piplus_v2 = unumpy.uarray(df[f'piplus_v2_{EP}'].values, df[f'piplus_v2_err_{EP}'].values) / resolution
        piminus_v2 = unumpy.uarray(df[f'piminus_v2_{EP}'].values, df[f'piminus_v2_err_{EP}'].values) / resolution
        piplus_v2_eff = unumpy.uarray(df_eff[f'piplus_v2_{EP}'].values, df_eff[f'piplus_v2_err_{EP}'].values) / resolution
        piminus_v2_eff = unumpy.uarray(df_eff[f'piminus_v2_{EP}'].values, df_eff[f'piminus_v2_err_{EP}'].values) / resolution
        antiproton_v2 = unumpy.uarray(df[f'antiproton_v2_{EP}'].values, df[f'antiproton_v2_err_{EP}'].values) / resolution
        antiproton_v2_eff = unumpy.uarray(df_eff[f'antiproton_v2_{EP}'].values, df_eff[f'antiproton_v2_err_{EP}'].values) / resolution
        x = np.array([75., 65., 55., 45., 35., 25., 15., 7.5, 2.5])

        shift = -1
        shift_2 = -0.5
        ax_eff[i].errorbar(x+shift-shift_2, unumpy.nominal_values(piplus_v2_eff), unumpy.std_devs(piplus_v2_eff), fmt='o', capsize=2, ms=8, mfc='white', color='C0')
        ax_eff[i].errorbar(x+shift+shift_2, unumpy.nominal_values(piplus_v2), unumpy.std_devs(piplus_v2), fmt='o', capsize=2, ms=8, color='C0')
        ax_eff[i].errorbar(x-shift-shift_2, unumpy.nominal_values(antiproton_v2_eff), unumpy.std_devs(antiproton_v2_eff), fmt='s', capsize=2, ms=8, mfc='white', color='C1')
        ax_eff[i].errorbar(x-shift+shift_2, unumpy.nominal_values(antiproton_v2), unumpy.std_devs(antiproton_v2), fmt='s', capsize=2, ms=8, color='C1')
        ax_eff[i].annotate(r'$\sqrt{s_{\text{NN}}}=$' + energy, xy=(0.15, 0.9), fontsize=15, xycoords='axes fraction', horizontalalignment='left')
    fig_eff.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel(r'$\text{Centrality (%)}$', fontsize=15)
    plt.ylabel(r'$v_2$', fontsize=15, labelpad=20)
    ax_eff[7].legend([r'$\pi^+$', r'$\pi^+$ (eff)', r'$\bar{p}$', r'$\bar{p}$ (eff)'], fontsize=15, frameon=False)
    plt.savefig(paper_plots_path + '/efficiency_comparison.pdf')
    plt.savefig(paper_plots_path + '/efficiency_comparison.eps', format='eps')
    plt.savefig(paper_plots_path + '/efficiency_comparison.svg', format='svg')
    figs.append(fig_eff)
    plt.close()
    return figs

def merge_helper(dict_arr, bin_groups):
    dict_v2 = dict_arr
                 
    for key, arr in dict_v2.items():
        temp = copy.copy(arr)
        # set the bins needed to be merged to 0, except the first
        last = 0 # number of bins merged in the last iteration
        start = 999 # start index of the bins to be merged
        for bins in bin_groups:
            # if the last bin to be merged is before the start of the previous
            # group, there is no need to adjust the bins
            if bins[-1] < start:
                last = 0
            for b in bins[1:]:
                temp[b-1-last] = ufloat(0, 0)
        
            temp = temp[unumpy.nominal_values(temp) != 0]
            # expand the array to 2xn or 3xn, n is the number of bins to be merged, 
            # other rows are set to be the same as the first row
            temp_2D = np.zeros((len(bins), len(temp)))*ufloat(0, 0)
            for b in range(len(bins)):
                temp_2D[b,:] = temp
                if b > 0:
                    temp_2D[b, bins[0]-1] = arr[bins[b]-1]
        
            # now average along the first axis (noting that these are measurements with uncertainties)
            # (so their weights are taken into account)
            weights = 1 / unumpy.std_devs(temp_2D)**2
            temp = np.average(temp_2D, axis=0, weights=weights)
            last = len(bins) - 1
            start = bins[0]
        # temp = np.mean(temp_2D, axis=0)
        # we need to adjust the error for the non-merged bins, as they are sqrt(2) smaller
        # actually, we don't need to since uncertainties package takes care of correlation
        dict_v2[key] = temp    
    return dict_v2

def neutron_skin_alpha_test(dict_input, figs, paper_plot_path):
    """
    Per-energy fit of interpolation parameter alpha between flat d/u and Glauber
    neutron-skin prediction, restricted to 40-80% centrality.
    Model: R(cen; alpha) = R_flat + alpha * (R_Glauber(cen) - R_flat)
      alpha=0 -> flat 315/276, alpha=1 -> full Glauber neutron skin.
    Analytic chi2 minimisation (linear in alpha). Uses stat errors only.
    """
    CEN_GLAUBER = np.array([2.5, 7.5, 15, 25, 35, 45, 55, 65, 75])
    RATIO_DU    = np.array([1.13304, 1.13351, 1.13777, 1.14464, 1.15486,
                             1.17016, 1.19128, 1.22017, 1.26102])
    R_FLAT = 315. / 276.
    CEN_MIN, CEN_MAX = 30., 80.   # 30-80% centrality window

    # Degree-3 polynomial fit to Glauber (max residual ~9e-4, negligible vs data)
    glauber_poly = np.poly1d(np.polyfit(CEN_GLAUBER, RATIO_DU, 3))

    _energy_key = lambda f: float(f.split('/')[-2].split('_')[1].replace('p', '.').replace('GeV', ''))
    files_TPC = sorted([f for f in dict_input['ratio']
                        if '_TPC' in os.path.basename(f)
                        and 'isobar' not in f],
                       key=_energy_key)
    def fit_alpha(f):
        """Return alpha fit dict for a single yaml file, or None if too few bins."""
        with open(f, 'r') as fh:
            d = yaml.load(fh, Loader=yaml.CLoader)
        cen       = np.array(d['x'])
        y         = np.array(d['y'])
        yerr_stat = np.array(d['yerr_stat'])
        mask = (cen >= CEN_MIN) & (cen <= CEN_MAX)
        cen_m, y_m, yerr_m = cen[mask], y[mask], yerr_stat[mask]
        n = len(cen_m)
        if n < 2:
            return None
        g     = glauber_poly(cen_m) - R_FLAT
        delta = y_m - R_FLAT
        w     = 1. / yerr_m**2
        alpha   = np.sum(delta * g * w) / np.sum(g**2 * w)
        sigma_a = 1. / np.sqrt(np.sum(g**2 * w))
        signif  = alpha / sigma_a
        model_m = R_FLAT + alpha * g
        return dict(alpha=alpha, sigma=sigma_a, signif=signif, n=n,
                    chi2_flat=np.sum(delta**2 * w),
                    chi2_glauber=np.sum((y_m - glauber_poly(cen_m))**2 * w),
                    chi2_best=np.sum((y_m - model_m)**2 * w))

    # --- Per-energy analytic alpha fit (30-80%, TPC only) ---
    results_TPC = {}
    print(f'\n===== Neutron Skin Alpha Test ({CEN_MIN:.0f}-{CEN_MAX:.0f}%, TPC) =====')
    for f in files_TPC:
        energy = f.split('/')[-2].split('_')[1].replace('p', '.')
        res = fit_alpha(f)
        results_TPC[energy] = res
        if res is None:
            print(f'  {energy:8s}: too few bins (n<2), skipping')
        else:
            print(f'  {energy:8s}: alpha={res["alpha"]:.3f}+/-{res["sigma"]:.3f} ({res["signif"]:+.2f}sigma)  '
                  f'chi2/ndf: flat={res["chi2_flat"]/res["n"]:.2f}  Glauber={res["chi2_glauber"]/res["n"]:.2f}  '
                  f'best={res["chi2_best"]/(res["n"]-1):.2f}  (n={res["n"]})')
    print('============================================\n')

    # --- Plot (same format as ratio.pdf) ---
    fig_ns = plt.figure(figsize=(16, 8))
    gs_ns  = fig_ns.add_gridspec(ncols=4, nrows=2, hspace=0.0, wspace=0.0)
    ax_ns  = gs_ns.subplots(sharex='col', sharey='row')
    ax_ns  = ax_ns.flatten()

    cen_smooth = np.linspace(35, 85, 200)
    lb, rb = 35., 85.

    for i, f in enumerate(files_TPC):
        energy = f.split('/')[-2].split('_')[1].replace('p', '.')
        ax = ax_ns[i]
        with open(f, 'r') as fh:
            d = yaml.load(fh, Loader=yaml.CLoader)
        cen       = np.array(d['x'])
        y         = np.array(d['y'])
        yerr_stat = np.array(d['yerr_stat'])
        yerr_sys  = np.array(d['yerr_sys'])
        mask = (cen >= CEN_MIN) & (cen <= CEN_MAX)

        shift = 1.
        ax.errorbar(cen[mask]+shift, y[mask], yerr=yerr_stat[mask], **marker_styles['TPC'])
        for j in np.where(mask)[0]:
            ax.fill_between(np.array([cen[j]-0.5, cen[j]+0.5])+shift,
                            y[j]-yerr_sys[j], y[j]+yerr_sys[j],
                            color=marker_styles['TPC']['color'], alpha=0.3)

        ax.hlines(R_FLAT, lb, rb, color='C3', linestyle='--')
        ax.hlines(1,      lb, rb, color='black', linestyle='--')
        ax.fill_between(cen_smooth,
                        glauber_poly(cen_smooth) - 1e-5,
                        glauber_poly(cen_smooth) + 1e-5,
                        color='C2', alpha=0.8)

        res = results_TPC.get(energy)
        if res is not None:
            model_s = R_FLAT + res['alpha'] * (glauber_poly(cen_smooth) - R_FLAT)
            ax.plot(cen_smooth, model_s, color='C1', lw=1.5, ls='-.')

        ax.annotate(r'$\sqrt{s_{\text{NN}}}=$' + energy,
                    xy=(0.15, 0.9), fontsize=15, xycoords='axes fraction',
                    horizontalalignment='left')
        if res is not None:
            ax.annotate(
                fr'$\alpha={res["alpha"]:.2f}\pm{res["sigma"]:.2f}$ ({res["signif"]:+.1f}$\sigma$)',
                xy=(0.97, 0.78), fontsize=12, xycoords='axes fraction', ha='right')
        ax.set_xlim(lb - 5, rb + 5)
        ax.set_ylim(0.509, 1.759)

    # Last panel: legend only
    ax_last = ax_ns[7]
    ax_last.plot([], [], ' ', label='AuAu')
    ax_last.errorbar([], [], yerr=[], label='TPC Event plane', **marker_styles['TPC'])
    ax_last.fill_between([], [], [], color='C2', alpha=0.8, label='Glauber d/u')
    ax_last.hlines(999, lb, rb, color='C3', linestyle='--', label='315/276')
    ax_last.hlines(999, lb, rb, color='black', linestyle='--', label='1')
    ax_last.plot([], [], color='C1', lw=1.5, ls='-.', label=r'Best-fit $\alpha$')
    ax_last.legend(fontsize=13, frameon=False, loc='center')
    ax_last.set_xlim(lb - 5, rb + 5)
    ax_last.set_ylim(0.509, 1.759)

    fig_ns.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel(r'$\text{Centrality (%)}$', fontsize=15)
    plt.ylabel(r'$\frac{v_2^{\pi^-}-\frac{2}{3}v_2^{\bar{p}}}{v_2^{\pi^+}-\frac{2}{3}v_2^{\bar{p}}}$', fontsize=18)

    plt.figure(fig_ns.number)
    plt.savefig(paper_plot_path + '/neutron_skin_test.pdf')
    plt.savefig(paper_plot_path + '/neutron_skin_test.eps', format='eps')
    figs.append(fig_ns)
    plt.close()
    return figs


def calculate_chi2_per_ndf(data_points, model_points, nparams):
    """
    Calculate chi2 per ndf for the given data points and model points. Use total errors. 
    """
    # chi2_array = (data_points - model_points).value**2 / (data_points.total_error()**2) # total error or stat only?
    chi2_array = (data_points - model_points).value**2 / (data_points.stat_error**2)
    ndf = len(data_points) - nparams
    chi2 = np.sum(chi2_array) / ndf
    return chi2


def main(dict_input, output_file=None):
    paper_plots_path = os.path.dirname(output_file)
    figs = []

    figs = plot_res(dict_input, figs, paper_plots_path)
    figs = plot_pion_v2(dict_input, figs, paper_plots_path)
    figs = plot_efficiency_comparison(dict_input, figs, paper_plots_path)
    figs = plot_ratio(dict_input, figs, paper_plots_path)
    figs = plot_ratio_one_energy(dict_input, figs, paper_plots_path)
    figs = plot_alternative_ratio(dict_input, figs, paper_plots_path)
    figs = plot_alternative_ratio_integrated(dict_input, figs, paper_plots_path)
    figs = plot_energy_dep(dict_input, figs, paper_plots_path)
    figs = plot_energy_dep_lambda(dict_input, figs, paper_plots_path)
    figs = plot_isobar_test(dict_input, figs, paper_plots_path)
    figs = neutron_skin_alpha_test(dict_input, figs, paper_plots_path)

    if output_file is not None:
        pdf = matplotlib.backends.backend_pdf.PdfPages(output_file)
        for fig in figs:
            pdf.savefig(fig)
        pdf.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_res', type=str, nargs='+')
    parser.add_argument('--input_ratio', type=str, nargs='+')
    parser.add_argument('--input_delta_v2', type=str, nargs='+')
    parser.add_argument('--input_delta_v2_isobar', type=str, nargs='+')
    parser.add_argument('--input_v2', type=str, nargs='+')
    parser.add_argument('--input_v2_eff', type=str, nargs='+')
    parser.add_argument('--output', type=str, help='a pdf report that includes all generated plots')
    args = parser.parse_args()

    delta_v2_files = {}
    delta_v2_files['lambda'] = {}
    delta_v2_files['lambda']['TPC'] = [f for f in args.input_delta_v2 if not f.split('/')[-2].split('_')[1].startswith('isobar') and 'TPC' in f and 'lambda' in f]
    delta_v2_files['lambda']['EPD'] = [f for f in args.input_delta_v2 if not f.split('/')[-2].split('_')[1].startswith('isobar') and 'EPD' in f and 'lambda' in f]
    for par_name in ['p', 'k', 'pi']:
        delta_v2_files[par_name] = {}
        delta_v2_files[par_name]['TPC'] = [f.replace('lambda', par_name) for f in delta_v2_files['lambda']['TPC']]
        delta_v2_files[par_name]['EPD'] = [f.replace('lambda', par_name) for f in delta_v2_files['lambda']['EPD']]
    delta_v2_files_isobar = {'pi': {'Ru': {'TPC': [], 'EPD': []}, 'Zr': {'TPC': [], 'EPD': []}}}
    for system in ['Ru', 'Zr']:
        delta_v2_files_isobar['pi'][system]['TPC'] = [f for f in args.input_delta_v2_isobar if f.split('/')[-2].split('energy_')[1] == f'isobar_{system}' and 'TPC' in f and 'pi' in f]
        delta_v2_files_isobar['pi'][system]['EPD'] = [f for f in args.input_delta_v2_isobar if f.split('/')[-2].split('energy_')[1] == f'isobar_{system}' and 'EPD' in f and 'pi' in f]
    dict_input = {'res': args.input_res, 'v2': args.input_v2, 'ratio': args.input_ratio,
                  'v2_eff': args.input_v2_eff, 'delta_v2': delta_v2_files, 'delta_v2_isobar': delta_v2_files_isobar}
    main(dict_input, args.output)

