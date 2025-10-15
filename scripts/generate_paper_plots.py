from importlib.resources import files
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
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
    markers = ['o', 's', 'd', '^', 'v', '<', '>', 'p', 'h', 'H', 'D', 'P', '*', 'X']
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    shift_range = 3 # shift the markers to avoid overlap, 2 means +- 0.5% shift
    for i, f in enumerate(files):
        df = pd.read_csv(f)
        energy = f.split('/')[-2].split('_')[1].replace('p', '.')
        if energy.startswith('isobar'):
            continue
        TPC_res = unumpy.uarray(df['TPC_res'].values, df['TPC_res_err'].values)
        EPD_res = unumpy.uarray(df['EPD_res'].values, df['EPD_res_err'].values)
        # mask non-physical resolution
        EPD_mask = unumpy.nominal_values(EPD_res) > 0

        x = np.array([75, 65, 55, 45, 35, 25, 15, 7.5, 2.5])
        x = x - 0.5 * shift_range + i / (len(files) - 1) * shift_range
        ax_res[0].errorbar(x, unumpy.nominal_values(TPC_res), unumpy.std_devs(TPC_res), label=energy,
                           fmt=markers[i], capsize=2, ms=8, color=colors[i])
        if energy == '14.6GeV':
            continue
        ax_res[1].errorbar(x[EPD_mask], unumpy.nominal_values(EPD_res[EPD_mask]), unumpy.std_devs(EPD_res[EPD_mask]),
                           fmt=markers[i], capsize=2, ms=8, color=colors[i])
    
    fig_res.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel(r'$\text{Centrality (%)}$', fontsize=15)
    plt.ylabel(r'$\text{Res}(\Psi_{EP})$', fontsize=15, labelpad=20)
    ax_res[0].legend(fontsize=12, frameon=False)
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
            piplus_v2 = unumpy.uarray(df[f'piplus_v2_{EP}'].values, df[f'piplus_v2_err_{EP}'].values) / resolution
            piminus_v2 = unumpy.uarray(df[f'piminus_v2_{EP}'].values, df[f'piminus_v2_err_{EP}'].values) / resolution
            antiproton_v2 = unumpy.uarray(df[f'antiproton_v2_{EP}'].values, df[f'antiproton_v2_err_{EP}'].values) / resolution
            x = np.array([75., 65., 55., 45., 35., 25., 15., 7.5, 2.5])  
            x = unumpy.uarray(x, np.zeros_like(x))

            x = unumpy.nominal_values(x)
            shift_EP = 1. if EP == 'TPC' else -1.
            shift_par = 0.5
            color = 'C0' if EP == 'TPC' else 'C1'
            # piminus marker is filled, piplus marker is not filled
            ax_coal[i].errorbar(x+shift_EP+shift_par, unumpy.nominal_values(piplus_v2), unumpy.std_devs(piplus_v2), label=r'$\pi^+$', mfc='white', **marker_styles[EP])
            ax_coal[i].errorbar(x+shift_EP-shift_par, unumpy.nominal_values(piminus_v2), unumpy.std_devs(piminus_v2), label=r'$\pi^-$', **marker_styles[EP])
            # ax_coal[i].errorbar(x+shift_EP, unumpy.nominal_values(antiproton_v2), unumpy.std_devs(antiproton_v2), label=r'$\bar{p}$', 
            #                     **{key:val for key, val in marker_styles[EP].items() if key != 'marker'}, marker='d')
            ax_coal[i].annotate(r'$\sqrt{s_{\text{NN}}}=$' + energy, xy=(0.15, 0.9), fontsize=15, xycoords='axes fraction', horizontalalignment='left')
            ax_coal[i].set_xlim(-5, 85)
            ax_coal[i].set_ylim(0, 0.07)
            lb, rb = ax_coal[i].get_xlim()
    fig_coal.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel(r'$\text{Centrality (%)}$', fontsize=15)
    plt.ylabel(r'$v_2$', fontsize=15, labelpad=20)
    ax_coal[7].errorbar([], [], yerr=[], fmt='o', capsize=2, ms=8, color='C0', label=r'$v_{2,\text{TPC}}^{\pi^-}$')
    ax_coal[7].errorbar([], [], yerr=[], fmt='o', capsize=2, ms=8, color='C0', mfc='white', label=r'$v_{2,\text{TPC}}^{\pi^+}$')
    ax_coal[7].errorbar([], [], yerr=[], fmt='o', capsize=2, ms=8, color='C1', label=r'$v_{2,\text{EPD}}^{\pi^-}$')
    ax_coal[7].errorbar([], [], yerr=[], fmt='o', capsize=2, ms=8, color='C1', mfc='white', label=r'$v_{2,\text{EPD}}^{\pi^+}$')    
    ax_coal[7].tick_params(axis='x', which='both', length=0)
    ax_coal[7].legend(fontsize=15, frameon=False, loc='center')   

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
    
    # remove isobar files
    files = [f for f in dict_input['v2'] if not f.split('/')[-2].split('_')[1].startswith('isobar')]
    resfiles = [f for f in dict_input['res'] if not f.split('/')[-2].split('_')[1].startswith('isobar')]

    files = {}
    for EP in ['TPC', 'EPD']:
        files[EP] = [f for f in dict_input['ratio'] if f.split('/')[-1].split('_')[1].startswith(EP)]

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
            ratio = unumpy.uarray(data_dict['y'], data_dict['yerr'])

            shift = 1. if EP == 'TPC' else -1.
            ax_coal[i].errorbar(x+shift, unumpy.nominal_values(ratio), unumpy.std_devs(ratio), **marker_styles[EP])
            ax_coal[i].annotate(r'AuAu, $\sqrt{s_{\text{NN}}}=$' + energy, xy=(0.85, 0.9), fontsize=15, xycoords='axes fraction', horizontalalignment='right')
            ax_coal[i].set_xlim(-5, 85)
            lb, rb = ax_coal[i].get_xlim()
            ax_coal[i].set_ylim(0.509, 1.759)
            ax_coal[i].hlines(315 / 276, lb, rb, color='C3', label='315/276', linestyle='--')
            ax_coal[i].hlines(1, lb, rb, color='black', label='1', linestyle='--')
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
    ax_coal[7].errorbar([], [], yerr=[], label='TPC Event plane', **marker_styles['TPC'])
    ax_coal[7].errorbar([], [], yerr=[], label='EPD Event plane', **marker_styles['EPD'])
    # ax_coal[7].annotate(r'$\bf{STAR}\;\it{Preliminary}$', xy=(0.15, 0.8), xycoords='axes fraction', fontsize=20)
    ax_coal[7].fill_between([], [], [], color='C2', alpha=0.8, label='Glauber d/u')
    ax_coal[7].hlines(999, lb, rb, color='C3', linestyle='--', label='315/276')
    ax_coal[7].hlines(999, lb, rb, color='black', linestyle='--', label='1')
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
    shutil.copyfile(paper_plot_path + '/ratio.pdf', paper_plot_path + '/ratio_old.pdf')
    plt.savefig(paper_plot_path + '/ratio.pdf')
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
    files = [f for f in dict_input['v2'] if not f.split('/')[-2].split('_')[1].startswith('isobar')]
    resfiles = [f for f in dict_input['res'] if not f.split('/')[-2].split('_')[1].startswith('isobar')]
    fig_dep, ax_dep = plt.subplots(figsize=(8, 6))

    ratio_1040_TPC = np.zeros(len(files)) * ufloat(0, 0)
    ratio_1040_EPD = np.zeros(len(files)) * ufloat(0, 0)

    for i, (f, fres) in enumerate(zip(files, resfiles)):
        df = pd.read_csv(f)
        df_res = pd.read_csv(fres)
        energy = f.split('/')[-2].split('_')[1].replace('p', '.')
        for EP in ['TPC', 'EPD']:
            resolution = unumpy.uarray(df_res[f'{EP}_res'].values, df_res[f'{EP}_res_err'].values)
            piplus_v2 = unumpy.uarray(df[f'piplus_v2_{EP}'].values, df[f'piplus_v2_err_{EP}'].values) / resolution
            piminus_v2 = unumpy.uarray(df[f'piminus_v2_{EP}'].values, df[f'piminus_v2_err_{EP}'].values) / resolution
            antiproton_v2 = unumpy.uarray(df[f'antiproton_v2_{EP}'].values, df[f'antiproton_v2_err_{EP}'].values) / resolution
            x = np.array([75., 65., 55., 45., 35., 25., 15., 7.5, 2.5])  
            x = unumpy.uarray(x, np.ones_like(x))

            # merge 10-40% centralities (35%, 25%, 15%)
            bins = np.array([5, 6, 7])
            piminus_sub = piminus_v2 - antiproton_v2 * 2. / 3.
            piplus_sub = piplus_v2 - antiproton_v2 * 2. / 3.
            ratio = piminus_sub / piplus_sub
            # print(f'energy: {energy}, EP: {EP}')
            # print(ratio)
            # dict_ratio = {'x': x, 'ratio': ratio}
            # dict_ratio = merge_helper(dict_ratio, [bins])
            # dict_delta = {'x': x, 'delta': piminus_v2 - piplus_v2}
            # dict_delta = merge_helper(dict_delta, [bins])
            merged_ratio = np.average(unumpy.nominal_values(ratio[bins-1]), weights=1./unumpy.std_devs(ratio[bins-1])**2)
            merged_ratio_err = np.sqrt(1./np.sum(1./unumpy.std_devs(ratio[bins-1])**2))

            x = unumpy.nominal_values(x)
            # now, what does the merged 10-40% bin correspond to?
            if EP == 'TPC':
                ratio_1040_TPC[i] = ufloat(merged_ratio, merged_ratio_err) # dict_ratio['ratio'][4]
            else:
                ratio_1040_EPD[i] = ufloat(merged_ratio, merged_ratio_err) # dict_ratio['ratio'][4]
    energy_str = [f.split('/')[-2].split('_')[1].replace('p', '.') for f in files ]
    energy_float = np.array([float(e.replace('GeV', '')) for e in energy_str])
    # we want to shift the markers to avoid overlap
    # but we are plotting the x-axis in log scale, so we need to shift the markers in log scale
    shift = 0.2 # multiply or divide by 1.5
    ax_dep.errorbar(energy_float + shift, unumpy.nominal_values(ratio_1040_TPC), unumpy.std_devs(ratio_1040_TPC), fmt='o', label='TPC', capsize=2, ms=8)
    # for EPD, show only 14.6 GeV and above
    ax_dep.errorbar((energy_float - shift)[energy_float >= 14.6], unumpy.nominal_values(ratio_1040_EPD)[energy_float >= 14.6], unumpy.std_devs(ratio_1040_EPD)[energy_float >= 14.6], fmt='o', label='EPD', capsize=2, ms=8)
    ax_dep.annotate(r'AuAu, 10-40%', xy=(0.05, 0.9), fontsize=15, xycoords='axes fraction', horizontalalignment='left')
    ax_dep.set_xlabel(r'$\sqrt{s_{\text{NN}}}$ (GeV)', fontsize=15)
    ax_dep.set_ylabel(r'$\frac{v_2^{\pi^-}-\frac{2}{3}v_2^{\bar{p}}}{v_2^{\pi^+}-\frac{2}{3}v_2^{\bar{p}}}$', fontsize=15)

    # print results to txt
    with open(paper_plots_path + '/energy_dep.txt', 'w') as f:
        for i, energy in enumerate(energy_str):
            f.write(f'{energy} {unumpy.nominal_values(ratio_1040_TPC)[i]:.4f} {unumpy.std_devs(ratio_1040_TPC)[i]:.4f} {unumpy.nominal_values(ratio_1040_EPD)[i]:.4f} {unumpy.std_devs(ratio_1040_EPD)[i]:.4f}\n')
    
    # ax_dep.set_xscale('log')
    ax_dep.set_xticks(energy_float, labels=energy_float)
    ax_dep.set_xlim(5, 30)
    ax_dep.set_ylim(1.10, 1.2)
    lb, rb = ax_dep.get_xlim()
    ax_dep.hlines(315 / 276, lb, rb, color='C3', linestyle='--', label='315/276')
    # ax_dep.hlines(1, lb, rb, color='C4', linestyle='--', label='1')
    ax_dep.legend(fontsize=15)
    # make a copy of the plot from the previous iteration
    shutil.copyfile(paper_plots_path + '/energy_dep.pdf', paper_plots_path + '/energy_dep_old.pdf')
    plt.savefig(paper_plots_path + '/energy_dep.pdf')
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
    
    mask = {'EPD': [14.6]}
    
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

                datapoints_lambda = DataPoint(unumpy.nominal_values(delta_lambda), unumpy.std_devs(delta_lambda))
                datapoints_pmk = DataPoint(unumpy.nominal_values(delta_pmk), unumpy.std_devs(delta_pmk))
                datapoints_p = DataPoint(unumpy.nominal_values(delta_p), unumpy.std_devs(delta_p))
                chi2ndf_2 = calculate_chi2_per_ndf(datapoints_lambda - datapoints_p, DataPoint(np.zeros(len(datapoints_lambda))), nparams=0)
                chi2ndf_1 = calculate_chi2_per_ndf(datapoints_lambda - datapoints_pmk, DataPoint(np.zeros(len(datapoints_lambda))), nparams=0)
                energy_str = [f.split('/')[-2].split('_')[1].replace('p', '.') for f in resfiles]
                energy_float = np.array([float(e.replace('GeV', '')) for e in energy_str])

                # masking
                current_mask = np.where(np.array([energy_float[energy_index] not in mask.get(EP, []) for energy_index in range(len(files_lambda))]))[0]
                delta_lambda = delta_lambda[current_mask]
                delta_pmk = delta_pmk[current_mask]
                delta_p = delta_p[current_mask]
                energy_float = energy_float[current_mask]

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
    figs = plot_energy_dep(dict_input, figs, paper_plots_path)
    figs = plot_energy_dep_lambda(dict_input, figs, paper_plots_path)
    figs = plot_isobar_test(dict_input, figs, paper_plots_path)

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

