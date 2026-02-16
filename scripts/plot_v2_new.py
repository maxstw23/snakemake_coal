import numpy as np
import platform
from uncertainties import unumpy
from uncertainties import ufloat
import uproot
import yaml
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import copy
from scipy.special import iv
from scipy.optimize import minimize
from iminuit import cost, Minuit
from simple_profile import SimpleProfile


# Before running:
# 1. Use v2_eff_correction.cpp script to correctly apply efficiency correction (note the pT range!)
# 2. Copy the output to the specified place below
# 3. Check the path contains the correct file name (with resolution profiles)

def res(chi):
    chi2_4 = chi ** 2 / 4.
    return np.pi ** 0.5 / 2 / 2 ** 0.5 * chi * np.exp(-chi2_4) * (iv(0, chi2_4) + iv(1, chi2_4))


def cosavg_to_res(cosavg):
    def func(x):
        return (res(x) - cosavg ** 0.5) ** 2

    m = minimize(func, x0=np.array([0.01]))
    return res(m.x[0] * 2 ** 0.5)


def autoscale_helper(arr, arr_err):
    # determine the proper y limits in the case of large errors
    median_err = np.median(arr_err)
    # if any error is larger than 10 times the median error, omit it for the purpose of autoscaling
    if np.any(arr_err > 10 * median_err):
        arr_err = np.where(arr_err > 10 * median_err, 0, arr_err)
    fig_temp, ax_temp = plt.subplots(1, 1)
    ax_temp.errorbar(np.arange(len(arr)), arr, arr_err, ls='none')
    lb, rb = ax_temp.get_ylim()
    plt.close(fig_temp)
    return lb, rb

def main(inputFile, inputFile_lambda, inputFile_lambdabar, resFile, outputDir, energy, yrange):
    df = pd.read_csv(inputFile)
    df_cen = [pd.read_csv(inputFile.replace('.csv', f'_cen{i+1}.csv')) for i in range(9)]
    df_res = pd.read_csv(resFile)
    print(df.keys())
    print(type(df['piminus_counts'].values))
    particles = ['piplus', 'piminus', 'kplus', 'kminus', 'proton', 'antiproton', 'lambda', 'lambdabar']
    nquarks = {'piplus': 2, 'piminus': 2, 'kplus': 2, 'kminus': 2, 'proton': 3, 'antiproton': 3, 'lambda': 3, 'lambdabar': 3}
    particles_latex = [r'$\pi^{+}$', r'$\pi^{-}$', r'$K^{+}$', r'$K^{-}$', r'$p$', r'$\bar{p}$', r'$\Lambda$', r'$\bar{\Lambda}$']

    # read lambda
    no_lambda = False
    try:
        df_lambda = pd.read_csv(inputFile_lambda, header=[0, 1], index_col=0)
        df_lambdabar = pd.read_csv(inputFile_lambdabar, header=[0, 1], index_col=0)
        lambda_v2_dict = {'EPD': {}, 'TPC': {}}
        lambdabar_v2_dict = {'EPD': {}, 'TPC': {}}

        for cen in df_lambda.columns.levels[0]:
            lambda_v2_dict['EPD'][int(cen)] = {s: df_lambda.loc[:, (cen, s)].values for s in ['values', 'errors', 'counts']}
            lambdabar_v2_dict['EPD'][int(cen)] = {s: df_lambdabar.loc[:, (cen, s)].values for s in ['values', 'errors', 'counts']}
        df_lambda_TPC = pd.read_csv(inputFile_lambda.replace('EPD', 'TPC'), header=[0, 1], index_col=0)
        df_lambdabar_TPC = pd.read_csv(inputFile_lambdabar.replace('EPD', 'TPC'), header=[0, 1], index_col=0)
        for centrality in df_lambda_TPC.columns.levels[0]:
            lambda_v2_dict['TPC'][int(centrality)] = {s: df_lambda_TPC.loc[:, (centrality, s)].values for s in ['values', 'errors', 'counts']}
            lambdabar_v2_dict['TPC'][int(centrality)] = {s: df_lambdabar_TPC.loc[:, (centrality, s)].values for s in ['values', 'errors', 'counts']}
    except (FileNotFoundError, pd.errors.ParserError, KeyError):
        print('No lambda data found, skipping lambda plots')
        no_lambda = True
    merged_bins = {
        '7.7GeV': {'EPD': [[7,8,9],[3,4]]},
         '9.2GeV': {'EPD': [[1,2,3]]},
        '11.5GeV': {'TPC': [[1,2]], 'EPD': [[1,2]]},
        '14.6GeV': {'TPC': [[1,2]], 'EPD': [[1,2]]},
        '17.3GeV': {'TPC': [[1,2]], 'EPD': [[1,2,3]]},
        '19.6GeV': {'TPC': [[1,2]], 'EPD': [[1,2]]},
        '27GeV': {'TPC': [[1,2,3]], 'EPD': [[1,2,3]]},
    }

    # masked_bins = {
    #     '7.7GeV': {'EPD': [3]},
    #     '9.2GeV': {'EPD': [1]}
    # }
    # mask all EPD bins for 7.7, 9.2 and 11.5 GeV
    masked_bins = {
        '7.7GeV': {'EPD': [1, 2, 3, 4, 5, 6]},
        '9.2GeV': {'EPD': [1, 2, 3, 4, 5, 6, 7]},
        '11.5GeV': {'EPD': [1, 2, 3, 4, 5, 6, 7, 8]},
        '17.3GeV': {'EPD': [1]},
        '19.6GeV': {'EPD': [1, 8]},
        '27GeV': {'EPD': [1]},
    }

    fig_c, ax_c = plt.subplots(1, 1, figsize=(8, 6))
    fig_dpi, ax_dpi = plt.subplots(1, 1, figsize=(8, 6))
    fig_dp, ax_dp = plt.subplots(1, 1, figsize=(8, 6))
    for i, EP in enumerate(['TPC', 'EPD']):
        resolution = unumpy.uarray(df_res[f'{EP}_res'].values, df_res[f'{EP}_res_err'].values)
        piplus_v2 = unumpy.uarray(df[f'piplus_v2_{EP}'].values, df[f'piplus_v2_err_{EP}'].values) / resolution
        piplus_counts = df[f'piplus_counts'].values
        piminus_v2 = unumpy.uarray(df[f'piminus_v2_{EP}'].values, df[f'piminus_v2_err_{EP}'].values) / resolution
        piminus_counts = df[f'piminus_counts'].values
        proton_v2 = unumpy.uarray(df[f'proton_v2_{EP}'].values, df[f'proton_v2_err_{EP}'].values) / resolution
        proton_counts = df[f'proton_counts'].values
        antiproton_v2 = unumpy.uarray(df[f'antiproton_v2_{EP}'].values, df[f'antiproton_v2_err_{EP}'].values) / resolution
        antiproton_counts = df[f'antiproton_counts'].values
        x = np.array([75., 65., 55., 45., 35., 25., 15., 7.5, 2.5])  
        x_edges = np.array([80., 70., 60., 50., 40., 30., 20., 10., 5., 0.])
        x_original = np.array([75., 65., 55., 45., 35., 25., 15., 7.5, 2.5])
        x = unumpy.uarray(x, np.ones_like(x))
        energy = energy.replace('p', '.')

        # merge certain cen bins for final measurments (not for the cen-dependent plots)
        cen_bins_correspondence = {'0-10%': np.array([8,9]), '10-40%': np.array([5,6,7]), '40-80%': np.array([1,2,3,4])}
        piplus_v2_merged = {}
        piminus_v2_merged = {}
        antiproton_v2_merged = {}
        for cb in cen_bins_correspondence.keys():
            piplus_v2_merged[cb] = ufloat(-999, 999)
            piminus_v2_merged[cb] = ufloat(-999, 999)
            antiproton_v2_merged[cb] = ufloat(-999, 999)
            if EP != 'EPD' or energy not in ['7.7GeV', '9.2GeV', '11.5GeV']:
                piplus_v2_merged[cb] = np.sum(piplus_v2[cen_bins_correspondence[cb]-1] * piplus_counts[cen_bins_correspondence[cb]-1]) / np.sum(piplus_counts[cen_bins_correspondence[cb]-1])
                piminus_v2_merged[cb] = np.sum(piminus_v2[cen_bins_correspondence[cb]-1] * piminus_counts[cen_bins_correspondence[cb]-1]) / np.sum(piminus_counts[cen_bins_correspondence[cb]-1])
                antiproton_v2_merged[cb] = np.sum(antiproton_v2[cen_bins_correspondence[cb]-1] * antiproton_counts[cen_bins_correspondence[cb]-1]) / np.sum(antiproton_counts[cen_bins_correspondence[cb]-1])

        if energy in merged_bins:
            if EP in merged_bins[energy]:
                bins = merged_bins[energy][EP]
                dict_v2 = {'x': x, 'pip': piplus_v2, 'pim': piminus_v2, 'p': proton_v2, 'ap': antiproton_v2}
                ### TEST ###
                # dict_v2 = {'x': x, 'pip': piplus_v2, 'pim': piminus_v2, 'ap': antiproton_v2, 'ap_pip': antiproton_pip_v2, 'ap_pim': antiproton_pim_v2}
                dict_v2 = merge_helper(dict_v2, bins)
                x = dict_v2['x']

                mask_edges = np.ones(len(x_edges), dtype=bool)
                flatten_merged_bins = []
                for b in bins:
                    flatten_merged_bins.extend(b[1:])
                mask_edges[np.array(flatten_merged_bins)-1] = False
                x_edges = x_edges[mask_edges]

                piplus_v2 = dict_v2['pip']
                piminus_v2 = dict_v2['pim']
                proton_v2 = dict_v2['p']
                antiproton_v2 = dict_v2['ap']

        ### Fill masked bins where resolution is invalid (+-nan or inf, or negative)
        index_masked = np.isnan(unumpy.nominal_values(resolution)) | np.isinf(unumpy.nominal_values(resolution)) | (unumpy.nominal_values(resolution) < 0)
        print(f'Energy: {energy}, EP: {EP}')
        print(f'    masked bins: {np.where(index_masked)[0]}')
        print(f'    merged bins: {merged_bins.get(energy, {}).get(EP, [])}')
        print(f'    x_original[np.where(index_masked)[0]]: {x_original[np.where(index_masked)[0]]}')
        print(f'    x_edges: {x_edges}')
        if len(np.where(index_masked)[0]) > 0:
            index_masked_merged = np.digitize(x_original[np.where(index_masked)[0]], x_edges)
            if energy not in masked_bins:
                masked_bins[energy] = {}
            if EP not in masked_bins[energy]:
                masked_bins[energy][EP] = []
            print(f'    index_masked_merged: {index_masked_merged}')
            masked_bins[energy][EP].extend(index_masked_merged.tolist())

        # now deal with masked bins
        if energy in masked_bins:
            if EP in masked_bins[energy]:
                bins = masked_bins[energy][EP]
                for b in bins:
                    x[b-1] = ufloat(0, 0)
                    piplus_v2[b-1] = ufloat(0, 0)
                    piminus_v2[b-1] = ufloat(0, 0)
                    proton_v2[b-1] = ufloat(0, 0)
                    antiproton_v2[b-1] = ufloat(0, 0)
                x = x[unumpy.nominal_values(x) != 0]
                piplus_v2 = piplus_v2[unumpy.nominal_values(piplus_v2) != 0]
                piminus_v2 = piminus_v2[unumpy.nominal_values(piminus_v2) != 0]
                proton_v2 = proton_v2[unumpy.nominal_values(proton_v2) != 0]
                antiproton_v2 = antiproton_v2[unumpy.nominal_values(antiproton_v2) != 0]

        x = unumpy.nominal_values(x)
        print(f'    x after masking and merging: {x}')
        cen = x

        fig_coal, ax_coal = plt.subplots(1, 3, figsize=(18, 4.8))
        ax_coal[0].errorbar(cen, unumpy.nominal_values(piminus_v2), unumpy.std_devs(piminus_v2), ls='none')
        ax_coal[0].scatter(cen, unumpy.nominal_values(piminus_v2), s=20, label=r'$\pi^{-}$')
        ax_coal[0].errorbar(cen, unumpy.nominal_values(piplus_v2), unumpy.std_devs(piplus_v2), ls='none')
        ax_coal[0].scatter(cen, unumpy.nominal_values(piplus_v2), s=20, label=r'$\pi^{+}$')
        ax_coal[0].errorbar(cen, unumpy.nominal_values(antiproton_v2), unumpy.std_devs(antiproton_v2), ls='none')
        ax_coal[0].scatter(cen, unumpy.nominal_values(antiproton_v2), s=20, label=r'$\bar{p}$')
        # print some values in reverse
        print(f'    piminus_v2: {piminus_v2[::-1]}')
        print(f'    piplus_v2: {piplus_v2[::-1]}')
        print(f'    antiproton_v2: {antiproton_v2[::-1]}')

        ### TEST ###
        # ax_coal[0].scatter(cen, unumpy.nominal_values(antiproton_pim_v2), s=20, label=r'$\bar{p} \pi^{-}$')
        # ax_coal[0].scatter(cen, unumpy.nominal_values(antiproton_pip_v2), s=20, label=r'$\bar{p} \pi^{+}$')
        ax_coal[0].set_xlabel('Centrality (%)', fontsize=16)
        ax_coal[0].set_ylabel(r'$v_2$', fontsize=16)
        ax_coal[0].tick_params(axis='x', labelsize=16)
        ax_coal[0].tick_params(axis='y', labelsize=16)
        ax_coal[0].legend(fontsize=20)

        pim_subtracted = piminus_v2 - antiproton_v2 * 2. / 3.
        pip_subtracted = piplus_v2 - antiproton_v2 * 2. / 3.
        ### TEST ###
        # pim_subtracted = piminus_v2 - antiproton_pim_v2 * 2. / 3.
        # pip_subtracted = piplus_v2 - antiproton_pip_v2 * 2. / 3.
        ax_coal[1].errorbar(cen, unumpy.nominal_values(pim_subtracted), unumpy.std_devs(pim_subtracted), ls='none')
        ax_coal[1].scatter(cen, unumpy.nominal_values(pim_subtracted), s=20, label=r'$\pi^{-}-\frac{2}{3}\bar{p}$')
        ax_coal[1].errorbar(cen, unumpy.nominal_values(pip_subtracted), unumpy.std_devs(pip_subtracted), ls='none')
        ax_coal[1].scatter(cen, unumpy.nominal_values(pip_subtracted), s=20, label=r'$\pi^{+}-\frac{2}{3}\bar{p}$')
        ax_coal[1].set_xlabel('Centrality (%)', fontsize=16)
        ax_coal[1].set_ylabel(r'$v_2$', fontsize=16)
        ax_coal[1].tick_params(axis='x', labelsize=16)
        ax_coal[1].tick_params(axis='y', labelsize=16)
        ax_coal[1].legend(fontsize=20)

        pim_subtracted_unc = unumpy.uarray(unumpy.nominal_values(pim_subtracted), unumpy.std_devs(pim_subtracted))
        pip_subtracted_unc = unumpy.uarray(unumpy.nominal_values(pip_subtracted), unumpy.std_devs(pip_subtracted))
        # ratio = pim_subtracted_unc / pip_subtracted_unc
        ratio = pim_subtracted / pip_subtracted
        # ratio = (piminus_v2 - piplus_v2) / (piplus_v2 - antiproton_v2 * 2. / 3.) + 1
        ratio_merged = {}
        for cb in cen_bins_correspondence.keys():
            ratio_merged[cb] = ufloat(-999, 999)
            if EP != 'EPD' or energy not in ['7.7GeV', '9.2GeV', '11.5GeV']:
                ratio_merged[cb] = (piminus_v2_merged[cb] - antiproton_v2_merged[cb] * 2. / 3.) / (piplus_v2_merged[cb] - antiproton_v2_merged[cb] * 2. / 3.)

        ax_coal[2].errorbar(cen, unumpy.nominal_values(ratio), unumpy.std_devs(ratio), fmt='o', ls='none',
                            label=r'$\frac{v_2^{\pi^-}-2v_2^{\bar{u}}}{v_2^{\pi^+}-2v_2^{\bar{u}}}$')
        shift = 1. if EP == 'TPC' else -1
        ax_c.errorbar(cen+shift, unumpy.nominal_values(ratio), unumpy.std_devs(ratio), fmt='o', ls='none', label=EP)
        ax_coal[2].set_xlabel('Centrality (%)', fontsize=16)
        ax_coal[2].set_ylabel('Ratio', fontsize=16)
        ax_coal[2].tick_params(axis='x', labelsize=16)
        ax_coal[2].tick_params(axis='y', labelsize=16)
        lb, rb = ax_coal[2].get_xlim()
        ax_coal[2].set_ylim(0.6, 1.5)
        ax_coal[2].hlines(315 / 276, lb, rb, color='C1', label='315/276')
        ax_coal[2].hlines(1, lb, rb, color='C2')
        ax_coal[2].legend(loc='lower left', fontsize=16)

        # ax_coal[2].annotate(
        #     f'Integrated ratio = {np.sum(pim_subtracted_unc) / np.sum(pip_subtracted_unc)}',
        #     xy=(0.1, 0.28), xycoords='axes fraction', fontsize=12)
        # print(piplus_v2_1040)
        # ax_coal[2].annotate(
        #     f'Integrated ratio (10-40%) = {(piminus_v2_1040 - 2. / 3. * antiproton_v2_1040) / (piplus_v2_1040 - 2. / 3. * antiproton_v2_1040)}',
        #     xy=(0.1, 0.22), xycoords='axes fraction', fontsize=12)
        # ax_coal[2].annotate(
        #     f'Integrated ratio (50-80%) = {(piminus_v2_5080 - 2. / 3. * antiproton_v2_5080) / (piplus_v2_5080 - 2. / 3. * antiproton_v2_5080)}',
        #     xy=(0.1, 0.16), xycoords='axes fraction', fontsize=12)

        plt.tight_layout()
        fig_coal.savefig(f'{outputDir}/coal_{EP}.pdf')
        with open(f'{outputDir}/coal_{EP}.yaml', 'w') as f:
            yaml.dump({'x': cen, 
                       'y': unumpy.nominal_values(ratio), 'yerr': unumpy.std_devs(ratio),
                       'y_010': unumpy.nominal_values(ratio_merged['0-10%']), 'yerr_010': unumpy.std_devs(ratio_merged['0-10%']),
                       'y_1040': unumpy.nominal_values(ratio_merged['10-40%']), 'yerr_1040': unumpy.std_devs(ratio_merged['10-40%']),
                       'y_4080': unumpy.nominal_values(ratio_merged['40-80%']), 'yerr_4080': unumpy.std_devs(ratio_merged['40-80%']),
                      }, f, default_flow_style=False)

        # delta pion
        delta_pion = piminus_v2 - piplus_v2
        ax_dpi.errorbar(cen, unumpy.nominal_values(delta_pion), unumpy.std_devs(delta_pion), fmt='o', ls='none', label=EP)

        # delta proton
        delta_proton = proton_v2 - antiproton_v2
        ax_dp.errorbar(cen, unumpy.nominal_values(delta_proton), unumpy.std_devs(delta_proton), fmt='o', ls='none', label=EP)
        
        # an alternative ratio plot
        fig_proton, ax_proton = plt.subplots(1, 1, figsize=(8, 6))
        denom = proton_v2 - antiproton_v2
        numer = 2*pim_subtracted + pip_subtracted
        ratio = numer / denom
        ax_proton.errorbar(cen, unumpy.nominal_values(ratio), unumpy.std_devs(ratio), fmt='o', ls='none', label=EP)
        ax_proton.set_xlabel('Centrality (%)', fontsize=16)
        ax_proton.set_ylabel('Ratio', fontsize=16)
        lb, rb = ax_proton.get_xlim()
        # ax_proton.set_ylim(0.6, 1.5)
        ax_proton.hlines(1, lb, rb, color='C1')
        fig_proton.savefig(f'{outputDir}/coal_proton_{EP}.pdf')
        with open(f'{outputDir}/coal_proton_{EP}.yaml', 'w') as f:
            yaml.dump({'x': cen, 'y': unumpy.nominal_values(ratio), 'yerr': unumpy.std_devs(ratio)}, f, default_flow_style=False)
        # plt.show()

        # another alternative ratio plot
        fig_proton2, ax_proton2 = plt.subplots(1, 1, figsize=(8, 6))
        AmB = piplus_v2 - piminus_v2 # A-B
        f = 0. # fraction of pion from resonance decays
        AmB = AmB / (1-f)
        twoApB = proton_v2 - antiproton_v2 # 2A+B
        A = (AmB + twoApB) / 3.
        B = A - AmB
        ratio = B / A
        ax_proton2.errorbar(cen, unumpy.nominal_values(ratio), unumpy.std_devs(ratio), fmt='o', ls='none', label=EP)
        ax_proton2.set_xlabel('Centrality (%)', fontsize=16)
        ax_proton2.set_ylabel('Ratio', fontsize=16)
        lb, rb = ax_proton2.get_xlim()
        ax_proton2.set_ylim(0.6, 1.5)
        ax_proton2.hlines(1, lb, rb, color='C1')
        fig_proton2.savefig(f'{outputDir}/coal_proton2_{EP}.pdf')
        with open(f'{outputDir}/coal_proton2_{EP}.yaml', 'w') as f:
            yaml.dump({'x': cen, 'y': unumpy.nominal_values(ratio), 'yerr': unumpy.std_devs(ratio)}, f, default_flow_style=False)

        # chat-gpt derived ratio
        fig_x, ax_x = plt.subplots(1, 1, figsize=(8, 6))
        numerator = proton_v2 + 2 * antiproton_v2 - piminus_v2 - 2 * piplus_v2
        denominator = 4 * piplus_v2 + 2 * piminus_v2 - 2 * proton_v2 - antiproton_v2
        ratio = numerator / denominator
        ax_x.errorbar(cen, unumpy.nominal_values(ratio), unumpy.std_devs(ratio), fmt='o', ls='none', label=EP)
        ax_x.set_xlabel('Centrality (%)', fontsize=16)
        ax_x.set_ylabel('Ratio', fontsize=16)
        lb, rb = ax_x.get_xlim()
        # ax_x.set_ylim(0.6, 1.5)
        ax_x.hlines(1, lb, rb, color='C1')
        fig_x.savefig(f'{outputDir}/emfield_d_u_{EP}.pdf')
        with open(f'{outputDir}/emfield_d_u_{EP}.yaml', 'w') as f:
            yaml.dump({'x': cen, 'y': unumpy.nominal_values(ratio), 'yerr': unumpy.std_devs(ratio)}, f, default_flow_style=False)
            
    # v2 vs pT stuff
    delta_IdPar_v2 = {}
    for p, par in enumerate(particles[::2]): 
        par_name = 'pi' if 'pi' in par else 'k' if 'k' in par else 'p' if 'proton' in par else 'lambda' if 'lambda' in par else 'unknown'
        nrebin = 10 if not par.startswith('lambda') else 1
        if par.startswith('lambda') and no_lambda:
            # save a blank yaml file
            with open(f'{outputDir}/lambda_delta_v2_TPC.yaml', 'w') as f:
                yaml.dump({'x': [], 'y': [], 'yerr': []}, f, default_flow_style=False)
            with open(f'{outputDir}/lambda_delta_v2_EPD.yaml', 'w') as f:
                yaml.dump({'x': [], 'y': [], 'yerr': []}, f, default_flow_style=False)
            continue
        delta_IdPar_v2[par_name] = {}
        for i, EP in enumerate(['TPC', 'EPD']):
            delta_IdPar_v2[par_name][EP] = {}
            fig_v2pt = plt.figure(figsize=(24, 18))
            gs_v2pt = fig_v2pt.add_gridspec(3, 3, hspace=0, wspace=0)
            axes_v2pt = gs_v2pt.subplots(sharex=True, sharey=True)
            fig_v2pt_delta = plt.figure(figsize=(24, 18))
            gs_v2pt_delta = fig_v2pt_delta.add_gridspec(3, 3, hspace=0, wspace=0)
            axes_v2pt_delta = gs_v2pt_delta.subplots(sharex=True, sharey=True)
            profile_pt_particle_merged = {'0-10%': None, '10-40%': None, '40-80%': None, '0-80%': None}
            profile_pt_antiparticle_merged = {'0-10%': None, '10-40%': None, '40-80%': None, '0-80%': None}
            profile_pt_delta_merged = {'0-10%': None, '10-40%': None, '40-80%': None, '0-80%': None}
            centrality_correspondence = {'0-10%': [8,9], '10-40%': [5,6,7], '40-80%': [1,2,3,4], '0-80%': list(range(1,10))}
            delta_v2 = np.zeros(9)
            delta_v2_err = np.zeros(9)
            for cen in range(1, 10):
                particle = par
                antiparticle = particles[2*p+1]
                ax = axes_v2pt[(cen-1)//3, (cen-1)%3]
                resolution = unumpy.uarray(df_res[f'{EP}_res'].values, df_res[f'{EP}_res_err'].values)
                
                if not par.startswith('lambda'):
                    pt = np.linspace(0.005, 1.995, 200)
                    v2_raw_particle = df_cen[cen-1][f'{particle}_v2_{EP}'].values[:200]
                    v2_raw_antiparticle = df_cen[cen-1][f'{antiparticle}_v2_{EP}'].values[:200]
                    v2_err_raw_particle = df_cen[cen-1][f'{particle}_v2_err_{EP}'].values[:200]
                    v2_err_raw_antiparticle = df_cen[cen-1][f'{antiparticle}_v2_err_{EP}'].values[:200]
                    v2_count_particle = df_cen[cen-1][f'{particle}_counts'].values[:200]
                    v2_count_antiparticle = df_cen[cen-1][f'{antiparticle}_counts'].values[:200]
                else:
                    pt = np.linspace(0.45, 1.75, 14)
                    v2_raw_particle = lambda_v2_dict[EP][cen]['values']
                    v2_raw_antiparticle = lambdabar_v2_dict[EP][cen]['values']
                    v2_err_raw_particle = lambda_v2_dict[EP][cen]['errors']
                    v2_err_raw_antiparticle = lambdabar_v2_dict[EP][cen]['errors']
                    v2_count_particle = lambda_v2_dict[EP][cen]['counts']
                    v2_count_antiparticle = lambdabar_v2_dict[EP][cen]['counts']

                profile_particle = SimpleProfile(v2_raw_particle, v2_count_particle, v2_err_raw_particle * v2_count_particle ** 0.5, pt, False) / resolution[cen-1]
                profile_antiparticle = SimpleProfile(v2_raw_antiparticle, v2_count_antiparticle, v2_err_raw_antiparticle * v2_count_antiparticle ** 0.5, pt, False) / resolution[cen-1]
                profile_delta = profile_particle.Add2(profile_antiparticle, 1., -1.)
                for key, val in centrality_correspondence.items():
                    if cen in val:
                        if profile_pt_particle_merged[key] is None:
                            profile_pt_particle_merged[key] = copy.deepcopy(profile_particle)
                            profile_pt_antiparticle_merged[key] = copy.deepcopy(profile_antiparticle)
                            profile_pt_delta_merged[key] = copy.deepcopy(profile_delta)
                        else:
                            profile_pt_particle_merged[key] = profile_pt_particle_merged[key] + profile_particle
                            profile_pt_antiparticle_merged[key] = profile_pt_antiparticle_merged[key] + profile_antiparticle
                            profile_pt_delta_merged[key] = profile_pt_delta_merged[key] + profile_delta
                profile_particle.Rebin(nrebin)
                profile_antiparticle.Rebin(nrebin)
                profile_delta_copy = copy.deepcopy(profile_delta)
                profile_delta.Rebin(nrebin)

                pt_merged_particle = profile_particle.bin_centers()
                pt_merged_antiparticle = profile_antiparticle.bin_centers()
                v2_pt_particle = unumpy.uarray(profile_particle.values(), profile_particle.errors())
                v2_pt_antiparticle = unumpy.uarray(profile_antiparticle.values(), profile_antiparticle.errors())
                ax.errorbar(pt_merged_particle, unumpy.nominal_values(v2_pt_particle), unumpy.std_devs(v2_pt_particle), fmt='o', ls='none', label=particles_latex[2*p])
                ax.errorbar(pt_merged_antiparticle, unumpy.nominal_values(v2_pt_antiparticle), unumpy.std_devs(v2_pt_antiparticle), fmt='o', ls='none', label=particles_latex[2*p+1])
                ax.annotate(f'cen {cen}', xy=(0.3, 0.9), xycoords='axes fraction', fontsize=20)
                ax.set_xlim(0.0, 1.999)
                ax.set_ylim(0, 0.199)
                ax.tick_params(axis='x', labelsize=20)
                ax.tick_params(axis='y', labelsize=20)
                if cen == 9:
                    ax.set_xlabel(r'$p_T$ (GeV/c)', fontsize=20)
                    ax.set_ylabel(r'$v_2$', fontsize=20)
                    ax.legend(fontsize=20)

                pt_merged_delta = profile_delta.bin_centers()
                v2_pt_delta = unumpy.uarray(profile_delta.values(), profile_delta.errors())
                ax_delta = axes_v2pt_delta[(cen-1)//3, (cen-1)%3]
                ax_delta.errorbar(pt_merged_delta, unumpy.nominal_values(v2_pt_delta), unumpy.std_devs(v2_pt_delta), fmt='o', ls='none', label=f'{particles_latex[2*p]} - {particles_latex[2*p+1]}')
                ax_delta.axhline(0, color='black', ls='--')

                # we can just do a weighted average (rebin)
                profile_delta_copy.Rebin(len(profile_delta_copy.values()))
                weighted_avg = profile_delta_copy.values()[0]
                # ax_delta.axhline(weighted_avg, color='red', ls='-', label=f'Avg: {weighted_avg:.4f} ± {profile_delta_copy.errors()[0]:.4f}')

                # or we can do a horizontal fit
                def constant_func(x, a):
                    return a + 0*x
                fit_region = (profile_delta.counts() > 1) & (pt_merged_delta > nquarks[par]*0.2) & (pt_merged_delta < nquarks[par]*0.6)
                c = cost.LeastSquares(pt_merged_delta[fit_region], unumpy.nominal_values(v2_pt_delta[fit_region])*100., unumpy.std_devs(v2_pt_delta[fit_region])*100., constant_func)
                # c.loss = 'soft_l1' # less sensitive to outliers
                a_init = weighted_avg*100.
                m = Minuit(c, a=a_init)
                # m.limits['a'] = (-np.max(np.abs(c.y))*10, np.max(np.abs(c.y))*10)

                m.migrad()
                m.hesse()
                fitted_a = m.values['a'] / 100.
                fitted_a_err = m.errors['a'] / 100.
                delta_v2[cen-1] = fitted_a
                delta_v2_err[cen-1] = fitted_a_err
                ax_delta.axhline(fitted_a, color='red', ls='-.', label=f'Fit: {fitted_a:.4f} ± {fitted_a_err:.4f}')

                ax_delta.annotate(f'cen {cen}', xy=(0.3, 0.9), xycoords='axes fraction', fontsize=20)
                ax_delta.set_xlim(0., 1.999)
                ax_delta.set_ylim(-0.099, 0.099)
                ax_delta.tick_params(axis='x', labelsize=20)
                ax_delta.tick_params(axis='y', labelsize=20)
                ax_delta.legend(fontsize=20)
                if cen == 9:
                    ax_delta.set_xlabel(r'$p_T$ (GeV/c)', fontsize=20)
                    ax_delta.set_ylabel(r'$\Delta v_2$', fontsize=20)
                    
            fig_v2pt.savefig(f'{outputDir}/{par_name}_v2_pt_{EP}.pdf')
            fig_v2pt_delta.savefig(f'{outputDir}/{par_name}_delta_v2_pt_{EP}.pdf')
            plt.close(fig_v2pt)
            plt.close(fig_v2pt_delta)

            fig_v2pt_merged = plt.figure(figsize=(24, 18))
            gs_v2pt_merged = fig_v2pt_merged.add_gridspec(2, 2, hspace=0, wspace=0)
            axes_v2pt_merged = gs_v2pt_merged.subplots(sharex=True, sharey=True)
            delta_v2_merged = np.zeros(4)
            delta_v2_merged_err = np.zeros(4)
            for i, (key, profile) in enumerate(profile_pt_particle_merged.items()):
                profile.Rebin(nrebin)
                pt_merged_particle = profile.bin_centers()
                v2_pt_particle = unumpy.uarray(profile.values(), profile.errors())
                profile_antiparticle = profile_pt_antiparticle_merged[key]
                profile_antiparticle.Rebin(nrebin)
                v2_pt_antiparticle = unumpy.uarray(profile_antiparticle.values(), profile_antiparticle.errors())
                pt_merged_antiparticle = profile_antiparticle.bin_centers()
                profile_delta = profile_pt_delta_merged[key]
                profile_delta.Rebin(nrebin)
                pt_merged_delta = profile_delta.bin_centers()
                v2_pt_delta = unumpy.uarray(profile_delta.values(), profile_delta.errors())

                # do the horizontal fit again
                def constant_func(x, a):
                    return a + 0*x
                fit_region = (profile_delta.counts() > 1) & (pt_merged_delta > nquarks[par]*0.2) & (pt_merged_delta < nquarks[par]*0.6)
                c = cost.LeastSquares(pt_merged_delta[fit_region], unumpy.nominal_values(v2_pt_delta[fit_region])*100., unumpy.std_devs(v2_pt_delta[fit_region])*100., constant_func)
                # c.loss = 'soft_l1' # less sensitive to outliers
                a_init = np.mean(unumpy.nominal_values(v2_pt_delta[fit_region]))*100.
                m = Minuit(c, a=a_init)
                # m.limits['a'] = (-np.max(np.abs(c.y))*10, np.max(np.abs(c.y))*10)
                m.migrad()
                m.hesse()
                fitted_a = m.values['a'] / 100.
                fitted_a_err = m.errors['a'] / 100.
                delta_v2_merged[i] = fitted_a
                delta_v2_merged_err[i] = fitted_a_err

                axes_v2pt_merged[i//2, i%2].errorbar(pt_merged_particle, unumpy.nominal_values(v2_pt_particle), unumpy.std_devs(v2_pt_particle), fmt='o', ls='none', label=particles_latex[2*p])
                axes_v2pt_merged[i//2, i%2].errorbar(pt_merged_antiparticle, unumpy.nominal_values(v2_pt_antiparticle), unumpy.std_devs(v2_pt_antiparticle), fmt='o', ls='none', label=particles_latex[2*p+1])
                axes_v2pt_merged[i//2, i%2].errorbar(pt_merged_delta, unumpy.nominal_values(v2_pt_delta), unumpy.std_devs(v2_pt_delta), fmt='o', ls='none', label=f'{particles_latex[2*p]} - {particles_latex[2*p+1]}')
                axes_v2pt_merged[i//2, i%2].axhline(0, color='black', ls='--')
                axes_v2pt_merged[i//2, i%2].axhline(fitted_a, color='red', ls='-.', label=f'Fit: {fitted_a:.4f} ± {fitted_a_err:.4f}')
                axes_v2pt_merged[i//2, i%2].set_xlim(0.0, 1.999)
                axes_v2pt_merged[i//2, i%2].set_ylim(-0.049, 0.199)
                axes_v2pt_merged[i//2, i%2].tick_params(axis='x', labelsize=20)
                axes_v2pt_merged[i//2, i%2].tick_params(axis='y', labelsize=20)
                axes_v2pt_merged[i//2, i%2].set_xlabel(r'$p_T$ (GeV/c)', fontsize=20)
                axes_v2pt_merged[i//2, i%2].set_ylabel(r'$v_2$', fontsize=20)
                axes_v2pt_merged[i//2, i%2].annotate(key, xy=(0.4, 0.9), xycoords='axes fraction', fontsize=20)
                axes_v2pt_merged[i//2, i%2].legend(fontsize=20)

            delta_IdPar_v2[par_name][EP]['split'] = {'x': np.array([75., 65., 55., 45., 35., 25., 15., 7.5, 2.5]),
                                               'y': delta_v2,
                                               'y_err': delta_v2_err}
            delta_IdPar_v2[par_name][EP]['merged'] = {'x': np.array([5., 25., 60., 40.]),
                                                 'x_label': ['0-10%', '10-40%', '40-80%', '0-80%'],
                                                 'y': delta_v2_merged,
                                                 'y_err': delta_v2_merged_err}
            fig_v2pt_merged.savefig(f'{outputDir}/{par_name}_v2_pt_{EP}_merged.pdf')
            with open(f'{outputDir}/{par_name}_delta_v2_{EP}.yaml', 'w') as f:
                yaml.dump({'split': delta_IdPar_v2[par_name][EP]['split'],
                           'merged': delta_IdPar_v2[par_name][EP]['merged']}, f, default_flow_style=False)

    # Glauber ratios
    cen_glauber = np.array([2.5,7.5,15,25,35,45,55,65,75])
    ratio_du = np.array([1.13304,1.13351,1.13777,1.14464,1.15486,1.17016,1.19128,1.22017,1.26102])
    ratio_du_err = np.array([1.56853e-08,4.95918e-08,9.90949e-08,1.74393e-07,3.19487e-07,5.92654e-07,1.24839e-06,3.09313e-06,5.11625e-06])
    ratio_upper = ratio_du + ratio_du_err
    ratio_lower = ratio_du - ratio_du_err
    ax_c.fill_between(cen_glauber, ratio_lower, ratio_upper, color='C2', alpha=0.8, label='Glauber')
    # ax_c.errorbar(cen_glauber, ratio_du, ratio_du_err, fmt='o', ls='none', label='Glauber')
    ax_c.set_xlabel('Centrality (%)', fontsize=16)
    ax_c.set_ylabel('Ratio', fontsize=16)
    ax_c.annotate(energy.replace('p', '.'), xy=(0.1, 0.9), xycoords='axes fraction', fontsize=16)
    ax_c.tick_params(axis='x', labelsize=16)
    ax_c.tick_params(axis='y', labelsize=16)
    lb, rb = ax_c.get_xlim()
    if yrange is not None:
        print(yrange)
        ax_c.set_ylim(*yrange)
    else:
        ax_c.set_ylim(autoscale_helper(unumpy.nominal_values(ratio), unumpy.std_devs(ratio))) # this uses the EPD ratios, which should have larger errors
    if energy == 'isobar_Zr':
        ax_c.hlines(152 / 136, lb, rb, color='C1', ls='-.', label='152/136')
    elif energy == 'isobar_Ru':
        ax_c.hlines(148 / 140, lb, rb, color='C1', ls='-.', label='148/140')
    else:
        ax_c.hlines(315 / 276, lb, rb, color='C2', ls='-.', label='315/276')
    ax_c.hlines(1, lb, rb, color='black', ls='--')
    ax_c.legend(loc='lower center', fontsize=20)
    fig_c.savefig(f'{outputDir}/coal_combined.pdf')
    # plt.show()

    # delta pion
    ax_dpi.set_xlabel('Centrality (%)', fontsize=16)
    ax_dpi.set_ylabel(r'$\Delta v_2$', fontsize=16)
    ax_dpi.tick_params(axis='x', labelsize=16)
    ax_dpi.tick_params(axis='y', labelsize=16)
    ax_dpi.annotate(energy.replace('p', '.'), xy=(0.1, 0.9), xycoords='axes fraction', fontsize=16)
    ax_dpi.annotate(r'$v_2^{\pi^-}-v_2^{\pi^+}$', xy=(0.1, 0.8), xycoords='axes fraction', fontsize=16)
    ax_dpi.legend(fontsize=16)
    ax_dpi.hlines(0, *ax_dpi.get_xlim(), color='black', ls='--')
    plt.tight_layout()
    fig_dpi.savefig(f'{outputDir}/delta_pion.pdf')
    with open(f'{outputDir}/delta_pion.yaml', 'w') as f:
        yaml.dump({'x': cen, 'y': unumpy.nominal_values(delta_pion), 'yerr': unumpy.std_devs(delta_pion)}, f, default_flow_style=False)

    # delta proton
    ax_dp.set_xlabel('Centrality (%)', fontsize=16)
    ax_dp.set_ylabel(r'$\Delta v_2$', fontsize=16)
    ax_dp.tick_params(axis='x', labelsize=16)
    ax_dp.tick_params(axis='y', labelsize=16)
    ax_dp.annotate(energy.replace('p', '.'), xy=(0.1, 0.9), xycoords='axes fraction', fontsize=16)
    ax_dp.annotate(r'$v_2^{p}-v_2^{\bar{p}}$', xy=(0.1, 0.8), xycoords='axes fraction', fontsize=16)
    ax_dp.legend(fontsize=16)
    ax_dp.hlines(0, *ax_dp.get_xlim(), color='black', ls='--')
    plt.tight_layout()
    fig_dp.savefig(f'{outputDir}/delta_proton.pdf')
    with open(f'{outputDir}/delta_proton.yaml', 'w') as f:
        yaml.dump({'x': cen, 'y': unumpy.nominal_values(delta_proton), 'yerr': unumpy.std_devs(delta_proton)}, f, default_flow_style=False)

    # delta lambda comparison
    if not no_lambda:
        for EP in ['TPC', 'EPD']:
            fig_dl, ax_dl = plt.subplots(1, 1, figsize=(8, 6))
            x = delta_IdPar_v2['lambda'][EP]['split']['x']
            delta_lambda = delta_IdPar_v2['lambda'][EP]['split']['y']
            delta_lambda_err = delta_IdPar_v2['lambda'][EP]['split']['y_err']

            delta_proton = delta_IdPar_v2['p'][EP]['split']['y']
            delta_proton_err = delta_IdPar_v2['p'][EP]['split']['y_err']
            delta_pminusk = delta_IdPar_v2['p'][EP]['split']['y'] - delta_IdPar_v2['k'][EP]['split']['y']
            delta_pminusk_err = (delta_IdPar_v2['p'][EP]['split']['y_err']**2 + delta_IdPar_v2['k'][EP]['split']['y_err']**2)**0.5

            ax_dl.errorbar(x-1, delta_lambda, delta_lambda_err, fmt='o', ls='none', label=r'$\Lambda-\bar{\Lambda}$')
            ax_dl.errorbar(x, delta_proton, delta_proton_err, fmt='s', ls='none', label=r'$p-\bar{p}$')
            ax_dl.errorbar(x+1, delta_pminusk, delta_pminusk_err, fmt='^', ls='none', label=r'$(p-\bar{p})-(K^+-K^-)$')

            ax_dl.set_xlabel('Centrality (%)', fontsize=16)
            ax_dl.set_ylabel(r'$\Delta v_2$', fontsize=16)
            ax_dl.set_ylim(-0.02, 0.04)
            ax_dl.tick_params(axis='x', labelsize=16)
            ax_dl.tick_params(axis='y', labelsize=16)
            ax_dl.legend(fontsize=16)
            plt.tight_layout()
            fig_dl.savefig(f'{outputDir}/coal_lambda_{EP}.pdf')
    else:
        print('No lambda data found, generating blank figure to fulfill Snakemake requirement')
        fig_blank, ax_blank = plt.subplots(1, 1)
        ax_blank.text(0.5, 0.5, 'No lambda data found', ha='center', va='center', fontsize=20)
        fig_blank.savefig(f'{outputDir}/coal_lambda_TPC.pdf')
        fig_blank.savefig(f'{outputDir}/coal_lambda_EPD.pdf')
    

def none_or_float(arg):
    if arg.lower() == 'none':
        return None
    return float(arg)


def merge_helper(dict_arr, bin_groups):
    dict_v2 = dict_arr
    # first check that all bin groups are disjoint
    # if not, raise an error
    for bins in bin_groups:
        for b in bins:
            for bins2 in bin_groups:
                if bins != bins2 and b in bins2:
                    raise ValueError(f'Bin {b} is in multiple groups: {bins} and {bins2}')
                 
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inputFile', type=str, help='Input file with efficiency corrected v2 values')
    parser.add_argument('inputFile_lambda', type=str, help='Input file with v2 values for lambda')
    parser.add_argument('inputFile_lambdabar', type=str, help='Input file with v2 values for anti-lambda')
    parser.add_argument('resFile', type=str, help='Input file with resolution profiles')
    parser.add_argument('outputDir', type=str, help='Output directory for the figures')
    parser.add_argument('energy', type=str, help='Energy')
    parser.add_argument('yrange', type=none_or_float, nargs='+', help='Y range for the ratio plot')
    args = parser.parse_args()
    if len(args.yrange) > 2:
        raise ValueError('Too many arguments for yrange')
    main(args.inputFile, args.inputFile_lambda, args.inputFile_lambdabar, args.resFile, args.outputDir, args.energy, args.yrange)
