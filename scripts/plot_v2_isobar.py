import numpy as np
import platform
from uncertainties import unumpy
from uncertainties import ufloat
import uproot
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import iv
from scipy.optimize import minimize


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

def main(inputFiles, inputRes, outputDir):
    # determine which of the two input files is for Ru, which is for Zr
    # if 'Zr' in inputFiles[0]:
    #     inputFiles = inputFiles[::-1]
    inputFile_Ru, inputFile_Zr = inputFiles
    res_Ru, res_Zr = inputRes
    df_Ru = pd.read_csv(inputFile_Ru, index_col=False)
    df_Zr = pd.read_csv(inputFile_Zr, index_col=False)
    df_res_Ru = pd.read_csv(res_Ru, index_col=False)
    df_res_Zr = pd.read_csv(res_Zr, index_col=False)

    fig_1, ax_1 = plt.subplots(1, 1, figsize=(8, 6))
    fig_2, ax_2 = plt.subplots(1, 1, figsize=(8, 6))
    fig_2_alt, ax_2_alt = plt.subplots(1, 1, figsize=(8, 6))
    fig_eq2_merged, ax_eq2_merged = plt.subplots(1, 1, figsize=(8, 6))
    fig_v2, ax_v2 = plt.subplots(2, 3, figsize=(24, 12)) # for v2 comparison
    fig_ratio, ax_ratio = plt.subplots(1, 2, figsize=(16, 6))
    fig_double_ratio, ax_double_ratio = plt.subplots(1, 1, figsize=(8, 6))
    scale_down = 1 # scale down the pion v2 to test for possible effect of resonance decay, does not affect antiproton
    for i, EP_method in enumerate(['TPC', 'EPD']): # , 'EPD']): # only TPC for now
        resolution_Ru = unumpy.uarray(df_res_Ru[f'{EP_method}_res'].values, df_res_Ru[f'{EP_method}_res_err'].values)
        piplus_v2_Ru = unumpy.uarray(df_Ru[f'piplus_v2_{EP_method}'].values, df_Ru[f'piplus_v2_err_{EP_method}'].values) * scale_down / resolution_Ru
        piminus_v2_Ru = unumpy.uarray(df_Ru[f'piminus_v2_{EP_method}'].values, df_Ru[f'piminus_v2_err_{EP_method}'].values) * scale_down / resolution_Ru
        proton_v2_Ru = unumpy.uarray(df_Ru[f'proton_v2_{EP_method}'].values, df_Ru[f'proton_v2_err_{EP_method}'].values) / resolution_Ru
        antiproton_v2_Ru = unumpy.uarray(df_Ru[f'antiproton_v2_{EP_method}'].values, df_Ru[f'antiproton_v2_err_{EP_method}'].values) / resolution_Ru
        resolution_Zr = unumpy.uarray(df_res_Zr[f'{EP_method}_res'].values, df_res_Zr[f'{EP_method}_res_err'].values)
        piplus_v2_Zr = unumpy.uarray(df_Zr[f'piplus_v2_{EP_method}'].values, df_Zr[f'piplus_v2_err_{EP_method}'].values) * scale_down / resolution_Zr
        piminus_v2_Zr = unumpy.uarray(df_Zr[f'piminus_v2_{EP_method}'].values, df_Zr[f'piminus_v2_err_{EP_method}'].values) * scale_down / resolution_Zr
        proton_v2_Zr = unumpy.uarray(df_Zr[f'proton_v2_{EP_method}'].values, df_Zr[f'proton_v2_err_{EP_method}'].values) / resolution_Zr
        antiproton_v2_Zr = unumpy.uarray(df_Zr[f'antiproton_v2_{EP_method}'].values, df_Zr[f'antiproton_v2_err_{EP_method}'].values) / resolution_Zr

        # proton counts for net proton
        proton_counts_Ru = df_Ru[f'proton_counts'].values
        antiproton_counts_Ru = df_Ru[f'antiproton_counts'].values
        proton_counts_Zr = df_Zr[f'proton_counts'].values
        antiproton_counts_Zr = df_Zr[f'antiproton_counts'].values

        # net proton v2
        net_proton_v2_Ru = (proton_v2_Ru * proton_counts_Ru - antiproton_v2_Ru * antiproton_counts_Ru) / (proton_counts_Ru - antiproton_counts_Ru)
        net_proton_v2_Zr = (proton_v2_Zr * proton_counts_Zr - antiproton_v2_Zr * antiproton_counts_Zr) / (proton_counts_Zr - antiproton_counts_Zr)
        v2_udbar_Ru = 1./3. * antiproton_v2_Ru
        v2_udbar_Zr = 1./3. * antiproton_v2_Zr
        r_Ru = antiproton_counts_Ru / proton_counts_Ru
        r_Zr = antiproton_counts_Zr / proton_counts_Zr
        print(f'r_Ru = {r_Ru}')
        print(f'r_Zr = {r_Zr}')
        mu_B = 21 # estimated from Tommy's plot
        T_ch = 160 # estimating from Tommy's plot
        N_transud_Ru = 3 * (1 - np.exp(-2. * mu_B / 3. / T_ch)) / (1 - r_Ru)
        N_transud_Zr = 3 * (1 - np.exp(-2. * mu_B / 3. / T_ch)) / (1 - r_Zr)
        print(f'N_transud_Ru = {N_transud_Ru}')
        print(f'N_transud_Zr = {N_transud_Zr}')
        v2_transud_Ru = (net_proton_v2_Ru - (3 - N_transud_Ru) * v2_udbar_Ru) / N_transud_Ru
        v2_transud_Zr = (net_proton_v2_Zr - (3 - N_transud_Zr) * v2_udbar_Zr) / N_transud_Zr

        # use antiproton v2 to scale between Ru and Zr
        # scale = antiproton_v2_Ru / antiproton_v2_Zr
        # piplus_v2_Zr = piplus_v2_Zr * scale
        # piminus_v2_Zr = piminus_v2_Zr * scale
        # antiproton_v2_Zr = antiproton_v2_Zr * scale

        cen = np.array([75, 65, 55, 45, 35, 25, 15, 7.5, 2.5])
        cen1 = np.array([55, 45, 35, 25, 15, 5])
        cen2 = np.array([35, 10])
        cen3 = np.array([60, 25, 5]) # for 0-10%, 10-40%, 40-80%

        # scaling isobar v2 ratios
        # pion_v2_Ru = (piplus_v2_Ru + piminus_v2_Ru) / 2
        # pion_v2_Zr = (piplus_v2_Zr + piminus_v2_Zr) / 2
        # pion_ratio = pion_v2_Ru / pion_v2_Zr
        # piplus_v2_Ru = piplus_v2_Ru / pion_ratio
        # piminus_v2_Ru = piminus_v2_Ru / pion_ratio
        # antiproton_v2_Ru = antiproton_v2_Ru / pion_ratio

        # plot the ratios again
        ratio_Ru = 1 + (piminus_v2_Ru - piplus_v2_Ru) / (piplus_v2_Ru - 2./3. * antiproton_v2_Ru)
        ratio_Zr = 1 + (piminus_v2_Zr - piplus_v2_Zr) / (piplus_v2_Zr - 2./3. * antiproton_v2_Zr)
        ax_ratio[0].errorbar(cen, unumpy.nominal_values(ratio_Ru), unumpy.std_devs(ratio_Ru), fmt='o', label=f'Ru {EP_method}')
        ax_ratio[1].errorbar(cen, unumpy.nominal_values(ratio_Zr), unumpy.std_devs(ratio_Zr), fmt='o', label=f'Zr {EP_method}')
        ax_ratio[0].set_xlabel('Centrality (%)')
        ax_ratio[1].set_xlabel('Centrality (%)')
        ax_ratio[0].set_ylabel('Ratio')
        ax_ratio[1].set_ylabel('Ratio')
        ax_ratio[0].legend()
        ax_ratio[1].legend()
        ax_ratio[0].hlines(1, 0, 80, linestyles='dashed', colors='gray', label='1')
        ax_ratio[1].hlines(1, 0, 80, linestyles='dashed', colors='gray', label='1')
        ax_ratio[0].set_ylim(0.8, 1.2)
        ax_ratio[1].set_ylim(0.8, 1.2)
        ax_ratio[0].hlines(148/140, 0, 80, linestyles='dashed', colors='gray', label='148/140')
        ax_ratio[1].hlines(148/140, 0, 80, linestyles='dashed', colors='gray', label='148/140')
        ax_ratio[0].hlines(152/136, 0, 80, linestyles='dashed', colors='gray', label='152/136')
        ax_ratio[1].hlines(152/136, 0, 80, linestyles='dashed', colors='gray', label='152/136')

        # for Eq1
        numerator = piminus_v2_Ru - piminus_v2_Zr
        denominator = piplus_v2_Ru - piplus_v2_Zr
        ratio = numerator / denominator
        ax_1.errorbar(cen, unumpy.nominal_values(ratio), unumpy.std_devs(ratio), fmt='o', label=EP_method)
        ax_1.set_xlabel('Centrality (%)')
        ax_1.set_ylabel(r'$\Delta v_2^{\pi^-}/\Delta v_2^{\pi^+}$')
        ax_1.legend()
        ax_1.hlines(-1, 0, 80, linestyles='dashed', colors='gray', label='-1')
        ax_1.set_ylim(-2, 2)

        # for Eq2
        numerator = piminus_v2_Ru - piplus_v2_Ru
        denominator = piminus_v2_Zr - piplus_v2_Zr
        adjustment = (v2_transud_Zr - v2_udbar_Zr) / (v2_transud_Ru - v2_udbar_Ru)
        print(f'Adjustment: {adjustment}')
        ratio = numerator / denominator * adjustment
        ax_2.errorbar(cen, unumpy.nominal_values(ratio), unumpy.std_devs(ratio), fmt='o', label=EP_method)
        ax_2.set_xlabel('Centrality (%)')
        ax_2.set_ylabel(r'$(v_2^{\pi^-} - v_2^{\pi^+})_{Ru+Ru}/(v_2^{\pi^-} - v_2^{\pi^+})_{Zr+Zr}$')
        ax_2.legend()
        ax_2.hlines(1/2, 0, 80, linestyles='dashed', colors='gray', label='1/2')
        ax_2.set_ylim(-1, 2)

        # do the above but for 0-10%, 10-40% and 40-80%
        demar = [0, 4, 7, 9]
        adjustment_merged = []
        ratio_eq2 = []
        for i in range(3):
            numerator = np.mean(piminus_v2_Ru[demar[i]:demar[i+1]]) - np.mean(piplus_v2_Ru[demar[i]:demar[i+1]])
            denominator = np.mean(piminus_v2_Zr[demar[i]:demar[i+1]]) - np.mean(piplus_v2_Zr[demar[i]:demar[i+1]])
            proton_v2_Ru_1040 = np.sum(proton_v2_Ru[demar[i]:demar[i+1]] * proton_counts_Ru[demar[i]:demar[i+1]]) / np.sum(proton_counts_Ru[demar[i]:demar[i+1]])
            antiproton_v2_Ru_1040 = np.sum(antiproton_v2_Ru[demar[i]:demar[i+1]] * antiproton_counts_Ru[demar[i]:demar[i+1]]) / np.sum(antiproton_counts_Ru[demar[i]:demar[i+1]])
            proton_counts_Ru_1040 = np.sum(proton_counts_Ru[demar[i]:demar[i+1]])
            antiproton_counts_Ru_1040 = np.sum(antiproton_counts_Ru[demar[i]:demar[i+1]])
            proton_v2_Zr_1040 = np.sum(proton_v2_Zr[demar[i]:demar[i+1]] * proton_counts_Zr[demar[i]:demar[i+1]]) / np.sum(proton_counts_Zr[demar[i]:demar[i+1]])
            antiproton_v2_Zr_1040 = np.sum(antiproton_v2_Zr[demar[i]:demar[i+1]] * antiproton_counts_Zr[demar[i]:demar[i+1]]) / np.sum(antiproton_counts_Zr[demar[i]:demar[i+1]])
            proton_counts_Zr_1040 = np.sum(proton_counts_Zr[demar[i]:demar[i+1]])
            antiproton_counts_Zr_1040 = np.sum(antiproton_counts_Zr[demar[i]:demar[i+1]])
            net_proton_v2_Ru_1040 = (proton_v2_Ru_1040 * proton_counts_Ru_1040 - antiproton_v2_Ru_1040 * antiproton_counts_Ru_1040) / (proton_counts_Ru_1040 - antiproton_counts_Ru_1040)
            net_proton_v2_Zr_1040 = (proton_v2_Zr_1040 * proton_counts_Zr_1040 - antiproton_v2_Zr_1040 * antiproton_counts_Zr_1040) / (proton_counts_Zr_1040 - antiproton_counts_Zr_1040)
            v2_udbar_Ru_1040 = 1./3. * antiproton_v2_Ru_1040
            v2_udbar_Zr_1040 = 1./3. * antiproton_v2_Zr_1040
            r_Ru_1040 = antiproton_counts_Ru_1040 / proton_counts_Ru_1040
            r_Zr_1040 = antiproton_counts_Zr_1040 / proton_counts_Zr_1040
            N_transud_Ru_1040 = 3 * (1 - np.exp(-2. * mu_B / 3. / T_ch)) / (1 - r_Ru_1040)
            N_transud_Zr_1040 = 3 * (1 - np.exp(-2. * mu_B / 3. / T_ch)) / (1 - r_Zr_1040)
            v2_transud_Ru_1040 = (net_proton_v2_Ru_1040 - (3 - N_transud_Ru_1040) * v2_udbar_Ru_1040) / N_transud_Ru_1040
            v2_transud_Zr_1040 = (net_proton_v2_Zr_1040 - (3 - N_transud_Zr_1040) * v2_udbar_Zr_1040) / N_transud_Zr_1040
            adj = (v2_transud_Zr_1040 - v2_udbar_Zr_1040) / (v2_transud_Ru_1040 - v2_udbar_Ru_1040)
            adjustment_merged.append(adj)
            ratio_1040 = numerator / denominator * adj
            ratio_eq2.append(ratio_1040)
        
        shift = 1 if EP_method == 'TPC' else -1
        ax_eq2_merged.errorbar(cen3+shift, unumpy.nominal_values(ratio_eq2), unumpy.std_devs(ratio_eq2), fmt='o', label=f'{EP_method} Merged Ratio')
        # ax_eq2_merged.errorbar(cen3, unumpy.nominal_values(adjustment_merged), unumpy.std_devs(adjustment_merged), fmt='o', label='Adjustment')
        ax_eq2_merged.set_xlabel('Centrality (%)')
        ax_eq2_merged.set_ylabel('Arb. Unit')
        ax_eq2_merged.legend()
        ax_eq2_merged.hlines(1, 0, 80, linestyles='dashed', colors='gray', label='1')
        ax_eq2_merged.set_ylim(-1, 1.5)

        # for Eq2 alternative
        double_ratio_pion = (piminus_v2_Zr / piplus_v2_Zr) / (piminus_v2_Ru / piplus_v2_Ru)
        delta_v2_Zr = piminus_v2_Zr - piplus_v2_Zr
        average_v2 = (piplus_v2_Ru + piminus_v2_Ru + piplus_v2_Zr + piminus_v2_Zr) / 4
        result = 1 - average_v2 * (double_ratio_pion - 1) / delta_v2_Zr
        result = result * adjustment
        print(f'Average alternative Eq2: {unumpy.nominal_values(np.mean(result))} +/- {unumpy.std_devs(np.mean(result))}')
        ax_2_alt.errorbar(cen, unumpy.nominal_values(result), unumpy.std_devs(result), fmt='o', label=EP_method)
        ax_2_alt.set_xlabel('Centrality (%)')
        ax_2_alt.set_ylabel('Alternative Eq2')
        ax_2_alt.legend()
        ax_2_alt.hlines(1/2, 0, 80, linestyles='dashed', colors='gray', label='1/2')
        ax_2_alt.set_ylim(-1, 2)

        # double ratio
        double_ratio = ratio_Zr / ratio_Ru
        ax_double_ratio.errorbar(cen, unumpy.nominal_values(double_ratio), unumpy.std_devs(double_ratio), fmt='o', label=EP_method)
        ax_double_ratio.set_xlabel('Centrality (%)')
        ax_double_ratio.set_ylabel('Double Ratio')
        ax_double_ratio.legend()
        ax_double_ratio.hlines(1, 0, 80, linestyles='dashed', colors='gray', label='1')
        ax_double_ratio.hlines((152/136)/(148/140), 0, 80, linestyles='dashed', colors='gray', label='(152/136)/(148/140)')
        ax_double_ratio.set_ylim(0.8, 1.2)        

        # for v2 comparison
        ax_v2[0][0].errorbar(cen, unumpy.nominal_values(piplus_v2_Ru), unumpy.std_devs(piplus_v2_Ru), fmt='o', label='Ru+Ru')
        ax_v2[0][0].errorbar(cen, unumpy.nominal_values(piplus_v2_Zr), unumpy.std_devs(piplus_v2_Zr), fmt='o', label='Zr+Zr')
        ax_v2[0][0].set_xlabel('Centrality (%)', fontsize=16)
        ax_v2[0][0].set_ylabel(r'$v_2^{\pi^+}$', fontsize=16)
        ax_v2[0][0].legend(fontsize=16)
        ax_v2[0][1].errorbar(cen, unumpy.nominal_values(piminus_v2_Ru), unumpy.std_devs(piminus_v2_Ru), fmt='o', label='Ru+Ru')
        ax_v2[0][1].errorbar(cen, unumpy.nominal_values(piminus_v2_Zr), unumpy.std_devs(piminus_v2_Zr), fmt='o', label='Zr+Zr')
        ax_v2[0][1].set_xlabel('Centrality (%)', fontsize=16)
        ax_v2[0][1].set_ylabel(r'$v_2^{\pi^-}$', fontsize=16)
        ax_v2[0][1].legend(fontsize=16)
        ax_v2[0][2].errorbar(cen, unumpy.nominal_values(piplus_v2_Ru - piplus_v2_Zr), unumpy.std_devs(piplus_v2_Ru - piplus_v2_Zr), fmt='o', label=r'$\Delta v_2^{\pi^+}$')
        ax_v2[0][2].errorbar(cen, unumpy.nominal_values(piminus_v2_Ru - piminus_v2_Zr), unumpy.std_devs(piminus_v2_Ru - piminus_v2_Zr), fmt='o', label=r'$\Delta v_2^{\pi^-}$')
        ax_v2[0][2].set_xlabel('Centrality (%)', fontsize=16)
        ax_v2[0][2].set_ylabel(r'$\Delta v_2$', fontsize=16)
        ax_v2[0][2].hlines(0, 0, 80, linestyles='dashed', colors='gray', label='0')
        ax_v2[0][2].annotate('Ru+Ru - Zr+Zr', (0.5, 0.5), xycoords='axes fraction', ha='center', va='center', fontsize=16)
        ax_v2[0][2].legend(fontsize=16)

        ax_v2[1][0].errorbar(cen, unumpy.nominal_values(piplus_v2_Ru), unumpy.std_devs(piplus_v2_Ru), fmt='o', label=r'$\pi^+$')
        ax_v2[1][0].errorbar(cen, unumpy.nominal_values(piminus_v2_Ru), unumpy.std_devs(piminus_v2_Ru), fmt='o', label=r'$\pi^-$')
        ax_v2[1][0].set_xlabel('Centrality (%)', fontsize=16)
        ax_v2[1][0].set_ylabel(r'$v_2^{Ru}$', fontsize=16)
        ax_v2[1][0].legend(fontsize=16)
        ax_v2[1][1].errorbar(cen, unumpy.nominal_values(piplus_v2_Zr), unumpy.std_devs(piplus_v2_Zr), fmt='o', label=r'$\pi^+$')
        ax_v2[1][1].errorbar(cen, unumpy.nominal_values(piminus_v2_Zr), unumpy.std_devs(piminus_v2_Zr), fmt='o', label=r'$\pi^-$')
        ax_v2[1][1].set_xlabel('Centrality (%)', fontsize=16)
        ax_v2[1][1].set_ylabel(r'$v_2^{Zr}$', fontsize=16)
        ax_v2[1][1].legend(fontsize=16)
        ax_v2[1][2].errorbar(cen, unumpy.nominal_values(piplus_v2_Ru - piminus_v2_Ru), unumpy.std_devs(piplus_v2_Ru - piminus_v2_Ru), fmt='o', label='Ru+Ru')
        ax_v2[1][2].errorbar(cen, unumpy.nominal_values(piplus_v2_Zr - piminus_v2_Zr), unumpy.std_devs(piplus_v2_Zr - piminus_v2_Zr), fmt='o', label='Zr+Zr')
        ax_v2[1][2].set_xlabel('Centrality (%)', fontsize=16)
        ax_v2[1][2].set_ylabel(r'$v_2^{\pi^+} - v_2^{\pi^-}$', fontsize=16)
        ax_v2[1][2].hlines(0, 0, 80, linestyles='dashed', colors='gray', label='0')
        ax_v2[1][2].legend(fontsize=16)

    fig_1.savefig(f'{outputDir}/eq1.pdf')
    fig_2.savefig(f'{outputDir}/eq2.pdf')
    fig_eq2_merged.savefig(f'{outputDir}/eq2_merged.pdf')
    fig_2_alt.savefig(f'{outputDir}/eq2_alt.pdf')
    fig_v2.savefig(f'{outputDir}/v2_comparison.pdf')
    fig_ratio.savefig(f'{outputDir}/ratio.pdf')
    fig_double_ratio.savefig(f'{outputDir}/double_ratio_{EP_method}.pdf')


def none_or_float(arg):
    if arg.lower() == 'none':
        return None
    return float(arg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputFile', type=str, nargs='+', help='Input file', required=True)
    parser.add_argument('--inputRes', type=str, nargs='+', help='Input file for resolution', required=True)
    parser.add_argument('--outputDir', type=str, help='Output directory for the figures', required=True)
    args = parser.parse_args()
    main(args.inputFile, args.inputRes, args.outputDir)
