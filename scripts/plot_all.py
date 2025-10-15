import numpy as np
import argparse
import glob
from uncertainties import unumpy
import matplotlib.pyplot as plt
import pandas as pd

def main(data_paths, output_paths):
    ratio_1040 = {}
    ratio_4080 = {}
    hasIsobarZr = False
    for data_path in data_paths:
        df = pd.read_csv(data_path)
        energy = glob.glob(data_path)[0].split('/')[-2]
        if energy == 'isobar_Zr':
            energy = '200GeV'
            hasIsobarZr = True
        energy = energy.replace('GeV', '')
        energy = energy.replace('p', '.')
        for EP in ['EPD', 'TPC']:
            piplus = unumpy.uarray(df[f'piplus_v2_{EP}'].values, df[f'piplus_v2_err_{EP}'].values)
            piminus = unumpy.uarray(df[f'piminus_v2_{EP}'].values, df[f'piminus_v2_err_{EP}'].values)
            antiproton = unumpy.uarray(df[f'antiproton_v2_{EP}'].values, df[f'antiproton_v2_err_{EP}'].values)
            if EP not in ratio_1040:
                ratio_1040[EP] = {}
            if EP not in ratio_4080:
                ratio_4080[EP] = {}
            ratio_1040[EP][energy] = np.mean((piminus[4:7] - 2./3. * antiproton[4:7]) / (piplus[4:7] - 2./3. * antiproton[4:7]))
            ratio_4080[EP][energy] = np.mean((piminus[0:4] - 2./3. * antiproton[0:4]) / (piplus[0:4] - 2./3. * antiproton[0:4]))

    print(ratio_1040)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for EP in ['EPD', 'TPC']:
        ratios = list(ratio_1040[EP].values())
        ax.errorbar([float(energy) for energy in ratio_1040[EP].keys()], unumpy.nominal_values(ratios), unumpy.std_devs(ratios), ls='none')
        ax.scatter([float(energy) for energy in ratio_1040[EP].keys()], unumpy.nominal_values(ratios), s=20, label=EP)
    ax.hlines(315/276, *ax.get_xlim(), linestyles='dashed', label=r'$\frac{315}{276}$')
    if hasIsobarZr:
        ax.hlines(152/136, *ax.get_xlim(), linestyles='dashed', label=r'$\frac{152}{136}$')
    ax.set_xlabel('Energy (GeV)', fontsize=16)
    ax.set_ylabel(r'$\frac{v_2(\pi^-) - \frac{2}{3}v_2(\bar{p})}{v_2(\pi^+) - \frac{2}{3}v_2(\bar{p})}$', fontsize=16)
    ax.annotate('Au+Au 10-40%', (0.05, 0.9), xycoords='axes fraction', fontsize=16)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_xscale('log')
    ax.legend(fontsize=20)
    plt.tight_layout()
    fig.savefig(output_paths[0])

    fig_peri, ax_peri = plt.subplots(1, 1, figsize=(8, 6))
    cen_glauber = np.array([2.5,7.5,15,25,35,45,55,65,75])
    ratio_du = np.array([1.13304,1.13351,1.13777,1.14464,1.15486,1.17016,1.19128,1.22017,1.26102])
    ratio_du_err = np.array([1.56853e-08,4.95918e-08,9.90949e-08,1.74393e-07,3.19487e-07,5.92654e-07,1.24839e-06,3.09313e-06,5.11625e-06])
    for EP in ['EPD', 'TPC']:
        ratios = list(ratio_4080[EP].values())
        ax_peri.errorbar([float(energy) for energy in ratio_4080[EP].keys()], unumpy.nominal_values(ratios), unumpy.std_devs(ratios), ls='none')
        ax_peri.scatter([float(energy) for energy in ratio_4080[EP].keys()], unumpy.nominal_values(ratios), s=20, label=EP)
    xlim = ax_peri.get_xlim()
    ax_peri.hlines(315/276, *xlim, linestyles='dashed', label=r'$\frac{315}{276}$')
    if hasIsobarZr:
        ax_peri.hlines(152/136, *xlim, linestyles='dashed', label=r'$\frac{152}{136}$')
    glauber_4080 = np.mean(unumpy.uarray(ratio_du, ratio_du_err)[-4:])
    x_glauber = np.linspace(*xlim, 100)
    y_up_glauber = np.ones_like(x_glauber) * glauber_4080.nominal_value + glauber_4080.std_dev
    y_down_glauber = np.ones_like(x_glauber) * glauber_4080.nominal_value - glauber_4080.std_dev
    ax_peri.set_ylim(0.0, 2.0)
    ax_peri.fill_between(x_glauber, y_down_glauber, y_up_glauber, color='C2', alpha=0.8, label='Glauber')
    ax_peri.set_xlabel('Energy (GeV)', fontsize=16)
    ax_peri.set_ylabel(r'$\frac{v_2(\pi^-) - \frac{2}{3}v_2(\bar{p})}{v_2(\pi^+) - \frac{2}{3}v_2(\bar{p})}$', fontsize=16)
    ax_peri.annotate('Au+Au 40-80%', (0.05, 0.9), xycoords='axes fraction', fontsize=16)
    ax_peri.tick_params(axis='x', labelsize=16)
    ax_peri.tick_params(axis='y', labelsize=16)
    ax_peri.set_xscale('log')
    ax_peri.legend(fontsize=16, loc='lower left')
    plt.tight_layout()
    fig_peri.savefig(output_paths[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot v2 ratio for all energies and EP methods')
    parser.add_argument('--data_paths', nargs='+', help='Paths to data files', required=True)
    parser.add_argument('--output_paths', nargs='+', help='Path to output plot', required=True)
    args = parser.parse_args()
    main(args.data_paths, args.output_paths)

