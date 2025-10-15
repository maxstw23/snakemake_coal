# Description: This script is used to calculate the number of d, u quarks present in mid-rapidity from published BES-I data
import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
from uncertainties import unumpy
from tabulate import tabulate
import argparse

def main(output):
    yields = {
        r'$\pi^+$':
        {
            'values': np.array([93.4, 123.9, 161.4, 172.9, 182.3]),
            'errors_stat': np.array([8.4, 12.4, 17.8, 19.1, 20.1]),
            'errors_sys': np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        },
        r'$\pi^-$':
        {
            'values': np.array([100.0, 129.8, 165.8, 177.1, 185.8]),
            'errors_stat': np.array([9.0, 13.0, 18.3, 19.5, 20.5]),
            'errors_sys': np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        },
        r'$K^+$':
        {
            'values': np.array([20.8, 25.0, 29.6, 31.1, 32.0]),
            'errors_stat': np.array([1.7, 2.5, 2.9, 2.8, 2.9]),
            'errors_sys': np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        },
        r'$K^-$':
        {
            'values': np.array([7.7, 12.3, 18.8, 22.6, 25.0]),
            'errors_stat': np.array([0.6, 1.2, 1.9, 2.0, 2.3]),
            'errors_sys': np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        },
        r'$p$':
        {
            'values': np.array([54.9, 44.0, 34.2, 31.7, 26.5]),
            'errors_stat': np.array([6.1, 5.3, 4.5, 3.8, 2.9]),
            'errors_sys': np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        },
        r'$\bar{p}$':
        {
            'values': np.array([0.39, 1.5, 4.2, 6.0, 8.5]),
            'errors_stat': np.array([0.05, 0.2, 0.5, 0.7, 1.0]),
            'errors_sys': np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        },
        r'$K^0_S$':
        {
            'values': np.array([12.67, 15.93, 20.89, 23.24, 24.92]),
            'errors_stat': np.array([0.12, 0.12, 0.08, 0.09, 0.1]),
            'errors_sys': np.array([0.44, 0.58, 0.67, 0.7, 1.7])
        },
        r'$\Lambda^0$':
        {
            'values': np.array([15.3, 14.17, 12.58, 11.67, 11.02]),
            'errors_stat': np.array([0.11, 0.08, 0.04, 0.04, 0.03]),
            'errors_sys': np.array([0.82, 0.78, 0.46, 0.51, 0.86]),
        },
        r'$\bar{\Lambda}^0$':
        {
            'values': np.array([0.193, 0.659, 1.858, 2.75, 3.82]),
            'errors_stat': np.array([0.009, 0.009, 0.01, 0.01, 0.01]),
            'errors_sys': np.array([0.019, 0.047, 0.099, 0.14, 0.36]),
        },
        r'\phi':
        {
            'values': np.array([1.21, 1.72, 2.57, 3.01, 3.38]),
            'errors_stat': np.array([0.06, 0.04, 0.04, 0.04, 0.03]),
            'errors_sys': np.array([0.21, 0.26, 0.35, 0.45, 0.48]),
        },
        r'$\Xi$':
        {
            'values': np.array([1.19, 1.35, 1.62, 1.57, 1.54]),
            'errors_stat': np.array([0.03, 0.02, 0.01, 0.01, 0.01]),
            'errors_sys': np.array([0.11, 0.11, 0.1, 0.11, 0.19]),
        },
        r'$\bar{\Xi}$':
        {
            'values': np.array([0.0657, 0.169, 0.421, 0.598, 0.78]),
            'errors_stat': np.array([0.0044, 0.004, 0.005, 0.006, 0.01]),
            'errors_sys': np.array([0.0068, 0.014, 0.03, 0.038, 0.11]),
        }
    }

    # number of quarks in the order of u, d, s, ubar, dbar, sbar
    num_quarks = {
        r'$\pi^+$': np.array([1, 0, 0, 0, 1, 0]),
        r'$\pi^-$': np.array([0, 1, 0, 1, 0, 0]),
        r'$\pi^0$': np.array([0.5, 0.5, 0, 0.5, 0.5, 0]),
        r'$K^+$': np.array([1, 0, 0, 0, 0, 1]),
        r'$K^-$': np.array([0, 0, 1, 1, 0, 0]),
        r'$K^0$': np.array([0, 1, 0, 0, 0, 1]),
        r'$\bar{K}^0$': np.array([0, 0, 1, 0, 1, 0]),
        r'$p$': np.array([2, 1, 0, 0, 0, 0]),
        r'$\bar{p}$': np.array([0, 0, 0, 2, 1, 0]),
        r'$n$': np.array([1, 2, 0, 0, 0, 0]),
        r'$\bar{n}$': np.array([0, 0, 0, 1, 2, 0]),
        # r'K^0_S': np.array([0, 0.5, 0.5, 0, 0.5, 0.5]),
        r'$\Lambda^0$': np.array([1, 1, 1, 0, 0, 0]),
        r'$\bar{\Lambda}^0$': np.array([0, 0, 0, 1, 1, 1]),
        r'\phi': np.array([0, 0, 1, 0, 0, 1]),
        r'$\Xi$': np.array([0, 1, 2, 0, 0, 0]),
        r'$\bar{\Xi}$': np.array([0, 0, 0, 0, 1, 2])
    }

    # make 5 tables like this
    # particle | yield | u | d | s | ubar | dbar | sbar 
    for i, energy in enumerate([7.7, 11.5, 19.6, 27, 39]):
        print(f'Energy: {energy} GeV')
        table = [['Particle', 'Yield', 'u', 'd', 's', 'ubar', 'dbar', 'sbar']]
        sum_quarks = np.zeros(6)
        for particle, yld in yields.items():
            if particle == r'$K^0_S$':
                continue
            yield_val = yld['values'][i]
            num_quark = yield_val * num_quarks[particle]
            table.append([particle, yield_val, *num_quark])
            sum_quarks += num_quark
        # Treat unmeasured particles separately
        for particle in [r'$K^0$', r'$\bar{K}^0$', r'$n$', r'$\bar{n}$']:
            if particle == r'$K^0$' or particle == r'$\bar{K}^0$':
                yield_val = yields[r'$K^0_S$']['values'][i]
            if particle == r'$n$':
                yield_val = yields[r'$p$']['values'][i] * (197-79) / 79 # n/p ratio
            if particle == r'$\bar{n}$':
                yield_val = yields[r'$\bar{p}$']['values'][i] / (197-79) * 79
            num_quark = yield_val * num_quarks[particle]
            table.append([particle, np.round(yield_val, 2), *np.round(num_quark,2)])
            sum_quarks += num_quark
        table.append(['Total', '', *np.round(sum_quarks,2)])
        # write to output file
        tabulated_table = tabulate(table, tablefmt='plain')
        print(tabulated_table)
        if output: 
            with open(output, 'w') as f:
                f.write(tabulate(table, tablefmt='latex'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate the number of d, u quarks present in mid-rapidity from published BES-I data')
    parser.add_argument('--output', help='Output file name', type=str, default=None)
    args = parser.parse_args()
    main(args.output)
    
    



