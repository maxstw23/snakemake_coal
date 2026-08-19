import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from uncertainties import unumpy

CEN_X = np.array([75., 65., 55., 45., 35., 25., 15., 7.5, 2.5])
PARTICLES = ['piplus', 'piminus', 'antiproton']
PARTICLE_LABELS = [r'$\pi^+$', r'$\pi^-$', r'$\bar{p}$']
COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
MARKERS = ['o', 's', '^', 'D']


def load_v2(v2_path, res_path):
    df = pd.read_csv(v2_path)
    df_res = pd.read_csv(res_path)
    resolution = unumpy.uarray(df_res['EPD_res'].values, df_res['EPD_res_err'].values)
    result = {}
    for p in PARTICLES:
        raw = unumpy.uarray(df[f'{p}_v2_EPD'].values, df[f'{p}_v2_err_EPD'].values)
        corrected = raw / resolution
        result[p] = (unumpy.nominal_values(corrected), unumpy.std_devs(corrected))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', nargs='+', required=True)
    parser.add_argument('--v2',  nargs='+', required=True)
    parser.add_argument('--res', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    assert len(args.labels) == len(args.v2) == len(args.res)

    datasets = []
    for v2_path, res_path, label in zip(args.v2, args.res, args.labels):
        datasets.append((label, load_v2(v2_path, res_path)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    fig.suptitle('14.6 GeV EPD plane comparison', fontsize=13)

    for ax, p, p_label in zip(axes, PARTICLES, PARTICLE_LABELS):
        for i, (label, data) in enumerate(datasets):
            vals, errs = data[p]
            # skip if all zero (histogram absent)
            if np.all(vals == 0):
                continue
            ax.errorbar(CEN_X, vals, errs,
                        fmt=MARKERS[i], color=COLORS[i], label=label,
                        capsize=3, markersize=5)
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.set_xlabel('Centrality (%)', fontsize=11)
        ax.set_ylabel(r'$v_2$', fontsize=11)
        ax.set_title(p_label, fontsize=12)
        ax.legend(fontsize=9)
        ax.invert_xaxis()

    fig.tight_layout()
    fig.savefig(args.out)
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
