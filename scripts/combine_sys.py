import argparse
import numpy as np
import yaml
import matplotlib.pyplot as plt


def main(default, regular_sys, special_sys, output, energy):
    with open (default, 'r') as f:
        default_cut = yaml.load(f, Loader=yaml.CLoader)
    sys_cut = {}
    if regular_sys is not None:
        if not regular_sys[0].startswith('result/blank/'):
            for sys in regular_sys:
                with open (sys, 'r') as f:
                    sys_tag = sys.split('/')[1].split('_')[-1]
                    sys_cut[sys_tag] = yaml.load(f, Loader=yaml.CLoader)

    if special_sys is not None:
        for sys in special_sys:
            with open (sys, 'r') as f:
                sys_tag = sys.split('/')[1].split('_')[-1]
                sys_cut[sys_tag] = yaml.load(f, Loader=yaml.CLoader)

    # iterate thru the centralities
    new_yerr = np.zeros_like(default_cut['yerr'])
    yerr_stat = np.zeros_like(default_cut['yerr'])
    yerr_sys = np.zeros_like(default_cut['yerr'])
    for cent in range(len(default_cut['x'])):
        print(f'Centrality: {default_cut["x"][cent]}')
        print('\t           Default')
        print(f'\t           {default_cut["y"][cent]:<10.4f}+-{default_cut["yerr"][cent]:<10.4f}')
        print('\tSys_tag    Delta      Delta_err  Significance')
        sum_of_unc = 0
        for sys_tag in sys_cut.keys():
            delta = np.abs(default_cut['y'][cent] - sys_cut[sys_tag]['y'][cent])
            delta_signed = sys_cut[sys_tag]['y'][cent] - default_cut['y'][cent]
            delta_err = np.sqrt(np.abs(sys_cut[sys_tag]['yerr'][cent]**2 - default_cut['yerr'][cent]**2))
            # significance = abs(delta) / delta_err if delta_err != 0 else 0
            significance = (delta_err < delta)
            print(f'\t{int(sys_tag):<10} {delta_signed:<10.4f} {delta_err:<10.4f} {significance}')
            if significance:
                sum_of_unc += delta**2 - delta_err**2
        print(f'\tTotal systematic uncertainty: {np.sqrt(sum_of_unc / 3):.4f}')
        new_yerr[cent] = np.sqrt(default_cut['yerr'][cent]**2 + sum_of_unc / 3)
        yerr_stat[cent] = default_cut['yerr'][cent]
        yerr_sys[cent] = np.sqrt(sum_of_unc / 3)

    # iterate thru the combined centrality bins
    new_yerr_combined = np.zeros(3)
    yerr_stat_combined = np.zeros(3)
    yerr_sys_combined = np.zeros(3)
    for i, cb in enumerate(['010', '1040', '4080']):
        print(f'Centrality bin: {cb}')
        print('\t           Default')
        print(f'\t           {default_cut["y_" + cb]:<10.4f}+-{default_cut["yerr_" + cb]:<10.4f}')
        print('\tSys_tag    Delta      Delta_err  Significance')
        sum_of_unc = 0
        for sys_tag in sys_cut.keys():
            delta = np.abs(default_cut['y_' + cb] - sys_cut[sys_tag]['y_' + cb])
            delta_signed = sys_cut[sys_tag]['y_' + cb] - default_cut['y_' + cb]
            delta_err = np.sqrt(np.abs(sys_cut[sys_tag]['yerr_' + cb]**2 - default_cut['yerr_' + cb]**2))
            # significance = abs(delta) / delta_err if delta_err != 0 else 0
            significance = (delta_err < delta)
            print(f'\t{int(sys_tag):<10} {delta_signed:<10.4f} {delta_err:<10.4f} {significance}')
            if significance:
                sum_of_unc += delta**2 - delta_err**2
        print(f'\tTotal systematic uncertainty: {np.sqrt(sum_of_unc / 3):.4f}')
        new_yerr_combined[i] = np.sqrt(default_cut['yerr_' + cb]**2 + sum_of_unc / 3)
        yerr_stat_combined[i] = default_cut['yerr_' + cb]
        yerr_sys_combined[i] = np.sqrt(sum_of_unc / 3)

    with open(output, 'w') as f:
        # basically use default_cut, but replace yerr with new_yerr
        default_cut['yerr'] = new_yerr.tolist()
        default_cut['yerr_stat'] = yerr_stat.tolist()
        default_cut['yerr_sys'] = yerr_sys.tolist()
        for i, cb in enumerate(['010', '1040', '4080']):
            default_cut['yerr_' + cb] = new_yerr_combined[i]
            default_cut['yerr_stat_' + cb] = yerr_stat_combined[i]
            default_cut['yerr_sys_' + cb] = yerr_sys_combined[i]
        yaml.dump(default_cut, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--default', type=str, help='Default cut', required=True)
    # it is OK if either or both of regular sys and special sys are empty, just don't do any systematic uncertainty calculation in that case
    parser.add_argument('--regular_sys', type=str, help='Regular systematic cut (subsets of default)', nargs='*')
    parser.add_argument('--special_sys', type=str, help='Special systematic cut (use the same dataset as default)', nargs='*')
    parser.add_argument('--output', type=str, help='Output file', required=True)
    parser.add_argument('--energy', type=str, help='Energy', required=True)
    args = parser.parse_args()
    main(args.default, args.regular_sys, args.special_sys, args.output, args.energy)