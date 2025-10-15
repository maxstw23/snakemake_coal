import argparse
import numpy as np
import yaml
import matplotlib.pyplot as plt


def main(default, regular_sys, special_sys, output, energy):
    with open (default, 'r') as f:
        default_cut = yaml.load(f, Loader=yaml.CLoader)
    sys_cut = {}
    if not regular_sys[0].startswith('result/blank/'):
        for sys in regular_sys:
            with open (sys, 'r') as f:
                sys_tag = sys.split('/')[1].split('_')[-1]
                sys_cut[sys_tag] = yaml.load(f, Loader=yaml.CLoader)
    for sys in special_sys:
        with open (sys, 'r') as f:
            sys_tag = sys.split('/')[1].split('_')[-1]
            sys_cut[sys_tag] = yaml.load(f, Loader=yaml.CLoader)
    
    # iterate thru the centralities
    new_yerr = np.zeros_like(default_cut['yerr'])
    for cent in range(len(default_cut['x'])):
        print(f'Centrality: {default_cut["x"][cent]}')
        print('\tSys_tag    Delta      Delta_err  Significance')
        sum_of_unc = 0
        for sys_tag in sys_cut.keys():
            delta = np.abs(default_cut['y'][cent] - sys_cut[sys_tag]['y'][cent])
            delta_err = np.sqrt(np.abs(sys_cut[sys_tag]['yerr'][cent]**2 - default_cut['yerr'][cent]**2))
            # significance = abs(delta) / delta_err if delta_err != 0 else 0
            significance = (delta_err < delta)
            print(f'\t{int(sys_tag):<10} {delta:<10.4f} {delta_err:<10.4f} {significance}')
            if significance:
                sum_of_unc += delta**2 - delta_err**2
        print(f'\tTotal systematic uncertainty: {np.sqrt(sum_of_unc / 12):.4f}')
        new_yerr[cent] = np.sqrt(default_cut['yerr'][cent]**2 + sum_of_unc / 12)

    with open(output, 'w') as f:
        # basically use default_cut, but replace yerr with new_yerr
        default_cut['yerr'] = new_yerr.tolist()
        yaml.dump(default_cut, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--default', type=str, help='Default cut', required=True)
    parser.add_argument('--regular_sys', type=str, help='Regular systematic cut (subsets of default)', nargs='+', required=True)
    parser.add_argument('--special_sys', type=str, help='Special systematic cut (use the same dataset as default)', nargs='+', required=True)
    parser.add_argument('--output', type=str, help='Output file', required=True)
    parser.add_argument('--energy', type=str, help='Energy', required=True)
    args = parser.parse_args()
    main(args.default, args.regular_sys, args.special_sys, args.output, args.energy)