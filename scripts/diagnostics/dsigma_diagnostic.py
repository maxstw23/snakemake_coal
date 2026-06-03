#!/usr/bin/env python3
"""C1 triage: D/sigma_D for the ratio denominator, plus linear-vs-toyMC ratio error.

D = v2_pi+ - 2/3 v2_pbar   (resolution-corrected)
sigma_D : variances-only (per-species stat errors + small res-err term)
R = (v2_pi- - 2/3 v2_pbar)/(v2_pi+ - 2/3 v2_pbar)

Read-only on existing CSVs. No covariance (PID-exclusivity decision, 2026-05-31).
"""
import numpy as np
import pandas as pd
from pathlib import Path

ENERGIES = ['7p7GeV','9p2GeV','11p5GeV','14p6GeV','17p3GeV','19p6GeV','27GeV']
CEN_LABEL = ['70-80','60-70','50-60','40-50','30-40','20-30','10-20','5-10','0-5']  # row0..8
BASE = Path('result/sys_tag_0')
N_TOY = 200_000
rng = np.random.default_rng(0)

def load(energy):
    d = BASE / f'energy_{energy}'
    df = pd.read_csv(d / 'v2_noeff_corrected.csv')
    res = pd.read_csv(d / 'v2_noeff_corrected_res.csv')
    return df, res

def analyze(df, res, ep):
    r   = res[f'{ep}_res'].values
    rerr= res[f'{ep}_res_err'].values
    pip = df[f'piplus_v2_{ep}'].values;      pip_e = df[f'piplus_v2_err_{ep}'].values
    pim = df[f'piminus_v2_{ep}'].values;     pim_e = df[f'piminus_v2_err_{ep}'].values
    ap  = df[f'antiproton_v2_{ep}'].values;  ap_e  = df[f'antiproton_v2_err_{ep}'].values

    rows = []
    for i in range(9):
        # resolution validity = the actual analysis mask
        bad_res = (not np.isfinite(r[i])) or (r[i] - 2*rerr[i] <= 0)
        # raw-space denominator and its variance-only error
        D0 = pip[i] - 2/3*ap[i]
        sD0 = np.sqrt(pip_e[i]**2 + (4/9)*ap_e[i]**2)
        # add the res-error term (D scales as 1/res); res itself cancels in D/sD
        rel_res = (rerr[i]/r[i]) if (np.isfinite(r[i]) and r[i] != 0) else np.nan
        sD = np.sqrt(sD0**2 + (D0*rel_res)**2)
        dsig = D0/sD if sD > 0 else np.nan

        # linear ratio + error (variances only, shared pbar handled analytically)
        N0 = pim[i] - 2/3*ap[i]
        R_lin = N0/D0 if D0 != 0 else np.nan
        # dR from pim, pip, ap (ap shared): R=N/D
        dR_dpim = 1/D0; dR_dpip = -N0/D0**2
        dR_dap  = (-2/3)/D0 - (-2/3)*N0/D0**2   # d/d ap of (pim-2/3ap)/(pip-2/3ap)
        sR_lin = np.sqrt((dR_dpim*pim_e[i])**2 + (dR_dpip*pip_e[i])**2 + (dR_dap*ap_e[i])**2)

        # toy-MC: 3 independent species draws (raw space), shared pbar into N and D
        s_pip = rng.normal(pip[i], pip_e[i], N_TOY)
        s_pim = rng.normal(pim[i], pim_e[i], N_TOY)
        s_ap  = rng.normal(ap[i],  ap_e[i],  N_TOY)
        Rt = (s_pim - 2/3*s_ap)/(s_pip - 2/3*s_ap)
        med, lo, hi = np.percentile(Rt, [50, 16, 84])

        rows.append((CEN_LABEL[i], D0, sD, dsig, R_lin, sR_lin, med, lo, hi, bad_res))
    return rows

def main():
    for energy in ENERGIES:
        try:
            df, res = load(energy)
        except FileNotFoundError:
            continue
        print(f'\n===== {energy} =====')
        for ep in ['TPC','EPD']:
            print(f'\n  [{ep}]  cen      D        sD     D/sD | R_lin  +-sigR | toy: med  [16,84]   (asym)   flag')
            for (cl,D0,sD,dsig,Rl,sRl,med,lo,hi,bad) in analyze(df,res,ep):
                flag = 'BADRES' if bad else ('<-- D/sD<3' if abs(dsig)<3 else '')
                asym = f'(+{hi-med:.2f}/-{med-lo:.2f})'
                print(f'        {cl:>6} {D0:8.4f} {sD:7.4f} {dsig:6.2f} | '
                      f'{Rl:6.2f} {sRl:6.2f} | {med:6.2f} [{lo:5.2f},{hi:5.2f}] {asym:>14} {flag}')

if __name__ == '__main__':
    main()
