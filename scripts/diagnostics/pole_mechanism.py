import numpy as np, pandas as pd
from pathlib import Path
rng = np.random.default_rng(1)
N = 1_000_000
BASE = Path('result/sys_tag_0')

def bin_stats(energy, ep, row):
    df = pd.read_csv(BASE/f'energy_{energy}'/'v2_noeff_corrected.csv')
    pip,pim,ap = (df[f'{s}_v2_{ep}'].values[row] for s in ['piplus','piminus','antiproton'])
    pe,me,ae   = (df[f'{s}_v2_err_{ep}'].values[row] for s in ['piplus','piminus','antiproton'])
    spip=rng.normal(pip,pe,N); spim=rng.normal(pim,me,N); sap=rng.normal(ap,ae,N)
    D = spip-2/3*sap; Nn = spim-2/3*sap; R = Nn/D
    D0 = pip-2/3*ap; sD=np.sqrt(pe**2+4/9*ae**2)
    print(f'\n{energy} {ep} row{row}:  D0/sD = {D0/sD:+.2f}')
    print(f'  toys with D<0 (sign flip):     {100*np.mean(D<0):5.1f}%')
    print(f'  toys with |R|>5 (tail blowup): {100*np.mean(np.abs(R)>5):5.1f}%')
    print(f'  MEAN of R:   {np.mean(R):10.2f}   (std {np.std(R):.1f})  <- unstable/garbage')
    print(f'  MEDIAN of R: {np.median(R):10.2f}   [16,84]=[{np.percentile(R,16):.2f},{np.percentile(R,84):.2f}]  <- robust')
    # re-run mean with a different seed to show non-reproducibility
    R2=(rng.normal(pim,me,N)-2/3*rng.normal(ap,ae,N))/(rng.normal(pip,pe,N)-2/3*rng.normal(ap,ae,N))
    print(f'  MEAN again (new seed): {np.mean(R2):10.2f}   MEDIAN again: {np.median(R2):.2f}')

bin_stats('19p6GeV','TPC',0)   # D/sD ~ -0.12, the R=3.13+-18.8 bin
bin_stats('27GeV','EPD',0)     # D/sD ~  0.11, the R=4.27+-29.7 bin
bin_stats('19p6GeV','EPD',2)   # D/sD ~  3.9, moderate/skewed
bin_stats('19p6GeV','TPC',5)   # D/sD ~ 84, clean control
