import numpy as np, pandas as pd
N=500; W=0.01; pt=(np.arange(N)+0.5)*W
PTNQ_HI=0.6
LOS=np.round(np.arange(0.08,0.401,0.04),2)
NQ={'piplus':2,'piminus':2,'antiproton':3}
def intv2(dfc, res, sp, lo, cens, EP):
    nq=NQ[sp]; m=(pt>=lo*nq)&(pt<=PTNQ_HI*nq)
    sw=swv=0.0
    for c in cens:
        r=res[f'{EP}_res'].values[c-1]
        if r<=0 or np.isnan(r): continue
        v=dfc[c-1][f'{sp}_v2_{EP}'].values[:N]/r
        w=dfc[c-1][f'{sp}_counts'].values[:N]
        sw+=w[m].sum(); swv+=(w[m]*v[m]).sum()
    return swv/sw if sw>0 else np.nan
for energy in ['19p6GeV','27GeV','7p7GeV']:
    base=f'result/sys_tag_0/energy_{energy}'
    res=pd.read_csv(f'{base}/v2_noeff_corrected_res.csv')
    dfc=[pd.read_csv(f'{base}/v2_noeff_corrected_cen{c}.csv') for c in range(1,10)]
    print(f'\n=== {energy}  (10-40%, cen 5-7) ===')
    print('ptnq_lo: ' + '  '.join(f'{x:.2f}' for x in LOS))
    for EP in ['TPC','EPD']:
        Rs=[]
        for lo in LOS:
            pip=intv2(dfc,res,'piplus',lo,[5,6,7],EP)
            pim=intv2(dfc,res,'piminus',lo,[5,6,7],EP)
            ap =intv2(dfc,res,'antiproton',lo,[5,6,7],EP)
            D=pip-2/3*ap; Rs.append((pim-2/3*ap)/D if D!=0 else np.nan)
        print(f'{EP:4} R:   ' + '  '.join(f'{x:.2f}' for x in Rs))
