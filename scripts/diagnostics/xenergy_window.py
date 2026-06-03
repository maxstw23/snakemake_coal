import numpy as np, pandas as pd
N=500; W=0.01; pt=(np.arange(N)+0.5)*W; PTNQ_HI=0.6
NQ={'piplus':2,'piminus':2,'antiproton':3}
ENS=['7p7GeV','9p2GeV','11p5GeV','14p6GeV','17p3GeV','19p6GeV','27GeV']
def intv2(dfc,res,sp,lo,cens,EP):
    nq=NQ[sp]; m=(pt>=lo*nq)&(pt<=PTNQ_HI*nq); sw=swv=0.0
    for c in cens:
        r=res[f'{EP}_res'].values[c-1]
        if r<=0 or np.isnan(r): continue
        v=dfc[c-1][f'{sp}_v2_{EP}'].values[:N]/r; w=dfc[c-1][f'{sp}_counts'].values[:N]
        sw+=w[m].sum(); swv+=(w[m]*v[m]).sum()
    return swv/sw if sw>0 else np.nan
def R(dfc,res,lo,EP):
    pip=intv2(dfc,res,'piplus',lo,[5,6,7],EP); pim=intv2(dfc,res,'piminus',lo,[5,6,7],EP)
    ap=intv2(dfc,res,'antiproton',lo,[5,6,7],EP); D=pip-2/3*ap
    return (pim-2/3*ap)/D if D else np.nan
LOS=[0.08,0.16,0.24,0.32]
print('10-40% TPC, R per energy at different ptnq_lo:')
print('energy   ' + '  '.join(f'lo={l:.2f}' for l in LOS))
allR={l:[] for l in LOS}
for e in ENS:
    base=f'result/sys_tag_0/energy_{e}'
    res=pd.read_csv(f'{base}/v2_noeff_corrected_res.csv')
    dfc=[pd.read_csv(f'{base}/v2_noeff_corrected_cen{c}.csv') for c in range(1,10)]
    row=[R(dfc,res,l,'TPC') for l in LOS]
    for l,v in zip(LOS,row): allR[l].append(v)
    print(f'{e:8} ' + '  '.join(f'{v:6.3f}' for v in row))
print('-'*48)
print('mean     ' + '  '.join(f'{np.mean(allR[l]):6.3f}' for l in LOS))
print('std      ' + '  '.join(f'{np.std(allR[l]):6.3f}' for l in LOS))
print('vs 315/276=%.3f'%(315/276))
