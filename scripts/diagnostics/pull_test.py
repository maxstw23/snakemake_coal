# Per-pT pull test: do the v2(pT) points scatter around their smooth trend by exactly
# their quoted errors? Uses 2nd differences (removes local slope) so only noise remains.
# ratio = measured point-to-point scatter / expected-from-errors. ~1 => errors honest.
import numpy as np, pandas as pd
N=500; pt=(np.arange(N)+0.5)*0.01
def ratio(energy, sp, EP, cen, lo=0.3, hi=1.4):
    df=pd.read_csv(f'result/sys_tag_0/energy_{energy}/v2_noeff_corrected_cen{cen}.csv')
    v=df[f'{sp}_v2_{EP}'].values[:N]; e=df[f'{sp}_v2_err_{EP}'].values[:N]
    m=(pt>=lo)&(pt<=hi)&(e>0)&np.isfinite(v)&np.isfinite(e)
    v,e=v[m],e[m]
    if len(v)<10: return np.nan
    d2=v[2:]-2*v[1:-1]+v[:-2]                       # 2nd difference: removes constant+slope
    exp=np.sqrt(e[:-2]**2+4*e[1:-1]**2+e[2:]**2)    # expected std of d2 if errors correct
    return np.std(d2)/np.sqrt(np.mean(exp**2))
print('19.6 GeV  ratio = measured scatter / expected-from-errors  (~1.0 = errors honest)')
print(f"{'species':12} {'TPC cen5':>9} {'cen6':>7} {'cen7':>7} | {'EPD cen5':>9} {'cen6':>7} {'cen7':>7}")
for sp in ['piplus','piminus','antiproton']:
    row=[ratio('19p6GeV',sp,'TPC',c) for c in [5,6,7]]+[ratio('19p6GeV',sp,'EPD',c) for c in [5,6,7]]
    print(f"{sp:12} "+ "  ".join(f'{x:6.2f}' for x in row[:3]) + " | " + "  ".join(f'{x:6.2f}' for x in row[3:]))
