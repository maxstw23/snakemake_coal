# M13: inverse-variance vs counts-weighted merge of cen 5,6,7 for quark_v2_bayes input.
import numpy as np, pandas as pd
N=500; pt=(np.arange(N)+0.5)*0.01
SP=['piplus','piminus','kplus','kminus','proton','antiproton']
NQ={'piplus':2,'piminus':2,'kplus':2,'kminus':2,'proton':3,'antiproton':3}
EP='EPD'; cens=[5,6,7]; base='result/sys_tag_0/energy_19p6GeV'
res=pd.read_csv(f'{base}/v2_noeff_corrected_res.csv')
dfc={c:pd.read_csv(f'{base}/v2_noeff_corrected_cen{c}.csv') for c in cens}
def merge(sp, mode):
    sw=np.zeros(N); swv=np.zeros(N)
    for c in cens:
        r=res[f'{EP}_res'].values[c-1]
        v=dfc[c][f'{sp}_v2_{EP}'].values/r; e=dfc[c][f'{sp}_v2_err_{EP}'].values/r
        cnt=dfc[c][f'{sp}_counts'].values
        valid=(e>0)&np.isfinite(e)&np.isfinite(v)
        w=np.where(valid, (1/e**2 if mode=='iv' else cnt), 0.0)
        sw+=w; swv+=w*v
    good=sw>0
    v2=np.where(good, swv/sw, np.nan)
    err=np.where(good, 1/np.sqrt(np.where(good,sw,1)), np.nan) if mode=='iv' else np.nan*v2
    return v2, err
print(f"{'species':12} {'window':>13} {'max|Δ|/err':>11} {'mean|Δ|/err':>12} {'int_iv':>9} {'int_cw':>9} {'rel%':>6}")
for sp in SP:
    nq=NQ[sp]; lo,hi=(0.16,2.0) if nq==2 else (0.24,3.0)
    m=(pt>=lo)&(pt<=hi)
    v_iv,e_iv=merge(sp,'iv'); v_cw,_=merge(sp,'cw')
    d=np.abs(v_iv-v_cw)
    rel=d[m]/np.where(e_iv[m]>0,e_iv[m],np.nan)
    rel=rel[np.isfinite(rel)]
    # counts-weighted integral over window (physical), both merges
    cnt_tot=sum(dfc[c][f'{sp}_counts'].values for c in cens)
    iiv=np.nansum((v_iv*cnt_tot)[m])/np.nansum(cnt_tot[m])
    icw=np.nansum((v_cw*cnt_tot)[m])/np.nansum(cnt_tot[m])
    print(f"{sp:12} [{lo:.2f},{hi:.2f}] {np.nanmax(rel):11.3f} {np.nanmean(rel):12.3f} {iiv:9.5f} {icw:9.5f} {100*(iiv-icw)/icw:6.2f}")
