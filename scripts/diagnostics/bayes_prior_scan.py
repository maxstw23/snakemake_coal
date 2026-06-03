import subprocess, sys, numpy as np, arviz as az
from pathlib import Path
PY=sys.executable
def run(sig):
    out=f'tmp/bayes_sig{sig}'
    subprocess.run([PY,'scripts/quark_v2_bayes.py','--energy','19p6GeV',
        '--f_prior_sigma',str(sig),'--out_dir',out,
        '--draws','600','--tune','600','--chains','2'],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return az.from_netcdf(f'{out}/trace.nc')
def rich(p,a,b,c,d,nu): return a[:,None]/(1+np.exp(-(p[None,:]-b[:,None])/c[:,None]))**(1/nu[:,None])-d[:,None]
def fl(t,n): return t.posterior[n].values.flatten()
p=np.linspace(0.08,0.6,150)
print(f"{'sigma':>6} {'f med[16,84]':>22} {'int(tr-prod) med[16,84]':>26} {'product med[16,84]':>24}")
for sig in [0.5, 2.0]:
    t=run(sig)
    f=fl(t,'f')
    tr=rich(p,fl(t,'a_tr'),fl(t,'b_tr'),fl(t,'c_tr'),fl(t,'d_tr'),fl(t,'nu_tr'))
    pr=rich(p,fl(t,'a_a'),fl(t,'b_a'),fl(t,'c_a'),fl(t,'d_a'),fl(t,'nu_a'))
    exc=(tr-pr).mean(1)            # integrated excess per sample
    prod=f*exc
    q=lambda x:f'{np.median(x):+.4f}[{np.percentile(x,16):+.4f},{np.percentile(x,84):+.4f}]'
    print(f"{sig:>6} {q(f):>22} {q(exc):>26} {q(prod):>24}")
