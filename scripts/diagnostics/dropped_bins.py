import pandas as pd, numpy as np, glob, re
# count dropped (NaN) v2 bins per energy/particle/EP from existing fit outputs
files = sorted(glob.glob('result/sys_tag_0/energy_*/fit_Lambda*_v2_*.csv'))
print(f"{'energy':14} {'particle':11} {'EP':4} {'bins':>5} {'dropped':>8} {'%':>6}")
for f in files:
    m = re.search(r'energy_([^/]+)/fit_(Lambda(?:bar)?)_v2_(TPC|EPD)\.csv', f)
    if not m: continue
    energy, part, ep = m.groups()
    try:
        df = pd.read_csv(f, header=[0,1], index_col=0)
    except Exception as e:
        print(f'{energy:14} {part:11} {ep:4}  parse error: {e}'); continue
    vals = df.xs('values', axis=1, level=1)
    n = vals.size
    ndrop = int(np.isnan(pd.to_numeric(vals.stack(), errors='coerce')).sum() + (n - vals.stack().size))
    print(f'{energy:14} {part:11} {ep:4} {n:5d} {ndrop:8d} {100*ndrop/n:6.1f}')
