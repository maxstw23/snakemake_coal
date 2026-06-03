import sys, numpy as np, yaml; sys.path.insert(0,'scripts')
from generate_paper_plots import build_sys_covariance, SYS_REGIME_SPLITS
ENS=['7p7GeV','9p2GeV','11p5GeV','14p6GeV','17p3GeV','19p6GeV','27GeV']
ef=np.array([7.7,9.2,11.5,14.6,17.3,19.6,27.0])
stat=np.zeros(7); src={}
for i,e in enumerate(ENS):
    d=yaml.load(open(f'tmp/final/energy_{e}/coal_TPC.yaml'),Loader=yaml.CLoader)
    stat[i]=d['yerr_stat_1040']
    for t,c in d['sys_sources_1040'].items():
        src.setdefault(int(t),np.zeros(7))[i]=float(c)
np.set_printoptions(linewidth=130, suppress=True)
print('energies:', ef)
print('\nPer-source signed contribution vectors s_s (over energies):')
print('  Vz   (tag1):', np.round(src[1],5))
print('  Nfit (tag2):', np.round(src[2],5))
print('\nSYS_REGIME_SPLITS:', SYS_REGIME_SPLITS, ' (Vz split: A={19.6,27}, B=rest)')

Vsys=build_sys_covariance(ef,src)
print('\nV_sys (x1e6):'); print(np.round(Vsys*1e6,2))
# worked entries
print('\nWorked diagonal at 9.2:  s_Vz^2 + s_Nfit^2 =',
      round(src[1][1]**2 + src[2][1]**2,8), ' = yerr_sys^2 ; sqrt=', round((src[1][1]**2+src[2][1]**2)**0.5,5))
print('Worked Cov(9.2,14.6): s_VzB(9.2)*s_VzB(14.6) + s_Nfit(9.2)*s_Nfit(14.6) =',
      round(src[1][1]*src[1][3] + src[2][1]*src[2][3], 8))
print('Worked Cov(19.6,14.6): VzA(19.6)*..=0 (regime) + Nfit(19.6)*Nfit(14.6) =',
      round(src[2][5]*src[2][3], 8), ' (Vz contributes 0 across the A-B block)')
V=np.diag(stat**2)+Vsys
D=np.diag(1/np.sqrt(np.diag(V))); corr=D@V@D
print('\ncorrelation matrix (stat+sys):'); print(np.round(corr,2))
