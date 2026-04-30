from iminuit import Minuit
from iminuit.cost import LeastSquares
from uncertainties import unumpy
import uproot
import platform
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse

def fcn(x, p0, p1, p2, p3, p4):
    return (p0 + p3 * x + p4 * x * x) * np.exp(-np.power(p1 / x, p2))


def fcn_eta(x, p0, p1, p2):
    return p0 * np.exp(-np.power(np.power(x/p1, 2), p2/2.))


def fcn_2D(x_y, p0, p1, p2, p3, p4, p5, p6):
    x, y = x_y
    return (p0 + p3 * x + p4 * x * x) * np.exp(-np.power(p1 / x, p2)) * np.exp(-np.power(np.power(y/p5, 2), p6/2.))

def file_name_matching(particles, paths):
    ### produce a dictionary with particle name and centrality as keys and file path as values
    file_dict = {}
    for path in paths:
        for particle in particles:
            for cen in range(1, 10):
                if f'cen{cen}.{particle}.root' == path.split('/')[-1]:
                    file_dict[(particle, cen)] = path
    return file_dict
        


def find_eff(particle, eta_cut, energy, paths, pt_lo, pt_hi):
    p_or_pt = 'Pt' # 'P' or 'Pt'
    # particle = 'kplus'
    particles = {'piplus': 'pip', 'piminus': 'pim', 'proton': 'P', 'antiproton': 'AP',
                 'kplus': 'Kp', 'kminus': 'Km'}
    file_dict = file_name_matching(particles.keys(), paths)
    if energy == '27GeV':
        embed = '/20203010/'
        paths_2 = [embed.join(path.rsplit('/', 1)) for path in paths]
        file_dict_2 = file_name_matching(particles.keys(), paths_2)

    pt_edge = uproot.open(file_dict[(particle, 1)])[f'hSel{p_or_pt}'].axis().edges()
    pt = (pt_edge[1:] + pt_edge[:-1]) / 2.
    eta_edge = uproot.open(file_dict[(particle, 1)])['hEta'].axis().edges()
    eta = (eta_edge[1:] + eta_edge[:-1]) / 2.
    cut = (pt > pt_lo) & (pt < pt_hi) # 0.15 for 14.6 and 19.6, 0.12 for 7.7
    cut_2 = (pt > 0.15) & (pt < 0.4) # for 27GeV
    # cut_eta = (eta < 1.5) & (eta > -1.5)
    tof_eta = (eta < eta_cut) & (eta > -eta_cut)

    fig, ax = plt.subplots(3, 3, figsize=(16, 9.6))
    p = np.zeros((5,9))

    status = np.zeros(9)
    for cen in range(1, 10):
        f = uproot.open(file_dict[(particle, cen)])
        SelPtEta = f[f'hSel{p_or_pt}Eta'].values()
        SelPtEtaMc = f[f'hSel{p_or_pt}EtaMc'].values()

        # SelPtEta is 2D. Poject 2D array to 1D with tof eta cut
        SelPtEta = SelPtEta[:,tof_eta]
        SelPtEtaMc = SelPtEtaMc[:,tof_eta]

        # project
        SelPtEta = np.sum(SelPtEta, axis=1)
        SelPtEtaMc = np.sum(SelPtEtaMc, axis=1)
        if energy == '27GeV':
            f2 = uproot.open(file_dict_2[(particle, cen)])
            SelPtEta_2 = f2[f'hSel{p_or_pt}Eta'].values()
            SelPtEtaMc_2 = f2[f'hSel{p_or_pt}EtaMc'].values()
            SelPtEta_2 = SelPtEta_2[:,tof_eta]
            SelPtEtaMc_2 = SelPtEtaMc_2[:,tof_eta]
            SelPtEta_2 = np.sum(SelPtEta_2, axis=1)
            SelPtEtaMc_2 = np.sum(SelPtEtaMc_2, axis=1)
            SelPtEta[cut_2] = SelPtEta_2[cut_2]
            SelPtEtaMc[cut_2] = SelPtEtaMc_2[cut_2]
        
        SelPtEta = SelPtEta[cut]
        SelPtEtaMc = SelPtEtaMc[cut]        

        eff_pteta = np.divide(SelPtEta, SelPtEtaMc, out=np.zeros_like(SelPtEta), where=SelPtEtaMc != 0)
        eff_err_pteta = np.sqrt(eff_pteta * (1 - eff_pteta) / SelPtEtaMc)
        # check if the error is nan and use the average of the neighboring 10 bins, excluding any nan bin
        is_nan = np.isnan(eff_err_pteta) | (eff_err_pteta == 0)

        ax[(cen - 1) // 3, (cen - 1) % 3].errorbar(pt[cut], eff_pteta, yerr=eff_err_pteta, fmt='o', ls='none',
                                                   color='black', markersize=5)
        ax[(cen - 1) // 3, (cen - 1) % 3].set_xlabel(r'$p_T$')
        ax[(cen - 1) // 3, (cen - 1) % 3].set_ylabel('Eff')
        lsq = LeastSquares(pt[cut][~is_nan], eff_pteta[~is_nan], eff_err_pteta[~is_nan], fcn)
        m = Minuit(lsq, p0=1.1, p1=0.2, p2=3.4, p3=-0.2, p4=0.04)
        if energy in ('27GeV', '7p7GeV', '9p2GeV', '11p5GeV'):
            m = Minuit(lsq, p0=0.8, p1=0.2, p2=5., p3=-0.2, p4=0.04)
        if energy in ('14p6GeV', '17p3GeV'):
            m = Minuit(lsq, p0=0.9, p1=0.2, p2=3.4, p3=-0.2, p4=0.04)
        print(m.migrad())
        if not m.valid:
            m.simplex()
            m.migrad()
        migrad_valid = m.valid
        m.hesse()
        if migrad_valid:
            status[cen - 1] = 1

        p[0][cen - 1] = m.values['p0']
        p[1][cen - 1] = m.values['p1']
        p[2][cen - 1] = m.values['p2']
        p[3][cen - 1] = m.values['p3']
        p[4][cen - 1] = m.values['p4']
        x = np.linspace(0.05, 3.5, 3301)
        ax[(cen - 1) // 3, (cen - 1) % 3].plot(x, fcn(x, *m.values))
        if status[cen - 1]:
            ax[(cen - 1) // 3, (cen - 1) % 3].annotate('converged', xy=(0.5, 0.5), xycoords='axes fraction',
                                                        ha='center', va='center')
        ax[(cen - 1) // 3, (cen - 1) % 3].set_ylim(0., 1.)

    fig.savefig(f'plots/QA/TPC_eff_{particle}_{energy}.pdf')
    # plt.show()
    if status.all():
        print('All fits converged')
    else:
        print(f'{particle} {np.arange(1,10)[status != 1]} not converged')
        raise ValueError('Not all fits converged')

    output = ''
    np.set_printoptions(precision=6, floatmode='fixed', linewidth=np.inf)
    # print(p)
    for i in range(5):
        output += f'const float Efficiency::P{i}_{particles[particle]}[] = {{'
        for j in range(8):
            output += f'{p[i][j]:.6f}, '
        output += f'{p[i][8]:.6f}}};\n'
    
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('energy', type=str, help='Energy, i.e. 7p7GeV')
    parser.add_argument('eta_cut', type=float, help='Eta cut')
    parser.add_argument('paths', type=str, nargs='+', help='Paths to the root files')
    parser.add_argument('pt_lo', type=float, help='Lower limit of pT for fitting purpose')
    parser.add_argument('pt_hi', type=float, help='Upper limit of pT for fitting purpose')
    args = parser.parse_args()
    energy = args.energy
    eta_cut = args.eta_cut
    paths = args.paths
    pt_lo = args.pt_lo
    pt_hi = args.pt_hi
    script_name = f'scripts/{energy}/Efficiency.cpp'
    with open(script_name, 'w') as f:
        f.write("""#include "Efficiency.h"
#include <string>
#include <cmath>""")
        f.write('\n')
        for p in ['piplus', 'piminus', 'kplus', 'kminus', 'proton', 'antiproton']:
            f.write(find_eff(p, float(eta_cut), energy, paths, float(pt_lo), float(pt_hi)))
            f.write('\n')
        f.write("""
const float Efficiency::P0_Kp_2D[] = {0.656673, 0.651768, 0.647041, 0.659321, 0.646929, 0.637643, 0.645916, 0.624893, 0.604222};
const float Efficiency::P1_Kp_2D[] = {0.143461, 0.136486, 0.132301, 0.140275, 0.13774, 0.138923, 0.146108, 0.152787, 0.152349};
const float Efficiency::P2_Kp_2D[] = {1.75571, 1.6679, 1.61186, 1.54363, 1.59639, 1.61874, 1.60391, 1.70599, 1.80471};
const float Efficiency::P3_Kp_2D[] = {0.14417, 0.150836, 0.156056, 0.145075, 0.150114, 0.153802, 0.141166, 0.149843, 0.162156};
const float Efficiency::P4_Kp_2D[] = {-0.0283527, -0.0297447, -0.0310676, -0.0289775, -0.0296311, -0.0304348, -0.0278383, -0.0293249, -0.0320393};
const float Efficiency::P5_Kp_2D[] = {1.10086, 1.09584, 1.10236, 1.09607, 1.09355, 1.09028, 1.08435, 1.08404, 1.0822};
const float Efficiency::P6_Kp_2D[] = {4.59476, 4.66841, 4.58455, 4.72552, 4.71154, 4.82421, 4.93087, 4.80386, 4.75507};
const float Efficiency::P0_Km_2D[] = {0.654812, 0.669162, 0.646511, 0.645435, 0.626828, 0.643098, 0.6192, 0.632903, 0.584627};
const float Efficiency::P1_Km_2D[] = {0.150512, 0.144169, 0.146853, 0.153886, 0.151368, 0.152935, 0.152321, 0.148042, 0.155633};
const float Efficiency::P2_Km_2D[] = {1.73642, 1.54927, 1.69996, 1.68169, 1.78667, 1.63821, 1.72796, 1.53378, 1.9663};
const float Efficiency::P3_Km_2D[] = {0.144394, 0.137028, 0.152111, 0.153102, 0.161691, 0.147432, 0.15978, 0.148994, 0.170449};
const float Efficiency::P4_Km_2D[] = {-0.0282313, -0.027081, -0.030181, -0.0303341, -0.0319985, -0.0291948, -0.0312043, -0.0299695, -0.0329565};
const float Efficiency::P5_Km_2D[] = {1.11151, 1.10048, 1.09567, 1.10665, 1.08959, 1.08616, 1.08686, 1.07772, 1.08155};
const float Efficiency::P6_Km_2D[] = {4.40872, 4.62938, 4.71322, 4.40107, 4.94297, 4.95828, 4.91221, 4.87859, 4.82713};
                
// functions
float Efficiency::GetEfficiency1D(float pT, int cent, std::string particle)
{
    if (particle == "pip")
        return (P0_pip[cent-1] + P3_pip[cent-1]*pT + P4_pip[cent-1]*pT*pT) * exp(-pow(P1_pip[cent-1]/pT,P2_pip[cent-1]));
    else if (particle == "pim")
        return (P0_pim[cent-1] + P3_pim[cent-1]*pT + P4_pim[cent-1]*pT*pT) * exp(-pow(P1_pim[cent-1]/pT,P2_pim[cent-1]));
    else if (particle == "Kp")
        return (P0_Kp[cent-1] + P3_Kp[cent-1]*pT + P4_Kp[cent-1]*pT*pT) * exp(-pow(P1_Kp[cent-1]/pT,P2_Kp[cent-1]));
    else if (particle == "Km")
        return (P0_Km[cent-1] + P3_Km[cent-1]*pT + P4_Km[cent-1]*pT*pT) * exp(-pow(P1_Km[cent-1]/pT,P2_Km[cent-1]));
    else if (particle == "P")
        return (P0_P[cent-1] + P3_P[cent-1]*pT + P4_P[cent-1]*pT*pT) * exp(-pow(P1_P[cent-1]/pT,P2_P[cent-1]));
    else if (particle == "AP")
        return (P0_AP[cent-1] + P3_AP[cent-1]*pT + P4_AP[cent-1]*pT*pT) * exp(-pow(P1_AP[cent-1]/pT,P2_AP[cent-1]));
    else
        return 0;
}

float Efficiency::GetEfficiency2D(float pT, float eta, int cent, std::string particle) // only for kplus and kminus
{
    if (particle == "Kp")
        return (P0_Kp_2D[cent-1] + P3_Kp_2D[cent-1]*pT + P4_Kp_2D[cent-1]*pT*pT) * exp(-pow(P1_Kp_2D[cent-1]/pT,P2_Kp_2D[cent-1])) * exp(-pow(pow(eta/P5_Kp_2D[cent-1],2),P6_Kp_2D[cent-1]));
    else if (particle == "Km")
        return (P0_Km_2D[cent-1] + P3_Km_2D[cent-1]*pT + P4_Km_2D[cent-1]*pT*pT) * exp(-pow(P1_Km_2D[cent-1]/pT,P2_Km_2D[cent-1])) * exp(-pow(pow(eta/P5_Km_2D[cent-1],2),P6_Km_2D[cent-1]));
    else
        return 0;
}
""")
