#!/usr/bin/env python3
"""
Bayesian NCQ quark v2 extraction — mixture model with S-curve pion correction.

Simultaneously fits π±, K±, p, p̄, Λ, Λ̄ v2(pT) to extract latent quark
v2 functions under a physically motivated mixture decomposition:

  v2^u(p) = f * v2^tr(p) + (1 - f) * v2^prod(p)
  v2^d(p) = f * v2^tr(p) + (1 - f) * v2^prod(p)         [f_d = f_u = f]
  v2^ā(p) = v2^prod(p)                                   [identity]
  v2^s(p) = independent Richards sigmoid

v2^prod is a fully independent Richards sigmoid, pinned by antiparticle data.
v2^tr   is a fully independent Richards sigmoid for transported quarks.
f is the transported fraction shared by u and d (isospin asymmetry ignored
for now):
  f = sigmoid(f_logit)
f_logit ~ N(logit(tanh(μ_B/3T_ch)), f_sigma)  from GCE (arXiv:1701.07065).

Pion v2 includes an S-curve hadronic correction (difference of two exponentials)
that absorbs ρ→ππ feed-down dilution and rescattering (see docs/experiments_2026-05-01.md):
  v2^π(pT) = v2_NCQ(pT) + corr(pT) + ε_0^π
  corr(pT) = A_fast·exp(-pT/τ_fast) - B_slow·exp(-pT/τ_slow)

Parameters:
  v2^tr:       5  (a_tr, b_tr, c_tr, d_tr, nu_tr)
  f_logit:     1  (logit-normal, transported fraction shared by u and d)
  v2^prod:     5  (a_a,  b_a,  c_a,  d_a,  nu_a)
  v2^s:        5  (a_s,  b_s,  c_s,  d_s,  nu_s)
  S-curve:     4  (A_fast, τ_fast, B_slow, τ_slow) — HalfNormal
  eps0_pi:     1  (flat pion offset, N(0,0.02))
  Total:      21

NCQ coalescence:
  v2^π+(pT)    = v2^u(pT/2) + v2^ā(pT/2) + corr(pT) + ε_0^π
  v2^π-(pT)    = v2^ā(pT/2) + v2^d(pT/2) + corr(pT) + ε_0^π
  v2^K+(pT)    = v2^u(pT/2) + v2^s(pT/2)
  v2^K-(pT)    = v2^ā(pT/2) + v2^s(pT/2)
  v2^p(pT)     = 2·v2^u(pT/3) + v2^d(pT/3)
  v2^p̄(pT)     = 3·v2^ā(pT/3)
  v2^Λ(pT)     = v2^u(pT/3) + v2^d(pT/3) + v2^s(pT/3)
  v2^Λ̄(pT)     = 2·v2^ā(pT/3) + v2^s(pT/3)

Usage (standalone):
  conda activate lambda_v1
  python scripts/quark_v2_bayes.py --energy 19p6GeV

Usage (Snakemake):
  snakemake plots/19p6GeV/quark_v2_functions.pdf
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from pathlib import Path

# ── Defaults ───────────────────────────────────────────────────────────────────

DEFAULTS = dict(
    energy          = '19p6GeV',
    cen_bins        = [5, 6, 7],
    ep              = 'EPD',
    meson_pt_lo     = 0.16,
    meson_pt_hi     = 2.00,
    baryon_pt_lo    = 0.24,
    baryon_pt_hi    = 3.00,
    rebin           = 5,
    sigma_ncq       = 0.01,
    sigma_ncq_slp   = 0.05,
    eps_p_ref       = 0.3,
    err_floor       = 0.001,
    f_prior_sigma   = 0.5,    # std in logit space
    thermal_csv     = 'data/chemical_freezeout_GCE_1701.07065.csv',
    thermal_cen_labels = ['10-20', '20-30', '30-40'],
    draws           = 2000,
    tune            = 1000,
    chains          = 4,
    target_accept   = 0.95,
    max_treedepth   = 12,
    random_seed     = 42,
)

PT_BIN_WIDTH = 0.01
N_BINS       = 500

MESON_SPECIES   = ['piplus', 'piminus', 'kplus', 'kminus']
BARYON_SPECIES  = ['proton', 'antiproton']
LAMBDA_SPECIES  = ['lambda', 'lambdabar']
V2_CSV_SPECIES  = MESON_SPECIES + BARYON_SPECIES
PARTICLES       = V2_CSV_SPECIES + LAMBDA_SPECIES

LAMBDA_PT_LO    = 0.45
LAMBDA_PT_HI    = 1.75
LAMBDA_N_BINS   = 14

LABELS = {
    'piplus':     r'$\pi^+$',
    'piminus':    r'$\pi^-$',
    'kplus':      r'$K^+$',
    'kminus':     r'$K^-$',
    'proton':     r'$p$',
    'antiproton': r'$\bar{p}$',
    'lambda':     r'$\Lambda$',
    'lambdabar':  r'$\bar{\Lambda}$',
}
COLORS = {
    'piplus': 'steelblue', 'piminus': 'darkorange',
    'kplus':  'forestgreen', 'kminus': 'firebrick',
    'proton': 'purple', 'antiproton': 'brown',
    'lambda': 'darkred', 'lambdabar': 'darkcyan',
}

ENERGY_TO_SQRTS = {
    '7p7GeV': 7.7, '9p2GeV': 9.2, '11p5GeV': 11.5, '14p6GeV': 14.6,
    '17p3GeV': 17.3, '19p6GeV': 19.6, '27GeV': 27.0, '39GeV': 39.0,
    '62p4GeV': 62.4, '200GeV': 200.0,
}

# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Bayesian NCQ quark v2 — mixture model')
    p.add_argument('--energy',            default=DEFAULTS['energy'])
    p.add_argument('--cen_bins',          nargs='+', type=int,
                   default=DEFAULTS['cen_bins'])
    p.add_argument('--ep',                default=DEFAULTS['ep'])
    p.add_argument('--data_dir',          default=None)
    p.add_argument('--csv_prefix',        default='v2_noeff_corrected',
                   help='Filename stem for v2 CSVs (e.g. v2_noeff_corrected or v2_eff_corrected)')
    p.add_argument('--out_dir',           default=None)
    p.add_argument('--meson_pt_lo',       type=float, default=DEFAULTS['meson_pt_lo'])
    p.add_argument('--meson_pt_hi',       type=float, default=DEFAULTS['meson_pt_hi'])
    p.add_argument('--baryon_pt_lo',      type=float, default=DEFAULTS['baryon_pt_lo'])
    p.add_argument('--baryon_pt_hi',      type=float, default=DEFAULTS['baryon_pt_hi'])
    p.add_argument('--rebin',             type=int,   default=DEFAULTS['rebin'])
    p.add_argument('--sigma_ncq',         type=float, default=DEFAULTS['sigma_ncq'])
    p.add_argument('--sigma_ncq_slp',     type=float, default=DEFAULTS['sigma_ncq_slp'])
    p.add_argument('--f_prior_sigma',     type=float, default=DEFAULTS['f_prior_sigma'])
    p.add_argument('--thermal_csv',       default=DEFAULTS['thermal_csv'])
    p.add_argument('--thermal_cen_labels',nargs='+',
                   default=DEFAULTS['thermal_cen_labels'])
    p.add_argument('--draws',             type=int,   default=DEFAULTS['draws'])
    p.add_argument('--tune',              type=int,   default=DEFAULTS['tune'])
    p.add_argument('--chains',            type=int,   default=DEFAULTS['chains'])
    p.add_argument('--target_accept',     type=float, default=DEFAULTS['target_accept'])
    p.add_argument('--max_treedepth',     type=int,   default=DEFAULTS['max_treedepth'])
    p.add_argument('--random_seed',       type=int,   default=DEFAULTS['random_seed'])
    return p.parse_args()

# ── Thermal model prior ────────────────────────────────────────────────────────

def load_thermal_prior(energy, thermal_csv, cen_labels):
    """
    Compute f_prior = tanh(μ_B / (3 T_ch)) from GCE thermal model.

    Uses GCEY (yield-fit) T_ch and μ_B, averaged over cen_labels rows.
    Falls back to f_prior=0.5 (logit=0) if energy not found.

    Returns (f_prior, T_ch_MeV, muB_MeV) — all scalars.
    """
    sqrts = ENERGY_TO_SQRTS.get(energy)
    if sqrts is None:
        print(f'  [thermal prior] Energy {energy} not in lookup; using f_prior=0.5')
        return 0.5, np.nan, np.nan

    try:
        df = pd.read_csv(thermal_csv)
    except FileNotFoundError:
        print(f'  [thermal prior] {thermal_csv} not found; using f_prior=0.5')
        return 0.5, np.nan, np.nan

    rows = df[(df['sqrts_GeV'] == sqrts) & (df['centrality'].isin(cen_labels))]
    if len(rows) == 0:
        print(f'  [thermal prior] No rows for {energy} / {cen_labels}; using f_prior=0.5')
        return 0.5, np.nan, np.nan

    if 'interpolated' in rows.columns and rows['interpolated'].any():
        print(f'  [thermal prior] WARNING: T_ch and μ_B for {energy} are INTERPOLATED '
              f'(cubic spline from neighbouring energies), not fitted to data.')

    T_ch = rows['Tch_GCEY_MeV'].mean()
    muB  = rows['muB_GCEY_MeV'].mean()
    f    = np.tanh(muB / (3.0 * T_ch))
    return float(f), float(T_ch), float(muB)

# ── Data loading ───────────────────────────────────────────────────────────────

def load_merged_v2(data_dir, cen_bins, ep, csv_prefix='v2_noeff_corrected'):
    data_dir   = Path(data_dir)
    pt_centers = (np.arange(N_BINS) + 0.5) * PT_BIN_WIDTH
    res_df     = pd.read_csv(data_dir / f'{csv_prefix}_res.csv')

    # Counts (yield) weighting across centralities, matching the main ratio pipeline
    # (M13): merging distinct centralities estimates the yield-weighted average v2, so the
    # weight is the particle count, not 1/err^2 (the latter would bias toward small-error bins
    # and does not correspond to any physical average). Counts-weighted-mean error:
    # sigma = sqrt(sum (w*err)^2) / sum w.
    sum_w   = {p: np.zeros(N_BINS) for p in V2_CSV_SPECIES}
    sum_wv  = {p: np.zeros(N_BINS) for p in V2_CSV_SPECIES}
    sum_w2e2 = {p: np.zeros(N_BINS) for p in V2_CSV_SPECIES}

    for cen in cen_bins:
        cen_df  = pd.read_csv(data_dir / f'{csv_prefix}_cen{cen}.csv')
        res_val = res_df.iloc[cen - 1][f'{ep}_res']

        for p in V2_CSV_SPECIES:
            v2_raw  = cen_df[f'{p}_v2_{ep}'].values
            err_raw = cen_df[f'{p}_v2_err_{ep}'].values
            cnt     = cen_df[f'{p}_counts'].values
            v2_c    = v2_raw  / res_val
            err_c   = err_raw / res_val
            valid   = (err_c > 0) & np.isfinite(err_c) & np.isfinite(v2_c) & (cnt > 0)
            w = np.where(valid, cnt, 0.0)
            sum_w[p]    += w
            sum_wv[p]   += w * v2_c
            sum_w2e2[p] += (w * err_c) ** 2

    v2_out, err_out = {}, {}
    for p in V2_CSV_SPECIES:
        good       = sum_w[p] > 0
        denom      = np.where(good, sum_w[p], 1.0)
        v2_out[p]  = np.where(good, sum_wv[p] / denom, np.nan)
        err_out[p] = np.where(good, np.sqrt(sum_w2e2[p]) / denom, np.nan)

    return pt_centers, v2_out, err_out


def load_lambda_v2(data_dir, cen_bins, ep):
    """
    Load Lambda/Lambdabar v2 from fit CSVs (output of fit_v2.py).

    The fit CSVs have MultiIndex columns: (centrality, [values, errors, counts]).
    Resolution correction is applied using the same v2_noeff_corrected_res.csv
    as for other particles.

    Returns (pt_lambda, v2_lambda, err_lambda, v2_lambdabar, err_lambdabar).
    Each v2/err is a 1D array over the merged pT bins.
    """
    data_dir   = Path(data_dir)
    res_df     = pd.read_csv(data_dir / 'v2_noeff_corrected_res.csv')
    pt_lambda  = np.linspace(LAMBDA_PT_LO, LAMBDA_PT_HI, LAMBDA_N_BINS)

    def _read_one(csv_path):
        df = pd.read_csv(csv_path, header=[0, 1], index_col=0)
        sum_w  = np.zeros(LAMBDA_N_BINS)
        sum_wv = np.zeros(LAMBDA_N_BINS)
        for cen in cen_bins:
            cen_str = str(cen)
            if cen_str not in df.columns.levels[0]:
                continue
            v2_raw  = df.loc[:, (cen_str, 'values')].values
            err_raw = df.loc[:, (cen_str, 'errors')].values
            res_val = res_df.iloc[cen - 1][f'{ep}_res']
            v2_c    = v2_raw  / res_val
            err_c   = err_raw / res_val
            valid   = (err_c > 0) & np.isfinite(err_c) & np.isfinite(v2_c)
            w       = np.where(valid, 1.0 / err_c**2, 0.0)
            sum_w   += w
            sum_wv  += w * v2_c
        v2  = np.where(sum_w > 0, sum_wv / sum_w, np.nan)
        err = np.where(sum_w > 0, 1.0 / np.sqrt(sum_w), np.nan)
        return v2, err

    v2_lam, err_lam = _read_one(data_dir / f'fit_Lambda_v2_{ep}.csv')
    v2_lbr, err_lbr = _read_one(data_dir / f'fit_Lambdabar_v2_{ep}.csv')
    return pt_lambda, v2_lam, err_lam, v2_lbr, err_lbr


def prepare_datasets(pt_centers, v2_all, err_all, cfg):
    def _cut_rebin(p, pt_lo, pt_hi):
        mask  = (pt_centers >= pt_lo) & (pt_centers <= pt_hi)
        pt    = pt_centers[mask]
        v2    = v2_all[p][mask]
        err   = err_all[p][mask]
        n_out = len(pt) // cfg['rebin']
        pt_r, v2_r, err_r = [], [], []
        for i in range(n_out):
            sl    = slice(i * cfg['rebin'], (i + 1) * cfg['rebin'])
            e_sl  = err[sl]; v_sl = v2[sl]
            valid = (e_sl > 0) & np.isfinite(e_sl) & np.isfinite(v_sl)
            if valid.sum() == 0:
                continue
            w = 1.0 / e_sl[valid]**2
            pt_r.append(pt[sl][valid].mean())
            v2_r.append(np.sum(w * v_sl[valid]) / np.sum(w))
            err_r.append(1.0 / np.sqrt(np.sum(w)))
        return np.array(pt_r), np.array(v2_r), np.array(err_r)

    data = {}
    for p in MESON_SPECIES:
        data[p] = _cut_rebin(p, cfg['meson_pt_lo'], cfg['meson_pt_hi'])
    for p in BARYON_SPECIES:
        data[p] = _cut_rebin(p, cfg['baryon_pt_lo'], cfg['baryon_pt_hi'])
    return data


def print_dataset_summary(data, lambda_data=None):
    print('\nDataset summary:')
    for sp, (pt, v2, err) in data.items():
        nq = 2 if sp in MESON_SPECIES else 3
        print(f'  {sp:12s}: {len(pt):2d} bins  '
              f'pT=[{pt.min():.3f},{pt.max():.3f}]  '
              f'pT/nQ=[{pt.min()/nq:.3f},{pt.max()/nq:.3f}]  '
              f'v2=[{v2.min():.4f},{v2.max():.4f}]')
    if lambda_data:
        for sp, (pt, v2, err) in lambda_data.items():
            nq = 3
            good = np.isfinite(v2) & np.isfinite(err) & (err > 0)
            v2g = v2[good]
            print(f'  {sp:12s}: {len(pt):2d} bins  '
                  f'pT=[{pt.min():.3f},{pt.max():.3f}]  '
                  f'pT/nQ=[{pt.min()/nq:.3f},{pt.max()/nq:.3f}]  '
                  f'v2=[{v2g.min():.4f},{v2g.max():.4f}]')

# ── ρ feed-down precomputation (not currently used in model) ──────────────────

def _build_interp_weights(pt_query, pt_grid):
    """Linear interpolation weight matrix, shape (n_query, n_grid)."""
    W = np.zeros((len(pt_query), len(pt_grid)))
    for i, pv in enumerate(pt_query):
        if pv <= pt_grid[0]:
            W[i, 0] = 1.0
        elif pv >= pt_grid[-1]:
            W[i, -1] = 1.0
        else:
            j = np.searchsorted(pt_grid, pv) - 1
            t = (pv - pt_grid[j]) / (pt_grid[j + 1] - pt_grid[j])
            W[i, j]     = 1.0 - t
            W[i, j + 1] = t
    return W


def build_rho_precomputed(data, T_kin=0.100, beta_avg=0.4,
                          N_mc=1_000_000, cache_path=None):
    """
    Precompute ρ→ππ transfer matrices for the pion feed-down correction.

    Returns a dict with:
      pt_grid   : (n,)     mother/daughter bin centres
      A         : (n, n)   T_v2 * spec_m[None,:] — numerator weights
      norm_grid : (n,)     sum_m T_yield[d,m]*spec_m[m] — denominator per daughter bin
      W_data    : (n_pi, n) interpolation weights from grid → pion data pT
    """
    if cache_path and os.path.exists(cache_path):
        print(f'  Loading rho transfer cache: {cache_path}')
        d = np.load(cache_path)
        return {k: d[k] for k in d.files}

    resflow_path = os.path.expanduser('~/Research/ResFlow')
    if resflow_path not in sys.path:
        sys.path.insert(0, resflow_path)
    from resflow.transfer import build_transfer_matrices
    from resflow.physics import blastwave_dndpt, M_RHO

    pt_edges = np.arange(0.0, 2.51, 0.05)       # 50 bins, 0–2.5 GeV
    pt_grid  = 0.5 * (pt_edges[:-1] + pt_edges[1:])

    print(f'  Building rho transfer matrices (N_mc={N_mc:,})...')
    T_yield, T_v2, _ = build_transfer_matrices(pt_edges, N_mc=N_mc)

    spec_m  = blastwave_dndpt(pt_grid, M_RHO, T_kin=T_kin, beta_avg=beta_avg)
    spec_m /= spec_m.sum()

    A         = T_v2   * spec_m[np.newaxis, :]                   # (n, n)
    norm_grid = (T_yield * spec_m[np.newaxis, :]).sum(axis=1)    # (n,)

    pt_pi  = data['piplus'][0]
    W_data = _build_interp_weights(pt_pi, pt_grid)               # (n_pi, n)

    result = dict(pt_grid=pt_grid, A=A, norm_grid=norm_grid, W_data=W_data)

    if cache_path:
        np.savez(cache_path, **result)
        print(f'  Saved cache: {cache_path}')

    return result


# ── Transfer cache loading ─────────────────────────────────────────────────────

def _load_transfer_cache(data, energy, cache_dir='plots'):
    """
    Load the ρ→ππ transfer cache. Returns (A, norm_grid, W_data, pt_grid)
    as numpy arrays, or None if cache missing.
    """
    cache_path = Path(cache_dir) / energy / 'rho_transfer_cache.npz'
    if not cache_path.exists():
        print(f'  [transfer] Cache not found: {cache_path}. No ρ correction available.')
        return None
    d = np.load(cache_path)
    return d['A'], d['norm_grid'], d['W_data'], d['pt_grid']


# ── D_eff / ΔpT_eff precomputation (Taylor expansion, kept for reference) ──────

def _compute_deff_dpteff(data, energy, cache_dir='plots'):
    """
    Compute D_eff(pT) and ΔpT_eff(pT) from the ρ→ππ transfer cache,
    interpolated to the pion data pT grid.

    Returns (D_eff_data, dpt_eff_data) — each shape (n_pi_bins,).
    """
    cache_path = Path(cache_dir) / energy / 'rho_transfer_cache.npz'
    if not cache_path.exists():
        print(f'  [deff] Cache not found: {cache_path}. Using D_eff=1, dpt_eff=0 (no ρ correction).')
        n_pi = len(data['piplus'][0])
        return np.ones(n_pi), np.zeros(n_pi)

    d = np.load(cache_path)
    pt_grid   = d['pt_grid']    # (N,) mother bin centres
    A         = d['A']          # (N,N) A[d,m] = T_v2[d,m] * s[m]
    norm_grid = d['norm_grid']  # (N,)  norm[d] = sum_m T_yield[d,m] * s[m]
    W_data    = d['W_data']     # (n_pi, N) interpolation grid -> data pT

    safe_norm = np.where(norm_grid > 0, norm_grid, 1e-10)

    # D_eff[d] = sum_m T_v2[d,m]*s[m] / sum_m T_yield[d,m]*s[m] = A.sum(1) / norm
    D_eff_grid = np.maximum(A.sum(axis=1) / safe_norm, 0.0)

    # ΔpT_eff[d] = sum_m A[d,m] * (pT_m - pT_d) / norm[d]
    pT_diff = pt_grid[None, :] - pt_grid[:, None]     # (N, N): pT_m - pT_d
    dpt_eff_grid = (A * pT_diff).sum(axis=1) / safe_norm

    # Interpolate to data pT
    D_eff_data   = np.maximum(W_data @ D_eff_grid, 0.0)
    dpt_eff_data = W_data @ dpt_eff_grid

    return D_eff_data, dpt_eff_data


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model(data, cfg, lambda_data=None):
    pT  = {sp: d[0] for sp, d in data.items()}
    v2  = {sp: d[1] for sp, d in data.items()}
    err = {sp: np.maximum(d[2], cfg['err_floor']) for sp, d in data.items()}
    if lambda_data:
        for sp, (pt_lam, v2_lam, err_lam) in lambda_data.items():
            pT[sp]  = pt_lam
            v2[sp]  = v2_lam
            err[sp] = np.maximum(err_lam, cfg['err_floor'])

    f_prior    = cfg['f_prior']
    f_sigma    = cfg['f_prior_sigma']
    f_logit_mu = float(np.log(f_prior / (1.0 - f_prior)))

    with pm.Model() as model:

        # ── Richards sigmoid helper (PyTensor) ────────────────────────────────
        def richards(p_arr, a, b, c, d, nu):
            return a / (1.0 + pt.exp(-(p_arr - b) / c)) ** (1.0 / nu) - d

        def richards_prime(p_arr, a, b, c, d, nu):
            """Derivative of Richards sigmoid w.r.t. p_arr."""
            t = pt.exp(-(p_arr - b) / c)
            return (a / (c * nu)) * t / (1.0 + t) ** (1.0 / nu + 1.0)

        # ── v2^prod: produced (anti)quarks, pinned by antiparticle data ───────
        a_a  = pm.HalfNormal('a_a',  sigma=0.15)
        b_a  = pm.Normal    ('b_a',  mu=0.3, sigma=0.3)
        c_a  = pm.HalfNormal('c_a',  sigma=0.2)
        d_a  = pm.Normal    ('d_a',  mu=0.0, sigma=0.02)
        nu_a = pm.LogNormal ('nu_a', mu=0.0, sigma=0.8)

        def v2_prod(p_arr):
            return richards(p_arr, a_a, b_a, c_a, d_a, nu_a)

        # ── v2^tr: transported quarks (independent Richards sigmoid) ──────────
        a_tr  = pm.HalfNormal('a_tr',  sigma=0.15)
        b_tr  = pm.Normal    ('b_tr',  mu=0.3, sigma=0.3)
        c_tr  = pm.HalfNormal('c_tr',  sigma=0.2)
        d_tr  = pm.Normal    ('d_tr',  mu=0.0, sigma=0.02)
        nu_tr = pm.LogNormal ('nu_tr', mu=0.0, sigma=0.8)

        def v2_tr(p_arr):
            return richards(p_arr, a_tr, b_tr, c_tr, d_tr, nu_tr)

        # ── Isospin-symmetric mixing fraction f (f_u = f_d = f) ──────────────
        # Isospin asymmetry (f_d > f_u from the Au neutron excess) is ignored
        # for now; both u and d share a single transported fraction f.
        # Prior: logit(f) ~ N(logit(tanh(μ_B/3T_ch)), f_sigma).
        f_logit = pm.Normal('f_logit', mu=f_logit_mu, sigma=f_sigma)
        f       = pm.Deterministic('f', pt.sigmoid(f_logit))

        # ── v2^s: strange (fully independent) ────────────────────────────────
        a_s  = pm.HalfNormal('a_s',  sigma=0.15)
        b_s  = pm.Normal    ('b_s',  mu=0.3, sigma=0.3)
        c_s  = pm.HalfNormal('c_s',  sigma=0.2)
        d_s  = pm.Normal    ('d_s',  mu=0.0, sigma=0.02)
        nu_s = pm.LogNormal ('nu_s', mu=0.0, sigma=0.8)

        def v2_s(p_arr):
            return richards(p_arr, a_s, b_s, c_s, d_s, nu_s)

        # ── Mixture: v2^u and v2^d ────────────────────────────────────────────
        def qv2(p_arr, flavor):
            if flavor in ('u', 'd'):
                return f * v2_tr(p_arr) + (1.0 - f) * v2_prod(p_arr)
            elif flavor == 'a':
                return v2_prod(p_arr)
            elif flavor == 's':
                return v2_s(p_arr)

        def qv2_prime(p_arr, flavor):
            """Derivative of quark v2 w.r.t. quark pT."""
            if flavor in ('u', 'd'):
                return f * richards_prime(p_arr, a_tr, b_tr, c_tr, d_tr, nu_tr) + \
                       (1.0 - f) * richards_prime(p_arr, a_a, b_a, c_a, d_a, nu_a)
            elif flavor == 'a':
                return richards_prime(p_arr, a_a, b_a, c_a, d_a, nu_a)
            elif flavor == 's':
                return richards_prime(p_arr, a_s, b_s, c_s, d_s, nu_s)

        # ── S-curve pion correction (difference of two exponentials) ─────────
        # See docs/experiments_2026-05-01.md attempt 7.
        A_fast   = pm.HalfNormal('A_fast',   sigma=0.10)
        tau_fast = pm.HalfNormal('tau_fast', sigma=1.50)
        B_slow   = pm.HalfNormal('B_slow',   sigma=0.20)
        tau_slow = pm.HalfNormal('tau_slow', sigma=1.00)

        def s_curve(p_arr):
            return A_fast * pt.exp(-p_arr / tau_fast) - B_slow * pt.exp(-p_arr / tau_slow)

        # ── Flat pion offset ─────────────────────────────────────────────────
        eps0_pi = pm.Normal('eps0_pi', mu=0.0, sigma=0.02)

        # ── Likelihood (all species except pions get pure NCQ) ───────────────
        def ncq_pred(sp):
            p = pT[sp]
            if sp == 'piplus':
                return qv2(p/2, 'u') + qv2(p/2, 'a') + s_curve(p) + eps0_pi
            elif sp == 'piminus':
                return qv2(p/2, 'a') + qv2(p/2, 'd') + s_curve(p) + eps0_pi
            elif sp == 'kplus':
                return qv2(p/2, 'u') + qv2(p/2, 's')
            elif sp == 'kminus':
                return qv2(p/2, 'a') + qv2(p/2, 's')
            elif sp == 'proton':
                return 2*qv2(p/3, 'u') + qv2(p/3, 'd')
            elif sp == 'antiproton':
                return 3*qv2(p/3, 'a')
            elif sp == 'lambda':
                return qv2(p/3, 'u') + qv2(p/3, 'd') + qv2(p/3, 's')
            elif sp == 'lambdabar':
                return 2*qv2(p/3, 'a') + qv2(p/3, 's')

        FIT_SPECIES = ['piplus', 'piminus', 'kplus', 'kminus', 'proton', 'antiproton']
        if lambda_data:
            FIT_SPECIES += ['lambda', 'lambdabar']
        for sp in FIT_SPECIES:
            pm.Normal(f'obs_{sp}', mu=ncq_pred(sp), sigma=err[sp], observed=v2[sp])

    return model

# ── Posterior helpers ──────────────────────────────────────────────────────────

def _richards_np(p_grid, a, b, c, d, nu):
    """Vectorized Richards sigmoid: shapes (S,) params, (N,) p_grid → (S, N)."""
    return (a[:,None] / (1 + np.exp(-(p_grid[None,:] - b[:,None]) / c[:,None]))
            ** (1.0 / nu[:,None]) - d[:,None])

def _richards_prime_np(p_grid, a, b, c, d, nu):
    """Vectorized Richards sigmoid derivative: shapes (S,) params, (N,) p_grid → (S, N)."""
    t = np.exp(-(p_grid[None,:] - b[:,None]) / c[:,None])
    return (a[:,None] / (c[:,None] * nu[:,None])) * t / (1.0 + t) ** (1.0 / nu[:,None] + 1.0)


def _flat(post, name):
    return post[name].values.flatten()


def eval_quark_v2_posterior(trace, flavor, p_grid):
    post = trace.posterior
    if flavor == 'a':
        return _richards_np(p_grid,
                            _flat(post,'a_a'), _flat(post,'b_a'),
                            _flat(post,'c_a'), _flat(post,'d_a'), _flat(post,'nu_a'))
    elif flavor == 'tr':
        return _richards_np(p_grid,
                            _flat(post,'a_tr'), _flat(post,'b_tr'),
                            _flat(post,'c_tr'), _flat(post,'d_tr'), _flat(post,'nu_tr'))
    elif flavor == 's':
        return _richards_np(p_grid,
                            _flat(post,'a_s'), _flat(post,'b_s'),
                            _flat(post,'c_s'), _flat(post,'d_s'), _flat(post,'nu_s'))
    elif flavor in ('u', 'd'):
        fq   = _flat(post, 'f')[:,None]
        prod = eval_quark_v2_posterior(trace, 'a', p_grid)
        tr   = eval_quark_v2_posterior(trace, 'tr', p_grid)
        return fq * tr + (1.0 - fq) * prod
    else:
        raise ValueError(f'Unknown flavor: {flavor}')


def eval_pion_decay_posterior(trace, rho, p_query):
    """
    Evaluate the ρ→ππ decay v2 at p_query for every posterior sample.
    Returns shape (n_samples, n_query).
    """
    post      = trace.posterior
    pt_grid   = rho['pt_grid']
    A         = rho['A']          # (n, n)
    norm_grid = rho['norm_grid']  # (n,)

    pt_m2 = pt_grid / 2  # NCQ: evaluate quark v2 at pT/2

    v2_prod_m = _richards_np(pt_m2,
                             _flat(post,'a_a'), _flat(post,'b_a'),
                             _flat(post,'c_a'), _flat(post,'d_a'), _flat(post,'nu_a'))
    v2_tr_m   = _richards_np(pt_m2,
                             _flat(post,'a_tr'), _flat(post,'b_tr'),
                             _flat(post,'c_tr'), _flat(post,'d_tr'), _flat(post,'nu_tr'))

    fq = _flat(post, 'f')[:,None]

    v2_ud_m = fq * v2_tr_m + (1 - fq) * v2_prod_m           # v2_u = v2_d (isospin-symmetric)
    # v2_rho = v2_a(pt/2) + 0.5*(v2_u(pt/2) + v2_d(pt/2))  [ρ0 NCQ]
    v2_rho_m = v2_prod_m + v2_ud_m                          # (S, n)

    # Transfer: v2_decay_grid[s,d] = sum_m A[d,m]*v2_rho[s,m] / norm_grid[d]
    safe_norm     = np.where(norm_grid > 0, norm_grid, 1e-10)
    v2_decay_grid = (v2_rho_m @ A.T) / safe_norm[None, :]    # (S, n)

    W = _build_interp_weights(p_query, pt_grid)               # (n_query, n)
    return v2_decay_grid @ W.T                                # (S, n_query)


def _s_curve_np(p, post):
    return (_flat(post, 'A_fast')[:,None] * np.exp(-p[None,:] / _flat(post, 'tau_fast')[:,None])
            - _flat(post, 'B_slow')[:,None] * np.exp(-p[None,:] / _flat(post, 'tau_slow')[:,None]))


def hadron_pred_posterior(trace, sp, pT_arr, rho_cache=None):
    p = pT_arr
    post = trace.posterior

    ncq_map = {
        'piplus':     lambda: (eval_quark_v2_posterior(trace,'u',p/2)
                               + eval_quark_v2_posterior(trace,'a',p/2)
                               + _s_curve_np(p, post)
                               + _flat(post, 'eps0_pi')[:,None]),
        'piminus':    lambda: (eval_quark_v2_posterior(trace,'a',p/2)
                               + eval_quark_v2_posterior(trace,'d',p/2)
                               + _s_curve_np(p, post)
                               + _flat(post, 'eps0_pi')[:,None]),
        'kplus':      lambda: eval_quark_v2_posterior(trace,'u',p/2) + eval_quark_v2_posterior(trace,'s',p/2),
        'kminus':     lambda: eval_quark_v2_posterior(trace,'a',p/2) + eval_quark_v2_posterior(trace,'s',p/2),
        'proton':     lambda: 2*eval_quark_v2_posterior(trace,'u',p/3) + eval_quark_v2_posterior(trace,'d',p/3),
        'antiproton': lambda: 3*eval_quark_v2_posterior(trace,'a',p/3),
        'lambda':     lambda: eval_quark_v2_posterior(trace,'u',p/3) + eval_quark_v2_posterior(trace,'d',p/3) + eval_quark_v2_posterior(trace,'s',p/3),
        'lambdabar':  lambda: 2*eval_quark_v2_posterior(trace,'a',p/3) + eval_quark_v2_posterior(trace,'s',p/3),
    }
    return ncq_map[sp]()


def _interp_deff_to_pT(pT_query, pt_grid, D_eff_grid, dpt_eff_grid):
    """Interpolate D_eff and dpt_eff from grid to query pT values."""
    from scipy.interpolate import interp1d
    valid = np.isfinite(D_eff_grid) & np.isfinite(dpt_eff_grid)
    D_interp = interp1d(pt_grid[valid], D_eff_grid[valid], kind='linear',
                         bounds_error=False, fill_value=(D_eff_grid[valid][0],
                                                          D_eff_grid[valid][-1]))
    dpt_interp = interp1d(pt_grid[valid], dpt_eff_grid[valid], kind='linear',
                           bounds_error=False, fill_value=(dpt_eff_grid[valid][0],
                                                            dpt_eff_grid[valid][-1]))
    return D_interp(pT_query), dpt_interp(pT_query)


def pct_band(curves, ax, x, color, zorder=1, alpha_fill=0.25):
    med  = np.median(curves, axis=0)
    lo68 = np.percentile(curves, 16,   axis=0)
    hi68 = np.percentile(curves, 84,   axis=0)
    lo95 = np.percentile(curves, 2.5,  axis=0)
    hi95 = np.percentile(curves, 97.5, axis=0)
    ax.fill_between(x, lo95, hi95, alpha=alpha_fill*0.6, color=color, zorder=zorder)
    ax.fill_between(x, lo68, hi68, alpha=alpha_fill,     color=color, zorder=zorder)
    ax.plot(x, med, color=color, lw=2, zorder=zorder+1)
    return med

# ── Plots ───────────────────────────────────────────────────────────────────────

def energy_label(energy):
    return energy.replace('p', '.').replace('GeV', ' GeV')


def plot_quark_v2_functions(trace, out_dir, energy, ext='pdf'):
    p_grid = np.linspace(0.08, 1.00, 400)
    flavor_info = [
        ('u',  r'$v_2^u$',              'steelblue'),
        ('d',  r'$v_2^d$',              'darkorange'),
        ('a',  r'$v_2^{\rm prod}$',     'forestgreen'),
        ('tr', r'$v_2^{\rm tr}$',       'crimson'),
        ('s',  r'$v_2^s$',              'firebrick'),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), sharey=True)
    for ax, (f, lbl, col) in zip(axes, flavor_info):
        curves = eval_quark_v2_posterior(trace, f, p_grid)
        pct_band(curves, ax, p_grid, col)
        ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.5)
        ax.set_xlabel(r'$p_T/n_q$ (GeV/$c$)', fontsize=12)
        ax.set_title(lbl, fontsize=12)
        ax.set_xlim(0.08, 1.00)
    axes[0].set_ylabel(r'$v_2^{\rm quark}$', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / f'quark_v2_functions.{ext}')
    plt.close(fig)
    print(f'  Saved quark_v2_functions.{ext}')


def plot_transported_signal(trace, out_dir, energy, cfg, ext='pdf'):
    """
    3×1 (vertical): v2^tr vs v2^prod / transported excess v2^tr-v2^prod / f posterior.
    """
    LBL, TIT, LEG, SUP, TICK = 20, 20, 17, 24, 16
    p_grid = np.linspace(0.08, 1.00, 400)
    post   = trace.posterior
    f_samples   = _flat(post, 'f')
    prod_curves = eval_quark_v2_posterior(trace, 'a', p_grid)
    tr_curves   = eval_quark_v2_posterior(trace, 'tr', p_grid)

    fig, axes = plt.subplots(3, 1, figsize=(7, 14))

    # Panel 0: v2^tr vs v2^prod
    ax0 = axes[0]
    pct_band(prod_curves, ax0, p_grid, 'forestgreen', alpha_fill=0.3)
    pct_band(tr_curves,   ax0, p_grid, 'crimson',     alpha_fill=0.3)
    ax0.plot([], [], color='forestgreen', lw=2, label=r'$v_2^{\rm prod}$')
    ax0.plot([], [], color='crimson',     lw=2, label=r'$v_2^{\rm tr}$')
    ax0.axhline(0, color='k', lw=0.5, ls='--', alpha=0.5)
    ax0.set_xlabel(r'$p_T/n_q$ (GeV/$c$)', fontsize=LBL)
    ax0.set_ylabel(r'$v_2^{\rm quark}$', fontsize=LBL)
    ax0.set_title(r'$v_2^{\rm tr}$ vs $v_2^{\rm prod}$', fontsize=TIT)
    ax0.legend(fontsize=LEG)
    ax0.set_xlim(0.08, 1.00)
    ax0.tick_params(axis='both', labelsize=TICK)

    # Panel 1: transported excess v2^tr - v2^prod
    ax1 = axes[1]
    diff_curves = tr_curves - prod_curves
    pct_band(diff_curves, ax1, p_grid, 'darkred', alpha_fill=0.3)
    ax1.axhline(0, color='k', lw=0.5, ls='--', alpha=0.5)
    ax1.set_xlabel(r'$p_T/n_q$ (GeV/$c$)', fontsize=LBL)
    ax1.set_ylabel(r'$v_2^{\rm tr} - v_2^{\rm prod}$', fontsize=LBL)
    ax1.set_title(r'Transported excess: $v_2^{\rm tr} - v_2^{\rm prod}$', fontsize=TIT)
    ax1.set_xlim(0.08, 1.00)
    ax1.tick_params(axis='both', labelsize=TICK)

    # Panel 2: f posterior, with prior mean marked
    ax2 = axes[2]
    ax2.hist(f_samples, bins=50, density=True, color='steelblue', alpha=0.6,
             label=r'Posterior $f$')
    ax2.axvline(cfg['f_prior'], color='k', ls='--', lw=1.5,
                label=f'Prior mean $= {cfg["f_prior"]:.2f}$')
    ax2.set_xlabel(r'$f$', fontsize=LBL)
    ax2.set_ylabel('Density', fontsize=LBL)
    ax2.set_title(r'Posterior of $f$', fontsize=TIT)
    ax2.set_xlim(0, 1)
    ax2.legend(fontsize=LEG)
    ax2.tick_params(axis='both', labelsize=TICK)

    fig.tight_layout()
    fig.savefig(out_dir / f'transported_signal.{ext}', bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved transported_signal.{ext}')


def plot_posterior_predictive(trace, data, out_dir, energy, lambda_data=None, ext='pdf'):
    LBL, ANN, LEG, SUP, TICK = 26, 20, 18, 26, 16
    p_fine = np.linspace(0.05, 3.2, 800)
    fig = plt.figure(figsize=(18, 9))
    gs  = fig.add_gridspec(ncols=4, nrows=2, hspace=0.0, wspace=0.0)
    axes = gs.subplots(sharex='col', sharey='row').flatten()
    for idx, sp in enumerate(PARTICLES):
        ax = axes[idx]
        if sp in data:
            pT_d, v2_d, err_d = data[sp]
        elif lambda_data and sp in lambda_data:
            pT_d, v2_d, err_d = lambda_data[sp]
        else:
            continue
        col    = COLORS[sp]
        curves = hadron_pred_posterior(trace, sp, p_fine)
        pct_band(curves, ax, p_fine, col)
        ax.errorbar(pT_d, v2_d, yerr=err_d, fmt='o', color='k',
                    ms=6, capsize=3, lw=1.5, label='Data', zorder=10)
        pred_at_data = hadron_pred_posterior(trace, sp, pT_d)
        pred_med     = np.median(pred_at_data, axis=0)
        valid        = np.isfinite(v2_d) & np.isfinite(err_d) & np.isfinite(pred_med) & (err_d > 0)
        if valid.sum() > 0:
            # Per-panel chi^2 over N data points (NOT a reduced chi^2: the ~21 model
            # parameters are global/shared across all species, so per-species N-p is not the
            # right ndf). Labeled chi^2/N to make that explicit; do not read as a GOF. (M15)
            chi2  = np.sum(((v2_d[valid] - pred_med[valid]) / err_d[valid]) ** 2)
            n_pts = valid.sum()
            ax.annotate(fr'$\chi^2/N={chi2/n_pts:.1f}$ ({n_pts})', xy=(0.95, 0.06),
                        xycoords='axes fraction', ha='right', va='bottom',
                        fontsize=ANN, color='dimgray')
        ax.annotate(LABELS[sp], xy=(0.06, 0.92), xycoords='axes fraction',
                    ha='left', va='top', fontsize=SUP)
        ax.set_xlim(0, 3.2)
        ax.tick_params(axis='both', labelsize=TICK)
    axes[0].set_ylim(-0.01, 0.165)   # meson row (shared)
    axes[4].set_ylim(-0.01, 0.235)   # baryon/Lambda row (shared)
    axes[0].legend(fontsize=LEG, frameon=False, loc='center right')

    # Shared x/y labels via an invisible full-figure overlay (matches ratio.pdf)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel(r'$p_T$ (GeV/$c$)', fontsize=LBL, labelpad=20)
    plt.ylabel(r'$v_2$', fontsize=LBL, labelpad=25)

    fig.savefig(out_dir / f'posterior_predictive.{ext}', bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved posterior_predictive.{ext}')


def plot_quark_comparison(trace, out_dir, energy, ext='pdf'):
    p_grid = np.linspace(0.08, 1.00, 400)
    flavor_info = [
        ('u',  r'$v_2^u$',          'steelblue'),
        ('d',  r'$v_2^d$',          'darkorange'),
        ('a',  r'$v_2^{\rm prod}$', 'forestgreen'),
        ('tr', r'$v_2^{\rm tr}$',   'crimson'),
        ('s',  r'$v_2^s$',          'firebrick'),
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    for f, lbl, col in flavor_info:
        curves = eval_quark_v2_posterior(trace, f, p_grid)
        med  = np.median(curves, 0)
        lo68 = np.percentile(curves, 16, 0)
        hi68 = np.percentile(curves, 84, 0)
        ax.fill_between(p_grid, lo68, hi68, alpha=0.20, color=col)
        ax.plot(p_grid, med, color=col, lw=2, label=lbl)
    ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.5)
    ax.set_xlabel(r'$p_T/n_q$ (GeV/$c$)', fontsize=13)
    ax.set_ylabel(r'$v_2^{\rm quark}$', fontsize=13)
    ax.set_xlim(0.08, 1.00)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / f'quark_v2_comparison.{ext}')
    plt.close(fig)
    print(f'  Saved quark_v2_comparison.{ext}')

# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    energy   = args.energy
    cen_bins = args.cen_bins
    ep       = args.ep
    data_dir = args.data_dir or f'result/sys_tag_0/energy_{energy}'
    out_dir  = Path(args.out_dir or f'plots/{energy}')
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'=== Quark v2 mixture model: {energy}, cen bins {cen_bins}, EP={ep} ===')

    print('\n=== Thermal model prior ===')
    f_prior, T_ch, muB = load_thermal_prior(
        energy, args.thermal_csv, args.thermal_cen_labels)
    print(f'  T_ch = {T_ch:.1f} MeV,  μ_B = {muB:.1f} MeV')
    print(f'  f_prior = tanh(μ_B/3T) = {f_prior:.3f}')
    print(f'  logit(f_prior) = {np.log(f_prior/(1-f_prior)):.3f},  σ_logit = {args.f_prior_sigma}')

    cfg = dict(
        rebin           = args.rebin,
        meson_pt_lo     = args.meson_pt_lo,
        meson_pt_hi     = args.meson_pt_hi,
        baryon_pt_lo    = args.baryon_pt_lo,
        baryon_pt_hi    = args.baryon_pt_hi,
        sigma_ncq       = args.sigma_ncq,
        sigma_ncq_slp   = args.sigma_ncq_slp,
        eps_p_ref       = DEFAULTS['eps_p_ref'],
        err_floor       = DEFAULTS['err_floor'],
        f_prior         = f_prior,
        f_prior_sigma   = args.f_prior_sigma,
    )

    print('\n=== Loading data ===')
    pt_centers, v2_all, err_all = load_merged_v2(data_dir, cen_bins, ep, args.csv_prefix)

    print('\n=== Loading Lambda/Lambdabar data ===')
    try:
        pt_lam, v2_lam, err_lam, v2_lbr, err_lbr = load_lambda_v2(data_dir, cen_bins, ep)
        lambda_data = {
            'lambda':    (pt_lam, v2_lam, err_lam),
            'lambdabar': (pt_lam, v2_lbr, err_lbr),
        }
    except (FileNotFoundError, KeyError):
        print('  No Lambda/Lambdabar fit data found; skipping Lambda predictions.')
        lambda_data = None

    print('\n=== Preparing datasets ===')
    data = prepare_datasets(pt_centers, v2_all, err_all, cfg)
    print_dataset_summary(data, lambda_data)

    print('\n=== Building model ===')
    model = build_model(data, cfg, lambda_data)

    print('\n=== Sampling ===')
    with model:
        trace = pm.sample(
            draws=args.draws, tune=args.tune,
            target_accept=args.target_accept, chains=args.chains,
            max_treedepth=args.max_treedepth,
            random_seed=args.random_seed, progressbar=True,
        )

    print('\n=== Convergence diagnostics ===')
    param_names = (
        ['a_tr', 'b_tr', 'c_tr', 'd_tr', 'nu_tr'] +
        ['a_a',  'b_a',  'c_a',  'd_a',  'nu_a']  +
        ['a_s',  'b_s',  'c_s',  'd_s',  'nu_s']  +
        ['f', 'eps0_pi'] +
        ['A_fast', 'tau_fast', 'B_slow', 'tau_slow']
    )
    hier_names = []
    summary    = az.summary(trace, var_names=param_names + hier_names)
    print(summary[['mean','sd','r_hat','ess_bulk']].to_string())
    max_rhat = summary['r_hat'].max()
    min_ess  = summary['ess_bulk'].min()
    print(f'\nMax R-hat:   {max_rhat:.4f}  (target <1.01)')
    print(f'Min ESS:     {min_ess:.0f}   (target >400)')
    n_div = int(trace.sample_stats['diverging'].values.sum())
    print(f'Divergences: {n_div}  (target 0)')

    print('\n=== Saving trace ===')
    trace_path = out_dir / 'trace.nc'
    az.to_netcdf(trace, trace_path)
    print(f'  Saved {trace_path}')

    print('\n=== Generating plots ===')
    plot_quark_v2_functions(trace, out_dir, energy)
    plot_transported_signal(trace, out_dir, energy, cfg)
    plot_posterior_predictive(trace, data, out_dir, energy, lambda_data)
    plot_quark_comparison(trace, out_dir, energy)

    print(f'\nDone. Output in {out_dir}/')


if __name__ == '__main__':
    main()
