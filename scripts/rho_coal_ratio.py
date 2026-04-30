#!/usr/bin/env python
"""Compute coalescence R ratio using rho v2 extracted via ResFlow inversion.

R = [v2(rho-) - (2/3)*v2(pbar)] / [v2(rho+) - (2/3)*v2(pbar)]

Usage:
    conda activate lambda_v1
    python scripts/rho_coal_ratio.py
    python scripts/rho_coal_ratio.py --n_bootstrap 50 --n_steps 3000  # faster
    python scripts/rho_coal_ratio.py --use_eff  # use efficiency-corrected data
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from uncertainties import ufloat, unumpy
from scipy.optimize import minimize
from resflow import invert_v2
from resflow.physics import blastwave_dndpt

# ── Constants ──────────────────────────────────────────────────────────────
N_CEN = 9
CEN_X = np.array([75., 65., 55., 45., 35., 25., 15., 7.5, 2.5])
CEN_LABELS = ["70-80%", "60-70%", "50-60%", "40-50%", "30-40%",
              "20-30%", "10-20%", "5-10%", "0-5%"]
PT_BIN_WIDTH_FINE = 0.01  # GeV, v2 TProfile has 1000 bins over [0,10], we store first 500
N_PT_BINS_FINE = 500
REBIN_EDGES = np.arange(0, 2.05, 0.05)  # 40 bins of 0.05 GeV
PT_LO = 0.16   # pT/nq=0.08, nq=2
PT_HI = 1.20   # pT/nq=0.60, nq=2
M_PION = 0.13957  # GeV

# Glauber d/u ratio (from generate_paper_plots.py)
CEN_GLAUBER = np.array([2.5, 7.5, 15, 25, 35, 45, 55, 65, 75])
RATIO_DU = np.array([1.13304, 1.13351, 1.13777, 1.14464, 1.15486,
                      1.17016, 1.19128, 1.22017, 1.26102])


# ── Data loading ───────────────────────────────────────────────────────────
def load_cen_v2(base_path, cen, EP, use_eff=False):
    """Load per-pT v2 for a single centrality bin."""
    prefix = "v2_eff_corrected" if use_eff else "v2_noeff_corrected"
    df = pd.read_csv(f"{base_path}/{prefix}_cen{cen}.csv")
    pt_centers = np.linspace(
        PT_BIN_WIDTH_FINE / 2,
        N_PT_BINS_FINE * PT_BIN_WIDTH_FINE - PT_BIN_WIDTH_FINE / 2,
        N_PT_BINS_FINE,
    )
    return dict(
        pt=pt_centers,
        pip_v2=df[f"piplus_v2_{EP}"].values,
        pip_err=df[f"piplus_v2_err_{EP}"].values,
        pip_counts=df["piplus_counts"].values,
        pim_v2=df[f"piminus_v2_{EP}"].values,
        pim_err=df[f"piminus_v2_err_{EP}"].values,
        pim_counts=df["piminus_counts"].values,
    )


def rebin_v2(pt_fine, v2, v2_err, counts, new_edges):
    """Rebin v2(pT) to coarser bins using counts-weighted average."""
    n_new = len(new_edges) - 1
    v2_out = np.full(n_new, np.nan)
    err_out = np.full(n_new, np.nan)
    counts_out = np.zeros(n_new)

    for i in range(n_new):
        mask = (pt_fine >= new_edges[i]) & (pt_fine < new_edges[i + 1])
        w = counts[mask]
        if w.sum() == 0:
            continue
        v2_out[i] = np.average(v2[mask], weights=w)
        err_out[i] = np.sqrt(np.sum((w * v2_err[mask]) ** 2)) / w.sum()
        counts_out[i] = w.sum()

    return v2_out, err_out, counts_out


# ── Blast-wave fitting ─────────────────────────────────────────────────────
def fit_blastwave(pt_centers, counts, mass=M_PION):
    """Fit BW spectrum to counts(pT). Return (T_kin, beta_avg, n_flow).

    counts = raw TProfile entries ∝ dN/dpT (tracks per pT bin).
    blastwave_dndpt returns pT * mT * spectrum ∝ dN/dpT.
    So we compare shapes directly after normalizing both.
    But at low pT the counts include TPC efficiency turn-on,
    so restrict fit to pT > 0.3 GeV where efficiency is flat.
    """
    mask = (counts > 0) & (pt_centers > 0.3)
    pt_fit = pt_centers[mask]
    counts_fit = counts[mask]
    counts_norm = counts_fit / counts_fit.max()

    def chi2(params):
        T, beta, n = params
        model = blastwave_dndpt(pt_fit, mass, T_kin=T, beta_avg=beta, n_flow=n)
        model_norm = model / model.max()
        return np.sum((model_norm - counts_norm) ** 2 / (counts_norm + 1e-10))

    result = minimize(chi2, x0=[0.1, 0.5, 1.0],
                      bounds=[(0.05, 0.2), (0.2, 0.7), (0.3, 3.0)],
                      method="L-BFGS-B")
    return result.x  # (T_kin, beta_avg, n_flow)


# ── Integration ────────────────────────────────────────────────────────────
def integrate_v2(v2_pt, pt_centers, lo=PT_LO, hi=PT_HI):
    """Integrate v2 over pT range (uniform weight, equal-width bins)."""
    mask = (pt_centers >= lo) & (pt_centers <= hi) & ~np.isnan(v2_pt)
    if mask.sum() == 0:
        return np.nan
    return np.mean(v2_pt[mask])


# ── Inversion ──────────────────────────────────────────────────────────────
def run_single_inversion(pt_edges, v2, v2_err, N_mc, n_steps, bw_params):
    """Run ResFlow inversion with given BW params, return InversionResult."""
    T_kin, beta_avg, n_flow = bw_params
    return invert_v2(
        pt_edges, v2, v2_err, pid="rho",
        T_kin=T_kin, beta_avg=beta_avg, n_flow=n_flow,
        N_mc=N_mc, n_steps=n_steps, verbose=False,
    )


def bootstrap_integrated_rho_v2(pt_edges, v2, v2_err, n_boot, N_mc, n_steps, bw_params):
    """Bootstrap: fluctuate input v2, invert, integrate. Return (samples,)."""
    pt_centers = (pt_edges[:-1] + pt_edges[1:]) / 2
    samples = np.empty(n_boot)
    for b in range(n_boot):
        v2_boot = v2 + np.random.normal(0, np.where(np.isnan(v2_err), 0, v2_err))
        try:
            result = run_single_inversion(pt_edges, v2_boot, v2_err, N_mc, n_steps, bw_params)
            samples[b] = integrate_v2(result.v2_mother, pt_centers)
        except Exception:
            samples[b] = np.nan
    return samples


# ── Plotting ───────────────────────────────────────────────────────────────
def plot_bw_fits(bw_params_dict, spectra_dict, out_dir):
    """3x3 panel: pion counts(pT) + BW fit per centrality."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 12), sharex=True)
    axes = axes.flatten()

    for cen_idx in range(N_CEN):
        ax = axes[cen_idx]
        pt = spectra_dict[cen_idx]["pt"]
        counts = spectra_dict[cen_idx]["counts"]
        mask = counts > 0

        # data
        ax.plot(pt[mask], counts[mask] / counts[mask].max(), "k.", ms=2, label="Data")

        # BW fit
        T, beta, n = bw_params_dict[cen_idx]
        model = blastwave_dndpt(pt[mask], M_PION, T_kin=T, beta_avg=beta, n_flow=n)
        model_norm = model / model.max()
        ax.plot(pt[mask], model_norm, "r-", lw=1.5, label="BW fit")

        ax.set_title(CEN_LABELS[cen_idx], fontsize=11)
        ax.annotate(
            f"T={T*1e3:.0f} MeV\n"
            rf"$\beta_{{avg}}$={beta:.3f}"
            f"\nn={n:.2f}",
            xy=(0.95, 0.95), xycoords="axes fraction",
            fontsize=9, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )
        if cen_idx >= 6:
            ax.set_xlabel(r"$p_T$ (GeV)", fontsize=11)
        if cen_idx % 3 == 0:
            ax.set_ylabel("Normalized yield", fontsize=11)
        ax.set_yscale("log")
        ax.set_xlim(0, 2)
        if cen_idx == 0:
            ax.legend(fontsize=9, loc="lower left")

    fig.suptitle("Blast-wave fits to eff-corrected pion spectra — 19.6 GeV", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "bw_fits.pdf")
    fig.savefig(out_dir / "bw_fits.png", dpi=150)
    print(f"Saved: {out_dir / 'bw_fits.pdf'}")
    plt.close()


def plot_v2_diagnostic(EP, inversion_results, data_dict, bw_params_dict, out_dir,
                       charge="pip"):
    """3x3 diagnostic: rho v2, daughter v2, pion v2 data per centrality.

    charge: "pip" for pi+ / rho+, "pim" for pi- / rho-.
    """
    charge_label = r"$\pi^+$" if charge == "pip" else r"$\pi^-$"
    rho_label = r"$\rho^+$" if charge == "pip" else r"$\rho^-$"
    v2_key = f"{charge}_v2"
    err_key = f"{charge}_err"

    fig, axes = plt.subplots(3, 3, figsize=(14, 12), sharex=True)
    axes = axes.flatten()

    for cen_idx in range(N_CEN):
        ax = axes[cen_idx]
        d = data_dict[cen_idx]
        pt_c = d["pt_coarse"]
        valid = ~np.isnan(d[v2_key])

        ax.errorbar(pt_c[valid], d[v2_key][valid], d[err_key][valid],
                     fmt="o", ms=3, color="gray", alpha=0.6,
                     label=f"{charge_label} data", capsize=1)

        if cen_idx in inversion_results and charge in inversion_results[cen_idx]:
            res, color_set = inversion_results[cen_idx][charge]
            c_pred, c_mother, c_decay = color_set

            ax.plot(res.pt_centers, res.v2_pred, "-", color=c_pred, lw=1.5,
                    label=f"Fit ({rho_label})")
            ax.plot(res.pt_centers, res.v2_mother, "--", color=c_mother, lw=1.5,
                    label=f"{rho_label} " + r"$v_2$")
            ax.plot(res.pt_centers, res.v2_decay, ":", color=c_decay, lw=1,
                    label=f"Decay ({rho_label})")

            ax.annotate(
                f"f={res.f:.2f}",
                xy=(0.05, 0.85), xycoords="axes fraction",
                fontsize=8, ha="left", va="top",
            )

        T, beta, n = bw_params_dict[cen_idx]
        ax.set_title(CEN_LABELS[cen_idx], fontsize=11)
        ax.annotate(
            f"T={T*1e3:.0f}, " + rf"$\beta$={beta:.2f}",
            xy=(0.05, 0.95), xycoords="axes fraction",
            fontsize=8, ha="left", va="top",
        )

        if cen_idx >= 6:
            ax.set_xlabel(r"$p_T$ (GeV)", fontsize=11)
        if cen_idx % 3 == 0:
            ax.set_ylabel(r"$v_2$", fontsize=11)
        ax.set_xlim(0, 2)

    axes[0].legend(fontsize=7, loc="lower right", ncol=1)
    fig.suptitle(f"ResFlow inversion diagnostic ({charge_label}, {EP}) — 19.6 GeV",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / f"rho_v2_diagnostic_{EP}_{charge}.pdf")
    fig.savefig(out_dir / f"rho_v2_diagnostic_{EP}_{charge}.png", dpi=150)
    print(f"Saved: {out_dir / f'rho_v2_diagnostic_{EP}_{charge}.pdf'}")
    plt.close()


def plot_ratio(results, out_dir):
    """Single-panel centrality dependence of rho R ratio."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    marker_styles = {
        "TPC": dict(fmt="o", color="C0", markersize=6, capsize=3),
        "EPD": dict(fmt="s", color="C1", markersize=6, capsize=3),
    }

    for EP in ["TPC", "EPD"]:
        if EP not in results:
            continue
        R = results[EP]
        shift = 1.0 if EP == "TPC" else -1.0
        valid = ~np.isnan(unumpy.nominal_values(R))
        ax.errorbar(
            CEN_X[valid] + shift,
            unumpy.nominal_values(R)[valid],
            unumpy.std_devs(R)[valid],
            label=f"{EP} Event Plane",
            **marker_styles[EP],
        )

    ax.fill_between(CEN_GLAUBER, RATIO_DU - 0.001, RATIO_DU + 0.001,
                     color="C2", alpha=0.8, label="Glauber d/u")
    ax.axhline(315 / 276, color="C3", ls="--", label="315/276")
    ax.axhline(1, color="black", ls="--", label="1")
    ax.set_xlim(-5, 85)
    ax.set_ylim(0.509, 1.759)
    ax.set_xlabel("Centrality (%)", fontsize=15)
    ax.set_ylabel(
        r"$\frac{v_2^{\rho^-}-\frac{2}{3}v_2^{\bar{p}}}"
        r"{v_2^{\rho^+}-\frac{2}{3}v_2^{\bar{p}}}$",
        fontsize=18,
    )
    ax.annotate(
        r"AuAu, $\sqrt{s_{\text{NN}}}=$ 19.6 GeV" + "\n(rho from ResFlow)",
        xy=(0.95, 0.92), xycoords="axes fraction",
        fontsize=13, ha="right", va="top",
    )
    ax.legend(fontsize=12, frameon=False, loc="upper left")
    ax.tick_params(labelsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "rho_ratio.pdf")
    fig.savefig(out_dir / "rho_ratio.png", dpi=150)
    print(f"Saved: {out_dir / 'rho_ratio.pdf'}")
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base_path", default="result/sys_tag_0/energy_19p6GeV")
    parser.add_argument("--output_dir", default="plots/sys_tag_0/energy_19p6GeV")
    parser.add_argument("--n_bootstrap", type=int, default=100)
    parser.add_argument("--n_steps", type=int, default=5000)
    parser.add_argument("--N_mc", type=int, default=5_000_000)
    parser.add_argument("--N_mc_boot", type=int, default=1_000_000)
    parser.add_argument("--use_eff", action="store_true",
                        help="Use efficiency-corrected data for BW fit and v2")
    args = parser.parse_args()

    base = Path(args.base_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load resolution and integrated antiproton v2
    res_prefix = "v2_eff_corrected" if args.use_eff else "v2_noeff_corrected"
    df_res = pd.read_csv(base / f"{res_prefix}_res.csv")
    df_int = pd.read_csv(base / f"{res_prefix}.csv")

    pt_fine = np.linspace(PT_BIN_WIDTH_FINE / 2, 2.0 - PT_BIN_WIDTH_FINE / 2, N_PT_BINS_FINE)
    pt_centers_coarse = (REBIN_EDGES[:-1] + REBIN_EDGES[1:]) / 2

    # ── Step 1: Fit BW params per centrality ──────────────────────────────
    print("Fitting blast-wave parameters per centrality...")
    bw_params_dict = {}
    spectra_dict = {}

    for cen in range(1, N_CEN + 1):
        # Use averaged pi+ + pi- counts for spectrum shape
        data = load_cen_v2(str(base), cen, "TPC", use_eff=args.use_eff)
        avg_counts = (data["pip_counts"] + data["pim_counts"]) / 2.0
        bw = fit_blastwave(data["pt"], avg_counts)
        bw_params_dict[cen - 1] = bw
        spectra_dict[cen - 1] = dict(pt=data["pt"], counts=avg_counts)
        print(f"  cen{cen} ({CEN_LABELS[cen-1]}): T={bw[0]*1e3:.1f} MeV, "
              f"beta={bw[1]:.3f}, n={bw[2]:.2f}")

    # Plot BW fits
    plot_bw_fits(bw_params_dict, spectra_dict, out_dir)

    # ── Step 2: Inversion per centrality and EP ───────────────────────────
    ratio_results = {}

    for EP in ["TPC", "EPD"]:
        print(f"\n{'='*60}")
        print(f"Event Plane: {EP}")
        print(f"{'='*60}")

        rho_plus_v2 = np.empty(N_CEN)
        rho_plus_err = np.empty(N_CEN)
        rho_minus_v2 = np.empty(N_CEN)
        rho_minus_err = np.empty(N_CEN)

        # For diagnostic plot
        inv_results_ep = {}
        data_dict_ep = {}

        for cen in range(1, N_CEN + 1):
            cen_idx = cen - 1
            res_val = df_res[f"{EP}_res"].iloc[cen_idx]
            bw = bw_params_dict[cen_idx]

            if res_val <= 0 or np.isnan(res_val):
                print(f"  cen{cen}: SKIPPED (invalid resolution)")
                rho_plus_v2[cen_idx] = np.nan
                rho_plus_err[cen_idx] = np.nan
                rho_minus_v2[cen_idx] = np.nan
                rho_minus_err[cen_idx] = np.nan
                continue

            data = load_cen_v2(str(base), cen, EP, use_eff=args.use_eff)

            # Resolution-correct
            pip_v2 = data["pip_v2"] / res_val
            pip_err = data["pip_err"] / res_val
            pim_v2 = data["pim_v2"] / res_val
            pim_err = data["pim_err"] / res_val

            # Rebin
            pip_v2_rb, pip_err_rb, pip_counts_rb = rebin_v2(
                data["pt"], pip_v2, pip_err, data["pip_counts"], REBIN_EDGES)
            pim_v2_rb, pim_err_rb, pim_counts_rb = rebin_v2(
                data["pt"], pim_v2, pim_err, data["pim_counts"], REBIN_EDGES)

            # Store for diagnostic plot
            data_dict_ep[cen_idx] = dict(
                pt_coarse=pt_centers_coarse,
                pip_v2=pip_v2_rb, pip_err=pip_err_rb,
                pim_v2=pim_v2_rb, pim_err=pim_err_rb,
            )

            # Inversion
            print(f"  cen{cen} ({CEN_LABELS[cen_idx]}): pi+ ...", end="", flush=True)
            try:
                res_pip = run_single_inversion(REBIN_EDGES, pip_v2_rb, pip_err_rb,
                                               args.N_mc, args.n_steps, bw)
                print(f" f={res_pip.f:.3f}", end="")
            except Exception as e:
                print(f" FAILED ({e})", end="")
                res_pip = None

            print(f" | pi- ...", end="", flush=True)
            try:
                res_pim = run_single_inversion(REBIN_EDGES, pim_v2_rb, pim_err_rb,
                                               args.N_mc, args.n_steps, bw)
                print(f" f={res_pim.f:.3f}")
            except Exception as e:
                print(f" FAILED ({e})")
                res_pim = None

            inv_results_ep[cen_idx] = {}
            if res_pip is not None:
                inv_results_ep[cen_idx]["pip"] = (res_pip, ("C0", "C0", "C0"))
            if res_pim is not None:
                inv_results_ep[cen_idx]["pim"] = (res_pim, ("C1", "C1", "C1"))

            rho_plus_v2[cen_idx] = integrate_v2(res_pip.v2_mother, pt_centers_coarse) if res_pip else np.nan
            rho_minus_v2[cen_idx] = integrate_v2(res_pim.v2_mother, pt_centers_coarse) if res_pim else np.nan

            # Bootstrap
            if args.n_bootstrap > 0:
                print(f"    bootstrap ({args.n_bootstrap}) ...", end="", flush=True)
                boot_pip = bootstrap_integrated_rho_v2(
                    REBIN_EDGES, pip_v2_rb, pip_err_rb,
                    args.n_bootstrap, args.N_mc_boot, args.n_steps, bw)
                boot_pim = bootstrap_integrated_rho_v2(
                    REBIN_EDGES, pim_v2_rb, pim_err_rb,
                    args.n_bootstrap, args.N_mc_boot, args.n_steps, bw)
                rho_plus_err[cen_idx] = np.nanstd(boot_pip)
                rho_minus_err[cen_idx] = np.nanstd(boot_pim)
                print(f" done")
            else:
                rho_plus_err[cen_idx] = 0
                rho_minus_err[cen_idx] = 0

        # Diagnostic plots for this EP (one per charge)
        for ch in ["pip", "pim"]:
            plot_v2_diagnostic(EP, inv_results_ep, data_dict_ep, bw_params_dict,
                               out_dir, charge=ch)

        # Antiproton v2 (already pT-integrated)
        pbar_v2_raw = df_int[f"antiproton_v2_{EP}"].values
        pbar_err_raw = df_int[f"antiproton_v2_err_{EP}"].values
        res_vals = df_res[f"{EP}_res"].values
        pbar_v2 = unumpy.uarray(pbar_v2_raw / res_vals, pbar_err_raw / res_vals)

        # R ratio
        rho_p = unumpy.uarray(rho_plus_v2, rho_plus_err)
        rho_m = unumpy.uarray(rho_minus_v2, rho_minus_err)
        R = (rho_m - 2.0 / 3.0 * pbar_v2) / (rho_p - 2.0 / 3.0 * pbar_v2)
        ratio_results[EP] = R

        print(f"\n  {'Cen%':>6s}  {'rho+ v2':>10s}  {'rho- v2':>10s}  {'pbar v2':>10s}  {'R':>10s}")
        for c in range(N_CEN):
            print(f"  {CEN_X[c]:6.1f}  {rho_plus_v2[c]:10.5f}  {rho_minus_v2[c]:10.5f}"
                  f"  {unumpy.nominal_values(pbar_v2[c]):10.5f}  {R[c]}")

    # ── Final ratio plot ──────────────────────────────────────────────────
    plot_ratio(ratio_results, out_dir)


if __name__ == "__main__":
    main()
