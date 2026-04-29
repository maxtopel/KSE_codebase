"""Three-panel figure: reconstruction RMSE vs observable (ordered by KSE),
with overlaid SNR levels. Run after the per-system CSVs exist.

Usage:
    python figures.py                      # default: latest results dir
    python figures.py --date 2026-04-27    # explicit date
    python figures.py --csvs A B C         # explicit CSVs in (L, HP, C24) order
"""
from __future__ import annotations
import argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=RuntimeWarning)

INTEGRAL_OBS = {"x_int_w100", "x_int_w1000", "z_int_w100",
                "resource_int_w50", "resource_int_w500"}
# C1--C2 and C1--C3 probe bond-stretching/angle-bending vibrations rather
# than slow chain dynamics; excluded from the published analysis.
BOND_MODE_OBS = {"C1_to_C2", "C1_to_C3"}
EXCLUDE = INTEGRAL_OBS | BOND_MODE_OBS

# SNRs are picked per system to span its three regimes:
#   ∞      = noiseless baseline
#   peak   = SNR where rho is largest (varies by system)
#   sat    = SNR in that system's saturation regime
SNR_STYLE = {
    np.inf: ("#000000", r"$\infty$",  "o"),
    80:     ("#1a3a8f", r"$80\,$dB",  "s"),
    30:     ("#1a3a8f", r"$30\,$dB",  "s"),
    15:     ("#e89c1f", r"$15\,$dB",  "D"),
    5:      ("#c5402d", r"$5\,$dB",   "v"),
    -5:     ("#7a1a3f", r"$-5\,$dB",  "v"),
}
# Per-system SNR overlays (clean ∞ is always drawn separately as bold)
SNRS_PER_SYSTEM = {
    "lorenz63":        [30, 5],     # equivalence at clean, peak at 30, sat at 5
    "hastings_powell": [80, 15],    # clean already discrim; 80 = peak, 15 = sat
    "c24h50":          [15, -5],    # 15 = ρ peak, -5 = sat onset
}
YLABEL = {
    "lorenz63":        r"RMSE of $(x,y,z)$  [state units]",
    "hastings_powell": r"RMSE of $(R,C,P)$  [population]",
    "c24h50":          r"RMSE of $\{\mathbf{r}_C\}$  [nm]",
}
TITLE = {"lorenz63": "Lorenz-63", "hastings_powell": "Hastings--Powell",
         "c24h50": r"$\mathrm{C}_{24}\mathrm{H}_{50}$"}

# Shorter tick labels for common observables
OBS_SHORT = {
    "x": "$x$", "y": "$y$", "z": "$z$",
    "dxdt": r"$\dot x$", "dzdt": r"$\dot z$", "d2xdt2": r"$\ddot x$",
    "x_squared": "$x^2$", "y_squared": "$y^2$", "z_squared": "$z^2$",
    "log1p_z": r"$\log(1{+}z)$", "x_plus_y": "$x{+}y$",
    "radial": r"$\|\mathbf{r}\|$", "x_times_y": "$xy$", "x_times_z": "$xz$",
    "resource": "$R$", "consumer": "$C$", "predator": "$P$",
    "log_resource": r"$\log R$", "log_consumer": r"$\log C$", "log_predator": r"$\log P$",
    "log_total_biomass": r"$\log(R{+}C{+}P)$",
    "dresource_dt": r"$\dot R$", "dconsumer_dt": r"$\dot C$", "dpredator_dt": r"$\dot P$",
    "total_biomass": "$R{+}C{+}P$", "resource_squared": "$R^2$",
    "predator_minus_resource": "$P{-}R$",
    "resource_times_consumer": "$RC$", "predator_over_consumer": "$P/C$",
}


def short(obs):
    if obs in OBS_SHORT:
        return OBS_SHORT[obs]
    if obs.startswith("x_smooth_") or obs.startswith("z_smooth_"):
        v, w = obs.split("_smooth_")
        return rf"$\langle {v}\rangle_{{{w}}}$"
    if obs.startswith("resource_smooth_"):
        return rf"$\langle R\rangle_{{{obs.split('_')[-1]}}}$"
    if obs.startswith(("consumer_smooth_", "predator_smooth_")):
        v = "C" if obs.startswith("c") else "P"
        return rf"$\langle {v}\rangle_{{{obs.split('_')[-1]}}}$"
    if obs.startswith("C1_to_C"):
        return rf"C$_1$--C$_{{{obs[7:]}}}$"
    return obs.replace("_", r"\_")


def boot_ci(xs, n_boot=800, seed=0):
    xs = np.asarray(xs, dtype=float); xs = xs[np.isfinite(xs)]
    if len(xs) < 2:
        m = float(np.mean(xs)) if len(xs) else np.nan
        return m, m
    rng = np.random.default_rng(seed)
    means = np.array([np.mean(rng.choice(xs, size=len(xs), replace=True))
                      for _ in range(n_boot)])
    return tuple(np.percentile(means, [2.5, 97.5]))


def plot_panel(ax, df, system_key):
    df = df[~df.observable.isin(EXCLUDE)]
    clean = df[np.isinf(df.snr_db)]
    cell = clean.groupby("observable").agg(
        kse=("kse_ub", "mean"), rmse=("rmse", "mean"), rmse_sd=("rmse", "std")
    ).reset_index().sort_values("kse").reset_index(drop=True)
    cell["rank"] = np.arange(1, len(cell) + 1)
    rank_of = dict(zip(cell.observable, cell["rank"]))

    # Per-system noisy-SNR overlays (clean ∞ drawn separately, bold)
    snr_list = SNRS_PER_SYSTEM.get(system_key, [])
    for snr in snr_list:
        sub = df[df.snr_db == snr]
        if sub.empty: continue
        n = sub.groupby("observable").agg(rmse=("rmse", "mean")).reset_index()
        n["rank"] = n.observable.map(rank_of)
        n = n.dropna(subset=["rank"]).sort_values("rank")
        c, lab, _ = SNR_STYLE.get(snr, ("gray", f"{snr}", "."))
        ax.plot(n["rank"], n["rmse"], "-", color=c, lw=1.0, alpha=0.7, label=lab)

    # Clean: bold black with shaded bootstrap CI
    lo = [boot_ci(clean[clean.observable == o].rmse.values, seed=hash(o) & 0xffff)[0] for o in cell.observable]
    hi = [boot_ci(clean[clean.observable == o].rmse.values, seed=hash(o) & 0xffff)[1] for o in cell.observable]
    ax.fill_between(cell["rank"], lo, hi, color="black", alpha=0.12, linewidth=0)
    ax.plot(cell["rank"], cell.rmse, "-", color="black", lw=1.6)
    ax.errorbar(cell["rank"], cell.rmse, yerr=cell.rmse_sd, fmt="o", color="black",
                ms=5.5, capsize=2.5, label=r"$\infty$")

    ax.set_xticks(cell["rank"])
    ax.set_xticklabels([short(o) for o in cell.observable], rotation=60, ha="right", fontsize=10)
    ax.set_xlabel(r"observable (ordered by $h^{KS,UB}_i$)", fontsize=11)
    ax.set_ylabel(YLABEL.get(system_key, "RMSE"), fontsize=11)
    ax.set_yscale("log")
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.4)
    ax.set_title(TITLE.get(system_key, system_key), fontsize=12)

    # Top axis: KSE values. Use ticks (min, q1, median, q3, max)
    n = len(cell)
    idx = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    idx = sorted(set(idx))
    ax2 = ax.twiny(); ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(cell["rank"].values[idx])
    ax2.set_xticklabels([f"{cell.kse.values[i]:.2f}" for i in idx], fontsize=10)
    ax2.set_xlabel(r"$h^{KS,UB}_i$", fontsize=11, labelpad=4)

    rho, p = spearmanr(cell.kse, cell.rmse)
    ax.text(0.04, 0.96, rf"$\mathrm{{SNR}}{{=}}\infty$: $\rho={rho:+.2f}$, "
                       rf"$p={p:.1e}$, $n={len(cell)}$",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.85))
    ax.legend(fontsize=9, frameon=False, loc="lower right", ncol=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results")
    ap.add_argument("--date", default=None)
    ap.add_argument("--csvs", nargs=3, default=None,
                    help="(lorenz, hastings_powell, c24h50) CSV paths")
    ap.add_argument("--out", default="../figures/fig_rmse_vs_kse.pdf")
    args = ap.parse_args()

    if args.csvs:
        paths = {k: Path(v) for k, v in zip(["lorenz63", "hastings_powell", "c24h50"], args.csvs)}
    else:
        root = Path(args.results)
        if args.date is None:
            args.date = sorted(p.name for p in root.iterdir() if p.is_dir())[-1]
        root = root / args.date
        paths = {k: root / k / "rmse_vs_kse.csv" for k in ["lorenz63", "hastings_powell", "c24h50"]}

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
    plt.subplots_adjust(wspace=0.32, bottom=0.22, top=0.84)
    for ax, (key, p) in zip(axes, paths.items()):
        if not p.exists():
            ax.set_title(f"{TITLE.get(key, key)}\n(no data)"); ax.set_axis_off(); continue
        df = pd.read_csv(p)
        df = df[np.isfinite(df.rmse) & np.isfinite(df.kse_ub)]
        plot_panel(ax, df, key)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
