"""Spearman rho(KSE, RMSE) vs SNR for each system, with bootstrap 95% CIs.

Reads per-system CSVs (one or many; multiple CSVs of the same system are
concatenated). Plots ρ(SNR) for all three systems on a single symlog-x axis.

Usage:
    python bell_curves.py --csvs <lorenz CSVs> -- <hp CSVs> -- <c24h50 CSVs>
    python bell_curves.py                              # autodetect from ../results/<latest>
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
BOND_MODE_OBS = {"C1_to_C2", "C1_to_C3"}
EXCLUDE = INTEGRAL_OBS | BOND_MODE_OBS
SYSTEM_STYLE = {
    "lorenz63":        ("#1f77b4", "Lorenz-63",        "o"),
    "hastings_powell": ("#2ca02c", "Hastings--Powell", "s"),
    "c24h50":          ("#d62728", r"$\mathrm{C}_{24}\mathrm{H}_{50}$", "^"),
}


def boot_rho(sub, n_boot=300, seed=0):
    """Per-obs mean RMSE/KSE; Spearman across obs; bootstrap by resampling
    per-trial rows within each observable."""
    g = sub.groupby("observable").agg(kse=("kse_ub", "mean"), rmse=("rmse", "mean")).reset_index()
    if len(g) < 4:
        return None
    rho, _ = spearmanr(g.kse, g.rmse)
    rng = np.random.default_rng(seed)
    rhos = []
    for _ in range(n_boot):
        rows = []
        for o in g.observable:
            ot = sub[sub.observable == o]
            if not len(ot): continue
            samp = ot.sample(len(ot), replace=True, random_state=rng.integers(2**31))
            rows.append((samp.kse_ub.mean(), samp.rmse.mean()))
        if len(rows) >= 4:
            arr = np.array(rows)
            r, _ = spearmanr(arr[:, 0], arr[:, 1])
            if not np.isnan(r): rhos.append(r)
    if not rhos: return rho, rho, rho
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return rho, lo, hi


def curves(csv_paths_per_system):
    out = {}
    for sys_name, paths in csv_paths_per_system.items():
        if not paths: continue
        df = pd.concat([pd.read_csv(p) for p in paths if Path(p).exists()], ignore_index=True)
        df = df[np.isfinite(df.rmse) & np.isfinite(df.kse_ub)]
        df = df[~df.observable.isin(EXCLUDE)]
        rows = []
        for snr in sorted(df.snr_db.unique()):
            res = boot_rho(df[df.snr_db == snr], seed=int(abs(snr) if np.isfinite(snr) else 999))
            if res:
                rho, lo, hi = res
                rows.append(dict(snr_db=snr, rho=rho, ci_lo=lo, ci_hi=hi))
        out[sys_name] = pd.DataFrame(rows)
    return out


def plot(curves_dict, out_pdf, snr_floor=-35, clean_x=2000):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for sys_name, df in curves_dict.items():
        if sys_name not in SYSTEM_STYLE or df.empty: continue
        c, lab, mk = SYSTEM_STYLE[sys_name]
        finite = df[np.isfinite(df.snr_db)].sort_values("snr_db")
        clean = df[~np.isfinite(df.snr_db)]
        x, y = finite.snr_db.values, finite.rho.values
        ax.fill_between(x, finite.ci_lo, finite.ci_hi, color=c, alpha=0.18, linewidth=0)
        ax.plot(x, y, "-", color=c, lw=1.4, marker=mk, ms=4, label=lab)
        if len(clean):
            yv = clean.rho.iloc[0]
            ax.plot([clean_x], [yv], marker=mk, color=c, ms=7,
                    markerfacecolor="white", markeredgewidth=1.4)

    ax.axhline(0, color="0.4", lw=0.6, linestyle=":")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("signal-to-noise ratio (dB)")
    ax.set_ylabel(r"Spearman $\rho(h^{KS,UB}_i,\, \mathrm{RMSE}_i)$")
    ax.set_xlim(snr_floor, clean_x * 1.3)
    ax.set_ylim(-1.0, 1.05)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.4)
    ax.legend(loc="upper right", frameon=False, bbox_to_anchor=(0.98, 0.98))
    fig.tight_layout()
    Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}")


def autodetect(results_root):
    """Walk ../results/<latest>/ and group all per-system CSVs (incl. sweep
    variants like lorenz63_dense_J1) by canonical system name."""
    root = Path(results_root)
    if not root.exists(): return {}
    date = sorted(p.name for p in root.iterdir() if p.is_dir())[-1]
    found = {"lorenz63": [], "hastings_powell": [], "c24h50": []}
    for sub in (root / date).iterdir():
        if not sub.is_dir(): continue
        for canon in found:
            if sub.name.startswith(canon):
                csv = sub / "rmse_vs_kse.csv"
                if csv.exists(): found[canon].append(csv)
                break
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results")
    ap.add_argument("--out", default="../figures/bell_curves.pdf")
    ap.add_argument("--lorenz",  nargs="*", default=None, help="explicit Lorenz CSVs")
    ap.add_argument("--hp",      nargs="*", default=None, help="explicit HP CSVs")
    ap.add_argument("--c24h50",  nargs="*", default=None, help="explicit C24H50 CSVs")
    args = ap.parse_args()

    if any(x is not None for x in (args.lorenz, args.hp, args.c24h50)):
        paths = {"lorenz63": args.lorenz or [], "hastings_powell": args.hp or [],
                 "c24h50": args.c24h50 or []}
    else:
        paths = autodetect(args.results)

    cdict = curves(paths)
    for k, df in cdict.items():
        print(f"\n=== {k} ===")
        print(df.to_string(index=False))
    plot(cdict, args.out)


if __name__ == "__main__":
    main()
