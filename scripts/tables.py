"""Per-system LaTeX tables: observable, tau, lambda_max, KSE, RMSE.

Reads ../results/<date>/<system>/rmse_vs_kse.csv (clean SNR rows only) and
writes one table_<system>.tex file per system.
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd

CAPTIONS = {
    "lorenz63":        "Lorenz-63 reconstruction quality (clean SNR; mean$\\pm$std over K-fold trials).",
    "hastings_powell": "Hastings--Powell reconstruction quality (clean SNR).",
    "c24h50":          "$\\mathrm{C}_{24}\\mathrm{H}_{50}$ reconstruction quality (clean SNR).",
}


def chain_key(name):
    m = re.search(r"C1_to_C(\d+)", name)
    return int(m.group(1)) if m else 999


def fmt(mu, sd, d=3):
    sd = sd if np.isfinite(sd) else 0.0
    return f"${mu:.{d}f}\\pm{sd:.{d}f}$"


def make_table(csv_path, label, caption):
    df = pd.read_csv(csv_path)
    if df.snr_db.nunique() > 1:
        df = df[np.isinf(df.snr_db)]
    g = df.groupby("observable").agg(kse_mu=("kse_ub","mean"), kse_sd=("kse_ub","std"),
                                      rmse_mu=("rmse","mean"), rmse_sd=("rmse","std"),
                                      tau_mu=("tau","mean"), lam1_mu=("lyap_max","mean")).reset_index()
    sort_key = chain_key if "c24h50" in label else None
    if sort_key:
        g["_k"] = g.observable.map(sort_key)
        g = g.sort_values("_k").drop(columns="_k")
    else:
        g = g.sort_values("kse_mu")

    rows = []
    for _, r in g.iterrows():
        obs = str(r.observable).replace("_", r"\_")
        tau = int(round(r.tau_mu)) if np.isfinite(r.tau_mu) else "--"
        rows.append(f"{obs} & {tau} & ${r.lam1_mu:+.3f}$ & "
                    f"{fmt(r.kse_mu, r.kse_sd)} & {fmt(r.rmse_mu, r.rmse_sd)} \\\\")

    return "\n".join([
        r"\begin{table}[ht]", r"\centering", r"\begin{tabular}{lrccc}", r"\hline",
        r"Observable & $\tau$ & $\lambda_1$ & $h^{KS,UB}$ & RMSE \\", r"\hline",
        *rows, r"\hline", r"\end{tabular}",
        rf"\caption{{{caption}}}", rf"\label{{tab:{label}}}", r"\end{table}"
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results")
    ap.add_argument("--date", default=None)
    ap.add_argument("--out", default="../tables")
    args = ap.parse_args()

    root = Path(args.results)
    if args.date is None:
        args.date = sorted(p.name for p in root.iterdir() if p.is_dir())[-1]
    root = root / args.date
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    for label, caption in CAPTIONS.items():
        csv = root / label / "rmse_vs_kse.csv"
        if not csv.exists():
            print(f"skip: {csv}"); continue
        (out / f"table_{label}.tex").write_text(make_table(csv, label, caption) + "\n")
        print(f"wrote {out / f'table_{label}.tex'}")


if __name__ == "__main__":
    main()
