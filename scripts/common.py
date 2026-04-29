"""Shared utilities for the KSE-vs-RMSE experiments.

Contains: noise injection, mutual-information tau, Eckmann KSE estimator,
and the unified TIR pipeline runner used by all three systems.
"""
from __future__ import annotations
import csv, json, os, sys, time, warnings
from pathlib import Path
import numpy as np
import nolds
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

TIR_PATH = os.environ.get("TIR_PATH", "../TAR/TIR_pkg")
sys.path.insert(0, TIR_PATH)
import TIR as tir  

CSV_FIELDS = ["system", "observable", "snr_db", "trial", "tau", "dim",
              "lyap_max", "kse_ub", "lyap_spec", "rmse"]


def add_gaussian_noise(x, snr_db, rng):
    if np.isinf(snr_db):
        return x.copy()
    nse_var = np.var(x) / (10 ** (snr_db / 10))
    return x + rng.normal(0.0, np.sqrt(nse_var), size=x.shape)


def _mi(x, y, nbins=32):
    H, _, _ = np.histogram2d(x, y, bins=nbins)
    p = H / H.sum()
    px = p.sum(1, keepdims=True); py = p.sum(0, keepdims=True)
    nz = p > 0
    return float(np.sum(p[nz] * np.log(p[nz] / (px * py + 1e-12)[nz])))


def auto_tau(x, max_tau=200, nbins=32):
    """First minimum of mutual information (Fraser--Swinney)."""
    mi = [0.0] + [_mi(x[:-t], x[t:], nbins) for t in range(1, max_tau + 1)]
    for t in range(2, max_tau):
        if mi[t] < mi[t - 1] and mi[t] < mi[t + 1]:
            return t
    return 1


def estimate_kse_upper_bound(x, tau, dim):
    """Returns (lambda_max, sum_positive_lyap, full_spectrum)."""
    try:
        spec = nolds.lyap_e(x, emb_dim=dim, matrix_dim=min(4, dim - 1), min_tsep=tau)
        return float(np.max(spec)), float(np.sum(np.clip(spec, 0, None))), list(map(float, spec))
    except Exception:
        lam = float(nolds.lyap_r(x, emb_dim=dim, lag=tau))
        return lam, max(lam, 0.0), [lam]


def _seed(trial, snr):
    s = int(snr) if np.isfinite(snr) else 999
    return 1000 * trial + abs(s) + 7


def run_pipeline(name, truth, observables, snrs, *, dim=7, int_dim=3,
                 divisions=3, epochs=250, ac_maxval=500, out_root="../results"):
    """Unified driver: standardize -> auto-tau -> Takens -> dMap -> Nystrom
    -> FFNN backmap, looping over (obs, snr, trial). Writes one CSV row per
    cell with append-and-flush, so partial output survives a timeout."""
    today = time.strftime("%Y-%m-%d")
    out_dir = Path(out_root) / today / name
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path(f"/tmp/{name}_scratch"); scratch.mkdir(exist_ok=True)
    csv_path = out_dir / "rmse_vs_kse.csv"
    f = open(csv_path, "w", newline="", buffering=1)
    w = csv.DictWriter(f, fieldnames=CSV_FIELDS); w.writeheader(); f.flush()

    n_rows = 0; t0 = time.time()
    for obs_name, raw in observables.items():
        raw = np.asarray(raw, dtype=np.float64)
        for snr in snrs:
            for trial in range(divisions):
                rng = np.random.default_rng(_seed(trial, snr))
                noisy = add_gaussian_noise(raw, snr, rng)
                sc = StandardScaler().fit(noisy[:len(noisy) // 2].reshape(-1, 1))
                s = sc.transform(noisy.reshape(-1, 1)).ravel()

                tau_kse = max(1, auto_tau(s))
                lam_max, kse, spec = estimate_kse_upper_bound(s, tau_kse, dim)
                try:
                    tau = max(1, int(tir.AC(s, obs_name, dim=dim, maxval=ac_maxval,
                                            tol=10, cut=0.2, save=False, plots=False,
                                            method="cutoff")))
                except Exception:
                    tau = tau_kse

                rmse = float("nan")
                try:
                    dvecs = tir.make_dvecs(s, obs_name, tau, dim, save=False,
                                           delaytau=0, mode="single", scaling="None")
                    tp = scratch / f"{obs_name}_{int(snr) if np.isfinite(snr) else 999}_{trial}"
                    tp.mkdir(parents=True, exist_ok=True)
                    n_dv = len(dvecs)
                    TL = min(10000, n_dv // (divisions + 1))
                    slow = np.arange(0, TL); fast = slow[::4]
                    P_fast = tir.make_P(obs_name, dvecs, fast, tau, dim, save=False, issymmetric=False)
                    eps = tir.eps_scan(obs_name, tau, dim, P_fast, dvecs, fast,
                                       fit=2, method="intrinsic", i_dim=int_dim,
                                       ll=-2, ul=12, lamulim=0.85, lamlim=0.3,
                                       lamllim=0.25, mineps=-2, plots=False, save=False)
                    P_slow = tir.make_P(obs_name, dvecs, slow, tau, dim, save=False, issymmetric=False)
                    lamb, psi, eps = tir.dMap(obs_name, tau, dim, P_slow, eps, dim + 5,
                                              save=True, save_loc=str(tp), plots=False)
                    tir.nystrom(obs_name, tau, dim, lamb, psi, dvecs, slow, eps,
                                save_loc=str(tp), save=True, plots=False)
                    Y = truth[(dim - 1) * tau:]
                    _, rmse = tir.train_and_evaluate_model2(
                        Y, f"{tp}/data_{obs_name}_tau{tau}_dim{dim}.npz",
                        int_dim, truth.shape[1], trial, hidden_layers=[50] * 5,
                        epochs=epochs, learning_rate=0.01, first_feature=1,
                        divisions=divisions, save=False)
                except Exception as e:
                    print(f"  [{obs_name} SNR={snr} tr={trial}] FAILED: {e}", flush=True)

                w.writerow(dict(system=name, observable=obs_name, snr_db=float(snr),
                                trial=trial, tau=tau, dim=dim, lyap_max=lam_max, kse_ub=kse,
                                lyap_spec=";".join(f"{v:.4f}" for v in spec), rmse=rmse))
                f.flush(); n_rows += 1
                print(f"  {obs_name:18s} SNR={snr!s:>5} tr={trial} tau={tau:3d} "
                      f"KSE={kse:.3f} RMSE={rmse:.4f}", flush=True)
    f.close()
    (out_dir / "run_meta.json").write_text(json.dumps(dict(
        system=name, date=today, n_obs=len(observables),
        snrs=[float(s) if np.isfinite(s) else "inf" for s in snrs],
        dim=dim, int_dim=int_dim, divisions=divisions, epochs=epochs,
        wall_sec=round(time.time() - t0, 1), n_rows=n_rows), indent=2))
    print(f"[{name}] {n_rows} rows -> {csv_path}", flush=True)
