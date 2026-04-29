# KSE-paper

Code to reproduce the empirical validation in *"Kolmogorov–Sinai entropies
identify optimal observables for prediction and dynamics reconstruction in
chaotic systems"* (Topel, 2026). Three systems — Lorenz-63, the
Hastings–Powell food chain, and a tetracosane (C24H50) MD trajectory — are
swept across Gaussian measurement noise; the pipeline reports per-observable
KSE upper bound and reconstruction RMSE. Figures and tables are produced
from the resulting CSVs.

## Layout

```
.
├── scripts/
│   ├── common.py             utilities + unified TAR pipeline runner
│   ├── lorenz63.py           ODE + 20 stationary observables
│   ├── hastings_powell.py    ODE + 20 stationary observables
│   ├── c24h50.py             MD trajectory + 21 chain-distance observables
│   ├── figures.py            3-panel RMSE-vs-KSE figure
│   ├── bell_curves.py        ρ(SNR) summary across systems
│   └── tables.py             per-system LaTeX tables
└── requirements.txt
```

## Dependencies

- Python 3.10+
- Packages in `requirements.txt`
- The TAR (Takens Reconstruction) package from
  https://github.com/maxtopel/TAR. Clone it adjacent and point `TIR_PATH`
  at it:

  ```
  git clone https://github.com/maxtopel/TAR
  export TIR_PATH=$(pwd)/TAR/TIR_pkg
  ```

- For C24H50, the MD trajectory from Ferguson *et al.*, PNAS 108 (2011) 13023.
  Place `md_protein_fit.xtc` somewhere readable and set `CTC_TRAJ` to its path
  (default `../data/c24h50/md_protein_fit.xtc`).

## Reproducing

Each system script writes `results/<date>/<system>/rmse_vs_kse.csv` (one row
per `(observable, SNR, trial)`, append-and-flush so partial runs survive).

```
cd scripts/
python lorenz63.py
python hastings_powell.py
python c24h50.py
```

Default SNR set per system is `{∞, 30, 20, 15, 10}` dB. Edit `SNRS` at the top
of each system script to extend the sweep. Pipeline parameters (embedding
dim, intrinsic dim, divisions, epochs) are arguments to `run_pipeline()` in
`common.py`; the system scripts pass system-appropriate values.

## Figures and tables

```
python figures.py                                    # 3-panel RMSE vs KSE
python bell_curves.py                                # ρ(SNR) summary
python tables.py                                     # LaTeX tables
```

By default each script picks the latest dated subdir of `../results/`. Each
accepts `--date YYYY-MM-DD` to pin a specific run, and `figures.py` /
`bell_curves.py` accept explicit CSV-path arguments for cross-run aggregation
(e.g. merging dense-sweep variants).

## Hardware

Reported runs were executed on Northwestern University's Quest HPC cluster
(4 CPU cores, 16–32 GB RAM per job; Intel Emerald Rapids / Skylake /
Broadwell nodes). One `(observable, SNR, trial)` cell takes 30 s – 2 min
depending on the system and embedding dimension.

## Citation

```
@article{topel2026kse,
  author = {Topel, Maximilian},
  title  = {{Kolmogorov-Sinai entropies identify optimal observables
             for prediction and dynamics reconstruction in chaotic systems}},
  year   = {2026}
}
```

The TAR pipeline used here is from Topel (2025), PhD thesis, The University
of Chicago (https://knowledge.uchicago.edu/record/14936).

## License

MIT.
