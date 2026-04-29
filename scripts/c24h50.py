"""C24H50 (tetracosane): 21 chain-distance observables (C1--Cn for n=4..24).

The shortest distances n=2,3 (bond length and 1--3 distance) are excluded:
they probe sub-picosecond bond-stretching/angle-bending vibrations rather than
the slow chain reorganizations the rest of the observables capture, and the
Eckmann--Ruelle KSE estimator returns noise-floor values for them.

Reads a pre-aligned MD trajectory in xtc format. Path configurable via the
CTC_TRAJ environment variable; default ../data/c24h50/md_protein_fit.xtc.
"""
from __future__ import annotations
import os
import numpy as np
import mdtraj as md
from common import run_pipeline

TRAJ = os.environ.get("CTC_TRAJ", "../data/c24h50/md_protein_fit.xtc")
N_CARBONS = 24
STRIDE = 25
SNRS = (np.inf, 30, 20, 15, 10)


def carbon_topology(n=N_CARBONS):
    top = md.Topology(); chain = top.add_chain(); res = top.add_residue("TCS", chain)
    for i in range(n):
        top.add_atom(f"C{i + 1}", md.element.carbon, res)
    return top


def main():
    t = md.load(TRAJ, top=carbon_topology(), stride=STRIDE)
    xyz = t.xyz
    truth = xyz.reshape(xyz.shape[0], -1)
    obs = {f"C1_to_C{n}": np.linalg.norm(xyz[:, 0] - xyz[:, n - 1], axis=1)
           for n in range(4, 25)}
    run_pipeline("c24h50", truth, obs, SNRS, dim=5, int_dim=2, ac_maxval=1000)


if __name__ == "__main__":
    main()
