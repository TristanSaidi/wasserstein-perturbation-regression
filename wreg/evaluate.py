"""Metrics copied from cellot/scripts/evaluate.py (compute_evaluations) so numbers are comparable."""
import numpy as np
import pandas as pd
from cellot.losses.mmd import mmd_distance
from sklearn.metrics.pairwise import euclidean_distances

GAMMAS = np.logspace(1, -3, num=50)


def mmd(lhs, rhs):
    """Same value as np.mean([cellot.losses.mmd.mmd_distance(lhs, rhs, g) for g in GAMMAS]) (RBF-kernel
    V-statistic averaged over 50 bandwidths), but the squared distances are computed once."""
    lhs, rhs = np.asarray(lhs, dtype=np.float64), np.asarray(rhs, dtype=np.float64)
    dxx, dxy, dyy = (euclidean_distances(a, b, squared=True) for a, b in ((lhs, lhs), (lhs, rhs), (rhs, rhs)))
    return float(np.mean([np.exp(-g * dxx).mean() + np.exp(-g * dyy).mean() - 2 * np.exp(-g * dxy).mean()
                          for g in GAMMAS]))


def evaluate(treated, imputed, markers, n_cells=(100, 250, 500, 1000, 1500), n_reps=10, seed=0):
    """treated, imputed: DataFrames (cells x genes). Returns long DataFrame like CellOT's evals.csv.
    MMD on top-50 marker genes; l2/r2 of feature means on all genes (as in the paper's Methods)."""
    rs = np.random.RandomState(seed)
    rows = []
    for nc in n_cells:
        if nc > min(len(treated), len(imputed)):
            break
        for r in range(n_reps):
            trt = treated.sample(nc, random_state=rs)
            imp = imputed.sample(nc, random_state=rs)
            rows.append((nc, 50, "mmd", mmd(trt[markers].values, imp[markers].values)))
            mut, mui = trt.mean(0), imp.mean(0)
            rows.append((nc, "all", "l2-means", float(np.linalg.norm(mut - mui))))
            rows.append((nc, "all", "r2-means", float(pd.Series.corr(mut, mui))))
            mut, mui = trt[markers].mean(0), imp[markers].mean(0)
            rows.append((nc, 50, "l2-means", float(np.linalg.norm(mut - mui))))
            rows.append((nc, 50, "r2-means", float(pd.Series.corr(mut, mui))))
    return pd.DataFrame(rows, columns=["ncells", "nfeatures", "metric", "value"])
