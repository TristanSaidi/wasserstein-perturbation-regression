"""Two instantiations of the Riemannian operations on P_2(R^d) needed by the regressions.

BW   : Bures-Wasserstein. Each measure is replaced by N(mean, cov) estimated from its cells;
       log map / parallel transport are the closed-form / ODE Gaussian versions from
       WassersteinPT (src.geom.log_map_gaussian, src.pt.BrenierGaussian.parallel_transport).
WPT  : empirical measures; log map by exact OT and parallel transport by WPT^- from
       WassersteinPT (W2EuclideanTangent.parallel_transport(..., project=False)).

Both expose the same interface, so the regressions in regress.py are written once.
A tangent at a measure P is always something that `field(P, tan, pts)` can evaluate on P's cells.
"""
import contextlib, io, os, sys
import numpy as np
import scipy.linalg as sla
from scipy.spatial import cKDTree
import ot

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "external/WassersteinPT"))
import matplotlib
matplotlib.use("Agg")
from src import pt as wpt  # noqa: E402
from src.geom import log_map_gaussian  # noqa: E402


def _sqrtm(S):
    w, U = np.linalg.eigh((S + S.T) / 2)
    return (U * np.sqrt(np.clip(w, 0, None))) @ U.T


class BW:
    name = "bw"

    def __init__(self, n_steps=100, eps=1e-6):
        self.n_steps, self.eps = n_steps, eps

    # measure = (mean, cov)
    def measure(self, X):
        return X.mean(0), np.cov(X.T) + self.eps * np.eye(X.shape[1])

    def w2sq(self, P, Q):
        (m0, S0), (m1, S1) = P, Q
        r = _sqrtm(S0)
        return float(np.sum((m0 - m1) ** 2) + np.trace(S0 + S1 - 2 * _sqrtm(r @ S1 @ r)))

    # tangent = (A, a): v(x) = A (x - mean) + a
    def log(self, P, Q):
        A, a = log_map_gaussian(P[0], P[1], Q[0], Q[1])
        A = np.real(A)
        return (A + A.T) / 2, a

    def pt(self, tan, P, Q):
        """Parallel transport tan in T_P along the geodesic P -> Q (user's Gaussian PT ODE)."""
        G, g = self.log(P, Q)
        geo = wpt.BrenierGaussian(P[0], P[1], G, g)
        As, as_ = wpt.BrenierGaussian(P[0], P[1], tan[0], tan[1]).parallel_transport(geo, self.n_steps)
        return As[-1], as_[-1]

    def inner(self, t1, t2, P):
        return float(np.trace(t1[0] @ P[1] @ t2[0]) + t1[1] @ t2[1])

    def lincomb(self, coefs, tans):
        return (sum(c * t[0] for c, t in zip(coefs, tans)), sum(c * t[1] for c, t in zip(coefs, tans)))

    def field(self, P, tan, X):
        """Displacement of each cell x in X (cells of the measure P) under exp_P(tan)."""
        return (X - P[0]) @ tan[0].T + tan[1]

    def barycenter(self, Ps, X_all=None, n_iter=100, tol=1e-10):
        m = np.mean([P[0] for P in Ps], 0)
        S = np.mean([P[1] for P in Ps], 0)
        for _ in range(n_iter):
            r = _sqrtm(S)
            ri = np.linalg.inv(r)
            T = np.mean([_sqrtm(r @ P[1] @ r) for P in Ps], 0)
            Sn = ri @ T @ T @ ri
            done = np.linalg.norm(Sn - S) < tol * np.linalg.norm(S)
            S = (Sn + Sn.T) / 2
            if done:
                break
        return m, S


class WPT:
    name = "wpt"

    def __init__(self, n_bary=1000, bary_iter=15, seed=0):
        self.n_bary, self.bary_iter, self.seed = n_bary, bary_iter, seed

    # measure = (n, d) array of cells with uniform weights
    def measure(self, X):
        return np.ascontiguousarray(X, dtype=float)

    def _em(self, X):
        return wpt.EmpiricalMeasure(X, np.ones(len(X)))

    def _plan(self, P, Q):
        M = ot.dist(P, Q)  # squared euclidean
        a, b = np.full(len(P), 1 / len(P)), np.full(len(Q), 1 / len(Q))
        G = ot.emd(a, b, M, numItermax=2_000_000)
        return G, M

    def w2sq(self, P, Q):
        G, M = self._plan(P, Q)
        return float((G * M).sum())

    # tangent = (n, d) velocities on the cells of the base measure, same row order
    def log(self, P, Q):
        tan = wpt.wasserstein_logmap(self._em(P), self._em(Q))
        if tan.vels is None:  # plan splits mass (unequal sizes): barycentric projection, as in WPT
            tan = wpt.barycentric_projection_tangent(tan)
        return np.asarray(tan.vels)

    def pt(self, tan, P, Q):
        """WPT^- : WassersteinPT parallel transport without the projection step."""
        t = wpt.W2EuclideanTangent(src_measure=self._em(P), vels=tan)
        with contextlib.redirect_stdout(io.StringIO()):
            out = t.parallel_transport(self._em(P), self._em(Q), n=1, project=False)
        # result lives on Q's support in np.unique order; put it back in Q's row order
        d, idx = cKDTree(out.src_measure.locs).query(Q, k=1)
        assert d.max() < 1e-8, d.max()
        return np.asarray(out.vels)[idx]

    def inner(self, t1, t2, P):
        return float(np.mean(np.sum(t1 * t2, 1)))

    def lincomb(self, coefs, tans):
        return sum(c * t for c, t in zip(coefs, tans))

    def field(self, P, tan, X):
        assert tan.shape == X.shape
        return tan

    def barycenter(self, Ps, X_all=None):
        rng = np.random.default_rng(self.seed)
        pool = np.concatenate(Ps)
        X0 = pool[rng.choice(len(pool), min(self.n_bary, len(pool)), replace=False)]
        ws = [np.full(len(P), 1 / len(P)) for P in Ps]
        return ot.lp.free_support_barycenter(Ps, ws, X0, numItermax=self.bary_iter)
