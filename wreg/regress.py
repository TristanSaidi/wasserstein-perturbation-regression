"""Kernel and linear regression on Wasserstein space (Perturbation.pdf, Sec. 2.1 / 2.2).

Everything is expressed through a geometry object (geometry.BW or geometry.WPT).
`Problem` caches log maps, distances, transports and barycenters so that hyper-parameter grids and
the inner leave-one-patient-out loop reuse them.
"""
import numpy as np


class Problem:
    def __init__(self, geom, ctrl, stim):
        """ctrl, stim: dicts name -> (n, d) cells (stim may miss the test measure)."""
        self.g = geom
        self.X = dict(ctrl)
        self.P = {k: geom.measure(v) for k, v in ctrl.items()}
        self.Ps = {k: geom.measure(v) for k, v in stim.items()}
        self._c = {}

    def _memo(self, key, fn):
        if key not in self._c:
            self._c[key] = fn()
        return self._c[key]

    def effect(self, j):  # Y_j = log_{rho_j}(rho_j^eps)
        return self._memo(("eff", j), lambda: self.g.log(self.P[j], self.Ps[j]))

    def w2sq(self, j, k):
        return self._memo(("w2", ) + tuple(sorted((str(j), str(k)))), lambda: self.g.w2sq(self.P[j], self.P[k]))

    def effect_at(self, j, k):  # PT_{rho_j -> rho_k} Y_j
        return self._memo(("pt", j, k), lambda: self.g.pt(self.effect(j), self.P[j], self.P[k]))

    # ---------------- 2.1 kernel regression ----------------
    def kernel(self, train, k, bandwidths):
        """Returns {c: displacement of each cell of rho_k}. Gaussian kernel exp(-W2^2 / 2h^2) with
        h = c * median_j W2(rho_k, rho_j); c = inf is the unweighted average."""
        d2 = np.array([self.w2sq(j, k) for j in train])
        tans = [self.effect_at(j, k) for j in train]
        med = np.median(np.sqrt(d2))
        out = {}
        for c in bandwidths:
            w = np.ones(len(train)) if np.isinf(c) else np.exp(-(d2 - d2.min()) / (2 * (c * med) ** 2))
            w = w / w.sum()
            out[c] = self.g.field(self.P[k], self.g.lincomb(w, tans), self.X[k])
        return out

    # ---------------- 2.2 linear regression ----------------
    def pivot(self, train):
        key = ("bary", tuple(train))
        def make():
            B = self.g.barycenter([self.P[j] for j in train])
            return B
        return self._memo(key, make)

    def linear(self, train, k, lambdas, intercept=True):
        """Pivot = barycenter of training controls. X_i = log_pivot(rho_i), Ybar_i = PT_{rho_i->pivot} Y_i.
        F = argmin sum_i ||F X_i - Ybar_i||^2_{L2(pivot)} (+ ridge lam * tr(G)/n * ||F||_HS^2; n << dim of
        the tangent space, so the ridge/min-norm solution is used, in dual form).
        Prediction: exp_{rho_k}( PT_{pivot->rho_k} (F log_pivot(rho_k)) ).
        Returns {lam: displacement of each cell of rho_k}."""
        g, train = self.g, list(train)
        B = self.pivot(train)
        tk = ("piv", tuple(train))
        X = [self._memo(tk + ("X", j), lambda j=j: g.log(B, self.P[j])) for j in train]
        Y = [self._memo(tk + ("Y", j), lambda j=j: g.pt(self.effect(j), self.P[j], B)) for j in train]
        Xk = self._memo(tk + ("X", k), lambda: g.log(B, self.P[k]))
        n = len(train)
        if intercept:
            Xm, Ym = g.lincomb(np.full(n, 1 / n), X), g.lincomb(np.full(n, 1 / n), Y)
            X = [g.lincomb([1, -1], [x, Xm]) for x in X]
            Yc = [g.lincomb([1, -1], [y, Ym]) for y in Y]
            Xk = g.lincomb([1, -1], [Xk, Xm])
        else:
            Yc = Y
        G = np.array([[g.inner(a, b, B) for b in X] for a in X])
        gk = np.array([g.inner(a, Xk, B) for a in X])
        out = {}
        for lam in lambdas:
            alpha = np.linalg.solve(G + lam * np.trace(G) / n * np.eye(n), gk)
            Yhat = g.lincomb(alpha, Yc)
            if intercept:
                Yhat = g.lincomb([1, 1], [Yhat, Ym])
            out[lam] = g.field(self.P[k], g.pt(Yhat, B, self.P[k]), self.X[k])
        return out
