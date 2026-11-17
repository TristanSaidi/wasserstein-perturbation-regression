"""Models of `transport_regression_random_effects.tex` ("Predicting distributions from distributions:
transport regression, random effects, shrinkage and Bures-Wasserstein approximations").

Every model returns, for a target measure k with cells X_k (PC coordinates), the displacement
T(x) - x of each cell, so it plugs into the same evaluation as the regressions in regress.py
(displacement lifted to gene space and added to the control cell).

Two map families
----------------
GaussMaps  (Sec. 6, "G-*"): each pair is replaced by its Gaussian surrogates G(P_i), G(Q_i); the per-pair
           map is the affine Gaussian OT map T_i(x) = b_i + A_i x, A_i symmetric positive definite.
           Common domain = R^d (PC space), so maps are globally defined.
OTMaps     (Secs. 2-5, "OT-*"): per-pair map = barycentric projection of the exact empirical OT plan
           between the control and stimulated cells of the pair, defined on supp(P_i) only. It is
           extended to a common domain by k-nearest-neighbour averaging of the displacement
           (the "specified extension" the note requires); this extension is not guaranteed to stay a
           convex gradient.

Section -> method name (suffix of the model label)
---------------------------------------------------
Sec. 2   direct Wasserstein (local Frechet) regression ............ frechet        (hyper: bandwidth c)
Sec. 3   random effects, mean-map pushforward  Tbar_# P ........... re-map
         random effects, barycenter of T_i# P .................... re-bary
         (Gaussian only) log-Euclidean mean of A_i ............... re-logmap
Sec. 4   conditional map regression  M(P,.) = sum_i w_i(P) T_i .... cond-map       (c)
         conditional barycenter of T_i# P with weights w_i(P) ..... cond-bary      (c)
         low-rank effects + regression of scores on P ........... lowrank        (rank r, ridge)
           (Gaussian: PSD-projected; OT: unconstrained)
Sec. 5.1 empirical-Bayes partial pooling of maps, then cond-map .. eb-cond-map    (c)   (Gaussian only)
         positive-part James-Stein interpolation, then cond-map .. js-cond-map    (c)   (Gaussian only)
Sec. 5.2 shrinkage for prediction (1-lam) T0 + lam M(P,.) ......... shrink        (c, lam)
Kernel weights are exp(-(d^2 - min d^2) / 2 (c med d)^2), c = inf -> uniform, exactly as in regress.py;
d = Gaussian (Bures-Wasserstein) distance for GaussMaps, exact W2 between control clouds for OTMaps.
"""
import zlib
import numpy as np
import ot
from scipy.spatial import cKDTree


# ------------------------------------------------------------------ helpers
def _sym(S):
    return (S + S.T) / 2


def _eig_fn(S, f):
    w, U = np.linalg.eigh(_sym(S))
    return (U * f(w)) @ U.T


def sqrtm(S):
    return _eig_fn(S, lambda w: np.sqrt(np.clip(w, 0, None)))


def psd_project(S, floor=1e-6):
    """Frobenius projection onto {S symmetric, S >= floor I} (closed convex cone, Sec. 4 remedy)."""
    return _eig_fn(S, lambda w: np.clip(w, floor, None))


def bw2(m0, S0, m1, S1):
    """Squared Gaussian W2 = ||m0 - m1||^2 + d_B^2(S0, S1)."""
    r = sqrtm(S0)
    return float(np.sum((m0 - m1) ** 2) + np.trace(S0) + np.trace(S1) - 2 * np.trace(sqrtm(r @ S1 @ r)))


def gauss_ot(m0, S0, m1, S1):
    """Affine OT map x -> b + A x from N(m0,S0) to N(m1,S1)."""
    r = sqrtm(S0)
    ri = np.linalg.inv(r)
    A = _sym(ri @ sqrtm(r @ S1 @ r) @ ri)
    return A, m1 - A @ m0


def bw_barycenter(ms, Ss, w, n_iter=200, tol=1e-10):
    """Weighted Bures-Wasserstein barycenter (fixed point of Alvarez-Esteban et al. 2016)."""
    w = np.asarray(w, float) / np.sum(w)
    m = np.einsum("i,ij->j", w, np.asarray(ms))
    S = np.einsum("i,ijk->jk", w, np.asarray(Ss))
    for _ in range(n_iter):
        r = sqrtm(S)
        ri = np.linalg.inv(r)
        T = sum(wi * sqrtm(r @ Si @ r) for wi, Si in zip(w, Ss) if wi > 0)
        Sn = _sym(ri @ T @ T @ ri)
        done = np.linalg.norm(Sn - S) < tol * np.linalg.norm(S)
        S = Sn
        if done:
            break
    return m, S


def kernel_weights(d2, bandwidths):
    """{c: normalized weights}; same kernel as regress.Problem.kernel."""
    d2 = np.asarray(d2, float)
    med = np.median(np.sqrt(d2))
    out = {}
    for c in bandwidths:
        w = np.ones(len(d2)) if np.isinf(c) else np.exp(-(d2 - d2.min()) / (2 * (c * med) ** 2))
        out[c] = w / w.sum()
    return out


def ridge_scores(Z_train, Z_new, xi, lam):
    """Linear ridge (dual form, ridge scaled by tr(K)/n as in regress.linear) of scores xi (n x r) on
    covariates Z (centered by the training mean). lam = inf -> predict the mean score (0)."""
    if np.isinf(lam):
        return np.zeros(xi.shape[1])
    mu = Z_train.mean(0)
    Zc, zk = Z_train - mu, Z_new - mu
    K = Zc @ Zc.T
    n = len(Zc)
    alpha = np.linalg.solve(K + lam * np.trace(K) / n * np.eye(n), xi)
    return (Zc @ zk) @ alpha


def covariate(m, S):
    """Euclidean feature vector of a measure for the low-rank score regression g_k(P):
    (mean, upper triangle of the covariance square root)."""
    iu = np.triu_indices(len(m))
    return np.concatenate([m, sqrtm(S)[iu]])


# ------------------------------------------------------------------ Gaussian / affine family
class GaussMaps:
    name = "gauss"

    def __init__(self, ctrl, stim, eps=1e-6, n_boot=20, seed=0):
        """ctrl, stim: dict name -> (n, d) PC coordinates. stim may lack the test measure."""
        self.X = dict(ctrl)
        d = next(iter(ctrl.values())).shape[1]
        mom = lambda Y: (Y.mean(0), np.cov(Y.T) + eps * np.eye(d))
        self.P = {k: mom(v) for k, v in ctrl.items()}
        self.Q = {k: mom(v) for k, v in stim.items()}
        self.maps = {k: gauss_ot(*self.P[k], *self.Q[k]) for k in stim}  # k -> (A_k, b_k)
        self.eps, self.n_boot, self.seed = eps, n_boot, seed
        self.stim = stim
        self._d2, self._boot = {}, {}

    # distances d_G(P_k, P_j) between control surrogates
    def d2(self, j, k):
        key = tuple(sorted((str(j), str(k))))
        if key not in self._d2:
            self._d2[key] = bw2(*self.P[j], *self.P[k])
        return self._d2[key]

    def _disp(self, A, b, k):
        X = self.X[k]
        return X @ A.T + b - X

    def _push_to(self, m, S, k):
        """Displacement of cells of k under the Gaussian OT map G(P_k) -> N(m, S)."""
        A, b = gauss_ot(*self.P[k], m, S)
        return self._disp(A, b, k)

    # ---- Sec. 2 / 6.2: direct (Gaussian) Frechet regression
    def frechet(self, train, k, bandwidths):
        W = kernel_weights([self.d2(j, k) for j in train], bandwidths)
        out = {}
        for c, w in W.items():
            m, S = bw_barycenter([self.Q[j][0] for j in train], [self.Q[j][1] for j in train], w)
            out[c] = self._push_to(m, S, k)
        return out

    # ---- Sec. 3 / 4: (conditional) mean-map and (conditional) barycenter of pushforwards
    def cond_map(self, train, k, bandwidths, maps=None):
        maps = maps or self.maps
        W = kernel_weights([self.d2(j, k) for j in train], bandwidths)
        out = {}
        for c, w in W.items():  # nonnegative weights: convex combination of SPD A_j stays SPD
            A = sum(wi * maps[j][0] for wi, j in zip(w, train))
            b = sum(wi * maps[j][1] for wi, j in zip(w, train))
            out[c] = self._disp(A, b, k)
        return out

    def cond_bary(self, train, k, bandwidths):
        mk, Sk = self.P[k]
        W = kernel_weights([self.d2(j, k) for j in train], bandwidths)
        ms = [self.maps[j][1] + self.maps[j][0] @ mk for j in train]
        Ss = [self.maps[j][0] @ Sk @ self.maps[j][0] for j in train]
        return {c: self._push_to(*bw_barycenter(ms, Ss, w), k) for c, w in W.items()}

    def re_logmap(self, train, k):
        """Sec. 6.3: shrink/average A in matrix-log coordinates, then exponentiate."""
        L = np.mean([_eig_fn(self.maps[j][0], lambda w: np.log(np.clip(w, 1e-12, None))) for j in train], 0)
        A = _eig_fn(L, np.exp)
        b = np.mean([self.maps[j][1] for j in train], 0)
        return self._disp(A, b, k)

    # ---- Sec. 4: low-rank effects, scores regressed on P, PSD projection
    @staticmethod
    def _theta(A, b):
        iu = np.triu_indices(len(b))
        return np.concatenate([b, A[iu]])

    @staticmethod
    def _untheta(t, d):
        b, A = t[:d], np.zeros((d, d))
        A[np.triu_indices(d)] = t[d:]
        return A + np.triu(A, 1).T, b

    def lowrank(self, train, k, ranks, ridges):
        d = len(self.P[k][0])
        Th = np.array([self._theta(*self.maps[j]) for j in train])
        T0 = Th.mean(0)
        U, s, Vt = np.linalg.svd(Th - T0, full_matrices=False)
        Z = np.array([covariate(*self.P[j]) for j in train])
        zk = covariate(*self.P[k])
        out = {}
        for r in ranks:
            xi = U[:, :r] * s[:r]  # scores xi_{ik}
            for lam in ridges:
                A, b = self._untheta(T0 + ridge_scores(Z, zk, xi, lam) @ Vt[:r], d)
                out[(r, lam)] = self._disp(psd_project(A), b, k)
        return out

    # ---- Sec. 5.1: shrinkage of individual maps toward the common map
    def boot_var(self, j):
        """Within-pair sampling variance of theta_j = (b_j, vech A_j): diagonal, by bootstrap over cells."""
        if j not in self._boot:
            rng = np.random.default_rng(self.seed + zlib.crc32(str(j).encode()) % 10_000)
            Xc, Xs = self.X[j], self.stim[j]
            d = Xc.shape[1]
            mom = lambda Y: (Y.mean(0), np.cov(Y.T) + self.eps * np.eye(d))
            th = [self._theta(*gauss_ot(*mom(Xc[rng.integers(0, len(Xc), len(Xc))]),
                                        *mom(Xs[rng.integers(0, len(Xs), len(Xs))])))
                  for _ in range(self.n_boot)]
            self._boot[j] = np.var(th, axis=0, ddof=1)
        return self._boot[j]

    def shrunk_maps(self, train, kind):
        """kind='eb': coordinatewise Gaussian random-effects posterior mean
              theta~_ij = mu_j + Omega_j / (Omega_j + V_ij) (theta^_ij - mu_j), mu, Omega by moments
              (diagonal working model), then PSD projection of A.
           kind='js': positive-part James-Stein factor a_i in [0,1] toward mu with sigma_i^2 = mean_j V_ij;
              the convex interpolation (1-a_i) T0 + a_i T_i keeps A SPD (no projection needed).
        Returns ({j: (A, b)}, shrinkage factors)."""
        d = len(self.maps[train[0]][1])
        Th = np.array([self._theta(*self.maps[j]) for j in train])
        V = np.array([self.boot_var(j) for j in train])
        mu = Th.mean(0)
        if kind == "eb":
            Om = np.clip(Th.var(0, ddof=1) - V.mean(0), 0, None)
            fac = Om / (Om + V + 1e-300)
            Tt = mu + fac * (Th - mu)
            maps = {}
            for j, t in zip(train, Tt):
                A, b = self._untheta(t, d)
                maps[j] = (psd_project(A), b)
            return maps, fac.mean(1)
        p = Th.shape[1]
        a = np.clip(1 - (p - 2) * V.mean(1) / np.sum((Th - mu) ** 2, 1), 0, 1)
        A0, b0 = self._untheta(mu, d)
        maps = {j: ((1 - ai) * A0 + ai * self.maps[j][0], (1 - ai) * b0 + ai * self.maps[j][1])
                for j, ai in zip(train, a)}
        return maps, a

    # ---- everything for one (train, k)
    def all_methods(self, train, k, bandwidths, lams, ranks, ridges):
        out = {}
        for c, v in self.frechet(train, k, bandwidths).items():
            out[("frechet", c)] = v
        cm = self.cond_map(train, k, bandwidths)
        for c, v in cm.items():
            out[("cond-map", c)] = v
        out[("re-map", None)] = cm[np.inf]
        cb = self.cond_bary(train, k, bandwidths)
        for c, v in cb.items():
            out[("cond-bary", c)] = v
        out[("re-bary", None)] = cb[np.inf]
        out[("re-logmap", None)] = self.re_logmap(train, k)
        for h, v in self.lowrank(train, k, ranks, ridges).items():
            out[("lowrank", h)] = v
        for kind in ("eb", "js"):
            maps, _ = self.shrunk_maps(train, kind)
            for c, v in self.cond_map(train, k, bandwidths, maps).items():
                out[(f"{kind}-cond-map", c)] = v
        for c, v in cm.items():
            for lam in lams:  # displacement is linear in the map
                out[("shrink", (c, lam))] = (1 - lam) * cm[np.inf] + lam * v
        return out


# ------------------------------------------------------------------ empirical OT family
def _emd(X, Y, numItermax=2_000_000):
    a, b = np.full(len(X), 1 / len(X)), np.full(len(Y), 1 / len(Y))
    M = ot.dist(X, Y)
    return ot.emd(a, b, M, numItermax=numItermax), M


def barycentric_disp(X, Y, b=None):
    """Barycentric projection of the exact OT plan from uniform X to Y (weights b): T(x) - x on X."""
    a = np.full(len(X), 1 / len(X))
    b = np.full(len(Y), 1 / len(Y)) if b is None else b
    G = ot.emd(a, b, ot.dist(X, Y), numItermax=2_000_000)
    return (G @ Y) / G.sum(1, keepdims=True) - X


class OTMaps:
    name = "ot"

    def __init__(self, ctrl, stim, knn=10, n_sub=500, bary_iter=10, seed=0):
        self.X, self.stim = dict(ctrl), stim
        self.knn, self.n_sub, self.bary_iter, self.seed = knn, n_sub, bary_iter, seed
        self._c = {}

    def _memo(self, key, fn):
        if key not in self._c:
            self._c[key] = fn()
        return self._c[key]

    def _sub(self, Y, tag):
        rng = np.random.default_rng(self.seed + zlib.crc32(tag.encode()) % 10_000)
        return Y if len(Y) <= self.n_sub else Y[rng.choice(len(Y), self.n_sub, replace=False)]

    def w2sq(self, j, k):
        key = ("w2",) + tuple(sorted((str(j), str(k))))
        return self._memo(key, lambda: float((lambda G, M: (G * M).sum())(*_emd(self.X[j], self.X[k]))))

    def D(self, j):
        """Per-pair map T_j - id on supp(P_j): barycentric projection of the exact OT plan."""
        return self._memo(("D", j), lambda: barycentric_disp(self.X[j], self.stim[j]))

    def ext(self, j, k):
        """T_j - id extended to the cells of k by kNN averaging over P_j's cells (common-domain extension)."""
        def f():
            _, idx = cKDTree(self.X[j]).query(self.X[k], k=self.knn)
            return self.D(j)[idx.reshape(len(self.X[k]), -1)].mean(1)
        return self._memo(("ext", j, k), f)

    def _bary_disp(self, clouds, w, k):
        """Weighted free-support barycenter of `clouds`, then exact OT from the cells of k onto it."""
        w = np.asarray(w, float)
        keep = w > 1e-8
        clouds = [c for c, kp in zip(clouds, keep) if kp]
        w = w[keep] / w[keep].sum()
        if len(clouds) == 1:
            B = clouds[0]
        else:
            X0 = clouds[int(np.argmax(w))]
            B = ot.lp.free_support_barycenter(clouds, [np.full(len(c), 1 / len(c)) for c in clouds], X0,
                                              weights=w, numItermax=self.bary_iter)
        return barycentric_disp(self.X[k], B)

    # ---- Sec. 2: direct Wasserstein (local Frechet) regression, weighted barycenter of the responses Q_j
    def frechet(self, train, k, bandwidths):
        W = kernel_weights([self.w2sq(j, k) for j in train], bandwidths)
        clouds = [self._sub(self.stim[j], f"q{j}") for j in train]
        return {c: self._bary_disp(clouds, w, k) for c, w in W.items()}

    # ---- Sec. 3 / 4
    def cond_map(self, train, k, bandwidths):
        W = kernel_weights([self.w2sq(j, k) for j in train], bandwidths)
        E = [self.ext(j, k) for j in train]
        return {c: sum(wi * e for wi, e in zip(w, E)) for c, w in W.items()}

    def cond_bary(self, train, k, bandwidths):
        W = kernel_weights([self.w2sq(j, k) for j in train], bandwidths)
        Xk = self.X[k]
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(Xk), min(self.n_sub, len(Xk)), replace=False)
        clouds = [Xk[idx] + self.ext(j, k)[idx] for j in train]  # T_j # P_k
        return {c: self._bary_disp(clouds, w, k) for c, w in W.items()}

    # ---- Sec. 4: low-rank effects on the common domain supp(P_k), no convexity constraint
    def lowrank(self, train, k, ranks, ridges, gauss):
        F = np.array([self.ext(j, k).ravel() for j in train])
        F0 = F.mean(0)
        U, s, Vt = np.linalg.svd(F - F0, full_matrices=False)
        Z = np.array([covariate(*gauss.P[j]) for j in train])
        zk = covariate(*gauss.P[k])
        out = {}
        for r in ranks:
            xi = U[:, :r] * s[:r]
            for lam in ridges:
                out[(r, lam)] = (F0 + ridge_scores(Z, zk, xi, lam) @ Vt[:r]).reshape(self.X[k].shape)
        return out

    def all_methods(self, train, k, bandwidths, lams, ranks, ridges, gauss):
        out = {}
        for c, v in self.frechet(train, k, bandwidths).items():
            out[("frechet", c)] = v
        cm = self.cond_map(train, k, bandwidths)
        for c, v in cm.items():
            out[("cond-map", c)] = v
        out[("re-map", None)] = cm[np.inf]
        cb = self.cond_bary(train, k, bandwidths)
        for c, v in cb.items():
            out[("cond-bary", c)] = v
        out[("re-bary", None)] = cb[np.inf]
        for h, v in self.lowrank(train, k, ranks, ridges, gauss).items():
            out[("lowrank", h)] = v
        for c, v in cm.items():
            for lam in lams:
                out[("shrink", (c, lam))] = (1 - lam) * cm[np.inf] + lam * v
        return out
