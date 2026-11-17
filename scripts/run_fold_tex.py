"""One held-out patient x one PCA dimension: models of transport_regression_random_effects.tex (wreg/tex.py).

Same protocol as run_fold.py (CellOT o.o.s. split, PCA fit on training cells, displacement in PC space,
hyper-parameters scored by leave-one-patient-out inside the 7 training patients, MMD on top-50 markers).

Usage: run_fold_tex.py <holdout> <pca_dim> [n_train_cells]
  n_train_cells (optional): subsample every training patient to this many control and this many
  stimulated cells (regime (b) of the note: many pairs, few cells per pair). output tag "_n<N>".
"""
import sys, os, time, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wreg.data import load_split, marker_genes, PCA, SAMPLES, ROOT
from wreg.tex import GaussMaps, OTMaps
from wreg.evaluate import mmd

K, d = int(sys.argv[1]), int(sys.argv[2])
NTR = int(sys.argv[3]) if len(sys.argv) > 3 else None
TAG = f"_n{NTR}" if NTR else ""
BANDW = [0.1, 0.25, 0.5, 1.0, 2.0, np.inf]
LAMS = [0.0, 0.25, 0.5, 0.75]          # shrink: lam = 1 is cond-map itself, lam = 0 is re-map
RANKS = [1, 2, 3]
RIDGES = [1e-2, 1e-1, 1.0, 10.0]
t0 = time.time()

ad = load_split(K); df = ad.to_df(); obs = ad.obs
mk = marker_genes(ad)
mk_idx = [df.columns.get_loc(m) for m in mk]
def sel(s, split, tr):
    return df[((obs.sample_id == s) & (obs.split == split) & (obs.transport == tr)).values]
train = [s for s in SAMPLES if s != K]
raw_c = {s: sel(s, "train", "source").values for s in train}
raw_t = {s: sel(s, "train", "target").values for s in train}
if NTR:
    rng = np.random.default_rng(K)
    sub = lambda A: A[rng.choice(len(A), min(NTR, len(A)), replace=False)]
    raw_c = {s: sub(v) for s, v in raw_c.items()}
    raw_t = {s: sub(v) for s, v in raw_t.items()}
    pca = PCA(np.concatenate(list(raw_c.values()) + list(raw_t.values())), d)
else:
    pca = PCA(df[(obs.split == "train").values].values, d)
test_c = sel(K, "ood", "source")
raw_c["test"] = test_c.values
ctrl = {s: pca.enc(v) for s, v in raw_c.items()}
stim = {s: pca.enc(v) for s, v in raw_t.items()}

def inner_score(k, disp, reps=2, nc=500):
    pred = raw_c[k][:, mk_idx] + pca.lift(disp)[:, mk_idx]
    true = raw_t[k][:, mk_idx]
    rs = np.random.RandomState(0); n = min(nc, len(pred), len(true))
    return float(np.mean([mmd(true[rs.choice(len(true), n, replace=False)],
                              pred[rs.choice(len(pred), n, replace=False)]) for _ in range(reps)]))

def hstr(h):
    if h is None: return "-"
    if isinstance(h, tuple): return ",".join(str(x) for x in h)
    return str(h)

G = GaussMaps(ctrl, stim)
O = OTMaps(ctrl, stim)
def all_methods(tr, k):
    out = {("gauss",) + m: v for m, v in G.all_methods(tr, k, BANDW, LAMS, RANKS, RIDGES).items()}
    out.update({("ot",) + m: v for m, v in O.all_methods(tr, k, BANDW, LAMS, RANKS, RIDGES, G).items()})
    return out

inner, disps = [], {}
for k in train:
    for (g, m, h), disp in all_methods([s for s in train if s != k], k).items():
        inner.append(dict(holdout=K, d=d, geom=g, method=m, hyper=hstr(h), inner_k=k, mmd=inner_score(k, disp)))
    print("inner", k, round(time.time() - t0), flush=True)
for (g, m, h), disp in all_methods(train, "test").items():
    disps[(g, m, hstr(h))] = disp.astype(np.float32)
shrink_fac = {kind: dict(zip(train, G.shrunk_maps(train, kind)[1])) for kind in ("eb", "js")}
os.makedirs(os.path.join(ROOT, "results/folds"), exist_ok=True)
pickle.dump(dict(inner=pd.DataFrame(inner), disps=disps, V=pca.V, holdout=K, d=d, ntrain=NTR, shrink_fac=shrink_fac),
            open(os.path.join(ROOT, f"results/folds/tex_{K}_{d}{TAG}.pkl"), "wb"))
print("DONE", K, d, NTR, round(time.time() - t0))
