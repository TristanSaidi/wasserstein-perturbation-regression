"""One held-out patient x one PCA dimension: run kernel / linear regression in BW and WPT^- geometry.

Protocol = CellOT o.o.s. lupus-patient task (configs/tasks/lupuspatients-ood.yaml):
  train on the 'train' cells of the 7 other patients, predict stim cells of the held-out patient from its
  'ood' control cells, compare to its 'ood' stim cells with cellot/scripts/evaluate.py metrics.
Hyper-parameters are scored by leave-one-patient-out *inside the 7 training patients* (no test cells).

Usage: run_fold.py <holdout> <pca_dim>
"""
import sys, os, time, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wreg.data import load_split, marker_genes, PCA, SAMPLES, ROOT
from wreg.geometry import BW, WPT
from wreg.regress import Problem
from wreg.evaluate import mmd

K, d = int(sys.argv[1]), int(sys.argv[2])
BANDW = [0.1, 0.25, 0.5, 1.0, 2.0, np.inf]
LAMS = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
t0 = time.time()

ad = load_split(K); df = ad.to_df(); obs = ad.obs
mk = marker_genes(ad)
mk_idx = [df.columns.get_loc(m) for m in mk]
def sel(s, split, tr):
    return df[((obs.sample_id == s) & (obs.split == split) & (obs.transport == tr)).values]
train = [s for s in SAMPLES if s != K]
pca = PCA(df[(obs.split == "train").values].values, d)
raw_c = {s: sel(s, "train", "source").values for s in train}
raw_t = {s: sel(s, "train", "target").values for s in train}
test_c, test_t = sel(K, "ood", "source"), sel(K, "ood", "target")
raw_c["test"] = test_c.values
ctrl = {s: pca.enc(v) for s, v in raw_c.items()}
stim = {s: pca.enc(v) for s, v in raw_t.items()}

def all_methods(pr, tr, k):
    """{(method, hyper): displacement field on cells of k}"""
    out = {}
    for c, v in pr.kernel(tr, k, BANDW).items(): out[("kernel", c)] = v
    for l, v in pr.linear(tr, k, LAMS).items(): out[("linear", l)] = v
    for l, v in pr.linear(tr, k, LAMS, intercept=False).items(): out[("linear-nointercept", l)] = v
    return out

def inner_score(k, disp, reps=2, nc=500):
    """MMD (top-50 markers) between predicted and true stim cells of training patient k."""
    pred = raw_c[k][:, mk_idx] + pca.lift(disp)[:, mk_idx]
    true = raw_t[k][:, mk_idx]
    rs = np.random.RandomState(0); n = min(nc, len(pred), len(true))
    return float(np.mean([mmd(true[rs.choice(len(true), n, replace=False)],
                              pred[rs.choice(len(pred), n, replace=False)]) for _ in range(reps)]))

inner, disps = [], {}
for geom in [BW(), WPT()]:
    pr = Problem(geom, ctrl, stim)
    for k in train:  # inner leave-one-patient-out
        for (m, h), disp in all_methods(pr, [s for s in train if s != k], k).items():
            inner.append(dict(holdout=K, d=d, geom=geom.name, method=m, hyper=h, inner_k=k, mmd=inner_score(k, disp)))
    print(geom.name, "inner CV done", round(time.time() - t0), flush=True)
    for (m, h), disp in all_methods(pr, train, "test").items():
        disps[(geom.name, m, h)] = disp.astype(np.float32)  # PC-space displacement of each test control cell
    print(geom.name, "test done", round(time.time() - t0), flush=True)

pickle.dump(dict(inner=pd.DataFrame(inner), disps=disps, V=pca.V, holdout=K, d=d),
            open(os.path.join(ROOT, f"results/folds/{K}_{d}.pkl"), "wb"))
print("DONE", K, d, round(time.time() - t0))
