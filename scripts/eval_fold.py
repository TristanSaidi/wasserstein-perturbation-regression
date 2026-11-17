"""Evaluate one held-out patient with CellOT's metric protocol.
Hyper-parameters (PCA dim d, bandwidth / ridge) are chosen by the inner leave-one-patient-out MMD only.
Usage: eval_fold.py <holdout>"""
import glob, os, pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wreg.data import load_split, marker_genes, SAMPLES, ROOT
from wreg.evaluate import evaluate

K = int(sys.argv[1])
R = {d: pickle.load(open(f"{ROOT}/results/folds/{K}_{d}.pkl", "rb")) for d in (10, 30, 50)}
inner = pd.concat([r["inner"] for r in R.values()])
key = ["geom", "method"]
cv = inner.groupby(key + ["d", "hyper"]).mmd.mean().reset_index()
best = cv.loc[cv.groupby(key).mmd.idxmin()].assign(holdout=K)

ad = load_split(K); df = ad.to_df(); obs = ad.obs; mk = marker_genes(ad)
def sel(s, split, tr):
    return df[((obs.sample_id == s) & (obs.split == split) & (obs.transport == tr)).values]
test_c, test_t = sel(K, "ood", "source"), sel(K, "ood", "target")
def imputed(d, g, m, h):
    return pd.DataFrame(test_c.values + R[d]["disps"][(g, m, h)] @ R[d]["V"].T, columns=df.columns)

out = []
for _, b in best.iterrows():  # selected models: full protocol (10 bootstraps, all cell counts)
    out.append(evaluate(test_t, imputed(b.d, b.geom, b.method, b.hyper), mk).assign(model=f"{b.geom.upper()} {b.method}"))
tr = (obs.split == "train").values
shift = df[tr & (obs.transport == "target").values].values.mean(0) - df[tr & (obs.transport == "source").values].values.mean(0)
out.append(evaluate(test_t, test_c, mk).assign(model="identity"))
out.append(evaluate(test_t, test_c + shift, mk).assign(model="mean-shift (gene space)"))
out.append(evaluate(test_t, sel(K, "ignore", "target"), mk).assign(model="observed (other half of treated cells)"))

sens = []  # every fixed (d, hyper): 3 bootstraps at 500 cells, MMD only
for d, r in R.items():
    for (g, m, h) in r["disps"]:
        e = evaluate(test_t, imputed(d, g, m, h), mk, n_cells=(500,), n_reps=3)
        sens.append(dict(holdout=K, d=d, geom=g, method=m, hyper=h, mmd=e[e.metric == "mmd"].value.mean()))
pickle.dump(dict(evals=pd.concat(out).assign(holdout=K), best=best, sens=pd.DataFrame(sens), inner=inner),
            open(f"{ROOT}/results/eval/{K}.pkl", "wb"))
print("DONE", K)
