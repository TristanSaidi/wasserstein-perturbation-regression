"""Evaluate the transport_regression_random_effects.tex models for one held-out patient (CellOT metric protocol,
identical to eval_fold.py). Hyper-parameters (PCA dim d, bandwidth / shrinkage / rank / ridge) are chosen by
the inner leave-one-patient-out MMD only.
Usage: eval_fold_tex.py <holdout> [n_train_cells]
  With n_train_cells, the original BW/WPT regressions (run_fold.py ... n_train_cells) are evaluated too,
  together with identity / mean-shift / observed baselines computed from the same subsampled training cells."""
import os, pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wreg.data import load_split, marker_genes, SAMPLES, ROOT
from wreg.evaluate import evaluate

K = int(sys.argv[1])
NTR = int(sys.argv[2]) if len(sys.argv) > 2 else None
TAG = f"_n{NTR}" if NTR else ""
DIMS = (10, 30, 50)
R = {("tex", d): pickle.load(open(f"{ROOT}/results/folds/tex_{K}_{d}{TAG}.pkl", "rb")) for d in DIMS}
if NTR:
    R.update({("orig", d): pickle.load(open(f"{ROOT}/results/folds/{K}_{d}{TAG}.pkl", "rb")) for d in DIMS})
for (src, d), r in R.items():
    r["inner"]["hyper"] = r["inner"].hyper.astype(str)
    r["disps"] = {(g, m, str(h)): v for (g, m, h), v in r["disps"].items()}
inner = pd.concat([r["inner"] for r in R.values()])
key = ["geom", "method"]
cv = inner.groupby(key + ["d", "hyper"]).mmd.mean().reset_index()
best = cv.loc[cv.groupby(key).mmd.idxmin()].assign(holdout=K, ntrain=NTR or 0)
src_of = {(g, m): src for (src, d), r in R.items() for (g, m, h) in r["disps"]}

ad = load_split(K); df = ad.to_df(); obs = ad.obs; mk = marker_genes(ad)
def sel(s, split, tr):
    return df[((obs.sample_id == s) & (obs.split == split) & (obs.transport == tr)).values]
test_c, test_t = sel(K, "ood", "source"), sel(K, "ood", "target")
def imputed(d, g, m, h):
    r = R[(src_of[(g, m)], d)]
    return pd.DataFrame(test_c.values + r["disps"][(g, m, h)] @ r["V"].T, columns=df.columns)

out = []
for _, b in best.iterrows():
    out.append(evaluate(test_t, imputed(b.d, b.geom, b.method, b.hyper), mk).assign(model=f"{b.geom.upper()} {b.method}"))
if NTR:  # baselines from the same subsampled training cells
    rng = np.random.default_rng(K)
    train = [s for s in SAMPLES if s != K]
    sub = lambda A: A[rng.choice(len(A), min(NTR, len(A)), replace=False)]
    rc = {s: sub(sel(s, "train", "source").values) for s in train}
    rt = {s: sub(sel(s, "train", "target").values) for s in train}
    shift = np.concatenate(list(rt.values())).mean(0) - np.concatenate(list(rc.values())).mean(0)
    out.append(evaluate(test_t, test_c, mk).assign(model="identity"))
    out.append(evaluate(test_t, test_c + shift, mk).assign(model="mean-shift (gene space)"))
    out.append(evaluate(test_t, sel(K, "ignore", "target"), mk).assign(model="observed (other half of treated cells)"))

sens = []
for (src, d), r in R.items():
    for (g, m, h) in r["disps"]:
        e = evaluate(test_t, imputed(d, g, m, h), mk, n_cells=(500,), n_reps=3)
        sens.append(dict(holdout=K, d=d, geom=g, method=m, hyper=h, mmd=e[e.metric == "mmd"].value.mean()))
shrink = pd.DataFrame([dict(holdout=K, d=d, kind=kind, patient=p, factor=np.mean(f))
                       for (src, d), r in R.items() if src == "tex" for kind, fs in r["shrink_fac"].items() for p, f in fs.items()])
os.makedirs(f"{ROOT}/results/eval_tex{TAG}", exist_ok=True)
pickle.dump(dict(evals=pd.concat(out).assign(holdout=K), best=best, sens=pd.DataFrame(sens), inner=inner, shrink=shrink),
            open(f"{ROOT}/results/eval_tex{TAG}/{K}.pkl", "wb"))
print("DONE", K, NTR)
