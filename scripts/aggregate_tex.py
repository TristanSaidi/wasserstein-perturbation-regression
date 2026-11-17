"""Compare the transport_regression_random_effects.tex models with every method already in the repo
(BW/WPT kernel & linear regression, baselines) and the numbers Bunne et al. reported for CellOT/scGen/cAE.
Usage: aggregate_tex.py [ncells] [n_train_cells]
  default: full training data; results/eval/*.pkl (existing methods) + results/eval_tex/*.pkl.
  n_train_cells: the few-cells-per-patient regime; results/eval_tex_n<N>/*.pkl only (contains all methods)."""
import glob, os, pickle, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NC = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
NTR = int(sys.argv[2]) if len(sys.argv) > 2 else None
TAG = f"_n{NTR}" if NTR else ""
SEC = {"frechet": "§2/6.2 direct Fréchet", "re-map": "§3 RE mean-map", "re-bary": "§3 RE barycenter",
       "re-logmap": "§6.3 RE log-mean map", "cond-map": "§4 cond. mean-map", "cond-bary": "§4 cond. barycenter",
       "lowrank": "§4 low-rank", "eb-cond-map": "§5.1 EB-pooled + cond-map", "js-cond-map": "§5.1 JS-pooled + cond-map",
       "shrink": "§5.2 shrink-for-prediction"}
FAM = {"GAUSS": "[tex] Gaussian", "OT": "[tex] OT"}
def relabel(m):
    fam, _, meth = m.partition(" ")
    return f"{FAM[fam]} {SEC[meth]}" if fam in FAM and meth in SEC else m

T = [pickle.load(open(f, "rb")) for f in sorted(glob.glob(f"{ROOT}/results/eval_tex{TAG}/*.pkl"))]
O = [] if NTR else [pickle.load(open(f, "rb")) for f in sorted(glob.glob(f"{ROOT}/results/eval/*.pkl"))]
ev = pd.concat([r["evals"] for r in T + O]); ev["model"] = ev.model.map(relabel)
best = pd.concat([r["best"] for r in T]); sens = pd.concat([r["sens"] for r in T]); shrink = pd.concat([r["shrink"] for r in T])
best.to_csv(f"{ROOT}/results/tex_selected_hyperparameters{TAG}.csv", index=False)
sens.to_csv(f"{ROOT}/results/tex_sensitivity_mmd_500cells{TAG}.csv", index=False)
allr = ev[["model", "holdout", "metric", "nfeatures", "ncells", "value"]]
if not NTR:
    rep = pd.read_csv(f"{ROOT}/data/bunne_reported/lupus.csv")
    rep["model"] = rep.model.map({"cellot": "CellOT", "scgen": "scGen", "cae": "cAE"}) + " [reported, " + rep["mode"].map({"ood": "o.o.s.", "iid": "i.i.d."}) + "]"
    allr = pd.concat([rep.assign(nfeatures=50, ncells=NC)[["model", "holdout", "metric", "nfeatures", "ncells", "value"]], allr])
allr = allr.assign(nfeatures=allr.nfeatures.astype(str))
allr.to_csv(f"{ROOT}/results/tex_all_results{TAG}.csv", index=False)
HOLDS = sorted(ev.holdout.unique())

def table(metric, nf, holds, fmt="{:.4f}", nc=NC, sd=True):
    x = allr[(allr.metric == metric) & (allr.nfeatures == str(nf)) & allr.holdout.isin(holds) & (allr.ncells == nc)]
    m = x.groupby(["model", "holdout"]).value.agg(["mean", "std"])
    out = ((m["mean"].map(fmt.format) + " ± " + m["std"].map(fmt.format)) if sd else m["mean"].map(fmt.format)).unstack()
    mm = m["mean"].unstack()
    out["mean"] = mm.mean(1).map(fmt.format)
    out = out[mm.notna().all(1)]
    return out.loc[mm.mean(1).loc[out.index].sort_values(ascending=not metric.startswith("r2")).index]

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 60)
title = f"training cells per patient and condition: {NTR}" if NTR else "full training data"
print(f"##### {title} #####\n")
print(f"=== MMD (top-50 marker genes, {NC} cells, mean ± sd over 10 bootstraps), held-out patients reported by Bunne et al. ===")
print(table("mmd", 50, [1488, 1015, 1256]).to_string())
print(f"\n=== MMD ({NC} cells), all 8 held-out patients ===")
print(table("mmd", 50, HOLDS, sd=False).to_string())
print("\n=== feature means over all 1000 genes, mean over 8 patients ===")
print(table("l2-means", "all", HOLDS, "{:.3f}", sd=False)[["mean"]].join(table("r2-means", "all", HOLDS, "{:.3f}", sd=False)[["mean"]], lsuffix=" l2", rsuffix=" r2").to_string())
print("\n=== selected hyper-parameters of the tex models (inner leave-one-patient-out); c = bandwidth, (c,lam) shrink, (r,ridge) low-rank ===")
print(best.assign(sel="d=" + best.d.astype(str) + ", " + best.hyper.astype(str)).pivot(index=["geom", "method"], columns="holdout", values="sel").to_string())
print("\n=== §5.1 shrinkage factors (1 = no shrinkage), mean over patients, by PCA dim ===")
print(shrink.groupby(["kind", "d"]).factor.agg(["mean", "min", "max"]).round(3).to_string())
print("\n=== sensitivity: test MMD (500 cells) at fixed hyper-parameters, mean over 8 patients, best and worst setting per model ===")
s = sens.groupby(["geom", "method", "d", "hyper"]).mmd.mean().reset_index()
print(s.groupby(["geom", "method"]).mmd.agg(["min", "median", "max"]).round(4).sort_values("min").to_string())
