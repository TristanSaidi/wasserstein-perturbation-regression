"""Tabulate test metrics next to Bunne et al.'s reported numbers (Source Data, Extended Data Fig. 3).
Usage: aggregate.py [ncells]"""
import glob, os, pickle, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NC = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
R = [pickle.load(open(f, "rb")) for f in sorted(glob.glob(f"{ROOT}/results/eval/*.pkl"))]
ev = pd.concat([r["evals"] for r in R]); best = pd.concat([r["best"] for r in R]); sens = pd.concat([r["sens"] for r in R])
best.to_csv(f"{ROOT}/results/selected_hyperparameters.csv", index=False)
sens.to_csv(f"{ROOT}/results/sensitivity_mmd_500cells.csv", index=False)
rep = pd.read_csv(f"{ROOT}/data/bunne_reported/lupus.csv")
rep["model"] = rep.model.map({"cellot": "CellOT", "scgen": "scGen", "cae": "cAE"}) + " [reported, " + rep["mode"].map({"ood": "o.o.s.", "iid": "i.i.d."}) + "]"
rep = rep.assign(nfeatures=50, ncells=NC)[["model", "holdout", "metric", "nfeatures", "ncells", "value"]]
allr = pd.concat([rep, ev[["model", "holdout", "metric", "nfeatures", "ncells", "value"]]])
allr["nfeatures"] = allr.nfeatures.astype(str)
allr.to_csv(f"{ROOT}/results/all_results.csv", index=False)
HOLDS = sorted(ev.holdout.unique())

def table(metric, nf, holds, fmt="{:.4f}", nc=NC, sd=True):
    x = allr[(allr.metric == metric) & (allr.nfeatures == str(nf)) & allr.holdout.isin(holds) & (allr.ncells == nc)]
    m = x.groupby(["model", "holdout"]).value.agg(["mean", "std"])
    out = ((m["mean"].map(fmt.format) + " ± " + m["std"].map(fmt.format)) if sd else m["mean"].map(fmt.format)).unstack()
    mm = m["mean"].unstack()
    out["mean"] = mm.mean(1).map(fmt.format)
    out = out[mm.notna().all(1)]
    return out.loc[mm.mean(1).loc[out.index].sort_values(ascending=not metric.startswith("r2")).index]

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(f"=== MMD (top-50 marker genes, {NC} cells, mean ± sd over 10 bootstraps), held-out patients reported by Bunne et al. ===")
print(table("mmd", 50, [1488, 1015, 1256]).to_string())
print(f"\n=== MMD, all 8 held-out patients ===")
print(table("mmd", 50, HOLDS, sd=False).to_string())
print("\n=== MMD of selected models vs number of evaluation cells (mean over 1488/1015/1256) ===")
x = allr[(allr.metric == "mmd") & allr.holdout.isin([1488, 1015, 1256]) & ~allr.model.str.contains("reported")]
print(x.groupby(["model", "ncells"]).value.mean().unstack().round(4).to_string())
print("\n=== feature means over all 1000 genes, mean over 8 patients ===")
print(table("l2-means", "all", HOLDS, "{:.3f}", sd=False)[["mean"]].join(table("r2-means", "all", HOLDS, "{:.3f}", sd=False)[["mean"]], lsuffix=" l2", rsuffix=" r2").to_string())
print("\n=== selected hyper-parameters (inner leave-one-patient-out) ===")
print(best.assign(sel="d=" + best.d.astype(str) + ", h=" + best.hyper.astype(str)).pivot(index=["geom", "method"], columns="holdout", values="sel").to_string())
print("\n=== sensitivity: test MMD (500 cells) for fixed hyper-parameters, mean over all 8 patients ===")
print(sens.groupby(["geom", "method", "d", "hyper"]).mmd.mean().unstack("hyper").round(4).to_string())
