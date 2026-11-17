"""Rebuild gene-space predictions for one held-out patient from the saved fold outputs."""
import pickle
import numpy as np, pandas as pd
from wreg.data import load_split, marker_genes, ROOT


def load_predictions(K, k_markers=50):
    """Returns dict(control, truth, preds={model: DataFrame}, markers, symbols, selected).
    Rows of every prediction correspond to the held-out control cells in `control`.
    Models use the (PCA dim, hyper-parameter) chosen by inner leave-one-patient-out (results/eval)."""
    ad = load_split(K); df = ad.to_df(); obs = ad.obs
    def sel(s, split, tr):
        return df[((obs.sample_id == s) & (obs.split == split) & (obs.transport == tr)).values]
    control, truth = sel(K, "ood", "source"), sel(K, "ood", "target")
    best = pickle.load(open(f"{ROOT}/results/eval/{K}.pkl", "rb"))["best"]
    preds = {}
    for _, b in best.iterrows():
        r = pickle.load(open(f"{ROOT}/results/folds/{K}_{b.d}.pkl", "rb"))
        disp = r["disps"][(b.geom, b.method, b.hyper)] @ r["V"].T
        preds[f"{b.geom.upper()} {b.method}"] = pd.DataFrame(control.values + disp, index=control.index, columns=df.columns)
    tr = (obs.split == "train").values
    shift = df[tr & (obs.transport == "target").values].values.mean(0) - df[tr & (obs.transport == "source").values].values.mean(0)
    preds["mean-shift"] = control + shift
    preds["identity"] = control.copy()
    return dict(control=control, truth=truth, preds=preds, markers=marker_genes(ad, k_markers),
                symbols=ad.var["symbol"].astype(str).to_dict(), selected=best,
                cell_type=obs.loc[control.index, "cell_type"])
