"""Data loading that reproduces the CellOT (Bunne et al. 2023) o.o.s. lupus-patient split exactly."""
import os
import numpy as np
import anndata
from ml_collections import ConfigDict
from cellot.data.cell import read_single_anndata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H5AD = os.path.join(ROOT, "data/datasets/scrna-lupuspatients/kang-hvg.h5ad")
SAMPLES = [101, 107, 1015, 1016, 1039, 1244, 1256, 1488]


def load_split(holdout):
    """AnnData with obs['split'] in {train,test,ood,ignore} and obs['transport'] in {source,target},
    built by CellOT's own code with its configs/tasks/lupuspatients-ood.yaml settings."""
    config = ConfigDict(dict(
        data=dict(type="cell", source="ctrl", target="stim", condition="condition", path=H5AD),
        datasplit=dict(holdout=int(holdout), key="sample_id", groupby="condition",
                       name="toggle_ood", mode="ood", test_size=0.2, random_state=0),
    ))
    return read_single_anndata(config)


def marker_genes(adata, k=50):
    """Top-k marker genes for 'stim', as in cellot/scripts/evaluate.py."""
    return list(adata.varm["marker_genes-condition-rank"]["stim"].sort_values().index[:k])


class PCA:
    """Orthonormal PCA fit on training cells only."""
    def __init__(self, X, d):
        self.mean = X.mean(0)
        _, _, Vt = np.linalg.svd(X - self.mean, full_matrices=False)
        self.V = Vt[:d].T  # (genes, d)

    def enc(self, X):
        return (X - self.mean) @ self.V

    def lift(self, V):
        """Lift a displacement in PC space to gene space."""
        return V @ self.V.T
