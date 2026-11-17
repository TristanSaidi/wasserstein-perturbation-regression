# Regression on Wasserstein space vs. CellOT (lupus patients, out-of-sample)

Implements the kernel and linear regressions of `Perturbation.pdf` in two geometries and compares
them with the numbers Bunne et al. (Nat. Methods 2023) published for their o.o.s. lupus-patient task.

## Setup

```bash
git clone --recursive <this repo> && cd <repo>
scripts/setup.sh        # submodules, .venv-cellot (Python 3.9), downloads the processed datasets
```

## Layout

- `wreg/geometry.py` — `BW` (Gaussian log map / parallel-transport ODE) and `WPT` (empirical OT log map / WPT^-)
- `wreg/regress.py` — kernel regression (Sec. 2.1) and linear regression with barycenter pivot (Sec. 2.2)
- `wreg/data.py`, `wreg/evaluate.py`, `wreg/predict.py` — CellOT's split, CellOT's metrics, prediction rebuild
- `external/WassersteinPT` (submodule) — github.com/TristanSaidi/WassersteinPT; WPT^- = `parallel_transport(project=False)`
- `external/cellot` (submodule) — github.com/bunnech/cellot; used for its split and metric code only, nothing retrained
- `data/bunne_reported/` — Bunne et al.'s Source Data (CC BY 4.0); `lupus.csv` = Extended Data Fig. 3 (MMD, patients 1488/1015/1256)
- `data/datasets/` — processed CellOT datasets (downloaded by `scripts/setup.sh`, not tracked)
- `notebooks/predictions_vs_truth.ipynb` — predicted vs. true stimulated cells for every method
- `results/` — selected hyper-parameters, all metrics (`all_results.csv`), summary tables, per-patient evals

## Reproduce

```bash
PY=.venv-cellot/bin/python
for s in 101 107 1015 1016 1039 1244 1256 1488; do for d in 10 30 50; do
  $PY scripts/run_fold.py $s $d      # -> results/folds/ (not tracked, ~86 MB)
done; done
for s in 101 107 1015 1016 1039 1244 1256 1488; do $PY scripts/eval_fold.py $s; done   # -> results/eval/
$PY scripts/aggregate.py 1000 > results/summary_ncells1000.txt
$PY scripts/make_notebook.py && $PY -m nbconvert --to notebook --execute --inplace notebooks/predictions_vs_truth.ipynb
```
Each `run_fold.py` job is single-patient / single-PCA-dim, so the 24 jobs can be spread over machines.

## Protocol

CellOT's `toggle_ood` split (train cells of the 7 other patients; test = a random half of the held-out patient),
PCA fit on training cells, displacement predicted in PC space and added to each held-out control cell,
MMD on the top-50 marker genes with CellOT's kernel bandwidths, 1000 cells x 10 bootstraps.
Hyper-parameters (PCA dim, kernel bandwidth, ridge) chosen by leave-one-patient-out inside the training patients.
The number of evaluation cells behind the reported CellOT values is not stated in the source data; 1000 is inferred.
