#!/bin/bash
# One-time setup: submodules, Python env, CellOT processed datasets.
set -e
cd "$(dirname "$0")/.."
git submodule update --init
uv venv --python 3.9 .venv-cellot
VIRTUAL_ENV=$PWD/.venv-cellot uv pip install -r requirements.txt --index-strategy unsafe-best-match
VIRTUAL_ENV=$PWD/.venv-cellot uv pip install -e external/cellot
# Processed datasets of Bunne et al. 2023 (ETH Research Collection, doi:10.3929/ethz-b-000609681), ~420 MB
mkdir -p data && cd data
curl -L -o processed_datasets_all.zip \
  "https://www.research-collection.ethz.ch/server/api/core/bitstreams/7c5fe615-a6fa-4464-8ae7-4482a02040db/content"
unzip -q -o processed_datasets_all.zip
