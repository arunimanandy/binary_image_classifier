#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/raw
kaggle datasets download -d bhavikjikadara/dog-and-cat-classification-dataset -p data/raw --unzip
find data/raw -maxdepth 3 -type d | sort
