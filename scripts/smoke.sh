#!/usr/bin/env bash
set -euo pipefail
python -m corecpu_lab selftest
python -m pytest -q
for config in A_baseline B_corecpu C_new_compute D_full_new; do
  python -m corecpu_lab train --config "configs/${config}.yaml" --steps 2 --batch 4 --eval-batches 1 --out "results/smoke_${config}_$(date +%s)"
done
for config in A_baseline B_corecpu C_new_compute D_full_new; do
  python -m corecpu_lab train --config "configs/${config}_language.yaml" --steps 2 --batch 2 --eval-batches 1 --out "results/smoke_lm_${config}_$(date +%s)"
done
