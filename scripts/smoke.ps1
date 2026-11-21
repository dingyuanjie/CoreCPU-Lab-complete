$ErrorActionPreference = 'Stop'
python -m corecpu_lab selftest
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m pytest -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
foreach ($case in @('A_baseline', 'B_corecpu', 'C_new_compute', 'D_full_new')) {
    python -m corecpu_lab train --config "configs/$case.yaml" --steps 2 --batch 4 --eval-batches 1 --out "results/smoke_${case}_$stamp"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
foreach ($case in @('A_baseline', 'B_corecpu', 'C_new_compute', 'D_full_new')) {
    python -m corecpu_lab train --config "configs/${case}_language.yaml" --steps 2 --batch 2 --eval-batches 1 --out "results/smoke_lm_${case}_$stamp"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
