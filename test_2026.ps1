$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

python discover_horses.py `
    --db pedigreeall_2026_test.db `
    --skip-tjk-list `
    --no-race-program `
    --no-graph `
    --max-pages 1 `
    --race-days 365 `
    --tjk-start 1 `
    --tjk-end 10000 `
    --rps 1 `
    --concurrency 3 `
    --timeout 30 `
    --retries 3 `
    --checkpoint-every 100

if ($LASTEXITCODE -ne 0) { throw "Discovery failed: $LASTEXITCODE" }

python normalize_data.py --db pedigreeall_2026_test.db
if ($LASTEXITCODE -ne 0) { throw "Normalization failed: $LASTEXITCODE" }

python scrape_pedigreeall.py `
    --db pedigreeall_2026_test.db `
    --rps 1 `
    --concurrency 3 `
    --batch-size 100 `
    --timeout 30 `
    --retries 3

if ($LASTEXITCODE -ne 0) { throw "Collection failed: $LASTEXITCODE" }

python analyze_dataset.py `
    --db pedigreeall_2026_test.db `
    --output lake/analytics_2026_test

if ($LASTEXITCODE -ne 0) { throw "Analysis failed: $LASTEXITCODE" }

Write-Host "2026 test pipeline completed."
