$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

# Sparse probes found populated records through ~127k and empty samples at 128k+.
python discover_horses.py `
  --skip-tjk-list `
  --no-race-program `
  --no-graph `
  --max-pages 1 `
  --tjk-start 1 `
  --tjk-end 128000 `
  --rps 1 `
  --concurrency 3 `
  --timeout 30 `
  --retries 3 `
  --checkpoint-every 100

if ($LASTEXITCODE -ne 0) { throw "Public TJK discovery failed: $LASTEXITCODE" }

python normalize_data.py
if ($LASTEXITCODE -ne 0) { throw "Normalization failed: $LASTEXITCODE" }

python scrape_pedigreeall.py --rps 1 --concurrency 3 --batch-size 200 --timeout 30 --retries 3
if ($LASTEXITCODE -ne 0) { throw "Detailed public collection failed: $LASTEXITCODE" }

python analyze_dataset.py --output lake/analytics
if ($LASTEXITCODE -ne 0) { throw "Analysis failed: $LASTEXITCODE" }
