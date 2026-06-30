$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

python discover_horses.py `
  --db pedigreeall_2026_only.db `
  --skip-tjk-list `
  --no-graph `
  --race-days 365 `
  --tjk-start 0 `
  --tjk-end 0 `
  --rps 1 `
  --concurrency 3 `
  --timeout 30 `
  --retries 3 `
  --checkpoint-every 100

python normalize_data.py --db pedigreeall_2026_only.db

python scrape_pedigreeall.py `
  --db pedigreeall_2026_only.db `
  --rps 1 `
  --concurrency 3 `
  --batch-size 200 `
  --timeout 30 `
  --retries 3

python normalize_data.py --db pedigreeall_2026_only.db

python analyze_dataset.py `
  --db pedigreeall_2026_only.db `
  --output lake/analytics_2026_only

Write-Host "2026-only discovery completed."
