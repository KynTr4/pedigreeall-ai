# Leakage Gate v2

Generated: 2026-06-30 12:50:49

## Result: **PASS**

- [x] All populated model features come from snapshots captured before race start.
- [x] Every admitted program/AGF/odds observation satisfies captured_at < race_start_at.
- [x] Future rows do not change an earlier feature row.
- [x] Mutating same-race outcome fields does not change model features.
- [x] Same-day histories are ordered by race_start_at and race_no.
- [x] The model feature contract contains no post-race or market columns.

This PASS applies only to `output/asof_features.parquet`; legacy historical datasets remain uncertified.
