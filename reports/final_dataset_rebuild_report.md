# Final Dataset Rebuild Report

Generated: 2026-06-30 12:36:12

## Result

- Source of truth: `pedigreeall_progress.db` / `horse_races`.
- Legacy `output/benter_features_with_komiser.csv` dependency: removed.
- Final shape: **961,695 rows x 68 columns**.
- CSV/Parquet row and column synchronization: **passed**.
- Duplicate `(horse_id, race_id)` keys: **0**.
- Date parser: `pd.to_datetime(..., dayfirst=True, errors="coerce")`.
- Unparseable dates: **0**.
- Historical rating source: post-race `GET:Tjk/Get.HP`; model input uses
  one-race-lagged `pre_race_handicap_rating` only.
- Internally complete race fields: **28,100** races (see
  `output/race_starter_coverage.csv`).

## Recent Year Distribution

| Year | Rows | Races |
| --- | ---: | ---: |
| 2024 | 21,262 | 5,336 |
| 2025 | 42,885 | 6,320 |
| 2026 | 23,954 | 2,930 |

## Root Cause

The old builder treated the existing final CSV (initially copied from the stale
`benter_features_with_komiser.csv`) as authoritative and only appended keys found
in `benter_features_base.csv`. The incremental feature job derived its query floor
from that CSV's maximum date, already in 2026, so 2024/2025 were never queried or
backfilled. Mixed ISO dates were also parsed without a stable day-first source
format. The rebuilt pipeline now starts from every current DB row and computes
rolling features using strictly earlier races for the same horse.
