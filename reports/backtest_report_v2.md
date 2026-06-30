# Production Backtest Report v2

Generated: 2026-06-30 12:43:07

## Executive Decision

- Production ready: **No, not yet**. The intended recent-year temporal test now runs successfully; betting-data quality and live validation gates remain.
- Best holdout model: **ensemble**.
- Expected winner accuracy from the 2026 holdout: **27.59%** across `1642` races.
- Observed top-1 ROI under stated assumptions: **-28.94%**. This is descriptive, not a guaranteed live return.
- Highest SHAP contributors: catboost: last_3_avg_position, pre_race_handicap_rating, draw, race_class, carried_weight; logistic: race_class, surface, track, last_3_avg_position, draw; xgboost: last_3_avg_position, pre_race_handicap_rating, draw, race_class, carried_weight.

## Temporal Design

| split | evaluation_year | train_rows | train_races | train_max_date | evaluation_rows | evaluation_races | evaluation_min_date | evaluation_max_date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation | 2024 | 181716 | 22792 | 2023-12-17 | 5862 | 799 | 2024-01-03 | 2024-12-31 |
| test | 2025 | 187578 | 23591 | 2024-12-31 | 26611 | 2867 | 2025-01-01 | 2025-12-31 |
| holdout | 2026 | 214189 | 26458 | 2025-12-31 | 15322 | 1642 | 2026-01-01 | 2026-06-26 |

Every fold was retrained from scratch with `train_date < evaluation_date`. Saved production model predictions were not reused. Validation, test and holdout evaluate 2024, 2025 and 2026 respectively.

## Data Integrity

- Source rows/columns: `961695` / `68`.
- Backtest as-of date: `2026-06-30`; future-dated rows excluded: `0`.
- Completed valid-race rows evaluated/trained: `229511`.
- Incomplete-field races excluded before training/evaluation: `116715`.
- Excluded races without exactly one winner or with fewer than two runners: `0`.
- Duplicate horse/race rows in source: `0`.
- Leakage columns intersecting model features: `[]`.
- AGF value-bet test remains unavailable because a reliable timestamped pre-race AGF snapshot is not present.

## Error Analysis

| model | lost_races | agf_favorite_analysis | predicted_horse_jockey_change_rate | predicted_horse_surface_change_rate | predicted_horse_distance_change_rate | predicted_horse_steward_incident_rate | actual_winner_jockey_change_rate | actual_winner_surface_change_rate | actual_winner_distance_change_rate | actual_winner_steward_incident_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic | 3944 | unavailable | 0.0000 | 0.3058 | 0.6235 | 0.0000 | 0.0000 | 0.3269 | 0.6650 | 0.0000 |
| catboost | 3904 | unavailable | 0.0000 | 0.3030 | 0.6378 | 0.0000 | 0.0000 | 0.3269 | 0.6650 | 0.0000 |
| xgboost | 3911 | unavailable | 0.0000 | 0.3199 | 0.6397 | 0.0000 | 0.0000 | 0.3269 | 0.6650 | 0.0000 |
| ensemble | 3894 | unavailable | 0.0000 | 0.3038 | 0.6366 | 0.0000 | 0.0000 | 0.3269 | 0.6650 | 0.0000 |

AGF-favorite loss analysis is unavailable. Commissioner, jockey, surface and distance indicators are reported as association rates only; they do not establish causality.

## Weaknesses And Final Work

- Preserve the DB-backed rebuild and rerun these recent-year splits after each material data refresh.
- Repair AGF ingestion before enabling value betting; preserve timestamped pre-race AGF snapshots.
- Confirm that historical odds are genuinely available pre-bet and encode dead heats, scratches, deductions, commissions and stake limits.
- Monitor live calibration and feature drift across the 2024/2025/2026 evaluation sequence.
- Use the selected model only after those gates pass; current results justify shadow mode, not unattended wagering.

## Artifacts

- `output/backtest_predictions_v2.csv`
- `output/model_scores_v2.csv`
- `output/roi_simulation_v2.csv`
- `output/calibration_table_v2.csv`
- `reports/model_comparison_v2.md`
- `reports/calibration_report_v2.md`
- `reports/calibration_curve_v2.png`
- `reports/roi_report_v2.md`
- `reports/feature_importance_v2.md`
