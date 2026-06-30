# Feature Provenance / As-of Leakage Audit

Generated: 2026-06-30 12:50:49

## Executive Verdict

**Overall leakage gate: NOT PROVEN / production-blocking provenance gap.**

The 13 derived model features pass sampled strict-history recomputation, and no
explicit outcome/final-market column is in the 20-feature model list. However,
the seven direct race features come from historical result rows without a
row-level `observed_at`, `captured_at`, race start timestamp, or link to the raw
snapshot that supplied that value. They are semantically expected before a race,
but this dataset cannot prove that the exact stored value was available then.

## Blocking Findings

1. **[BLOCKING] Direct feature as-of time is unprovable.** `horse_races` capture-time column present: `False`. The table stores current normalized values, not time-versioned snapshots.
2. **[BLOCKING] Program snapshots are not auditable.** `race_program_entries` capture-time column present: `False`; it currently represents only a mutable daily snapshot.
3. **[BLOCKING if enabled] Odds/GNY is not a safe pre-race feature.** In `horse_races`, `GNY` comes from the historical result table and is treated as final/result-history information. It is correctly excluded from model features.
4. **[BLOCKING if enabled] AGF timing is not proven.** The historical table has `0` populated AGF rows out of `88,101` since 2024, while `output/agf_data.csv` has no capture timestamp. AGF is correctly excluded.

## Important Findings

1. **Date-only history ordering has an edge case.** There are `2` horse/date collision groups (`4` rows). Row-wise `shift()` can treat one same-day race as history for another without a race start time. These are old training rows, not 2024–2026 evaluation rows, but should be quarantined or ordered by verified start time.
2. `jockey`, `trainer`, track/weather, workout and commissioner fields also lack immutable as-of capture metadata. They are not in the current model, so they do not affect the reported backtest score.
3. Historical raw responses were fetched only in `2026-06-20T18:15:39.111146+00:00` → `2026-06-26T19:48:13.154304+00:00` (`128,211` responses). A 2026 fetch of a 2024 result does not demonstrate 2024 pre-race availability.

## Automated Evidence

- Dataset audited: `961,695` rows × `68` columns.
- Provenance rows written: `68` (every final dataset column).
- Model features: `20`.
- Explicit outcome/final-market intersection with model list: `[]`.
- Strict `history.race_date < target.race_date` reference sample: `300` recent rows / `3,900` feature comparisons.
- Reference mismatches: `0`; examples: `none`.
- Model verdict counts: `{'PASS_WITH_CAVEAT': 13, 'CONDITIONAL': 7}`.

The reference implementation never reads the target row's finish, time, odds,
AGF or prize. Matching stored rolling values therefore supports absence of
future-race and same-row outcome use for sampled 2024–2026 records. This is
calculation evidence, not source-time evidence.

## Model Feature Matrix

| Feature | Source | Known before race? | Future race used? | Verdict |
| --- | --- | --- | --- | --- |
| `track` | horse_races.hippodrome | Semantically pre-race; capture time not recorded | No calculation; direct target-race value | **CONDITIONAL** |
| `distance` | horse_races.distance | Semantically pre-race; capture time not recorded | No calculation; direct target-race value | **CONDITIONAL** |
| `surface` | horse_races.surface | Semantically pre-race; capture time not recorded | No calculation; direct target-race value | **CONDITIONAL** |
| `race_class` | horse_races.race_class | Semantically pre-race; capture time not recorded | No calculation; direct target-race value | **CONDITIONAL** |
| `carried_weight` | horse_races.weight | Semantically pre-race; capture time not recorded | No calculation; direct target-race value | **CONDITIONAL** |
| `draw` | horse_races.gate | Semantically pre-race; capture time not recorded | No calculation; direct target-race value | **CONDITIONAL** |
| `pre_race_handicap_rating` | lag(horse_races.rating/GET:Tjk/Get.HP) | Semantically pre-race; capture time not recorded | No calculation; direct target-race value | **CONDITIONAL** |
| `days_since_last_race` | Derived from horse_races | Computed from earlier races | Uses shift/cumulative prior values; strict-date audit sampled | **PASS_WITH_CAVEAT** |
| `last_3_avg_position` | Derived from horse_races | Computed from earlier races | Uses shift/cumulative prior values; strict-date audit sampled | **PASS_WITH_CAVEAT** |
| `last_5_avg_position` | Derived from horse_races | Computed from earlier races | Uses shift/cumulative prior values; strict-date audit sampled | **PASS_WITH_CAVEAT** |
| `last_10_avg_position` | Derived from horse_races | Computed from earlier races | Uses shift/cumulative prior values; strict-date audit sampled | **PASS_WITH_CAVEAT** |
| `surface_win_rate` | Derived from horse_races | Computed from earlier races | Uses shift/cumulative prior values; strict-date audit sampled | **PASS_WITH_CAVEAT** |
| `distance_win_rate` | Derived from horse_races | Computed from earlier races | Uses shift/cumulative prior values; strict-date audit sampled | **PASS_WITH_CAVEAT** |
| `track_win_rate` | Derived from horse_races | Computed from earlier races | Uses shift/cumulative prior values; strict-date audit sampled | **PASS_WITH_CAVEAT** |
| `jockey_horse_win_rate` | Derived from horse_races | Computed from earlier races | Uses shift/cumulative prior values; strict-date audit sampled | **PASS_WITH_CAVEAT** |
| `trainer_horse_win_rate` | Derived from horse_races | Computed from earlier races | Uses shift/cumulative prior values; strict-date audit sampled | **PASS_WITH_CAVEAT** |
| `weight_change` | Derived from horse_races | Computed from earlier races | Uses shift/cumulative prior values; strict-date audit sampled | **PASS_WITH_CAVEAT** |
| `class_change` | Derived from horse_races | Computed from earlier races | Uses shift/cumulative prior values; strict-date audit sampled | **PASS_WITH_CAVEAT** |
| `distance_change` | Derived from horse_races | Computed from earlier races | Uses shift/cumulative prior values; strict-date audit sampled | **PASS_WITH_CAVEAT** |
| `surface_change` | Derived from horse_races | Computed from earlier races | Uses shift/cumulative prior values; strict-date audit sampled | **PASS_WITH_CAVEAT** |

## Odds / Ganyan Decision

- `horse_races.odds` maps from `HORSE_TABLE.GNY`, a historical results payload.
- It has `88,068` populated rows since 2024, but no quote timestamp or pre-race snapshot identity.
- Treat it as **post-race/final-market data** for modeling. It may be used only for retrospective payout/ROI accounting with that limitation stated.
- Live odds require an append-only table keyed by `(race_id, horse_id, captured_at)` and a selection rule enforcing `captured_at < race_start_at`.

## Required Remediation Before a Leakage-Safe Claim

1. Add immutable `race_start_at`, `captured_at`, `source_request_key`, and `source_endpoint` to versioned program/market snapshots.
2. Build training rows with an as-of join: `captured_at = max(captured_at) where captured_at < race_start_at`.
3. Keep results (`finish`, `DERECE`, final `GNY`, margin, prize) in a separate post-race table and deny them at feature-schema validation.
4. Store AGF/odds snapshots append-only; never overwrite the pre-race observation with final values.
5. Replace date-only rolling order with verified race timestamps, or exclude same-horse/same-date collisions.
6. Add CI tests for prefix invariance, target-outcome mutation invariance, and source timestamp constraints.

## Scope

The full 62-column matrix is in `reports/feature_provenance_matrix.csv`. This
audit establishes what the current code and artifacts can demonstrate. Where
source capture timestamps are absent, it deliberately reports uncertainty rather
than inferring pre-race availability from field names.
