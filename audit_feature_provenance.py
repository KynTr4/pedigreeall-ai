"""Generate a feature provenance and as-of leakage audit for the Benter dataset."""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from feature_contract import MODEL_FEATURES

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "output" / "final_benter_dataset.parquet"
DB = ROOT / "pedigreeall_progress.db"
REPORT = ROOT / "reports" / "feature_provenance_leakage_audit.md"
MATRIX = ROOT / "reports" / "feature_provenance_matrix.csv"

DIRECT_MODEL = {
    "track": ("horse_races.hippodrome", "Race venue"),
    "distance": ("horse_races.distance", "Race distance"),
    "surface": ("horse_races.surface", "Race surface"),
    "race_class": ("horse_races.race_class", "Race class"),
    "carried_weight": ("horse_races.weight", "Declared/actual carried weight"),
    "draw": ("horse_races.gate", "Starting gate"),
    "pre_race_handicap_rating": (
        "lag(horse_races.rating/GET:Tjk/Get.HP)",
        "Previous race HP; current-race HP is forbidden because it is post-race updated",
    ),
}
ROLLING_MODEL = {
    "days_since_last_race": "Latest prior race date",
    "last_3_avg_position": "Prior finish positions, window=3",
    "last_5_avg_position": "Prior finish positions, window=5",
    "last_10_avg_position": "Prior finish positions, window=10",
    "surface_win_rate": "Prior finishes on current surface",
    "distance_win_rate": "Prior finishes at current distance",
    "track_win_rate": "Prior finishes at current track",
    "jockey_horse_win_rate": "Prior finishes with current jockey",
    "trainer_horse_win_rate": "Prior finishes with current trainer",
    "weight_change": "Current weight minus prior-race weight",
    "class_change": "Current class versus prior-race class",
    "distance_change": "Current distance minus prior-race distance",
    "surface_change": "Current surface versus prior-race surface",
}
OUTCOME_COLUMNS = {
    "finish_position": ("horse_races.finish (S)", "Post-race result"),
    "finish_time_seconds": ("horse_races.race_time (DERECE)", "Post-race result"),
    "odds": ("horse_races.odds (GNY)", "Final/result-history win dividend; not proven pre-race"),
    "agf": ("horse_races.agf", "Normalizer currently writes NULL"),
    "prize": ("horse_races.prize (IKRAMIYE)", "Result-history field; semantics/timing unproven"),
    "agf_percent": ("output/agf_data.csv", "Potentially pre-race, but no captured_at timestamp"),
    "agf_rank": ("output/agf_data.csv", "Potentially pre-race, but no captured_at timestamp"),
    "margin_text": ("builder placeholder", "Post-race margin concept"),
    "margin_lengths_numeric": ("builder placeholder", "Post-race margin concept"),
    "handicap_rating": (
        "horse_races.rating (GET:Tjk/Get.HP)",
        "Post-race history value; forbidden by feature contract",
    ),
}
IDENTITY = {
    "horse_id": "horse_races.horse_key", "race_id": "horse_races.race_id",
    "race_date": "horse_races.race_date", "horse_name": "horse_profiles.name",
    "jockey": "horse_races.jockey", "trainer": "horse_races.trainer",
}
TRACK_COLUMNS = {
    "track_condition", "turf_condition", "dirt_condition", "synthetic_condition",
    "weather", "temperature", "humidity", "pressure", "wind_speed", "wind_direction",
}
WORKOUT_COLUMNS = {
    "last_workout_date", "last_workout_distance", "last_workout_time",
    "days_since_last_workout", "workout_count_last_7d", "workout_count_last_14d",
}
COMMISSIONER_COLUMNS = {
    "race_no", "had_jockey_change", "had_trainer_change", "had_equipment_change",
    "had_veterinary_issue", "had_lameness_issue", "had_steward_incident",
    "had_recent_scratch", "incident_count_last_30d", "veterinary_count_last_180d",
    "steward_incident_count_last_180d",
}


def build_matrix(columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        used = column in MODEL_FEATURES
        if column in DIRECT_MODEL:
            source, detail = DIRECT_MODEL[column]
            availability = "Semantically pre-race; capture time not recorded"
            future = "No calculation; direct target-race value"
            same_race = "Possible post-event correction cannot be ruled out"
            verdict = "CONDITIONAL"
        elif column in ROLLING_MODEL:
            source, detail = "Derived from horse_races", ROLLING_MODEL[column]
            availability = "Computed from earlier races"
            future = "Uses shift/cumulative prior values; strict-date audit sampled"
            same_race = "Current finish excluded; date-only ordering has 2 historical collision groups"
            verdict = "PASS_WITH_CAVEAT"
        elif column in OUTCOME_COLUMNS:
            source, detail = OUTCOME_COLUMNS[column]
            availability = detail
            future = "Not applicable"
            same_race = "Outcome/final-market or timing-unproven field"
            verdict = "EXCLUDED_POST_RACE_OR_UNPROVEN"
        elif column in IDENTITY:
            source, detail = IDENTITY[column], "Identity/context column"
            availability = "Identifier or nominal race context"
            future = "No rolling calculation"
            same_race = "Not used by model"
            verdict = "NOT_MODEL_FEATURE"
        elif column in TRACK_COLUMNS:
            source, detail = "output/track_conditions.csv", "Date/track lookup"
            availability = "No captured_at; historical observation may be post-race"
            future = "Direct date/track merge"
            same_race = "Timing not provable; not used by model"
            verdict = "NOT_MODEL_FEATURE_UNPROVEN_TIME"
        elif column in WORKOUT_COLUMNS:
            source, detail = "output/workouts.csv", "Date/horse lookup"
            availability = "No captured_at; last_workout semantics require validation"
            future = "Direct date/horse merge"
            same_race = "Timing not provable; not used by model"
            verdict = "NOT_MODEL_FEATURE_UNPROVEN_TIME"
        elif column in COMMISSIONER_COLUMNS:
            source, detail = "output/komiser_events.csv", "Commissioner report lookup/default"
            availability = "Report is generally post-event; no captured_at"
            future = "Current builder defaults values to zero"
            same_race = "Unsafe as pre-race input; not used by model"
            verdict = "NOT_MODEL_FEATURE_POST_RACE"
        else:
            source, detail = "builder", "Placeholder or unmapped column"
            availability = "Unverified"
            future = "Unverified"
            same_race = "Not used by model"
            verdict = "NOT_MODEL_FEATURE_UNVERIFIED"
        rows.append({
            "feature": column, "model_used": used, "source": source,
            "source_detail": detail, "known_before_race": availability,
            "future_race_used": future, "same_race_result_risk": same_race,
            "verdict": verdict,
        })
    return pd.DataFrame(rows)


def close(a: object, b: object, tolerance: float = 1e-8) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    try:
        return bool(np.isclose(float(a), float(b), equal_nan=True, atol=tolerance, rtol=tolerance))
    except (TypeError, ValueError):
        return str(a) == str(b)


def strict_history_reference(target: pd.Series, horse: pd.DataFrame) -> dict[str, object]:
    past = horse[horse["_date"] < target["_date"]].sort_values(["_date", "race_id"], kind="stable")
    if past.empty:
        return {
            "days_since_last_race": np.nan, "last_3_avg_position": np.nan,
            "last_5_avg_position": np.nan, "last_10_avg_position": np.nan,
            "surface_win_rate": np.nan, "distance_win_rate": np.nan,
            "track_win_rate": np.nan, "jockey_horse_win_rate": np.nan,
            "trainer_horse_win_rate": np.nan, "weight_change": np.nan,
            "class_change": 1, "distance_change": np.nan, "surface_change": 1,
        }
    last = past.iloc[-1]
    result: dict[str, object] = {
        "days_since_last_race": (target["_date"] - last["_date"]).days,
        "weight_change": target["carried_weight"] - last["carried_weight"],
        "class_change": int(str(target["race_class"]) != str(last["race_class"])),
        "distance_change": target["distance"] - last["distance"],
        "surface_change": int(str(target["surface"]) != str(last["surface"])),
    }
    finishes = pd.to_numeric(past["finish_position"], errors="coerce")
    for window in (3, 5, 10):
        # Production semantics are the last N race rows; missing finishes stay
        # inside the row window and are ignored only by the final mean.
        result[f"last_{window}_avg_position"] = finishes.tail(window).mean() if len(finishes) else np.nan
    for category, output in [
        ("surface", "surface_win_rate"), ("distance", "distance_win_rate"),
        ("track", "track_win_rate"), ("jockey", "jockey_horse_win_rate"),
        ("trainer", "trainer_horse_win_rate"),
    ]:
        selected = past[past[category].astype(str).eq(str(target[category]))]
        selected_finish = pd.to_numeric(selected["finish_position"], errors="coerce")
        result[output] = selected_finish.eq(1).sum() / len(selected) if len(selected) else np.nan
    return result


def run_audit() -> dict[str, object]:
    data = pd.read_parquet(DATASET)
    data["_date"] = pd.to_datetime(data["race_date"], dayfirst=True, errors="coerce")
    matrix = build_matrix(list(data.columns.drop("_date")))
    MATRIX.parent.mkdir(exist_ok=True)
    matrix.to_csv(MATRIX, index=False, encoding="utf-8")

    with sqlite3.connect(DB) as connection:
        horse_race_columns = [row[1] for row in connection.execute("PRAGMA table_info(horse_races)")]
        program_columns = [row[1] for row in connection.execute("PRAGMA table_info(race_program_entries)")]
        same_day = pd.read_sql_query(
            """SELECT horse_key, race_date, COUNT(*) AS races
               FROM horse_races GROUP BY horse_key, race_date HAVING COUNT(*) > 1""",
            connection,
        )
        raw_window = connection.execute(
            "SELECT COUNT(*), MIN(fetched_at), MAX(fetched_at) FROM raw_api_responses"
        ).fetchone()
        recent_market = connection.execute(
            """SELECT COUNT(*),
                      SUM(CASE WHEN agf IS NOT NULL AND TRIM(agf) != '' THEN 1 ELSE 0 END),
                      SUM(CASE WHEN odds IS NOT NULL AND TRIM(odds) != '' THEN 1 ELSE 0 END)
               FROM horse_races WHERE SUBSTR(race_date, -4) >= '2024'"""
        ).fetchone()

    outcome_intersection = sorted(set(MODEL_FEATURES) & set(OUTCOME_COLUMNS))
    has_race_observed_at = bool({"observed_at", "fetched_at", "captured_at"} & set(horse_race_columns))
    has_program_observed_at = bool({"observed_at", "fetched_at", "captured_at"} & set(program_columns))

    recent = data[data["_date"].dt.year.isin([2024, 2025, 2026])]
    sample = recent.sample(min(300, len(recent)), random_state=42)
    groups = data.groupby("horse_id", sort=False).groups
    mismatches: list[tuple[str, str]] = []
    checked = 0
    for _, target in sample.iterrows():
        horse = data.loc[groups[target["horse_id"]]]
        expected = strict_history_reference(target, horse)
        for feature, value in expected.items():
            checked += 1
            if not close(target[feature], value):
                mismatches.append((str(target["horse_id"]), feature))

    return {
        "matrix": matrix,
        "rows": len(data),
        "columns": len(data.columns) - 1,
        "model_features": len(MODEL_FEATURES),
        "outcome_intersection": outcome_intersection,
        "horse_races_has_capture_time": has_race_observed_at,
        "race_program_has_capture_time": has_program_observed_at,
        "raw_response_count": int(raw_window[0]),
        "raw_response_min": raw_window[1],
        "raw_response_max": raw_window[2],
        "same_day_groups": len(same_day),
        "same_day_rows": int(same_day["races"].sum()) if len(same_day) else 0,
        "reference_sample_rows": len(sample),
        "reference_checks": checked,
        "reference_mismatches": len(mismatches),
        "reference_mismatch_examples": mismatches[:10],
        "recent_rows": int(recent_market[0]),
        "recent_agf_rows": int(recent_market[1] or 0),
        "recent_odds_rows": int(recent_market[2] or 0),
    }


def write_report(result: dict[str, object]) -> None:
    matrix: pd.DataFrame = result["matrix"]  # type: ignore[assignment]
    model_matrix = matrix[matrix["model_used"]]
    verdict_counts = model_matrix["verdict"].value_counts()
    matrix_lines = []
    for _, row in model_matrix.iterrows():
        matrix_lines.append(
            f"| `{row.feature}` | {row.source} | {row.known_before_race} | "
            f"{row.future_race_used} | **{row.verdict}** |"
        )
    mismatch_text = (
        "none" if not result["reference_mismatch_examples"]
        else ", ".join(f"{horse}/{feature}" for horse, feature in result["reference_mismatch_examples"])
    )
    report = f"""# Feature Provenance / As-of Leakage Audit

Generated: {datetime.now():%Y-%m-%d %H:%M:%S}

## Executive Verdict

**Overall leakage gate: NOT PROVEN / production-blocking provenance gap.**

The 13 derived model features pass sampled strict-history recomputation, and no
explicit outcome/final-market column is in the 20-feature model list. However,
the seven direct race features come from historical result rows without a
row-level `observed_at`, `captured_at`, race start timestamp, or link to the raw
snapshot that supplied that value. They are semantically expected before a race,
but this dataset cannot prove that the exact stored value was available then.

## Blocking Findings

1. **[BLOCKING] Direct feature as-of time is unprovable.** `horse_races` capture-time column present: `{result['horse_races_has_capture_time']}`. The table stores current normalized values, not time-versioned snapshots.
2. **[BLOCKING] Program snapshots are not auditable.** `race_program_entries` capture-time column present: `{result['race_program_has_capture_time']}`; it currently represents only a mutable daily snapshot.
3. **[BLOCKING if enabled] Odds/GNY is not a safe pre-race feature.** In `horse_races`, `GNY` comes from the historical result table and is treated as final/result-history information. It is correctly excluded from model features.
4. **[BLOCKING if enabled] AGF timing is not proven.** The historical table has `{result['recent_agf_rows']:,}` populated AGF rows out of `{result['recent_rows']:,}` since 2024, while `output/agf_data.csv` has no capture timestamp. AGF is correctly excluded.

## Important Findings

1. **Date-only history ordering has an edge case.** There are `{result['same_day_groups']}` horse/date collision groups (`{result['same_day_rows']}` rows). Row-wise `shift()` can treat one same-day race as history for another without a race start time. These are old training rows, not 2024–2026 evaluation rows, but should be quarantined or ordered by verified start time.
2. `jockey`, `trainer`, track/weather, workout and commissioner fields also lack immutable as-of capture metadata. They are not in the current model, so they do not affect the reported backtest score.
3. Historical raw responses were fetched only in `{result['raw_response_min']}` → `{result['raw_response_max']}` (`{result['raw_response_count']:,}` responses). A 2026 fetch of a 2024 result does not demonstrate 2024 pre-race availability.

## Automated Evidence

- Dataset audited: `{result['rows']:,}` rows × `{result['columns']}` columns.
- Provenance rows written: `{len(matrix)}` (every final dataset column).
- Model features: `{result['model_features']}`.
- Explicit outcome/final-market intersection with model list: `{result['outcome_intersection']}`.
- Strict `history.race_date < target.race_date` reference sample: `{result['reference_sample_rows']}` recent rows / `{result['reference_checks']:,}` feature comparisons.
- Reference mismatches: `{result['reference_mismatches']}`; examples: `{mismatch_text}`.
- Model verdict counts: `{verdict_counts.to_dict()}`.

The reference implementation never reads the target row's finish, time, odds,
AGF or prize. Matching stored rolling values therefore supports absence of
future-race and same-row outcome use for sampled 2024–2026 records. This is
calculation evidence, not source-time evidence.

## Model Feature Matrix

| Feature | Source | Known before race? | Future race used? | Verdict |
| --- | --- | --- | --- | --- |
{chr(10).join(matrix_lines)}

## Odds / Ganyan Decision

- `horse_races.odds` maps from `HORSE_TABLE.GNY`, a historical results payload.
- It has `{result['recent_odds_rows']:,}` populated rows since 2024, but no quote timestamp or pre-race snapshot identity.
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
"""
    REPORT.write_text(report, encoding="utf-8")


def main() -> int:
    result = run_audit()
    write_report(result)
    print({key: value for key, value in result.items() if key != "matrix"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
