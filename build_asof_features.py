"""Build certified pre-race features from immutable snapshots only.

This module intentionally has no dependency on the legacy horse_races table.
"""
from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from app_config import DB_PATH, OUTPUT_DIR, PROJECT_ROOT
from feature_contract import MODEL_FEATURES, POST_RACE_COLUMNS, validate_model_feature_contract
from migrate_provenance_schema import apply_migrations

ROOT = PROJECT_ROOT
DB = DB_PATH
CSV = OUTPUT_DIR / "asof_features.csv"
PARQUET = OUTPUT_DIR / "asof_features.parquet"


def latest_program_asof(connection: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """WITH eligible AS (
               SELECT *, ROW_NUMBER() OVER (
                   PARTITION BY race_id,horse_id
                   ORDER BY captured_at DESC,snapshot_id DESC
               ) AS rn
               FROM program_snapshots
               WHERE julianday(captured_at) < julianday(race_start_at)
           )
           SELECT * FROM eligible WHERE rn=1""",
        connection,
    )


def latest_results(connection: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """WITH ranked AS (
               SELECT *, ROW_NUMBER() OVER (
                   PARTITION BY race_id,horse_id
                   ORDER BY captured_at DESC,result_id DESC
               ) AS rn
               FROM race_results
               WHERE result_status='finished'
           )
           SELECT * FROM ranked WHERE rn=1""",
        connection,
    )


def latest_market_asof(
    targets: pd.DataFrame, values: pd.DataFrame, value_columns: list[str]
) -> pd.DataFrame:
    if targets.empty or values.empty:
        return pd.DataFrame(columns=["race_id", "horse_id", *value_columns])
    joined = targets[["race_id", "horse_id", "race_start_at"]].merge(
        values, on=["race_id", "horse_id"], how="left", suffixes=("", "_market")
    )
    joined["_start"] = pd.to_datetime(joined["race_start_at"], utc=True, errors="coerce")
    joined["_captured"] = pd.to_datetime(joined["captured_at"], utc=True, errors="coerce")
    joined = joined[joined["_captured"] < joined["_start"]]
    if joined.empty:
        return pd.DataFrame(columns=["race_id", "horse_id", *value_columns])
    joined = joined.sort_values("_captured").drop_duplicates(["race_id", "horse_id"], keep="last")
    return joined[["race_id", "horse_id", *value_columns]]


def history_features(target: pd.Series, history: pd.DataFrame) -> dict[str, object]:
    target_start = pd.to_datetime(target["race_start_at"], utc=True)
    past = history[
        history["horse_id"].eq(target["horse_id"])
        & history["_start"].lt(target_start)
    ].sort_values(["_start", "race_no", "race_id"], kind="stable")
    if past.empty:
        return {
            "days_since_last_race": np.nan,
            "last_3_avg_position": np.nan,
            "last_5_avg_position": np.nan,
            "last_10_avg_position": np.nan,
            "surface_win_rate": np.nan,
            "distance_win_rate": np.nan,
            "track_win_rate": np.nan,
            "jockey_horse_win_rate": np.nan,
            "trainer_horse_win_rate": np.nan,
            "weight_change": np.nan,
            "class_change": np.nan,
            "distance_change": np.nan,
            "surface_change": np.nan,
        }
    last = past.iloc[-1]
    result = {
        "days_since_last_race": (target_start - last["_start"]).total_seconds() / 86400.0,
        "weight_change": target["carried_weight"] - last["carried_weight"],
        "class_change": int(str(target["race_class"]) != str(last["race_class"])),
        "distance_change": target["distance"] - last["distance"],
        "surface_change": int(str(target["surface"]) != str(last["surface"])),
    }
    finishes = pd.to_numeric(past["finish_position"], errors="coerce")
    for window in (3, 5, 10):
        result[f"last_{window}_avg_position"] = finishes.tail(window).mean()
    for category, output in [
        ("surface", "surface_win_rate"), ("distance", "distance_win_rate"),
        ("track", "track_win_rate"), ("jockey", "jockey_horse_win_rate"),
        ("trainer", "trainer_horse_win_rate"),
    ]:
        selected = past[past[category].astype(str).eq(str(target[category]))]
        finish = pd.to_numeric(selected["finish_position"], errors="coerce")
        result[output] = finish.eq(1).sum() / len(selected) if len(selected) else np.nan
    return result


def build_frame(db_path: str | Path = DB) -> pd.DataFrame:
    validate_model_feature_contract(MODEL_FEATURES)
    apply_migrations(db_path)
    connection = sqlite3.connect(str(db_path), timeout=60)
    try:
        program = latest_program_asof(connection)
        results = latest_results(connection)
        agf = pd.read_sql_query("SELECT * FROM agf_snapshots", connection)
        odds = pd.read_sql_query("SELECT * FROM odds_snapshots", connection)
    finally:
        connection.close()
    metadata = [
        "race_id", "horse_id", "horse_name", "race_start_at", "race_no",
        "captured_at", "source_endpoint", "source_request_id", "snapshot_id",
    ]
    output_columns = metadata + MODEL_FEATURES + [
        "agf_percent", "agf_rank", "agf_captured_at", "odds", "odds_captured_at"
    ]
    if program.empty:
        return pd.DataFrame(columns=output_columns)

    program["_start"] = pd.to_datetime(program["race_start_at"], utc=True, errors="raise")
    program["_captured"] = pd.to_datetime(program["captured_at"], utc=True, errors="raise")
    if not (program["_captured"] < program["_start"]).all():
        raise AssertionError("Program as-of join admitted captured_at >= race_start_at")
    # The race-program HANDICAP field is a genuine pre-race value because the
    # selected immutable program snapshot is strictly earlier than race start.
    # Keep the raw storage name out of the model contract to prevent accidental
    # reuse of post-race GET:Tjk/Get.HP in historical builds.
    program["pre_race_handicap_rating"] = pd.to_numeric(
        program["handicap_rating"], errors="coerce"
    )

    if results.empty:
        history = pd.DataFrame(columns=list(program.columns) + ["finish_position"])
    else:
        history = program.merge(
            results[["race_id", "horse_id", "finish_position"]],
            on=["race_id", "horse_id"], how="inner",
        )
        history["_start"] = pd.to_datetime(history["race_start_at"], utc=True, errors="raise")

    derived = pd.DataFrame([history_features(row, history) for _, row in program.iterrows()])
    frame = pd.concat([program.reset_index(drop=True), derived], axis=1)
    agf_latest = latest_market_asof(
        program, agf, ["agf_percent", "agf_rank", "captured_at"]
    ).rename(columns={"captured_at": "agf_captured_at"})
    odds_latest = latest_market_asof(
        program, odds, ["odds", "captured_at"]
    ).rename(columns={"captured_at": "odds_captured_at"})
    frame = frame.merge(agf_latest, on=["race_id", "horse_id"], how="left")
    frame = frame.merge(odds_latest, on=["race_id", "horse_id"], how="left")

    forbidden = sorted(set(MODEL_FEATURES) & POST_RACE_COLUMNS)
    if forbidden:
        raise AssertionError(f"Post-race columns entered model features: {forbidden}")
    return frame[output_columns].sort_values(
        ["race_start_at", "race_no", "race_id", "horse_id"], kind="stable"
    ).reset_index(drop=True)


def write_frame(frame: pd.DataFrame) -> None:
    CSV.parent.mkdir(exist_ok=True)
    frame.to_csv(CSV, index=False, encoding="utf-8")
    frame.to_parquet(PARQUET, index=False)
    csv = pd.read_csv(CSV, low_memory=False)
    parquet = pd.read_parquet(PARQUET)
    if csv.shape != parquet.shape or list(csv.columns) != list(parquet.columns):
        raise AssertionError("As-of CSV/Parquet synchronization failed")


def main() -> int:
    frame = build_frame()
    write_frame(frame)
    # A certified build is atomic from the caller's perspective: invariant CI
    # and provenance validation must both pass before success is returned.
    subprocess.run([sys.executable, str(ROOT / "run_leakage_ci.py")], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / "validate_feature_provenance.py")], cwd=ROOT, check=True)
    print({"rows": len(frame), "races": frame["race_id"].nunique() if len(frame) else 0})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
