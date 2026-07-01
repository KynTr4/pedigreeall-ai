"""Generate and immutably archive pre-race predictions without retraining."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from app_config import DB_PATH, MODELS_DIR, OUTPUT_DIR, PROJECT_ROOT
from feature_contract import (
    CATEGORICAL_FEATURES, FEATURE_CONTRACT_VERSION, MODEL_FEATURES,
    validate_model_feature_contract,
)
from migrate_provenance_schema import apply_migrations
from race_scope import is_turkey_track

ROOT = PROJECT_ROOT
DB = DB_PATH
FEATURES = OUTPUT_DIR / "asof_features.parquet"
SHADOW_CSV = OUTPUT_DIR / "shadow_predictions.csv"
HISTORY_CSV = OUTPUT_DIR / "prediction_history.csv"
MODEL_PATHS = {
    "logistic": MODELS_DIR / "benter_baseline_logistic.pkl",
    "catboost": MODELS_DIR / "benter_baseline_catboost.pkl",
    "xgboost": MODELS_DIR / "xgboost_production.pkl",
}
PIPELINE_FILES = [
    "shadow_mode.py", "build_asof_features.py", "feature_contract.py",
    "validate_feature_provenance.py",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def version_info() -> tuple[str, str]:
    model_parts = [f"{name}:{sha256_file(path)[:16]}" for name, path in MODEL_PATHS.items()]
    pipeline_digest = hashlib.sha256()
    for name in PIPELINE_FILES:
        path = ROOT / name
        pipeline_digest.update(name.encode())
        pipeline_digest.update(path.read_bytes())
    return "|".join(model_parts), pipeline_digest.hexdigest()[:24]


def load_models() -> dict[str, object]:
    missing = [str(path) for path in MODEL_PATHS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Shadow mode requires all fixed models: {missing}")
    models = {}
    for name, path in MODEL_PATHS.items():
        with path.open("rb") as stream:
            models[name] = pickle.load(stream)
    return models


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    validate_model_feature_contract(MODEL_FEATURES)
    output = frame[MODEL_FEATURES].copy()
    for column in MODEL_FEATURES:
        if column in CATEGORICAL_FEATURES:
            output[column] = output[column].astype(object)
            output.loc[output[column].isna(), column] = "missing"
            output[column] = output[column].map(lambda value: "missing" if pd.isna(value) else str(value))
        else:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def normalize_by_race(frame: pd.DataFrame, column: str) -> pd.Series:
    total = frame.groupby("race_id")[column].transform("sum")
    count = frame.groupby("race_id")[column].transform("size")
    return pd.Series(np.where(total > 0, frame[column] / total, 1.0 / count), index=frame.index)


def score_fixed_models(frame: pd.DataFrame, models: dict[str, object]) -> pd.DataFrame:
    features = prepare_features(frame)
    cat_features = features.copy()
    for column in CATEGORICAL_FEATURES:
        cat_features[column] = cat_features[column].astype(str)
    raw = {
        "logistic": models["logistic"].predict_proba(features)[:, 1],
        "catboost": models["catboost"].predict_proba(cat_features)[:, 1],
        "xgboost": models["xgboost"].predict_proba(features)[:, 1],
    }
    extra_cols = [c for c in ["agf_percent", "agf_rank", "odds"] if c in frame.columns]
    output = frame[[
        "race_id", "horse_id", "horse_name", "race_start_at", "snapshot_id",
        "source_request_id", *MODEL_FEATURES,
    ] + extra_cols].copy()
    for model, values in raw.items():
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"Invalid {model} probabilities")
        output[f"{model}_probability"] = values
        output[f"{model}_probability"] = normalize_by_race(output, f"{model}_probability")
    output["ensemble_probability"] = output[
        ["logistic_probability", "catboost_probability", "xgboost_probability"]
    ].mean(axis=1)
    output["predicted_rank"] = output.groupby("race_id")["ensemble_probability"].rank(
        method="first", ascending=False
    ).astype(int)
    return output


def feature_hash(row: pd.Series) -> str:
    def clean(value):
        if pd.isna(value):
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value
    payload = {
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "snapshot_id": int(row["snapshot_id"]),
        "source_request_id": str(row["source_request_id"]),
        "race_start_at": str(row["race_start_at"]),
        "features": {column: clean(row[column]) for column in MODEL_FEATURES},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def feature_values_json(row: pd.Series) -> str:
    def clean(value):
        if pd.isna(value):
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value
    return json.dumps(
        {column: clean(row[column]) for column in MODEL_FEATURES},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    )


def archive_predictions(
    scored: pd.DataFrame,
    db_path: str | Path = DB,
    prediction_time: str | None = None,
) -> pd.DataFrame:
    apply_migrations(db_path)
    stamp = prediction_time or datetime.now(timezone.utc).isoformat()
    predicted_at = pd.Timestamp(stamp)
    starts = pd.to_datetime(scored["race_start_at"], utc=True, errors="raise")
    if not (predicted_at < starts).all():
        bad = scored.loc[~(predicted_at < starts), "race_id"].unique().tolist()
        raise ValueError(f"Shadow prediction is not pre-race: {bad[:10]}")
    # Fetch live AGF & Odds at prediction time from DB
    race_ids = scored["race_id"].unique().tolist()
    agf_map = {}
    odds_map = {}
    connection = sqlite3.connect(str(db_path), timeout=60)
    try:
        connection.row_factory = sqlite3.Row
        for rid in race_ids:
            # Query AGF
            agf_rows = connection.execute("""
                SELECT horse_id, agf_percent, agf_rank
                FROM (
                    SELECT horse_id, agf_percent, agf_rank,
                           ROW_NUMBER() OVER (PARTITION BY horse_id ORDER BY captured_at DESC) AS rn
                    FROM agf_snapshots
                    WHERE race_id = ? AND julianday(captured_at) <= julianday(?)
                ) WHERE rn = 1
            """, (rid, stamp)).fetchall()
            for r in agf_rows:
                agf_map[(rid, r["horse_id"])] = (r["agf_percent"], r["agf_rank"])
                
            # Query Odds
            odds_rows = connection.execute("""
                SELECT horse_id, odds
                FROM (
                    SELECT horse_id, odds,
                           ROW_NUMBER() OVER (PARTITION BY horse_id ORDER BY captured_at DESC) AS rn
                    FROM odds_snapshots
                    WHERE race_id = ? AND julianday(captured_at) <= julianday(?)
                ) WHERE rn = 1
            """, (rid, stamp)).fetchall()
            for r in odds_rows:
                odds_map[(rid, r["horse_id"])] = r["odds"]
    finally:
        connection.close()

    model_version, pipeline_version = version_info()
    rows = []
    run_id = uuid.uuid4().hex
    for index, row in scored.reset_index(drop=True).iterrows():
        rid = row["race_id"]
        hid = row["horse_id"]
        
        # Get from DB map or fallback to row
        db_agf = agf_map.get((rid, hid), (None, None))
        db_odds = odds_map.get((rid, hid), None)
        
        agf_percent = db_agf[0] if db_agf[0] is not None else (float(row["agf_percent"]) if "agf_percent" in row and pd.notna(row["agf_percent"]) else None)
        agf_rank = db_agf[1] if db_agf[1] is not None else (int(row["agf_rank"]) if "agf_rank" in row and pd.notna(row["agf_rank"]) else None)
        odds = db_odds if db_odds is not None else (float(row["odds"]) if "odds" in row and pd.notna(row["odds"]) else None)

        rows.append({
            "prediction_id": f"{run_id}:{index}",
            "model_version": model_version,
            "pipeline_version": pipeline_version,
            "race_id": rid, "horse_id": hid,
            "prediction_time": stamp, "race_start_at": row["race_start_at"],
            "logistic_probability": float(row["logistic_probability"]),
            "catboost_probability": float(row["catboost_probability"]),
            "xgboost_probability": float(row["xgboost_probability"]),
            "ensemble_probability": float(row["ensemble_probability"]),
            "predicted_rank": int(row["predicted_rank"]),
            "feature_hash": feature_hash(row),
            "feature_values_json": feature_values_json(row),
            "feature_contract_version": FEATURE_CONTRACT_VERSION,
            "feature_snapshot_id": int(row["snapshot_id"]),
            "source_request_id": row["source_request_id"],
            "agf_percent": agf_percent,
            "agf_rank": agf_rank,
            "odds": odds,
        })
    archive = pd.DataFrame(rows)
    connection = sqlite3.connect(str(db_path), timeout=60)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executemany(
            """INSERT INTO prediction_snapshots(
                   prediction_id,model_version,pipeline_version,race_id,horse_id,
                   prediction_time,race_start_at,logistic_probability,
                   catboost_probability,xgboost_probability,ensemble_probability,
                   predicted_rank,feature_hash,feature_values_json,feature_contract_version,
                   feature_snapshot_id,source_request_id,agf_percent,agf_rank,odds)
               VALUES(:prediction_id,:model_version,:pipeline_version,:race_id,:horse_id,
                      :prediction_time,:race_start_at,:logistic_probability,
                      :catboost_probability,:xgboost_probability,:ensemble_probability,
                      :predicted_rank,:feature_hash,:feature_values_json,:feature_contract_version,
                      :feature_snapshot_id,:source_request_id,:agf_percent,:agf_rank,:odds)""",
            archive.to_dict("records"),
        )
        feature_rows = [{
            "prediction_id": row["prediction_id"], "race_id": row["race_id"],
            "horse_id": row["horse_id"], "prediction_time": row["prediction_time"],
            "race_start_at": row["race_start_at"],
            "feature_values_json": row["feature_values_json"], "feature_hash": row["feature_hash"],
            "feature_contract_version": row["feature_contract_version"], "archived_at": stamp,
        } for row in rows]
        connection.executemany(
            """INSERT INTO prediction_feature_snapshots(
                   prediction_id,race_id,horse_id,prediction_time,race_start_at,
                   feature_values_json,feature_hash,feature_contract_version,archived_at)
               VALUES(:prediction_id,:race_id,:horse_id,:prediction_time,:race_start_at,
                      :feature_values_json,:feature_hash,:feature_contract_version,:archived_at)""",
            feature_rows,
        )
        connection.commit()
    finally:
        connection.close()
    return archive


def export_prediction_history(db_path: str | Path = DB) -> None:
    connection = sqlite3.connect(str(db_path), timeout=60)
    try:
        history = pd.read_sql_query(
            """SELECT p.*,r.finish_position,r.winner,r.official_odds,
                      r.official_time,r.payout,r.matched_at
               FROM prediction_snapshots p
               LEFT JOIN prediction_results r USING(prediction_id)
               ORDER BY p.prediction_time,p.race_id,p.predicted_rank""",
            connection,
        )
    finally:
        connection.close()
    HISTORY_CSV.parent.mkdir(exist_ok=True)
    history.to_csv(HISTORY_CSV, index=False, encoding="utf-8")


def eligible_today(frame: pd.DataFrame, day: str, now: pd.Timestamp) -> pd.DataFrame:
    starts = pd.to_datetime(frame["race_start_at"], utc=True, errors="coerce")
    local_day = starts.dt.tz_convert(ZoneInfo("Europe/Istanbul")).dt.date.astype(str)
    return frame[local_day.eq(day) & starts.gt(now)].copy()


def final_prediction_exists(race_id: str, race_start_at: str, db_path: str | Path = DB) -> bool:
    start = pd.Timestamp(race_start_at)
    window_start = (start - pd.Timedelta(minutes=15)).isoformat()
    with sqlite3.connect(str(db_path), timeout=30) as connection:
        return connection.execute(
            """SELECT 1 FROM prediction_snapshots
               WHERE race_id=? AND julianday(prediction_time)>=julianday(?)
                 AND julianday(prediction_time)<julianday(race_start_at) LIMIT 1""",
            (race_id, window_start),
        ).fetchone() is not None


def main() -> int:
    parser = argparse.ArgumentParser(description="Immutable production shadow predictions")
    parser.add_argument("--date", default=datetime.now(ZoneInfo("Europe/Istanbul")).date().isoformat())
    parser.add_argument("--race-id")
    parser.add_argument("--final-freeze", action="store_true")
    args = parser.parse_args()
    if args.final_freeze and not args.race_id:
        parser.error("--final-freeze requires --race-id")
    apply_migrations(DB)
    if not FEATURES.exists():
        raise FileNotFoundError(FEATURES)
    all_features = pd.read_parquet(FEATURES)
    now = pd.Timestamp.now(tz="UTC")
    targets = eligible_today(all_features, args.date, now)
    if args.race_id:
        targets = targets[targets["race_id"].astype(str).eq(args.race_id)].copy()
    if args.final_freeze and not targets.empty:
        starts = pd.to_datetime(targets["race_start_at"], utc=True, errors="raise")
        lower = starts - pd.Timedelta(minutes=15)
        upper = starts
        if not ((now >= lower) & (now < upper)).all():
            raise RuntimeError("Final prediction is outside the race_start_at -15/0 minute window")
        if final_prediction_exists(args.race_id, targets.iloc[0]["race_start_at"], DB):
            print({"mode": "shadow_mode", "status": "final_prediction_already_frozen", "race_id": args.race_id})
            return 0
    if targets.empty:
        SHADOW_CSV.parent.mkdir(exist_ok=True)
        pd.DataFrame(columns=[
            "prediction_id", "race_id", "horse_id", "prediction_time", "race_start_at",
            "logistic_probability", "catboost_probability", "xgboost_probability",
            "ensemble_probability", "predicted_rank", "feature_hash",
        ]).to_csv(SHADOW_CSV, index=False)
        export_prediction_history()
        print({"mode": "shadow_mode", "status": "no_future_races", "rows": 0})
        return 0
    scored = score_fixed_models(targets, load_models())
    prediction_time = datetime.now(timezone.utc).isoformat()
    still_future = pd.to_datetime(scored["race_start_at"], utc=True).gt(pd.Timestamp(prediction_time))
    scored = scored[still_future].copy()
    if scored.empty:
        raise RuntimeError("All races crossed race_start_at during model inference")
    archive = archive_predictions(scored, prediction_time=prediction_time)
    archive.to_csv(SHADOW_CSV, index=False, encoding="utf-8")
    export_prediction_history()
    print({
        "mode": "shadow_mode", "status": "archived", "rows": len(archive),
        "races": archive["race_id"].nunique(), "prediction_time": prediction_time,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
