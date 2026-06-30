"""Match shadow outcomes and produce deterministic live monitoring artifacts."""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from app_config import DB_PATH, OUTPUT_DIR, PROJECT_ROOT, REPORTS_DIR
from feature_contract import (
    CATEGORICAL_FEATURES, FEATURE_CONTRACT_VERSION, MODEL_FEATURES,
    validate_model_feature_contract,
)
from migrate_provenance_schema import apply_migrations
from shadow_mode import export_prediction_history
from validate_feature_provenance import validate as validate_provenance

ROOT = PROJECT_ROOT
DB = DB_PATH
OUTPUT = OUTPUT_DIR
REPORTS = REPORTS_DIR
MODEL_PROBS = {
    "logistic": "logistic_probability",
    "catboost": "catboost_probability",
    "xgboost": "xgboost_probability",
    "ensemble": "ensemble_probability",
}
THRESHOLDS = {
    "psi": 0.25, "js": 0.20, "kl": 0.50,
    "confidence_shift": 0.15, "ece": 0.15,
    "missing_rate_delta": 0.20, "unseen_category_rate": 0.10,
    "target_rate_shift": 0.10,
}
MIN_CURRENT_ROWS = 30
MIN_REFERENCE_ROWS = 100
MIN_CALIBRATION_RACES = 20
LONGSHOT_ODDS = 10.0


def match_prediction_results(db_path: str | Path = DB) -> int:
    apply_migrations(db_path)
    stamp = datetime.now(timezone.utc).isoformat()
    connection = sqlite3.connect(str(db_path), timeout=60)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """WITH latest_result AS (
                   SELECT *,ROW_NUMBER() OVER(
                       PARTITION BY race_id,horse_id ORDER BY captured_at DESC,result_id DESC
                   ) AS rn FROM race_results WHERE result_status='finished'
               )
               SELECT p.prediction_id,r.finish_position,r.result_odds,r.finish_time
               FROM prediction_snapshots p
               JOIN latest_result r ON r.race_id=p.race_id AND r.horse_id=p.horse_id AND r.rn=1
               LEFT JOIN prediction_results pr ON pr.prediction_id=p.prediction_id
               WHERE pr.prediction_id IS NULL"""
        ).fetchall()
        payload = []
        for row in rows:
            winner = int(row["finish_position"] == 1)
            odds = row["result_odds"]
            payload.append((
                row["prediction_id"], row["finish_position"], winner, odds,
                row["finish_time"], float(odds) if winner and odds is not None else 0.0, stamp,
            ))
        connection.executemany(
            """INSERT INTO prediction_results(
                   prediction_id,finish_position,winner,official_odds,official_time,payout,matched_at
               ) VALUES(?,?,?,?,?,?,?)""",
            payload,
        )
        connection.commit()
        return len(payload)
    finally:
        connection.close()


def load_history(db_path: str | Path = DB) -> tuple[pd.DataFrame, pd.DataFrame]:
    connection = sqlite3.connect(str(db_path), timeout=60)
    try:
        history = pd.read_sql_query(
            """WITH latest_result AS (
                   SELECT race_id,horse_id,result_status,ROW_NUMBER() OVER(
                       PARTITION BY race_id,horse_id ORDER BY captured_at DESC,result_id DESC
                   ) AS rn FROM race_results
               )
               SELECT p.*,r.finish_position,r.winner,r.official_odds,r.official_time,
                      r.payout,r.matched_at,rr.result_status
               FROM prediction_snapshots p
               LEFT JOIN prediction_results r USING(prediction_id)
               LEFT JOIN latest_result rr ON rr.race_id=p.race_id
                    AND rr.horse_id=p.horse_id AND rr.rn=1""",
            connection,
        )
        odds = pd.DataFrame(columns=["race_id", "horse_id", "captured_at", "odds"])
    finally:
        connection.close()
    if history.empty:
        return history, odds
    history["prediction_time_parsed"] = pd.to_datetime(history["prediction_time"], utc=True)
    history["race_start_parsed"] = pd.to_datetime(history["race_start_at"], utc=True)
    history["race_date"] = history["race_start_parsed"].dt.tz_convert(
        ZoneInfo("Europe/Istanbul")
    ).dt.date.astype(str)
    return history, odds


def latest_prediction_runs(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history.copy()
    latest = history.groupby("race_id")["prediction_time_parsed"].transform("max")
    return history[history["prediction_time_parsed"].eq(latest)].copy()


def attach_certified_odds(frame: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["pre_race_odds"] = output["odds"]
    output["pre_race_odds_captured_at"] = output["prediction_time_parsed"]
    return output


def completed_races(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    quality = frame.groupby("race_id").agg(
        rows=("horse_id", "size"), matched=("winner", "count"), winners=("winner", "sum")
    )
    valid = quality[(quality["rows"] == quality["matched"]) & (quality["winners"] == 1)].index
    return frame[frame["race_id"].isin(valid)].copy()


def ece_mce(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> tuple[float, float, list[dict[str, float]]]:
    edges = np.linspace(0, 1, bins + 1)
    assigned = np.clip(np.digitize(probability, edges, right=True) - 1, 0, bins - 1)
    ece = 0.0
    gaps = []
    details = []
    for index in range(bins):
        mask = assigned == index
        predicted = float(probability[mask].mean()) if mask.any() else np.nan
        observed = float(y[mask].mean()) if mask.any() else np.nan
        gap = abs(predicted - observed) if mask.any() else np.nan
        if mask.any():
            ece += float(mask.mean()) * gap
            gaps.append(gap)
        details.append({
            "bin": index + 1, "bin_lower": edges[index], "bin_upper": edges[index + 1],
            "count": int(mask.sum()), "mean_probability": predicted,
            "observed_rate": observed, "absolute_gap": gap,
        })
    return float(ece), float(max(gaps)) if gaps else float("nan"), details


def model_metric_row(frame: pd.DataFrame, model: str, period: str, window: str) -> dict[str, object]:
    probability_col = MODEL_PROBS[model]
    race_count = frame["race_id"].nunique()
    ranks = frame.groupby("race_id")[probability_col].rank(method="first", ascending=False)
    y = frame["winner"].astype(int).to_numpy()
    probability = frame[probability_col].astype(float).to_numpy()
    winner_probability = frame.loc[frame["winner"].eq(1), probability_col]
    certified = frame.groupby("race_id")["pre_race_odds"].apply(lambda value: value.notna().all())
    certified_races = certified[certified].index
    market = frame[frame["race_id"].isin(certified_races)]
    if market.empty:
        favorite_accuracy = np.nan
    else:
        favorites = market.loc[market.groupby("race_id")["pre_race_odds"].idxmin()]
        favorite_accuracy = favorites["winner"].mean()
    top = frame.loc[ranks.eq(1)].copy()
    top_odds = pd.to_numeric(top["pre_race_odds"], errors="coerce")
    longshots = top[top_odds.ge(LONGSHOT_ODDS)]
    ece, _, _ = ece_mce(y, probability)
    try:
        auc = roc_auc_score(y, probability) if len(np.unique(y)) > 1 else np.nan
    except ValueError:
        auc = np.nan
    return {
        "metric_date": period, "window": window, "model": model,
        "rows": len(frame), "races": race_count,
        "top1_accuracy": float(frame.loc[ranks.le(1)].groupby("race_id")["winner"].max().mean()),
        "top3_accuracy": float(frame.loc[ranks.le(3)].groupby("race_id")["winner"].max().mean()),
        "top5_accuracy": float(frame.loc[ranks.le(5)].groupby("race_id")["winner"].max().mean()),
        "log_loss": float(log_loss(y, np.clip(probability, 1e-12, 1 - 1e-12), labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, probability)),
        "roc_auc": float(auc), "calibration_error": ece,
        "average_winner_probability": float(winner_probability.mean()),
        "favorite_accuracy": float(favorite_accuracy) if pd.notna(favorite_accuracy) else np.nan,
        "longshot_accuracy": float(longshots["winner"].mean()) if len(longshots) else np.nan,
        "certified_odds_races": len(certified_races), "status": "OK",
    }


def calculate_live_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    valid = completed_races(frame)
    columns = [
        "metric_date", "window", "model", "rows", "races", "top1_accuracy",
        "top3_accuracy", "top5_accuracy", "log_loss", "brier_score", "roc_auc",
        "calibration_error", "average_winner_probability", "favorite_accuracy",
        "longshot_accuracy", "certified_odds_races", "status",
    ]
    if valid.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for day, group in valid.groupby("race_date"):
        for model in MODEL_PROBS:
            rows.append(model_metric_row(group, model, day, "daily"))
    max_day = pd.to_datetime(valid["race_date"]).max()
    rolling = valid[pd.to_datetime(valid["race_date"]).ge(max_day - pd.Timedelta(days=89))]
    for model in MODEL_PROBS:
        rows.append(model_metric_row(rolling, model, max_day.date().isoformat(), "rolling_90d"))
    return pd.DataFrame(rows, columns=columns)


def distributions(reference: pd.Series, current: pd.Series, bins: np.ndarray | None = None) -> tuple[float, float, float]:
    reference = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(float)
    current = pd.to_numeric(current, errors="coerce").dropna().to_numpy(float)
    if not len(reference) or not len(current):
        return np.nan, np.nan, np.nan
    if bins is None:
        combined = np.concatenate([reference, current])
        low, high = float(np.min(combined)), float(np.max(combined))
        if low == high:
            return 0.0, 0.0, 0.0
        bins = np.linspace(low, high, 11)
    ref = np.histogram(reference, bins=bins)[0].astype(float) + 1e-9
    cur = np.histogram(current, bins=bins)[0].astype(float) + 1e-9
    ref /= ref.sum(); cur /= cur.sum()
    psi = float(np.sum((cur - ref) * np.log(cur / ref)))
    midpoint = 0.5 * (ref + cur)
    js = float(0.5 * np.sum(ref * np.log(ref / midpoint)) + 0.5 * np.sum(cur * np.log(cur / midpoint)))
    kl = float(np.sum(cur * np.log(cur / ref)))
    return psi, js, kl


def categorical_distances(reference: pd.Series, current: pd.Series) -> tuple[float, float, float, float]:
    reference = reference.fillna("__MISSING__").astype(str)
    current = current.fillna("__MISSING__").astype(str)
    categories = sorted(set(reference) | set(current))
    ref = reference.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy(float) + 1e-9
    cur = current.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy(float) + 1e-9
    ref /= ref.sum(); cur /= cur.sum(); midpoint = 0.5 * (ref + cur)
    psi = float(np.sum((cur - ref) * np.log(cur / ref)))
    js = float(0.5 * np.sum(ref * np.log(ref / midpoint)) + 0.5 * np.sum(cur * np.log(cur / midpoint)))
    kl = float(np.sum(cur * np.log(cur / ref)))
    unseen = float((~current.isin(set(reference))).mean())
    return psi, js, kl, unseen


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["prediction_id", "race_date", *MODEL_FEATURES])
    values = pd.DataFrame([json.loads(value) for value in frame["feature_values_json"]])
    values.insert(0, "race_date", frame["race_date"].to_numpy())
    values.insert(0, "prediction_id", frame["prediction_id"].to_numpy())
    return values


def feature_drift(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    features = feature_frame(frame)
    columns = [
        "as_of_date", "feature", "feature_type", "current_rows", "reference_rows",
        "current_mean", "reference_mean", "current_median", "reference_median",
        "current_std", "reference_std", "current_min", "reference_min",
        "current_max", "reference_max", "current_missing_rate", "reference_missing_rate",
        "missing_rate_delta", "unseen_category_rate", "psi", "js_distance",
        "kl_divergence", "status",
    ]
    if features.empty:
        return pd.DataFrame(columns=columns), "INSUFFICIENT_DATA"
    as_of = pd.to_datetime(features["race_date"]).max()
    current = features[pd.to_datetime(features["race_date"]).eq(as_of)]
    reference = features[
        pd.to_datetime(features["race_date"]).between(as_of - pd.Timedelta(days=30), as_of - pd.Timedelta(days=1))
    ]
    rows = []
    overall = "INSUFFICIENT_DATA" if len(current) < MIN_CURRENT_ROWS or len(reference) < MIN_REFERENCE_ROWS else "PASS"
    for feature in MODEL_FEATURES:
        cur, ref = current[feature], reference[feature]
        base = {
            "as_of_date": as_of.date().isoformat(), "feature": feature,
            "feature_type": "categorical" if feature in CATEGORICAL_FEATURES else "numeric",
            "current_rows": len(cur), "reference_rows": len(ref),
            "current_missing_rate": cur.isna().mean() if len(cur) else np.nan,
            "reference_missing_rate": ref.isna().mean() if len(ref) else np.nan,
        }
        base["missing_rate_delta"] = abs(base["current_missing_rate"] - base["reference_missing_rate"]) if len(ref) else np.nan
        if feature in CATEGORICAL_FEATURES:
            psi, js, kl, unseen = categorical_distances(ref, cur) if len(ref) and len(cur) else (np.nan,) * 4
            base.update({
                "current_mean": np.nan, "reference_mean": np.nan,
                "current_median": np.nan, "reference_median": np.nan,
                "current_std": np.nan, "reference_std": np.nan,
                "current_min": np.nan, "reference_min": np.nan,
                "current_max": np.nan, "reference_max": np.nan,
                "unseen_category_rate": unseen,
            })
        else:
            cur_num, ref_num = pd.to_numeric(cur, errors="coerce"), pd.to_numeric(ref, errors="coerce")
            psi, js, kl = distributions(ref_num, cur_num)
            base.update({
                "current_mean": cur_num.mean(), "reference_mean": ref_num.mean(),
                "current_median": cur_num.median(), "reference_median": ref_num.median(),
                "current_std": cur_num.std(), "reference_std": ref_num.std(),
                "current_min": cur_num.min(), "reference_min": ref_num.min(),
                "current_max": cur_num.max(), "reference_max": ref_num.max(),
                "unseen_category_rate": np.nan,
            })
        enough = len(cur) >= MIN_CURRENT_ROWS and len(ref) >= MIN_REFERENCE_ROWS
        critical = enough and (
            psi >= THRESHOLDS["psi"] or js >= THRESHOLDS["js"] or kl >= THRESHOLDS["kl"]
            or base["missing_rate_delta"] >= THRESHOLDS["missing_rate_delta"]
            or (pd.notna(base["unseen_category_rate"]) and base["unseen_category_rate"] >= THRESHOLDS["unseen_category_rate"])
        )
        status = "CRITICAL" if critical else "PASS" if enough else "INSUFFICIENT_DATA"
        if status == "CRITICAL": overall = "CRITICAL"
        base.update({"psi": psi, "js_distance": js, "kl_divergence": kl, "status": status})
        rows.append(base)
    return pd.DataFrame(rows, columns=columns), overall


def model_drift(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    columns = [
        "as_of_date", "drift_type", "model", "current_rows", "reference_rows",
        "psi", "js_distance", "kl_divergence", "winner_probability_shift",
        "confidence_shift", "winner_rate_shift", "class_js_distance",
        "scratch_rate_shift", "status",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns), "INSUFFICIENT_DATA"
    as_of = pd.to_datetime(frame["race_date"]).max()
    current = frame[pd.to_datetime(frame["race_date"]).eq(as_of)]
    reference = frame[pd.to_datetime(frame["race_date"]).between(as_of - pd.Timedelta(days=30), as_of - pd.Timedelta(days=1))]
    rows = []
    overall = "INSUFFICIENT_DATA" if len(current) < MIN_CURRENT_ROWS or len(reference) < MIN_REFERENCE_ROWS else "PASS"
    for model, probability_col in MODEL_PROBS.items():
        psi, js, kl = distributions(reference[probability_col], current[probability_col], np.linspace(0, 1, 11))
        current_winner = current.loc[current["winner"].eq(1), probability_col].mean()
        ref_winner = reference.loc[reference["winner"].eq(1), probability_col].mean()
        cur_conf = current.groupby("race_id")[probability_col].max().mean()
        ref_conf = reference.groupby("race_id")[probability_col].max().mean()
        winner_shift = abs(current_winner - ref_winner) if pd.notna(current_winner) and pd.notna(ref_winner) else np.nan
        confidence_shift = abs(cur_conf - ref_conf) if pd.notna(cur_conf) and pd.notna(ref_conf) else np.nan
        enough = len(current) >= MIN_CURRENT_ROWS and len(reference) >= MIN_REFERENCE_ROWS
        critical = enough and (
            psi >= THRESHOLDS["psi"] or js >= THRESHOLDS["js"] or kl >= THRESHOLDS["kl"]
            or (pd.notna(confidence_shift) and confidence_shift >= THRESHOLDS["confidence_shift"])
        )
        status = "CRITICAL" if critical else "PASS" if enough else "INSUFFICIENT_DATA"
        if status == "CRITICAL": overall = "CRITICAL"
        rows.append({
            "as_of_date": as_of.date().isoformat(), "drift_type": "prediction",
            "model": model, "current_rows": len(current), "reference_rows": len(reference),
            "psi": psi, "js_distance": js, "kl_divergence": kl,
            "winner_probability_shift": winner_shift, "confidence_shift": confidence_shift,
            "winner_rate_shift": np.nan, "class_js_distance": np.nan,
            "scratch_rate_shift": np.nan, "status": status,
        })
    completed_current, completed_ref = completed_races(current), completed_races(reference)
    winner_rate_shift = abs(completed_current["winner"].mean() - completed_ref["winner"].mean()) if len(completed_current) and len(completed_ref) else np.nan
    current_features, ref_features = feature_frame(current), feature_frame(reference)
    _, class_js, _, _ = categorical_distances(ref_features.get("race_class", pd.Series(dtype=str)), current_features.get("race_class", pd.Series(dtype=str))) if len(ref_features) and len(current_features) else (np.nan,) * 4
    current_scratch = current["result_status"].eq("scratched").mean() if "result_status" in current else np.nan
    ref_scratch = reference["result_status"].eq("scratched").mean() if "result_status" in reference else np.nan
    scratch_shift = abs(current_scratch - ref_scratch) if pd.notna(current_scratch) and pd.notna(ref_scratch) else np.nan
    target_enough = len(completed_current) >= MIN_CURRENT_ROWS and len(completed_ref) >= MIN_REFERENCE_ROWS
    target_critical = target_enough and (
        (pd.notna(winner_rate_shift) and winner_rate_shift >= THRESHOLDS["target_rate_shift"])
        or (pd.notna(class_js) and class_js >= THRESHOLDS["js"])
        or (pd.notna(scratch_shift) and scratch_shift >= THRESHOLDS["target_rate_shift"])
    )
    target_status = "CRITICAL" if target_critical else "PASS" if target_enough else "INSUFFICIENT_DATA"
    if target_status == "CRITICAL":
        overall = "CRITICAL"
    rows.append({
        "as_of_date": as_of.date().isoformat(), "drift_type": "target", "model": "all",
        "current_rows": len(completed_current), "reference_rows": len(completed_ref),
        "psi": np.nan, "js_distance": np.nan, "kl_divergence": np.nan,
        "winner_probability_shift": np.nan, "confidence_shift": np.nan,
        "winner_rate_shift": winner_rate_shift, "class_js_distance": class_js,
        "scratch_rate_shift": scratch_shift, "status": target_status,
    })
    return pd.DataFrame(rows, columns=columns), overall


def calibration_history(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    valid = completed_races(frame)
    columns = [
        "week", "model", "bin", "bin_lower", "bin_upper", "count",
        "mean_probability", "observed_rate", "absolute_gap", "ece", "mce", "status",
    ]
    if valid.empty:
        return pd.DataFrame(columns=columns), "INSUFFICIENT_DATA"
    dates = pd.to_datetime(valid["race_date"])
    valid = valid.assign(week=dates.dt.to_period("W-SUN").astype(str))
    rows = []
    overall = "PASS"
    for week, group in valid.groupby("week"):
        for model, probability_col in MODEL_PROBS.items():
            ece, mce, bins = ece_mce(group["winner"].astype(int).to_numpy(), group[probability_col].to_numpy(float))
            enough = group["race_id"].nunique() >= MIN_CALIBRATION_RACES
            status = "CRITICAL" if enough and ece >= THRESHOLDS["ece"] else "PASS" if enough else "INSUFFICIENT_DATA"
            if status == "CRITICAL": overall = "CRITICAL"
            elif status == "INSUFFICIENT_DATA" and overall != "CRITICAL": overall = "INSUFFICIENT_DATA"
            for detail in bins:
                rows.append({"week": week, "model": model, **detail, "ece": ece, "mce": mce, "status": status})
    return pd.DataFrame(rows, columns=columns), overall


def roi_report_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    valid = completed_races(frame)
    columns = ["strategy", "bets", "stake", "payout", "profit", "roi", "status"]
    if valid.empty:
        return pd.DataFrame(columns=columns), "NOT CERTIFIED"
    coverage = valid.groupby("race_id")["pre_race_odds"].apply(lambda values: values.notna().all())
    certified = valid[valid["race_id"].isin(coverage[coverage].index)]
    if certified.empty:
        return pd.DataFrame(columns=columns), "NOT CERTIFIED"
    selections = {}
    ensemble_rank = certified.groupby("race_id")["ensemble_probability"].rank(method="first", ascending=False)
    selections["flat_betting"] = certified[ensemble_rank.eq(1)].assign(stake=1.0)
    selections["favorite"] = certified.loc[certified.groupby("race_id")["pre_race_odds"].idxmin()].assign(stake=1.0)
    value = certified[(certified["ensemble_probability"] * certified["pre_race_odds"]).ge(1.05)].copy()
    selections["value_bet"] = value.assign(stake=1.0)
    longshot_pool = certified[certified["pre_race_odds"].ge(LONGSHOT_ODDS)]
    selections["longshot"] = longshot_pool.loc[longshot_pool.groupby("race_id")["ensemble_probability"].idxmax()].assign(stake=1.0) if len(longshot_pool) else longshot_pool.assign(stake=pd.Series(dtype=float))
    kelly = certified.copy()
    b = kelly["pre_race_odds"] - 1
    q = 1 - kelly["ensemble_probability"]
    kelly["stake"] = (((b * kelly["ensemble_probability"] - q) / b.replace(0, np.nan)).clip(0, 0.25)).fillna(0)
    selections["kelly"] = kelly[kelly["stake"].gt(0)]
    rows = []
    for strategy, selected in selections.items():
        if selected.empty:
            rows.append({"strategy": strategy, "bets": 0, "stake": 0.0, "payout": 0.0, "profit": 0.0, "roi": np.nan, "status": "CERTIFIED_NO_BETS"})
            continue
        payout = (selected["stake"] * selected["pre_race_odds"] * selected["winner"]).sum()
        stake = selected["stake"].sum(); profit = payout - stake
        rows.append({"strategy": strategy, "bets": len(selected), "stake": stake, "payout": payout, "profit": profit, "roi": profit / stake if stake else np.nan, "status": "CERTIFIED"})
    return pd.DataFrame(rows, columns=columns), "CERTIFIED"


def verify_feature_hashes(frame: pd.DataFrame) -> bool:
    for _, row in frame.iterrows():
        features = json.loads(row["feature_values_json"])
        payload = {
            "feature_contract_version": row["feature_contract_version"],
            "snapshot_id": int(row["feature_snapshot_id"]),
            "source_request_id": str(row["source_request_id"]),
            "race_start_at": str(row["race_start_at"]),
            "features": features,
        }
        actual = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        if actual != row["feature_hash"]:
            return False
    return True


def snapshot_coverage_pass(db_path: str | Path, history: pd.DataFrame) -> tuple[bool, list[str]]:
    if history.empty:
        archive_integrity = True
    else:
        archive_integrity = bool((history["prediction_time_parsed"] < history["race_start_parsed"]).all())
    connection = sqlite3.connect(str(db_path), timeout=60)
    try:
        bad = connection.execute(
            """SELECT COUNT(*) FROM prediction_snapshots p
               LEFT JOIN program_snapshots s ON s.snapshot_id=p.feature_snapshot_id
               WHERE s.snapshot_id IS NULL OR s.race_id!=p.race_id OR s.horse_id!=p.horse_id
                  OR s.source_request_id!=p.source_request_id
                  OR julianday(s.captured_at)>=julianday(p.race_start_at)"""
        ).fetchone()[0]
        inception_row = connection.execute(
            "SELECT applied_at FROM schema_migrations WHERE migration_name='007_prediction_snapshots.sql'"
        ).fetchone()
    finally:
        connection.close()
    missed: list[str] = []
    feature_path = ROOT / "output" / "asof_features.parquet"
    if inception_row and feature_path.exists():
        feature_rows = pd.read_parquet(feature_path, columns=["race_id", "race_start_at"])
        starts = pd.to_datetime(feature_rows["race_start_at"], utc=True, errors="coerce")
        local_days = starts.dt.tz_convert(ZoneInfo("Europe/Istanbul")).dt.date
        inception = pd.Timestamp(inception_row[0])
        today = datetime.now(ZoneInfo("Europe/Istanbul")).date()
        candidates = set(feature_rows.loc[starts.gt(inception) & local_days.le(today), "race_id"])
        predicted = set(history["race_id"]) if not history.empty else set()
        missed = sorted(candidates - predicted)
    return archive_integrity and bad == 0 and verify_feature_hashes(history) and not missed, missed


def markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 100) -> str:
    if frame.empty:
        return "_No observations._"
    display = frame[columns].head(limit).copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(lambda value: "N/A" if pd.isna(value) else f"{value:.4f}")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def write_calibration_plot(calibration: pd.DataFrame) -> None:
    path = REPORTS / "live_calibration_curve.png"
    plt.figure(figsize=(8, 7))
    if not calibration.empty:
        latest_week = calibration["week"].max()
        for model in MODEL_PROBS:
            subset = calibration[(calibration["week"] == latest_week) & (calibration["model"] == model) & calibration["count"].gt(0)]
            plt.plot(subset["mean_probability"], subset["observed_rate"], marker="o", label=model)
    plt.plot([0, 1], [0, 1], "--", color="black", label="perfect")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Observed winner rate")
    plt.title("Live weekly reliability diagram"); plt.grid(alpha=0.25); plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()


def archive_monitoring_run(
    db_path: str | Path, checks: dict[str, object], shadow_days: int,
    healthy_shadow_days: int, monitor_date: str, production_ready: bool
) -> None:
    stamp = datetime.now(timezone.utc).isoformat()
    connection = sqlite3.connect(str(db_path), timeout=60)
    try:
        connection.execute(
            """INSERT INTO shadow_monitoring_runs(
                   run_id,run_at,shadow_date,leakage_gate_pass,feature_contract_pass,
                   snapshot_coverage_pass,prediction_drift_status,calibration_status,
                   feature_drift_status,pipeline_status,production_ready,details_json)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (uuid.uuid4().hex, stamp, monitor_date, int(checks["leakage_gate_pass"]),
             int(checks["feature_contract_pass"]), int(checks["snapshot_coverage_pass"]),
             checks["prediction_drift_status"], checks["calibration_status"],
             checks["feature_drift_status"], checks["pipeline_status"],
             int(production_ready), json.dumps({"shadow_days": shadow_days, "healthy_shadow_days": healthy_shadow_days, **checks}, sort_keys=True)),
        )
        connection.commit()
    finally:
        connection.close()


def main() -> int:
    OUTPUT.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True); apply_migrations(DB)
    matched = match_prediction_results(DB)
    export_prediction_history(DB)
    history, odds = load_history(DB)
    latest = attach_certified_odds(latest_prediction_runs(history), odds)
    metrics = calculate_live_metrics(latest)
    drift, prediction_drift_status = model_drift(latest)
    features, feature_drift_status = feature_drift(latest)
    calibration, calibration_status = calibration_history(latest)
    roi, roi_status = roi_report_data(latest)
    metrics.to_csv(OUTPUT / "live_metrics.csv", index=False, encoding="utf-8")
    drift.to_csv(OUTPUT / "model_drift.csv", index=False, encoding="utf-8")
    calibration.to_csv(OUTPUT / "calibration_history.csv", index=False, encoding="utf-8")
    features.to_csv(OUTPUT / "feature_drift.csv", index=False, encoding="utf-8")
    write_calibration_plot(calibration)

    provenance = validate_provenance(DB)
    try:
        validate_model_feature_contract(MODEL_FEATURES)
        contract_pass = True
    except ValueError:
        contract_pass = False
    coverage_pass, missed_races = snapshot_coverage_pass(DB, history)
    critical = (
        not provenance["passed"] or not contract_pass or not coverage_pass
        or prediction_drift_status == "CRITICAL" or calibration_status == "CRITICAL"
        or feature_drift_status == "CRITICAL"
    )
    warmup = any(status == "INSUFFICIENT_DATA" for status in (
        prediction_drift_status, calibration_status, feature_drift_status
    ))
    pipeline_status = "FAIL" if critical else "SHADOW_WARMUP" if warmup else "PASS"
    completed = completed_races(latest)
    completed_dates = set(completed["race_date"].unique()) if len(completed) else set()
    shadow_days = len(completed_dates)
    monitor_date = max(completed_dates) if completed_dates else datetime.now(ZoneInfo("Europe/Istanbul")).date().isoformat()
    connection = sqlite3.connect(str(DB), timeout=60)
    try:
        healthy_dates = {
            row[0] for row in connection.execute(
                """SELECT DISTINCT shadow_date FROM shadow_monitoring_runs
                   WHERE leakage_gate_pass=1 AND feature_contract_pass=1
                     AND snapshot_coverage_pass=1 AND pipeline_status!='FAIL'"""
            )
        }
    finally:
        connection.close()
    if not critical and completed_dates:
        healthy_dates.add(monitor_date)
    healthy_shadow_days = len(completed_dates & healthy_dates)
    production_ready = bool(
        shadow_days >= 90 and healthy_shadow_days >= 90 and not critical and not warmup
        and not metrics.empty and metrics[metrics["window"].eq("rolling_90d")]["status"].eq("OK").all()
    )
    checks = {
        "leakage_gate_pass": bool(provenance["passed"]),
        "feature_contract_pass": contract_pass,
        "snapshot_coverage_pass": coverage_pass,
        "prediction_drift_status": prediction_drift_status,
        "calibration_status": calibration_status,
        "feature_drift_status": feature_drift_status,
        "pipeline_status": pipeline_status,
        "missed_shadow_races": len(missed_races),
    }
    archive_monitoring_run(DB, checks, shadow_days, healthy_shadow_days, monitor_date, production_ready)
    stamp = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    latest_daily = metrics[metrics["window"].eq("daily")]
    if not latest_daily.empty:
        latest_daily = latest_daily[latest_daily["metric_date"].eq(latest_daily["metric_date"].max())]
    (REPORTS / "daily_shadow_report.md").write_text(
        f"# Daily Shadow Report\n\nGenerated: {stamp}\n\n- Mode: `shadow_mode`\n- Pipeline status: **{pipeline_status}**\n- Archived predictions: **{len(history)}**\n- Latest-run races: **{latest['race_id'].nunique() if len(latest) else 0}**\n- Missed eligible races since shadow inception: **{len(missed_races)}**\n- Newly matched results: **{matched}**\n- Completed shadow days: **{shadow_days} / 90**\n- Model retraining: **disabled**\n\n"
        + markdown_table(latest_daily, ["model", "races", "top1_accuracy", "top3_accuracy", "top5_accuracy", "log_loss", "brier_score", "roc_auc", "calibration_error"]),
        encoding="utf-8",
    )
    (REPORTS / "live_accuracy_report.md").write_text(
        f"# Live Accuracy Report\n\nGenerated: {stamp}\n\nModels are compared on identical completed races; no model is selected automatically.\n\n"
        + markdown_table(metrics, ["metric_date", "window", "model", "races", "top1_accuracy", "top3_accuracy", "top5_accuracy", "log_loss", "brier_score", "roc_auc", "calibration_error", "average_winner_probability", "favorite_accuracy", "longshot_accuracy"]),
        encoding="utf-8",
    )
    (REPORTS / "model_drift_report.md").write_text(
        f"# Model Drift Report\n\nGenerated: {stamp}\n\nOverall prediction drift: **{prediction_drift_status}**. Reference window is the previous 30 calendar days.\n\n"
        + markdown_table(drift, ["drift_type", "model", "current_rows", "reference_rows", "psi", "js_distance", "kl_divergence", "winner_probability_shift", "confidence_shift", "winner_rate_shift", "class_js_distance", "scratch_rate_shift", "status"]),
        encoding="utf-8",
    )
    (REPORTS / "feature_drift_report.md").write_text(
        f"# Feature Drift Report\n\nGenerated: {stamp}\n\nOverall feature drift: **{feature_drift_status}**.\n\n"
        + markdown_table(features, ["feature", "feature_type", "current_rows", "reference_rows", "current_mean", "reference_mean", "current_median", "reference_median", "current_std", "reference_std", "current_min", "reference_min", "current_max", "reference_max", "current_missing_rate", "reference_missing_rate", "unseen_category_rate", "psi", "js_distance", "kl_divergence", "status"]),
        encoding="utf-8",
    )
    calibration_summary = calibration.drop_duplicates(["week", "model"])[["week", "model", "ece", "mce", "status"]] if len(calibration) else calibration
    (REPORTS / "calibration_monitor.md").write_text(
        f"# Calibration Monitor\n\nGenerated: {stamp}\n\nOverall calibration: **{calibration_status}**. ECE critical threshold: `{THRESHOLDS['ece']}`.\n\n"
        + markdown_table(calibration_summary, list(calibration_summary.columns) if len(calibration_summary.columns) else ["week"])
        + "\n\nReliability diagram: `reports/live_calibration_curve.png`.\n",
        encoding="utf-8",
    )
    (REPORTS / "live_roi_report.md").write_text(
        f"# Live ROI Report\n\nGenerated: {stamp}\n\nROI certification: **{roi_status}**. Only odds snapshots captured before both prediction and race start are admissible.\n\n"
        + (markdown_table(roi, ["strategy", "bets", "stake", "payout", "profit", "roi", "status"]) if roi_status == "CERTIFIED" else "`ROI = NOT CERTIFIED`"),
        encoding="utf-8",
    )
    (REPORTS / "model_health_dashboard.md").write_text(
        f"# Model Health Dashboard\n\nGenerated: {stamp}\n\n## Status: **{pipeline_status}**\n\n- Production ready: **{'YES' if production_ready else 'NO'}**\n- Shadow days: **{shadow_days} / 90**\n- Healthy gate days: **{healthy_shadow_days} / 90**\n- Leakage gate: **{'PASS' if provenance['passed'] else 'FAIL'}**\n- Feature contract: **{'PASS' if contract_pass else 'FAIL'}**\n- Snapshot coverage: **{'PASS' if coverage_pass else 'FAIL'}**\n- Prediction drift: **{prediction_drift_status}**\n- Feature drift: **{feature_drift_status}**\n- Calibration: **{calibration_status}**\n- ROI: **{roi_status}**\n\nProduction readiness cannot be granted before 90 completed, healthy shadow dates.\n",
        encoding="utf-8",
    )
    print({"pipeline_status": pipeline_status, "production_ready": production_ready, "shadow_days": shadow_days, "matched": matched, "checks": checks})
    return 1 if critical else 0


if __name__ == "__main__":
    raise SystemExit(main())
