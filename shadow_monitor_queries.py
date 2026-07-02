"""Read-only 60-90 day analytics for the live shadow prediction system."""
from __future__ import annotations

import csv
import io
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
from statistics import mean, median
from typing import Any

from diagnostics_queries import DIAGNOSTICS_CTE
from performance_queries import MODELS
from race_scope import configure_sqlite

MODEL_ORDER = ("Ensemble", "Logistic", "CatBoost", "XGBoost")


def normalize_shadow_filters(
    date_from: str, date_to: str, track: str | None = None,
    model: str | None = None, distance: str | None = None,
    surface: str | None = None, field_size: str | None = None,
    evaluated_only: bool = True, odds_only: bool = False,
) -> dict[str, Any]:
    for label, value in (("date_from", date_from), ("date_to", date_to)):
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"{label} must be YYYY-MM-DD") from exc
    if date_from > date_to:
        raise ValueError("date_from must be on or before date_to")
    selected = model or ""
    if selected and selected not in MODELS:
        raise ValueError(f"model must be one of: {', '.join(MODELS)}")
    return {
        "date_from": date_from, "date_to": date_to, "track": track or None,
        "model": selected or None, "distance": distance or None,
        "surface": surface or None, "field_size": field_size or None,
        "evaluated_only": bool(evaluated_only), "odds_only": bool(odds_only),
    }


def _selected_models(filters: dict[str, Any]) -> list[str]:
    return [filters["model"]] if filters.get("model") else list(MODEL_ORDER)


def _load_rows(connection: sqlite3.Connection, filters: dict[str, Any]) -> list[dict[str, Any]]:
    configure_sqlite(connection)
    clauses = ["d.race_date BETWEEN ? AND ?"]
    params: list[Any] = [filters["date_from"], filters["date_to"]]
    columns = {
        "track": "d.track", "distance": "d.distance_group",
        "surface": "d.surface_group", "field_size": "d.field_size_group",
    }
    for key, column in columns.items():
        if filters.get(key):
            clauses.append(f"track_key({column})=track_key(?)" if key == "track" else f"{column}=?")
            params.append(filters[key])
    where = " AND ".join(clauses)
    rows: list[dict[str, Any]] = []
    for model in _selected_models(filters):
        query = DIAGNOSTICS_CTE + f"""
        SELECT d.*,COALESCE(pr.official_odds,predicted_result.result_odds) AS bet_official_odds
        FROM diagnostics d
        LEFT JOIN prediction_snapshots top_snapshot
          ON top_snapshot.race_id=d.race_id
         AND top_snapshot.prediction_time=d.prediction_time
         AND top_snapshot.horse_id=d.top1_horse_id
        LEFT JOIN prediction_results pr ON pr.prediction_id=top_snapshot.prediction_id
        LEFT JOIN results predicted_result
          ON predicted_result.race_id=d.race_id
         AND predicted_result.horse_id=d.top1_horse_id
        WHERE {where}
        ORDER BY d.race_start_at,d.race_id"""
        for row in connection.execute(query, [model, *params]):
            item = dict(row)
            if filters["odds_only"] and item["bet_official_odds"] is None:
                continue
            rows.append(item)
    return rows


def _metrics(rows: list[dict[str, Any]], stake: float = 1.0) -> dict[str, Any]:
    ranks = [int(row["winner_rank"]) for row in rows]
    odds_rows = [row for row in rows if row["bet_official_odds"] is not None]
    net_profit = sum(
        stake * (float(row["bet_official_odds"]) - 1.0)
        if int(row["winner_rank"]) == 1 else -stake
        for row in odds_rows
    )
    count = len(rows)
    model_over = sum(
        row["winner_agf_rank"] is not None
        and int(row["winner_rank"]) < int(row["winner_agf_rank"])
        for row in rows
    )
    agf_over = sum(
        row["winner_agf_rank"] is not None
        and int(row["winner_agf_rank"]) < int(row["winner_rank"])
        for row in rows
    )
    tied = sum(
        row["winner_agf_rank"] is not None
        and int(row["winner_agf_rank"]) == int(row["winner_rank"])
        for row in rows
    )
    return {
        "evaluated_race_count": count,
        "top1_accuracy": 100.0 * sum(rank == 1 for rank in ranks) / count if count else 0.0,
        "top3_accuracy": 100.0 * sum(rank <= 3 for rank in ranks) / count if count else 0.0,
        "top5_accuracy": 100.0 * sum(rank <= 5 for rank in ranks) / count if count else 0.0,
        "average_winner_rank": mean(ranks) if ranks else 0.0,
        "median_winner_rank": median(ranks) if ranks else 0.0,
        "roi_bet_count": len(odds_rows),
        "roi_percent": 100.0 * net_profit / (stake * len(odds_rows)) if odds_rows else 0.0,
        "net_profit": net_profit,
        "model_over_agf": model_over, "agf_over_model": agf_over, "tied_with_agf": tied,
    }


def _group(rows: list[dict[str, Any]], key, label: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(key(row) or "Bilinmiyor")].append(row)
    return [
        {"group": group, "dimension": label, **_metrics(items)}
        for group, items in sorted(grouped.items())
    ]


def models(connection: sqlite3.Connection, filters: dict[str, Any]) -> dict[str, Any]:
    rows = _load_rows(connection, filters)
    return {
        "models": [
            {"model": model, **_metrics([row for row in rows if row["model"] == model])}
            for model in _selected_models(filters)
        ]
    }


def segments(connection: sqlite3.Connection, filters: dict[str, Any]) -> dict[str, Any]:
    rows = _load_rows(connection, filters)

    def odds_band(row):
        value = row["bet_official_odds"]
        if value is None: return "Odds yok"
        value = float(value)
        if value < 2: return "<2"
        if value < 4: return "2-3.99"
        if value < 8: return "4-7.99"
        return "8+"

    def agf_band(row):
        value = row["winner_agf_rank"]
        if value is None: return "AGF yok"
        value = int(value)
        if value == 1: return "1"
        if value <= 3: return "2-3"
        if value <= 6: return "4-6"
        return "7+"

    return {
        "track": _group(rows, lambda row: row["track"], "Pist"),
        "distance": _group(rows, lambda row: row["distance_group"], "Mesafe"),
        "surface": _group(rows, lambda row: row["surface_group"], "Zemin"),
        "field_size": _group(rows, lambda row: row["field_size_group"], "At Sayısı"),
        "race_type": _group(rows, lambda row: row["race_type_group"], "Yarış Tipi"),
        "odds": _group(rows, odds_band, "Odds Aralığı"),
        "agf_rank": _group(rows, agf_band, "AGF Rank"),
    }


def trends(connection: sqlite3.Connection, filters: dict[str, Any]) -> dict[str, Any]:
    rows = _load_rows(connection, filters)
    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_model_date: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_date[row["race_date"]].append(row)
        by_model_date[(row["model"], row["race_date"])].append(row)
    dates = sorted(by_date)
    daily = []
    for date in dates:
        metrics = _metrics(by_date[date])
        daily.append({"date": date, **metrics})
    for item in daily:
        current = datetime.strptime(item["date"], "%Y-%m-%d")
        for days in (7, 30):
            start = (current - timedelta(days=days - 1)).strftime("%Y-%m-%d")
            rolling = [row for date in dates if start <= date <= item["date"] for row in by_date[date]]
            item[f"rolling_{days}_top1_accuracy"] = _metrics(rolling)["top1_accuracy"]
    model_rank = [
        {"model": model, "date": date,
         "average_winner_rank": _metrics(items)["average_winner_rank"]}
        for (model, date), items in sorted(by_model_date.items())
    ]
    return {"daily": daily, "model_winner_rank": model_rank}


def health(connection: sqlite3.Connection, filters: dict[str, Any]) -> dict[str, Any]:
    configure_sqlite(connection)
    date_params = [filters["date_from"], filters["date_to"]]
    program_races = connection.execute(
        """SELECT COUNT(DISTINCT race_id) FROM program_snapshots
           WHERE date(race_start_at,'+3 hours') BETWEEN ? AND ?""", date_params
    ).fetchone()[0]
    predicted_races = connection.execute(
        """SELECT COUNT(DISTINCT race_id) FROM prediction_snapshots
           WHERE date(race_start_at,'+3 hours') BETWEEN ? AND ?
             AND julianday(prediction_time)<julianday(race_start_at)""", date_params
    ).fetchone()[0]
    result_races = connection.execute(
        """SELECT COUNT(DISTINCT race_id) FROM race_results
           WHERE date(race_start_at,'+3 hours') BETWEEN ? AND ?
             AND result_status='finished' AND finish_position=1""", date_params
    ).fetchone()[0]
    rows = _load_rows(connection, filters)
    monitoring = connection.execute(
        "SELECT * FROM shadow_monitoring_runs ORDER BY run_at DESC LIMIT 1"
    ).fetchone()
    monitoring_data = dict(monitoring) if monitoring else {}
    feature_rows = connection.execute(
        """WITH latest AS (
             SELECT race_id,horse_id,MAX(prediction_time) prediction_time
             FROM prediction_snapshots
             WHERE date(race_start_at,'+3 hours') BETWEEN ? AND ?
               AND julianday(prediction_time)<julianday(race_start_at)
             GROUP BY race_id,horse_id)
           SELECT p.feature_values_json FROM prediction_snapshots p JOIN latest l
             ON l.race_id=p.race_id AND l.horse_id=p.horse_id
            AND l.prediction_time=p.prediction_time""", date_params
    ).fetchall()
    feature_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in feature_rows:
        try: values = json.loads(row[0] or "{}")
        except (TypeError, json.JSONDecodeError): values = {}
        for key, value in values.items():
            feature_counts[key][1] += 1
            feature_counts[key][0] += int(value is None)
    null_rates = [
        {"feature": key, "null_rate": 100.0 * nulls / total if total else 0.0}
        for key, (nulls, total) in feature_counts.items()
    ]
    null_rates.sort(key=lambda row: (-row["null_rate"], row["feature"]))
    lifecycle = [
        {"status": row[0], "count": row[1]}
        for row in connection.execute(
            """SELECT status,COUNT(*) FROM race_prediction_lifecycle
               WHERE date(race_start_at,'+3 hours') BETWEEN ? AND ?
               GROUP BY status ORDER BY status""", date_params
        )
    ]
    return {
        "prediction_missing_races": max(0, int(program_races) - int(predicted_races)),
        "result_unmatched_races": max(0, int(predicted_races) - int(result_races)),
        "missing_odds_count": sum(row["bet_official_odds"] is None for row in rows),
        "missing_agf_count": sum(row["winner_agf_rank"] is None for row in rows),
        "feature_null_rates": null_rates[:25],
        "drift_status": {
            "prediction": monitoring_data.get("prediction_drift_status", "UNKNOWN"),
            "feature": monitoring_data.get("feature_drift_status", "UNKNOWN"),
        },
        "snapshot_coverage": monitoring_data.get("snapshot_coverage_pass"),
        "pipeline_status": monitoring_data.get("pipeline_status", "UNKNOWN"),
        "latest_monitoring_run": monitoring_data.get("run_at"),
        "lifecycle_statuses": lifecycle,
    }


def summary(connection: sqlite3.Connection, filters: dict[str, Any]) -> dict[str, Any]:
    rows = _load_rows(connection, filters)
    overall = _metrics(rows)
    model_rows = models(connection, filters)["models"]
    track_rows = _group(rows, lambda row: row["track"], "Pist")
    best_model = max(model_rows, key=lambda row: row["top1_accuracy"], default=None)
    best_track = max(track_rows, key=lambda row: row["top1_accuracy"], default=None)
    worst_track = min(track_rows, key=lambda row: row["top1_accuracy"], default=None)
    health_data = health(connection, filters)
    production_ready = (
        health_data["pipeline_status"] != "FAIL"
        and health_data["snapshot_coverage"] in (1, True)
    )
    return {
        **overall,
        "live_prediction_days": len({row["race_date"] for row in rows}),
        "total_evaluated_races": len({row["race_id"] for row in rows}),
        "total_predictions": len(rows),
        "best_model": best_model["model"] if best_model else None,
        "best_track": best_track["group"] if best_track else None,
        "worst_track": worst_track["group"] if worst_track else None,
        "production_ready": production_ready,
    }


def export_csv(connection: sqlite3.Connection, filters: dict[str, Any]) -> str:
    rows = _load_rows(connection, filters)
    fields = [
        "race_date", "race_id", "track", "race_no", "model", "top1_horse",
        "winner_horse", "winner_rank", "winner_agf_rank", "bet_official_odds",
        "surface_group", "distance_group", "field_size_group", "race_type_group",
    ]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()
