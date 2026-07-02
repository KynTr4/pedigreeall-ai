"""Read-only SQL query layer for prediction performance reporting."""
from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Any
from race_scope import configure_sqlite

MODELS = ("Logistic", "CatBoost", "XGBoost", "Ensemble")
OUTCOMES = ("all", "correct", "incorrect")
PAGE_SIZE = 100

PERFORMANCE_CTE = """
WITH model_names(model) AS (
    SELECT 'Logistic' UNION ALL SELECT 'CatBoost'
    UNION ALL SELECT 'XGBoost' UNION ALL SELECT 'Ensemble'
),
latest_runs AS (
    SELECT race_id,MAX(prediction_time) AS prediction_time
    FROM prediction_snapshots
    WHERE julianday(prediction_time)<julianday(race_start_at)
    GROUP BY race_id
),
run_maxima AS (
    SELECT p.race_id,p.prediction_time,
           MAX(logistic_probability) AS logistic_max,
           MAX(catboost_probability) AS catboost_max,
           MAX(xgboost_probability) AS xgboost_max,
           MAX(ensemble_probability) AS ensemble_max
    FROM prediction_snapshots p JOIN latest_runs l
      ON l.race_id=p.race_id AND l.prediction_time=p.prediction_time
    GROUP BY p.race_id,p.prediction_time
),
scored AS (
    SELECT p.prediction_id,p.race_id,p.horse_id,p.prediction_time,p.race_start_at,
           p.model_version,m.model,p.odds,
           CASE m.model
             WHEN 'Logistic' THEN p.logistic_probability
             WHEN 'CatBoost' THEN p.catboost_probability
             WHEN 'XGBoost' THEN p.xgboost_probability
             ELSE p.ensemble_probability
           END AS probability
    FROM prediction_snapshots p
    JOIN run_maxima x ON x.race_id=p.race_id AND x.prediction_time=p.prediction_time
    CROSS JOIN model_names m
    WHERE CASE m.model
            WHEN 'Logistic' THEN p.logistic_probability=x.logistic_max
            WHEN 'CatBoost' THEN p.catboost_probability=x.catboost_max
            WHEN 'XGBoost' THEN p.xgboost_probability=x.xgboost_max
            ELSE p.ensemble_probability=x.ensemble_max
          END
),
ranked_predictions AS (
    SELECT s.*,ROW_NUMBER() OVER(
        PARTITION BY race_id,prediction_time,model
        ORDER BY probability DESC,horse_id
    ) AS prediction_rank
    FROM scored s
),
ranked_results AS (
    SELECT r.*,ROW_NUMBER() OVER(
        PARTITION BY race_id,horse_id ORDER BY captured_at DESC,result_id DESC
    ) AS result_rank
    FROM race_results r
    WHERE result_status='finished' AND finish_position IS NOT NULL
),
latest_results AS (
    SELECT * FROM ranked_results WHERE result_rank=1
),
ranked_programs AS (
    SELECT p.*,ROW_NUMBER() OVER(
        PARTITION BY race_id,horse_id ORDER BY captured_at DESC,snapshot_id DESC
    ) AS program_rank
    FROM program_snapshots p
),
latest_programs AS (
    SELECT * FROM ranked_programs WHERE program_rank=1
),
race_programs AS (
    SELECT race_id,MAX(track) AS city,MAX(race_no) AS race_no
    FROM latest_programs GROUP BY race_id
),
winners AS (
    SELECT r.race_id,
           GROUP_CONCAT(r.horse_id,'|') AS winner_ids,
           GROUP_CONCAT(COALESCE(p.horse_name,r.horse_id),', ') AS winner_name,
           MAX(r.result_odds) AS winner_decimal_odds
    FROM latest_results r
    LEFT JOIN latest_programs p ON p.race_id=r.race_id AND p.horse_id=r.horse_id
    WHERE r.finish_position=1
    GROUP BY r.race_id
),
evaluation_core AS (
    SELECT p.prediction_id,p.race_id,p.prediction_time,p.race_start_at,p.model,p.model_version,p.probability,
           p.horse_id AS predicted_horse_id,
           COALESCE(pp.horse_name,p.horse_id) AS predicted_horse,
           w.winner_name,w.winner_ids,
           COALESCE(rp.city,'Bilinmiyor') AS city,rp.race_no,
           COALESCE(pp.race_class, 'Bilinmiyor') AS race_class,
           COALESCE(pp.surface, 'Bilinmiyor') AS surface,
           COALESCE(pp.distance, 0) AS distance,
           date(p.race_start_at,'+3 hours') AS race_date,
           strftime('%H:%M',p.race_start_at,'+3 hours') AS race_time,
           CASE WHEN instr('|'||w.winner_ids||'|','|'||p.horse_id||'|')>0 THEN 1 ELSE 0 END AS correct,
           CASE WHEN instr('|'||w.winner_ids||'|','|'||p.horse_id||'|')>0
                THEN COALESCE(pr.official_odds,predicted_result.result_odds,w.winner_decimal_odds)
                ELSE NULL END AS decimal_odds,
           w.winner_decimal_odds
    FROM ranked_predictions p
    JOIN winners w ON w.race_id=p.race_id
    LEFT JOIN prediction_results pr ON pr.prediction_id=p.prediction_id
    LEFT JOIN latest_results predicted_result
      ON predicted_result.race_id=p.race_id
     AND predicted_result.horse_id=p.horse_id
     AND predicted_result.result_rank=1
    LEFT JOIN latest_programs pp ON pp.race_id=p.race_id AND pp.horse_id=p.horse_id
    LEFT JOIN race_programs rp ON rp.race_id=p.race_id
    WHERE p.prediction_rank=1
),
evaluated AS (
    SELECT e.*,
           CASE WHEN correct=0 THEN -1.0
                WHEN decimal_odds>0 THEN decimal_odds-1.0
                ELSE NULL END AS net_return
    FROM evaluation_core e
)
"""


def normalize_filters(
    date: str | None = None,
    track: str | None = None,
    model: str | None = None,
    outcome: str = "all",
    race_no: str | int | None = None,
    **kwargs,
) -> dict[str, Any]:
    if date:
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError("date must be YYYY-MM-DD") from exc
    if model:
        models_list = [m.strip() for m in model.split(",")]
        for m in models_list:
            if m not in MODELS:
                raise ValueError(f"model must be one of: {', '.join(MODELS)}")
    if outcome not in OUTCOMES:
        raise ValueError("outcome must be all, correct or incorrect")
    return {"date": date or None, "track": track or None, "model": model or None, "outcome": outcome, "race_no": race_no}


def _where(filters: dict[str, Any], *, include_model: bool = True) -> tuple[str, list[Any]]:
    clauses, params = [], []
    if filters.get("date"):
        clauses.append("race_date=?"); params.append(filters["date"])
    if filters.get("track"):
        clauses.append("track_key(city)=track_key(?)"); params.append(filters["track"])
    if filters.get("race_no") not in (None, ""):
        clauses.append("race_no=?"); params.append(int(filters["race_no"]))
    if include_model and filters.get("model"):
        models_list = [m.strip() for m in filters["model"].split(",")]
        placeholders = ",".join("?" for _ in models_list)
        clauses.append(f"model IN ({placeholders})")
        params.extend(models_list)
    if filters.get("outcome") == "correct":
        clauses.append("correct=1")
    elif filters.get("outcome") == "incorrect":
        clauses.append("correct=0")
    return (" WHERE " + " AND ".join(clauses) if clauses else ""), params


def summary(connection: sqlite3.Connection, filters: dict[str, str | None]) -> dict[str, Any]:
    configure_sqlite(connection)
    where, params = _where(filters)
    row = connection.execute(
        PERFORMANCE_CTE + f"""
        SELECT COUNT(*) AS total_predictions,COUNT(*)>0 AS has_data,COALESCE(SUM(correct),0) AS correct_predictions,
               COALESCE(100.0*AVG(correct),0) AS accuracy_percent,
               COALESCE(100.0*SUM(net_return)/NULLIF(COUNT(net_return),0),0) AS roi_percent,
               COALESCE(SUM(net_return),0) AS net_profit,
               COUNT(net_return) AS roi_bets,COUNT(DISTINCT race_id) AS processed_races
        FROM evaluated{where}""",
        params,
    ).fetchone()
    return dict(row)


def model_comparison(connection: sqlite3.Connection, filters: dict[str, str | None]) -> list[dict[str, Any]]:
    configure_sqlite(connection)
    where, params = _where(filters, include_model=False)
    rows = connection.execute(
        PERFORMANCE_CTE + f"""
        SELECT model,COUNT(*) AS predictions,COALESCE(SUM(correct),0) AS correct,
               COALESCE(100.0*AVG(correct),0) AS accuracy_percent,
               COALESCE(100.0*SUM(net_return)/NULLIF(COUNT(net_return),0),0) AS roi_percent,
               COALESCE(SUM(net_return),0) AS net_profit,COUNT(net_return) AS roi_bets
        FROM evaluated{where} GROUP BY model
        ORDER BY CASE model WHEN 'Logistic' THEN 1 WHEN 'CatBoost' THEN 2
                            WHEN 'XGBoost' THEN 3 ELSE 4 END""",
        params,
    ).fetchall()
    found = {row["model"]: dict(row) for row in rows}
    return [found.get(model, {"model": model, "predictions": 0, "correct": 0,
                              "accuracy_percent": 0.0, "roi_percent": 0.0,
                              "net_profit": 0.0, "roi_bets": 0}) for model in MODELS]


def history(
    connection: sqlite3.Connection,
    filters: dict[str, str | None],
    page: int = 1,
) -> dict[str, Any]:
    configure_sqlite(connection)
    page = max(1, int(page)); where, params = _where(filters)
    rows = connection.execute(
        PERFORMANCE_CTE + f"""
        SELECT race_date,city,race_no,race_time,predicted_horse,winner_name,correct,
               decimal_odds,net_return,model,prediction_time,race_start_at,race_id,
               COUNT(*) OVER() AS total_count
        FROM evaluated{where}
        ORDER BY race_start_at DESC,prediction_time DESC,
                 CASE model WHEN 'Ensemble' THEN 1 WHEN 'XGBoost' THEN 2
                            WHEN 'CatBoost' THEN 3 ELSE 4 END
        LIMIT ? OFFSET ?""",
        [*params, PAGE_SIZE, (page - 1) * PAGE_SIZE],
    ).fetchall()
    total = int(rows[0]["total_count"]) if rows else 0
    if not rows and page > 1:
        total = int(connection.execute(PERFORMANCE_CTE + f"SELECT COUNT(*) FROM evaluated{where}", params).fetchone()[0])
    items = []
    for row in rows:
        item = dict(row); item.pop("total_count", None); items.append(item)
    return {
        "page": page, "page_size": PAGE_SIZE, "total": total,
        "pages": max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE),
        "rows": items,
    }


def chart_data(connection: sqlite3.Connection, filters: dict[str, str | None]) -> dict[str, Any]:
    configure_sqlite(connection)
    where, params = _where(filters)
    daily = connection.execute(
        PERFORMANCE_CTE + f"""
        SELECT race_date,COUNT(*) AS predictions,100.0*AVG(correct) AS accuracy_percent,
               100.0*SUM(net_return)/NULLIF(COUNT(net_return),0) AS roi_percent,
               COALESCE(SUM(net_return),0) AS daily_profit
        FROM evaluated{where} GROUP BY race_date ORDER BY race_date DESC LIMIT 30""",
        params,
    ).fetchall()
    days = [dict(row) for row in reversed(daily)]
    cumulative = 0.0
    for row in days:
        cumulative += float(row["daily_profit"] or 0)
        row["cumulative_profit"] = cumulative
    models = model_comparison(connection, filters)
    return {"daily": days, "models": models}


def race_filters(connection: sqlite3.Connection) -> dict[str, Any]:
    configure_sqlite(connection)
    rows = connection.execute(
        PERFORMANCE_CTE + """
        SELECT city,COUNT(DISTINCT race_id) AS races,MIN(race_date) AS first_date,
               MAX(race_date) AS last_date FROM evaluated GROUP BY track_key(city) ORDER BY city"""
    ).fetchall()
    return {"tracks": [dict(row) for row in rows], "models": list(MODELS), "outcomes": list(OUTCOMES)}
