"""Read-only SQL analytics for the Model Diagnostics dashboard."""
from __future__ import annotations

import sqlite3
import json
from datetime import datetime
from typing import Any, Iterator
from race_scope import configure_sqlite

MODELS = ("Logistic", "CatBoost", "XGBoost", "Ensemble")
PAGE_SIZE = 100

DETAIL_CTE = r"""
WITH settings(model,race_id) AS (VALUES (?,?)),
latest_run AS (
    SELECT p.race_id,MAX(p.prediction_time) AS prediction_time
    FROM prediction_snapshots p,settings s
    WHERE p.race_id=s.race_id AND julianday(p.prediction_time)<julianday(p.race_start_at)
),
scored AS (
    SELECT p.*,s.model,
           CASE s.model WHEN 'Logistic' THEN p.logistic_probability
                        WHEN 'CatBoost' THEN p.catboost_probability
                        WHEN 'XGBoost' THEN p.xgboost_probability
                        ELSE p.ensemble_probability END AS probability
    FROM prediction_snapshots p JOIN latest_run l
      ON l.race_id=p.race_id AND l.prediction_time=p.prediction_time
    CROSS JOIN settings s
),
ranked AS (
    SELECT scored.*,ROW_NUMBER() OVER(ORDER BY probability DESC,horse_id) AS model_rank
    FROM scored
),
result_ranked AS (
    SELECT r.*,ROW_NUMBER() OVER(
        PARTITION BY r.race_id,r.horse_id ORDER BY r.captured_at DESC,r.result_id DESC
    ) AS rn FROM race_results r,settings s WHERE r.race_id=s.race_id
),
results AS (SELECT * FROM result_ranked WHERE rn=1)
"""

DIAGNOSTICS_CTE = r"""
WITH settings(model) AS (VALUES (?)),
latest_prediction_runs AS (
    SELECT race_id,MAX(prediction_time) AS prediction_time
    FROM prediction_snapshots
    WHERE julianday(prediction_time)<julianday(race_start_at)
    GROUP BY race_id
),
scored AS (
    SELECT p.*,
           CASE s.model WHEN 'Logistic' THEN p.logistic_probability
                        WHEN 'CatBoost' THEN p.catboost_probability
                        WHEN 'XGBoost' THEN p.xgboost_probability
                        ELSE p.ensemble_probability END AS model_probability,
           s.model
    FROM prediction_snapshots p
    JOIN latest_prediction_runs l
      ON l.race_id=p.race_id AND l.prediction_time=p.prediction_time
    CROSS JOIN settings s
),
ranked_predictions AS (
    SELECT scored.*,ROW_NUMBER() OVER(
        PARTITION BY race_id ORDER BY model_probability DESC,horse_id
    ) AS model_rank
    FROM scored
),
program_ranked AS (
    SELECT p.*,ROW_NUMBER() OVER(
        PARTITION BY race_id,horse_id ORDER BY captured_at DESC,snapshot_id DESC
    ) AS rn
    FROM program_snapshots p
    WHERE julianday(captured_at)<julianday(race_start_at)
),
programs AS (SELECT * FROM program_ranked WHERE rn=1),
race_meta AS (
    SELECT race_id,MAX(race_start_at) AS race_start_at,MAX(race_no) AS race_no,
           COALESCE(MAX(track),'Bilinmiyor') AS track,MAX(surface) AS surface,
           MAX(distance) AS distance,MAX(race_class) AS race_class,COUNT(*) AS horse_count
    FROM programs GROUP BY race_id
),
result_ranked AS (
    SELECT r.*,ROW_NUMBER() OVER(
        PARTITION BY race_id,horse_id ORDER BY captured_at DESC,result_id DESC
    ) AS rn
    FROM race_results r
    WHERE result_status='finished' AND finish_position IS NOT NULL
),
results AS (SELECT * FROM result_ranked WHERE rn=1),
winner_candidates AS (
    SELECT r.race_id,r.horse_id,r.result_odds,p.model_rank,p.model_probability,
           p.agf_percent,p.agf_rank,
           ROW_NUMBER() OVER(PARTITION BY r.race_id ORDER BY p.model_rank,r.horse_id) AS rn
    FROM results r JOIN ranked_predictions p
      ON p.race_id=r.race_id AND p.horse_id=r.horse_id
    WHERE r.finish_position=1
),
winners AS (SELECT * FROM winner_candidates WHERE rn=1),
agf_favorite_ranked AS (
    SELECT a.*,ROW_NUMBER() OVER(
        PARTITION BY race_id ORDER BY CASE WHEN agf_rank IS NULL THEN 1 ELSE 0 END,
                 agf_rank,agf_percent DESC,horse_id
    ) AS favorite_rank
    FROM ranked_predictions a
),
agf_favorites AS (SELECT * FROM agf_favorite_ranked WHERE favorite_rank=1),
top_predictions AS (SELECT * FROM ranked_predictions WHERE model_rank=1),
diagnostics AS (
    SELECT m.race_id,date(m.race_start_at,'+3 hours') AS race_date,m.track,m.race_no,
           m.race_start_at,strftime('%H:%M',m.race_start_at,'+3 hours') AS race_time,
           m.surface,m.distance,m.race_class,m.horse_count,t.model,t.model_version,
           t.prediction_time,t.horse_id AS top1_horse_id,
           COALESCE(tp.horse_name,t.horse_id) AS top1_horse,
           t.model_probability AS top1_probability,w.horse_id AS winner_horse_id,
           COALESCE(wp.horse_name,w.horse_id) AS winner_horse,w.model_rank AS winner_rank,
           w.model_probability AS winner_probability,
           t.model_probability-w.model_probability AS probability_difference,
           w.agf_percent AS winner_agf,w.agf_rank AS winner_agf_rank,
           af.horse_id AS agf_favorite_id,COALESCE(ap.horse_name,af.horse_id) AS agf_favorite,
           af.agf_percent AS agf_favorite_percent,
           CASE WHEN t.horse_id=w.horse_id THEN 1 ELSE 0 END AS correct,
           CASE WHEN w.model_rank<=2 THEN 1 ELSE 0 END AS top2_correct,
           CASE WHEN w.model_rank<=3 THEN 1 ELSE 0 END AS top3_correct,
           CASE WHEN w.model_rank<=5 THEN 1 ELSE 0 END AS top5_correct,
           CASE WHEN t.horse_id=w.horse_id AND af.horse_id<>w.horse_id THEN 'Model > AGF'
                WHEN af.horse_id=w.horse_id AND t.horse_id<>w.horse_id THEN 'AGF > Model'
                ELSE 'Beraber' END AS agf_comparison,
           w.result_odds AS official_odds,t.odds AS pre_race_odds,
           CASE WHEN t.odds IS NULL OR t.odds<=0 THEN NULL
                WHEN t.horse_id<>w.horse_id THEN -1.0 ELSE t.odds-1.0 END AS net_return,
           CASE WHEN lower(COALESCE(m.surface,'')) LIKE '%sent%' THEN 'Sentetik'
                WHEN lower(COALESCE(m.surface,'')) LIKE '%kum%' THEN 'Kum'
                WHEN lower(COALESCE(m.surface,'')) LIKE '%çim%' OR lower(COALESCE(m.surface,'')) LIKE '%cim%' THEN 'Çim'
                ELSE COALESCE(NULLIF(m.surface,''),'Bilinmiyor') END AS surface_group,
           CASE WHEN lower(COALESCE(m.race_class,'')) LIKE '%ingiliz%' OR lower(COALESCE(m.race_class,'')) LIKE '%İngiliz%' THEN 'İngiliz'
                WHEN lower(COALESCE(m.race_class,'')) LIKE '%arap%' THEN 'Arap' ELSE 'Belirsiz' END AS breed_group,
           CASE WHEN upper(COALESCE(m.race_class,'')) LIKE '%G1%' THEN 'G1'
                WHEN upper(COALESCE(m.race_class,'')) LIKE '%G2%' THEN 'G2'
                WHEN upper(COALESCE(m.race_class,'')) LIKE '%G3%' THEN 'G3'
                WHEN upper(COALESCE(m.race_class,'')) LIKE '%KV%' THEN 'KV'
                WHEN lower(COALESCE(m.race_class,'')) LIKE '%maiden%' THEN 'Maiden'
                WHEN lower(COALESCE(m.race_class,'')) LIKE '%şart%' OR lower(COALESCE(m.race_class,'')) LIKE '%sart%' THEN 'Şartlı'
                WHEN lower(COALESCE(m.race_class,'')) LIKE '%hand%' THEN 'Handikap'
                ELSE COALESCE(NULLIF(m.race_class,''),'Diğer') END AS race_type_group,
           CASE WHEN m.distance IS NULL THEN 'Bilinmiyor' WHEN m.distance<=1200 THEN '≤1200'
                WHEN m.distance<=1600 THEN '1300-1600' WHEN m.distance<=2000 THEN '1700-2000'
                ELSE '2000+' END AS distance_group,
           CASE WHEN m.horse_count<=6 THEN '≤6' WHEN m.horse_count<=10 THEN '7-10'
                WHEN m.horse_count<=14 THEN '11-14' ELSE '15+' END AS field_size_group
    FROM race_meta m JOIN top_predictions t ON t.race_id=m.race_id
    JOIN winners w ON w.race_id=m.race_id
    LEFT JOIN programs tp ON tp.race_id=t.race_id AND tp.horse_id=t.horse_id
    LEFT JOIN programs wp ON wp.race_id=w.race_id AND wp.horse_id=w.horse_id
    LEFT JOIN agf_favorites af ON af.race_id=m.race_id
    LEFT JOIN programs ap ON ap.race_id=af.race_id AND ap.horse_id=af.horse_id
)
"""


def normalize_filters(date: str | None = None, track: str | None = None,
                      model: str | None = None, race_type: str | None = None,
                      distance: str | None = None, surface: str | None = None,
                      field_size: str | None = None) -> dict[str, str | None]:
    if date:
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError("date must be YYYY-MM-DD") from exc
    model = model or "Ensemble"
    if model not in MODELS:
        raise ValueError(f"model must be one of: {', '.join(MODELS)}")
    return {"date": date or None, "track": track or None, "model": model,
            "race_type": race_type or None, "distance": distance or None,
            "surface": surface or None, "field_size": field_size or None}


def _where(filters: dict[str, str | None]) -> tuple[str, list[Any]]:
    columns = {"date": "race_date", "track": "track", "race_type": "race_type_group",
               "distance": "distance_group", "surface": "surface_group",
               "field_size": "field_size_group"}
    clauses, params = [], []
    for key, column in columns.items():
        if filters.get(key):
            clauses.append(f"track_key({column})=track_key(?)" if key == "track" else f"{column}=?")
            params.append(filters[key])
    return (" WHERE " + " AND ".join(clauses) if clauses else ""), params


def _params(filters: dict[str, str | None], extra: list[Any] | None = None) -> list[Any]:
    _, where_params = _where(filters)
    return [filters["model"], *where_params, *(extra or [])]


def summary(connection: sqlite3.Connection, filters: dict[str, str | None]) -> dict[str, Any]:
    configure_sqlite(connection)
    where, _ = _where(filters)
    row = connection.execute(DIAGNOSTICS_CTE + f"""
        SELECT COUNT(*) AS race_count,COALESCE(100.0*AVG(correct),0) AS top1_accuracy,
               COALESCE(100.0*AVG(top2_correct),0) AS top2_accuracy,
               COALESCE(100.0*AVG(top3_correct),0) AS top3_accuracy,
               COALESCE(100.0*AVG(top5_correct),0) AS top5_accuracy,
               COALESCE(SUM(correct),0) AS correct_races,
               COALESCE(SUM(agf_comparison='Model > AGF'),0) AS model_over_agf,
               COALESCE(SUM(agf_comparison='AGF > Model'),0) AS agf_over_model,
               COALESCE(SUM(agf_comparison='Beraber'),0) AS tied
        FROM diagnostics{where}""", _params(filters)).fetchone()
    return dict(row)


def races(connection: sqlite3.Connection, filters: dict[str, str | None], page: int = 1) -> dict[str, Any]:
    configure_sqlite(connection)
    page = max(1, int(page)); where, _ = _where(filters)
    rows = connection.execute(DIAGNOSTICS_CTE + f"""
        SELECT *,COUNT(*) OVER() AS total_count FROM diagnostics{where}
        ORDER BY race_start_at DESC,race_no DESC LIMIT ? OFFSET ?""",
        _params(filters, [PAGE_SIZE, (page - 1) * PAGE_SIZE])).fetchall()
    total = int(rows[0]["total_count"]) if rows else 0
    if not rows and page > 1:
        total = int(connection.execute(DIAGNOSTICS_CTE + f"SELECT COUNT(*) FROM diagnostics{where}",
                                       _params(filters)).fetchone()[0])
    items = []
    for row in rows:
        item = dict(row); item.pop("total_count", None); items.append(item)
    return {"page": page, "page_size": PAGE_SIZE, "total": total,
            "pages": max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE), "rows": items}


def winner_ranks(connection: sqlite3.Connection, filters: dict[str, str | None]) -> list[dict[str, Any]]:
    configure_sqlite(connection)
    where, _ = _where(filters)
    return [dict(row) for row in connection.execute(DIAGNOSTICS_CTE + f"""
        SELECT winner_rank,COUNT(*) AS race_count FROM diagnostics{where}
        GROUP BY winner_rank ORDER BY winner_rank""", _params(filters)).fetchall()]


def group_performance(connection: sqlite3.Connection, filters: dict[str, str | None]) -> list[dict[str, Any]]:
    configure_sqlite(connection)
    where, _ = _where(filters)
    dimensions = (("Pist Türü", "surface_group"), ("Irk", "breed_group"),
                  ("Yarış Tipi", "race_type_group"), ("Mesafe", "distance_group"),
                  ("At Sayısı", "field_size_group"))
    selects = []
    for label, column in dimensions:
        selects.append(f"""SELECT '{label}' AS dimension,{column} AS group_name,
            COUNT(*) AS race_count,100.0*AVG(correct) AS top1_accuracy,
            100.0*AVG(top3_correct) AS top3_accuracy,
            100.0*SUM(net_return)/NULLIF(COUNT(net_return),0) AS roi,
            COALESCE(SUM(net_return),0) AS net_profit FROM filtered GROUP BY {column}""")
    sql = DIAGNOSTICS_CTE + f", filtered AS (SELECT * FROM diagnostics{where})\n" + " UNION ALL ".join(selects)
    return [dict(row) for row in connection.execute(sql, _params(filters)).fetchall()]


def extremes(connection: sqlite3.Connection, filters: dict[str, str | None]) -> dict[str, Any]:
    configure_sqlite(connection)
    where, _ = _where(filters)
    base = DIAGNOSTICS_CTE + f", filtered AS (SELECT * FROM diagnostics{where}) "
    columns = "race_id,race_date,track,race_no,top1_horse,top1_probability,winner_horse,winner_probability,probability_difference"
    errors = connection.execute(base + f"SELECT {columns} FROM filtered WHERE correct=0 ORDER BY top1_probability DESC LIMIT 50", _params(filters)).fetchall()
    successes = connection.execute(base + f"SELECT {columns} FROM filtered WHERE correct=1 ORDER BY top1_probability ASC LIMIT 50", _params(filters)).fetchall()
    return {"errors": [dict(row) for row in errors], "successes": [dict(row) for row in successes]}


def filter_options(connection: sqlite3.Connection, model: str = "Ensemble") -> dict[str, Any]:
    configure_sqlite(connection)
    filters = normalize_filters(model=model)
    row = connection.execute(DIAGNOSTICS_CTE + """
        SELECT GROUP_CONCAT(DISTINCT track) AS tracks,GROUP_CONCAT(DISTINCT race_type_group) AS race_types,
               GROUP_CONCAT(DISTINCT distance_group) AS distances,GROUP_CONCAT(DISTINCT surface_group) AS surfaces,
               GROUP_CONCAT(DISTINCT field_size_group) AS field_sizes FROM diagnostics""", [filters["model"]]).fetchone()
    split = lambda value: sorted(x for x in (value or "").split(",") if x)
    return {"models": list(MODELS), "tracks": split(row["tracks"]),
            "race_types": split(row["race_types"]), "distances": split(row["distances"]),
            "surfaces": split(row["surfaces"]), "field_sizes": split(row["field_sizes"])}


def export_rows(connection: sqlite3.Connection, filters: dict[str, str | None]) -> Iterator[dict[str, Any]]:
    configure_sqlite(connection)
    where, _ = _where(filters)
    cursor = connection.execute(DIAGNOSTICS_CTE + f"SELECT * FROM diagnostics{where} ORDER BY race_start_at DESC", _params(filters))
    for row in cursor:
        yield dict(row)


def feature_contribution_status() -> dict[str, Any]:
    return {"available": False, "reason": "Arşivde SHAP katkı değerleri yok; read-only diagnostics modeli yeniden çalıştırmaz."}


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _decode_features(value: Any) -> dict[str, Any]:
    try:
        decoded = json.loads(value or "{}")
        return decoded if isinstance(decoded, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _confidence(top_probability: float, second_probability: float, correct: bool) -> dict[str, Any]:
    margin = max(0.0, top_probability - second_probability)
    if top_probability >= 0.25 or margin >= 0.10:
        level = "High"
    elif top_probability < 0.15 and margin < 0.03:
        level = "Low"
    else:
        level = "Medium"
    label = "Medium Confidence" if level == "Medium" else f"{level} Confidence {'Correct' if correct else 'Wrong'}"
    return {"level": level, "label": label, "top_probability": top_probability,
            "top2_margin": margin, "rules_version": "probability-margin-v1"}


def _horse_card(row: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    def first(*names):
        return next((features[name] for name in names if name in features and features[name] is not None), None)
    return {
        "horse_name": row.get("horse_name"), "horse_id": row.get("horse_id"),
        "model_rank": row.get("model_rank"), "probability": row.get("probability"),
        "agf": row.get("agf_percent"), "agf_rank": row.get("agf_rank"),
        "odds_at_prediction": row.get("odds"), "jockey": row.get("jockey"),
        "trainer": row.get("trainer"), "carried_weight": row.get("carried_weight"),
        "pre_race_handicap_rating": first("pre_race_handicap_rating"),
        "age": first("age"), "sex": first("sex", "gender"),
        "pedigree": first("pedigree", "pedigree_score"),
        "last_race_date": first("last_race_date"),
        "last_3_races": first("last_3_races", "last_3_avg_position"),
        "last_5_races": first("last_5_races", "last_5_avg_position"),
        "average_finish": first("average_finish", "last_10_avg_position"),
        "career_starts": first("career_starts", "starts"),
        "career_wins": first("career_wins", "wins"),
        "career_earnings": first("career_earnings", "earnings"),
        "features": features,
    }


def race_detail(connection: sqlite3.Connection, race_id: str, model: str = "Ensemble") -> dict[str, Any] | None:
    configure_sqlite(connection)
    if model not in MODELS:
        raise ValueError(f"model must be one of: {', '.join(MODELS)}")
    rows = connection.execute(DETAIL_CTE + """
        SELECT p.prediction_id,p.race_id,p.horse_id,p.prediction_time,p.race_start_at,
               p.model,p.model_version,p.model_rank,p.probability,p.feature_hash,
               p.feature_values_json,p.feature_contract_version,p.feature_snapshot_id,
               COALESCE(fs.horse_name,p.horse_id) AS horse_name,fs.race_no,fs.track,
               fs.surface,fs.distance,fs.race_class,fs.jockey,fs.trainer,fs.carried_weight,
               fs.draw,fs.handicap_rating,p.agf_percent,p.agf_rank,p.odds,
               r.finish_position,r.result_status
        FROM ranked p
        LEFT JOIN program_snapshots fs ON fs.snapshot_id=p.feature_snapshot_id
        LEFT JOIN results r ON r.race_id=p.race_id AND r.horse_id=p.horse_id
        ORDER BY p.model_rank""", (model, race_id)).fetchall()
    if not rows:
        return None
    ranking = [dict(row) for row in rows]
    archive_rows: dict[str, sqlite3.Row] = {}
    if _table_exists(connection, "prediction_feature_snapshots"):
        ids = [row["prediction_id"] for row in ranking]
        placeholders = ",".join("?" for _ in ids)
        archive_rows = {row["prediction_id"]: row for row in connection.execute(
            f"SELECT * FROM prediction_feature_snapshots WHERE prediction_id IN ({placeholders})", ids
        )}
    for row in ranking:
        archived = archive_rows.get(row["prediction_id"])
        raw = archived["feature_values_json"] if archived else row["feature_values_json"]
        row["features"] = _decode_features(raw)
        row["feature_snapshot_available"] = bool(row["features"])
        row["feature_snapshot_source"] = "prediction_feature_snapshots" if archived else (
            "prediction_snapshots.feature_values_json" if row["features"] else None
        )
        row["feature_hash_verified"] = bool(archived and archived["feature_hash"] == row["feature_hash"])
        row["winner"] = row.get("finish_position") == 1

    selected = ranking[0]
    winner = next((row for row in ranking if row["winner"]), None)
    if winner is None:
        return None
    selected_features, winner_features = selected["features"], winner["features"]
    feature_names = list(dict.fromkeys([*selected_features.keys(), *winner_features.keys()]))
    comparison = []
    for name in feature_names:
        left, right = selected_features.get(name), winner_features.get(name)
        difference = None
        if isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)) and not isinstance(right, bool):
            difference = float(left) - float(right)
        comparison.append({"feature": name, "model_selection": left, "winner": right,
                           "difference": difference, "equal": left == right})

    correct = selected["horse_id"] == winner["horse_id"]
    second_probability = float(ranking[1]["probability"]) if len(ranking) > 1 else 0.0
    confidence = _confidence(float(selected["probability"]), second_probability, correct)
    gap = float(selected["probability"]) - float(winner["probability"])
    if correct:
        message = "Modelin ilk tercihi yarışı kazandı."
    elif winner["model_rank"] <= 3 and gap <= 0.03:
        message = "Model kazananı güçlü ve rekabetçi bir aday olarak değerlendirdi; fark küçüktü."
    elif winner["model_rank"] <= 5 or gap <= 0.07:
        message = "Kazanan modelin aday kümesindeydi ancak ilk tercih değildi."
    else:
        message = "Kazanan, arşivlenmiş olasılıklara göre gerçekçi bir ilk sıra adayı olarak değerlendirilmedi."
    first = ranking[0]
    metadata = {key: first.get(key) for key in (
        "race_id", "race_start_at", "race_no", "track", "surface", "distance",
        "race_class", "prediction_time", "model", "model_version", "feature_contract_version"
    )}
    return {
        "race": metadata, "model_selection": _horse_card(selected, selected_features),
        "winner": _horse_card(winner, winner_features), "feature_comparison": comparison,
        "ranking": ranking, "top10": ranking[:10], "correct": correct,
        "winner_rank": winner["model_rank"], "probability_difference": gap,
        "confidence": confidence,
        "why_missed": {"winner_rank": winner["model_rank"], "probability_difference": gap,
                       "confidence": confidence["label"], "message": message},
        "feature_snapshot_available": bool(selected_features and winner_features),
        "feature_snapshot_message": None if selected_features and winner_features else
            "Prediction feature snapshot was not archived. Historical feature comparison is unavailable.",
        "shap": feature_contribution_status(),
    }
