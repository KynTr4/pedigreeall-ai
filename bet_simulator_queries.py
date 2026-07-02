"""Read-only flat-stake simulation over archived Top-1 predictions."""
from __future__ import annotations

import sqlite3
from collections import Counter
from statistics import mean, median
from typing import Any, Iterator

from diagnostics_queries import DIAGNOSTICS_CTE
from performance_queries import PERFORMANCE_CTE, PAGE_SIZE, normalize_filters, _where
from race_scope import configure_sqlite

MODEL_COMPARISON_ORDER = ("Ensemble", "Logistic", "CatBoost", "XGBoost")


def normalize_bet_filters(date=None, track=None, model="Ensemble", outcome="all", stake=20, race_no=None) -> dict[str, Any]:
    # Support comma-separated models in validation
    base = normalize_filters(date, track, model or "Ensemble", outcome, race_no=race_no)
    try:
        amount = float(stake)
    except (TypeError, ValueError) as exc:
        raise ValueError("stake must be numeric") from exc
    if not 0 < amount <= 1_000_000:
        raise ValueError("stake must be greater than 0 and at most 1000000")
    return {**base, "stake": amount}


def _parts(filters: dict[str, Any]) -> tuple[str, list[Any]]:
    return _where(filters)


def _trend_parts(filters: dict[str, Any]) -> tuple[str, list[Any]]:
    clauses, params = [], []
    if filters.get("date"):
        from datetime import datetime, timedelta
        dt = datetime.strptime(filters["date"], "%Y-%m-%d")
        # Query 5 days prior to the target date (total 6 days, plus 1 baseline prepended day = 7 days)
        start_date = (dt - timedelta(days=5)).strftime("%Y-%m-%d")
        clauses.append("race_date BETWEEN ? AND ?")
        params.extend([start_date, filters["date"]])
    if filters.get("track"):
        clauses.append("track_key(city)=track_key(?)")
        params.append(filters["track"])
    if filters.get("race_no") not in (None, ""):
        clauses.append("race_no=?")
        params.append(int(filters["race_no"]))
    if filters.get("model"):
        models_list = [m.strip() for m in filters["model"].split(",")]
        placeholders = ",".join("?" for _ in models_list)
        clauses.append(f"model IN ({placeholders})")
        params.extend(models_list)
    if filters.get("outcome") == "correct":
        clauses.append("correct=1")
    elif filters.get("outcome") == "incorrect":
        clauses.append("correct=0")
    return (" WHERE " + " AND ".join(clauses) if clauses else ""), params


def _row_sql(where: str) -> str:
    return f"""SELECT race_date,city AS track,race_no,race_time,model,predicted_horse,
               winner_name,correct,decimal_odds,winner_decimal_odds,race_start_at,prediction_time,race_id,
               probability,surface,race_class,distance,
               CASE WHEN correct=0 OR decimal_odds>0 THEN 1 ELSE 0 END AS bet_eligible
        FROM evaluated{where}"""


def _money(row: dict[str, Any], stake: float) -> dict[str, Any]:
    eligible = bool(row["bet_eligible"])
    if not eligible:
        returned = net = None
    elif row["correct"]:
        returned = stake * float(row["decimal_odds"]); net = returned - stake
    else:
        returned = 0.0; net = -stake
    return {**row, "stake": stake, "return_amount": returned, "net_profit": net,
            "odds_status": "AVAILABLE" if eligible else "ODDS_MISSING"}


def history(connection: sqlite3.Connection, filters: dict[str, Any], page: int = 1) -> dict[str, Any]:
    configure_sqlite(connection); page=max(1,int(page)); where,params=_parts(filters)
    
    # Get distinct race_ids for this page (paginated by race rather than model rows)
    race_ids_rows = connection.execute(
        PERFORMANCE_CTE + f"""
        SELECT DISTINCT race_id, race_start_at 
        FROM evaluated {where}
        ORDER BY race_start_at DESC, race_id
        LIMIT ? OFFSET ?""",
        [*params, PAGE_SIZE, (page - 1) * PAGE_SIZE]
    ).fetchall()
    
    race_ids = [row["race_id"] for row in race_ids_rows]
    
    if not race_ids:
        return {"page":page,"page_size":PAGE_SIZE,"total":0,"pages":1,"rows":[]}
        
    total = connection.execute(
        PERFORMANCE_CTE + f"SELECT COUNT(DISTINCT race_id) FROM evaluated{where}",
        params
    ).fetchone()[0]
    
    # Query evaluated details for the selected race_ids (query all models to pivot)
    placeholders = ",".join("?" for _ in race_ids)
    rows = connection.execute(
        PERFORMANCE_CTE + f"""
        SELECT race_date, city AS track, race_no, race_time, model, predicted_horse,
               winner_name, correct, decimal_odds, winner_decimal_odds, race_start_at, prediction_time, race_id,
               probability, surface, race_class, distance,
               CASE WHEN correct=0 OR decimal_odds>0 THEN 1 ELSE 0 END AS bet_eligible
        FROM evaluated
        WHERE race_id IN ({placeholders})
        ORDER BY race_start_at DESC, model""",
        race_ids
    ).fetchall()
    
    pivoted = {}
    stake = filters["stake"]
    # Get primary model to copy to root for backwards compatibility
    primary_model = [m.strip() for m in (filters.get("model") or "Ensemble").split(",")][0]
    
    for r in rows:
        row_dict = _money(dict(r), stake)
        race_id = row_dict["race_id"]
        if race_id not in pivoted:
            pivoted[race_id] = {
                "race_id": race_id,
                "race_date": row_dict["race_date"],
                "track": row_dict["track"],
                "race_no": row_dict["race_no"],
                "race_time": row_dict["race_time"],
                "winner_name": row_dict["winner_name"],
                "decimal_odds": row_dict["winner_decimal_odds"],
                "race_start_at": row_dict["race_start_at"],
                "surface": row_dict["surface"],
                "race_class": row_dict["race_class"],
                "distance": row_dict["distance"],
                "models": {}
            }
        m = row_dict["model"]
        m_data = {
            "predicted_horse": row_dict["predicted_horse"],
            "correct": row_dict["correct"],
            "probability": row_dict["probability"],
            "net_profit": row_dict["net_profit"],
            "bet_eligible": row_dict["bet_eligible"],
            "odds_status": row_dict["odds_status"],
            "return_amount": row_dict["return_amount"],
            "stake": stake,
            "model": m,
            "official_odds": row_dict["decimal_odds"],
        }
        pivoted[race_id]["models"][m] = m_data

        if row_dict["correct"] and row_dict["decimal_odds"] is not None:
            pivoted[race_id]["decimal_odds"] = row_dict["decimal_odds"]
        
        # Copy to root if it is the primary model for legacy compatibility
        if m == primary_model:
            pivoted[race_id].update(m_data)
            
    # For compatibility, ensure root fields are populated even if primary model had no prediction
    for race_id, data in pivoted.items():
        if "model" not in data and data["models"]:
            first_m = list(data["models"].keys())[0]
            data.update(data["models"][first_m])
            
    rows_list = [pivoted[race_id] for race_id in race_ids if race_id in pivoted]
    
    return {"page":page,"page_size":PAGE_SIZE,"total":int(total),
            "pages":max(1,(int(total)+PAGE_SIZE-1)//PAGE_SIZE),
            "rows":rows_list}


def summary(connection: sqlite3.Connection, filters: dict[str, Any]) -> dict[str, Any]:
    configure_sqlite(connection); where,params=_parts(filters); stake=filters["stake"]
    
    # Fetch all matching rows for the models
    rows_raw = connection.execute(
        PERFORMANCE_CTE + _row_sql(where) + " ORDER BY race_start_at,prediction_time", params
    ).fetchall()
    
    model_rows = {}
    for r in rows_raw:
        row_dict = _money(dict(r), stake)
        model_rows.setdefault(row_dict["model"], []).append(row_dict)
        
    models_summary = {}
    selected_models = [m.strip() for m in (filters.get("model") or "Ensemble,Logistic,CatBoost,XGBoost").split(",")]
    
    for m in selected_models:
        rows = model_rows.get(m, [])
        bets = [row for row in rows if row["bet_eligible"]]
        correct = sum(int(row["correct"]) for row in rows)
        invested = stake * len(bets)
        returned = sum(float(row["return_amount"] or 0) for row in bets)
        
        # Calculate streaks
        max_win_streak = current_win = 0
        max_loss_streak = current_loss = 0
        for row in rows:
            if row["bet_eligible"]:
                if row["correct"]:
                    current_win += 1
                    max_win_streak = max(max_win_streak, current_win)
                    current_loss = 0
                else:
                    current_loss += 1
                    max_loss_streak = max(max_loss_streak, current_loss)
                    current_win = 0
                    
        profitable = [row for row in bets if row["net_profit"] is not None]
        best = max(profitable, key=lambda row: row["net_profit"], default=None)
        worst = min(profitable, key=lambda row: row["net_profit"], default=None)
        
        avg_odds = sum(float(row["decimal_odds"] or 0) for row in bets if row["correct"]) / sum(1 for row in bets if row["correct"]) if sum(1 for row in bets if row["correct"]) > 0 else 0
        avg_prob = sum(float(row["probability"] or 0) for row in bets) / len(bets) if bets else 0
        
        models_summary[m] = {
            "has_data": bool(rows),
            "stake": stake,
            "total_races": len(rows),
            "bet_races": len(bets),
            "correct_predictions": correct,
            "incorrect_predictions": len(rows) - correct,
            "accuracy_percent": 100 * correct / len(rows) if rows else 0,
            "total_invested": invested,
            "total_return": returned,
            "net_profit": returned - invested,
            "roi_percent": 100 * (returned - invested) / invested if invested else 0,
            "largest_winning_streak": max_win_streak,
            "largest_losing_streak": max_loss_streak,
            "odds_missing_races": sum(not row["bet_eligible"] for row in rows),
            "average_odds": avg_odds,
            "average_probability": avg_prob,
            "best_race": best,
            "worst_race": worst,
        }
        
    # Track-level profitability
    track_profit = {}
    for r in rows_raw:
        row_dict = _money(dict(r), stake)
        if row_dict["bet_eligible"] and row_dict["net_profit"] is not None:
            t = row_dict["track"]
            track_profit[t] = track_profit.get(t, 0.0) + row_dict["net_profit"]
            
    best_track = max(track_profit.keys(), key=lambda k: track_profit[k], default="—")
    worst_track = min(track_profit.keys(), key=lambda k: track_profit[k], default="—")
    if best_track != "—":
        best_track = f"{best_track} ({track_profit[best_track]:+.2f} TL)"
    if worst_track != "—":
        worst_track = f"{worst_track} ({track_profit[worst_track]:+.2f} TL)"
        
    # Surface-level profitability
    surface_profit = {}
    for r in rows_raw:
        row_dict = _money(dict(r), stake)
        if row_dict["bet_eligible"] and row_dict["net_profit"] is not None:
            s = row_dict.get("surface", "—")
            surface_profit[s] = surface_profit.get(s, 0.0) + row_dict["net_profit"]
            
    best_surface = max(surface_profit.keys(), key=lambda k: surface_profit[k], default="—")
    worst_surface = min(surface_profit.keys(), key=lambda k: surface_profit[k], default="—")
    
    # Correct bets for odds stats
    correct_bets = [row for row in (
        [_money(dict(r), stake) for r in rows_raw]
    ) if row["bet_eligible"] and row["correct"]]
    
    highest_odds_winner = max(correct_bets, key=lambda row: float(row["decimal_odds"] or 0), default=None)
    lowest_odds_winner = min(correct_bets, key=lambda row: float(row["decimal_odds"] or 9999), default=None)
    
    avg_winning_odds = sum(float(row["decimal_odds"] or 0) for row in correct_bets) / len(correct_bets) if correct_bets else 0
    total_unbetted = sum(1 for r in rows_raw if not _money(dict(r), stake)["bet_eligible"])
    
    # Fetch trend data over a 7-day range
    trend_where, trend_params = _trend_parts(filters)
    trend_rows = connection.execute(
        PERFORMANCE_CTE + _row_sql(trend_where) + " ORDER BY race_start_at,prediction_time", trend_params
    ).fetchall()
    
    trend_model_rows = {}
    for r in trend_rows:
        row_dict = _money(dict(r), stake)
        trend_model_rows.setdefault(row_dict["model"], []).append(row_dict)
        
    trend_dates = sorted(list(set(row["race_date"] for row in trend_rows)))
    
    if filters.get("date") and trend_dates:
        from datetime import datetime, timedelta
        first_date_dt = datetime.strptime(trend_dates[0], "%Y-%m-%d")
        baseline_date = (first_date_dt - timedelta(days=1)).strftime("%Y-%m-%d")
        all_dates = [baseline_date] + trend_dates
    else:
        all_dates = trend_dates
        
    cumulative_data = {}
    for m in selected_models:
        m_rows = trend_model_rows.get(m, [])
        
        # calculate daily net profit for this model
        daily_profit = {}
        for row in m_rows:
            if row["bet_eligible"] and row["net_profit"] is not None:
                d = row["race_date"]
                daily_profit[d] = daily_profit.get(d, 0.0) + row["net_profit"]
                
        # build cumulative list
        cum_sum = 0.0
        cum_list = []
        
        if filters.get("date") and trend_dates:
            cum_list.append(0.0)
            
        for d in trend_dates:
            cum_sum += daily_profit.get(d, 0.0)
            cum_list.append(cum_sum)
            
        cumulative_data[m] = cum_list
        
    formatted_dates = []
    for d_str in all_dates:
        try:
            from datetime import datetime
            dt = datetime.strptime(d_str, "%Y-%m-%d")
            formatted_dates.append(dt.strftime("%d.%m"))
        except ValueError:
            formatted_dates.append(d_str)
            
    stats_panel = {
        "best_track": best_track,
        "worst_track": worst_track,
        "best_surface": best_surface,
        "worst_surface": worst_surface,
        "highest_odds_winner": f"{highest_odds_winner['predicted_horse']} ({highest_odds_winner['decimal_odds']:.2f})" if highest_odds_winner else "—",
        "lowest_odds_winner": f"{lowest_odds_winner['predicted_horse']} ({lowest_odds_winner['decimal_odds']:.2f})" if lowest_odds_winner else "—",
        "avg_winning_odds": avg_winning_odds,
        "total_unbetted": total_unbetted,
        "odds_missing_count": total_unbetted,
    }
    
    # Fallback/Backward compatible root metrics using the first model
    default_model = selected_models[0] if selected_models else "Ensemble"
    result = dict(models_summary.get(default_model, {
        "has_data": False, "stake": stake, "total_races": 0, "bet_races": 0,
        "correct_predictions": 0, "accuracy_percent": 0, "total_invested": 0,
        "total_return": 0, "net_profit": 0, "roi_percent": 0, "largest_winning_streak": 0,
        "largest_losing_streak": 0, "odds_missing_races": 0, "average_odds": 0,
        "average_probability": 0, "best_race": None, "worst_race": None
    }))
    
    result.update({
        "models": models_summary,
        "cumulative_data": {
            "labels": formatted_dates,
            "series": cumulative_data,
        },
        "stats_panel": stats_panel,
    })
    return result


def model_comparison(connection: sqlite3.Connection, filters: dict[str, Any]) -> dict[str, Any]:
    configure_sqlite(connection)
    selected = [
        model for model in MODEL_COMPARISON_ORDER
        if model in {value.strip() for value in str(filters.get("model") or "").split(",")}
    ]
    if not selected:
        selected = list(MODEL_COMPARISON_ORDER)

    clauses, params = [], []
    if filters.get("date"):
        clauses.append("race_date=?"); params.append(filters["date"])
    if filters.get("track"):
        clauses.append("track_key(track)=track_key(?)"); params.append(filters["track"])
    if filters.get("race_no") not in (None, ""):
        clauses.append("race_no=?"); params.append(int(filters["race_no"]))
    if filters.get("outcome") == "correct":
        clauses.append("winner_rank=1")
    elif filters.get("outcome") == "incorrect":
        clauses.append("winner_rank>1")
    where = " WHERE " + " AND ".join(clauses) if clauses else ""

    stake = float(filters["stake"])
    models = []
    for model in selected:
        rows = connection.execute(
            DIAGNOSTICS_CTE + f"""
            SELECT winner_rank,winner_agf_rank,net_return
            FROM diagnostics{where}
            ORDER BY race_start_at,race_id""",
            [model, *params],
        ).fetchall()
        ranks = [int(row["winner_rank"]) for row in rows]
        agf_rows = [row for row in rows if row["winner_agf_rank"] is not None]
        model_over = sum(row["winner_rank"] < row["winner_agf_rank"] for row in agf_rows)
        agf_over = sum(row["winner_agf_rank"] < row["winner_rank"] for row in agf_rows)
        tied = len(agf_rows) - model_over - agf_over
        bet_returns = [float(row["net_return"]) for row in rows if row["net_return"] is not None]
        net_profit = stake * sum(bet_returns)
        invested = stake * len(bet_returns)
        distribution = Counter(ranks)
        count = len(ranks)
        models.append({
            "model": model,
            "evaluated_race_count": count,
            "top1_count": sum(rank == 1 for rank in ranks),
            "top3_count": sum(rank <= 3 for rank in ranks),
            "top5_count": sum(rank <= 5 for rank in ranks),
            "top1_accuracy": 100.0 * sum(rank == 1 for rank in ranks) / count if count else 0.0,
            "top3_accuracy": 100.0 * sum(rank <= 3 for rank in ranks) / count if count else 0.0,
            "top5_accuracy": 100.0 * sum(rank <= 5 for rank in ranks) / count if count else 0.0,
            "average_winner_rank": mean(ranks) if ranks else 0.0,
            "median_winner_rank": median(ranks) if ranks else 0.0,
            "net_profit": net_profit,
            "roi_percent": 100.0 * net_profit / invested if invested else 0.0,
            "model_over_agf": model_over,
            "agf_over_model": agf_over,
            "tied_with_agf": tied,
            "agf_evaluated_race_count": len(agf_rows),
            "model_advantage_percent": 100.0 * model_over / len(agf_rows) if agf_rows else 0.0,
            "rank_distribution": [
                {"winner_rank": rank, "race_count": distribution[rank]}
                for rank in sorted(distribution)
            ],
        })
    return {"models": models}


def export_rows(connection: sqlite3.Connection, filters: dict[str, Any]) -> Iterator[dict[str, Any]]:
    configure_sqlite(connection); where,params=_parts(filters)
    for row in connection.execute(PERFORMANCE_CTE+_row_sql(where)+" ORDER BY race_start_at DESC",params):
        yield _money(dict(row),filters["stake"])
