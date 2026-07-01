"""Race-level final AGF/odds refresh and immutable prediction freeze runner."""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from app_config import DB_PATH, TZ_NAME
from migrate_provenance_schema import apply_migrations
from pipeline_runner import run_step, runner_lock, write_run_log
from results_coverage import clean_track, track_policy


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def classify_race(now: datetime, start: datetime, has_final: bool, has_result: bool,
                  supported: bool) -> str:
    if not supported:
        return "SOURCE_UNSUPPORTED"
    if has_result:
        return "RESULT_CAPTURED"
    if has_final:
        return "RESULT_PENDING" if now >= start else "FINAL_PREDICTION_DONE"
    if now >= start - timedelta(minutes=5):
        return "MISSED_FINAL_WINDOW"
    if now >= start - timedelta(minutes=15):
        return "FINAL_REFRESH_DUE"
    return "WAITING"


def _races(connection: sqlite3.Connection, target_date: str) -> list[dict[str, Any]]:
    connection.row_factory = sqlite3.Row
    return [dict(row) for row in connection.execute(
        """WITH ranked AS (
               SELECT *,ROW_NUMBER() OVER(
                   PARTITION BY race_id,horse_id ORDER BY captured_at DESC,snapshot_id DESC
               ) rn FROM program_snapshots WHERE date(race_start_at,'+3 hours')=?
           )
           SELECT race_id,MAX(race_start_at) race_start_at,MAX(track) track,MAX(race_no) race_no
           FROM ranked WHERE rn=1 GROUP BY race_id ORDER BY race_start_at""", (target_date,)
    )]


def _facts(connection: sqlite3.Connection, race_id: str, start: datetime) -> dict[str, Any]:
    window = (start - timedelta(minutes=15)).isoformat()
    prediction = connection.execute(
        """SELECT prediction_id,prediction_time FROM prediction_snapshots
           WHERE race_id=? AND julianday(prediction_time)>=julianday(?)
             AND julianday(prediction_time)<julianday(race_start_at)
           ORDER BY prediction_time DESC LIMIT 1""", (race_id, window)
    ).fetchone()
    result = connection.execute(
        """SELECT 1 FROM race_results WHERE race_id=? AND result_status='finished'
           AND finish_position=1 LIMIT 1""", (race_id,)
    ).fetchone()
    limit = prediction[1] if prediction else start.isoformat()
    agf = connection.execute(
        "SELECT MAX(captured_at) FROM agf_snapshots WHERE race_id=? AND julianday(captured_at)<=julianday(?)",
        (race_id, limit),
    ).fetchone()[0]
    odds = connection.execute(
        "SELECT MAX(captured_at) FROM odds_snapshots WHERE race_id=? AND julianday(captured_at)<=julianday(?)",
        (race_id, limit),
    ).fetchone()[0]
    return {"prediction": prediction, "has_result": bool(result), "agf": agf, "odds": odds}


def _save(connection: sqlite3.Connection, race: dict[str, Any], now: datetime,
          status: str, facts: dict[str, Any], warning: str | None = None) -> None:
    start = datetime.fromisoformat(str(race["race_start_at"]).replace("Z", "+00:00"))
    stamp = now.isoformat(); prediction = facts.get("prediction")
    connection.execute(
        """INSERT INTO race_prediction_lifecycle(
               race_id,race_start_at,track,final_refresh_due_at,final_prediction_due_at,
               final_prediction_done_at,final_prediction_status,agf_snapshot_done_at,
               odds_snapshot_done_at,prediction_run_id,status,warning,created_at,updated_at)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(race_id) DO UPDATE SET
               race_start_at=excluded.race_start_at,track=excluded.track,
               final_refresh_due_at=excluded.final_refresh_due_at,
               final_prediction_due_at=excluded.final_prediction_due_at,
               final_prediction_done_at=COALESCE(excluded.final_prediction_done_at,race_prediction_lifecycle.final_prediction_done_at),
               final_prediction_status=excluded.final_prediction_status,
               agf_snapshot_done_at=COALESCE(excluded.agf_snapshot_done_at,race_prediction_lifecycle.agf_snapshot_done_at),
               odds_snapshot_done_at=COALESCE(excluded.odds_snapshot_done_at,race_prediction_lifecycle.odds_snapshot_done_at),
               prediction_run_id=COALESCE(excluded.prediction_run_id,race_prediction_lifecycle.prediction_run_id),
               status=excluded.status,warning=excluded.warning,updated_at=excluded.updated_at""",
        (race["race_id"], race["race_start_at"], clean_track(race.get("track")),
         (start - timedelta(minutes=15)).isoformat(), (start - timedelta(minutes=10)).isoformat(),
         prediction[1] if prediction else None, "DONE" if prediction else status,
         facts.get("agf"), facts.get("odds"), prediction[0].split(":", 1)[0] if prediction else None,
         status, warning, stamp, stamp),
    )


def process(target_date: str, now: datetime, db_path: str | Path = DB_PATH,
            step_runner=run_step) -> dict[str, Any]:
    apply_migrations(db_path)
    with sqlite3.connect(str(db_path), timeout=60) as connection:
        races = _races(connection, target_date)
        due = []
        for race in races:
            start = datetime.fromisoformat(str(race["race_start_at"]).replace("Z", "+00:00"))
            facts = _facts(connection, race["race_id"], start)
            supported = track_policy(race.get("track")) != "unsupported"
            status = classify_race(now, start, bool(facts["prediction"]), facts["has_result"], supported)
            _save(connection, race, now, status, facts)
            if status == "FINAL_REFRESH_DUE":
                due.append(race)
        connection.commit()

    steps = []
    if due:
        for script, args, timeout in (
            ("update_race_programs.py", [], 900),
            ("run_agf_update.py", [], 900),
            ("build_asof_features.py", [], 1800),
            ("validate_feature_provenance.py", [], 900),
        ):
            result = step_runner(script, args, timeout); steps.append(result)
            if int(result["exit_code"]) != 0:
                break
        prerequisite_ok = not steps or all(int(step["exit_code"]) == 0 for step in steps)
        if prerequisite_ok:
            for race in due:
                result = step_runner(
                    "shadow_mode.py", ["--date", target_date, "--race-id", race["race_id"], "--final-freeze"], 1800
                ); steps.append(result)
                if int(result["exit_code"]) != 0:
                    break
        with sqlite3.connect(str(db_path), timeout=60) as connection:
            for race in due:
                start = datetime.fromisoformat(str(race["race_start_at"]).replace("Z", "+00:00"))
                facts = _facts(connection, race["race_id"], start)
                status = classify_race(now, start, bool(facts["prediction"]), facts["has_result"], True)
                warning = None if facts["prediction"] else "Final prediction step did not create an immutable snapshot"
                _save(connection, race, now, status, facts, warning)
            connection.commit()
    return {"date": target_date, "now": now.isoformat(), "race_count": len(races),
            "due_races": [race["race_id"] for race in due], "steps": steps}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=datetime.now(ZoneInfo(TZ_NAME)).date().isoformat())
    parser.add_argument("--now", help="UTC ISO timestamp for deterministic validation")
    args = parser.parse_args()
    now = datetime.fromisoformat(args.now.replace("Z", "+00:00")) if args.now else datetime.now(timezone.utc)
    payload = {"runner": "race_freeze", "started_at": utc_now(), "status": "FAILED"}
    with runner_lock("race_freeze", skip_if_active=True) as lock:
        if not lock.acquired:
            payload.update({"status": "SKIPPED_ALREADY_RUNNING", "owner": lock.metadata})
        else:
            payload.update(process(args.date, now)); failed = [s for s in payload["steps"] if int(s["exit_code"]) != 0]
            payload["status"] = "FAILED" if failed else "SUCCESS"
    payload["ended_at"] = utc_now(); write_run_log("race_freeze", payload)
    print(json.dumps(payload, ensure_ascii=False, default=str))
    return 1 if payload["status"] == "FAILED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
