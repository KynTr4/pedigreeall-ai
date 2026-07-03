"""Race-level final prediction freeze — multi-window capture with failure reason codes.

Freeze state machine (freeze_state column)
──────────────────────────────────────────
  WAITING            start > T+30 min
  PRE_WINDOW         T-30 to T-15 (monitoring only, no action)
  CAPTURING          T-15 to T-2  (run_agf_update + shadow_mode)
  FINAL_CAPTURING    T-2  to T+0  (emergency: skip all prereqs, just shadow_mode)
  POST_START_RETRY   T+0  to T+120s (TJK late-publish retry, up to 3 attempts)
  FINAL_CAPTURED     immutable prediction archived
  RESULT_PENDING     race finished, result not yet matched
  RESULT_CAPTURED    result matched
  SOURCE_UNSUPPORTED track not in supported list
  FAILED             all windows exhausted without prediction

Failure reason codes (failure_reason column)
─────────────────────────────────────────────
  NO_AGF           agf_snapshots empty within prediction window
  NO_ODDS          odds_snapshots empty within prediction window
  NO_FEATURES      asof_features.parquet has no rows for this race
  LATE_DISCOVERY   race entered program_snapshots after window closed
  LATE_TIMER       timer fired but window already at/past T-2
  CLOCK_DRIFT      server NTP offset > 2000 ms
  MODEL_ERROR      shadow_mode.py exited non-zero
  AGF_FETCH_FAILED run_agf_update.py exited non-zero
  RACE_STARTED     race already started when first discovered
  PIPELINE_FAILED  a prerequisite script exited non-zero (PermissionError etc.)
  UNKNOWN          unclassified
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from app_config import DB_PATH, TZ_NAME
from migrate_provenance_schema import apply_migrations
from pipeline_runner import run_step, runner_lock, write_run_log
from results_coverage import clean_track, track_policy

# ── Window constants ──────────────────────────────────────────────────────────
PRE_WINDOW_OPEN = timedelta(minutes=30)  # start monitoring
CAPTURE_WINDOW = timedelta(minutes=15)  # run AGF + predict
EMERGENCY_WINDOW = timedelta(minutes=2)  # skip prereqs, direct predict
POST_START_MAX = timedelta(seconds=120)  # post-start retry limit
POST_START_RETRIES_MAX = 3
NTP_WARN_MS = 2000


class FS:  # Freeze States
    WAITING = "WAITING"
    PRE_WINDOW = "PRE_WINDOW"
    CAPTURING = "CAPTURING"
    FINAL_CAPTURING = "FINAL_CAPTURING"
    POST_START_RETRY = "POST_START_RETRY"
    FINAL_CAPTURED = "FINAL_CAPTURED"
    RESULT_PENDING = "RESULT_PENDING"
    RESULT_CAPTURED = "RESULT_CAPTURED"
    SOURCE_UNSUPPORTED = "SOURCE_UNSUPPORTED"
    FAILED = "FAILED"


class FR:  # Failure Reasons
    NO_AGF = "NO_AGF"
    NO_ODDS = "NO_ODDS"
    NO_FEATURES = "NO_FEATURES"
    LATE_DISCOVERY = "LATE_DISCOVERY"
    LATE_TIMER = "LATE_TIMER"
    CLOCK_DRIFT = "CLOCK_DRIFT"
    MODEL_ERROR = "MODEL_ERROR"
    AGF_FETCH_FAILED = "AGF_FETCH_FAILED"
    RACE_STARTED = "RACE_STARTED"
    PIPELINE_FAILED = "PIPELINE_FAILED"
    UNKNOWN = "UNKNOWN"


def check_ntp_offset_ms() -> float | None:
    """Return the system clock offset in milliseconds, or None if unavailable."""
    # Try chronyc first (field index 4 = system time offset in seconds)
    try:
        proc = subprocess.run(
            ["chronyc", "-c", "tracking"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            fields = proc.stdout.strip().split(",")
            if len(fields) > 4:
                return float(fields[4]) * 1000.0
    except Exception:
        pass
    # Fall back to ntpq
    try:
        proc = subprocess.run(
            ["ntpq", "-c", "rv 0"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0:
            for token in proc.stdout.replace(",", " ").split():
                if token.startswith("offset="):
                    return float(token.split("=", 1)[1])
    except Exception:
        pass
    return None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _races(connection: sqlite3.Connection, target_date: str) -> list[dict[str, Any]]:
    connection.row_factory = sqlite3.Row
    return [
        dict(row)
        for row in connection.execute(
            """WITH ranked AS (
               SELECT *,ROW_NUMBER() OVER(
                   PARTITION BY race_id,horse_id ORDER BY captured_at DESC,snapshot_id DESC
               ) rn FROM program_snapshots WHERE date(race_start_at,'+3 hours')=?
           )
           SELECT race_id,MAX(race_start_at) race_start_at,MAX(track) track,MAX(race_no) race_no
           FROM ranked WHERE rn=1 GROUP BY race_id ORDER BY race_start_at""",
            (target_date,),
        )
    ]


def _facts(
    connection: sqlite3.Connection, race_id: str, start: datetime
) -> dict[str, Any]:
    window = (start - timedelta(minutes=15)).isoformat()
    prediction = connection.execute(
        """SELECT prediction_id,prediction_time FROM prediction_snapshots
           WHERE race_id=? AND prediction_time>=? AND prediction_time<race_start_at
           ORDER BY prediction_time DESC LIMIT 1""",
        (race_id, window),
    ).fetchone()
    result = connection.execute(
        """SELECT 1 FROM race_results
           WHERE race_id=? AND result_status='finished' AND finish_position=1 LIMIT 1""",
        (race_id,),
    ).fetchone()
    limit = prediction[1] if prediction else start.isoformat()
    agf = connection.execute(
        "SELECT MAX(captured_at) FROM agf_snapshots WHERE race_id=? AND captured_at<=?",
        (race_id, limit),
    ).fetchone()[0]
    odds = connection.execute(
        "SELECT MAX(captured_at) FROM odds_snapshots WHERE race_id=? AND captured_at<=?",
        (race_id, limit),
    ).fetchone()[0]
    lc = connection.execute(
        "SELECT first_seen_at,post_start_retries FROM race_prediction_lifecycle WHERE race_id=?",
        (race_id,),
    ).fetchone()
    return {
        "prediction": prediction,
        "has_result": bool(result),
        "agf": agf,
        "odds": odds,
        "first_seen_at": lc[0] if lc else None,
        "post_start_retries": lc[1] if lc else 0,
    }


def _classify_state(
    now: datetime,
    start: datetime,
    has_final: bool,
    has_result: bool,
    supported: bool,
) -> str:
    if not supported:
        return FS.SOURCE_UNSUPPORTED
    if has_result:
        return FS.RESULT_CAPTURED
    if has_final:
        return FS.RESULT_PENDING if now >= start else FS.FINAL_CAPTURED
    # No prediction yet:
    if now >= start + POST_START_MAX:
        return FS.FAILED
    if now >= start:
        return FS.POST_START_RETRY
    if now >= start - EMERGENCY_WINDOW:
        return FS.FINAL_CAPTURING
    if now >= start - CAPTURE_WINDOW:
        return FS.CAPTURING
    if now >= start - PRE_WINDOW_OPEN:
        return FS.PRE_WINDOW
    return FS.WAITING


def _determine_failure_reason(
    facts: dict[str, Any], now: datetime, start: datetime
) -> str:
    first_seen = facts.get("first_seen_at")
    if first_seen:
        try:
            seen_dt = datetime.fromisoformat(first_seen.replace("Z", "+00:00"))
            if seen_dt >= start:
                return FR.RACE_STARTED
            if seen_dt >= start - EMERGENCY_WINDOW:
                return FR.LATE_DISCOVERY
        except (ValueError, TypeError):
            pass
    if now >= start:
        return FR.RACE_STARTED
    if not facts.get("agf"):
        return FR.NO_AGF
    if not facts.get("odds"):
        return FR.NO_ODDS
    return FR.UNKNOWN


def _legacy_status(freeze_state: str) -> str:
    return {
        FS.WAITING: "WAITING",
        FS.PRE_WINDOW: "WAITING",
        FS.CAPTURING: "FINAL_REFRESH_DUE",
        FS.FINAL_CAPTURING: "FINAL_REFRESH_DUE",
        FS.POST_START_RETRY: "MISSED_FINAL_WINDOW",
        FS.FINAL_CAPTURED: "FINAL_PREDICTION_DONE",
        FS.RESULT_PENDING: "RESULT_PENDING",
        FS.RESULT_CAPTURED: "RESULT_CAPTURED",
        FS.SOURCE_UNSUPPORTED: "SOURCE_UNSUPPORTED",
        FS.FAILED: "MISSED_FINAL_WINDOW",
    }.get(freeze_state, freeze_state)


def _save(
    connection: sqlite3.Connection,
    race: dict[str, Any],
    now: datetime,
    freeze_state: str,
    facts: dict[str, Any],
    failure_reason: str | None = None,
    ntp_offset_ms: float | None = None,
) -> None:
    start = datetime.fromisoformat(str(race["race_start_at"]).replace("Z", "+00:00"))
    stamp = now.isoformat()
    prediction = facts.get("prediction")
    connection.execute(
        """INSERT INTO race_prediction_lifecycle(
               race_id,race_start_at,track,
               final_refresh_due_at,final_prediction_due_at,
               final_prediction_done_at,final_prediction_status,
               agf_snapshot_done_at,odds_snapshot_done_at,prediction_run_id,
               status,warning,created_at,updated_at,
               freeze_state,failure_reason,post_start_retries,ntp_offset_ms,first_seen_at
           )VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(race_id) DO UPDATE SET
               race_start_at=excluded.race_start_at,
               track=excluded.track,
               final_refresh_due_at=excluded.final_refresh_due_at,
               final_prediction_due_at=excluded.final_prediction_due_at,
               final_prediction_done_at=COALESCE(
                   excluded.final_prediction_done_at,
                   race_prediction_lifecycle.final_prediction_done_at),
               final_prediction_status=excluded.final_prediction_status,
               agf_snapshot_done_at=COALESCE(
                   excluded.agf_snapshot_done_at,
                   race_prediction_lifecycle.agf_snapshot_done_at),
               odds_snapshot_done_at=COALESCE(
                   excluded.odds_snapshot_done_at,
                   race_prediction_lifecycle.odds_snapshot_done_at),
               prediction_run_id=COALESCE(
                   excluded.prediction_run_id,
                   race_prediction_lifecycle.prediction_run_id),
               status=excluded.status,
               warning=excluded.warning,
               updated_at=excluded.updated_at,
               freeze_state=excluded.freeze_state,
               failure_reason=COALESCE(
                   excluded.failure_reason,
                   race_prediction_lifecycle.failure_reason),
               ntp_offset_ms=excluded.ntp_offset_ms,
               first_seen_at=COALESCE(
                   race_prediction_lifecycle.first_seen_at,
                   excluded.first_seen_at)""",
        (
            race["race_id"],
            race["race_start_at"],
            clean_track(race.get("track")),
            (start - timedelta(minutes=15)).isoformat(),  # final_refresh_due_at
            (start - timedelta(minutes=10)).isoformat(),  # final_prediction_due_at
            prediction[1] if prediction else None,  # final_prediction_done_at
            "DONE" if prediction else freeze_state,  # final_prediction_status
            facts.get("agf"),  # agf_snapshot_done_at
            facts.get("odds"),  # odds_snapshot_done_at
            prediction[0].split(":", 1)[0] if prediction else None,  # prediction_run_id
            _legacy_status(freeze_state),  # status
            failure_reason if freeze_state == FS.FAILED else None,  # warning
            stamp,  # created_at
            stamp,  # updated_at
            freeze_state,  # freeze_state
            failure_reason,  # failure_reason
            0,  # post_start_retries (new rows default)
            ntp_offset_ms,  # ntp_offset_ms
            stamp,  # first_seen_at (kept by COALESCE if exists)
        ),
    )


def process(
    target_date: str,
    now: datetime,
    db_path: str | Path = DB_PATH,
    step_runner=run_step,
) -> dict[str, Any]:
    apply_migrations(db_path)
    ntp_ms = check_ntp_offset_ms()
    ntp_warning = ntp_ms is not None and abs(ntp_ms) > NTP_WARN_MS

    with sqlite3.connect(str(db_path), timeout=60) as conn:
        races = _races(conn, target_date)
        capture_now: list[tuple[dict[str, Any], str]] = []
        emergency_now: list[
            tuple[dict[str, Any], str]
        ] = []  # FINAL_CAPTURING only, future use
        post_retry: list[dict[str, Any]] = []

        for race in races:
            start = datetime.fromisoformat(
                str(race["race_start_at"]).replace("Z", "+00:00")
            )
            facts = _facts(conn, race["race_id"], start)
            supported = track_policy(race.get("track")) != "unsupported"
            state = _classify_state(
                now, start, bool(facts["prediction"]), facts["has_result"], supported
            )
            failure = (
                _determine_failure_reason(facts, now, start)
                if state == FS.FAILED
                else None
            )
            _save(conn, race, now, state, facts, failure, ntp_ms)
            if not facts["prediction"] and supported and not facts["has_result"]:
                if state in (FS.CAPTURING, FS.FINAL_CAPTURING):
                    capture_now.append((race, state))
                elif (
                    state == FS.POST_START_RETRY
                    and facts.get("post_start_retries", 0) < POST_START_RETRIES_MAX
                ):
                    post_retry.append(race)
        conn.commit()

    steps: list[dict[str, Any]] = []
    agf_ok = True

    # --- MAIN CAPTURE: run_agf_update then shadow_mode ---
    if capture_now:
        agf_result = step_runner("run_agf_update.py", [], 300)
        steps.append(agf_result)
        agf_ok = int(agf_result["exit_code"]) == 0
        if not agf_ok:
            # Mark failure reason for all due races (best-effort, per-race connection)
            for race, _ in capture_now:
                with sqlite3.connect(str(db_path), timeout=60) as conn:
                    conn.execute(
                        "UPDATE race_prediction_lifecycle"
                        " SET failure_reason=?,updated_at=?"
                        " WHERE race_id=? AND failure_reason IS NULL",
                        (FR.AGF_FETCH_FAILED, now.isoformat(), race["race_id"]),
                    )
                    conn.commit()
        # Still try shadow_mode even with stale AGF data
        for race, state in capture_now:
            result = step_runner(
                "shadow_mode.py",
                ["--date", target_date, "--race-id", race["race_id"], "--final-freeze"],
                600,
            )
            steps.append(result)
            if int(result["exit_code"]) != 0 and state == FS.CAPTURING:
                break  # stop on CAPTURING failure; emergency races get separate attempt

    # --- POST-START RETRY: best-effort AGF refresh, then shadow_mode ---
    if post_retry:
        step_runner("run_agf_update.py", [], 120)  # best-effort, don't track failure
        for race in post_retry:
            result = step_runner(
                "shadow_mode.py",
                ["--date", target_date, "--race-id", race["race_id"], "--final-freeze"],
                120,
            )
            steps.append(result)

    # --- Re-evaluate all processed races ---
    all_processed = [r for r, _ in capture_now] + post_retry
    if all_processed:
        now_updated = datetime.now(timezone.utc)
        with sqlite3.connect(str(db_path), timeout=60) as conn:
            for race in all_processed:
                start = datetime.fromisoformat(
                    str(race["race_start_at"]).replace("Z", "+00:00")
                )
                facts = _facts(conn, race["race_id"], start)
                supported = track_policy(race.get("track")) != "unsupported"
                state = _classify_state(
                    now_updated,
                    start,
                    bool(facts["prediction"]),
                    facts["has_result"],
                    supported,
                )
                failure = None
                warning_txt = None
                if state == FS.FAILED:
                    failure = _determine_failure_reason(facts, now_updated, start)
                    if not agf_ok:
                        failure = FR.AGF_FETCH_FAILED
                    warning_txt = f"No prediction created [{failure}]"
                elif state == FS.POST_START_RETRY:
                    conn.execute(
                        "UPDATE race_prediction_lifecycle"
                        " SET post_start_retries=post_start_retries+1 WHERE race_id=?",
                        (race["race_id"],),
                    )
                _save(conn, race, now_updated, state, facts, failure, ntp_ms)
                if warning_txt:
                    conn.execute(
                        "UPDATE race_prediction_lifecycle SET warning=? WHERE race_id=?",
                        (warning_txt, race["race_id"]),
                    )
            conn.commit()

    return {
        "date": target_date,
        "now": now.isoformat(),
        "ntp_offset_ms": ntp_ms,
        "ntp_warning": ntp_warning,
        "race_count": len(races),
        "capture_races": [r["race_id"] for r, _ in capture_now],
        "post_retry_races": [r["race_id"] for r in post_retry],
        "steps": steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date", default=datetime.now(ZoneInfo(TZ_NAME)).date().isoformat()
    )
    parser.add_argument("--now", help="UTC ISO timestamp for deterministic validation")
    args = parser.parse_args()
    now = (
        datetime.fromisoformat(args.now.replace("Z", "+00:00"))
        if args.now
        else datetime.now(timezone.utc)
    )
    payload: dict[str, Any] = {
        "runner": "race_freeze",
        "started_at": utc_now(),
        "status": "FAILED",
    }
    with runner_lock("race_freeze", skip_if_active=True) as lock:
        if not lock.acquired:
            payload.update(
                {"status": "SKIPPED_ALREADY_RUNNING", "owner": lock.metadata}
            )
        else:
            payload.update(process(args.date, now))
            failed = [s for s in payload["steps"] if int(s["exit_code"]) != 0]
            payload["status"] = "FAILED" if failed else "SUCCESS"
    payload["ended_at"] = utc_now()
    write_run_log("race_freeze", payload)
    print(json.dumps(payload, ensure_ascii=False, default=str))
    return 1 if payload["status"] == "FAILED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
