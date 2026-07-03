"""Production-safe post-race result fetch, matching and monitoring runner."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from app_config import DB_PATH, LOG_DIR, OUTPUT_DIR
from live_results_status import read_status_file, write_live_status
from pipeline_runner import run_step, runner_lock, write_run_log
from race_scope import normalize_country
from results_coverage import coverage_warnings

REQUIRED_COLUMNS = {
    "race_results": {
        "race_id",
        "horse_id",
        "race_start_at",
        "captured_at",
        "finish_position",
        "result_status",
    },
    "prediction_results": {"prediction_id", "matched_at"},
    "shadow_monitoring_runs": {
        "run_id",
        "run_at",
        "snapshot_coverage_pass",
        "pipeline_status",
        "details_json",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def emit(event: str, **fields: Any) -> None:
    print(
        json.dumps({"event": event, "at": utc_now(), **fields}, ensure_ascii=False),
        flush=True,
    )


def db_state() -> dict[str, Any]:
    """Read counts and validate the result store without mutating it."""
    connection = sqlite3.connect(
        f"file:{DB_PATH.as_posix()}?mode=ro", uri=True, timeout=30
    )
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        quick = connection.execute("PRAGMA quick_check").fetchone()[0]
        if quick != "ok":
            raise RuntimeError(f"SQLite quick_check failed: {quick}")
        for table, required in REQUIRED_COLUMNS.items():
            columns = {
                row[1] for row in connection.execute(f"PRAGMA table_info({table})")
            }
            missing = required - columns
            if missing:
                raise RuntimeError(f"Schema invalid: {table} missing {sorted(missing)}")
        integrity = []
        for table in ("race_results", "prediction_results"):
            integrity.extend(
                connection.execute(f"PRAGMA foreign_key_check({table})").fetchall()
            )
        if integrity:
            raise RuntimeError(
                f"Foreign-key integrity failed: {len(integrity)} violation(s)"
            )
        monitor = connection.execute(
            """SELECT run_id,run_at,leakage_gate_pass,feature_contract_pass,
                      snapshot_coverage_pass,prediction_drift_status,calibration_status,
                      feature_drift_status,pipeline_status,details_json
               FROM shadow_monitoring_runs ORDER BY run_at DESC LIMIT 1"""
        ).fetchone()
        return {
            "result_rows": connection.execute(
                "SELECT COUNT(*) FROM race_results"
            ).fetchone()[0],
            "prediction_result_rows": connection.execute(
                "SELECT COUNT(*) FROM prediction_results"
            ).fetchone()[0],
            "distinct_result_races_today": connection.execute(
                """SELECT COUNT(DISTINCT race_id) FROM race_results
                   WHERE date(race_start_at,'+3 hours')=date('now','+3 hours')"""
            ).fetchone()[0],
            "monitor": dict(monitor) if monitor else None,
        }
    finally:
        connection.close()


def monitoring_warnings(monitor: dict[str, Any] | None) -> list[str]:
    if not monitor:
        return ["shadow_monitor produced no monitoring record"]
    warnings: list[str] = []
    try:
        details = json.loads(monitor.get("details_json") or "{}")
    except json.JSONDecodeError:
        details = {}
        warnings.append("shadow_monitor details_json is invalid")
    if not bool(monitor.get("snapshot_coverage_pass")):
        warnings.append("snapshot_coverage_fail")
    missed = int(details.get("missed_shadow_races") or 0)
    if missed:
        warnings.append(f"missed_shadow_races={missed}")
    for field in (
        "prediction_drift_status",
        "calibration_status",
        "feature_drift_status",
    ):
        value = str(monitor.get(field) or details.get(field) or "")
        if value.upper() == "INSUFFICIENT_DATA":
            warnings.append(f"{field}=INSUFFICIENT_DATA")
        elif value.upper() == "CRITICAL":
            warnings.append(f"{field}=CRITICAL")
    if not bool(monitor.get("leakage_gate_pass")):
        warnings.append("leakage_gate_fail")
    if not bool(monitor.get("feature_contract_pass")):
        warnings.append("feature_contract_fail")
    pipeline_status = str(monitor.get("pipeline_status") or "")
    if pipeline_status and pipeline_status not in {"PASS"}:
        warnings.append(f"shadow_pipeline_status={pipeline_status}")
    return list(dict.fromkeys(warnings))


def result_coverage_warnings() -> list[str]:
    path = OUTPUT_DIR / "results_coverage_run.json"
    if not path.is_file():
        return ["results coverage artifact missing"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"results coverage artifact invalid: {exc}"]
    warnings = []
    for coverage in payload.get("dates", []):
        for warning in coverage_warnings(coverage or {}):
            warnings.append(f"date={coverage.get('date')} {warning}")
    return warnings


def forward_step_output(result: dict[str, Any]) -> None:
    stdout = str(result.get("stdout") or "")
    stderr = str(result.get("stderr") or "")
    if stdout:
        print(stdout, end="" if stdout.endswith("\n") else "\n", flush=True)
    if stderr:
        print(
            stderr,
            file=sys.stderr,
            end="" if stderr.endswith("\n") else "\n",
            flush=True,
        )


def base_payload() -> dict[str, Any]:
    return {
        "runner": "results_update",
        "status": "FAILED",
        "started_at": utc_now(),
        "ended_at": None,
        "duration_seconds": 0,
        "update_results_exit_code": 0,
        "shadow_monitor_exit_code": 0,
        "inserted_results_count": 0,
        "distinct_result_races_today": 0,
        "matched_predictions_count": 0,
        "warnings": [],
        "errors": [],
    }


def default_options() -> argparse.Namespace:
    return argparse.Namespace(
        date=None,
        today_tracks=False,
        country="ALL",
        track=[],
        skip_monitor=False,
        live_status=False,
    )


def main(options: argparse.Namespace | None = None) -> int:
    options = options or default_options()
    country = normalize_country(getattr(options, "country", "ALL"))
    target_date = getattr(options, "date", None)
    if getattr(options, "today_tracks", False) and not target_date:
        target_date = datetime.now(ZoneInfo("Europe/Istanbul")).date().isoformat()
    payload = base_payload()
    started = time.monotonic()
    service_exit_code = 1
    emit("results_update_started", started_at=payload["started_at"])
    try:
        if getattr(options, "live_status", False):
            live_date = (
                target_date
                or datetime.now(ZoneInfo("Europe/Istanbul")).date().isoformat()
            )
            previous = read_status_file(LOG_DIR / "live_results_status.json")
            write_live_status(
                DB_PATH,
                LOG_DIR / "live_results_status.json",
                live_date,
                country,
                {
                    "status": "RUNNING",
                    "started_at": payload["started_at"],
                    "last_run_at": previous.get("last_run_at")
                    or previous.get("ended_at"),
                    "warnings": [],
                    "errors": [],
                },
            )
        with runner_lock("results_update", skip_if_active=True) as lock:
            if not lock.acquired:
                payload["status"] = "SKIPPED_ALREADY_RUNNING"
                payload["warnings"].append(f"active_lock_owner={lock.metadata}")
                service_exit_code = 0
                emit(
                    "results_update_skipped",
                    reason="already_running",
                    owner=lock.metadata,
                )
            else:
                if lock.stale_lock_removed:
                    payload["warnings"].append("stale_lock_removed")
                    emit("stale_lock_removed", path=str(lock.path))
                before = db_state()

                update_args = ["--country", country]
                if target_date:
                    update_args += ["--date", target_date]
                if getattr(options, "today_tracks", False):
                    update_args.append("--today-tracks")
                for track in getattr(options, "track", []) or []:
                    update_args += ["--track", track]
                update = run_step("update_results.py", update_args, 1800)
                payload["update_results_exit_code"] = int(update["exit_code"])
                forward_step_output(update)
                emit(
                    "step_completed",
                    script="update_results.py",
                    exit_code=update["exit_code"],
                    started_at=update["started_at"],
                    ended_at=update["ended_at"],
                    duration_seconds=update["duration_seconds"],
                )

                # Pull same-day results directly from TJK CDN (CSV).
                # Runs regardless of API step outcome — CDN is the ground truth.
                cdn_args = ["--today"]
                if target_date:
                    cdn_args = ["--date", target_date]
                cdn = run_step("import_race_results_csv.py", cdn_args, 120)
                forward_step_output(cdn)
                emit(
                    "step_completed",
                    script="import_race_results_csv.py",
                    exit_code=cdn["exit_code"],
                    started_at=cdn["started_at"],
                    ended_at=cdn["ended_at"],
                    duration_seconds=cdn["duration_seconds"],
                )
                if int(cdn["exit_code"]) != 0:
                    payload["warnings"].append(
                        f"cdn_csv_import exit_code={cdn['exit_code']}"
                    )

                if int(update["exit_code"]) != 0:
                    payload["errors"].append(
                        f"update_results.py exit_code={update['exit_code']}"
                    )
                    payload["status"] = "FAILED"
                else:
                    after_update = db_state()
                    payload["warnings"].extend(result_coverage_warnings())
                    payload["inserted_results_count"] = max(
                        0, int(after_update["result_rows"]) - int(before["result_rows"])
                    )

                    if getattr(options, "skip_monitor", False):
                        final = after_update
                        payload["distinct_result_races_today"] = int(
                            final["distinct_result_races_today"]
                        )
                        payload["warnings"] = list(dict.fromkeys(payload["warnings"]))
                        payload["status"] = (
                            "WARNING" if payload["warnings"] else "SUCCESS"
                        )
                        service_exit_code = 0
                    else:
                        monitor_before_id = (after_update.get("monitor") or {}).get(
                            "run_id"
                        )
                        monitor_result = run_step("shadow_monitor.py", [], 1800)
                        payload["shadow_monitor_exit_code"] = int(
                            monitor_result["exit_code"]
                        )
                        forward_step_output(monitor_result)
                        emit(
                            "step_completed",
                            script="shadow_monitor.py",
                            exit_code=monitor_result["exit_code"],
                            started_at=monitor_result["started_at"],
                            ended_at=monitor_result["ended_at"],
                            duration_seconds=monitor_result["duration_seconds"],
                        )

                        final = db_state()
                        payload["distinct_result_races_today"] = int(
                            final["distinct_result_races_today"]
                        )
                        payload["matched_predictions_count"] = max(
                            0,
                            int(final["prediction_result_rows"])
                            - int(after_update["prediction_result_rows"]),
                        )
                        latest_monitor = final.get("monitor")
                        monitor_completed = bool(
                            latest_monitor
                            and latest_monitor.get("run_id") != monitor_before_id
                        )
                        monitor_exit = int(monitor_result["exit_code"])
                        has_traceback = "Traceback" in str(
                            monitor_result.get("stderr") or ""
                        )
                        monitoring_only_failure = (
                            monitor_exit == 1
                            and monitor_completed
                            and not has_traceback
                        )
                        if monitor_exit != 0 and not monitoring_only_failure:
                            payload["errors"].append(
                                f"shadow_monitor.py technical failure exit_code={monitor_result['exit_code']}"
                            )
                            payload["status"] = "FAILED"
                        else:
                            payload["warnings"].extend(
                                monitoring_warnings(latest_monitor)
                            )
                            if monitor_exit != 0:
                                payload["warnings"].append(
                                    f"shadow_monitor monitoring exit_code={monitor_result['exit_code']}"
                                )
                            payload["warnings"] = list(
                                dict.fromkeys(payload["warnings"])
                            )
                            payload["status"] = (
                                "WARNING" if payload["warnings"] else "SUCCESS"
                            )
                            service_exit_code = 0
    except Exception as exc:
        payload["status"] = "FAILED"
        payload["errors"].append(f"{type(exc).__name__}: {exc}")
        traceback.print_exc(file=sys.stderr)
        service_exit_code = 1
    finally:
        payload["ended_at"] = utc_now()
        payload["duration_seconds"] = round(time.monotonic() - started, 3)
        if getattr(options, "live_status", False):
            live_date = (
                target_date
                or datetime.now(ZoneInfo("Europe/Istanbul")).date().isoformat()
            )
            try:
                write_live_status(
                    DB_PATH,
                    LOG_DIR / "live_results_status.json",
                    live_date,
                    country,
                    {
                        "status": payload["status"],
                        "started_at": payload["started_at"],
                        "ended_at": payload["ended_at"],
                        "duration_seconds": payload["duration_seconds"],
                        "warnings": payload["warnings"],
                        "errors": payload["errors"],
                    },
                )
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                payload["status"] = "FAILED"
                payload["errors"].append(f"live status write failed: {exc}")
                service_exit_code = 1
        try:
            write_run_log("results_update", payload)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            payload["status"] = "FAILED"
            payload["errors"].append(f"results_update_latest.json write failed: {exc}")
            service_exit_code = 1
        emit(
            "results_update_finished",
            status=payload["status"],
            ended_at=payload["ended_at"],
            duration_seconds=payload["duration_seconds"],
            exit_code=service_exit_code,
            inserted_results_count=payload["inserted_results_count"],
            distinct_result_races_today=payload["distinct_result_races_today"],
            matched_predictions_count=payload["matched_predictions_count"],
            warnings=payload["warnings"],
            errors=payload["errors"],
        )
    return service_exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--today-tracks", action="store_true")
    parser.add_argument("--country", default="ALL")
    parser.add_argument("--track", action="append", default=[])
    parser.add_argument("--skip-monitor", action="store_true")
    parser.add_argument("--live-status", action="store_true")
    raise SystemExit(main(parser.parse_args()))
