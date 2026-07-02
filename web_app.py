"""Read-only FastAPI dashboard for the shadow prediction pipeline."""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import platform
import secrets
import shutil
import sqlite3
import subprocess
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from app_config import (
    BACKUP_DIR,
    DB_PATH,
    LOG_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    WEB_HOST,
    WEB_PASSWORD,
    WEB_PORT,
    WEB_USERNAME,
    TZ_NAME,
)
from performance_queries import (
    chart_data as query_performance_chart,
    history as query_performance_history,
    model_comparison as query_performance_models,
    normalize_filters,
    race_filters as query_performance_races,
    summary as query_performance_summary,
)
from diagnostics_queries import (
    export_rows as query_diagnostics_export,
    extremes as query_diagnostics_extremes,
    feature_contribution_status,
    filter_options as query_diagnostics_filters,
    group_performance as query_diagnostics_groups,
    normalize_filters as normalize_diagnostics_filters,
    races as query_diagnostics_races,
    race_detail as query_diagnostics_race_detail,
    summary as query_diagnostics_summary,
    winner_ranks as query_diagnostics_winner_ranks,
)
from results_coverage import build_results_coverage
from race_day_queries import missing_horses, race_day_performance, race_day_summary, race_day_view, validate_date
from race_scope import configure_sqlite, normalize_country, track_key
from live_results_status import read_status_file, verified_status
from bet_simulator_queries import (
    export_rows as query_bet_export, history as query_bet_history,
    normalize_bet_filters, summary as query_bet_summary,
)

WEB_ROOT = PROJECT_ROOT / "web"
TEMPLATES = Jinja2Templates(directory=str(WEB_ROOT / "templates"))
ALLOWED_REPORTS = (
    "model_health_dashboard.md", "daily_shadow_report.md", "live_accuracy_report.md",
    "model_drift_report.md", "feature_drift_report.md", "calibration_monitor.md",
    "live_roi_report.md", "leakage_gate_v2.md", "vps_healthcheck.md",
    "results_coverage_latest.md", "izmir_results_debug.md",
    "race_day_dashboard_validation.md",
    "model_diagnostics_validation.md",
    "turkey_live_results_validation.md",
    "race_diagnostics_explainability_validation.md",
)
ALLOWED_LOGS = (
    "daily.log", "daily.err.log", "agf.log", "agf.err.log",
    "results.log", "results.err.log", "web.log", "web.err.log",
    "race-freeze.log", "race-freeze.err.log",
)
SYSTEMD_UNITS = (
    "at-yaris-web.service", "at-yaris-daily.timer", "at-yaris-agf-update.timer",
    "at-yaris-results-update.timer", "at-yaris-backup.timer",
    "at-yaris-live-results.timer",
    "at-yaris-race-freeze.timer",
)
_PERFORMANCE_CACHE: dict[tuple[Any, ...], tuple[float, Any]] = {}
_PERFORMANCE_CACHE_LOCK = threading.Lock()

app = FastAPI(title="AT Yarış Shadow Dashboard", docs_url=None, redoc_url=None)


def _unauthorized() -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={"detail": "Authentication required"},
        headers={"WWW-Authenticate": 'Basic realm="AT Yaris Dashboard"'},
    )


@app.middleware("http")
async def basic_auth(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)
    header = request.headers.get("Authorization", "")
    authenticated = False
    if header.startswith("Basic "):
        try:
            decoded = base64.b64decode(header[6:], validate=True).decode("utf-8")
            username, password = decoded.split(":", 1)
            authenticated = secrets.compare_digest(username, WEB_USERNAME) and secrets.compare_digest(
                password, WEB_PASSWORD
            )
        except (ValueError, UnicodeDecodeError):
            authenticated = False
    if not authenticated:
        return _unauthorized()
    return await call_next(request)


@app.get("/health")
async def health():
    try:
        connection = readonly_connection()
        connection.execute("SELECT 1").fetchone()
        connection.close()
        db_status = "ok"
    except Exception as e:
        db_status = f"error: {e}"

    commit_hash = "unknown"
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
        ).strip()
    except Exception:
        pass

    return {
        "status": "healthy" if db_status == "ok" else "unhealthy",
        "database": db_status,
        "commit": commit_hash,
    }


def readonly_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    configure_sqlite(connection)
    return connection


def _iso(value: Any) -> str | None:
    if value in (None, ""):
        return None
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(parsed) else parsed.isoformat()


def _latest_runner(name: str) -> dict[str, Any]:
    for directory in (LOG_DIR, PROJECT_ROOT / "logs"):
        path = directory / f"{name}_latest.json"
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                return {"status": "invalid", "error": str(exc)}
    return {"status": "missing"}


def _report_pass(name: str, positive: str = "PASS") -> bool:
    path = REPORTS_DIR / name
    if not path.is_file():
        return False
    return positive.lower() in path.read_text(encoding="utf-8", errors="replace").lower()


def systemd_status() -> dict[str, str]:
    if platform.system() != "Linux" or not shutil.which("systemctl"):
        return {unit: "unavailable_on_this_host" for unit in SYSTEMD_UNITS}
    result = {}
    for unit in SYSTEMD_UNITS:
        process = subprocess.run(
            ["systemctl", "is-active", unit], capture_output=True, text=True, timeout=5, check=False
        )
        result[unit] = process.stdout.strip() or "unknown"
    return result


def dashboard_status() -> dict[str, Any]:
    with readonly_connection() as connection:
        monitor = connection.execute(
            """SELECT * FROM shadow_monitoring_runs ORDER BY run_at DESC LIMIT 1"""
        ).fetchone()
        shadow_days = connection.execute("SELECT COUNT(DISTINCT shadow_date) FROM shadow_monitoring_runs").fetchone()[0]
        last_prediction = connection.execute("SELECT MAX(prediction_time) FROM prediction_snapshots").fetchone()[0]
    monitor = dict(monitor) if monitor else {}
    shadow_days = int(shadow_days or 0)
    daily = _latest_runner("run")
    agf = _latest_runner("agf_update")
    results = _latest_runner("results_update")
    backups = sorted(BACKUP_DIR.glob("*/*.tar.gz"), key=lambda path: path.stat().st_mtime, reverse=True)
    usage = shutil.disk_usage(PROJECT_ROOT)
    error_logs = []
    for name in ALLOWED_LOGS:
        path = LOG_DIR / name
        if path.is_file() and path.stat().st_size:
            tail = "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-200:])
            if any(token in tail.lower() for token in ("error", "critical", "traceback")):
                error_logs.append(name)
    leakage = bool(monitor.get("leakage_gate_pass")) or _report_pass("leakage_gate_v2.md")
    contract = bool(monitor.get("feature_contract_pass"))
    coverage = bool(monitor.get("snapshot_coverage_pass"))
    production_ready = (
        bool(monitor.get("production_ready")) and shadow_days >= 90
        and leakage and contract and coverage and not error_logs
    )
    return {
        "system_status": "healthy" if leakage and contract and coverage and not error_logs else "attention",
        "production_ready": production_ready,
        "shadow_days": shadow_days,
        "leakage_gate": leakage,
        "feature_contract": contract,
        "snapshot_coverage": coverage,
        "last_pipeline_at": daily.get("ended_at"),
        "last_pipeline_status": daily.get("status"),
        "last_agf_at": agf.get("ended_at"),
        "last_agf_status": agf.get("status"),
        "last_results_at": results.get("ended_at"),
        "last_results_status": results.get("status"),
        "last_prediction_at": _iso(last_prediction),
        "last_backup_at": datetime.fromtimestamp(backups[0].stat().st_mtime, timezone.utc).isoformat() if backups else None,
        "disk_used_percent": round((usage.used / usage.total) * 100, 1),
        "disk_free_gb": round(usage.free / 1024**3, 2),
        "active_errors": error_logs,
        "latest_monitor": monitor,
    }


def today_races() -> list[dict[str, Any]]:
    now = pd.Timestamp.now(tz="UTC")
    local_date = pd.Timestamp.now(tz=os.environ.get("TZ", "Europe/Istanbul")).date().isoformat()
    with readonly_connection() as connection:
        rows = connection.execute(
            """WITH latest AS (
                   SELECT *,ROW_NUMBER() OVER(
                       PARTITION BY race_id,horse_id ORDER BY captured_at DESC,snapshot_id DESC
                   ) AS rn FROM program_snapshots
               ), agf AS (
                   SELECT race_id,MAX(captured_at) AS last_agf_at FROM agf_snapshots GROUP BY race_id
               )
               SELECT l.race_id,l.race_start_at,l.race_no,l.track,l.surface,
                      COUNT(*) AS horse_count,
                      SUM(CASE WHEN julianday(l.captured_at)<julianday(l.race_start_at) THEN 1 ELSE 0 END) AS covered,
                      a.last_agf_at
               FROM latest l LEFT JOIN agf a USING(race_id)
               WHERE l.rn=1 AND substr(l.race_start_at,1,10)=?
               GROUP BY l.race_id,l.race_start_at,l.race_no,l.track,l.surface,a.last_agf_at
               ORDER BY l.race_start_at,l.race_no""",
            (local_date,),
        ).fetchall()
        result_coverage = build_results_coverage(connection, local_date)
    coverage_by_race = {row["race_id"]: row for row in result_coverage["races"]}
    races = []
    for row in rows:
        item = dict(row)
        count = int(item["horse_count"] or 0)
        item["snapshot_coverage"] = round(100 * int(item["covered"] or 0) / count, 1) if count else 0.0
        start = pd.to_datetime(item["race_start_at"], utc=True, errors="coerce")
        item["started"] = bool(not pd.isna(start) and now >= start)
        item["race_start_at"] = _iso(item["race_start_at"])
        item["last_agf_at"] = _iso(item["last_agf_at"])
        result = coverage_by_race.get(item["race_id"], {})
        item["result_status"] = result.get("status", "Sonuç bekleniyor")
        item["result_missing_reason"] = result.get("missing_reason", "source_not_published")
        item["tjk_id_missing_horse_count"] = result.get("tjk_id_missing_horse_count", 0)
        item["source_not_published_count"] = result.get("source_not_published_count", 0)
        races.append(item)
    return races


def current_predictions() -> list[dict[str, Any]]:
    with readonly_connection() as connection:
        rows = connection.execute(
            """WITH predictions AS (
                   SELECT *,ROW_NUMBER() OVER(
                       PARTITION BY race_id,horse_id ORDER BY prediction_time DESC,prediction_id DESC
                   ) AS rn FROM prediction_snapshots
               ), programs AS (
                   SELECT race_id,horse_id,horse_name,track,ROW_NUMBER() OVER(
                       PARTITION BY race_id,horse_id ORDER BY captured_at DESC,snapshot_id DESC
                   ) AS rn FROM program_snapshots
               )
               SELECT p.race_id,p.race_start_at,p.horse_id,COALESCE(g.horse_name,p.horse_id) AS horse_name,
                      p.logistic_probability,p.catboost_probability,p.xgboost_probability,
                      p.ensemble_probability,p.predicted_rank,p.feature_hash,p.prediction_time
               FROM predictions p LEFT JOIN programs g
                 ON g.race_id=p.race_id AND g.horse_id=p.horse_id AND g.rn=1
               WHERE p.rn=1
               ORDER BY p.race_start_at,p.race_id,p.predicted_rank"""
        ).fetchall()
    items = [dict(row) for row in rows]
    sums: dict[str, float] = defaultdict(float)
    for item in items:
        sums[item["race_id"]] += float(item["ensemble_probability"] or 0)
    for item in items:
        item["probability_sum"] = round(sums[item["race_id"]], 8)
        item["probability_sum_valid"] = abs(sums[item["race_id"]] - 1.0) <= 1e-6
        item["race_start_at"] = _iso(item["race_start_at"])
        item["prediction_time"] = _iso(item["prediction_time"])
    return items


def _performance_filters(date=None, track=None, model=None, outcome="all") -> dict[str, str | None]:
    try:
        return normalize_filters(date=date, track=track, model=model, outcome=outcome)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _performance_cached(name: str, filters: dict[str, Any], loader, page: int | None = None):
    key = (name, *sorted(filters.items()), page)
    now = time.monotonic()
    with _PERFORMANCE_CACHE_LOCK:
        cached = _PERFORMANCE_CACHE.get(key)
        if cached and now - cached[0] <= 30:
            return cached[1]
    with readonly_connection() as connection:
        value = loader(connection, filters) if page is None else loader(connection, filters, page)
    with _PERFORMANCE_CACHE_LOCK:
        _PERFORMANCE_CACHE[key] = (now, value)
        if len(_PERFORMANCE_CACHE) > 256:
            oldest = min(_PERFORMANCE_CACHE, key=lambda item: _PERFORMANCE_CACHE[item][0])
            _PERFORMANCE_CACHE.pop(oldest, None)
    return value


def _diagnostics_filters(date=None, track=None, model=None, race_type=None,
                         distance=None, surface=None, field_size=None) -> dict[str, str | None]:
    try:
        return normalize_diagnostics_filters(date, track, model, race_type, distance, surface, field_size)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _race_day_date(value: str | None) -> str:
    candidate = value or pd.Timestamp.now(tz=os.environ.get("TZ", "Europe/Istanbul")).date().isoformat()
    try:
        return validate_date(candidate)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _race_day_cached(name: str, target_date: str, track: str | None, country: str, loader):
    key = (f"race_day_{name}", target_date, track, country)
    now = time.monotonic()
    with _PERFORMANCE_CACHE_LOCK:
        cached = _PERFORMANCE_CACHE.get(key)
        if cached and now - cached[0] <= 30:
            return cached[1]
    with readonly_connection() as connection:
        value = loader(connection)
    with _PERFORMANCE_CACHE_LOCK:
        _PERFORMANCE_CACHE[key] = (now, value)
    return value


def _allowed_text(directory: Path, name: str, allowlist: tuple[str, ...], lines: int | None = None) -> str:
    if name not in allowlist:
        raise HTTPException(status_code=404, detail="File is not allowed")
    path = directory / name
    if not path.is_file():
        return "File is not available."
    content = path.read_text(encoding="utf-8", errors="replace")
    return "\n".join(content.splitlines()[-lines:]) if lines else content


@app.get("/static/style.css", response_class=FileResponse)
def static_style():
    return FileResponse(WEB_ROOT / "static" / "style.css", media_type="text/css")


@app.get("/static/live-results.js", response_class=FileResponse)
def static_live_results():
    return FileResponse(WEB_ROOT / "static" / "live-results.js", media_type="application/javascript")


def server_today() -> str:
    return datetime.now(ZoneInfo(TZ_NAME)).date().isoformat()


def _page_date(value: str | None) -> str:
    if value is None:
        return server_today()
    try:
        return validate_date(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return TEMPLATES.TemplateResponse(request, "dashboard.html", {"data": dashboard_status()})


@app.get("/races", response_class=HTMLResponse)
def races_page(request: Request, date: str | None = None):
    return TEMPLATES.TemplateResponse(request, "races.html", {"selected_date": _page_date(date)})


@app.get("/predictions", response_class=HTMLResponse)
def predictions_page(request: Request):
    return TEMPLATES.TemplateResponse(request, "predictions.html", {"predictions": current_predictions()})


@app.get("/performance", response_class=HTMLResponse)
def performance_page(request: Request, date: str | None = None):
    return TEMPLATES.TemplateResponse(
        request, "performance.html", {"selected_date": _page_date(date)}
    )


@app.get("/diagnostics", response_class=HTMLResponse)
def diagnostics_page(request: Request, date: str | None = None):
    return TEMPLATES.TemplateResponse(
        request, "diagnostics.html", {"selected_date": _page_date(date)}
    )


@app.get("/diagnostics/race/{race_id}", response_class=HTMLResponse)
def diagnostics_race_page(request: Request, race_id: str):
    return TEMPLATES.TemplateResponse(request, "diagnostics_race.html", {"race_id": race_id})


@app.get("/bet-simulator", response_class=HTMLResponse)
def bet_simulator_page(request: Request, date: str | None = None):
    return TEMPLATES.TemplateResponse(
        request, "bet_simulator.html", {"selected_date": _page_date(date)}
    )


@app.get("/reports", response_class=HTMLResponse)
def reports_page(request: Request, name: str | None = None):
    selected = name if name in ALLOWED_REPORTS else None
    content = _allowed_text(REPORTS_DIR, selected, ALLOWED_REPORTS) if selected else None
    return TEMPLATES.TemplateResponse(
        request, "reports.html", {"files": ALLOWED_REPORTS, "selected": selected, "content": content}
    )


@app.get("/logs", response_class=HTMLResponse)
def logs_page(request: Request, name: str | None = None):
    selected = name if name in ALLOWED_LOGS else None
    content = _allowed_text(LOG_DIR, selected, ALLOWED_LOGS, 200) if selected else None
    return TEMPLATES.TemplateResponse(
        request, "logs.html", {"files": ALLOWED_LOGS, "selected": selected, "content": content}
    )


@app.get("/api/health")
def api_health():
    data = dashboard_status()
    return {"status": data["system_status"], "production_ready": data["production_ready"], "checks": data}


@app.get("/api/today-races")
def api_today_races():
    return {"count": len(races := today_races()), "races": races}


@app.get("/api/predictions")
def api_predictions():
    predictions = current_predictions()
    invalid = sorted({row["race_id"] for row in predictions if not row["probability_sum_valid"]})
    return {"count": len(predictions), "invalid_probability_sum_races": invalid, "predictions": predictions}


@app.get("/api/shadow-status")
def api_shadow_status():
    data = dashboard_status()
    return {key: data[key] for key in ("shadow_days", "production_ready", "latest_monitor")}


@app.get("/api/systemd-status")
def api_systemd_status():
    return systemd_status()


@app.get("/api/performance/summary")
def api_performance_summary(date: str | None = None, track: str | None = None,
                            model: str | None = None, outcome: str = "all", country: str = "ALL"):
    _country(country)
    filters = _performance_filters(date, track, model, outcome)
    return _performance_cached("summary", filters, query_performance_summary)


@app.get("/api/performance/models")
def api_performance_models(date: str | None = None, track: str | None = None,
                           outcome: str = "all", country: str = "ALL"):
    _country(country)
    filters = _performance_filters(date, track, None, outcome)
    return {"models": _performance_cached("models", filters, query_performance_models)}


@app.get("/api/performance/history")
def api_performance_history(page: int = 1, date: str | None = None, track: str | None = None,
                            model: str | None = None, outcome: str = "all", country: str = "ALL"):
    _country(country)
    if page < 1:
        raise HTTPException(status_code=400, detail="page must be >= 1")
    filters = _performance_filters(date, track, model, outcome)
    return _performance_cached("history", filters, query_performance_history, page)


@app.get("/api/performance/chart")
def api_performance_chart(date: str | None = None, track: str | None = None,
                          model: str | None = None, outcome: str = "all", country: str = "ALL"):
    _country(country)
    filters = _performance_filters(date, track, model, outcome)
    return _performance_cached("chart", filters, query_performance_chart)


@app.get("/api/performance/races")
def api_performance_races():
    filters = _performance_filters()
    return _performance_cached("races", filters, lambda connection, _: query_performance_races(connection))


def _country(value: str | None) -> str:
    try:
        return normalize_country(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/race-day/summary")
def api_race_day_summary(date: str | None = None, country: str = "ALL"):
    target = _race_day_date(date)
    scope = _country(country)
    return _race_day_cached("summary", target, None, scope,
                            lambda connection: race_day_summary(connection, target, scope))


@app.get("/api/race-day/tracks")
def api_race_day_tracks(date: str | None = None, country: str = "ALL"):
    target = _race_day_date(date)
    scope = _country(country)
    view = _race_day_cached("view", target, None, scope, lambda connection: race_day_view(connection, target, scope))
    return {"date": target, "country": scope, "count": len(view["tracks"]), "tracks": view["tracks"], "warnings": view["warnings"]}


@app.get("/api/race-day/races")
def api_race_day_races(date: str | None = None, track: str | None = None, country: str = "ALL"):
    target = _race_day_date(date)
    scope = _country(country)
    view = _race_day_cached("view", target, None, scope, lambda connection: race_day_view(connection, target, scope))
    rows = [row for row in view["races"] if not track or track_key(row["track"]) == track_key(track)]
    return {"date": target, "country": scope, "track": track, "count": len(rows), "races": rows}


@app.get("/api/race-day/performance")
def api_race_day_performance(date: str | None = None, track: str | None = None, country: str = "ALL"):
    target = _race_day_date(date)
    scope = _country(country)
    return _race_day_cached(
        "performance", target, track, scope,
        lambda connection: race_day_performance(connection, target, track, scope),
    )


@app.get("/api/race-day/missing-horses")
def api_race_day_missing_horses(date: str | None = None, track: str | None = None):
    target = _race_day_date(date)
    with readonly_connection() as connection:
        rows = missing_horses(connection, target, track)
    return {"date": target, "track": track, "count": len(rows), "rows": rows}


@app.get("/api/race-day/missing-horses/export.csv")
def api_race_day_missing_horses_export(date: str | None = None, track: str | None = None):
    target = _race_day_date(date)
    with readonly_connection() as connection:
        rows = missing_horses(connection, target, track)
    fields = ["date","track","race_no","race_id","race_start_at","missing_reason",
              "horse_id","horse_name","draw","jockey","trainer","tjk_id","missing_fields"]
    buffer = io.StringIO(); writer = csv.DictWriter(buffer, fieldnames=fields); writer.writeheader()
    for row in rows:
        item = dict(row); item["missing_fields"] = ",".join(item["missing_fields"]); writer.writerow(item)
    return StreamingResponse(iter(["\ufeff" + buffer.getvalue()]), media_type="text/csv; charset=utf-8",
                             headers={"Content-Disposition": f'attachment; filename="missing_horses_{target}.csv"'})


@app.get("/api/results-refresh/status")
def api_results_refresh_status(date: str | None = None, country: str = "ALL"):
    target = _race_day_date(date); scope = _country(country)
    metadata = read_status_file(LOG_DIR / "live_results_status.json")
    if not metadata:
        metadata = {"status": "UNKNOWN", "warnings": ["Live refresh has not run yet"]}
    elif metadata.get("date") not in (None, target) or metadata.get("country") not in (None, scope):
        metadata = {"status": "UNKNOWN", "warnings": ["No live refresh has run for this date"]}
    with readonly_connection() as connection:
        return verified_status(connection, target, scope, metadata)


def _diagnostics_args(date=None, track=None, model=None, race_type=None,
                      distance=None, surface=None, field_size=None):
    return _diagnostics_filters(date, track, model, race_type, distance, surface, field_size)


@app.get("/api/diagnostics/summary")
def api_diagnostics_summary(date: str | None = None, track: str | None = None,
                            model: str | None = None, race_type: str | None = None,
                            distance: str | None = None, surface: str | None = None,
                            field_size: str | None = None):
    filters = _diagnostics_args(date, track, model, race_type, distance, surface, field_size)
    return _performance_cached("diagnostics_summary", filters, query_diagnostics_summary)


@app.get("/api/diagnostics/races")
def api_diagnostics_races(page: int = 1, date: str | None = None, track: str | None = None,
                          model: str | None = None, race_type: str | None = None,
                          distance: str | None = None, surface: str | None = None,
                          field_size: str | None = None):
    if page < 1:
        raise HTTPException(status_code=400, detail="page must be >= 1")
    filters = _diagnostics_args(date, track, model, race_type, distance, surface, field_size)
    return _performance_cached("diagnostics_races", filters, query_diagnostics_races, page)


@app.get("/api/diagnostics/winner-ranks")
def api_diagnostics_winner_ranks(date: str | None = None, track: str | None = None,
                                 model: str | None = None, race_type: str | None = None,
                                 distance: str | None = None, surface: str | None = None,
                                 field_size: str | None = None):
    filters = _diagnostics_args(date, track, model, race_type, distance, surface, field_size)
    return {"rows": _performance_cached("diagnostics_ranks", filters, query_diagnostics_winner_ranks)}


@app.get("/api/diagnostics/groups")
def api_diagnostics_groups(date: str | None = None, track: str | None = None,
                           model: str | None = None, race_type: str | None = None,
                           distance: str | None = None, surface: str | None = None,
                           field_size: str | None = None):
    filters = _diagnostics_args(date, track, model, race_type, distance, surface, field_size)
    return {"rows": _performance_cached("diagnostics_groups", filters, query_diagnostics_groups)}


@app.get("/api/diagnostics/extremes")
def api_diagnostics_extremes(date: str | None = None, track: str | None = None,
                             model: str | None = None, race_type: str | None = None,
                             distance: str | None = None, surface: str | None = None,
                             field_size: str | None = None):
    filters = _diagnostics_args(date, track, model, race_type, distance, surface, field_size)
    return _performance_cached("diagnostics_extremes", filters, query_diagnostics_extremes)


@app.get("/api/diagnostics/filters")
def api_diagnostics_filters(model: str = "Ensemble"):
    filters = _diagnostics_args(model=model)
    return _performance_cached("diagnostics_filters", filters,
                               lambda connection, _: query_diagnostics_filters(connection, model))


@app.get("/api/diagnostics/feature-contribution")
def api_diagnostics_feature_contribution():
    return feature_contribution_status()


@app.get("/api/diagnostics/race/{race_id}")
def api_diagnostics_race_detail(race_id: str, model: str = "Ensemble"):
    try:
        filters = normalize_diagnostics_filters(model=model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    detail = _performance_cached(
        "diagnostics_race_detail", {"race_id": race_id, "model": filters["model"]},
        lambda connection, _: query_diagnostics_race_detail(connection, race_id, str(filters["model"])),
    )
    if detail is None:
        raise HTTPException(status_code=404, detail="Evaluated Turkish race was not found")
    return detail


@app.get("/api/diagnostics/export.csv")
def api_diagnostics_export(date: str | None = None, track: str | None = None,
                           model: str | None = None, race_type: str | None = None,
                           distance: str | None = None, surface: str | None = None,
                           field_size: str | None = None):
    filters = _diagnostics_args(date, track, model, race_type, distance, surface, field_size)

    def stream():
        with readonly_connection() as connection:
            rows = iter(query_diagnostics_export(connection, filters))
            first = next(rows, None)
            if first is None:
                yield "race_id\n"
                return
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=list(first))
            writer.writeheader(); writer.writerow(first)
            yield "\ufeff" + buffer.getvalue()
            for row in rows:
                buffer.seek(0); buffer.truncate(0); writer.writerow(row); yield buffer.getvalue()

    filename = f"model_diagnostics_{filters.get('date') or 'all'}.csv"
    return StreamingResponse(stream(), media_type="text/csv; charset=utf-8",
                             headers={"Content-Disposition": f'attachment; filename="{filename}"'})


def _bet_filters(date=None, track=None, model="Ensemble", outcome="all", stake=20, race_no=None):
    try:
        return normalize_bet_filters(date, track, model, outcome, stake, race_no=race_no)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/bet-simulator/summary")
def api_bet_summary(date: str | None = None, track: str | None = None, model: str = "Ensemble",
                    outcome: str = "all", stake: float = 20, race_no: str | None = None):
    filters=_bet_filters(date,track,model,outcome,stake,race_no)
    return _performance_cached("bet_summary",filters,query_bet_summary)


@app.get("/api/bet-simulator/history")
def api_bet_history(page: int = 1, date: str | None = None, track: str | None = None,
                    model: str = "Ensemble", outcome: str = "all", stake: float = 20, race_no: str | None = None):
    if page<1: raise HTTPException(status_code=400,detail="page must be >= 1")
    filters=_bet_filters(date,track,model,outcome,stake,race_no)
    return _performance_cached("bet_history",filters,query_bet_history,page)


@app.get("/api/bet-simulator/export.csv")
def api_bet_export(date: str | None = None, track: str | None = None, model: str = "Ensemble",
                   outcome: str = "all", stake: float = 20, race_no: str | None = None):
    filters=_bet_filters(date,track,model,outcome,stake,race_no)
    def stream():
        with readonly_connection() as connection:
            rows=iter(query_bet_export(connection,filters)); first=next(rows,None)
            if first is None: yield "race_id\n"; return
            buffer=io.StringIO(); writer=csv.DictWriter(buffer,fieldnames=list(first)); writer.writeheader();writer.writerow(first)
            yield "\ufeff"+buffer.getvalue()
            for row in rows: buffer.seek(0);buffer.truncate(0);writer.writerow(row);yield buffer.getvalue()
    return StreamingResponse(stream(),media_type="text/csv; charset=utf-8",
                             headers={"Content-Disposition":'attachment; filename="bet_simulator.csv"'})


def self_check() -> dict[str, Any]:
    required_templates = ("base.html", "dashboard.html", "races.html", "predictions.html",
                          "performance.html", "diagnostics.html", "diagnostics_race.html", "bet_simulator.html",
                          "reports.html", "logs.html")
    with readonly_connection() as connection:
        db_ok = connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        query_only = connection.execute("PRAGMA query_only").fetchone()[0] == 1
    return {
        "database_readable": db_ok,
        "database_query_only": query_only,
        "templates_present": all((WEB_ROOT / "templates" / name).is_file() for name in required_templates),
        "static_present": all((WEB_ROOT / "static" / name).is_file() for name in ("style.css", "live-results.js")),
        "basic_auth_configured": bool(WEB_USERNAME and WEB_PASSWORD),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        checks = self_check()
        print(json.dumps(checks, indent=2))
        return 0 if all(checks.values()) else 1
    uvicorn.run(app, host=WEB_HOST, port=WEB_PORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
