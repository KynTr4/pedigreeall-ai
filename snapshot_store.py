"""Immutable snapshot parsing and insertion helpers."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from app_config import DB_PATH
from migrate_provenance_schema import apply_migrations

ISTANBUL = ZoneInfo("Europe/Istanbul")
UTC = timezone.utc


def as_float(value):
    if value is None:
        return None
    val_str = str(value).strip()
    if val_str in {"", "-"}:
        return None
    val_str = val_str.replace("%", "")
    match = re.match(r"^(\d+(?:[.,]\d+)?)", val_str)
    if match:
        try:
            return float(match.group(1).replace(",", "."))
        except ValueError:
            return None
    return None


def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    return "".join(ch for ch in text if not unicodedata.combining(ch)).upper().strip()


def horse_entity(entry: dict) -> str:
    info = entry.get("HORSE_INFO") or {}
    horse_id = info.get("HORSE_ID")
    tjk_id = entry.get("TJK_ID")
    if horse_id not in (None, "", 0, "0"):
        return f"horse:{int(horse_id)}"
    if tjk_id not in (None, "", 0, "0"):
        return f"tjk:{tjk_id}"
    digest = hashlib.sha256(
        normalize_name(entry.get("HORSE_NAME")).encode()
    ).hexdigest()[:20]
    return f"name:{digest}"


def parse_program_date(value: str) -> datetime:
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%Y%m%d", "%d%m%Y"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported program date: {value}")


def parse_race_start(program_date: str, tab: dict) -> str | None:
    label = " ".join(str(tab.get(key) or "") for key in ("RACE_NO", "RACE_TAB_NAME"))
    match = re.search(r"(?<!\d)([01]?\d|2[0-3])[.:](\d{2})(?!\d)", label)
    if not match:
        return None
    base = parse_program_date(program_date)
    local = base.replace(
        hour=int(match.group(1)), minute=int(match.group(2)), tzinfo=ISTANBUL
    )
    return local.astimezone(UTC).isoformat()


def parse_race_no(tab: dict) -> int | None:
    match = re.search(
        r"(\d+)\s*[.]?\s*Ko",
        str(tab.get("RACE_NO") or tab.get("RACE_TAB_NAME") or ""),
        re.I,
    )
    return int(match.group(1)) if match else None


def parse_title(tab: dict) -> tuple[str | None, str | None, float | None]:
    raw = re.sub(r"\s+", " ", str(tab.get("TITLE") or "")).strip()
    distance_matches = re.findall(r"(?<!\d)(\d{3,4})(?!\d)", raw)
    distance = float(distance_matches[-1]) if distance_matches else None
    folded = normalize_name(raw)
    if "SENTETIK" in folded:
        surface = "S:"
    elif "CIM" in folded:
        surface = "Ç:"
    elif "KUM" in folded:
        surface = "K:"
    else:
        surface = None
    return raw or None, surface, distance


def parse_agf(value: object) -> tuple[float | None, int | None]:
    text = str(value or "")
    percent = re.search(r"%\s*([\d.,]+)", text)
    rank = re.search(r"[(]\s*(\d+)\s*[)]", text)
    return as_float(percent.group(1)) if percent else None, int(
        rank.group(1)
    ) if rank else None


def insert_program_capture(
    db_path: str | Path,
    payload: list,
    program_date: str,
    captured_at: str,
    source_request_id: str,
    source_endpoint: str = "GET:Tjk/GetRaceProgram",
) -> dict[str, int]:
    """Append one network capture to program/AGF/odds snapshot tables."""
    apply_migrations(db_path)
    captured = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
    if captured.tzinfo is None:
        raise ValueError("captured_at must be timezone-aware")
    counts = {"program": 0, "agf": 0, "odds": 0, "missing_start": 0}
    connection = sqlite3.connect(str(db_path), timeout=60)
    try:
        for city in payload or []:
            city_id = city.get("CITY_ID")
            track = city.get("CITY_NAME")
            for tab in city.get("SUB_TAB", []) or []:
                race_start_at = parse_race_start(program_date, tab)
                if not race_start_at:
                    counts["missing_start"] += len(tab.get("PROGRAM_LIST", []) or [])
                    continue
                race_id = f"prog_{parse_program_date(program_date):%Y-%m-%d}_{city_id}_{tab.get('RACE_TAB_ID')}"
                race_no = parse_race_no(tab)
                race_class, surface, distance = parse_title(tab)
                for entry in tab.get("PROGRAM_LIST", []) or []:
                    horse_id = horse_entity(entry)
                    horse_name = (
                        str(entry.get("HORSE_NAME"))
                        .split("\n")[0]
                        .split("\r")[0]
                        .strip()
                        if entry.get("HORSE_NAME")
                        else None
                    )
                    cursor = connection.execute(
                        """INSERT OR IGNORE INTO program_snapshots(
                               race_id,horse_id,race_start_at,race_no,captured_at,
                               source_endpoint,source_request_id,draw,carried_weight,
                               jockey,trainer,handicap_rating,race_class,track,
                               surface,distance,horse_name)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            race_id,
                            horse_id,
                            race_start_at,
                            race_no,
                            captured_at,
                            source_endpoint,
                            source_request_id,
                            as_float(entry.get("START")),
                            as_float(entry.get("WEIGHT")),
                            entry.get("JOCKEY"),
                            entry.get("COACH"),
                            as_float(entry.get("HANDICAP")),
                            race_class,
                            track,
                            surface,
                            distance,
                            horse_name,
                        ),
                    )
                    counts["program"] += max(cursor.rowcount, 0)
                    agf_percent, agf_rank = parse_agf(entry.get("AGF"))
                    if agf_percent is not None or agf_rank is not None:
                        cursor = connection.execute(
                            """INSERT OR IGNORE INTO agf_snapshots(
                                   race_id,horse_id,captured_at,agf_percent,agf_rank,
                                   source_request_id,source_endpoint)
                               VALUES(?,?,?,?,?,?,?)""",
                            (
                                race_id,
                                horse_id,
                                captured_at,
                                agf_percent,
                                agf_rank,
                                source_request_id,
                                source_endpoint,
                            ),
                        )
                        counts["agf"] += max(cursor.rowcount, 0)
                    odds = as_float(entry.get("GNY"))
                    if odds is not None:
                        cursor = connection.execute(
                            """INSERT OR IGNORE INTO odds_snapshots(
                                   race_id,horse_id,captured_at,odds,source_request_id,source_endpoint)
                               VALUES(?,?,?,?,?,?)""",
                            (
                                race_id,
                                horse_id,
                                captured_at,
                                odds,
                                source_request_id,
                                source_endpoint,
                            ),
                        )
                        counts["odds"] += max(cursor.rowcount, 0)
        connection.commit()
    finally:
        connection.close()
    return counts


def backfill_program_captures_from_raw(db_path: str | Path) -> dict[str, int]:
    """Backfill only captures with genuine raw fetched_at evidence; never invent time."""
    apply_migrations(db_path)
    connection = sqlite3.connect(str(db_path), timeout=60)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """SELECT request_key,request_params_json,fetched_at,response_json
           FROM raw_api_responses WHERE endpoint_key='GET:Tjk/GetRaceProgram'
           ORDER BY fetched_at"""
    ).fetchall()
    connection.close()
    total = {"captures": 0, "program": 0, "agf": 0, "odds": 0, "missing_start": 0}
    for row in rows:
        params = json.loads(row["request_params_json"] or "{}")
        date_value = params.get("p_sDate")
        body = json.loads(row["response_json"])
        payload = body.get("m_cData", body) if isinstance(body, dict) else body
        if not date_value or not isinstance(payload, list):
            continue
        counts = insert_program_capture(
            db_path, payload, date_value, row["fetched_at"], row["request_key"]
        )
        total["captures"] += 1
        for key, value in counts.items():
            total[key] += value
    return total


def append_normalized_result(
    db_path: str | Path,
    horse_id: str,
    race_date_dot: str,
    captured_at: str,
    source_request_id: str,
    source_endpoint: str = "GET:Tjk/Get",
) -> int:
    """Copy a newly normalized post-race value into the isolated result store.

    This is the only compatibility boundary allowed to read horse_races. The
    certified feature builder never imports or queries that legacy table.
    """
    apply_migrations(db_path)
    target_date = parse_program_date(race_date_dot).date().isoformat()
    connection = sqlite3.connect(str(db_path), timeout=60)
    connection.row_factory = sqlite3.Row
    try:
        programs = connection.execute(
            """SELECT race_id,race_start_at,race_no,track FROM program_snapshots
               WHERE horse_id=? AND substr(race_start_at,1,10)=?
               GROUP BY race_id,race_start_at,race_no,track""",
            (horse_id, target_date),
        ).fetchall()

        # Exclude composite/unsupported tracks (e.g., Karma) that duplicate a horse
        # across its real race AND a synthetic composite race on the same day.
        if len(programs) > 1:
            from results_coverage import track_policy

            programs = [
                p for p in programs if track_policy(p["track"]) != "unsupported"
            ]

        # Fallback: if the program snapshot was stored as name:DIGEST (no TJK_ID at
        # capture time), look it up by name via horse_profiles.
        if not programs and not horse_id.startswith("name:"):
            hp_row = connection.execute(
                "SELECT name FROM horse_profiles WHERE horse_key=? LIMIT 1",
                (horse_id,),
            ).fetchone()
            if hp_row and hp_row["name"]:
                name_key = (
                    "name:"
                    + hashlib.sha256(
                        normalize_name(hp_row["name"]).encode()
                    ).hexdigest()[:20]
                )
                programs = connection.execute(
                    """SELECT race_id,race_start_at,race_no FROM program_snapshots
                       WHERE horse_id=? AND substr(race_start_at,1,10)=?
                       GROUP BY race_id,race_start_at,race_no""",
                    (name_key, target_date),
                ).fetchall()

        results = connection.execute(
            """SELECT finish,race_time,prize,odds FROM horse_races
               WHERE horse_key=? AND race_date=?""",
            (horse_id, race_date_dot),
        ).fetchall()
        # Ambiguity is rejected rather than guessed; race time identity can be
        # added when the upstream result endpoint exposes it reliably.
        if len(programs) != 1 or len(results) != 1:
            return 0
        program, result = programs[0], results[0]
        status = "finished" if result["finish"] not in (None, "") else "unknown"
        cursor = connection.execute(
            """INSERT OR IGNORE INTO race_results(
                   race_id,horse_id,race_start_at,race_no,captured_at,
                   source_endpoint,source_request_id,finish_position,finish_time,
                   prize,margin,result_odds,result_status)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                program["race_id"],
                horse_id,
                program["race_start_at"],
                program["race_no"],
                captured_at,
                source_endpoint,
                source_request_id,
                as_float(result["finish"]),
                result["race_time"],
                as_float(result["prize"]),
                None,
                as_float(result["odds"]),
                status,
            ),
        )
        connection.commit()
        return max(cursor.rowcount, 0)
    finally:
        connection.close()


def diagnose_append_failure(
    db_path: str | Path,
    horse_id: str,
    race_date_dot: str,
) -> str:
    """Return the specific reason why append_normalized_result returned 0.

    Possible reasons:
      no_program_snapshot    – horse not found in program_snapshots for the date
      ambiguous_program      – horse appears in multiple races on the date
      result_not_published   – race not yet in horse_races (provider hasn't published)
      ambiguous_result       – horse has multiple horse_races rows for the date
      already_inserted       – race_results row already exists (success on prior run)
    """
    target_date = parse_program_date(race_date_dot).date().isoformat()
    connection = sqlite3.connect(str(db_path), timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    try:
        from results_coverage import track_policy

        prog_rows = connection.execute(
            """SELECT DISTINCT race_id, track FROM program_snapshots
               WHERE horse_id=? AND substr(race_start_at,1,10)=?""",
            (horse_id, target_date),
        ).fetchall()
        # Mirror the same unsupported-track filter applied in append_normalized_result
        supported_progs = [
            r for r in prog_rows if track_policy(r["track"]) != "unsupported"
        ]
        prog_count = len(supported_progs)
        result_count = connection.execute(
            """SELECT COUNT(*) FROM horse_races
               WHERE horse_key=? AND race_date=?""",
            (horse_id, race_date_dot),
        ).fetchone()[0]
        already = connection.execute(
            """SELECT 1 FROM race_results
               WHERE horse_id=? AND date(race_start_at,'+3 hours')=?
               LIMIT 1""",
            (horse_id, target_date),
        ).fetchone()
        if already:
            return "already_inserted"
        if prog_count == 0:
            return "no_program_snapshot"
        if prog_count > 1:
            return "ambiguous_program"
        if result_count == 0:
            return "result_not_published"
        if result_count > 1:
            return "ambiguous_result"
        # prog_count==1 and result_count==1 and not already inserted
        # → data is ready; append should succeed on next call.
        return "ready_to_insert"
    finally:
        connection.close()


if __name__ == "__main__":
    print({"migrations_applied": apply_migrations(DB_PATH), "database": str(DB_PATH)})
