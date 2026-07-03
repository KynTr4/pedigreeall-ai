"""Download and parse today's race program, updating database tables."""

import asyncio
import json
import logging
import os
import sys
from datetime import date, datetime

import pandas as pd

from app_config import DB_PATH, LOG_DIR, ensure_runtime_dirs
from discover_horses import f, i, record, text, upsert
from pedigreeall_core import APIClient, connect, init_db, unwrap
from snapshot_store import insert_program_capture

# Setup logging
ensure_runtime_dirs()
log_date = datetime.now().strftime("%Y_%m_%d")
log_file = LOG_DIR / f"update_{log_date}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("update_race_programs")

FAILED_UPDATES_CSV = "failed_updates.csv"


def log_failure(entity, error_type, message):
    """Write failure record to failed_updates.csv."""
    row = pd.DataFrame(
        [
            {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "script": "update_race_programs.py",
                "entity": str(entity),
                "error_type": str(error_type),
                "error_message": str(message),
            }
        ]
    )
    file_exists = os.path.exists(FAILED_UPDATES_CSV)
    row.to_csv(
        FAILED_UPDATES_CSV,
        mode="a",
        index=False,
        header=not file_exists,
        encoding="utf-8",
    )


async def public_request(c, key, path, entity, params):
    try:
        return await c.request(key, path, params=params, entity_key=entity)
    except Exception as e:
        logger.error(f"Request failed for {path} ({entity}): {e}")
        log_failure(entity, type(e).__name__, str(e))
        return None


def resolve_name_ids_in_snapshots(db, program_date: str) -> int:
    """Replace name:DIGEST horse_ids in program_snapshots with proper tjk:/horse: IDs.

    When the race program API entry has no TJK_ID or HORSE_ID, insert_program_capture
    stores the horse as name:SHA256DIGEST.  That format never matches the tjk:/horse:
    entity keys used by update_results.py.  This function resolves the mismatch by
    looking up the horse name in race_program_entries (which stores TJK_ID separately)
    and updating the snapshot row in-place.
    """
    name_rows = db.execute(
        """SELECT ps.snapshot_id, ps.source_request_id, ps.race_id, ps.horse_name
           FROM program_snapshots ps
           WHERE ps.horse_id LIKE 'name:%'
             AND substr(ps.race_start_at, 1, 10) = ?""",
        (program_date,),
    ).fetchall()
    if not name_rows:
        return 0
    updated = 0
    for row in name_rows:
        horse_name = row["horse_name"] if hasattr(row, "keys") else row[3]
        if not horse_name:
            continue
        rpe = db.execute(
            """SELECT tjk_id, horse_id FROM race_program_entries
               WHERE program_date = ? AND horse_name = ?
               LIMIT 1""",
            (program_date, horse_name),
        ).fetchone()
        if not rpe:
            continue
        tjk_id_val = rpe["tjk_id"] if hasattr(rpe, "keys") else rpe[0]
        horse_id_val = rpe["horse_id"] if hasattr(rpe, "keys") else rpe[1]
        new_id = None
        if horse_id_val and horse_id_val != 0:
            new_id = f"horse:{horse_id_val}"
        elif tjk_id_val:
            new_id = f"tjk:{tjk_id_val}"
        if not new_id:
            continue
        snapshot_id = row["snapshot_id"] if hasattr(row, "keys") else row[0]
        src_req = row["source_request_id"] if hasattr(row, "keys") else row[1]
        race_id = row["race_id"] if hasattr(row, "keys") else row[2]
        # Only update if new_id does not already exist for this (source_request_id, race_id)
        conflict = db.execute(
            """SELECT 1 FROM program_snapshots
               WHERE source_request_id=? AND race_id=? AND horse_id=?""",
            (src_req, race_id, new_id),
        ).fetchone()
        if conflict:
            continue
        db.execute(
            "UPDATE program_snapshots SET horse_id=? WHERE snapshot_id=?",
            (new_id, snapshot_id),
        )
        updated += 1
    return updated


async def main():
    logger.info("Starting update_race_programs.py...")
    db_path = str(DB_PATH)
    init_db(db_path)

    today_str = date.today().isoformat()
    today_dot = date.today().strftime("%d.%m.%Y")
    marker = f"race_program:{today_str}"

    # Load state
    state_path = "update_state.json"
    state = {}
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f_state:
                state = json.load(f_state)
        except Exception as e:
            logger.warning(f"Failed to read state: {e}")

    # Check if already done today
    with connect(db_path) as db:
        done = db.execute(
            "SELECT status FROM progress WHERE work_type='public_discovery' AND entity_key=? AND endpoint_key='race_program'",
            (marker,),
        ).fetchone()

    if done and done[0] == "completed" and state.get("last_run_date") == today_str:
        logger.info(
            "Race program was already collected today; fetching another immutable "
            "snapshot so program/odds history is preserved."
        )

    # Initialize client
    c = APIClient(db_path, rps=0.75, concurrency=2)
    found = 0

    async with c.open():
        await c.checkpoint("public_discovery", marker, "race_program", "running")
        logger.info(f"Fetching race program for date: {today_dot} / {today_str}")

        payload = None
        # Try both ddmmyyyy and yyyymmdd formats
        for date_value in (today_dot, today_str):
            payload = await public_request(
                c,
                "GET:Tjk/GetRaceProgram",
                "Tjk/GetRaceProgram",
                marker,
                {"p_sDate": date_value},
            )
            if payload:
                break

        if payload is None:
            logger.error("Could not retrieve race program from TJK API.")
            await c.checkpoint(
                "public_discovery",
                marker,
                "race_program",
                "failed",
                message="Empty payload",
            )
            log_failure(marker, "APIError", "GetRaceProgram returned empty payload")
            return 1

        payload = unwrap(payload)
        cities = payload if isinstance(payload, list) else []
        if not cities:
            logger.info(
                f"No race program found for {today_str}; marking as no_race_today."
            )
            await c.checkpoint(
                "public_discovery",
                marker,
                "race_program",
                "not_found",
                message="no_race_today",
            )
            state["last_run_date"] = today_str
            state["race_program_status"] = "no_race_today"
            with open(state_path, "w", encoding="utf-8") as f_state:
                json.dump(state, f_state, indent=2, ensure_ascii=False)
            return 0

        # Immutable provenance path. The legacy race_program_entries table below
        # remains for compatibility, while every actual download is appended here.
        source_request_id = c.last_source_request_id
        if not source_request_id:
            raise RuntimeError(
                "Race program response has no immutable source request id"
            )
        with connect(db_path) as db:
            source_row = db.execute(
                "SELECT fetched_at FROM raw_api_responses WHERE request_key=?",
                (source_request_id,),
            ).fetchone()
        if not source_row:
            raise RuntimeError(f"Raw capture not found: {source_request_id}")
        snapshot_counts = insert_program_capture(
            db_path, cities, today_str, source_row[0], source_request_id
        )
        logger.info("Appended immutable snapshots: %s", snapshot_counts)

        with connect(db_path) as db:
            resolved = resolve_name_ids_in_snapshots(db, today_str)
        if resolved:
            logger.info(
                "Resolved %d name:DIGEST horse_ids to proper tjk:/horse: IDs.", resolved
            )

        with connect(db_path) as db:
            db.execute(
                "DELETE FROM race_program_entries WHERE program_date=?", (today_str,)
            )
            for city in cities:
                if not isinstance(city, dict):
                    continue
                city_id = i(city.get("CITY_ID"))
                city_name = city.get("CITY_NAME")
                logger.info(f"Processing city: {city_name} (ID: {city_id})")

                for tab in city.get("SUB_TAB", []) or []:
                    if not isinstance(tab, dict):
                        continue
                    race_tab_id = i(tab.get("RACE_TAB_ID"))
                    race_name = tab.get("RACE_TAB_NAME") or tab.get("TITLE")
                    race_no = tab.get("RACE_NO")

                    for entry in tab.get("PROGRAM_LIST", []) or []:
                        if not isinstance(entry, dict):
                            continue
                        hi = entry.get("HORSE_INFO") or {}
                        x = {
                            **hi,
                            **{k: v for k, v in entry.items() if k != "HORSE_INFO"},
                        }

                        # Save discovered horse
                        r_rec = record("race_program", x, is_turkey=True)
                        upsert(db, r_rec)
                        found += 1

                        # Save race program entry
                        tjk_id = (
                            str(entry.get("TJK_ID")) if entry.get("TJK_ID") else None
                        )
                        horse_id = i(hi.get("HORSE_ID"))
                        horse_name = text(
                            entry.get("HORSE_NAME") or hi.get("HORSE_NAME")
                        )

                        db.execute(
                            "INSERT OR REPLACE INTO race_program_entries VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (
                                today_str,
                                city_id,
                                city_name,
                                race_tab_id,
                                race_name,
                                race_no,
                                tjk_id,
                                horse_id,
                                horse_name,
                                i(entry.get("AGE")),
                                entry.get("WEIGHT"),
                                entry.get("JOCKEY"),
                                entry.get("OWNER"),
                                entry.get("COACH"),
                                entry.get("START"),  # starting gate
                                entry.get("HANDICAP"),
                                entry.get("LAST_6_RACE"),
                                entry.get("KGS"),
                                entry.get("GNY"),
                                entry.get("AGF"),
                                entry.get("DERECE"),
                                json.dumps(hi, ensure_ascii=False),
                            ),
                        )

        await c.checkpoint(
            "public_discovery",
            marker,
            "race_program",
            "completed",
            message=f"horses={found}",
        )
        logger.info(f"Successfully processed race program. Discovered {found} horses.")

        # Update state
        state["last_run_date"] = today_str
        state["race_program_status"] = "completed" if found else "no_race_today"
        with open(state_path, "w", encoding="utf-8") as f_state:
            json.dump(state, f_state, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()) or 0)
    except Exception as exc:
        logger.exception("Unhandled error in update_race_programs.py")
        log_failure("race_program", type(exc).__name__, str(exc))
        sys.exit(1)
