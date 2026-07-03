"""Download and normalize today's finished race results from Pedigreeall."""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import date, datetime, timedelta

import pandas as pd

from app_config import DB_PATH, LOG_DIR, OUTPUT_DIR, REPORTS_DIR, ensure_runtime_dirs
from normalize_data import normalize_entity
from pedigreeall_core import APIClient, connect, init_db, now, resolve_tjk_id
from race_scope import normalize_country, track_in_country
from results_coverage import (
    clean_track,
    coverage_warnings,
    track_policy,
    write_results_coverage,
)
from snapshot_store import append_normalized_result, diagnose_append_failure

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
logger = logging.getLogger("update_results")

FAILED_UPDATES_CSV = "failed_updates.csv"
PUBLIC_ONLY_MODE = True


def filter_program_horses(horses, country="ALL", tracks=None, completed_tracks=None):
    """Select only in-scope, incomplete program tracks before any detail request."""
    selected_tracks = {clean_track(track) for track in (tracks or [])}
    completed = {clean_track(track) for track in (completed_tracks or [])}
    return [
        horse
        for horse in horses
        if (
            track_in_country(clean_track(horse.get("city_name")), country)
            and (
                not selected_tracks
                or clean_track(horse.get("city_name")) in selected_tracks
            )
            and clean_track(horse.get("city_name")) not in completed
        )
    ]


def log_failure(entity, error_type, message):
    """Write failure record to failed_updates.csv."""
    row = pd.DataFrame(
        [
            {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "script": "update_results.py",
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


async def fetch_endpoint(c, key, path, entity, params, force=False):
    with connect(c.db_path) as db:
        restricted = db.execute(
            "SELECT 1 FROM access_restrictions WHERE endpoint_key=?", (key,)
        ).fetchone()
    if PUBLIC_ONLY_MODE and restricted:
        return None
    try:
        return await c.request(
            key, path, params=params, entity_key=entity, store_raw=True
        )
    except Exception as e:
        logger.warning(f"Failed to fetch {key} for {entity}: {e}")
        log_failure(entity, type(e).__name__, f"Endpoint {key} failed: {e}")
        return None


def publish_coverage(db_path, target_date: date):
    coverage = write_results_coverage(
        db_path, target_date.isoformat(), OUTPUT_DIR, REPORTS_DIR
    )
    warnings = coverage_warnings(coverage)
    logger.info(
        "RESULTS_COVERAGE %s", json.dumps(coverage["tracks"], ensure_ascii=False)
    )
    for warning in warnings:
        logger.warning("RESULTS_COVERAGE_WARNING %s", warning)
    return coverage


def isolated_result_exists(connection, entity: str, target_date: str) -> bool:
    return (
        connection.execute(
            """SELECT 1 FROM race_results
           WHERE horse_id=? AND date(race_start_at,'+3 hours')=?
             AND result_status='finished' AND finish_position IS NOT NULL
           LIMIT 1""",
            (entity, target_date),
        ).fetchone()
        is not None
    )


async def update_date(
    target_date: date,
    country: str = "ALL",
    tracks: list[str] | None = None,
    skip_completed_tracks: bool = False,
):
    logger.info("Starting update_results.py for %s...", target_date.isoformat())
    db_path = str(DB_PATH)
    init_db(db_path)

    today_str = target_date.isoformat()
    # Format dates in TJK / Pedigreeall (stored as DD.MM.YYYY in race history)
    today_dot = target_date.strftime("%d.%m.%Y")

    # 1. Query today's race program horses
    with connect(db_path) as db:
        horses = [
            dict(r)
            for r in db.execute(
                """SELECT DISTINCT tjk_id,horse_id,horse_name,city_name
               FROM race_program_entries WHERE program_date=?""",
                (today_str,),
            ).fetchall()
        ]

    if not horses:
        logger.info(
            "No horses found in %s race program. Results cannot be updated.", today_str
        )
        return publish_coverage(db_path, target_date)

    completed_tracks: set[str] = set()
    if skip_completed_tracks:
        coverage = publish_coverage(db_path, target_date)
        completed_tracks = {
            clean_track(row["track"])
            for row in coverage["tracks"]
            if row["program_races"] and row["program_races"] == row["result_races"]
        }
    scoped_horses = filter_program_horses(horses, country, tracks, completed_tracks)
    for track in sorted(completed_tracks):
        logger.info("Skipping completed track without detail requests: track=%s", track)
    supported_horses = []
    for horse in scoped_horses:
        policy = track_policy(horse.get("city_name"))
        if policy == "unsupported":
            logger.warning(
                "Skipping unsupported result source: track=%s horse=%s tjk_id=%s",
                clean_track(horse.get("city_name")),
                horse.get("horse_name"),
                horse.get("tjk_id"),
            )
            continue
        supported_horses.append(horse)
    horses = supported_horses

    logger.info(
        f"Found {len(horses)} horses in today's race program. Checking if results are already in DB..."
    )

    # 2. The completion source of truth is the isolated race_results table.
    # A legacy horse_races row must not suppress a missing immutable result row.
    missing_results = []
    with connect(db_path) as db:
        for h in horses:
            horse_id = h["horse_id"]
            name = h["horse_name"]
            track = clean_track(h.get("city_name"))

            # Use unified resolver
            res = resolve_tjk_id(
                db, horse_id if horse_id else f"tjk:{h['tjk_id']}", name, today_str
            )
            tjk_id = res["tjk_id"]

            # Form entity key
            if horse_id:
                entity = f"horse:{horse_id}"
            else:
                entity = f"tjk:{tjk_id}" if tjk_id else f"tjk:{h['tjk_id']}"

            if isolated_result_exists(db, entity, today_str):
                continue

            # Log resolver result for each missing record
            logger.info(
                "RESOLVER_RESULT horse_name=%s horse_id=%s resolved_tjk_id=%s source_table=%s reason=%s",
                name,
                horse_id,
                tjk_id,
                res["source_table"],
                res["reason"],
            )

            missing_results.append((entity, tjk_id, horse_id, name, track))

    logger.info(
        f"{len(horses) - len(missing_results)} results already exist in DB. {len(missing_results)} results are missing and will be requested."
    )

    if not missing_results:
        logger.info("All supported race results are up to date.")
        return publish_coverage(db_path, target_date)

    # 3. Request race results from Pedigreeall API (force=True to bypass cached responses)
    c = APIClient(db_path, rps=0.75, concurrency=2)
    success_count = 0
    technical_failures = []

    async with c.open():
        for entity, tjk_id, horse_id, name, track in missing_results:
            if not tjk_id:
                logger.warning(
                    "TJK_ID_MISSING track=%s horse=%s entity=%s",
                    track,
                    name,
                    entity,
                )
                continue

            logger.info(f"Fetching latest results for {name} ({entity})")
            # We call fetch_endpoint with force=True (which is not directly in c.request but we bypass caching by deleting or using custom check)
            # Wait, in pedigreeall_core.py, c.request does NOT check raw_api_responses for cache!
            # The cache checking is done in scrape_pedigreeall.py!
            # So calling c.request directly here will AUTOMATICALLY fetch from the network and overwrite the cache!
            # This is exactly what we want!
            response = await fetch_endpoint(
                c, "GET:Tjk/Get", "Tjk/Get", entity, {"p_iTjkId": tjk_id}
            )
            if response is None:
                continue

            try:
                # Normalize new data
                normalize_entity(db_path, entity, tjk_id, horse_id)
                source_request_id = c.last_source_request_id
                if not source_request_id:
                    raise RuntimeError(
                        "Result response has no immutable source request id"
                    )
                with connect(db_path) as db:
                    capture = db.execute(
                        "SELECT fetched_at FROM raw_api_responses WHERE request_key=?",
                        (source_request_id,),
                    ).fetchone()
                inserted = append_normalized_result(
                    db_path, entity, today_dot, capture[0], source_request_id
                )
                logger.info("Appended %s isolated post-race result row(s).", inserted)
                if inserted:
                    success_count += 1
                    logger.info(f"Successfully normalized results for {name}.")
                else:
                    reason = diagnose_append_failure(db_path, entity, today_dot)
                    logger.warning(
                        "RESULT_NOT_APPENDED track=%s horse=%s entity=%s reason=%s",
                        track,
                        name,
                        entity,
                        reason,
                    )
            except Exception as e:
                logger.error(f"Failed to normalize results for {name}: {e}")
                log_failure(entity, "NormalizationError", str(e))
                technical_failures.append(f"{entity}: {type(e).__name__}: {e}")

    logger.info(
        f"Completed update_results.py. Successfully updated results for {success_count}/{len(missing_results)} horses."
    )
    if technical_failures:
        raise RuntimeError(
            f"Result normalization/write failed for {len(technical_failures)} horse(s): "
            + "; ".join(technical_failures[:10])
        )
    return publish_coverage(db_path, target_date)


async def main(
    target_date: date | None = None,
    lookback_days: int = 1,
    country: str = "ALL",
    tracks: list[str] | None = None,
    today_tracks: bool = False,
):
    country = normalize_country(country)
    dates = (
        [target_date]
        if target_date
        else [
            date.today() - timedelta(days=offset)
            for offset in range(max(0, lookback_days), -1, -1)
        ]
    )
    coverages = []
    for selected_date in dates:
        coverages.append(
            await update_date(selected_date, country, tracks, today_tracks)
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "results_coverage_run.json").write_text(
        json.dumps({"dates": coverages}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return coverages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="Single result date in YYYY-MM-DD format")
    parser.add_argument("--lookback-days", type=int, default=1)
    parser.add_argument("--country", default="ALL")
    parser.add_argument("--track", action="append", default=[])
    parser.add_argument("--today-tracks", action="store_true")
    args = parser.parse_args()
    selected = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else None
    if args.today_tracks and selected is None:
        selected = date.today()
    asyncio.run(
        main(selected, args.lookback_days, args.country, args.track, args.today_tracks)
    )
