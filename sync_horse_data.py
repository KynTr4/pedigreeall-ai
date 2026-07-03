"""Copy historical horse data from the local pedigreeall_progress.db to the configured DB.

Usage:
    python sync_horse_data.py [--dry-run] [--source PATH]

By default, source is ./pedigreeall_progress.db and destination is DB_PATH from app_config.
Tables synced: discovered_horses, horse_profiles, horse_races, horse_links, horse_mapping
All inserts use INSERT OR IGNORE to preserve existing data.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from app_config import DB_PATH

TABLES = [
    "discovered_horses",
    "horse_profiles",
    "horse_races",
    "horse_links",
    "horse_mapping",
]


def sync_tables(
    source: Path, dest: Path, tables: list[str], dry_run: bool = False
) -> dict[str, int]:
    """Attach source DB and INSERT OR IGNORE each table into dest.

    Returns a dict of {table: rows_copied}.
    """
    if not source.exists():
        raise FileNotFoundError(f"Source DB not found: {source}")

    conn = sqlite3.connect(str(dest), timeout=60)
    try:
        # Use ATTACH so SQLite does the bulk copy without Python round-trips.
        # Use the plain path string (no URI) for cross-platform compatibility.
        conn.execute("ATTACH DATABASE ? AS src", (str(source),))

        counts: dict[str, int] = {}
        for table in tables:
            # Check table exists in source
            src_exists = conn.execute(
                "SELECT 1 FROM src.sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            if not src_exists:
                counts[table] = 0
                print(f"  {table}: not found in source, skipping")
                continue

            src_count = conn.execute(f"SELECT COUNT(*) FROM src.{table}").fetchone()[0]
            if src_count == 0:
                counts[table] = 0
                print(f"  {table}: 0 rows in source, skipping")
                continue

            before = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

            if not dry_run:
                conn.execute(f"INSERT OR IGNORE INTO {table} SELECT * FROM src.{table}")
                conn.commit()

            after = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            copied = after - before
            counts[table] = copied
            status = "(dry-run)" if dry_run else ""
            print(
                f"  {table}: {src_count} source rows → {copied} new rows inserted {status}"
            )

        conn.execute("DETACH DATABASE src")
    finally:
        conn.close()

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=str(Path(__file__).resolve().parent / "pedigreeall_progress.db"),
        help="Path to the source (local) SQLite database",
    )
    parser.add_argument(
        "--dest",
        default=str(DB_PATH),
        help="Path to the destination (production) SQLite database",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without writing anything",
    )
    args = parser.parse_args()

    source = Path(args.source)
    dest = Path(args.dest)

    print(f"Source : {source}")
    print(f"Dest   : {dest}")
    if args.dry_run:
        print("DRY-RUN mode – no changes will be written\n")
    else:
        print()

    try:
        counts = sync_tables(source, dest, TABLES, dry_run=args.dry_run)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    total = sum(counts.values())
    print(f"\nDone. {total} total rows {'would be' if args.dry_run else ''} synced.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
