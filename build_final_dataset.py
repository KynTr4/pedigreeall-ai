"""Rebuild the final Benter dataset from the current SQLite horse_races table.

The database is the source of truth.  Legacy feature CSV files are deliberately
not read: they can be stale or incomplete and were the reason 2024/2025 vanished.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "pedigreeall_progress.db"
OUTPUT = ROOT / "output"
REPORTS = ROOT / "reports"
FINAL_CSV = OUTPUT / "final_benter_dataset.csv"
FINAL_PARQUET = OUTPUT / "final_benter_dataset.parquet"

BASE_COLUMNS = [
    "horse_id", "race_id", "race_date", "track", "distance", "surface",
    "race_class", "horse_name", "jockey", "trainer", "carried_weight",
    "draw", "finish_position", "finish_time_seconds", "odds", "agf",
    "handicap_rating", "pre_race_handicap_rating", "prize",
    "race_field_complete", "found_starters", "expected_starters_min",
    "missing_starters_min", "history_order_certified", "days_since_last_race",
    "last_3_avg_position", "last_5_avg_position", "last_10_avg_position",
    "surface_win_rate", "distance_win_rate", "track_win_rate",
    "jockey_horse_win_rate", "trainer_horse_win_rate", "weight_change",
    "class_change", "distance_change", "surface_change",
]
ENRICHMENT_COLUMNS = [
    "track_condition", "turf_condition", "dirt_condition",
    "synthetic_condition", "weather", "temperature", "humidity", "pressure",
    "wind_speed", "wind_direction", "agf_percent", "agf_rank", "margin_text",
    "margin_lengths_numeric", "last_workout_date", "last_workout_distance",
    "last_workout_time", "days_since_last_workout", "workout_count_last_7d",
    "workout_count_last_14d", "race_no", "had_jockey_change",
    "had_trainer_change", "had_equipment_change", "had_veterinary_issue",
    "had_lameness_issue", "had_steward_incident", "had_recent_scratch",
    "incident_count_last_30d", "veterinary_count_last_180d",
    "steward_incident_count_last_180d",
]

OUTPUT.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)
os.makedirs(ROOT / "logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "logs" / f"update_{datetime.now():%Y_%m_%d}.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("build_final_dataset")


def parse_dates(values: pd.Series) -> pd.Series:
    """Canonical required parser; DB values are uniformly DD.MM.YYYY."""
    return pd.to_datetime(values, dayfirst=True, errors="coerce")


def numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values.astype("string").str.replace(",", ".", regex=False), errors="coerce")


def parse_time(values: pd.Series) -> pd.Series:
    parts = values.astype("string").str.extract(r"^(?:(\d+)[.:])?(\d+)[.](\d+)$")
    minutes = pd.to_numeric(parts[0], errors="coerce").fillna(0)
    seconds = pd.to_numeric(parts[1], errors="coerce")
    fraction = pd.to_numeric("0." + parts[2].fillna(""), errors="coerce")
    return minutes * 60 + seconds + fraction


def normalize_surface(values: pd.Series) -> pd.Series:
    text = values.fillna("").astype(str).str.strip()
    return text.str.extract(r"^([^:]+)", expand=False).fillna("").str[:1] + ":"


def clean_horse_name(values: pd.Series) -> pd.Series:
    replacements = str.maketrans("ÇĞİÖŞÜçğıöşü", "CGIOSUcgiosu")
    return (
        values.fillna("").astype(str).str.translate(replacements)
        .str.replace(r"[^A-Za-z ]", "", regex=True).str.upper()
        .str.replace(r"\s+", " ", regex=True).str.strip()
    )


def prior_rate(frame: pd.DataFrame, category: str) -> pd.Series:
    key = frame[category].fillna("__MISSING__").astype(str)
    keys = [frame["horse_id"], key]
    starts = frame.groupby(keys, sort=False, dropna=False).cumcount()
    wins_before = frame.groupby(keys, sort=False, dropna=False)["_is_win"].cumsum() - frame["_is_win"]
    return wins_before.div(starts.where(starts > 0))


def add_historical_pre_race_rating(frame: pd.DataFrame) -> pd.DataFrame:
    """Use only the prior race's post-race HP as the next race's known rating."""
    output = frame.copy()
    same_day_count = output.groupby(
        ["horse_id", "_date"], dropna=False
    )["race_id"].transform("size")
    output["history_order_certified"] = same_day_count.eq(1)
    output["pre_race_handicap_rating"] = output.groupby(
        "horse_id", sort=False, dropna=False
    )["handicap_rating"].shift()
    output.loc[~output["history_order_certified"], "pre_race_handicap_rating"] = np.nan
    return output


def load_base_from_db() -> pd.DataFrame:
    logger.info("Loading horse_races from %s", DB_PATH)
    with sqlite3.connect(DB_PATH) as connection:
        raw = pd.read_sql_query(
            """
            SELECT r.horse_key, r.race_id, r.race_date, r.hippodrome, r.distance,
                   r.surface, r.race_class, r.finish, r.race_time, r.agf, r.odds,
                   r.jockey, r.trainer, r.prize, r.weight, r.gate, r.rating,
                   p.name AS horse_name
            FROM horse_races AS r
            LEFT JOIN horse_profiles AS p ON p.horse_key = r.horse_key
            """,
            connection,
        )

    frame = pd.DataFrame({
        "horse_id": raw["horse_key"].astype("string"),
        "race_id": raw["race_id"].astype("string"),
        "race_date": raw["race_date"].astype("string"),
        "track": raw["hippodrome"],
        "distance": numeric(raw["distance"]),
        "surface": normalize_surface(raw["surface"]),
        "race_class": raw["race_class"],
        "horse_name": raw["horse_name"].fillna("Unknown"),
        "jockey": raw["jockey"],
        "trainer": raw["trainer"],
        "carried_weight": numeric(raw["weight"]),
        "draw": numeric(raw["gate"]),
        "finish_position": numeric(raw["finish"]),
        "finish_time_seconds": parse_time(raw["race_time"]),
        "odds": numeric(raw["odds"]),
        "agf": numeric(raw["agf"]),
        "handicap_rating": numeric(raw["rating"]),
        "prize": numeric(raw["prize"]),
    })
    frame["_date"] = parse_dates(frame["race_date"])
    bad_dates = int(frame["_date"].isna().sum())
    if bad_dates:
        raise ValueError(f"DB contains {bad_dates} unparseable race_date values")
    duplicate_count = int(frame.duplicated(["horse_id", "race_id"]).sum())
    if duplicate_count:
        logger.warning("Dropping %s duplicate horse/race keys", duplicate_count)
        frame = frame.drop_duplicates(["horse_id", "race_id"], keep="last")

    frame = frame.sort_values(["horse_id", "_date", "race_id"], kind="stable").reset_index(drop=True)
    horses = frame.groupby("horse_id", sort=False, dropna=False)
    # GET:Tjk/Get.HP is fetched from the post-race history endpoint and changes
    # with that race's result.  Only the preceding race's HP is admissible for
    # historical training.  Same-day order cannot be certified without a start
    # time, so those rows receive no historical pre-race rating.
    frame = add_historical_pre_race_rating(frame)
    horses = frame.groupby("horse_id", sort=False, dropna=False)
    previous_date = horses["_date"].shift()
    frame["days_since_last_race"] = (frame["_date"] - previous_date).dt.days.astype(float)
    for window in (3, 5, 10):
        frame[f"last_{window}_avg_position"] = horses["finish_position"].transform(
            lambda series, w=window: series.shift().rolling(w, min_periods=1).mean()
        )
    frame["_is_win"] = frame["finish_position"].eq(1).fillna(False).astype(int)
    for category, output in [
        ("surface", "surface_win_rate"), ("distance", "distance_win_rate"),
        ("track", "track_win_rate"), ("jockey", "jockey_horse_win_rate"),
        ("trainer", "trainer_horse_win_rate"),
    ]:
        frame[output] = prior_rate(frame, category)

    previous_weight = horses["carried_weight"].shift()
    previous_class = horses["race_class"].shift()
    previous_distance = horses["distance"].shift()
    previous_surface = horses["surface"].shift()
    frame["weight_change"] = frame["carried_weight"] - previous_weight
    frame["class_change"] = frame["race_class"].ne(previous_class).astype(int)
    frame["distance_change"] = frame["distance"] - previous_distance
    frame["surface_change"] = frame["surface"].ne(previous_surface).astype(int)
    frame = add_race_field_audit(frame)
    return frame


def add_race_field_audit(frame: pd.DataFrame) -> pd.DataFrame:
    """Prove the minimum internally complete field without inventing starters.

    A race is admitted for training only when every observed row has one unique,
    positive integer finish and the set is exactly 1..N.  This cannot prove
    scratched/non-finishing starters that never reached the source, so the
    expected count is explicitly named a minimum rather than a program count.
    """
    finish = pd.to_numeric(frame["finish_position"], errors="coerce")
    valid_integer = finish.notna() & finish.gt(0) & finish.mod(1).eq(0)
    audit = pd.DataFrame({"race_id": frame["race_id"], "finish": finish.where(valid_integer)})
    grouped = audit.groupby("race_id", sort=False, dropna=False)
    stats = grouped.agg(
        found_starters=("race_id", "size"),
        numeric_finishes=("finish", "count"),
        unique_finishes=("finish", "nunique"),
        min_finish=("finish", "min"),
        expected_starters_min=("finish", "max"),
    )
    stats["race_field_complete"] = (
        stats["found_starters"].ge(2)
        & stats["numeric_finishes"].eq(stats["found_starters"])
        & stats["unique_finishes"].eq(stats["found_starters"])
        & stats["min_finish"].eq(1)
        & stats["expected_starters_min"].eq(stats["found_starters"])
    )
    stats["missing_starters_min"] = (
        stats["expected_starters_min"] - stats["unique_finishes"]
    ).clip(lower=0)
    stats["expected_starters_min"] = stats["expected_starters_min"].fillna(stats["found_starters"])
    coverage = stats.reset_index()[[
        "race_id", "found_starters", "expected_starters_min",
        "missing_starters_min", "race_field_complete",
    ]]
    coverage.to_csv(OUTPUT / "race_starter_coverage.csv", index=False, encoding="utf-8")
    return frame.merge(coverage, on="race_id", how="left", validate="many_to_one")


def merge_optional_enrichments(frame: pd.DataFrame) -> pd.DataFrame:
    """Enrich from current raw lookup outputs, never from legacy Benter datasets."""
    unavailable = "not_available_in_tjk_checked"
    for column in ["track_condition", "turf_condition", "dirt_condition", "synthetic_condition",
                   "weather", "temperature", "humidity", "pressure", "wind_speed", "wind_direction"]:
        frame[column] = unavailable
    for column in ["agf_percent", "agf_rank", "margin_text", "last_workout_date",
                   "last_workout_distance", "last_workout_time", "days_since_last_workout",
                   "workout_count_last_7d", "workout_count_last_14d"]:
        frame[column] = "not_found"
    for column in ["margin_lengths_numeric", "race_no"]:
        frame[column] = np.nan
    for column in ["had_jockey_change", "had_trainer_change", "had_equipment_change",
                   "had_veterinary_issue", "had_lameness_issue", "had_steward_incident",
                   "had_recent_scratch", "incident_count_last_30d",
                   "veterinary_count_last_180d", "steward_incident_count_last_180d"]:
        frame[column] = 0

    frame["_date_key"] = frame["_date"].dt.strftime("%Y-%m-%d")
    frame["_name_key"] = clean_horse_name(frame["horse_name"])

    track_path = OUTPUT / "track_conditions.csv"
    if track_path.exists():
        lookup = pd.read_csv(track_path, low_memory=False)
        lookup["_date_key"] = parse_dates(lookup["race_date"]).dt.strftime("%Y-%m-%d")
        lookup = lookup.drop_duplicates(["_date_key", "track"], keep="last")
        columns = [c for c in ["track_condition", "turf_condition", "dirt_condition",
                   "synthetic_condition", "weather", "temperature", "humidity", "pressure",
                   "wind_speed", "wind_direction"] if c in lookup]
        incoming = lookup[["_date_key", "track", *columns]].rename(columns={c: f"_new_{c}" for c in columns})
        frame = frame.merge(incoming, on=["_date_key", "track"], how="left", sort=False)
        for column in columns:
            frame[column] = frame.pop(f"_new_{column}").combine_first(frame[column])

    agf_path = OUTPUT / "agf_data.csv"
    if agf_path.exists():
        lookup = pd.read_csv(agf_path, low_memory=False)
        lookup["_date_key"] = parse_dates(lookup["race_date"]).dt.strftime("%Y-%m-%d")
        lookup["_name_key"] = clean_horse_name(lookup["horse_name"])
        lookup = lookup.drop_duplicates(["_date_key", "track", "_name_key"], keep="last")
        columns = [c for c in ["agf_percent", "agf_rank"] if c in lookup]
        incoming = lookup[["_date_key", "track", "_name_key", *columns]].rename(columns={c: f"_new_{c}" for c in columns})
        frame = frame.merge(incoming, on=["_date_key", "track", "_name_key"], how="left", sort=False)
        for column in columns:
            frame[column] = frame.pop(f"_new_{column}").combine_first(frame[column])

    workout_path = OUTPUT / "workouts.csv"
    if workout_path.exists():
        lookup = pd.read_csv(workout_path, low_memory=False)
        lookup["_date_key"] = parse_dates(lookup["race_date"]).dt.strftime("%Y-%m-%d")
        lookup["_name_key"] = clean_horse_name(lookup["horse_name"])
        lookup = lookup.drop_duplicates(["_date_key", "_name_key"], keep="last")
        columns = [c for c in ["last_workout_date", "last_workout_distance", "last_workout_time",
                   "days_since_last_workout", "workout_count_last_7d", "workout_count_last_14d"] if c in lookup]
        incoming = lookup[["_date_key", "_name_key", *columns]].rename(columns={c: f"_new_{c}" for c in columns})
        frame = frame.merge(incoming, on=["_date_key", "_name_key"], how="left", sort=False)
        for column in columns:
            frame[column] = frame.pop(f"_new_{column}").combine_first(frame[column])
    return frame


def save_final_dataset(frame: pd.DataFrame) -> tuple[int, int]:
    final = frame[BASE_COLUMNS + ENRICHMENT_COLUMNS].copy()
    final = final.drop_duplicates(["horse_id", "race_id"], keep="last")
    final = final.sort_values(["race_date", "race_id", "horse_id"], kind="stable").reset_index(drop=True)
    final.to_csv(FINAL_CSV, index=False, encoding="utf-8")
    parquet_frame = final.copy()
    for column in parquet_frame.select_dtypes(include=["object", "string"]).columns:
        parquet_frame[column] = parquet_frame[column].astype("string")
    pq.write_table(pa.Table.from_pandas(parquet_frame, preserve_index=False), FINAL_PARQUET, compression="zstd")
    csv_check = pd.read_csv(FINAL_CSV, low_memory=False)
    parquet_check = pd.read_parquet(FINAL_PARQUET)
    if csv_check.shape != parquet_check.shape or list(csv_check.columns) != list(parquet_check.columns):
        raise AssertionError(f"CSV/Parquet mismatch: {csv_check.shape} vs {parquet_check.shape}")
    duplicates = int(csv_check.duplicated(["horse_id", "race_id"]).sum())
    if duplicates:
        raise AssertionError(f"Final dataset has {duplicates} duplicate horse/race keys")
    parsed = parse_dates(csv_check["race_date"])
    years = parsed.dt.year.value_counts().sort_index()
    for year in (2024, 2025, 2026):
        if int(years.get(year, 0)) == 0:
            raise AssertionError(f"Final dataset is missing year {year}")
    return len(csv_check), len(csv_check.columns)


def write_report(frame: pd.DataFrame, rows: int, columns: int) -> None:
    parsed = parse_dates(frame["race_date"])
    complete_races = int(frame.loc[frame["race_field_complete"].fillna(False), "race_id"].nunique())
    distribution = (
        pd.DataFrame({"year": parsed.dt.year, "race_id": frame["race_id"]})
        .dropna(subset=["year"]).groupby("year").agg(rows=("race_id", "size"), races=("race_id", "nunique"))
    )
    recent = distribution.loc[distribution.index.intersection([2024, 2025, 2026])]
    table = "\n".join(f"| {int(year)} | {int(row.rows):,} | {int(row.races):,} |" for year, row in recent.iterrows())
    report = f"""# Final Dataset Rebuild Report

Generated: {datetime.now():%Y-%m-%d %H:%M:%S}

## Result

- Source of truth: `pedigreeall_progress.db` / `horse_races`.
- Legacy `output/benter_features_with_komiser.csv` dependency: removed.
- Final shape: **{rows:,} rows x {columns} columns**.
- CSV/Parquet row and column synchronization: **passed**.
- Duplicate `(horse_id, race_id)` keys: **0**.
- Date parser: `pd.to_datetime(..., dayfirst=True, errors="coerce")`.
- Unparseable dates: **{int(parsed.isna().sum())}**.
- Historical rating source: post-race `GET:Tjk/Get.HP`; model input uses
  one-race-lagged `pre_race_handicap_rating` only.
- Internally complete race fields: **{complete_races:,}** races (see
  `output/race_starter_coverage.csv`).

## Recent Year Distribution

| Year | Rows | Races |
| --- | ---: | ---: |
{table}

## Root Cause

The old builder treated the existing final CSV (initially copied from the stale
`benter_features_with_komiser.csv`) as authoritative and only appended keys found
in `benter_features_base.csv`. The incremental feature job derived its query floor
from that CSV's maximum date, already in 2026, so 2024/2025 were never queried or
backfilled. Mixed ISO dates were also parsed without a stable day-first source
format. The rebuilt pipeline now starts from every current DB row and computes
rolling features using strictly earlier races for the same horse.
"""
    (REPORTS / "final_dataset_rebuild_report.md").write_text(report, encoding="utf-8")


def main() -> int:
    frame = load_base_from_db()
    frame = merge_optional_enrichments(frame)
    rows, columns = save_final_dataset(frame)
    write_report(frame, rows, columns)
    logger.info("Rebuilt final dataset: %s rows x %s columns", rows, columns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
