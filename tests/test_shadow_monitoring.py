import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from feature_contract import FEATURE_CONTRACT_VERSION, MODEL_FEATURES
from migrate_provenance_schema import apply_migrations
from shadow_mode import archive_predictions
from shadow_monitor import (
    calculate_live_metrics, feature_drift, latest_prediction_runs,
    match_prediction_results, model_drift, roi_report_data,
)


class ShadowMonitoringTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "shadow.db"
        apply_migrations(self.db)
        with sqlite3.connect(self.db) as db:
            db.execute(
                """INSERT INTO program_snapshots(
                       race_id,horse_id,race_start_at,race_no,captured_at,source_endpoint,
                       source_request_id,draw,carried_weight,jockey,trainer,
                       handicap_rating,race_class,track,surface,distance,horse_name)
                   VALUES('r1','h1','2020-01-01T12:00:00+00:00',1,
                          '2020-01-01T09:00:00+00:00','test','source1',1,55,'J','T',
                          70,'A','Track','K:',1400,'Horse')"""
            )
            self.snapshot_id = db.execute("SELECT snapshot_id FROM program_snapshots").fetchone()[0]

    def tearDown(self):
        self.tmp.cleanup()

    def scored(self):
        row = {
            "race_id": "r1", "horse_id": "h1", "horse_name": "Horse",
            "race_start_at": "2020-01-01T12:00:00+00:00",
            "snapshot_id": self.snapshot_id, "source_request_id": "source1",
            "logistic_probability": 1.0, "catboost_probability": 1.0,
            "xgboost_probability": 1.0, "ensemble_probability": 1.0,
            "predicted_rank": 1,
        }
        row.update({feature: ("missing" if feature in {"track", "surface", "race_class"} else 0.0) for feature in MODEL_FEATURES})
        return pd.DataFrame([row])

    def test_prediction_archive_is_append_only_and_traceable(self):
        archive = archive_predictions(self.scored(), self.db, "2020-01-01T10:00:00+00:00")
        self.assertEqual(len(archive), 1)
        with sqlite3.connect(self.db) as db:
            row = db.execute("SELECT feature_hash,feature_values_json,feature_contract_version FROM prediction_snapshots").fetchone()
            self.assertTrue(row[0]); self.assertTrue(row[1]); self.assertEqual(row[2], FEATURE_CONTRACT_VERSION)
            feature_row = db.execute(
                "SELECT prediction_id,feature_values_json,feature_hash FROM prediction_feature_snapshots"
            ).fetchone()
            self.assertEqual(feature_row[0], archive.iloc[0].prediction_id)
            self.assertEqual(feature_row[1], row[1]); self.assertEqual(feature_row[2], row[0])
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute("UPDATE prediction_snapshots SET predicted_rank=2")
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute("UPDATE prediction_feature_snapshots SET feature_hash='changed'")

    def test_agf_and_odds_archived_from_db(self):
        with sqlite3.connect(self.db) as db:
            db.execute(
                """INSERT INTO agf_snapshots(race_id, horse_id, captured_at, agf_percent, agf_rank, source_request_id)
                   VALUES('r1', 'h1', '2020-01-01T09:30:00+00:00', 35.5, 2, 'source_agf')"""
            )
            db.execute(
                """INSERT INTO odds_snapshots(race_id, horse_id, captured_at, odds, source_request_id)
                   VALUES('r1', 'h1', '2020-01-01T09:30:00+00:00', 4.5, 'source_odds')"""
            )
        scored = self.scored()
        archive = archive_predictions(scored, self.db, "2020-01-01T10:00:00+00:00")
        self.assertEqual(len(archive), 1)
        with sqlite3.connect(self.db) as db:
            db.row_factory = sqlite3.Row
            row = db.execute("SELECT agf_percent, agf_rank, odds FROM prediction_snapshots").fetchone()
            self.assertEqual(row["agf_percent"], 35.5)
            self.assertEqual(row["agf_rank"], 2)
            self.assertEqual(row["odds"], 4.5)

    def test_post_start_prediction_is_rejected(self):
        with self.assertRaises(ValueError):
            archive_predictions(self.scored(), self.db, "2020-01-01T13:00:00+00:00")

    def test_result_matching_is_separate_and_append_only(self):
        archive = archive_predictions(self.scored(), self.db, "2020-01-01T10:00:00+00:00")
        with sqlite3.connect(self.db) as db:
            db.execute(
                """INSERT INTO race_results(
                       race_id,horse_id,race_start_at,race_no,captured_at,source_endpoint,
                       source_request_id,finish_position,finish_time,prize,margin,
                       result_odds,result_status)
                   VALUES('r1','h1','2020-01-01T12:00:00+00:00',1,
                          '2020-01-01T13:00:00+00:00','result','result1',1,
                          '1.20.00',100,NULL,3.5,'finished')"""
            )
        self.assertEqual(match_prediction_results(self.db), 1)
        self.assertEqual(match_prediction_results(self.db), 0)
        with sqlite3.connect(self.db) as db:
            result = db.execute("SELECT winner,payout FROM prediction_results WHERE prediction_id=?", (archive.iloc[0].prediction_id,)).fetchone()
            self.assertEqual(result, (1, 3.5))

    def test_latest_prediction_run_is_used(self):
        frame = pd.DataFrame({
            "race_id": ["r1", "r1", "r1", "r1"],
            "prediction_time_parsed": pd.to_datetime([
                "2030-01-01T09:00:00Z", "2030-01-01T09:00:00Z",
                "2030-01-01T10:00:00Z", "2030-01-01T10:00:00Z",
            ]),
            "horse_id": ["h1", "h2", "h1", "h2"],
        })
        latest = latest_prediction_runs(frame)
        self.assertEqual(len(latest), 2)
        self.assertEqual(latest.prediction_time_parsed.nunique(), 1)

    def test_roi_requires_certified_pre_race_odds(self):
        frame = pd.DataFrame({
            "race_id": ["r1", "r1"], "horse_id": ["h1", "h2"],
            "winner": [1, 0], "ensemble_probability": [0.7, 0.3],
            "pre_race_odds": [float("nan"), float("nan")],
        })
        _, status = roi_report_data(frame)
        self.assertEqual(status, "NOT CERTIFIED")

    def monitoring_frame(self, shifted=False):
        rows = []
        for day, high in [("2029-12-15", False), ("2030-01-01", shifted)]:
            for race in range(25):
                for horse in range(4):
                    probability = [0.55, 0.25, 0.15, 0.05][horse]
                    if high:
                        probability = [0.05, 0.15, 0.25, 0.55][horse]
                    features = {
                        feature: ("NEW" if high and feature in {"track", "surface", "race_class"}
                                  else "BASE" if feature in {"track", "surface", "race_class"}
                                  else 100.0 if high else float(horse))
                        for feature in MODEL_FEATURES
                    }
                    rows.append({
                        "prediction_id": f"{day}-{race}-{horse}",
                        "race_id": f"{day}-r{race}", "horse_id": f"h{horse}",
                        "race_date": day, "winner": int(horse == 0),
                        "result_status": "finished", "pre_race_odds": [2, 4, 8, 12][horse],
                        "feature_values_json": __import__("json").dumps(features),
                        "logistic_probability": probability,
                        "catboost_probability": probability,
                        "xgboost_probability": probability,
                        "ensemble_probability": probability,
                    })
        return pd.DataFrame(rows)

    def test_daily_metrics_compare_all_models(self):
        metrics = calculate_live_metrics(self.monitoring_frame())
        self.assertEqual(set(metrics.model), {"logistic", "catboost", "xgboost", "ensemble"})
        latest = metrics[(metrics.window == "daily") & (metrics.metric_date == "2030-01-01")]
        self.assertTrue(latest.top1_accuracy.eq(1.0).all())

    def test_critical_prediction_and_feature_drift_detected(self):
        frame = self.monitoring_frame(shifted=True)
        _, prediction_status = model_drift(frame)
        _, feature_status = feature_drift(frame)
        self.assertEqual(prediction_status, "CRITICAL")
        self.assertEqual(feature_status, "CRITICAL")

    def test_certified_roi_is_computed(self):
        roi, status = roi_report_data(self.monitoring_frame())
        self.assertEqual(status, "CERTIFIED")
        self.assertIn("flat_betting", set(roi.strategy))


if __name__ == "__main__":
    unittest.main()
