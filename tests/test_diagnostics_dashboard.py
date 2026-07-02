import base64
import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

import web_app
from migrate_provenance_schema import apply_migrations


class DiagnosticsDashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db = Path(self.temp.name) / "diagnostics.db"
        apply_migrations(self.db)
        self.original_db = web_app.DB_PATH; web_app.DB_PATH = self.db
        web_app._PERFORMANCE_CACHE.clear()
        token = base64.b64encode(f"{web_app.WEB_USERNAME}:{web_app.WEB_PASSWORD}".encode()).decode()
        self.auth = {"Authorization": f"Basic {token}"}
        self.client = TestClient(web_app.app)
        self._seed("race-1", 1, (0.60, 0.30, 0.10), winner=3, odds=6.0,
                   agf=(2, 3, 1), surface="Çim", race_class="İngiliz Maiden", distance=1200)
        self._seed("race-2", 2, (0.40, 0.35, 0.25), winner=1, odds=4.0,
                   agf=(2, 1, 3), surface="Kum", race_class="Arap Handikap", distance=1800)

    def tearDown(self):
        web_app.DB_PATH = self.original_db
        web_app._PERFORMANCE_CACHE.clear()
        self.temp.cleanup()

    def _seed(self, race_id, race_no, probabilities, winner, odds, agf, surface, race_class, distance):
        start = datetime(2030, 6, 1 + race_no, 15, 0, tzinfo=timezone.utc)
        captured = start - timedelta(hours=2); prediction_time = start - timedelta(minutes=30)
        with sqlite3.connect(self.db) as db:
            snapshots = []
            for horse in range(1, 4):
                cursor = db.execute("""INSERT INTO program_snapshots(
                    race_id,horse_id,race_start_at,race_no,captured_at,source_endpoint,
                    source_request_id,horse_name,track,surface,distance,race_class)
                    VALUES(?,?,?,?,?,'test',?,?,?,?,?,?)""",
                    (race_id, f"h{horse}", start.isoformat(), race_no, captured.isoformat(),
                     f"program-{race_id}-{horse}", f"AT {horse}", "İstanbul", surface, distance, race_class))
                snapshots.append(cursor.lastrowid)
                p = probabilities[horse - 1]
                db.execute("""INSERT INTO prediction_snapshots(
                    prediction_id,model_version,pipeline_version,race_id,horse_id,prediction_time,
                    race_start_at,logistic_probability,catboost_probability,xgboost_probability,
                    ensemble_probability,predicted_rank,feature_hash,feature_values_json,
                    feature_contract_version,feature_snapshot_id,source_request_id,
                    agf_percent,agf_rank,odds)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,'v1',?,?,?,?,?)""",
                    (f"pred-{race_id}-{horse}", "model-v1", "pipe-v1", race_id, f"h{horse}",
                     prediction_time.isoformat(), start.isoformat(), p, p, p, p, horse,
                     f"hash-{race_id}-{horse}", json.dumps({
                         "last_3_avg_position": float(horse), "last_5_avg_position": float(horse + 1),
                         "jockey_horse_win_rate": 0.20 / horse, "carried_weight": 55 + horse,
                     }) if race_id == "race-1" else "{}", snapshots[-1], f"program-{race_id}-{horse}",
                     float(50 - 10 * agf[horse - 1]), int(agf[horse - 1]), float(odds if horse == 1 else 2.0)))
                db.execute("""INSERT INTO agf_snapshots(
                    race_id,horse_id,captured_at,agf_percent,agf_rank,source_request_id)
                    VALUES(?,?,?,?,?,?)""", (race_id, f"h{horse}", prediction_time.isoformat(),
                                             50 - 10 * agf[horse - 1], agf[horse - 1],
                                             f"agf-{race_id}-{horse}"))
                db.execute("""INSERT INTO odds_snapshots(
                    race_id,horse_id,captured_at,odds,source_request_id)
                    VALUES(?,?,?,?,?)""", (race_id, f"h{horse}", prediction_time.isoformat(),
                                            odds if horse == 1 else 2.0,
                                            f"odds-{race_id}-{horse}"))
                db.execute("""INSERT INTO race_results(
                    race_id,horse_id,race_start_at,race_no,captured_at,source_endpoint,
                    source_request_id,finish_position,result_odds,result_status)
                    VALUES(?,?,?,?,?,'test',?,?,?,'finished')""",
                    (race_id, f"h{horse}", start.isoformat(), race_no,
                     (start + timedelta(hours=1)).isoformat(), f"result-{race_id}-{horse}",
                     1 if horse == winner else horse + 1, odds if horse == winner else 2.0))

    def get(self, path):
        response = self.client.get(path, headers=self.auth)
        self.assertEqual(response.status_code, 200, response.text)
        return response

    def test_top_n_probability_and_agf_analysis(self):
        summary = self.get("/api/diagnostics/summary").json()
        self.assertEqual(summary["race_count"], 2)
        self.assertAlmostEqual(summary["top1_accuracy"], 50.0)
        self.assertAlmostEqual(summary["top2_accuracy"], 50.0)
        self.assertAlmostEqual(summary["top3_accuracy"], 100.0)
        self.assertAlmostEqual(summary["top5_accuracy"], 100.0)
        self.assertEqual(summary["agf_over_model"], 1)
        races = self.get("/api/diagnostics/races").json()
        self.assertEqual(races["page_size"], 100)
        wrong = next(row for row in races["rows"] if row["race_id"] == "race-1")
        self.assertEqual(wrong["winner_rank"], 3)
        self.assertAlmostEqual(wrong["winner_probability"], 0.10)
        self.assertAlmostEqual(wrong["probability_difference"], 0.50)
        self.assertEqual(wrong["winner_agf_rank"], 1)
        self.assertEqual(wrong["agf_favorite"], "AT 3")

    def test_rank_groups_extremes_filters_and_csv(self):
        ranks = self.get("/api/diagnostics/winner-ranks").json()["rows"]
        self.assertEqual({row["winner_rank"]: row["race_count"] for row in ranks}, {1: 1, 3: 1})
        groups = self.get("/api/diagnostics/groups").json()["rows"]
        maiden = next(row for row in groups if row["dimension"] == "Yarış Tipi" and row["group_name"] == "Maiden")
        self.assertEqual(maiden["race_count"], 1)
        self.assertAlmostEqual(maiden["top3_accuracy"], 100.0)
        extremes = self.get("/api/diagnostics/extremes").json()
        self.assertEqual(extremes["errors"][0]["race_id"], "race-1")
        filtered = self.get("/api/diagnostics/races?surface=Çim").json()
        self.assertEqual(filtered["total"], 1)
        csv_response = self.get("/api/diagnostics/export.csv?model=Ensemble")
        self.assertIn("text/csv", csv_response.headers["content-type"])
        self.assertIn("winner_rank", csv_response.text)
        self.assertIn("race-1", csv_response.text)

    def test_shap_unavailable_auth_and_read_only(self):
        self.assertEqual(self.client.get("/diagnostics").status_code, 401)
        page = self.get("/diagnostics").text
        self.assertIn("Model Diagnostics", page)
        for model in ("Ensemble", "Logistic", "CatBoost", "XGBoost"):
            self.assertIn(f'value="{model}"', page)
        self.assertIn('<select id="diag-model" name="model" class="form-select">', page)
        self.assertIn('<option value="">Tümü</option>', page)
        self.assertNotIn('role="radiogroup"', page)
        shap = self.get("/api/diagnostics/feature-contribution").json()
        self.assertFalse(shap["available"])
        self.assertEqual(self.client.get("/api/diagnostics/summary?model=Unknown", headers=self.auth).status_code, 400)
        with web_app.readonly_connection() as db:
            self.assertEqual(db.execute("PRAGMA query_only").fetchone()[0], 1)

    def test_race_detail_explains_selection_without_rerunning_model(self):
        detail = self.get("/api/diagnostics/race/race-1?model=Ensemble").json()
        self.assertEqual(detail["model_selection"]["horse_name"], "AT 1")
        self.assertEqual(detail["winner"]["horse_name"], "AT 3")
        self.assertEqual(detail["winner_rank"], 3)
        self.assertAlmostEqual(detail["probability_difference"], 0.5)
        self.assertEqual(detail["confidence"]["label"], "High Confidence Wrong")
        self.assertEqual(len(detail["ranking"]), 3)
        self.assertTrue(detail["feature_snapshot_available"])
        self.assertTrue(any(row["feature"] == "last_3_avg_position" for row in detail["feature_comparison"]))
        self.assertFalse(detail["shap"]["available"])
        page = self.get("/diagnostics/race/race-1?model=Ensemble")
        self.assertIn("NEDEN BU TAHMİN YAPILDI?", page.text)

    def test_race_detail_reports_missing_feature_snapshot(self):
        detail = self.get("/api/diagnostics/race/race-2").json()
        self.assertFalse(detail["feature_snapshot_available"])
        self.assertIn("not archived", detail["feature_snapshot_message"])


if __name__ == "__main__":
    unittest.main()
