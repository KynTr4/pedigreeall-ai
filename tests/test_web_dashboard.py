import base64
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

import web_app
from migrate_provenance_schema import apply_migrations


class WebDashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = Path(self.temp.name)
        self.db = root / "test.db"
        apply_migrations(self.db)
        self.logs = root / "logs"; self.logs.mkdir()
        self.reports = root / "reports"; self.reports.mkdir()
        self.backups = root / "backups"; self.backups.mkdir()
        self.originals = (web_app.DB_PATH, web_app.LOG_DIR, web_app.REPORTS_DIR, web_app.BACKUP_DIR)
        web_app.DB_PATH, web_app.LOG_DIR, web_app.REPORTS_DIR, web_app.BACKUP_DIR = (
            self.db, self.logs, self.reports, self.backups,
        )
        web_app._PERFORMANCE_CACHE.clear()
        self.client = TestClient(web_app.app)
        token = base64.b64encode(f"{web_app.WEB_USERNAME}:{web_app.WEB_PASSWORD}".encode()).decode()
        self.auth = {"Authorization": f"Basic {token}"}

    def seed_performance_race(self, index: int = 0):
        race_id = f"performance-race-{index}"
        start = datetime(2030, 1, 1, 15, 30, tzinfo=timezone.utc) + timedelta(days=index)
        captured = start - timedelta(hours=2); predicted = start - timedelta(hours=1)
        with sqlite3.connect(self.db) as connection:
            snapshot_ids = []
            for horse_no, horse_name in ((1, "KARA ŞİMŞEK"), (2, "RÜZGAR")):
                source = f"program-{index}-{horse_no}"
                cursor = connection.execute(
                    """INSERT INTO program_snapshots(
                           race_id,horse_id,race_start_at,race_no,captured_at,
                           source_endpoint,source_request_id,horse_name,track)
                       VALUES(?,?,?,?,?,'test',?,?,?)""",
                    (race_id, f"horse-{horse_no}", start.isoformat(), index + 1,
                     captured.isoformat(), source, horse_name, "İstanbul"),
                )
                snapshot_ids.append(cursor.lastrowid)
            probabilities = ((0.60, 0.20, 0.70, 0.65), (0.40, 0.80, 0.30, 0.35))
            for horse_no, values in enumerate(probabilities, 1):
                connection.execute(
                    """INSERT INTO prediction_snapshots(
                           prediction_id,model_version,pipeline_version,race_id,horse_id,
                           prediction_time,race_start_at,logistic_probability,
                           catboost_probability,xgboost_probability,ensemble_probability,
                           predicted_rank,feature_hash,feature_values_json,
                           feature_contract_version,feature_snapshot_id,source_request_id,
                           odds)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,'{}','v1',?,?,?)""",
                    (f"prediction-{index}-{horse_no}", "model-v1", "pipeline-v1", race_id,
                     f"horse-{horse_no}", predicted.isoformat(), start.isoformat(), *values,
                     horse_no, f"hash-{index}-{horse_no}", snapshot_ids[horse_no - 1],
                     f"program-{index}-{horse_no}", 3.40 if horse_no == 1 else 2.10),
                )
                connection.execute(
                    """INSERT INTO race_results(
                           race_id,horse_id,race_start_at,race_no,captured_at,
                           source_endpoint,source_request_id,finish_position,result_odds,result_status)
                       VALUES(?,?,?,?,?,'test',?,?,?,'finished')""",
                    (race_id, f"horse-{horse_no}", start.isoformat(), index + 1,
                     (start + timedelta(hours=1)).isoformat(), f"result-{index}-{horse_no}",
                     horse_no, 3.40 if horse_no == 1 else 2.10),
                )
        web_app._PERFORMANCE_CACHE.clear()

    def tearDown(self):
        web_app.DB_PATH, web_app.LOG_DIR, web_app.REPORTS_DIR, web_app.BACKUP_DIR = self.originals
        self.temp.cleanup()

    def test_basic_auth_is_required_for_html_api_and_static(self):
        for path in ("/", "/performance", "/api/health", "/api/performance/summary", "/static/style.css"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 401)
            self.assertIn("Basic", response.headers["www-authenticate"])
            self.assertEqual(self.client.get(path, headers=self.auth).status_code, 200)

    @patch("web_app.server_today", return_value="2026-07-02")
    def test_date_pages_default_to_server_today_and_honor_url(self, _server_today):
        inputs = {
            "/bet-simulator": '<input name="date" type="date" value="{}"',
            "/performance": '<input id="filter-date" name="date" type="date" value="{}"',
            "/diagnostics": '<input name="date" type="date" value="{}"',
            "/races": '<input id="race-day-date" type="date" value="{}"',
        }
        for path, input_html in inputs.items():
            with self.subTest(path=path, source="server"):
                page = self.client.get(path, headers=self.auth).text
                self.assertIn(input_html.format("2026-07-02"), page)
                self.assertNotIn("localStorage", page)
                self.assertNotIn("sessionStorage", page)
            with self.subTest(path=path, source="url"):
                page = self.client.get(f"{path}?date=2026-06-30", headers=self.auth).text
                self.assertIn(input_html.format("2026-06-30"), page)

        bet_page = self.client.get("/bet-simulator", headers=self.auth).text
        self.assertNotIn('value = "2026-06-30"', bet_page)

    def test_database_connection_is_enforced_read_only(self):
        with web_app.readonly_connection() as connection:
            self.assertEqual(connection.execute("PRAGMA query_only").fetchone()[0], 1)
            with self.assertRaises(sqlite3.OperationalError):
                connection.execute("CREATE TABLE forbidden_write(x)")

    def test_report_allowlist_blocks_path_traversal(self):
        (self.reports / "leakage_gate_v2.md").write_text("PASS", encoding="utf-8")
        self.assertEqual(
            web_app._allowed_text(self.reports, "leakage_gate_v2.md", web_app.ALLOWED_REPORTS),
            "PASS",
        )
        with self.assertRaises(HTTPException):
            web_app._allowed_text(self.reports, "../.env", web_app.ALLOWED_REPORTS)

    def test_prediction_probability_sum_is_checked_per_race(self):
        with sqlite3.connect(self.db) as connection:
            snapshot_ids = []
            for index in (1, 2):
                cursor = connection.execute(
                    """INSERT INTO program_snapshots(
                           race_id,horse_id,race_start_at,race_no,captured_at,
                           source_endpoint,source_request_id,horse_name,track)
                       VALUES('race-1',?,'2030-01-01T12:00:00+00:00',1,
                              '2030-01-01T10:00:00+00:00','test',?,?,?)""",
                    (f"horse-{index}", f"program-{index}", f"Horse {index}", "İstanbul"),
                )
                snapshot_ids.append(cursor.lastrowid)
            for index, probability in enumerate((0.6, 0.4), 1):
                connection.execute(
                    """INSERT INTO prediction_snapshots(
                           prediction_id,model_version,pipeline_version,race_id,horse_id,
                           prediction_time,race_start_at,logistic_probability,
                           catboost_probability,xgboost_probability,ensemble_probability,
                           predicted_rank,feature_hash,feature_values_json,
                           feature_contract_version,feature_snapshot_id,source_request_id)
                       VALUES(?,?,?,?,?,'2030-01-01T11:00:00+00:00','2030-01-01T12:00:00+00:00',
                              ?,?,?,?,?,?,'{}','v1',?,?)""",
                    (f"prediction-{index}", "model", "pipeline", "race-1", f"horse-{index}",
                     probability, probability, probability, probability, index, f"hash-{index}",
                     snapshot_ids[index - 1], f"program-{index}"),
                )
        predictions = web_app.current_predictions()
        self.assertEqual(len(predictions), 2)
        self.assertTrue(all(row["probability_sum_valid"] for row in predictions))
        self.assertTrue(all(row["probability_sum"] == 1.0 for row in predictions))

    def test_performance_summary_models_history_and_filters(self):
        self.seed_performance_race()
        summary = self.client.get("/api/performance/summary", headers=self.auth).json()
        self.assertEqual(summary["total_predictions"], 4)
        self.assertEqual(summary["correct_predictions"], 3)
        self.assertAlmostEqual(summary["accuracy_percent"], 75.0)
        self.assertAlmostEqual(summary["net_profit"], 6.2)
        self.assertAlmostEqual(summary["roi_percent"], 155.0)
        self.assertEqual(summary["processed_races"], 1)

        models = self.client.get("/api/performance/models", headers=self.auth).json()["models"]
        by_name = {row["model"]: row for row in models}
        self.assertEqual(by_name["CatBoost"]["correct"], 0)
        self.assertEqual(by_name["Ensemble"]["correct"], 1)

        history = self.client.get("/api/performance/history", headers=self.auth).json()
        self.assertEqual(history["page_size"], 100)
        self.assertEqual(history["total"], 4)
        ensemble = next(row for row in history["rows"] if row["model"] == "Ensemble")
        self.assertEqual(ensemble["city"], "İstanbul")
        self.assertEqual(ensemble["race_time"], "18:30")
        self.assertEqual(ensemble["predicted_horse"], "KARA ŞİMŞEK")
        self.assertEqual(ensemble["winner_name"], "KARA ŞİMŞEK")
        self.assertEqual(ensemble["correct"], 1)
        self.assertAlmostEqual(ensemble["net_return"], 2.4)

        filtered = self.client.get(
            "/api/performance/history?model=CatBoost&outcome=correct", headers=self.auth
        ).json()
        self.assertEqual(filtered["total"], 0)
        self.assertEqual(self.client.get(
            "/api/performance/history?model=Unknown", headers=self.auth
        ).status_code, 400)
        normalized = self.client.get("/api/performance/summary", params={
            "date":"2030-01-01","track":"İstanbul (33. Yarış Günü)","model":"Ensemble"
        }, headers=self.auth).json()
        self.assertTrue(normalized["has_data"]); self.assertEqual(normalized["processed_races"], 1)
        for path in ("summary", "models", "history", "chart"):
            response = self.client.get(
                f"/api/performance/{path}?date=2030-01-01&track=İstanbul&outcome=correct",
                headers=self.auth,
            )
            self.assertEqual(response.status_code, 200, response.text)

    def test_bet_simulator_default_and_variable_stake(self):
        self.seed_performance_race()
        base=self.client.get("/api/bet-simulator/summary?model=Ensemble&stake=20",headers=self.auth).json()
        self.assertEqual(base["total_races"],1);self.assertEqual(base["bet_races"],1)
        self.assertAlmostEqual(base["total_invested"],20);self.assertAlmostEqual(base["total_return"],68)
        self.assertAlmostEqual(base["net_profit"],48);self.assertAlmostEqual(base["roi_percent"],240)
        larger=self.client.get("/api/bet-simulator/summary?model=Ensemble&stake=50",headers=self.auth).json()
        self.assertAlmostEqual(larger["net_profit"],120)
        wrong=self.client.get("/api/bet-simulator/summary?model=CatBoost&stake=20",headers=self.auth).json()
        self.assertAlmostEqual(wrong["net_profit"],-20)
        history=self.client.get("/api/bet-simulator/history?model=Ensemble&stake=20",headers=self.auth).json()
        self.assertAlmostEqual(history["rows"][0]["return_amount"],68)
        export=self.client.get("/api/bet-simulator/export.csv?model=Ensemble&stake=20",headers=self.auth)
        self.assertEqual(export.status_code,200);self.assertIn("net_profit",export.text)
        self.assertEqual(self.client.get("/api/bet-simulator/summary?stake=0",headers=self.auth).status_code,400)

    def test_performance_history_paginates_at_100_rows(self):
        for index in range(26):
            self.seed_performance_race(index)
        first = self.client.get("/api/performance/history?page=1", headers=self.auth).json()
        second = self.client.get("/api/performance/history?page=2", headers=self.auth).json()
        self.assertEqual(first["total"], 104)
        self.assertEqual(len(first["rows"]), 100)
        self.assertEqual(len(second["rows"]), 4)
        self.assertEqual(second["pages"], 2)

    def test_performance_chart_and_race_filter_endpoints(self):
        self.seed_performance_race()
        chart = self.client.get("/api/performance/chart", headers=self.auth).json()
        self.assertEqual(len(chart["daily"]), 1)
        self.assertAlmostEqual(chart["daily"][0]["accuracy_percent"], 75.0)
        self.assertAlmostEqual(chart["daily"][0]["cumulative_profit"], 6.2)
        races = self.client.get("/api/performance/races", headers=self.auth).json()
        self.assertEqual(races["tracks"][0]["city"], "İstanbul")
        self.assertEqual(races["models"], ["Logistic", "CatBoost", "XGBoost", "Ensemble"])


if __name__ == "__main__":
    unittest.main()
