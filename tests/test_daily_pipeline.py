import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import run_daily_pipeline


class Lock:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class DailyPipelineTests(unittest.TestCase):
    def result(self, script, exit_code=0):
        return {
            "script": script,
            "args": [],
            "command": ["python", script],
            "exit_code": exit_code,
            "stdout": "child stdout",
            "stderr": "child stderr",
            "exception_traceback": "",
            "duration_seconds": 1.25,
        }

    def test_nonzero_step_is_printed_and_stops_pipeline(self):
        calls = []

        def fake_run(script, args, timeout):
            calls.append(script)
            return self.result(script, 1 if script == "snapshot_store.py" else 0)

        stderr = io.StringIO()
        with patch("run_daily_pipeline.run_step", side_effect=fake_run), \
             patch("run_daily_pipeline.runner_lock", return_value=Lock()), \
             patch("run_daily_pipeline.write_run_log") as write_log, \
             redirect_stdout(io.StringIO()), redirect_stderr(stderr):
            exit_code = run_daily_pipeline.main()

        report = stderr.getvalue()
        self.assertEqual(exit_code, 1)
        self.assertEqual(calls, ["update_race_programs.py", "snapshot_store.py"])
        self.assertIn("script: snapshot_store.py", report)
        self.assertIn("command: python snapshot_store.py", report)
        self.assertIn("exit code: 1", report)
        self.assertIn("Python traceback:", report)
        self.assertIn("child stderr", report)
        self.assertIn("child stdout", report)
        payload = write_log.call_args.args[1]
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["failed_step"], "snapshot_store.py")

    def test_all_successful_steps_return_zero(self):
        with patch(
            "run_daily_pipeline.run_step",
            side_effect=lambda script, args, timeout: self.result(script),
        ), patch("run_daily_pipeline.runner_lock", return_value=Lock()), \
             patch("run_daily_pipeline.write_run_log"), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(run_daily_pipeline.main(), 0)


if __name__ == "__main__":
    unittest.main()
