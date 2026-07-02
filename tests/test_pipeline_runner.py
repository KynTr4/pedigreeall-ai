import subprocess
import unittest
from unittest.mock import patch

from pipeline_runner import run_step


class PipelineRunnerTests(unittest.TestCase):
    def test_runner_records_command_and_spawn_traceback(self):
        with patch("pipeline_runner.subprocess.run", side_effect=OSError("spawn failed")):
            result = run_step("broken.py", ["--flag"], 10)

        self.assertEqual(result["exit_code"], 1)
        self.assertTrue(str(result["command"][-2]).endswith("broken.py"))
        self.assertEqual(result["command"][-1], "--flag")
        self.assertIn("OSError: spawn failed", result["stderr"])
        self.assertIn("OSError: spawn failed", result["exception_traceback"])

    def test_runner_records_timeout_traceback(self):
        timeout = subprocess.TimeoutExpired(["python", "slow.py"], 5)
        with patch("pipeline_runner.subprocess.run", side_effect=timeout):
            result = run_step("slow.py", [], 5)

        self.assertEqual(result["exit_code"], 124)
        self.assertIn("Timeout after 5s", result["stderr"])
        self.assertIn("TimeoutExpired", result["exception_traceback"])


if __name__ == "__main__":
    unittest.main()
