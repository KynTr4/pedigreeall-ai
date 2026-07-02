"""Linux/systemd daily pipeline with fail-closed step execution."""
from __future__ import annotations

import shlex
import sys
import traceback
from datetime import datetime, timezone

from pipeline_runner import run_step, runner_lock, write_run_log

STEPS = [
    ("update_race_programs.py", [], 1800),
    ("snapshot_store.py", [], 300),
    ("download_agfv2.py", ["--today", "--tables", "1", "2", "--force-refresh"], 1800),
    ("komiser.py", ["--today"], 1800),
    ("process_komiser.py", ["--today"], 1800),
    ("update_track_conditions.py", [], 900),
    ("update_workouts.py", [], 900),
    ("update_results.py", [], 1800),
    ("build_asof_features.py", [], 1800),
    ("validate_feature_provenance.py", [], 900),
    ("shadow_monitor.py", [], 1800),
]


class PipelineStepError(RuntimeError):
    def __init__(self, result: dict[str, object]):
        self.result = result
        super().__init__(
            f"{result['script']} exited with code {result['exit_code']}"
        )


def failure_report(result: dict[str, object], parent_traceback: str) -> str:
    command = result.get("command") or [result["script"], *result.get("args", [])]
    command_text = shlex.join(str(part) for part in command)
    child_traceback = str(result.get("exception_traceback") or "").strip()
    tracebacks = parent_traceback.strip()
    if child_traceback and child_traceback not in tracebacks:
        tracebacks += f"\n\nChild process traceback:\n{child_traceback}"
    return "\n".join([
        "DAILY PIPELINE STEP FAILED",
        f"script: {result['script']}",
        f"command: {command_text}",
        f"exit code: {result['exit_code']}",
        f"elapsed time: {result.get('duration_seconds', 0)} seconds",
        "Python traceback:",
        tracebacks or "(child returned non-zero without raising a Python exception)",
        "stderr:",
        str(result.get("stderr") or "(empty)"),
        "stdout:",
        str(result.get("stdout") or "(empty)"),
    ])


def main() -> int:
    payload = {"runner": "daily", "started_at": datetime.now(timezone.utc).isoformat(), "steps": []}
    current_script = None
    try:
        with runner_lock("daily_pipeline"):
            for script, args, timeout in STEPS:
                current_script = script
                print(f"[daily] START {script}", flush=True)
                result = run_step(script, args, timeout)
                payload["steps"].append(result)
                if int(result["exit_code"]) != 0:
                    raise PipelineStepError(result)
                print(
                    f"[daily] OK {script} ({result['duration_seconds']}s)",
                    flush=True,
                )
        payload["status"] = "success"
    except PipelineStepError as exc:
        payload.update({
            "status": "failed",
            "failed_step": exc.result["script"],
            "traceback": traceback.format_exc(),
        })
        print(
            failure_report(exc.result, str(payload["traceback"])),
            file=sys.stderr,
            flush=True,
        )
    except Exception:
        payload.update({
            "status": "failed",
            "failed_step": current_script or "daily_pipeline",
            "traceback": traceback.format_exc(),
        })
        print(str(payload["traceback"]), file=sys.stderr, flush=True)
    finally:
        payload["ended_at"] = datetime.now(timezone.utc).isoformat()
        write_run_log("run", payload)
    return 0 if payload["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
