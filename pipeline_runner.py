"""Shared subprocess execution and JSON audit logging for VPS runners."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app_config import LOG_DIR, PROJECT_ROOT, ensure_runtime_dirs


class RunnerAlreadyActive(RuntimeError):
    """Raised for callers that do not opt into graceful overlap skipping."""


@dataclass(frozen=True)
class RunnerLockState:
    acquired: bool
    path: Path
    metadata: dict[str, object]
    stale_lock_removed: bool = False


def pid_is_running(pid: object) -> bool:
    try:
        process_id = int(pid)
        if process_id <= 0:
            return False
        if os.name == "nt":
            import ctypes
            process_query_limited_information = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(
                process_query_limited_information, False, process_id
            )
            if not handle:
                return False
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        os.kill(process_id, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (TypeError, ValueError, OSError):
        return False


def read_lock_metadata(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def lock_owner_is_running(metadata: dict[str, object]) -> bool:
    return bool(
        metadata.get("hostname") == socket.gethostname()
        and pid_is_running(metadata.get("pid"))
    )


@contextmanager
def runner_lock(name: str, *, skip_if_active: bool = False):
    """Acquire an atomic PID lock and recover abandoned lock files.

    Callers that set ``skip_if_active`` receive ``acquired=False`` instead of
    an exception when the recorded process is still alive.
    """
    ensure_runtime_dirs()
    path = LOG_DIR / f"{name}.lock"
    stale_removed = False
    owned: dict[str, object] | None = None
    try:
        for _ in range(5):
            metadata = {
                "pid": os.getpid(),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "hostname": socket.gethostname(),
                "lock_id": uuid.uuid4().hex,
                "runner": name,
            }
            try:
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
            except FileExistsError:
                existing = read_lock_metadata(path)
                if lock_owner_is_running(existing):
                    state = RunnerLockState(False, path, existing, stale_removed)
                    if skip_if_active:
                        yield state
                        return
                    raise RunnerAlreadyActive(f"Runner already active: {name}; owner={existing}")
                try:
                    path.unlink()
                    stale_removed = True
                except FileNotFoundError:
                    pass
                continue
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(metadata, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            owned = metadata
            yield RunnerLockState(True, path, metadata, stale_removed)
            return
        raise RuntimeError(f"Could not acquire runner lock after stale-lock recovery: {name}")
    finally:
        if owned is not None:
            current = read_lock_metadata(path)
            if current.get("lock_id") == owned.get("lock_id"):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass


def run_step(script: str, args: list[str] | None = None, timeout: int = 3600) -> dict[str, object]:
    command = [sys.executable, str(PROJECT_ROOT / script), *(args or [])]
    started = datetime.now(timezone.utc)
    monotonic = time.monotonic()
    exception_traceback = ""
    try:
        result = subprocess.run(
            command, cwd=PROJECT_ROOT, env=os.environ.copy(), text=True,
            capture_output=True, timeout=timeout, check=False,
        )
        exit_code, stdout, stderr = result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as exc:
        exit_code = 124
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + f"\nTimeout after {timeout}s"
        exception_traceback = traceback.format_exc()
    except Exception:
        exit_code = 1
        stdout = ""
        stderr = traceback.format_exc()
        exception_traceback = stderr
    ended = datetime.now(timezone.utc)
    return {
        "script": script, "args": args or [], "command": command,
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(), "duration_seconds": round(time.monotonic() - monotonic, 3),
        "exit_code": exit_code, "stdout": stdout, "stderr": stderr,
        "exception_traceback": exception_traceback,
    }


def write_run_log(name: str, payload: dict[str, object]) -> tuple[Path, Path]:
    ensure_runtime_dirs()
    project_logs = PROJECT_ROOT / "logs"
    project_logs.mkdir(parents=True, exist_ok=True)
    day = datetime.now().strftime("%Y_%m_%d")
    dated = project_logs / f"{name}_{day}.json"
    latest = LOG_DIR / f"{name}_latest.json"
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    for path in (dated, latest):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    return dated, latest
