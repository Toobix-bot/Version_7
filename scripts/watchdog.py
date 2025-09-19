"""
Simple watchdog to keep a FastAPI/Uvicorn server running.

Features
- Starts uvicorn with given host/port and app path.
- Restarts on exit with exponential backoff (resets after stable uptime).
- Graceful shutdown on Ctrl+C (terminates child); optional stop file support.

Usage (Windows PowerShell)
  # Using workspace venv
  .venv\\Scripts\\python.exe scripts\\watchdog.py --port 8001

  # Custom app
  .venv\\Scripts\\python.exe scripts\\watchdog.py --app api.app:app --host 127.0.0.1 --port 8003
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def build_uvicorn_cmd(python: str, app: str, host: str, port: int, reload: bool = False) -> list[str]:
    cmd: list[str] = [python, "-m", "uvicorn", app, "--host", host, "--port", str(port)]
    # Tweak logging if desired
    # cmd += ["--log-level", "info"]
    if reload:
        cmd.append("--reload")
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Watchdog for uvicorn")
    parser.add_argument("--app", default=os.getenv("UVICORN_APP", "api.app:app"), help="Uvicorn app path, e.g. module:app")
    parser.add_argument("--host", default=os.getenv("APP_HOST", "127.0.0.1"), help="Bind host")
    parser.add_argument("--port", type=int, default=int(os.getenv("APP_PORT", "8001")), help="Bind port")
    parser.add_argument("--python", default=os.getenv("PYTHON_EXE", sys.executable), help="Python executable to use")
    parser.add_argument("--reload", action="store_true", help="Pass --reload to uvicorn (dev only)")
    parser.add_argument("--stop-file", default=os.getenv("WATCHDOG_STOP_FILE", "scripts/.watchdog.stop"), help="Path to stop-file; if present at loop boundary, watchdog exits")
    parser.add_argument("--max-backoff", type=float, default=float(os.getenv("WATCHDOG_MAX_BACKOFF", "30")), help="Maximum backoff seconds")
    parser.add_argument("--reset-after", type=float, default=float(os.getenv("WATCHDOG_RESET_AFTER", "60")), help="If child ran at least this many seconds, backoff resets")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    stop_file = (project_root / args.stop_file).resolve() if not Path(args.stop_file).is_absolute() else Path(args.stop_file)

    backoff = 1.0
    max_backoff = max(1.0, float(args.max_backoff))
    reset_after = max(1.0, float(args.reset_after))

    print(f"[{ts()}] Watchdog starting in {project_root}")
    print(f"[{ts()}] App={args.app} Host={args.host} Port={args.port} Python={args.python}")
    print(f"[{ts()}] Stop file: {stop_file}")

    child: subprocess.Popen | None = None

    def shutdown_child():
        nonlocal child
        if child and child.poll() is None:
            try:
                print(f"[{ts()}] Stopping child (pid={child.pid})…")
                # Try graceful first
                if os.name == "nt":
                    child.terminate()  # sends CTRL-BREAK equivalent on Windows console
                else:
                    child.send_signal(signal.SIGTERM)
                try:
                    child.wait(timeout=8)
                except Exception:
                    pass
                if child.poll() is None:
                    print(f"[{ts()}] Child still running, killing…")
                    child.kill()
            except Exception as e:
                print(f"[{ts()}] Error stopping child: {e}")

    try:
        while True:
            if stop_file.exists():
                print(f"[{ts()}] Stop file detected at {stop_file}. Exiting watchdog.")
                try:
                    stop_file.unlink(missing_ok=True)  # type: ignore[arg-type]
                except Exception:
                    pass
                return 0

            cmd = build_uvicorn_cmd(args.python, args.app, args.host, args.port, reload=args.reload)
            print(f"[{ts()}] Launch: {' '.join(cmd)}")
            started_at = time.monotonic()
            # Note: inherit env and stdio to see logs
            child = subprocess.Popen(cmd, cwd=str(project_root))
            ret = None
            try:
                ret = child.wait()
            except KeyboardInterrupt:
                print(f"[{ts()}] KeyboardInterrupt received (watchdog). Shutting down…")
                shutdown_child()
                return 0

            uptime = time.monotonic() - started_at
            print(f"[{ts()}] Child exited with code {ret} (uptime {uptime:.1f}s)")

            # Backoff handling
            if uptime >= reset_after:
                backoff = 1.0
            else:
                backoff = min(max_backoff, backoff * 2)

            print(f"[{ts()}] Restarting in {backoff:.1f}s… (Ctrl+C to stop)")
            # Sleep in small chunks to notice stop-file sooner
            slept = 0.0
            while slept < backoff:
                if stop_file.exists():
                    print(f"[{ts()}] Stop file detected. Exiting watchdog.")
                    try:
                        stop_file.unlink(missing_ok=True)  # type: ignore[arg-type]
                    except Exception:
                        pass
                    return 0
                time.sleep(0.5)
                slept += 0.5

    finally:
        shutdown_child()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
