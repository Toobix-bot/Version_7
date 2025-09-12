#!/usr/bin/env python3
"""Create a git snapshot tag and optional SQLite DB copy before applying a plan."""
from __future__ import annotations
import subprocess, time, os, shutil, sys

def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()

def main() -> None:
    ts = int(time.time())
    tag = f"snapshot-{ts}"
    # ensure clean index
    try:
        status = run(["git", "status", "--porcelain"])
    except Exception as e:  # noqa: BLE001
        print(f"Git unavailable: {e}")
        sys.exit(1)
    if status:
        print("Working tree not clean; aborting snapshot.")
        sys.exit(2)
    run(["git", "tag", tag])
    db_path = os.getenv("DB_PATH", "agent.db")
    if os.path.exists(db_path):
        backup_name = f"{db_path}.bak.{ts}"
        shutil.copy2(db_path, backup_name)
        print(f"Copied DB to {backup_name}")
    print(f"Created tag {tag}")
    print("Restore: git checkout <tag>")

if __name__ == "__main__":
    main()
