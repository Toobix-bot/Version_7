"""Infrastructure primitives: database init, event emission, event queue access.
Phase 1 extracted from monolithic app.py
"""
from __future__ import annotations
import os, json, time, asyncio, sqlite3, random
from typing import Any, Dict
from fastapi import HTTPException

DB_PATH = os.getenv("DB_PATH", "agent.db")
# global connection (kept for compatibility; later DI)
db_conn: sqlite3.Connection | None = None
# per-loop event queues
_event_queues: dict[int, asyncio.Queue[dict[str, Any]]] = {}
_last_activity_ts = time.time()

def get_db() -> sqlite3.Connection | None:
    return db_conn

def get_last_activity() -> float:
    return _last_activity_ts

def _get_event_queue() -> asyncio.Queue[dict[str, Any]]:
    loop = asyncio.get_running_loop()
    q = _event_queues.get(id(loop))
    if q is None:
        q = asyncio.Queue()
        _event_queues[id(loop)] = q
    return q

async def emit_event(kind: str, data: dict[str, Any]) -> None:  # replicated contract
    global _last_activity_ts
    record = {"ts": time.time(), "kind": kind, "data": data}
    if db_conn:
        try:
            db_conn.execute(
                "INSERT INTO events(ts, kind, data) VALUES (?,?,?)",
                (record["ts"], record["kind"], json.dumps(record["data"]))
            )
            db_conn.commit()
        except Exception:
            pass
    await _get_event_queue().put(record)
    if not kind.startswith("thought."):
        _last_activity_ts = time.time()

# Expose queue getter for SSE
get_event_queue = _get_event_queue

RISK_PATTERNS = [r"\b(os\.system|subprocess\.)", r"\.\./", r"/etc/passwd", r"\b(?:ssh|https?|ftp)://", r"(?i)api[_-]?key", r"(?i)\b(token|secret)\b"]

def init_db() -> None:
    global db_conn
    first = not os.path.exists(DB_PATH)
    db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    db_conn.execute("PRAGMA journal_mode=WAL;")
    db_conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            kind TEXT NOT NULL,
            data TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            agent_id TEXT NOT NULL,
            action_type TEXT NOT NULL,
            latency_ms REAL NOT NULL,
            input_size INTEGER,
            output_size INTEGER,
            score REAL,
            meta TEXT
        );
        CREATE TABLE IF NOT EXISTS proposals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            description TEXT NOT NULL,
            artifact_path TEXT NOT NULL,
            target_files TEXT,
            meta TEXT
        );
        CREATE TABLE IF NOT EXISTS llm_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            model TEXT NOT NULL,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            total_tokens INTEGER,
            latency_ms REAL,
            meta TEXT
        );
        CREATE TABLE IF NOT EXISTS thoughts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            text TEXT NOT NULL,
            kind TEXT NOT NULL,
            meta TEXT
        );
        CREATE TABLE IF NOT EXISTS story_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            ts REAL NOT NULL,
            epoch INTEGER NOT NULL,
            mood TEXT NOT NULL,
            arc TEXT NOT NULL,
            resources TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS story_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            epoch INTEGER NOT NULL,
            kind TEXT NOT NULL,
            text TEXT NOT NULL,
            mood TEXT NOT NULL,
            deltas TEXT,
            tags TEXT,
            option_ref TEXT
        );
        CREATE TABLE IF NOT EXISTS story_options (
            id TEXT PRIMARY KEY,
            created_at REAL NOT NULL,
            label TEXT NOT NULL,
            rationale TEXT,
            risk INTEGER NOT NULL,
            expected TEXT,
            tags TEXT,
            expires_at REAL
        );
        CREATE TABLE IF NOT EXISTS story_companions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            archetype TEXT,
            mood TEXT,
            stats TEXT,
            acquired_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS story_buffs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT NOT NULL,
            kind TEXT,
            magnitude INTEGER,
            expires_at REAL,
            meta TEXT
        );
        CREATE TABLE IF NOT EXISTS story_skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            level INTEGER NOT NULL,
            xp INTEGER NOT NULL,
            category TEXT,
            updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS idle_quests (
            id INTEGER PRIMARY KEY,
            goal TEXT NOT NULL,
            required INTEGER NOT NULL,
            progress INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL,
            tags TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        """
    )
    db_conn.commit()
    # seed minimal story companions / skills if empty (replicated subset)
    try:
        cur = db_conn.execute("SELECT COUNT(*) FROM story_companions")
        if cur.fetchone()[0] == 0:
            now = time.time()
            seeds = [
                ("Mentor", "weise", "ruhig", {"bonus_wissen":5,"empatie":3}),
                ("Späher", "wendig", "wachsam", {"sicht":7,"tempo":2}),
            ]
            for name, archetype, mood, stats in seeds:
                db_conn.execute(
                    "INSERT INTO story_companions(name,archetype,mood,stats,acquired_at) VALUES (?,?,?,?,?)",
                    (name, archetype, mood, json.dumps(stats), now)
                )
        cur = db_conn.execute("SELECT COUNT(*) FROM story_skills")
        if cur.fetchone()[0] == 0:
            for name,lvl,xp,cat in [("fokus",1,0,"mental"),("reflexion",1,0,"meta"),("ideenfindung",1,0,"kreativ")]:
                db_conn.execute(
                    "INSERT INTO story_skills(name,level,xp,category,updated_at) VALUES (?,?,?,?,?)",
                    (name,lvl,xp,cat,time.time())
                )
        db_conn.commit()
    except Exception:
        pass

__all__ = ["init_db","emit_event","get_event_queue","get_db","RISK_PATTERNS","get_last_activity","db_conn"]
