from __future__ import annotations

import os
import time
import asyncio
import json
import random
import sqlite3
from functools import wraps
from typing import Any, AsyncGenerator, Dict, Callable, Awaitable, TypeVar, Coroutine

from fastapi import FastAPI, Depends, HTTPException, status, Request, Header
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, CollectorRegistry, generate_latest
import yaml

APP_START = time.time()
SEED = int(os.getenv("DETERMINISTIC_SEED", "42"))
random.seed(SEED)
API_TOKEN = os.getenv("API_TOKEN", "changeme-dev-token")
KILL_SWITCH = os.getenv("KILL_SWITCH", "off")
DB_PATH = os.getenv("DB_PATH", "agent.db")
POLICY_DIR = os.getenv("POLICY_DIR", "policies")
AGENT_WHITELIST_DIRS = [d.strip() for d in os.getenv("AGENT_WHITELIST_DIRS", "api,plans").split(",") if d.strip()]

app = FastAPI(title="Life-Agent Dev Environment", version="0.1.0")

STATE: Dict[str, Any] = {"agents": {}, "meta": {"version": app.version, "start": APP_START}}
EVENT_QUEUE: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
POLICIES: dict[str, Any] = {"loaded_at": time.time(), "rules": []}
DB_CONN: sqlite3.Connection | None = None

REGISTRY = CollectorRegistry()
METR_REQ = Counter("api_requests_total", "Total API requests", ["endpoint"], registry=REGISTRY)
METR_ERR = Counter("api_errors_total", "Total API errors", ["endpoint"], registry=REGISTRY)
METR_LAT = Histogram("action_latency_ms", "Latency of observed actions (ms)", ["action_type"], registry=REGISTRY)

# ---------------- Models ------------------
class ActRequest(BaseModel):
    agent_id: str
    action: str
    payload: dict[str, Any] | None = None

class TurnRequest(BaseModel):
    agent_id: str
    input: str

class PlanRequest(BaseModel):
    description: str
    target_files: list[str] | None = None

# ---------------- Database ----------------
def init_db() -> None:
    global DB_CONN
    first = not os.path.exists(DB_PATH)
    DB_CONN = sqlite3.connect(DB_PATH, check_same_thread=False)
    DB_CONN.execute("PRAGMA journal_mode=WAL;")
    if first:
        DB_CONN.executescript(
            """
            CREATE TABLE events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                kind TEXT NOT NULL,
                data TEXT NOT NULL
            );
            CREATE TABLE actions (
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
            CREATE TABLE proposals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                description TEXT NOT NULL,
                artifact_path TEXT NOT NULL,
                target_files TEXT,
                meta TEXT
            );
            """
        )
        DB_CONN.commit()

def load_policies() -> None:
    rules: list[dict[str, Any]] = []
    if os.path.isdir(POLICY_DIR):
        for fname in os.listdir(POLICY_DIR):
            if not fname.endswith(('.yml', '.yaml')):
                continue
            path = os.path.join(POLICY_DIR, fname)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    doc = yaml.safe_load(f) or {}
                for r in doc.get('rules', []):  # type: ignore[assignment]
                    rules.append(r)
            except Exception:
                continue
    POLICIES['rules'] = rules
    POLICIES['loaded_at'] = time.time()

# update startup to load policies
@app.on_event("startup")
async def on_startup() -> None:  # pragma: no cover - simple init
    init_db()
    load_policies()

# ---------------- Security ------------------
async def require_auth(authorization: str = Header("", alias="Authorization")) -> str:
    if KILL_SWITCH.lower() == "on":
        raise HTTPException(status_code=503, detail="Kill switch active")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if token != API_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    # For future multi-agent: agent id could be embedded; return token owner
    return "default-agent"

# ---------------- Observability -------------
T = TypeVar("T")

def observer(action_type: str) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Coroutine[Any, Any, T]]]:
    def decorator(func: Callable[..., Awaitable[T]]):
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:  # type: ignore[override]
            start = time.time()
            # record request count
            METR_REQ.labels(endpoint=action_type).inc()
            input_repr = ""
            if args:
                try:
                    input_repr = json.dumps(args[1].__dict__)[:500]  # skip self if method
                except Exception:
                    input_repr = "<unserializable>"
            try:
                result = await func(*args, **kwargs)
                ok = True
            except Exception as e:  # noqa: BLE001
                result = {"error": str(e)}  # type: ignore[assignment]
                ok = False
                raise
            finally:
                end = time.time()
                latency_ms = (end - start) * 1000
                METR_LAT.labels(action_type=action_type).observe(latency_ms)
                output_repr = ""
                try:
                    output_repr = json.dumps(result)[:500]
                except Exception:
                    output_repr = "<unserializable>"
                score = None  # placeholder for future quality scoring
                evt = {
                    "ts": end,
                    "kind": f"obs.{action_type}",
                    "data": {
                        "latency_ms": latency_ms,
                        "ok": ok,
                        "action_type": action_type,
                    },
                }
                await emit_event(evt["kind"], evt["data"])  # queue + DB
                if DB_CONN:
                    try:
                        DB_CONN.execute(
                            "INSERT INTO actions(ts, agent_id, action_type, latency_ms, input_size, output_size, score, meta) VALUES (?,?,?,?,?,?,?,?)",
                            (
                                end,
                                kwargs.get("agent", "default-agent"),
                                action_type,
                                latency_ms,
                                len(input_repr),
                                len(output_repr),
                                score,
                                json.dumps({}),
                            ),
                        )
                        DB_CONN.commit()
                    except Exception:
                        pass
            return result
        return wrapper
    return decorator

async def emit_event(kind: str, data: dict[str, Any]) -> None:
    record = {"ts": time.time(), "kind": kind, "data": data}
    if DB_CONN:
        try:
            DB_CONN.execute(
                "INSERT INTO events(ts, kind, data) VALUES (?,?,?)",
                (record["ts"], record["kind"], json.dumps(record["data"]))
            )
            DB_CONN.commit()
        except Exception:
            pass
    await EVENT_QUEUE.put(record)

# --------------- Endpoints ------------------
@app.post("/act")
@observer("act")
async def act(req: ActRequest, agent: str = Depends(require_auth)) -> dict[str, Any]:
    STATE["agents"].setdefault(req.agent_id, {}).setdefault("actions", []).append(req.action)
    return {"status": "ok", "echo": req.model_dump()}


@app.post("/turn")
@observer("turn")
async def turn(req: TurnRequest, agent: str = Depends(require_auth)) -> dict[str, Any]:
    STATE["agents"].setdefault(req.agent_id, {}).setdefault("turns", []).append(req.input)
    return {"status": "ok", "echo": req.model_dump(), "response": f"Processed {req.input}"}


@app.get("/state")
async def get_state(agent_id: str | None = None, agent: str = Depends(require_auth)) -> dict[str, Any]:
    # Basic BOLA guard: if agent_id specified and not existing or belongs to different owner (future), restrict
    if agent_id:
        return {"agent_id": agent_id, "state": STATE["agents"].get(agent_id, {})}
    # Without explicit agent_id provide only meta + requesting agent subset (future expansion)
    return {"meta": STATE["meta"], "agent": STATE["agents"].get(agent, {})}


@app.get("/meta")
async def get_meta(agent: str = Depends(require_auth)) -> dict[str, Any]:
    uptime = time.time() - APP_START
    return {"meta": STATE["meta"], "uptime": uptime, "policies_loaded": len(POLICIES.get("rules", []))}


@app.get("/events")
async def events_stream(request: Request, agent: str = Depends(require_auth)) -> StreamingResponse:
    async def event_gen() -> AsyncGenerator[bytes, None]:
        while True:
            if await request.is_disconnected():
                break
            try:
                evt = await asyncio.wait_for(EVENT_QUEUE.get(), timeout=1.0)
                payload = f"event: {evt['kind']}\ndata: {json.dumps(evt)}\n\n".encode()
                yield payload
            except asyncio.TimeoutError:
                # send heartbeat comment
                yield b":heartbeat\n\n"
    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    uptime = time.time() - APP_START
    # custom gauge style lines appended to generated metrics
    extra = f"app_uptime_seconds {uptime:.2f}\npolicy_rules_loaded {len(POLICIES.get('rules', []))}\n"
    data = generate_latest(REGISTRY).decode() + extra
    return PlainTextResponse(data, media_type="text/plain")


@app.post("/policy/reload")
async def policy_reload(agent: str = Depends(require_auth)) -> dict[str, Any]:
    load_policies()
    await emit_event("policy.reload", {"loaded_at": POLICIES["loaded_at"], "rules": len(POLICIES['rules'])})
    return {"status": "reloaded", "loaded_at": POLICIES["loaded_at"], "rule_count": len(POLICIES['rules'])}

# helper enforcement

def enforce_plan(target_files: list[str]) -> None:
    deny_paths = []
    for r in POLICIES.get('rules', []):
        if 'paths_deny' in r:
            deny_paths.extend(r['paths_deny'])
    for tf in target_files:
        # whitelist check
        if not any(tf.startswith(w + "/") or tf == w for w in AGENT_WHITELIST_DIRS):
            raise HTTPException(status_code=400, detail=f"Target file not in whitelist: {tf}")
        for dp in deny_paths:
            if dp and dp in tf:
                raise HTTPException(status_code=400, detail=f"Target path denied by policy: {dp}")


@app.post("/plan")
@observer("plan")
async def plan(req: PlanRequest, agent: str = Depends(require_auth)) -> dict[str, Any]:
    targets = req.target_files or []
    enforce_plan(targets)
    ts = int(time.time())
    os.makedirs("plans", exist_ok=True)
    plan_filename = f"plans/plan_{ts}.json"
    artifact = {"description": req.description, "target_files": targets, "created_at": ts, "policies": [r.get('id') for r in POLICIES.get('rules', [])]}
    with open(plan_filename, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    if DB_CONN:
        try:
            DB_CONN.execute(
                "INSERT INTO proposals(ts, description, artifact_path, target_files, meta) VALUES (?,?,?,?,?)",
                (time.time(), req.description, plan_filename, json.dumps(req.target_files or []), json.dumps({}))
            )
            DB_CONN.commit()
        except Exception:
            pass
    await emit_event("plan.created", {"file": plan_filename})
    return {"status": "created", "artifact": plan_filename, "applied_policies": artifact['policies']}
