from __future__ import annotations

import os
import time
import asyncio
import json
import random
import sqlite3
import re
from functools import wraps
from typing import Any, AsyncGenerator, Dict, Callable, Awaitable, TypeVar, Coroutine

from fastapi import FastAPI, Depends, HTTPException, status, Request, Header, Body
from fastapi.responses import StreamingResponse, PlainTextResponse
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST  # type: ignore
import yaml
from policy.loader import load_policy
from policy.model import Policy
from policy.opa_gate import opa_allow

APP_START = time.time()
SEED = int(os.getenv("DETERMINISTIC_SEED", "42"))
random.seed(SEED)
API_TOKEN = os.getenv("API_TOKEN", "changeme-dev-token")
API_TOKENS = {t.strip() for t in os.getenv("API_TOKENS", API_TOKEN).split(",") if t.strip()}
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
ACTIONS_TOTAL = Counter("actions_total", "Count of observed actions", ["kind"], registry=REGISTRY)
POLICY_DENIED_TOTAL = Counter("policy_denied_total", "Count of blocked plans/policies", ["reason"], registry=REGISTRY)
PLAN_SECONDS = Histogram("plan_seconds", "Plan endpoint latency (s)", registry=REGISTRY)
ACT_SECONDS = Histogram("act_seconds", "Act/Turn latency (s)", ["kind"], registry=REGISTRY)

RISK_PATTERNS = [r"\b(os\.system|subprocess\.)", r"\.\./", r"/etc/passwd", r"\b(?:ssh|https?|ftp)://", r"(?i)api[_-]?key", r"(?i)\b(token|secret)\b"]

# ---------------- Models ------------------
class ActRequest(BaseModel):
    agent_id: str
    action: str
    payload: dict[str, Any] | None = None

class TurnRequest(BaseModel):
    agent_id: str
    input: str

class PlanRequest(BaseModel):
    # legacy fields
    description: str | None = None
    target_files: list[str] | None = None
    # new schema
    intent: str | None = None
    context: str | None = None
    target_paths: list[str] | None = None

    def normalized(self) -> dict[str, Any]:
        return {
            "intent": self.intent or (self.description or "").split("\n", 1)[0],
            "context": self.context or (self.description or ""),
            "targets": self.target_paths or self.target_files or [],
        }

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

CURRENT_POLICY: Policy | None = None
# modify load_policies to use schema

def load_policies() -> None:
    global CURRENT_POLICY
    rules: list[dict[str, Any]] = []
    # existing YAML rule loading (legacy)
    if os.path.isdir(POLICY_DIR):
        for fname in os.listdir(POLICY_DIR):
            if fname == "policy.yaml":
                try:
                    CURRENT_POLICY = load_policy(os.path.join(POLICY_DIR, fname))
                except HTTPException:
                    CURRENT_POLICY = None
            if not fname.endswith((".yml", ".yaml")):
                continue
            path = os.path.join(POLICY_DIR, fname)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    doc = yaml.safe_load(f) or {}
                for r in doc.get('rules', []):
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
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
BEARER_SCHEME = HTTPBearer(auto_error=False)

async def require_auth(authorization: str = Header("", alias="Authorization"), api_key: str | None = Depends(API_KEY_HEADER), bearer: HTTPAuthorizationCredentials | None = Depends(BEARER_SCHEME)) -> str:
    """Authorize request via API key (preferred) or fallback bearer token.
    Returns agent identifier (currently single default).
    """
    if KILL_SWITCH.lower() == "on":
        raise HTTPException(status_code=503, detail="Kill switch active")
    provided = None
    if api_key:
        provided = api_key
    elif bearer and bearer.scheme.lower() == "bearer":
        provided = bearer.credentials
    else:
        # fallback legacy header
        if authorization.startswith("Bearer "):
            provided = authorization.split(" ", 1)[1].strip()
    if not provided:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing credentials")
    if provided not in API_TOKENS:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    return "default-agent"

# ---------------- Observability -------------
T = TypeVar("T")

def observer(action_type: str) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Coroutine[Any, Any, T]]]:
    def decorator(func: Callable[..., Awaitable[T]]):
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:  # type: ignore[override]
            start = time.time()
            # metrics request count
            ACTIONS_TOTAL.labels(kind=action_type).inc()
            input_repr = ""
            if args:
                try:
                    input_repr = json.dumps(getattr(args[1], "__dict__", {}))[:500]
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
                if action_type == "plan":
                    PLAN_SECONDS.observe(end - start)
                elif action_type in ("act", "turn"):
                    ACT_SECONDS.labels(kind=action_type).observe(end - start)
                output_repr = ""
                try:
                    output_repr = json.dumps(result)[:500]
                except Exception:
                    output_repr = "<unserializable>"
                score = None
                evt = {
                    "ts": end,
                    "kind": f"obs.{action_type}",
                    "data": {
                        "latency_ms": latency_ms,
                        "ok": ok,
                        "action_type": action_type,
                    },
                }
                await emit_event(evt["kind"], evt["data"])
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
        yield b"retry: 5000\n"
        yield b"event: ready\ndata: {}\n\n"
        last_keepalive = time.time()
        while True:
            if await request.is_disconnected():
                break
            now = time.time()
            try:
                evt = await asyncio.wait_for(EVENT_QUEUE.get(), timeout=1.0)
                payload = f"event: {evt['kind']}\ndata: {json.dumps(evt)}\n\n".encode()
                yield payload
            except asyncio.TimeoutError:
                pass
            if now - last_keepalive >= 15:
                yield b":keepalive\n\n"
                last_keepalive = now
    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    data = generate_latest(REGISTRY)
    return PlainTextResponse(data, media_type=CONTENT_TYPE_LATEST)


@app.post("/policy/reload")
async def policy_reload(agent: str = Depends(require_auth), path: str | None = Body(default=None)) -> dict[str, Any]:
    env_path = os.getenv("POLICY_PATH")
    if path:
        os.environ["POLICY_PATH"] = path
    use_path = path or env_path
    if use_path and os.path.exists(use_path):
        try:
            global CURRENT_POLICY
            CURRENT_POLICY = load_policy(use_path)
        except HTTPException as e:
            raise e
    load_policies()
    await emit_event("policy.reload", {"loaded_at": POLICIES["loaded_at"], "rules": len(POLICIES['rules'])})
    return {"status": "reloaded", "loaded_at": POLICIES["loaded_at"], "rule_count": len(POLICIES['rules'])}

# helper enforcement

def risk_gate(text: str):
    for pat in RISK_PATTERNS:
        if re.search(pat, text or ""):
            POLICY_DENIED_TOTAL.labels(reason="risk_pattern").inc()
            raise HTTPException(status_code=422, detail={"code": "prompt_risky", "pattern": pat})

def enforce_whitelist(paths: list[str], allowed_dirs: list[str]):
    for p in paths or []:
        if not any(p.startswith(d.rstrip("/") + "/") or p == d for d in allowed_dirs):
            POLICY_DENIED_TOTAL.labels(reason="path_forbidden").inc()
            raise HTTPException(status_code=403, detail={"code": "path_forbidden", "path": p})


@app.post("/plan")
@observer("plan")
async def plan(req: PlanRequest, agent: str = Depends(require_auth)) -> dict[str, Any]:
    norm = req.normalized()
    description_joined = f"{norm['intent']} {norm['context']}"[:2000]
    risk_gate(description_joined)
    targets = list(norm['targets'])
    allowed_dirs = AGENT_WHITELIST_DIRS
    if CURRENT_POLICY:
        allowed_dirs = [str(p) for p in CURRENT_POLICY.allowed_dirs]
    enforce_whitelist(targets, allowed_dirs)
    if not opa_allow({"intent": norm['intent'], "context": norm['context'], "targets": targets}):
        POLICY_DENIED_TOTAL.labels(reason="opa_deny").inc()
        raise HTTPException(status_code=403, detail={"code": "opa_deny"})
    ts = int(time.time())
    os.makedirs("plans", exist_ok=True)
    plan_filename = f"plans/plan_{ts}.json"
    artifact = {"intent": norm['intent'], "context": norm['context'], "target_files": targets, "created_at": ts, "policies": [r.get('id') for r in POLICIES.get('rules', [])]}
    with open(plan_filename, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    if DB_CONN:
        try:
            DB_CONN.execute(
                "INSERT INTO proposals(ts, description, artifact_path, target_files, meta) VALUES (?,?,?,?,?)",
                (time.time(), norm['intent'], plan_filename, json.dumps(targets), json.dumps({}))
            )
            DB_CONN.commit()
        except Exception:
            pass
    await emit_event("plan.created", {"file": plan_filename})
    return {"status": "created", "artifact": plan_filename, "applied_policies": artifact['policies']}
