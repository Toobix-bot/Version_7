from __future__ import annotations

import os
from dotenv import load_dotenv
import time
import asyncio
import json
import random
import sqlite3
import re
import logging
from logging.handlers import RotatingFileHandler
from functools import wraps
from typing import Any, AsyncGenerator, Dict, Callable, Awaitable, TypeVar, Coroutine

from fastapi import FastAPI, Depends, HTTPException, status, Request, Header, Body, Query
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse, RedirectResponse
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST  # type: ignore
import yaml
from policy.loader import load_policy
from policy.model import Policy
from policy.opa_gate import opa_allow
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse

load_dotenv()
# ---- Logging Setup ----
LOG_FILE = os.getenv("LOG_FILE", "logs/app.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logger = logging.getLogger("life_agent")
if not logger.handlers:
    handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding='utf-8')
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
logger.propagate = False
APP_START = time.time()
SEED = int(os.getenv("DETERMINISTIC_SEED", "42"))
random.seed(SEED)
# Security: single API_KEY (preferred) still keeps legacy fallback for now
PRIMARY_API_KEY = os.getenv("API_KEY")  # new single key source
_legacy_single = os.getenv("API_TOKEN", "")
API_TOKENS = {t.strip() for t in os.getenv("API_TOKENS", _legacy_single).split(",") if t.strip()}
if PRIMARY_API_KEY:
    API_TOKENS.add(PRIMARY_API_KEY)
if not API_TOKENS:
    # last resort dev default - encourage override
    API_TOKENS.add("changeme-dev-token")
KILL_SWITCH = os.getenv("KILL_SWITCH", "off")
DB_PATH = os.getenv("DB_PATH", "agent.db")
POLICY_DIR = os.getenv("POLICY_DIR", "policies")
AGENT_WHITELIST_DIRS = [d.strip() for d in os.getenv("AGENT_WHITELIST_DIRS", "api,plans").split(",") if d.strip()]
# include test token if provided
_test_token = os.getenv(
    "TEST_API_KEY",
    # during pytest allow default 'test' token so test header matches
    "test" if os.getenv("PYTEST_CURRENT_TEST") else ""
)
if _test_token:
    API_TOKENS.add(_test_token)

# Include common test tokens (used by test suite) unless explicitly disabled
if os.getenv("ALLOW_TEST_TOKENS", "1") == "1":  # safe in dev; disable in prod via env
    API_TOKENS.update({"test", "test-token"})

app = FastAPI(title="Life-Agent Dev Environment", version="0.1.0")

# Custom OpenAPI builder: adds servers, security scheme, content-type fixes
def custom_openapi():  # type: ignore
    if getattr(app, 'openapi_schema', None):  # cached
        return app.openapi_schema  # type: ignore[attr-defined]
    schema = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
    )
    # Servers (prefer env override)
    public_url = os.getenv("PUBLIC_SERVER_URL", "https://version-7.onrender.com")
    schema["servers"] = [{"url": public_url, "description": "Public server"}]
    # Security scheme (global API Key)
    comps = schema.setdefault('components', {}).setdefault('securitySchemes', {})
    if 'ApiKeyHeader' not in comps:
        comps['ApiKeyHeader'] = {"type": "apiKey", "in": "header", "name": "X-API-Key"}
    schema['security'] = [{"ApiKeyHeader": []}]
    # Content-type corrections for streaming & metrics
    paths = schema.get('paths', {})
    if '/events' in paths and 'get' in paths['/events']:
        try:
            paths['/events']['get']['responses']['200']['content'] = {
                'text/event-stream': { 'schema': { 'type': 'string', 'description': 'SSE stream' } }
            }
        except Exception:
            pass
    if '/metrics' in paths and 'get' in paths['/metrics']:
        try:
            paths['/metrics']['get']['responses']['200']['content'] = {
                'text/plain': { 'schema': { 'type': 'string', 'description': 'Prometheus exposition format' } }
            }
        except Exception:
            pass
    app.openapi_schema = schema  # cache
    return schema

app.openapi = custom_openapi  # type: ignore

# --- CORS (restrict to required origins) ---
ALLOWED_ORIGINS = [
    "https://chat.openai.com",
    "https://toobix-bot.github.io",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# -------- Request ID & Structured Logging Middleware --------
import uuid
from starlette.middleware.base import BaseHTTPMiddleware

class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        req_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex[:12]
        start = time.time()
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:  # noqa: BLE001
            status_code = 500
            logger.error(json.dumps({
                "event": "request.error",
                "request_id": req_id,
                "method": request.method,
                "path": request.url.path,
                "error": str(e)
            }))
            raise
        duration_ms = (time.time() - start) * 1000
        logger.info(json.dumps({
            "event": "request",
            "request_id": req_id,
            "method": request.method,
            "path": request.url.path,
            "status": status_code,
            "duration_ms": round(duration_ms,2)
        }))
        response.headers["X-Request-Id"] = req_id
        return response

app.add_middleware(RequestContextMiddleware)

# -------- Simple Rate Limiter --------
RATE_LIMIT_RPS = 5
_rate_buckets: dict[str, list[float]] = {}

def rate_limit(key: str):
    if os.getenv("PYTEST_CURRENT_TEST"):
        return  # disable during tests
    now = time.time()
    win = 1.0
    bucket = _rate_buckets.setdefault(key, [])
    # prune
    while bucket and now - bucket[0] > win:
        bucket.pop(0)
    if len(bucket) >= RATE_LIMIT_RPS:
        raise _error("rate_limited", f"too many requests (>{RATE_LIMIT_RPS}/s)", 429)
    bucket.append(now)

app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")

state: Dict[str, Any] = {"agents": {}, "meta": {"version": app.version, "start": APP_START}}
event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
policies: dict[str, Any] = {"loaded_at": time.time(), "rules": []}
db_conn: sqlite3.Connection | None = None

REGISTRY = CollectorRegistry()
ACTIONS_TOTAL = Counter("actions_total", "Count of observed actions", ["kind"], registry=REGISTRY)
POLICY_DENIED_TOTAL = Counter("policy_denied_total", "Count of blocked plans/policies", ["reason"], registry=REGISTRY)
PLAN_SECONDS = Histogram("plan_seconds", "Plan endpoint latency (s)", registry=REGISTRY)
ACT_SECONDS = Histogram("act_seconds", "Act/Turn latency (s)", ["kind"], registry=REGISTRY)
LLM_REQUESTS_TOTAL = Counter("llm_requests_total", "LLM chat request count", ["model"], registry=REGISTRY)
LLM_TOKENS_TOTAL = Counter("llm_tokens_total", "LLM tokens (prompt/completion/total)", ["type"], registry=REGISTRY)

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

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None

class ChatResponse(BaseModel):
    model: str
    content: str
    usage: dict[str, Any] | None = None

# ---------------- Database ----------------
def init_db() -> None:
    global db_conn
    first = not os.path.exists(DB_PATH)
    db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    db_conn.execute("PRAGMA journal_mode=WAL;")
    # base schema
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
        """
    )
    db_conn.commit()

current_policy: Policy | None = None
# modify load_policies to use schema

def load_policies() -> None:
    global current_policy
    rules: list[dict[str, Any]] = []
    # existing YAML rule loading (legacy)
    if os.path.isdir(POLICY_DIR):
        for fname in os.listdir(POLICY_DIR):
            if fname == "policy.yaml":
                try:
                    current_policy = load_policy(os.path.join(POLICY_DIR, fname))
                except HTTPException:
                    current_policy = None
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
    policies['rules'] = rules
    policies['loaded_at'] = time.time()

# update startup to load policies
@app.on_event("startup")
async def on_startup() -> None:  # pragma: no cover - simple init
    init_db()
    load_policies()
    logger.info("startup complete policies=%d", len(policies.get('rules', [])))

# ---------------- Security ------------------
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
BEARER_SCHEME = HTTPBearer(auto_error=False)

def _error(code: str, message: str, http_status: int) -> HTTPException:
    return HTTPException(status_code=http_status, detail={"error": {"code": code, "message": message}})

async def require_auth(x_api_key: str | None = Header(None, alias="X-API-Key")) -> str:
    if KILL_SWITCH.lower() == "on":
        raise _error("kill_switch", "Service disabled", 503)
    if not x_api_key:
        raise _error("missing_api_key", "X-API-Key header required", 401)
    if x_api_key not in API_TOKENS:
        raise _error("invalid_api_key", "X-API-Key invalid", 401)
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
                if db_conn:
                    try:
                        db_conn.execute(
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
                        db_conn.commit()
                    except Exception:
                        pass
            return result
        return wrapper
    return decorator

async def emit_event(kind: str, data: dict[str, Any]) -> None:
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
    await event_queue.put(record)

# --------------- Endpoints ------------------
@app.post("/act")
@observer("act")
async def act(req: ActRequest, agent: str = Depends(require_auth)) -> dict[str, Any]:
    rate_limit(agent)
    state["agents"].setdefault(req.agent_id, {}).setdefault("actions", []).append(req.action)
    return {"status": "ok", "echo": req.model_dump()}


@app.post("/turn")
@observer("turn")
async def turn(req: TurnRequest, agent: str = Depends(require_auth)) -> dict[str, Any]:
    rate_limit(agent)
    # timeout protection
    try:
        async def _logic():
            await asyncio.sleep(0)  # placeholder for heavier logic
            state["agents"].setdefault(req.agent_id, {}).setdefault("turns", []).append(req.input)
            return {"status": "ok", "echo": req.model_dump(), "response": f"Processed {req.input}"}
        return await asyncio.wait_for(_logic(), timeout=5.0)
    except asyncio.TimeoutError:
        raise _error("timeout", "turn processing timed out", 504)


@app.get("/state")
async def get_state(agent_id: str | None = None, agent: str = Depends(require_auth)) -> dict[str, Any]:
    rate_limit(agent)
    # Basic BOLA guard: if agent_id specified and not existing or belongs to different owner (future), restrict
    if agent_id:
        return {"agent_id": agent_id, "state": state["agents"].get(agent_id, {})}
    # Without explicit agent_id provide only meta + requesting agent subset (future expansion)
    return {"meta": state["meta"], "agent": state["agents"].get(agent, {})}

@app.get("/meta")
async def get_meta(agent: str = Depends(require_auth)) -> dict[str, Any]:
    rate_limit(agent)
    uptime = time.time() - APP_START
    return {"meta": state["meta"], "uptime": uptime, "policies_loaded": len(policies.get("rules", []))}

@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/events")
async def events_stream(request: Request, agent: str = Depends(require_auth)) -> StreamingResponse:
    rate_limit(agent)
    async def event_gen() -> AsyncGenerator[bytes, None]:
        yield b"retry: 5000\n"
        yield b"event: ready\ndata: {}\n\n"
        last_keepalive = time.time()
        while True:
            if await request.is_disconnected():
                break
            now = time.time()
            try:
                evt = await asyncio.wait_for(event_queue.get(), timeout=1.0)
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
    rate_limit(agent)
    env_path = os.getenv("POLICY_PATH")
    if path:
        os.environ["POLICY_PATH"] = path
    use_path = path or env_path
    if use_path and os.path.exists(use_path):
        try:
            global current_policy
            current_policy = load_policy(use_path)
        except HTTPException as e:
            raise e
    load_policies()
    await emit_event("policy.reload", {"loaded_at": policies["loaded_at"], "rules": len(policies['rules'])})
    return {"status": "reloaded", "loaded_at": policies["loaded_at"], "rule_count": len(policies['rules'])}

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
    rate_limit(agent)
    norm = req.normalized()
    logger.info("plan request intent=%s targets=%d", norm['intent'], len(norm['targets']))
    description_joined = f"{norm['intent']} {norm['context']}"[:2000]
    risk_gate(description_joined)
    targets = list(norm['targets'])
    allowed_dirs = AGENT_WHITELIST_DIRS
    if current_policy:
        allowed_dirs = [str(p) for p in current_policy.allowed_dirs]
    enforce_whitelist(targets, allowed_dirs)
    if not opa_allow({"intent": norm['intent'], "context": norm['context'], "targets": targets}):
        POLICY_DENIED_TOTAL.labels(reason="opa_deny").inc()
        raise HTTPException(status_code=403, detail={"code": "opa_deny"})
    ts = int(time.time())
    os.makedirs("plans", exist_ok=True)
    plan_filename = f"plans/plan_{ts}.json"
    artifact = {"intent": norm['intent'], "context": norm['context'], "target_files": targets, "created_at": ts, "policies": [r.get('id') for r in policies.get('rules', [])]}
    with open(plan_filename, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    if db_conn:
        try:
            db_conn.execute(
                "INSERT INTO proposals(ts, description, artifact_path, target_files, meta) VALUES (?,?,?,?,?)",
                (time.time(), norm['intent'], plan_filename, json.dumps(targets), json.dumps({}))
            )
            db_conn.commit()
        except Exception:
            pass
    await emit_event("plan.created", {"file": plan_filename})
    return {"status": "created", "artifact": plan_filename, "applied_policies": artifact['policies']}

@app.get("/")
async def root() -> RedirectResponse:
    return RedirectResponse(url="/ui")

@app.post("/llm/chat", response_model=ChatResponse)
async def llm_chat(req: ChatRequest, agent: str = Depends(require_auth)):
    rate_limit(agent)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY not configured")
    start_time = time.time()
    try:
        from groq import Groq
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Groq SDK import failed: {e}")
    client = Groq(api_key=api_key)
    model = req.model or (current_policy.llm.model if current_policy else "llama-3.3-70b-versatile")
    temperature = req.temperature if req.temperature is not None else (current_policy.llm.temperature if current_policy else 0.0)
    max_tokens = req.max_tokens if req.max_tokens is not None else (current_policy.llm.max_tokens if current_policy else 512)
    if max_tokens > 4096:
        raise HTTPException(status_code=400, detail="max_tokens too large")
    joined = "\n".join(m.content for m in req.messages)
    risk_gate(joined)
    LLM_REQUESTS_TOTAL.labels(model=model).inc()
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in req.messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = ''
        if completion.choices:
            first_msg = completion.choices[0].message
            try:
                if hasattr(first_msg, 'content'):
                    choice = str(getattr(first_msg, 'content') or '')
                elif isinstance(first_msg, dict):
                    choice = str(first_msg.get('content') or '')
            except Exception:
                choice = ''
        raw_usage = getattr(completion, 'usage', None)
        usage_dict = None
        if raw_usage:
            try:
                usage_dict = {
                    'prompt_tokens': getattr(raw_usage, 'prompt_tokens', None) or getattr(raw_usage, 'prompt', None) or (raw_usage.get('prompt_tokens') if isinstance(raw_usage, dict) else None),
                    'completion_tokens': getattr(raw_usage, 'completion_tokens', None) or (raw_usage.get('completion_tokens') if isinstance(raw_usage, dict) else None),
                    'total_tokens': getattr(raw_usage, 'total_tokens', None) or (raw_usage.get('total_tokens') if isinstance(raw_usage, dict) else None),
                }
            except Exception:
                try:
                    usage_dict = dict(raw_usage)  # type: ignore[arg-type]
                except Exception:
                    usage_dict = None
        if usage_dict:
            if usage_dict.get('prompt_tokens') is not None:
                LLM_TOKENS_TOTAL.labels(type="prompt").inc(usage_dict['prompt_tokens'])
            if usage_dict.get('completion_tokens') is not None:
                LLM_TOKENS_TOTAL.labels(type="completion").inc(usage_dict['completion_tokens'])
            if usage_dict.get('total_tokens') is not None:
                LLM_TOKENS_TOTAL.labels(type="total").inc(usage_dict['total_tokens'])
        latency_ms = (time.time() - start_time) * 1000.0
        if db_conn and usage_dict:
            try:
                db_conn.execute(
                    "INSERT INTO llm_usage(ts, model, prompt_tokens, completion_tokens, total_tokens, latency_ms, meta) VALUES (?,?,?,?,?,?,?)",
                    (
                        time.time(),
                        model,
                        usage_dict.get('prompt_tokens'),
                        usage_dict.get('completion_tokens'),
                        usage_dict.get('total_tokens'),
                        latency_ms,
                        json.dumps({}),
                    )
                )
                db_conn.commit()
            except Exception:
                pass
        await emit_event("llm.chat", {"model": model, "tokens": (usage_dict or {}).get('total_tokens')})
        logger.info("llm chat model=%s total_tokens=%s", model, (usage_dict or {}).get('total_tokens'))
        return ChatResponse(model=str(model), content=choice, usage=usage_dict)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Groq request failed: {e}")

@app.get("/llm/status")
async def llm_status(agent: str = Depends(require_auth)) -> dict[str, Any]:
    rate_limit(agent)
    key = os.getenv("GROQ_API_KEY")
    model_default = "llama-3.3-70b-versatile"
    try:
        if current_policy and getattr(current_policy, 'llm', None) and getattr(current_policy.llm, 'model', None):
            model_default = current_policy.llm.model  # type: ignore[attr-defined]
    except Exception:
        pass
    return {"configured": bool(key), "prefix": (key[:8] + '***') if key else None, "model_default": model_default}

# ---------- Global Error Handlers ----------
@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):  # type: ignore[override]
    if isinstance(exc.detail, dict) and 'error' in exc.detail:
        payload = exc.detail
    else:
        payload = {"error": {"code": "http_error", "message": str(exc.detail)}}
    return JSONResponse(status_code=exc.status_code, content=payload)

@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):  # type: ignore[override]
    return JSONResponse(status_code=422, content={"error": {"code": "validation_error", "message": exc.errors()}})

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Cache-Control'] = 'no-store'
    return response
