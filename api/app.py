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
from typing import Any, AsyncGenerator, Dict, Callable, Awaitable, TypeVar, Coroutine, cast
from .routes import wizard as wizard_routes  # Phase1: extracted wizard route
from .routes import help as help_routes
from .routes import advisor as advisor_routes
from .routes import templates as templates_routes
from .core import infra as core_infra

from fastapi import FastAPI, Depends, HTTPException, status, Request, Header, Body, Query
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse, RedirectResponse, HTMLResponse
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST  # type: ignore
from prometheus_client import Gauge  # type: ignore
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

# Early auth dependency (placed early so that later endpoint definitions referencing require_auth do not raise NameError).
# The more elaborate security section further below re-defines the same function (idempotent logic) to keep original structure.
from fastapi import Header  # already imported above, safe re-import for clarity
async def require_auth(x_api_key: str | None = Header(None, alias="X-API-Key")) -> str:  # type: ignore[override]
    if KILL_SWITCH.lower() == "on":
        raise HTTPException(status_code=503, detail={"error": {"code": "kill_switch", "message": "Service disabled"}})
    if not x_api_key:
        raise HTTPException(status_code=401, detail={"error": {"code": "missing_api_key", "message": "X-API-Key header required"}})
    if x_api_key not in API_TOKENS:
        raise HTTPException(status_code=401, detail={"error": {"code": "invalid_api_key", "message": "X-API-Key invalid"}})
    return "default-agent"

from contextlib import asynccontextmanager

@asynccontextmanager
async def _lifespan(app_: FastAPI):  # pragma: no cover - setup wiring
    # initialize DB + policies
    try:
        init_db()
    except Exception:
        pass
    try:
        load_policies()
    except Exception:
        pass
    if os.getenv("THOUGHT_STREAM_ENABLE", "1") == "1":
        try:
            asyncio.create_task(_thought_loop())
        except Exception:
            logger.warning("failed to start thought loop")
    try:
        asyncio.create_task(_idle_background_loop())
    except Exception:
        logger.warning("failed to start idle background loop")
    yield

app = FastAPI(title="Life-Agent Dev Environment", version="0.1.0", lifespan=_lifespan)

# Persona presets (added after refactor to ensure constant exists for chat)
PERSONA_PRESETS: dict[str, str] = {
    "mentor": "Du bist ein geduldiger Mentor. Antworte klar, knapp und lehrreich.",
    "auditor": "Du bist ein kritischer Auditor. Fokussiere Risiken, Lücken, Compliance.",
    "architekt": "Du bist ein Software-Architekt. Entwirf klare modulare Strukturen.",
    "erzähler": "Du bist ein Erzähler. Verwandle technische Ziele in eine kurze Story.",
    "default": "Du bist ein hilfreicher Assistent für Entwicklungsaufgaben."
}

# Custom OpenAPI builder: adds servers, security scheme, content-type fixes, and schema hardening
def custom_openapi():  # type: ignore
    if getattr(app, 'openapi_schema', None):  # cached
        return app.openapi_schema  # type: ignore[attr-defined]
    schema = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
    )
    # Force OpenAPI version (robust for some Action builders)
    schema["openapi"] = "3.0.3"
    # Servers (prefer env override)
    public_url = os.getenv("PUBLIC_SERVER_URL", "https://version-7.onrender.com")
    schema["servers"] = [{"url": public_url, "description": "Public server"}]
    # Security scheme (global API Key)
    comps = schema.setdefault('components', {}).setdefault('securitySchemes', {})
    if 'ApiKeyHeader' not in comps:
        comps['ApiKeyHeader'] = {"type": "apiKey", "in": "header", "name": "X-API-Key"}
    schema['security'] = [{"ApiKeyHeader": []}]
    # Iterate paths for request/response schema hardening
    paths = schema.get('paths', {})
    for pth, methods in list(paths.items()):
        for method, op in list(methods.items()):
            if not isinstance(op, dict):
                continue
            # 3a) Shape /policy/reload requestBody into object { path?: string }
            if pth == "/policy/reload" and "requestBody" in op:
                try:
                    content = op["requestBody"].get("content", {})
                    if "application/json" in content:
                        content["application/json"]["schema"] = {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "nullable": True}
                            }
                        }
                except Exception:
                    pass
            # 3b) Harden 200 (and other) object responses lacking properties/additionalProperties
            responses = cast(dict[str, Any], op.get("responses", {}))
            for _, resp in responses.items():
                content = cast(dict[str, Any], resp.get("content", {}) if isinstance(resp, dict) else {})
                for _, c in content.items():
                    sch = c.get("schema") if isinstance(c, dict) else None  # type: ignore[index]
                    if isinstance(sch, dict) and sch.get("type") == "object":
                        if "properties" not in sch and "additionalProperties" not in sch:
                            sch["additionalProperties"] = True
    # Content-type corrections for streaming & metrics
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
app.include_router(wizard_routes.router, dependencies=[Depends(require_auth)])
app.include_router(help_routes.router, dependencies=[Depends(require_auth)])
app.include_router(advisor_routes.router, dependencies=[Depends(require_auth)])
app.include_router(templates_routes.router, dependencies=[Depends(require_auth)])

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
try:
    # serve static documentation (openapi.yaml etc.)
    if os.path.isdir("docs"):
        app.mount("/static-docs", StaticFiles(directory="docs"), name="static-docs")
except Exception:
    pass

state: Dict[str, Any] = {"agents": {}, "meta": {"version": app.version, "start": APP_START}}
# Event queue must be bound to the active loop; tests may recreate loops. Use a mapping per loop id.
_event_queues: dict[int, asyncio.Queue[dict[str, Any]]] = {}

def _get_event_queue() -> asyncio.Queue[dict[str, Any]]:
    loop = asyncio.get_running_loop()
    q = _event_queues.get(id(loop))
    if q is None:
        q = asyncio.Queue()
        _event_queues[id(loop)] = q
    return q
policies: dict[str, Any] = {"loaded_at": time.time(), "rules": []}
# ensure db_conn has an explicit typed declaration near top (if not already)
try:
    db_conn  # type: ignore  # noqa: F821
except NameError:
    from typing import Optional as _Opt
    db_conn: 'sqlite3.Connection | None' = None  # runtime assigned in init_db

# Some test harnesses (httpx ASGITransport without lifespan) skip lifespan startup.
# Provide a lightweight lazy initializer so endpoints can still function when
# lifespan events did not run (e.g., direct import in tests).
def _ensure_db() -> None:
    global db_conn
    if db_conn is None:
        try:
            init_db()
        except Exception as e:  # pragma: no cover - defensive
            raise RuntimeError(f"database_init_failed: {e}")

REGISTRY = CollectorRegistry()
ACTIONS_TOTAL = Counter("actions_total", "Count of observed actions", ["kind"], registry=REGISTRY)
POLICY_DENIED_TOTAL = Counter("policy_denied_total", "Count of blocked plans/policies", ["reason"], registry=REGISTRY)
PLAN_SECONDS = Histogram("plan_seconds", "Plan endpoint latency (s)", registry=REGISTRY)
ACT_SECONDS = Histogram("act_seconds", "Act/Turn latency (s)", ["kind"], registry=REGISTRY)
LLM_REQUESTS_TOTAL = Counter("llm_requests_total", "LLM chat request count", ["model"], registry=REGISTRY)
LLM_TOKENS_TOTAL = Counter("llm_tokens_total", "LLM tokens (prompt/completion/total)", ["type"], registry=REGISTRY)
IDLE_TICKS_TOTAL = Counter("idle_ticks_total", "Idle game tick count", registry=REGISTRY)
ENV_INFO_TOTAL = Counter("env_info_total", "Environment info requests", registry=REGISTRY)
SUGGEST_GENERATED_TOTAL = Counter("suggestions_generated_total", "Suggestions generated", registry=REGISTRY)
SUGGEST_REVIEW_TOTAL = Counter("suggestions_review_total", "Suggestion review actions", ["action"], registry=REGISTRY)
SUGGEST_OPEN_GAUGE = Gauge("suggestions_open_total", "Open (draft/revised) suggestions", registry=REGISTRY)
THOUGHT_GENERATED_TOTAL = Counter("thought_generated_total", "Generated background thoughts", registry=REGISTRY)
THOUGHT_CATEGORY_TOTAL = Counter("thought_category_total", "Thoughts per category", ["category"], registry=REGISTRY)
QUEST_COMPLETED_TOTAL = Counter("quest_completed_total", "Completed quests", registry=REGISTRY)
STORY_EVENTS_TOTAL = Counter("story_events_total", "Story events count", ["kind"], registry=REGISTRY)
STORY_OPTIONS_OPEN = Gauge("story_options_open", "Offene Story Optionen", registry=REGISTRY)
PLAN_VARIANT_GENERATED_TOTAL = Counter(
    "plan_variant_generated_total",
    "Generated plan variants",
    ["variant_id", "risk_budget", "explore"],
    registry=REGISTRY,
)

# ---- Balance Parameter (env overrides) ----
XP_BASE = float(os.getenv("STORY_XP_BASE", "80"))
XP_EXP = float(os.getenv("STORY_XP_EXP", "1.4"))
INSP_TICK_THRESHOLD = int(os.getenv("STORY_INSP_TICK_THRESHOLD", "120"))
INSP_SOFT_CAP = int(os.getenv("STORY_INSP_SOFT_CAP", "150"))
INSP_MIN_ENERGY = int(os.getenv("STORY_INSP_MIN_ENERGY", "10"))

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

class PlanVariant(BaseModel):
    id: str
    label: str
    risk_level: str
    knobs: dict[str, Any]
    summary: str
    explanation: str | None = None
    patch_preview: str | None = None

class PlanResponse(BaseModel):
    status: str
    artifact: str | None = None
    applied_policies: list[str] | None = None
    variants: list[PlanVariant]

class PlanIdeasResponse(BaseModel):
    status: str
    ideas: list[dict[str, str]]

class HealthDevResponse(BaseModel):
    status: str
    meta_ok: bool
    policy_ok: bool
    llm_ok: bool
    auth_mode: str
    version: str | None = None

class PRFromPlanRequest(BaseModel):
    intent: str
    variant_id: str | None = None
    risk_budget: str | None = None
    branch: str | None = None
    dry_run: bool | None = False

class PRFromPlanResponse(BaseModel):
    status: str
    branch: str | None = None
    artifact: str | None = None
    variant: str | None = None
    message: str | None = None
    pr_url: str | None = None  # optional URL of created PR

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    persona: str | None = None  # NEW persona selector
    agent_id: str | None = None  # multi-agent memory scope

class ChatResponse(BaseModel):
    model: str
    content: str
    usage: dict[str, Any] | None = None

# ---- Groq JSON Mode Wrapper (deterministic structured responses) ----
class GroqJSONError(Exception):
    pass

async def call_groq_json(prompt: str, schema: dict[str, Any], timeout: float = 22.0, temperature: float = 0.0, model: str | None = None) -> dict[str, Any]:
    """Call Groq (if configured) requesting a deterministic JSON object.

    Falls kein GROQ_API_KEY gesetzt ist, wird ein Dummy-Response erzeugt, der Schema-Keys auffüllt.
    Schema ist ein einfaches Dict mit optionalen default Werten; tiefe Validierung minimal.
    """
    api_key = os.getenv("GROQ_API_KEY")
    # naive default fill function
    def _fill(s: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k,v in s.items():
            if isinstance(v, dict):
                # nested schema or default object
                if any(isinstance(x, (dict,list)) for x in v.values()):
                    out[k] = _fill(v)
                else:
                    out[k] = v if not isinstance(v, (int,float,str,bool)) else v
            else:
                # primitive default placeholder
                if isinstance(v, (int,float,str,bool)):
                    out[k] = v
                else:
                    out[k] = "" if v is None else str(v)
        return out
    if not api_key:
        return _fill(schema)
    try:
        import groq  # type: ignore
    except Exception as e:  # pragma: no cover
        raise GroqJSONError(f"groq_sdk_import_failed: {e}")
    client = groq.Groq(api_key=api_key)  # type: ignore[attr-defined]
    sys_prompt = (
        "Du gibst ausschließlich gültiges JSON zurück. Keine Erklärungen. Nur ein einzelnes JSON Objekt passend zum Schema."  # noqa: E501
    )
    user_prompt = (
        f"Schema (Keys / Struktur, Werte = Defaults oder Platzhalter):\n{json.dumps(schema, indent=2, ensure_ascii=False)}\n\n"  # noqa: E501
        f"Anforderung:\n{prompt}\n\nGib nur JSON zurück."  # noqa: E501
    )
    import asyncio as _asyncio
    async def _do_call():  # isolate for timeout
        completion = await _asyncio.get_event_loop().run_in_executor(None, lambda: client.chat.completions.create(
            model = model or os.getenv("GROQ_MODEL","llama-3.3-70b-versatile"),
            temperature = temperature,
            max_tokens = 800,
            messages = [
                {"role":"system","content": sys_prompt},
                {"role":"user","content": user_prompt},
            ],
            response_format={"type":"json_object"},  # Structured response
        ))
        return completion
    try:
        completion = await asyncio.wait_for(_do_call(), timeout=timeout)
    except Exception as e:  # noqa: BLE001
        raise GroqJSONError(f"groq_call_failed: {e}")
    try:
        content = completion.choices[0].message.content  # type: ignore[attr-defined]
    except Exception as e:  # noqa: BLE001
        raise GroqJSONError(f"groq_no_content: {e}")
    try:
        obj = json.loads(content)
    except Exception as e:  # noqa: BLE001
        raise GroqJSONError(f"groq_invalid_json: {e}")
    # minimal key presence check
    for key in schema.keys():
        if key not in obj:
            obj[key] = schema[key] if not isinstance(schema[key], dict) else _fill(schema[key])
    return obj
    persona: str | None = None  # echo persona

class EnvInfoResponse(BaseModel):
    version: str
    file_count: int
    files: list[dict[str, Any]]
    env: dict[str, Any]
    python: dict[str, Any]
    metrics: dict[str, Any]

class SuggestGenerateRequest(BaseModel):
    goal: str
    focus_paths: list[str] | None = None

class Suggestion(BaseModel):
    id: str
    created_at: float
    status: str
    goal: str
    focus_paths: list[str] | None = None
    summary: str
    rationale: str
    recommended_steps: list[str]
    potential_patches: list[dict[str, Any]]
    risk_notes: list[str]
    weaknesses: list[str] = []
    metrics_impact: dict[str, Any]
    version: str = "1"
    tags: list[str] = []  # NEW for quests
    impact: dict[str, Any] | None = None  # impact scoring (score, rationale, approved_at)

class SuggestReviewRequest(BaseModel):
    id: str
    approve: bool
    adjustments: str | None = None

class SuggestReviewResponse(BaseModel):
    suggestion: Suggestion
    revised: bool

class SuggestListItem(BaseModel):
    id: str
    status: str
    goal: str
    created_at: float
    weaknesses: list[str] | None = None

class SuggestListResponse(BaseModel):
    total: int
    open: int
    items: list[SuggestListItem]

class SuggestLLMRequest(BaseModel):
    id: str
    instruction: str | None = None

class SuggestLLMResponse(BaseModel):
    suggestion: Suggestion
    refined: bool
# Impact model
class ImpactInfo(BaseModel):
    id: str
    score: float
    rationale: str
    approved_at: float
# Thought models
class Thought(BaseModel):
    id: str
    ts: float
    text: str
    kind: str = "thought"
    meta: dict[str, Any] | None = None
    category: str | None = None
    pinned: bool | None = False
class ThoughtStreamResponse(BaseModel):
    items: list[Thought]
    total: int

# ---- Memory & Quest Models ----
class MemoryItem(BaseModel):
    id: str
    ts: float
    role: str
    content: str
    persona: str | None = None
    agent_id: str | None = None

class MemoryList(BaseModel):
    items: list[MemoryItem]

class QuestGenerateRequest(BaseModel):
    theme: str | None = None
    difficulty: str | None = None  # easy|normal|hard

class QuestItem(BaseModel):
    id: str
    goal: str
    status: str
    created_at: float
    difficulty: str | None = None

class QuestListResponse(BaseModel):
    items: list[QuestItem]

from .story_core import (
    StoryOption, StoryEvent, StoryState,
    Companion, CompanionCreate, Buff, BuffCreate, Skill, SkillCreate,
    eval_arc as _core_eval_arc,
    maybe_arc_shift as _core_maybe_arc_shift,
    get_story_state as _get_story_state,
    list_story_options as _list_story_options,
    generate_story_options as _generate_story_options,
    persist_options as _persist_options,
    refresh_story_options as _refresh_story_options,
    apply_story_option as _apply_story_option,
    story_tick as _story_tick,
    fetch_events as _story_log,
)

def _story_now() -> float:
    return time.time()

## Story logic now imported from story_core

def _list_story_options(conn: sqlite3.Connection) -> list[StoryOption]:
    cur = conn.execute("SELECT id, label, rationale, risk, expected, tags, expires_at FROM story_options ORDER BY created_at DESC")
    out: list[StoryOption] = []
    now = _story_now()
    for row in cur.fetchall():
        oid, label, rationale, risk, expected_json, tags_json, expires_at = row
        if expires_at and expires_at < now:
            continue
        expected: dict[str, int] | None
        if expected_json:
            try:
                raw = json.loads(expected_json)
                if isinstance(raw, dict):
                    expected = {str(k): int(v) for k, v in raw.items() if isinstance(v, (int, float))}
                else:
                    expected = None
            except Exception:
                expected = None
        else:
            expected = None
        tags: list[str]
        if tags_json:
            try:
                raw_t = json.loads(tags_json)
                if isinstance(raw_t, list):
                    tags = [str(t) for t in raw_t]
                else:
                    tags = []
            except Exception:
                tags = []
        else:
            tags = []
        out.append(StoryOption(id=oid, label=label, rationale=rationale, risk=risk, expected=expected, tags=tags, expires_at=expires_at))
    return out

def _generate_story_options(state: StoryState) -> list[StoryOption]:
    opts: list[StoryOption] = []
    res = state.resources
    # heuristic examples
    if res.get("energie", 0) < 40:
        opts.append(
            StoryOption(
                id=f"opt_rest_{int(_story_now())}",
                label="Meditieren und Energie sammeln",
                rationale="Niedrige Energie erkannt",
                risk=0,
                expected={"energie": +15, "inspiration": +2},
                tags=["resource:energie"],
            )
        )
    if res.get("inspiration", 0) > 10 and res.get("wissen", 0) < 50:
        opts.append(
            StoryOption(
                id=f"opt_write_{int(_story_now())}",
                label="Ideen schriftlich strukturieren",
                rationale="Inspiration in Wissen umwandeln",
                risk=1,
                expected={"inspiration": -5, "wissen": +8, "erfahrung": +5},
                tags=["convert", "resource:wissen"],
            )
        )
    # progressive XP curve (env driven)
    _lvl = state.resources.get("level", 1)
    xp_needed = int((_lvl ** XP_EXP) * XP_BASE)
    if res.get("erfahrung", 0) >= xp_needed:
        xp_cost = xp_needed
        opts.append(
            StoryOption(
                id=f"opt_level_{int(_story_now())}",
                label="Reflektion und Level-Aufstieg",
                rationale="Erfahrungsschwelle erreicht",
                risk=1,
                expected={"erfahrung": -xp_cost, "level": +1, "stabilitaet": +5, "inspiration": +3},
                tags=["levelup"],
            )
        )
    # fallback
    if not opts:
        opts.append(
            StoryOption(
                id=f"opt_explore_{int(_story_now())}",
                label="Neuen Gedankenpfad erkunden",
                rationale="Kein dringendes Bedürfnis",
                risk=1,
                expected={"inspiration": +5, "energie": -5, "erfahrung": +3},
                tags=["explore"],
            )
        )
    return opts

def _persist_options(conn: sqlite3.Connection, options: list[StoryOption]) -> None:
    now = _story_now()
    for o in options:
        conn.execute("INSERT OR REPLACE INTO story_options(id, created_at, label, rationale, risk, expected, tags, expires_at) VALUES (?,?,?,?,?,?,?,?)",
                     (o.id, now, o.label, o.rationale, o.risk, json.dumps(o.expected) if o.expected else None, json.dumps(o.tags), o.expires_at))
    conn.commit()

def _refresh_story_options(conn: sqlite3.Connection, state: StoryState) -> list[StoryOption]:
    # clear existing (simple strategy MVP)
    conn.execute("DELETE FROM story_options")
    opts = _generate_story_options(state)
    # Mentor (Gefährte) senkt Risiko aller Optionen leicht
    try:
        cur = conn.execute("SELECT name FROM story_companions")
        names = {str(r[0]).lower() for r in cur.fetchall()}
        if any(n.startswith("mentor") for n in names):
            for o in opts:
                if o.risk > 0:
                    o.risk -= 1
                if "mentor" not in o.tags:
                    o.tags.append("mentor")
    except Exception:
        pass
    _persist_options(conn, opts)
    return opts

def _apply_story_option(conn: sqlite3.Connection, state: StoryState, option_id: str) -> StoryEvent:
    cur = conn.execute("SELECT id, label, rationale, risk, expected FROM story_options WHERE id=?", (option_id,))
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail={"error": {"code": "story.option_not_found", "message": "Option nicht gefunden"}})
    _, label, rationale, risk, expected_json = row
    expected: dict[str, int] = {}
    if expected_json:
        try:
            raw = json.loads(expected_json)
            if isinstance(raw, dict):
                for k, v in raw.items():
                    if isinstance(v, (int, float)):
                        expected[str(k)] = int(v)
        except Exception:
            expected = {}
    # apply delta
    new_resources = dict(state.resources)
    deltas: dict[str,int] = {}
    # Mindest-Erfahrungsgewinn skaliert nach Risiko, falls Option keinen XP Effekt hat
    if "erfahrung" not in expected:
        base_xp = max(1, risk)
        expected["erfahrung"] = base_xp
    for k, delta in expected.items():
        before = new_resources.get(k, 0)
        after = before + delta
        new_resources[k] = after
        deltas[k] = delta
    epoch = int(getattr(state, 'epoch', 0)) + 1
    mood = state.mood
    # simple mood tweak
    if deltas.get("energie",0) > 0:
        mood = "calm"
    if deltas.get("inspiration",0) > 0:
        mood = "curious"
    # possible arc shift before commit
    new_arc = _core_eval_arc(new_resources, state.arc)
    conn.execute("UPDATE story_state SET ts=?, epoch=?, mood=?, arc=?, resources=? WHERE id=1", (_story_now(), epoch, mood, new_arc, json.dumps(new_resources)))
    ev_id = f"sev_{int(_story_now()*1000)}"
    conn.execute("INSERT INTO story_events(ts, epoch, kind, text, mood, deltas, tags, option_ref) VALUES (?,?,?,?,?,?,?,?)", (_story_now(), epoch, "action", label, mood, json.dumps(deltas), json.dumps(["action"]), option_id))
    arc_event = _core_maybe_arc_shift(conn, state.arc, new_arc, epoch, mood)
    # refresh options after action
    conn.execute("DELETE FROM story_options")
    conn.commit()
    # return primary event (arc shift will also be in log)
    return StoryEvent(id=ev_id, ts=_story_now(), epoch=epoch, kind="action", text=label, mood=mood, deltas=deltas, tags=["action"], option_ref=option_id)

def _story_tick(conn: sqlite3.Connection) -> StoryEvent:
    state = _get_story_state(conn)
    # passive drift
    resources = dict(state.resources)
    resources["energie"] = max(0, resources.get("energie",0) - 1)
    # Passive Inspiration nur wenn unter Schwelle und genug Energie
    if resources.get("inspiration",0) < INSP_TICK_THRESHOLD and resources.get("energie",0) > INSP_MIN_ENERGY:
        resources["inspiration"] = resources.get("inspiration",0) + 1
    # Soft Cap
    if resources.get("inspiration",0) > INSP_SOFT_CAP:
        resources["inspiration"] = INSP_SOFT_CAP
    epoch = state.epoch + 1
    mood = state.mood
    if resources.get("energie",0) < 30:
        mood = "strained"
    new_arc = _core_eval_arc(resources, state.arc)
    conn.execute("UPDATE story_state SET ts=?, epoch=?, mood=?, arc=?, resources=? WHERE id=1", (_story_now(), epoch, mood, new_arc, json.dumps(resources)))
    ev_id = f"sev_{int(_story_now()*1000)}"
    text = "Zeit vergeht. Eine stille Verschiebung im inneren Raum."
    deltas_tick = {"energie": -1}
    if resources.get("inspiration",0) != state.resources.get("inspiration",0):
        # inspiration actually increased
        inc = resources.get("inspiration",0) - state.resources.get("inspiration",0)
        if inc>0:
            deltas_tick["inspiration"] = inc
    conn.execute(
        "INSERT INTO story_events(ts, epoch, kind, text, mood, deltas, tags, option_ref) VALUES (?,?,?,?,?,?,?,?)",
        (_story_now(), epoch, "tick", text, mood, json.dumps(deltas_tick), json.dumps(["tick"]), None),
    )
    _core_maybe_arc_shift(conn, state.arc, new_arc, epoch, mood)
    # regenerate options occasionally
    _refresh_story_options(conn, StoryState(epoch=epoch, mood=mood, arc=state.arc, resources=resources, options=[]))
    conn.commit()
    return StoryEvent(id=ev_id, ts=_story_now(), epoch=epoch, kind="tick", text=text, mood=mood, deltas=deltas_tick, tags=["tick"], option_ref=None)

# ---------------- Story LLM Support -----------------
async def _story_llm_generate(prompt: str, max_tokens: int = 120) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return prompt.split("\n\n")[-1][:max_tokens]  # fallback simple
    try:
        from groq import Groq  # type: ignore
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=(current_policy.llm.model if (current_policy and getattr(current_policy, 'llm', None) and getattr(current_policy.llm,'model',None)) else "llama-3.3-70b-versatile"),
            messages=[{"role":"system","content":"Du bist ein knapper literarischer Erzähler auf Deutsch."},{"role":"user","content": prompt}],
            temperature=0.6,
            max_tokens=max_tokens,
        )
        txt = resp.choices[0].message.content.strip()
        return txt
    except Exception:
        return prompt.split("\n\n")[-1][:max_tokens]

def _record_story_event(ev: StoryEvent) -> None:  # type: ignore[override]
    try:
        STORY_EVENTS_TOTAL.labels(ev.kind).inc()
        if db_conn:
            try:
                cur = db_conn.execute("SELECT COUNT(*) FROM story_options")
                cnt = cur.fetchone()[0]
                STORY_OPTIONS_OPEN.set(int(cnt))
            except Exception:
                pass
    except Exception:
        pass

# ---------------- Story API Endpoints -----------------

@app.get("/story/state", response_model=StoryState)
async def story_get_state(agent: str = Depends(require_auth)) -> StoryState:  # type: ignore[override]
    rate_limit(agent)
    _ensure_db()
    # Aufräumen abgelaufener Buffs vor Zustandsabruf
    try:
        now = _story_now()
        db_conn.execute("DELETE FROM story_buffs WHERE expires_at IS NOT NULL AND expires_at < ?", (now,))  # type: ignore[arg-type]
        db_conn.commit()
    except Exception:
        pass
    st = _get_story_state(db_conn)  # type: ignore[arg-type]
    return st

@app.get("/story/export")
async def story_export(limit: int = 100, agent: str = Depends(require_auth)) -> dict[str, Any]:  # type: ignore[override]
    rate_limit(agent)
    _ensure_db()
    st = _get_story_state(db_conn)  # type: ignore[arg-type]
    cur = db_conn.execute("SELECT id, ts, epoch, kind, text, mood, deltas, tags, option_ref FROM story_events ORDER BY id DESC LIMIT ?", (limit,))  # type: ignore[arg-type]
    evs: list[dict[str, Any]] = []
    for row in cur.fetchall():
        rid, ts, epoch, kind, text, mood, deltas_json, tags_json, option_ref = row
        try: deltas = json.loads(deltas_json) if deltas_json else None
        except Exception: deltas = None
        try: tags = json.loads(tags_json) if tags_json else []
        except Exception: tags = []
        evs.append({"id": rid, "ts": ts, "epoch": epoch, "kind": kind, "text": text, "mood": mood, "deltas": deltas, "tags": tags, "option_ref": option_ref})
    return {"state": st.model_dump(), "events": list(reversed(evs)), "exported_at": time.time()}

@app.post("/story/reset")
async def story_reset(agent: str = Depends(require_auth)) -> dict[str, Any]:  # type: ignore[override]
    rate_limit(agent)
    _ensure_db()
    base_resources = {"energie": 80, "inspiration": 20, "wissen": 0, "erfahrung": 0, "stabilitaet": 50, "level": 1}
    db_conn.execute("UPDATE story_state SET ts=?, epoch=?, mood=?, arc=?, resources=? WHERE id=1", (_story_now(), 0, "neutral", "beginn", json.dumps(base_resources)))  # type: ignore[arg-type]
    try:
        db_conn.execute("DELETE FROM story_events")
        db_conn.execute("DELETE FROM story_options")
    except Exception:
        pass
    st = _get_story_state(db_conn)  # type: ignore[arg-type]
    _refresh_story_options(db_conn, st)
    db_conn.commit()
    await emit_event("story.reset", {})
    return {"status": "reset", "state": st.model_dump()}

@app.get("/story/log", response_model=list[StoryEvent])
async def story_log(limit: int = 50, agent: str = Depends(require_auth)) -> list[StoryEvent]:  # type: ignore[override]
    rate_limit(agent)
    cur = db_conn.execute("SELECT id, ts, epoch, kind, text, mood, deltas, tags, option_ref FROM story_events ORDER BY id DESC LIMIT ?", (limit,))  # type: ignore[arg-type]
    out: list[StoryEvent] = []
    for row in cur.fetchall():
        rid, ts, epoch, kind, text, mood, deltas_json, tags_json, option_ref = row
        deltas: dict[str, int] | None
        if deltas_json:
            try:
                raw_d = json.loads(deltas_json)
                if isinstance(raw_d, dict):
                    deltas = {str(k): int(v) for k, v in raw_d.items() if isinstance(v, (int, float))}
                else:
                    deltas = None
            except Exception:
                deltas = None
        else:
            deltas = None
        tags: list[str]
        if tags_json:
            try:
                raw_t = json.loads(tags_json)
                if isinstance(raw_t, list):
                    tags = [str(t) for t in raw_t]
                else:
                    tags = []
            except Exception:
                tags = []
        else:
            tags = []
        out.append(StoryEvent(id=str(rid), ts=ts, epoch=epoch, kind=kind, text=text, mood=mood, deltas=deltas, tags=tags, option_ref=option_ref))
    return list(reversed(out))

@app.get("/story/options", response_model=list[StoryOption])
async def story_options(agent: str = Depends(require_auth)) -> list[StoryOption]:  # type: ignore[override]
    rate_limit(agent)
    st = _get_story_state(db_conn)  # type: ignore[arg-type]
    return st.options

class StoryActionRequest(BaseModel):
    option_id: str | None = None
    free_text: str | None = None

@app.post("/story/action", response_model=StoryEvent)
async def story_action(req: StoryActionRequest, agent: str = Depends(require_auth)) -> StoryEvent:  # type: ignore[override]
    rate_limit(agent)
    st = _get_story_state(db_conn)  # type: ignore[arg-type]
    if req.option_id:
        ev = _apply_story_option(db_conn, st, req.option_id)  # type: ignore[arg-type]
    else:
        # free text becomes a minor inspiration action
        label = req.free_text.strip() if req.free_text else "Freies Nachdenken"
        ev = _apply_story_option(db_conn, st, _persist_free_action(label))  # pseudo id from helper will raise for now
        # Above placeholder; for MVP we simply create a synthetic event
    # enrich text with LLM
    prompt = f"Aktueller Zustand: Ressourcen={st.resources}\nAktion: {ev.text}\nFormuliere einen kurzen erzählerischen Satz im Präteritum (<=25 Wörter)."
    ev.text = await _story_llm_generate(prompt, max_tokens=60)
    _record_story_event(ev)
    await emit_event("story.event", ev.model_dump())
    await emit_event("story.state", {"epoch": ev.epoch})
    return ev

def _persist_free_action(label: str) -> str:
    # create transient option to reuse apply logic with neutral deltas
    oid = f"opt_free_{int(_story_now())}"
    opt = StoryOption(id=oid, label=label, rationale="Freitext Aktion", risk=1, expected={"inspiration": +1, "energie": -1})
    _persist_options(db_conn, [opt])  # type: ignore[arg-type]
    return oid

@app.post("/story/advance", response_model=StoryEvent)
async def story_advance(agent: str = Depends(require_auth)) -> StoryEvent:  # type: ignore[override]
    rate_limit(agent)
    ev = _story_tick(db_conn)  # type: ignore[arg-type]
    # decorate tick text via LLM
    st = _get_story_state(db_conn)  # type: ignore[arg-type]
    prompt = f"Ressourcen: {st.resources}\nEreignis: {ev.text}\nSchreibe einen knappen poetischen Tick-Satz auf Deutsch (<=18 Wörter)."
    ev.text = await _story_llm_generate(prompt, max_tokens=50)
    _record_story_event(ev)
    await emit_event("story.event", ev.model_dump())
    await emit_event("story.state", {"epoch": ev.epoch})
    return ev

@app.post("/story/options/regen", response_model=list[StoryOption])
async def story_options_regen(agent: str = Depends(require_auth)) -> list[StoryOption]:  # type: ignore[override]
    rate_limit(agent)
    st = _get_story_state(db_conn)  # type: ignore[arg-type]
    opts = _refresh_story_options(db_conn, st)  # type: ignore[arg-type]
    STORY_OPTIONS_OPEN.set(len(opts))
    await emit_event("story.state", {"options": len(opts)})
    return opts

# ---------------- Meta Resource Endpoints ----------------

@app.get("/story/meta/companions", response_model=list[Companion])
async def story_meta_companions(agent: str = Depends(require_auth)) -> list[Companion]:  # type: ignore[override]
    rate_limit(agent)
    rows = db_conn.execute("SELECT id,name,archetype,mood,stats,acquired_at FROM story_companions ORDER BY id ASC").fetchall()  # type: ignore[arg-type]
    out: list[Companion] = []
    for r in rows:
        cid,name,arch,mood,stats_json,acq = r
        try: stats = json.loads(stats_json) if stats_json else {}
        except Exception: stats = {}
        out.append(Companion(id=cid,name=name,archetype=arch,mood=mood,stats=stats,acquired_at=acq))
    return out

@app.post("/story/meta/companions", response_model=Companion, status_code=201)
async def story_meta_companions_create(req: CompanionCreate, agent: str = Depends(require_auth)) -> Companion:  # type: ignore[override]
    rate_limit(agent)
    now = time.time()
    stats_json = json.dumps(req.stats or {})
    cur = db_conn.execute("INSERT INTO story_companions(name,archetype,mood,stats,acquired_at) VALUES (?,?,?,?,?)", (req.name, req.archetype, req.mood, stats_json, now))  # type: ignore[arg-type]
    db_conn.commit()
    cid = cur.lastrowid
    comp = Companion(id=cid, name=req.name, archetype=req.archetype, mood=req.mood, stats=req.stats or {}, acquired_at=now)
    await emit_event("story.meta.companion.add", comp.model_dump())
    await emit_event("story.state", {"meta":"companions"})
    return comp

@app.get("/story/meta/buffs", response_model=list[Buff])
async def story_meta_buffs(agent: str = Depends(require_auth)) -> list[Buff]:  # type: ignore[override]
    rate_limit(agent)
    rows = db_conn.execute("SELECT id,label,kind,magnitude,expires_at,meta FROM story_buffs ORDER BY id ASC").fetchall()  # type: ignore[arg-type]
    out: list[Buff] = []
    for r in rows:
        bid,label,kind,mag,exp,meta_json = r
        try: meta = json.loads(meta_json) if meta_json else {}
        except Exception: meta = {}
        out.append(Buff(id=bid,label=label,kind=kind,magnitude=mag,expires_at=exp,meta=meta))
    return out

@app.post("/story/meta/buffs", response_model=Buff, status_code=201)
async def story_meta_buffs_create(req: BuffCreate, agent: str = Depends(require_auth)) -> Buff:  # type: ignore[override]
    rate_limit(agent)
    cur = db_conn.execute("INSERT INTO story_buffs(label,kind,magnitude,expires_at,meta) VALUES (?,?,?,?,?)", (req.label, req.kind, req.magnitude, req.expires_at, json.dumps(req.meta or {})))  # type: ignore[arg-type]
    db_conn.commit()
    bid = cur.lastrowid
    buff = Buff(id=bid,label=req.label,kind=req.kind,magnitude=req.magnitude,expires_at=req.expires_at,meta=req.meta or {})
    await emit_event("story.meta.buff.add", buff.model_dump())
    await emit_event("story.state", {"meta":"buffs"})
    return buff

@app.get("/story/meta/skills", response_model=list[Skill])
async def story_meta_skills(agent: str = Depends(require_auth)) -> list[Skill]:  # type: ignore[override]
    rate_limit(agent)
    rows = db_conn.execute("SELECT id,name,level,xp,category,updated_at FROM story_skills ORDER BY id ASC").fetchall()  # type: ignore[arg-type]
    out: list[Skill] = []
    for r in rows:
        sid,name,lvl,xp,cat,upd = r
        out.append(Skill(id=sid,name=name,level=lvl,xp=xp,category=cat,updated_at=upd))
    return out

@app.post("/story/meta/skills", response_model=Skill, status_code=201)
async def story_meta_skills_create(req: SkillCreate, agent: str = Depends(require_auth)) -> Skill:  # type: ignore[override]
    rate_limit(agent)
    now = time.time()
    level = req.level if (req.level and req.level>0) else 1
    xp = req.xp if req.xp is not None and req.xp>=0 else 0
    cur = db_conn.execute("INSERT INTO story_skills(name,level,xp,category,updated_at) VALUES (?,?,?,?,?)", (req.name, level, xp, req.category, now))  # type: ignore[arg-type]
    db_conn.commit()
    sid = cur.lastrowid
    skill = Skill(id=sid,name=req.name,level=level,xp=xp,category=req.category,updated_at=now)
    await emit_event("story.meta.skill.add", skill.model_dump())
    await emit_event("story.state", {"meta":"skills"})
    return skill

# ---------------- Database ----------------
def init_db() -> None:
    global db_conn
    core_infra.init_db()
    db_conn = core_infra.get_db()

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
## (startup handler removed in favor of lifespan context)

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
    await _get_event_queue().put(record)
    # mark activity (exclude thought noise to allow idle auto ticks)
    global _last_activity_ts
    if not kind.startswith("thought."):
        _last_activity_ts = time.time()

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
async def events_stream(request: Request, agent: str = Depends(require_auth), kinds: str | None = Query(default=None)) -> StreamingResponse:  # type: ignore[override]
    rate_limit(agent)
    heartbeat_interval = float(os.getenv("SSE_HEARTBEAT_INTERVAL", "15"))
    poll_timeout = float(os.getenv("SSE_POLL_TIMEOUT", "1"))
    allowed: set[str] | None = None
    if kinds:
        try:
            allowed = {k.strip() for k in kinds.split(',') if k.strip()}
            if not allowed:
                allowed = None
        except Exception:
            allowed = None
    async def event_gen() -> AsyncGenerator[bytes, None]:
        # Initial directives + ready event for clients/tests
        yield b"retry: 5000\n"
        yield b"event: ready\ndata: {}\n\n"
        # Test shortcut: allow immediate termination (no infinite stream) using ?test=1
        if request.query_params.get("test") == "1" or os.getenv("SSE_TEST_MODE") == "1":
            return
        last_keepalive = time.time()
        while True:
            if await request.is_disconnected():
                break
            now = time.time()
            try:
                evt = await asyncio.wait_for(_get_event_queue().get(), timeout=poll_timeout)
                if allowed is None or evt['kind'] in allowed:
                    payload = f"event: {evt['kind']}\ndata: {json.dumps(evt)}\n\n".encode()
                    yield payload
            except asyncio.TimeoutError:
                pass
            if now - last_keepalive >= heartbeat_interval:
                # unified short heartbeat marker
                yield b":-hb\n\n"
                last_keepalive = now
    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    data = generate_latest(REGISTRY)
    return PlainTextResponse(data, media_type=CONTENT_TYPE_LATEST)

# ---------- Minimal Story UI (inline) ----------
@app.get("/story/ui")
async def story_ui() -> HTMLResponse:  # type: ignore[override]
        html = """<!DOCTYPE html><html lang=de><head><meta charset=utf-8><title>Story</title>
<style>
body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:16px;background:#0f1115;color:#f2f2f2}
section{margin-bottom:18px;padding:12px;background:#1b1f27;border:1px solid #2c333f;border-radius:8px}
h2{margin:4px 0 12px;font-size:18px}
button{background:#2563eb;color:#fff;border:0;padding:6px 12px;margin:4px 4px 4px 0;border-radius:4px;cursor:pointer;font-size:13px}
button.secondary{background:#374151}
code{font-size:12px;background:#111722;padding:2px 4px;border-radius:4px;white-space:pre-wrap;word-break:break-word}
.opts button{display:block;width:100%;text-align:left;position:relative}
.opts button{white-space:normal;word-break:break-word;overflow-wrap:anywhere}
.opts button.levelup{border:1px solid #f59e0b;background:#92400e}
.opts button .risk{position:absolute;right:6px;top:6px;font-size:10px;padding:2px 5px;border-radius:10px;background:#374151;color:#fff;opacity:.85}
.opts button .risk.r0{background:#059669}
.opts button .risk.r1{background:#2563eb}
.opts button .risk.r2{background:#d97706}
.opts button .risk.r3{background:#dc2626}
.badge{display:inline-block;margin-left:6px;padding:1px 4px;font-size:10px;border-radius:4px;background:#1f2937;color:#fff}
.badge.level{background:#f59e0b;color:#000}
.badge.insp{background:#ec4899}
.log-item{margin:0 0 4px;line-height:1.25;word-break:break-word;overflow-wrap:anywhere}
.kind-action{color:#10b981}.kind-tick{color:#9ca3af}.kind-arc_shift{color:#f59e0b}
.flex{display:flex;gap:16px;flex-wrap:wrap}
.col{flex:1 1 300px;min-width:280px}
input[type=text]{width:100%;padding:6px;background:#111722;color:#fff;border:1px solid #2c333f;border-radius:4px}
.res-list{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:8px;margin-top:8px}
.res{background:#111722;padding:6px;border:1px solid #2a3140;border-radius:6px;font-size:11px}
.bar{height:6px;border-radius:3px;background:#222;margin-top:4px;overflow:hidden;position:relative}
.bar span{display:block;height:100%;background:linear-gradient(90deg,#2563eb,#38bdf8)}
.res[data-k="energie"] .bar span{background:linear-gradient(90deg,#f59e0b,#fbbf24)}
.res[data-k="wissen"] .bar span{background:linear-gradient(90deg,#6366f1,#8b5cf6)}
.res[data-k="inspiration"] .bar span{background:linear-gradient(90deg,#ec4899,#f472b6)}
.res[data-k="ruf"] .bar span{background:linear-gradient(90deg,#0ea5e9,#06b6d4)}
.res[data-k="stabilitaet"] .bar span{background:linear-gradient(90deg,#10b981,#34d399)}
.res[data-k="erfahrung"] .bar span{background:linear-gradient(90deg,#9333ea,#a855f7)}
.res[data-k="level"] .bar span{background:linear-gradient(90deg,#ef4444,#f87171)}
/* safer text wrapping across panels incl. help overlay */
#stateBox,#logBox,#companionsBox,#buffsBox,#skillsBox,#helpOverlay,#helpOverlay *{word-break:break-word;overflow-wrap:anywhere}
</style></head><body>
<h1>Story <small id=sseStatus style='font-size:12px;color:#888'>[SSE: init]</small> <small id=keyStatus style='font-size:12px;color:#888;margin-left:8px'>[Key: ?]</small></h1>
<div id="helpOverlay" style="position:fixed;right:16px;bottom:16px;z-index:9999;background:#111827;border:1px solid #374151;padding:12px 14px;width:300px;max-height:60vh;overflow:auto;border-radius:10px;box-shadow:0 4px 18px -2px #0009;font-size:12px;display:none">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
        <strong style="font-size:13px">Hilfe & Kontext</strong>
        <button id="helpClose" style="background:#1f2937;border:0;color:#9ca3af;font-size:16px;line-height:16px;cursor:pointer">×</button>
    </div>
    <div id="helpTabs" style="display:flex;gap:6px;margin-bottom:6px">
        <button data-tab="tips" class="htab active" style="flex:1;background:#2563eb;border:0;padding:4px 6px;border-radius:4px;color:#fff;cursor:pointer;font-size:11px">Tipps</button>
        <button data-tab="gloss" class="htab" style="flex:1;background:#374151;border:0;padding:4px 6px;border-radius:4px;color:#fff;cursor:pointer;font-size:11px">Glossar</button>
    </div>
    <div id="helpContent"></div>
    <div style="margin-top:8px;text-align:right">
        <button id="helpPin" style="background:#374151;border:0;color:#d1d5db;padding:3px 8px;font-size:11px;border-radius:4px;cursor:pointer">Pin</button>
    </div>
</div>
<button id="helpToggle" style="position:fixed;right:16px;bottom:16px;z-index:9998;background:#2563eb;border:none;color:#fff;padding:10px 14px;border-radius:50%;font-size:18px;cursor:pointer;box-shadow:0 4px 12px -2px #000a">?</button>
<div class=flex>
 <div class=col>
    <section id=stateSec><h2>Zustand</h2><div id=stateBox>lade...</div>
        <div id=metaWrap style='margin-top:12px'>
            <details open><summary style='cursor:pointer'>Gefährten</summary><div id=companionsBox style='font-size:12px;margin-top:4px'></div></details>
            <details open><summary style='cursor:pointer'>Buffs</summary><div id=buffsBox style='font-size:12px;margin-top:4px'></div></details>
            <details open><summary style='cursor:pointer'>Skills</summary><div id=skillsBox style='font-size:12px;margin-top:4px'></div></details>
        </div>
    </section>
    <section><h2>Aktion</h2>
        <div class=opts id=optsBox></div>
        <input id=freeText placeholder="Freitext Aktion" />
        <button id=btnFree>Freitext senden</button>
        <button class=secondary id=btnAdvance>Zeit voranschreiten</button>
        <button class=secondary id=btnRegen>Optionen neu</button>
    </section>
 </div>
 <div class=col>
        <section><h2>Log</h2>
            <input id=logFilter placeholder="Filter (Regex / Text)" style="margin-bottom:6px;width:100%;padding:4px 6px;background:#111722;color:#eee;border:1px solid #2c333f;border-radius:4px;font-size:12px" />
            <div id=logBox style="max-height:520px;overflow:auto;font-size:12px"></div>
        </section>
 </div>
</div>
<script>
const apiKey = localStorage.getItem('api_key') || prompt('API Key?'); if(apiKey) localStorage.setItem('api_key', apiKey);
const H = {'X-API-Key': apiKey};
function j(el, html){document.getElementById(el).innerHTML=html}
function fetchJSON(p,opt={}){opt.headers={...(opt.headers||{}),...H,'Content-Type':'application/json'};return fetch(p,opt).then(r=>{if(!r.ok) throw new Error(r.status); return r.json()})}
function setKeyStatus(txt,color){ const el=document.getElementById('keyStatus'); if(el){ el.textContent='[Key: '+txt+']'; el.style.color=color||'#888'; } }
async function checkAuth(){ try{ const r = await fetchJSON('/health/dev'); const mode = (r && r.auth_mode) ? r.auth_mode : 'ok'; setKeyStatus(mode,'#10b981'); } catch(e){ setKeyStatus('invalid','#ef4444'); } }
function fmtResBlocks(r){const maxMap={energie:100,wissen:200,inspiration:100,ruf:100,stabilitaet:100,erfahrung:1000,level:20};return '<div class="res-list">'+Object.entries(r).map(([k,v])=>{const max=maxMap[k]||100;const pct=Math.min(100,Math.round((v/max)*100));return `<div class='res' data-k='${k}'><div><b>${k}</b> <span style='float:right'>${v}</span></div><div class='bar'><span style='width:${pct}%'></span></div></div>`}).join('')+'</div>'}
function renderCompanions(list){const box=document.getElementById('companionsBox'); if(!list||!list.length){box.textContent='(keine)';return;} box.innerHTML=list.map(c=>`<div><b>${c.name}</b> <small>${c.archetype||''}</small> – ${c.mood||''} ${c.stats?'<code>'+Object.entries(c.stats).map(([k,v])=>k+':'+v).join(', ')+'</code>':''}</div>`).join('')}
function renderBuffs(list){const now=Date.now()/1000;const box=document.getElementById('buffsBox'); if(!list||!list.length){box.textContent='(keine)';return;} box.innerHTML=list.map(b=>{const rem=b.expires_at?Math.max(0,Math.round(b.expires_at-now)):null; return `<div><b>${b.label}</b> <small>${b.kind||''}</small> ${b.magnitude!=null?('['+b.magnitude+']'):''} ${rem!=null?(' <span style=color:#f59e0b>'+rem+'s</span>'):''}</div>`}).join('')}
function renderSkills(list){const box=document.getElementById('skillsBox'); if(!list||!list.length){box.textContent='(keine)';return;} box.innerHTML=list.map(s=>`<div><b>${s.name}</b> Lv ${s.level} <small>${s.category||''}</small> <span style='opacity:.6'>xp:${s.xp}</span></div>`).join('')}
function loadState(){fetchJSON('/story/state').then(st=>{const energy=st.resources.energie||0; const energyWarn = energy<30?"<div style='color:#f59e0b;font-size:11px;margin-top:6px'>Niedrige Energie</div>":""; const arcColor = st.arc.includes('krise')?'#dc2626':(st.arc.includes('aufbau')?'#2563eb':(st.arc.includes('ruhe')?'#10b981':'#6b7280')); const arcBadge = `<span style='background:${arcColor};padding:2px 8px;border-radius:12px;font-size:11px;margin-left:8px'>${st.arc}</span>`; j('stateBox',`Arc: <b>${st.arc}</b>${arcBadge}<br>Mood: ${st.mood}<br>Epoch: ${st.epoch}<br>${fmtResBlocks(st.resources)}${energyWarn}`);renderCompanions(st.companions);renderBuffs(st.buffs);renderSkills(st.skills);renderOpts(st.options)}).catch(e=>console.error(e))}
function renderOpts(opts){const box=document.getElementById('optsBox');box.innerHTML=''; if(!opts.length){box.textContent='(keine)';return} opts.forEach(o=>{const b=document.createElement('button'); b.textContent=`${o.label}`; if((o.tags||[]).includes('levelup')){b.classList.add('levelup')} const riskSpan=document.createElement('span'); riskSpan.className='risk r'+(o.risk||0); riskSpan.textContent='R'+(o.risk||0); b.appendChild(riskSpan); b.onclick=()=>act(o.id); box.appendChild(b)})}
function loadLog(){fetchJSON('/story/log?limit=80').then(events=>{const box=document.getElementById('logBox'); box.innerHTML=events.map(e=>{let badges=''; if(e.deltas){ if(e.deltas.level>0){badges+=`<span class='badge level'>Level +${e.deltas.level}</span>`;} if(e.deltas.inspiration && e.deltas.inspiration>5){badges+=`<span class='badge insp'>+${e.deltas.inspiration} Insp</span>`;} } return `<div class='log-item kind-${e.kind}'>[${e.epoch}] <span>${e.kind}</span>: ${e.text} ${badges}</div>`}).join('')}).catch(e=>{})}
function act(id){fetchJSON('/story/action',{method:'POST',body:JSON.stringify({option_id:id})}).then(e=>{loadState();prependLog(e)})}
function free(){const t=document.getElementById('freeText').value.trim(); if(!t) return; fetchJSON('/story/action',{method:'POST',body:JSON.stringify({free_text:t})}).then(e=>{document.getElementById('freeText').value=''; loadState(); prependLog(e)})}
function advance(){fetchJSON('/story/advance',{method:'POST'}).then(e=>{loadState();prependLog(e)})}
function regen(){fetchJSON('/story/options/regen',{method:'POST'}).then(_=>loadState())}
function prependLog(ev){const box=document.getElementById('logBox'); const div=document.createElement('div');div.className='log-item kind-'+ev.kind; div.innerHTML=`[${ev.epoch}] <span>${ev.kind}</span>: ${ev.text}`; box.prepend(div)}
document.getElementById('btnFree').onclick=free; document.getElementById('btnAdvance').onclick=advance; document.getElementById('btnRegen').onclick=regen;
loadState(); loadLog(); checkAuth();
// Help Overlay Logic
const helpToggle=document.getElementById('helpToggle');
const helpOverlay=document.getElementById('helpOverlay');
const helpContent=document.getElementById('helpContent');
const helpClose=document.getElementById('helpClose');
const helpPin=document.getElementById('helpPin');
let helpPinned=false; let lastEvents=[]; let glossary=[];
function renderHelpTips(){
    const latest=lastEvents.slice(-5).reverse().map(e=>`<li><code>${e.kind}</code> – ${(e.data&&e.data.intent)?e.data.intent:''}</li>`).join('')||'<li>(noch keine)</li>';
    helpContent.innerHTML=`<div><b>Letzte Events</b><ul style='margin:4px 0 8px 16px;padding:0'>${latest}</ul><b>Schnelle Aktionen</b><ul style='margin:4px 0 0 16px;padding:0'><li>Arc prüfen → Zustand Tab</li><li>Neue Optionen → "Optionen neu"</li><li>Policy Wizard testen → /policy/wizard</li></ul></div>`;
}
function renderHelpGloss(){
    if(!glossary.length){helpContent.innerHTML='<em>lade...</em>';return;}
    helpContent.innerHTML='<div style="max-height:36vh;overflow:auto">'+glossary.map(g=>`<div style='margin-bottom:6px'><b>${g.term}</b><br><span style='opacity:.8'>${g.short}</span></div>`).join('')+'</div>';
}
function setHelpTab(tab){document.querySelectorAll('#helpTabs .htab').forEach(b=>{b.classList.toggle('active',b.dataset.tab===tab); if(b.classList.contains('active')){b.style.background='#2563eb'} else {b.style.background='#374151'} }); if(tab==='gloss'){renderHelpGloss()} else {renderHelpTips()}}
helpToggle.onclick=()=>{helpOverlay.style.display = helpOverlay.style.display==='none'?'block':'none'; if(helpOverlay.style.display==='block'){renderHelpTips()}};
helpClose.onclick=()=>{ if(!helpPinned){helpOverlay.style.display='none'} };
helpPin.onclick=()=>{helpPinned=!helpPinned; helpPin.textContent=helpPinned?'Unpin':'Pin'; helpPin.style.background=helpPinned?'#10b981':'#374151'}
document.getElementById('helpTabs').addEventListener('click',e=>{const t=e.target.closest('button'); if(!t)return; setHelpTab(t.dataset.tab) });
fetchJSON('/glossary').then(g=>{glossary=g||[]; if(helpOverlay.style.display==='block'){renderHelpGloss()}}).catch(()=>{});
// Capture events for overlay
function trackEvent(ev){try{lastEvents.push(ev); if(lastEvents.length>50) lastEvents=lastEvents.slice(-50); if(helpOverlay.style.display==='block' && !document.querySelector('#helpTabs .htab.active[data-tab="gloss"]')){renderHelpTips();}}catch(_){}}
// SSE
let es; let esRetry=0; const maxRetry=10; const statusEl=document.getElementById('sseStatus');
function sseSet(st, color){ if(statusEl){ statusEl.textContent='[SSE: '+st+']'; statusEl.style.color=color||'#888'; } }
function startSSE(){
    try{ if(es){ es.close(); }
        es = new EventSource('/events',{withCredentials:false});
        sseSet('verbunden','#10b981'); esRetry=0;
        es.addEventListener('open',()=>sseSet('offen','#10b981'));
        es.onmessage=()=>{};
        es.addEventListener('story.state',()=>{loadState()});
        es.addEventListener('story.event',()=>{loadLog()});
                es.addEventListener('story.event',e=>{try{trackEvent(JSON.parse(e.data))}catch(_){}});
        es.onerror=()=>{ es.close(); sseSet('getrennt','#f59e0b'); if(esRetry<maxRetry){ const t = Math.min(10000, 500 * Math.pow(1.6, esRetry)); esRetry++; setTimeout(startSSE,t);} else { sseSet('fail','#ef4444'); } };
    }catch(e){ console.warn('SSE fail',e); sseSet('fehler','#ef4444'); }
}
startSSE();
setInterval(()=>{loadState();},20000);
// Log Filter
document.getElementById('logFilter').addEventListener('input',()=>{ const v = document.getElementById('logFilter').value.trim(); const items=[...document.querySelectorAll('#logBox .log-item')]; let rx=null; try{ rx = v? new RegExp(v,'i'):null;}catch(_){rx=null;} items.forEach(it=>{ const txt=it.textContent||''; it.style.display = !v? '' : (rx? (rx.test(txt)?'':'none') : (txt.toLowerCase().includes(v.toLowerCase())?'':'none')); }); });
</script>
</body></html>"""
        return HTMLResponse(content=html)


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

@app.get("/policy/current")
async def policy_current(agent: str = Depends(require_auth)) -> dict[str, Any]:
    rate_limit(agent)
    # Provide serialized policy + legacy aggregated rules meta
    pol = None
    try:
        if current_policy:
            pol = current_policy.model_dump()
    except Exception:
        pol = None
    return {"policy": pol, "rules_count": len(policies.get('rules', [])), "loaded_at": policies.get('loaded_at')}

class PolicyApplyRequest(BaseModel):
    content: str
    dry_run: bool | None = False

class PolicyDryRunRequest(BaseModel):
    content: str

@app.post("/policy/dry-run")
async def policy_dry_run(req: PolicyDryRunRequest, agent: str = Depends(require_auth)) -> dict[str, Any]:  # type: ignore[override]
    rate_limit(agent)
    import yaml as _yaml
    try:
        data = _yaml.safe_load(req.content) or {}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": "yaml_parse", "message": str(e)}
    from policy.model import Policy as _Pol
    try:
        new_pol = _Pol.model_validate(data)
    except Exception as e:  # noqa: BLE001
        return {"status": "invalid", "error": "validation", "message": str(e)}
    return {"status": "ok", "policy": new_pol.model_dump()}

@app.post("/policy/apply")
async def policy_apply(req: PolicyApplyRequest, agent: str = Depends(require_auth)) -> dict[str, Any]:
    rate_limit(agent)
    import yaml as _yaml
    try:
        data = _yaml.safe_load(req.content) or {}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail={"code": "yaml_parse_error", "message": str(e)})
    from policy.model import Policy as _Pol
    try:
        new_pol = _Pol.model_validate(data)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=422, detail={"code": "policy_invalid", "message": str(e)})
    if req.dry_run:
        return {"status": "validated", "dry_run": True, "policy": new_pol.model_dump()}
    # Persist to policies/policy.yaml
    os.makedirs(POLICY_DIR, exist_ok=True)
    target_path = os.path.join(POLICY_DIR, "policy.yaml")
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(req.content if req.content.endswith("\n") else req.content + "\n")
    # Reload
    try:
        global current_policy
        current_policy = new_pol
    except Exception:
        pass
    policies['loaded_at'] = time.time()
    await emit_event("policy.apply", {"path": target_path, "version": getattr(new_pol, 'version', None)})
    return {"status": "applied", "path": target_path, "policy": new_pol.model_dump()}

"""Policy Wizard models & endpoint moved to services/policy_wizard.py and routes/wizard.py (Phase1)."""

# -------- Multi-IO Scaffold --------
class MultiIORequest(BaseModel):
    user_input: str | None = None
    shared_context: str | None = None
    system_guidance: str | None = None
    mode: str | None = "default"

    def combined(self) -> str:
        parts = []
        if self.system_guidance:
            parts.append(f"[system]\n{self.system_guidance.strip()}")
        if self.shared_context:
            parts.append(f"[context]\n{self.shared_context.strip()}")
        if self.user_input:
            parts.append(f"[user]\n{self.user_input.strip()}")
        return "\n\n".join(parts)

# Placeholder service for variant building reuse
def build_variants(kind: str, base: dict[str, Any], profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in profiles:
        d = {"kind": kind, **p}
        # simple summary heuristics
        depth = p.get("depth_limit")
        explore = p.get("explore")
        d["summary"] = f"Variante {p.get('id')} (Tiefe={depth}, Explore={explore})" if depth or explore else p.get("id")
        out.append(d)
    return out

TEMPLATE_DIR = os.path.join(POLICY_DIR, "templates")

def _load_template(name: str) -> dict[str, Any]:
    import yaml as _yaml
    safe = re.sub(r"[^a-zA-Z0-9_-]", "", name)
    path = os.path.join(TEMPLATE_DIR, f"{safe}.yaml")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail={"code": "template_not_found", "template": name})
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = _yaml.safe_load(f) or {}
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail={"code": "template_yaml_error", "message": str(e)})
    if not isinstance(data, dict):
        raise HTTPException(status_code=422, detail={"code": "template_invalid", "message": "root not mapping"})
    return data

def _apply_risk_profile(doc: dict[str, Any], risk: str | None, notes: list[str]):
    if not risk:
        return
    risk = risk.lower()
    if risk not in ("low","medium","high"):
        notes.append(f"ignoriert unbekanntes risk_profile '{risk}'")
        return
    # heuristic tweaks: add synthetic rule or adjust llm temperature if present
    if risk == "low":
        # enforce deterministic llm
        llm = doc.setdefault("llm", {})
        if isinstance(llm, dict):
            prev = llm.get("temperature")
            llm["temperature"] = 0.0
            # Immer vermerken (auch wenn bereits 0.0), damit Tests & Nutzer Feedback erhalten
            if prev != 0.0:
                notes.append("Risk Profile low → temperature=0.0 (angepasst)")
            else:
                notes.append("Risk Profile low bestätigt (temperature bereits 0.0)")
    elif risk == "high":
        llm = doc.setdefault("llm", {})
        if isinstance(llm, dict):
            prev = llm.get("temperature")
            llm.setdefault("temperature", 0.2)
            if prev is None:
                notes.append("Risk Profile high → default temperature=0.2 gesetzt")

def _merge_overrides(base: dict[str, Any], overrides: dict[str, Any] | None, notes: list[str]):
    if not overrides:
        return
    allow = {"allowed_dirs","rules","llm","branching","reviews","name"}
    for k, v in overrides.items():
        if k not in allow:
            notes.append(f"override feld '{k}' nicht erlaubt – ignoriert")
            continue
        base[k] = v
        notes.append(f"override angewendet: {k}")

def _compute_diff(orig: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    added: dict[str, Any] = {}
    removed: dict[str, Any] = {}
    changed: dict[str, dict[str, Any]] = {}
    for k in new.keys() - orig.keys():
        added[k] = new[k]
    for k in orig.keys() - new.keys():
        removed[k] = orig[k]
    for k in orig.keys() & new.keys():
        if orig[k] != new[k]:
            changed[k] = {"from": orig[k], "to": new[k]}
    return {"added": added, "removed": removed, "changed": changed}

"""/policy/wizard provided by router include above."""

# -------- Glossary --------

@app.get("/glossary")
async def glossary(agent: str = Depends(require_auth)) -> list[dict[str, Any]]:  # type: ignore[override]
    rate_limit(agent)
    path = os.path.join(os.getcwd(), "glossary.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []

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


@app.post("/plan", response_model=PlanResponse)
@observer("plan")
async def plan(req: PlanRequest, agent: str = Depends(require_auth), risk_budget: str | None = Query(default=None)) -> PlanResponse:  # type: ignore[override]
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

    # Heuristic variant generation (placeholder until Groq integration for variants)
    base_intent = norm['intent'] or 'plan'
    rb = (risk_budget or 'balanced').lower()
    variant_specs = [
        {"id": "v_safe", "label": "Vorsichtig", "risk_level": "vorsichtig", "temperature": 0.0, "depth_limit": 3, "risk_budget": "low", "explore": 0.1, "review_enforcement": True},
        {"id": "v_bal", "label": "Balanciert", "risk_level": "balanciert", "temperature": 0.0, "depth_limit": 5, "risk_budget": "medium", "explore": 0.25, "review_enforcement": True},
        {"id": "v_focus", "label": "Fokussiert", "risk_level": "fokussiert", "temperature": 0.1, "depth_limit": 6, "risk_budget": "medium", "explore": 0.15, "review_enforcement": True},
        {"id": "v_exp", "label": "Experimentell", "risk_level": "experimentell", "temperature": 0.35, "depth_limit": 7, "risk_budget": "high", "explore": 0.5, "review_enforcement": False},
        {"id": "v_bold", "label": "Mutig", "risk_level": "mutig", "temperature": 0.2, "depth_limit": 8, "risk_budget": "high", "explore": 0.4, "review_enforcement": False},
    ]
    # Reorder to emphasize requested risk_budget first if provided
    if rb in ("low","medium","high"):
        priority = {"low":0, "medium":1, "high":2}
        variant_specs.sort(key=lambda v: priority.get(v['risk_budget'], 99) + (0 if v['risk_budget']==rb else 10))

    variants: list[PlanVariant] = []
    for spec in variant_specs:
        summary = f"{spec['label']} Variante für '{base_intent}' (Tiefe {spec['depth_limit']}, Explore {spec['explore']})"
        explanation = (
            "Konservativer Ansatz mit minimalem Risiko." if spec['risk_budget']=="low" else
            ("Ausgewogene Änderungen mit moderatem Umfang." if spec['risk_budget']=="medium" else "Aggressivere / explorative Variante mit erweiterten Änderungen.")
        )
        patch_preview = None
        try:
            # simple illustrative patch preview (no real diff yet)
            target_preview = (targets[0] if targets else 'README.md')
            patch_preview = (
                f"--- a/{target_preview}\n+++ b/{target_preview}\n@@\n-// TODO old\n+// {spec['label']} plan update placeholder\n"
            )
        except Exception:
            pass
        variants.append(PlanVariant(
            id=spec['id'],
            label=spec['label'],
            risk_level=spec['risk_level'],
            knobs={
                "temperature": spec['temperature'],
                "risk_budget": spec['risk_budget'],
                "depth_limit": spec['depth_limit'],
                "explore": spec['explore'],
                "review_enforcement": spec['review_enforcement'],
            },
            summary=summary,
            explanation=explanation,
            patch_preview=patch_preview,
        ))
        try:
            PLAN_VARIANT_GENERATED_TOTAL.labels(
                variant_id=spec['id'],
                risk_budget=spec['risk_budget'],
                explore=str(spec['explore'])
            ).inc()
        except Exception:
            # metrics should never break endpoint
            pass

    return PlanResponse(
        status="created",
        artifact=plan_filename,
        applied_policies=artifact['policies'],
        variants=variants,
    )

@app.get("/plan/ideas", response_model=PlanIdeasResponse)
async def plan_ideas(agent: str = Depends(require_auth)) -> PlanIdeasResponse:  # type: ignore[override]
    """Einfache Liste möglicher Plan-Intents für Einsteiger.

    Statisch / heuristisch – später kann dies dynamisch (Code-Analyse, Metriken) generiert werden.
    """
    base_ideas = [
        {"intent": "Health Endpoint hinzufügen", "desc": "Neuen /health oder /health/dev Endpunkt bereitstellen."},
        {"intent": "Logging vereinheitlichen", "desc": "Bestehende print/log Stellen in strukturiertes Logging umwandeln."},
        {"intent": "Tests erweitern", "desc": "Fehlende Tests für Plan Varianten & Policy Validierung ergänzen."},
        {"intent": "README Quick Start verbessern", "desc": "Kurzanleitung + Badges + erste Schritte."},
        {"intent": "Story Metriken hinzufügen", "desc": "Prometheus Counter/Gauge für story_beats_total etc."},
        {"intent": "Policy Templates anlegen", "desc": "Vorlagen (solo_dev/team/sandbox) unter /policies hinzufügen."},
        {"intent": "SSE Heartbeat härten", "desc": "Heartbeat Format vereinheitlichen und Client-Retry verbessern."},
        {"intent": "Groq JSON Wrapper bauen", "desc": "Abstraktion für deterministische JSON Responses."},
        {"intent": "Suggestion Impact Score verfeinern", "desc": "Gewichtung nach Dateityp / Änderungstiefe."},
        {"intent": "Story Onboarding vereinfachen", "desc": "Defaults + Mini Tutorial + 3 Start-Optionen."},
    ]
    return PlanIdeasResponse(status="ok", ideas=base_ideas)

@app.get("/health/dev", response_model=HealthDevResponse)
async def health_dev(agent: str = Depends(require_auth)) -> HealthDevResponse:  # type: ignore[override]
    """Aggregierter Dev-Health: vereinfacht für UI Badges.

    meta_ok: Basis-Meta verfügbar
    policy_ok: aktuelle Policy geladen (oder nicht leer)
    llm_ok: LLM konfiguriert (GROQ_API_KEY gesetzt oder interner Flag)
    auth_mode: 'multi' wenn mehrere Tokens, sonst 'single'
    """
    meta_ok = True if state.get('meta') else False
    policy_ok = bool(policies.get('rules'))
    llm_ok = bool(os.getenv('GROQ_API_KEY'))  # simple heuristic
    auth_mode = 'multi' if len(API_TOKENS) > 1 else 'single'
    return HealthDevResponse(status="ok", meta_ok=meta_ok, policy_ok=policy_ok, llm_ok=llm_ok, auth_mode=auth_mode, version=state.get('meta',{}).get('version'))

@app.post("/dev/pr-from-plan", response_model=PRFromPlanResponse)
async def pr_from_plan(req: PRFromPlanRequest, agent: str = Depends(require_auth)) -> PRFromPlanResponse:  # type: ignore[override]
    """Create a git branch, generate a plan, persist selected variant artifact and (optionally) push.
    Requires local git repo and optionally gh CLI for PR creation (if available).
    """
    rate_limit(agent)
    if not os.path.isdir('.git'):
        return PRFromPlanResponse(status="error", message="not a git repository")
    # generate plan first (reuse plan logic via internal call)
    plan_req = PlanRequest(intent=req.intent, context=req.intent, target_paths=[])
    plan_resp = await plan(plan_req, agent, req.risk_budget)  # type: ignore[arg-type]
    variant = None
    if req.variant_id:
        for v in plan_resp.variants:
            if v.id == req.variant_id:
                variant = v
                break
    if not variant and plan_resp.variants:
        variant = plan_resp.variants[0]
    branch = req.branch or f"plan/{int(time.time())}_{(variant.id if variant else 'base')}"  # type: ignore[union-attr]
    # safety: sanitize branch name
    branch = branch.replace('..','.')[:120]
    if req.dry_run:
        return PRFromPlanResponse(status="dry-run", branch=branch, artifact=plan_resp.artifact, variant=(variant.id if variant else None))
    import subprocess
    try:
        subprocess.run(["git","checkout","-b",branch], check=True, capture_output=True)
    except Exception as e:  # noqa: BLE001
        return PRFromPlanResponse(status="error", message=f"branch create failed: {e}")
    # commit artifact (already written by plan())
    try:
        if plan_resp.artifact:
            subprocess.run(["git","add", plan_resp.artifact], check=True, capture_output=True)
            msg = f"Add plan artifact for {req.intent} ({variant.id if variant else 'base'})"
            subprocess.run(["git","commit","-m",msg], check=True, capture_output=True)
    except Exception as e:  # noqa: BLE001
        return PRFromPlanResponse(status="error", message=f"commit failed: {e}")
    # optional push
    pr_url: str | None = None
    try:
        subprocess.run(["git","push","--set-upstream","origin",branch], check=True, capture_output=True)
        # optional PR via gh (draft + labels)
        try:
            cp = subprocess.run([
                "gh","pr","create",
                "--title", f"Plan: {req.intent}",
                "--body", f"Automatisch erzeugter Plan PR für {req.intent}",
                "--label","plan",
                "--label","ready-for-copilot",
                "--draft"
            ], check=True, capture_output=True)
            try:
                out_txt = cp.stdout.decode(errors='ignore').strip()
                import re as _re
                m = _re.findall(r'https?://\S+', out_txt)
                if m:
                    pr_url = m[-1]
            except Exception:
                pass
        except Exception:
            pass
    except Exception as e:  # noqa: BLE001
        return PRFromPlanResponse(status="error", branch=branch, artifact=plan_resp.artifact, variant=(variant.id if variant else None), message=f"push failed: {e}")
    await emit_event("plan.pr", {"branch": branch, "variant": (variant.id if variant else None)})
    return PRFromPlanResponse(status="created", branch=branch, artifact=plan_resp.artifact, variant=(variant.id if variant else None), pr_url=pr_url)

@app.post("/plan/pr", response_model=PRFromPlanResponse)
async def plan_pr(req: PRFromPlanRequest, agent: str = Depends(require_auth)) -> PRFromPlanResponse:  # type: ignore[override]
    """Alias für /dev/pr-from-plan (zukünftige Stable Route).

    Nutzt identische Logik (Branch-Erstellung, Commit, optionaler PR Draft) und ermöglicht clients das stabilere
    Präfix /plan/pr zu verwenden.
    """
    return await pr_from_plan(req, agent)

@app.get("/")
async def root() -> HTMLResponse:  # type: ignore[override]
    html = """<html><head><title>Index</title><meta charset='utf-8'>
<style>body{font-family:system-ui;background:#0f1115;color:#f5f5f5;padding:32px}a{color:#60a5fa;text-decoration:none;font-weight:600}ul{line-height:1.6}code{background:#1e2530;padding:2px 5px;border-radius:4px}</style></head>
<body>
<h1>Übersicht</h1>
<p>Wichtige Bereiche:</p>
<ul>
 <li><a href='/ui'>Haupt-UI</a></li>
 <li><a href='/story/ui'>Story</a> (lebendiges Erzählsystem)</li>
 <li><a href='/docs'>OpenAPI Docs</a></li>
 <li><a href='/metrics'>Prometheus Metriken</a></li>
</ul>
<p>Setze ggf. zuerst deinen <code>API Key</code> in der Haupt-UI; die Story-UI nutzt lokal gespeicherten Schlüssel.</p>
</body></html>"""
    return HTMLResponse(content=html)

@app.post("/llm/chat", response_model=ChatResponse)
async def llm_chat(req: ChatRequest, agent: str = Depends(require_auth)):
    rate_limit(agent)
    persona = (req.persona or "default").lower()
    pref = PERSONA_PRESETS.get(persona)
    if pref:
        req.messages = [ChatMessage(role="system", content=pref)] + req.messages
    # memory capture
    try:
        last_user = next((m.content for m in reversed(req.messages) if m.role == 'user'), None)
        if last_user:
            _store_memory('user', last_user, persona, req.agent_id)
    except Exception:
        pass
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY not configured")
    try:
        from groq import Groq  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Groq SDK import failed: {e}")
    # model params
    base_model = "llama-3.3-70b-versatile"
    try:
        if current_policy and getattr(current_policy, 'llm', None) and getattr(current_policy.llm, 'model', None):
            base_model = current_policy.llm.model  # type: ignore[attr-defined]
    except Exception:
        pass
    model = req.model or base_model
    temperature = req.temperature if req.temperature is not None else 0.0
    try:
        if current_policy and getattr(current_policy, 'llm', None) and getattr(current_policy.llm, 'temperature', None) is not None and req.temperature is None:
            temperature = current_policy.llm.temperature  # type: ignore[attr-defined]
    except Exception:
        pass
    max_tokens = req.max_tokens if req.max_tokens is not None else 512
    try:
        if current_policy and getattr(current_policy, 'llm', None) and getattr(current_policy.llm, 'max_tokens', None) and req.max_tokens is None:
            max_tokens = current_policy.llm.max_tokens  # type: ignore[attr-defined]
    except Exception:
        pass
    if max_tokens > 4096:
        raise HTTPException(status_code=400, detail="max_tokens too large")
    risk_gate("\n".join(m.content for m in req.messages))
    LLM_REQUESTS_TOTAL.labels(model=model).inc()
    client = Groq(api_key=api_key)
    start = time.time()
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in req.messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        out_text = ''
        try:
            if getattr(completion, 'choices', None):
                first = completion.choices[0].message  # type: ignore[index]
                if hasattr(first, 'content'):
                    out_text = str(getattr(first, 'content') or '')
                elif isinstance(first, dict):
                    out_text = str(first.get('content') or '')
        except Exception:
            out_text = ''
        usage = getattr(completion, 'usage', None)
        usage_dict: dict[str, Any] | None = None
        if usage:
            try:
                usage_dict = {
                    'prompt_tokens': getattr(usage, 'prompt_tokens', None) or (usage.get('prompt_tokens') if isinstance(usage, dict) else None),
                    'completion_tokens': getattr(usage, 'completion_tokens', None) or (usage.get('completion_tokens') if isinstance(usage, dict) else None),
                    'total_tokens': getattr(usage, 'total_tokens', None) or (usage.get('total_tokens') if isinstance(usage, dict) else None),
                }
            except Exception:
                usage_dict = None
        if usage_dict:
            if usage_dict.get('prompt_tokens') is not None:
                LLM_TOKENS_TOTAL.labels(type="prompt").inc(int(usage_dict['prompt_tokens']))
            if usage_dict.get('completion_tokens') is not None:
                LLM_TOKENS_TOTAL.labels(type="completion").inc(int(usage_dict['completion_tokens']))
            if usage_dict.get('total_tokens') is not None:
                LLM_TOKENS_TOTAL.labels(type="total").inc(int(usage_dict['total_tokens']))
            if db_conn:
                try:
                    db_conn.execute(
                        "INSERT INTO llm_usage(ts, model, prompt_tokens, completion_tokens, total_tokens, latency_ms, meta) VALUES (?,?,?,?,?,?,?)",
                        (
                            time.time(),
                            model,
                            usage_dict.get('prompt_tokens'),
                            usage_dict.get('completion_tokens'),
                            usage_dict.get('total_tokens'),
                            (time.time() - start) * 1000.0,
                            json.dumps({}),
                        )
                    )
                    db_conn.commit()
                except Exception:
                    pass
        await emit_event("llm.chat", {"model": model, "tokens": (usage_dict or {}).get('total_tokens'), "persona": persona, "agent_id": req.agent_id})
        _store_memory('assistant', out_text, persona, req.agent_id)
        return ChatResponse(model=model, content=out_text, usage=usage_dict, persona=persona)
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

# ---------------- Idle Game Plugin -----------------
idle_state: dict[str, Any] = {
    "tick": 0,
    "resources": 0,
    "rules_version": 1,
}
_last_activity_ts = time.time()
IDLE_BACKGROUND_INTERVAL = int(os.getenv("IDLE_BACKGROUND_INTERVAL","30"))  # seconds between checks
IDLE_INACTIVITY_THRESHOLD = int(os.getenv("IDLE_INACTIVITY_THRESHOLD","120"))  # seconds idle before auto tick

# -------- Idle Quest System (progress via ticks) --------
class IdleQuest(BaseModel):
    id: str
    goal: str
    required: int
    progress: int
    status: str  # active|completed
    tags: list[str] | None = None
    created_at: float
    updated_at: float

class IdleQuestList(BaseModel):
    items: list[IdleQuest]

_idle_tick_history: list[dict[str, Any]] = []  # ring buffer of recent ticks
IDLE_TICK_HISTORY_LIMIT = 200

def _ensure_idle_quests_seed(conn: sqlite3.Connection) -> None:
    try:
        cur = conn.execute("SELECT COUNT(*) FROM idle_quests")
        if cur.fetchone()[0] == 0:
            now = time.time()
            seeds = [
                ("q_ticks_3","Erreiche 3 Idle Ticks",3,0,"active",json.dumps(["idle","starter"]),now,now),
                ("q_ticks_5","Erreiche 5 Idle Ticks",5,0,"active",json.dumps(["idle","starter"]),now,now),
            ]
            conn.executemany("INSERT INTO idle_quests(id,goal,required,progress,status,tags,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?)", seeds)
            conn.commit()
    except Exception:
        pass

def _list_idle_quests(conn: sqlite3.Connection) -> list[IdleQuest]:
    items: list[IdleQuest] = []
    try:
        cur = conn.execute("SELECT id,goal,required,progress,status,tags,created_at,updated_at FROM idle_quests ORDER BY created_at ASC")
        for row in cur.fetchall():
            qid, goal, req, prog, status, tags_json, ca, ua = row
            tags: list[str] | None = None
            if tags_json:
                try:
                    tj = json.loads(tags_json)
                    if isinstance(tj, list):
                        tags = [str(t) for t in tj]
                except Exception:
                    tags = None
            items.append(IdleQuest(id=qid, goal=goal, required=int(req), progress=int(prog), status=status, tags=tags, created_at=ca, updated_at=ua))
    except Exception:
        pass
    return items

def _update_idle_quest_progress(delta: int = 1) -> list[dict[str, Any]]:
    """Increment progress for all active quests; return list of progress change events."""
    changes: list[dict[str, Any]] = []
    if not db_conn:
        return changes
    try:
        cur = db_conn.execute("SELECT id,progress,required FROM idle_quests WHERE status='active'")
        rows = cur.fetchall()
        now = time.time()
        for qid, prog, req in rows:
            new_prog = prog + delta
            status = 'active'
            completed = False
            if new_prog >= req:
                new_prog = req
                status = 'completed'
                completed = True
            db_conn.execute("UPDATE idle_quests SET progress=?, status=?, updated_at=? WHERE id=?", (new_prog, status, now, qid))
            changes.append({"id": qid, "progress": new_prog, "required": req, "completed": completed})
        db_conn.commit()
    except Exception:
        pass
    return changes

async def _idle_background_loop():
    while True:
        try:
            now = time.time()
            if now - _last_activity_ts > IDLE_INACTIVITY_THRESHOLD:
                try:
                    idle_state["tick"] += 1
                    idle_state["resources"] += 1
                    changes = _update_idle_quest_progress(1)
                    payload = {"tick": idle_state["tick"], "resources": idle_state["resources"], "auto": True, "quests": changes}
                    _idle_tick_history.append({"ts": time.time(), **payload})
                    if len(_idle_tick_history) > IDLE_TICK_HISTORY_LIMIT:
                        del _idle_tick_history[0:len(_idle_tick_history)-IDLE_TICK_HISTORY_LIMIT]
                    await emit_event("idle.tick.auto", payload)
                    # emit quest events
                    for ch in changes:
                        await emit_event("idle.quest.progress", ch)
                        if ch.get("completed"):
                            await emit_event("idle.quest.completed", {"id": ch["id"]})
                    # alle 2 Auto-Ticks eine sanfte Hinweis-Empfehlung
                    if idle_state["tick"] % 2 == 0:
                        await emit_event("idle.suggest", {"tip": "Optionen erneuern oder Plan prüfen?"})
                except Exception:
                    pass
            await asyncio.sleep(IDLE_BACKGROUND_INTERVAL)
        except Exception as e:  # noqa: BLE001
            logger.warning("idle background loop error: %s", e)
            await asyncio.sleep(IDLE_BACKGROUND_INTERVAL)

# ---------------- In-Memory Conversational Memory (multi-agent) -----------------
MEMORY_LIMIT = int(os.getenv("MEMORY_LIMIT", "200"))
_memory_buffer: list[MemoryItem] = []

def _store_memory(role: str, content: str, persona: str | None = None, agent_id: str | None = None) -> None:
    try:
        item = MemoryItem(id=f"mem_{int(time.time()*1000)}_{random.randint(100,999)}", ts=time.time(), role=role, content=content[:4000], persona=persona, agent_id=agent_id)
        _memory_buffer.append(item)
        if len(_memory_buffer) > MEMORY_LIMIT:
            del _memory_buffer[0:len(_memory_buffer)-MEMORY_LIMIT]
    except Exception:
        pass

@app.get("/memory/list", response_model=MemoryList)
async def memory_list(limit: int = 50, persona: str | None = None, agent_id: str | None = None, agent: str = Depends(require_auth)) -> MemoryList:  # type: ignore[override]
    rate_limit(agent)
    view = [m for m in _memory_buffer if (persona is None or m.persona==persona) and (agent_id is None or m.agent_id==agent_id)]
    items = list(view)[-limit:][::-1]
    return MemoryList(items=items)

@app.get("/game/idle/state")
async def idle_get_state(agent: str = Depends(require_auth)) -> dict[str, Any]:
    rate_limit(agent)
    return {"state": idle_state}

@app.post("/game/idle/tick")
async def idle_tick(agent: str = Depends(require_auth)) -> dict[str, Any]:
    rate_limit(agent)
    idle_state["tick"] += 1
    # simple resource accumulation
    gained = 1 + (1 if idle_state["tick"] % 5 == 0 else 0)
    idle_state["resources"] += gained
    proposal: dict[str, Any] | None = None
    # every 7 ticks propose a tiny rule mutation as a 'plan variant' conceptually
    if idle_state["tick"] % 7 == 0:
        idle_state["rules_version"] += 1
        proposal = {
            "id": f"idle_rule_{idle_state['rules_version']}",
            "summary": "Increase passive gain",
            "patch_preview": "--- a/plugins/games/idle/rules.yaml\n+++ b/plugins/games/idle/rules.yaml\n@@\n-passive_gain: 1\n+passive_gain: 2\n",
        }
    # progress quests
    changes = _update_idle_quest_progress(1)
    for ch in changes:
        await emit_event("idle.quest.progress", ch)
        if ch.get("completed"):
            await emit_event("idle.quest.completed", {"id": ch["id"]})
    evt_payload: dict[str, Any] = {"tick": idle_state["tick"], "resources": idle_state["resources"], "proposal": bool(proposal), "quests": changes}
    IDLE_TICKS_TOTAL.inc()
    _idle_tick_history.append({"ts": time.time(), **evt_payload})
    if len(_idle_tick_history) > IDLE_TICK_HISTORY_LIMIT:
        del _idle_tick_history[0:len(_idle_tick_history)-IDLE_TICK_HISTORY_LIMIT]
    await emit_event("idle.tick", evt_payload)
    return {"state": idle_state, "gained": gained, "proposal": proposal}

@app.get("/game/idle/quests", response_model=IdleQuestList)
async def idle_quests(agent: str = Depends(require_auth)) -> IdleQuestList:  # type: ignore[override]
    rate_limit(agent)
    _ensure_db()
    _ensure_idle_quests_seed(db_conn)  # type: ignore[arg-type]
    items = _list_idle_quests(db_conn)  # type: ignore[arg-type]
    return IdleQuestList(items=items)

class IdleQuestCreateRequest(BaseModel):
    goal: str
    required: int = 5
    tags: list[str] | None = None

@app.post("/game/idle/quests", response_model=IdleQuest)
async def idle_quest_create(req: IdleQuestCreateRequest, agent: str = Depends(require_auth)) -> IdleQuest:  # type: ignore[override]
    rate_limit(agent)
    _ensure_db()
    if not db_conn:
        raise HTTPException(status_code=500, detail={"error":{"code":"db_unavailable"}})
    _ensure_idle_quests_seed(db_conn)  # type: ignore[arg-type]
    now = time.time()
    qid = f"q_{int(now)}_{random.randint(100,999)}"
    try:
        db_conn.execute(
            "INSERT INTO idle_quests(id,goal,required,progress,status,tags,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?)",
            (qid, req.goal[:200], int(req.required), 0, 'active', json.dumps(req.tags or []), now, now)
        )
        db_conn.commit()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail={"error":{"code":"quest_insert_failed","message":str(e)}})
    quest = IdleQuest(id=qid, goal=req.goal[:200], required=int(req.required), progress=0, status='active', tags=req.tags or [], created_at=now, updated_at=now)
    await emit_event("idle.quest.create", quest.model_dump())
    return quest

@app.get("/game/idle/log")
async def idle_log(limit: int = 10, agent: str = Depends(require_auth)) -> dict[str, Any]:  # type: ignore[override]
    rate_limit(agent)
    lim = max(1, min(200, limit))
    return {"items": list(_idle_tick_history)[-lim:][::-1], "total": len(_idle_tick_history)}

# ---------------- Environment & Suggestions -----------------
SAFE_ENV_KEYS = {"APP_VERSION","KILL_SWITCH","PUBLIC_SERVER_URL"}
EXCLUDE_DIRS = {".git","__pycache__","node_modules",".venv","venv"}

def _scan_files(limit: int = 200) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for root, dirs, files in os.walk('.'):
        # prune dirs
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for fn in files:
            if len(out) >= limit:
                return out
            p = os.path.join(root, fn).replace('..','')
            try:
                st = os.stat(p)
                out.append({"path": p.lstrip('./'), "size": st.st_size})
            except Exception:
                continue
    return out

@app.get("/env/info", response_model=EnvInfoResponse)
async def env_info(agent: str = Depends(require_auth)) -> EnvInfoResponse:  # type: ignore[override]
    rate_limit(agent)
    files = _scan_files()
    env_filtered = {k: os.getenv(k) for k in SAFE_ENV_KEYS if os.getenv(k) is not None}
    import platform, sys
    py = {"version": platform.python_version(), "executable": sys.executable.split(os.sep)[-1]}
    # simple metrics snapshot
    metrics = {"actions_total": None, "plans": None}
    try:
        metrics["actions_total"] = sum(int(s.samples[0].value) for s in ACTIONS_TOTAL.collect())  # type: ignore[attr-defined]
    except Exception:
        pass
    ENV_INFO_TOTAL.inc()
    payload = EnvInfoResponse(
        version=app.version,
        file_count=len(files),
        files=files,
        env=env_filtered,
        python=py,
        metrics=metrics,
    )
    await emit_event("env.info", {"files": payload.file_count})
    return payload

SUGGEST_DIR = "suggestions"
os.makedirs(SUGGEST_DIR, exist_ok=True)

def _suggestion_path(sid: str) -> str:
    return os.path.join(SUGGEST_DIR, f"{sid}.json")

def _save_suggestion(s: Suggestion) -> None:
    try:
        with open(_suggestion_path(s.id), 'w', encoding='utf-8') as f:
            json.dump(s.model_dump(), f, indent=2, ensure_ascii=False)
    except Exception:
        pass

def _load_suggestion(sid: str) -> Suggestion | None:
    try:
        with open(_suggestion_path(sid), 'r', encoding='utf-8') as f:
            data = json.load(f)
        return Suggestion.model_validate(data)
    except Exception:
        return None

def _iter_suggestions() -> list[Suggestion]:
    items: list[Suggestion] = []
    try:
        for fn in os.listdir(SUGGEST_DIR):
            if not fn.endswith('.json'): continue
            s = _load_suggestion(fn[:-5])
            if s: items.append(s)
    except Exception:
        pass
    return items

def update_open_gauge() -> None:
    items = _iter_suggestions()
    open_count = sum(1 for s in items if s.status in ("draft","revised"))
    try:
        SUGGEST_OPEN_GAUGE.set(open_count)
    except Exception:
        pass
    # fire event asynchronously (best-effort)
    try:
        asyncio.create_task(emit_event("suggest.open", {"open": open_count}))
    except Exception:
        pass

# -------- Auto Suggest (Static Heuristics) --------
AUTO_SCAN_MAX_FILES = int(os.getenv("AUTO_SUGGEST_MAX_FILES", "300"))
LARGE_FILE_THRESHOLD = int(os.getenv("AUTO_SUGGEST_LARGE_FILE", "1200"))  # lines
LARGE_FUNC_THRESHOLD = int(os.getenv("AUTO_SUGGEST_LARGE_FUNC", "80"))    # lines

def _read_text(path: str) -> str:
    try:
        if os.path.getsize(path) > 2_000_000:
            return ""  # skip huge
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception:
        return ""

def _scan_codebase_for_improvements() -> dict[str, Any]:
    findings: dict[str, Any] = {"todos": [], "large_files": [], "large_functions": [], "broad_except": [], "dup_literals": []}
    literal_counts: dict[str, int] = {}
    scanned = 0
    for root, dirs, files in os.walk('.'):  # simplistic walk respecting EXCLUDE_DIRS
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for fn in files:
            if scanned >= AUTO_SCAN_MAX_FILES:
                break
            if not fn.endswith(('.py', '.md', '.txt', '.yaml', '.yml')):
                continue
            path = os.path.join(root, fn).lstrip('./')
            text = _read_text(path)
            if not text:
                continue
            scanned += 1
            # TODO / FIXME
            for m in re.finditer(r'(?i)\b(TODO|FIXME|HACK)\b(.{0,80})', text):
                snippet = m.group(0).strip()
                findings['todos'].append({"file": path, "snippet": snippet})
            # large file (by lines)
            lines = text.splitlines()
            if len(lines) >= LARGE_FILE_THRESHOLD:
                findings['large_files'].append({"file": path, "lines": len(lines)})
            # naive function size detection (def ... : then until blank lines decrease indentation)
            if path.endswith('.py'):
                for i, line in enumerate(lines):
                    if line.startswith('def ') and line.rstrip().endswith(':'):
                        start_indent = len(line) - len(line.lstrip())
                        block_len = 0
                        for j in range(i+1, len(lines)):
                            l2 = lines[j]
                            if l2.strip()=='' and block_len>5:  # soft break after some lines
                                break
                            indent = len(l2) - len(l2.lstrip())
                            if l2 and indent <= start_indent and l2.lstrip().startswith('def '):
                                break
                            block_len += 1
                        if block_len >= LARGE_FUNC_THRESHOLD:
                            findings['large_functions'].append({"file": path, "line": i+1, "lines": block_len, "signature": line.strip()})
                # broad except
                for m in re.finditer(r'except\s+Exception\s*:', text):
                    findings['broad_except'].append({"file": path, "pos": m.start()})
            # duplicate string literals (very naive)
            for lit in re.findall(r'"([A-Za-z0-9 _-]{5,40})"|\'([A-Za-z0-9 _-]{5,40})\'', text):
                val = (lit[0] or lit[1]).strip()
                if not val or ' ' not in val:
                    continue  # skip single words
                literal_counts[val] = literal_counts.get(val, 0) + 1
    # collect duplicates above threshold
    for k,v in literal_counts.items():
        if v >= 4:
            findings['dup_literals'].append({"literal": k, "count": v})
    findings['scanned_files'] = scanned
    return findings

def _build_auto_suggestions(findings: dict[str, Any]) -> list[Suggestion]:
    suggestions: list[Suggestion] = []
    now = time.time()
    # Helper to create suggestion
    def make_suggestion(goal: str, rationale: str, steps: list[str], weaknesses: list[str], tags: list[str]) -> Suggestion:
        sid = f"sug_{int(now)}_{random.randint(1000,9999)}_auto"
        s = Suggestion(
            id=sid,
            created_at=time.time(),
            status="draft",
            goal=goal,
            focus_paths=None,
            summary=goal,
            rationale=rationale,
            recommended_steps=steps,
            potential_patches=[],
            risk_notes=["Nur statische Codeanalyse, keine geheimen Daten gelesen."],
            weaknesses=weaknesses,
            metrics_impact={"tests": "+" if 'test' in goal.lower() else "~", "docs": "+" if 'doku' in goal.lower() or 'doc' in goal.lower() else "~", "observability": "+" if 'metr' in goal.lower() else "~"},
            tags=tags,
        )
        return s
    todos = findings.get('todos', [])
    if todos:
        rationale = f"{len(todos)} TODO/FIXME/HACK Fundstellen erkannt. Konsolidierung erhöht Wartbarkeit."
        steps = ["TODOs gruppieren und in Issues überführen", "Veraltete TODOs entfernen", "Klare Ownership definieren"]
        weaknesses = ["Akkumulation von technischen Schulden", "Unklare Priorisierung von TODO Items"]
        suggestions.append(make_suggestion("Offene TODOs reduzieren", rationale, steps, weaknesses, ["auto","static"]))
    if findings.get('large_files'):
        lf = findings['large_files']
        rationale = f"{len(lf)} sehr große Dateien >= {LARGE_FILE_THRESHOLD} Zeilen erhöhen kognitive Last."
        steps = ["Dateien modularisieren", "Gemeinsame Utilities extrahieren", "Tests für kritische ausgelagerte Module ergänzen"]
        weaknesses = ["Erschwerte Navigation", "Höheres Merge-Konflikt Risiko"]
        suggestions.append(make_suggestion("Große Dateien aufteilen", rationale, steps, weaknesses, ["auto","static"]))
    if findings.get('large_functions'):
        lf = findings['large_functions']
        rationale = f"{len(lf)} Funktionen über {LARGE_FUNC_THRESHOLD} Zeilen gefunden – Refactoring senkt Fehlerrisiko."
        steps = ["Long Functions in kleinere Hilfsfunktionen extrahieren", "Parameterzahl prüfen", "Gezielte Unit Tests schreiben"]
        weaknesses = ["Schwierige Testbarkeit", "Erhöhte Fehleranfälligkeit"]
        suggestions.append(make_suggestion("Lange Funktionen refaktorieren", rationale, steps, weaknesses, ["auto","static"]))
    if findings.get('broad_except'):
        be = findings['broad_except']
        rationale = f"{len(be)} broad 'except Exception:' Blöcke – Präzisere Exceptions verbessern Fehlerdiagnose."
        steps = ["Konkrete Exception-Typen einsetzen", "Logging differenzieren", "Fehlerrouten testen"]
        weaknesses = ["Schluckt spezifische Fehler", "Monitoring erschwert"]
        suggestions.append(make_suggestion("Exception Handling präzisieren", rationale, steps, weaknesses, ["auto","static"]))
    if findings.get('dup_literals'):
        dl = findings['dup_literals']
        rationale = f"{len(dl)} häufig wiederholte Textbausteine (>=4x) – Konstante/Config reduziert Redundanz."
        steps = ["Wiederholte Strings als Konstante extrahieren", "Konfiguration prüfen (env/policy)", "Tests zur Konsistenz hinzufügen"]
        weaknesses = ["Erhöhte Inkonsistenz-Gefahr", "Schwieriger globale Anpassungen"]
        suggestions.append(make_suggestion("Duplizierte Literale konsolidieren", rationale, steps, weaknesses, ["auto","static"]))
    return suggestions

@app.post("/suggest/auto", response_model=list[Suggestion])
async def suggest_auto(agent: str = Depends(require_auth)) -> list[Suggestion]:  # type: ignore[override]
    rate_limit(agent)
    findings = _scan_codebase_for_improvements()
    sugs = _build_auto_suggestions(findings)
    for s in sugs:
        _save_suggestion(s)
        SUGGEST_GENERATED_TOTAL.inc()
        await emit_event("suggest.generated", {"id": s.id, "auto": True})
    update_open_gauge()
    return sugs

@app.post("/suggest/generate", response_model=Suggestion)
async def suggest_generate(req: SuggestGenerateRequest, agent: str = Depends(require_auth)) -> Suggestion:  # type: ignore[override]
    rate_limit(agent)
    # gather env snapshot (reuse scan lightly to avoid heavy call each time)
    files = _scan_files(limit=80)
    ts = time.time()
    sid = f"sug_{int(ts)}_{abs(hash(req.goal))%10000}"
    steps: list[str] = []
    rationale: list[str] = []
    weaknesses: list[str] = []
    # heuristic: recommend tests, docs, metrics based on existing dirs
    file_paths = [f['path'] for f in files]
    if not any(p.startswith('tests') for p in file_paths):
        steps.append("Add initial pytest suite (smoke tests for /plan and /env/info)")
        rationale.append("Keine tests/ Ordner gefunden → Grundabdeckung fehlt.")
        weaknesses.append("Fehlende Testabdeckung kann Regressionen unbemerkt lassen.")
    if 'README.md' in file_paths:
        steps.append("Erweitere README mit Abschnitt 'Vorschlags-Workflow'")
        rationale.append("README vorhanden, neuer Workflow dokumentieren.")
    else:
        weaknesses.append("README.md fehlt – Einstieg für neue Nutzer erschwert.")
    steps.append("UI Panel 'Suggestions' hinzufügen (Liste + Detail + Approve/Revise)")
    steps.append("Prometheus Gauge für offene Vorschläge anlegen")
    weaknesses.append("Keine Kennzahl für offene Vorschläge → Fortschritt schwer messbar.")
    patches = []
    if req.focus_paths:
        for p in req.focus_paths[:5]:
            patches.append({
                "target": p,
                "diff": f"--- a/{p}\n+++ b/{p}\n@@\n// Vorschlag Placeholder für {req.goal}\n",
            })
    suggestion = Suggestion(
        id=sid,
        created_at=ts,
        status="draft",
        goal=req.goal,
        focus_paths=req.focus_paths,
        summary=f"Vorschlag für Ziel: {req.goal}",
        rationale="\n".join(rationale) or "Heuristische Analyse basierend auf Dateiliste.",
        recommended_steps=steps,
        potential_patches=patches,
        risk_notes=["Nur Metadaten verwendet, keine geheimen ENV keys."],
        weaknesses=weaknesses,
        metrics_impact={"tests": "+", "docs": "+", "observability": "+"},
    )
    _save_suggestion(suggestion)
    SUGGEST_GENERATED_TOTAL.inc()
    update_open_gauge()
    await emit_event("suggest.generated", {"id": suggestion.id})
    return suggestion

@app.get("/suggest/review", response_model=Suggestion)
async def suggest_review_get(id: str, agent: str = Depends(require_auth)) -> Suggestion:  # type: ignore[override]
    rate_limit(agent)
    s = _load_suggestion(id)
    if not s:
        raise HTTPException(status_code=404, detail={"code": "suggest_not_found"})
    return s

@app.post("/suggest/review", response_model=SuggestReviewResponse)
async def suggest_review_post(req: SuggestReviewRequest, agent: str = Depends(require_auth)) -> SuggestReviewResponse:  # type: ignore[override]
    rate_limit(agent)
    s = _load_suggestion(req.id)
    if not s:
        raise HTTPException(status_code=404, detail={"code": "suggest_not_found"})
    revised = False
    if req.approve:
        if s.status != "approved":
            s.status = "approved"
            # attach impact scoring if not already present
            if not s.impact:
                # simple heuristic scoring: count of recommended steps & weaknesses
                base = len(s.recommended_steps)
                penalty = len(s.weaknesses or []) * 0.1
                score = max(0.1, min(1.0, (base / 10.0) - penalty + 0.3))
                rationale = f"Heuristisch: {base} Schritte, {len(s.weaknesses or [])} Schwächen, Basisgewichtung mit Startoffset."
                s.impact = {"score": round(score,2), "rationale": rationale, "approved_at": time.time()}
            # quest completion detection: any suggestion with tag starting 'quest:'
            if any(t.startswith("quest:") for t in s.tags):
                await emit_event("quest.completed", {"suggestion": s.id, "tags": s.tags})
                QUEST_COMPLETED_TOTAL.inc()
            _save_suggestion(s)
            await emit_event("suggest.approved", {"id": s.id})
            SUGGEST_REVIEW_TOTAL.labels(action="approve").inc()
    else:
        # create revised copy by augmenting rationale and steps
        revised = True
        new_id = s.id + "_rev"
        s = Suggestion(
            id=new_id,
            created_at=time.time(),
            status="revised",
            goal=s.goal,
            focus_paths=s.focus_paths,
            summary=s.summary + " (überarbeitet)",
            rationale=(s.rationale + (f"\nAnpassung: {req.adjustments}" if req.adjustments else ""))[:4000],
            recommended_steps=s.recommended_steps + ([f"Berücksichtige Anpassung: {req.adjustments}"] if req.adjustments else []),
            potential_patches=s.potential_patches,
            risk_notes=s.risk_notes,
            metrics_impact=s.metrics_impact,
        )
        _save_suggestion(s)
        await emit_event("suggest.revised", {"id": s.id})
        SUGGEST_REVIEW_TOTAL.labels(action="revise").inc()
    update_open_gauge()
    return SuggestReviewResponse(suggestion=s, revised=revised)

@app.get("/suggest/impact", response_model=ImpactInfo)
async def suggest_impact(id: str, agent: str = Depends(require_auth)) -> ImpactInfo:  # type: ignore[override]
    rate_limit(agent)
    s = _load_suggestion(id)
    if not s:
        raise HTTPException(status_code=404, detail={"code": "suggest_not_found"})
    if not s.impact:
        raise HTTPException(status_code=404, detail={"code": "impact_not_found"})
    return ImpactInfo(id=s.id, score=float(s.impact.get("score",0.0)), rationale=str(s.impact.get("rationale","")), approved_at=float(s.impact.get("approved_at",0.0)))

@app.get("/suggest/list", response_model=SuggestListResponse)
async def suggest_list(limit: int = 50, full: bool = False, agent: str = Depends(require_auth)) -> SuggestListResponse:  # type: ignore[override]
    rate_limit(agent)
    items = _iter_suggestions()
    items.sort(key=lambda s: s.created_at, reverse=True)
    total = len(items)
    open_cnt = sum(1 for s in items if s.status in ("draft","revised"))
    view: list[SuggestListItem] = []
    for s in items[:limit]:
        view.append(SuggestListItem(id=s.id, status=s.status, goal=s.goal, created_at=s.created_at, weaknesses=(s.weaknesses if full else None)))
    return SuggestListResponse(total=total, open=open_cnt, items=view)

# Quest listing derived from suggestions tagged with quest:
@app.get("/quest/list", response_model=QuestListResponse)
async def quest_list(agent: str = Depends(require_auth)) -> QuestListResponse:  # type: ignore[override]
    rate_limit(agent)
    items = []
    for s in _iter_suggestions():
        qtags = [t for t in s.tags if t.startswith("quest:")]
        if not qtags:
            continue
        status = "done" if s.status == "approved" else ("revised" if s.status == "revised" else "pending")
        diff = None
        # optional difficulty tag quest:hard etc.
        for t in qtags:
            if ":" in t:
                parts = t.split(":",2)
                if len(parts)==3:
                    diff = parts[2]
        items.append(QuestItem(id=s.id, goal=s.goal, status=status, created_at=s.created_at, difficulty=diff))
    # sort newest first
    items.sort(key=lambda x: x.created_at, reverse=True)
    return QuestListResponse(items=items)

@app.post("/suggest/llm", response_model=SuggestLLMResponse)
async def suggest_llm(req: SuggestLLMRequest, agent: str = Depends(require_auth)) -> SuggestLLMResponse:  # type: ignore[override]
    rate_limit(agent)
    s = _load_suggestion(req.id)
    if not s:
        raise HTTPException(status_code=404, detail={"code": "suggest_not_found"})
    refined = False
    api_key = os.getenv("GROQ_API_KEY")
    new_summary = s.summary
    new_rationale = s.rationale
    if api_key:
        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            prompt = (
                f"Verbessere die folgende Vorschlags-Zusammenfassung und rationale. Ziel: {s.goal}. "
                f"Schwächen: {', '.join(s.weaknesses or [])}. "
                f"Anweisung: {req.instruction or 'Nutze präzisere Formulierungen, bleibe knapp.'}"
            )
            completion = client.chat.completions.create(
                model=os.getenv("GROQ_MODEL","llama-3.3-70b-versatile"),
                messages=[{"role":"system","content":"Du bist ein hilfsbereiter Assistent für Software-Refinement"},
                          {"role":"user","content": prompt + "\n---\nSummary: " + s.summary + "\nRationale:\n" + s.rationale}],
                temperature=0.2,
                max_tokens=400,
            )
            if completion.choices:
                txt = getattr(completion.choices[0].message, 'content', '') or ''
                if txt:
                    new_summary = (txt.split('\n',1)[0][:200]).strip() or new_summary
                    new_rationale = (txt[:2000]).strip()
                    refined = True
        except Exception:
            pass
    else:
        # Heuristic refinement
        new_summary = (s.summary + " (präzisiert)")[:200]
        if req.instruction:
            new_rationale = (s.rationale + f"\nInstruktionshinweis: {req.instruction}")[:2000]
        refined = True
    if refined:
        s = Suggestion(
            id=s.id + ("_llm" if not s.id.endswith("_llm") else ""),
            created_at=time.time(),
            status="revised" if s.status != "approved" else s.status,
            goal=s.goal,
            focus_paths=s.focus_paths,
            summary=new_summary,
            rationale=new_rationale,
            recommended_steps=s.recommended_steps,
            potential_patches=s.potential_patches,
            risk_notes=s.risk_notes,
            weaknesses=s.weaknesses,
            metrics_impact=s.metrics_impact,
        )
        _save_suggestion(s)
        SUGGEST_REVIEW_TOTAL.labels(action="llm_refine").inc()
        update_open_gauge()
        await emit_event("suggest.refined", {"id": s.id})
    return SuggestLLMResponse(suggestion=s, refined=refined)

# Thought Stream Feature
THOUGHT_INTERVAL_MIN = float(os.getenv("THOUGHT_STREAM_INTERVAL_MIN", "5"))
THOUGHT_INTERVAL_MAX = float(os.getenv("THOUGHT_STREAM_INTERVAL_MAX", "10"))
THOUGHT_LIMIT = int(os.getenv("THOUGHT_STREAM_LIMIT", "200"))
_recent_thoughts: list[Thought] = []
async def _generate_thought() -> Thought:
    now = time.time()
    open_sugs = 0
    try:
        open_sugs = int(SUGGEST_OPEN_GAUGE._value.get())  # type: ignore
    except Exception:
        pass
    idle_tick = 0
    try: idle_tick = int(idle_state.get('tick',0))
    except Exception: pass
    theme = random.choice(["reflexion","idee","notiz","vision","risiko","beobachtung"])
    fragments = [f"open_suggestions={open_sugs}", f"idle_tick={idle_tick}"]
    if open_sugs>5: fragments.append("fokus: approvals beschleunigen")
    elif open_sugs==0: fragments.append("fenster für neue initiative")
    if idle_tick and idle_tick % 7 == 0: fragments.append("idle schwellwert erreicht")
    text = f"[{theme}] " + "; ".join(fragments)
    # simple keyword category mapping
    cat = "neutral"
    lower = text.lower()
    if "risiko" in lower: cat = "risk"
    elif "initiative" in lower or "vision" in lower: cat = "opportunity"
    elif "approval" in lower or "approvals" in lower or "fokus" in lower: cat = "action"
    elif "idle" in lower: cat = "system"
    t = Thought(id=f"th_{int(now)}_{random.randint(100,999)}", ts=now, text=text, meta={"open":open_sugs,"idle":idle_tick}, category=cat)
    return t
async def _thought_loop():
    while True:
        try:
            t = await _generate_thought()
            _recent_thoughts.append(t)
            if len(_recent_thoughts) > THOUGHT_LIMIT:
                _recent_thoughts[:] = _recent_thoughts[-THOUGHT_LIMIT:]
            if db_conn:
                try:
                    db_conn.execute("INSERT INTO thoughts(ts, text, kind, meta) VALUES (?,?,?,?)", (t.ts, t.text, t.kind, json.dumps(t.meta or {})))
                    db_conn.commit()
                except Exception: pass
            THOUGHT_GENERATED_TOTAL.inc()
            THOUGHT_CATEGORY_TOTAL.labels(category=t.category or "unknown").inc()
            await emit_event("thought.stream", {"id": t.id, "text": t.text, "category": t.category})
        except Exception as e:
            logger.warning("thought loop error: %s", e)
        await asyncio.sleep(random.uniform(THOUGHT_INTERVAL_MIN, THOUGHT_INTERVAL_MAX))
@app.get("/thought/stream", response_model=ThoughtStreamResponse)
async def thought_stream(limit: int = 50, category: str | None = None, agent: str = Depends(require_auth)) -> ThoughtStreamResponse:  # type: ignore
    rate_limit(agent)
    items_full = list(_recent_thoughts)
    if category:
        items_full = [t for t in items_full if (t.category or "") == category]
    items = items_full[-limit:]
    return ThoughtStreamResponse(items=items[::-1], total=len(items_full))
@app.post("/thought/generate", response_model=Thought)
async def thought_generate(agent: str = Depends(require_auth)) -> Thought:  # type: ignore
    rate_limit(agent)
    t = await _generate_thought()
    _recent_thoughts.append(t)
    if len(_recent_thoughts) > THOUGHT_LIMIT:
        _recent_thoughts[:] = _recent_thoughts[-THOUGHT_LIMIT:]
    if db_conn:
        try:
            db_conn.execute("INSERT INTO thoughts(ts, text, kind, meta) VALUES (?,?,?,?)", (t.ts, t.text, t.kind, json.dumps(t.meta or {})))
            db_conn.commit()
        except Exception: pass
    THOUGHT_GENERATED_TOTAL.inc()
    THOUGHT_CATEGORY_TOTAL.labels(category=t.category or "unknown").inc()
    await emit_event("thought.stream", {"id": t.id, "text": t.text, "category": t.category, "manual": True})
    return t

class ThoughtPinRequest(BaseModel):
    pinned: bool

@app.patch("/thought/{thought_id}/pin", response_model=Thought)
async def thought_pin(thought_id: str, req: ThoughtPinRequest, agent: str = Depends(require_auth)) -> Thought:  # type: ignore
    rate_limit(agent)
    # locate in recent list
    found: Thought | None = None
    for t in _recent_thoughts:
        if t.id == thought_id:
            found = t
            break
    if not found:
        raise HTTPException(status_code=404, detail={"error":{"code":"thought_not_found"}})
    found.pinned = bool(req.pinned)
    # emit event
    await emit_event("pinned_insight", {"id": found.id, "pinned": found.pinned, "category": found.category})
    return found

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
