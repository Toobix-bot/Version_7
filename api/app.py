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

from fastapi import FastAPI, Depends, HTTPException, status, Request, Header, Body, Query
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse, RedirectResponse
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

app = FastAPI(title="Life-Agent Dev Environment", version="0.1.0")

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
# ensure db_conn has an explicit typed declaration near top (if not already)
try:
    db_conn  # type: ignore  # noqa: F821
except NameError:
    from typing import Optional as _Opt
    db_conn: 'sqlite3.Connection | None' = None  # runtime assigned in init_db

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

# ---------------- Story / Narrative Models (MVP Phase 1) ----------------
class StoryOption(BaseModel):
    id: str
    label: str
    rationale: str | None = None
    risk: int = 0  # 0–3
    expected: dict[str, int] | None = None
    tags: list[str] = []
    expires_at: float | None = None

class StoryEvent(BaseModel):
    id: str
    ts: float
    epoch: int
    kind: str  # tick|action|system
    text: str
    mood: str
    deltas: dict[str, int] | None = None
    tags: list[str] = []
    option_ref: str | None = None

class StoryState(BaseModel):
    epoch: int
    mood: str
    arc: str
    resources: dict[str, int]
    options: list[StoryOption] = []

# --- Arc Evaluation (simple progression) ---
_ARC_ORDER = ["foundations", "exploration", "mastery"]

def _eval_arc(resources: dict[str,int], current: str) -> str:
    lvl = resources.get("level",1)
    # simple thresholds
    if lvl >= 10:
        return "mastery"
    if lvl >= 3:
        return "exploration"
    return "foundations"

def _maybe_arc_shift(conn: sqlite3.Connection, prev_arc: str, new_arc: str, epoch: int, mood: str) -> StoryEvent | None:
    if new_arc == prev_arc:
        return None
    ev_id = f"sev_{int(_story_now()*1000)}"
    text = f"Arc-Wechsel: {prev_arc} -> {new_arc}"
    conn.execute(
        "INSERT INTO story_events(ts, epoch, kind, text, mood, deltas, tags, option_ref) VALUES (?,?,?,?,?,?,?,?)",
        (_story_now(), epoch, "arc_shift", text, mood, json.dumps({}), json.dumps(["arc_shift"]), None)
    )
    return StoryEvent(id=ev_id, ts=_story_now(), epoch=epoch, kind="arc_shift", text=text, mood=mood, deltas={}, tags=["arc_shift"], option_ref=None)

# Default resources baseline
# Story Ressource Keys (deutsch)
# energie, wissen, inspiration, ruf, stabilitaet, erfahrung, level
_STORY_RESOURCE_KEYS = [
    "energie",
    "wissen",
    "inspiration",
    "ruf",
    "stabilitaet",
    "erfahrung",
    "level",
]
_STORY_OLD_KEY_MAP = {
    "energy": "energie",
    "knowledge": "wissen",
    "inspiration": "inspiration",
    "reputation": "ruf",
    "stability": "stabilitaet",
    "xp": "erfahrung",
    "level": "level",
}

def _story_now() -> float:
    return time.time()

def _get_story_state(conn: sqlite3.Connection) -> StoryState:
    cur = conn.execute("SELECT epoch, mood, arc, resources FROM story_state WHERE id=1")
    row = cur.fetchone()
    if not row:
        # initialize
        resources = {k: 0 for k in _STORY_RESOURCE_KEYS}
        resources.update({"energie": 80, "stabilitaet": 80, "level": 1})
        conn.execute(
            "INSERT INTO story_state(id, ts, epoch, mood, arc, resources) VALUES (1, ?, ?, ?, ?, ?)",
            (_story_now(), 0, "calm", "foundations", json.dumps(resources)),
        )
        conn.commit()
        return StoryState(epoch=0, mood="calm", arc="foundations", resources=resources, options=[])
    epoch, mood, arc, resources_json = row
    resources = json.loads(resources_json)
    # backward compatibility mapping englische keys -> deutsch
    migrated = False
    for old, new in _STORY_OLD_KEY_MAP.items():
        if old in resources:
            resources[new] = resources.get(new, 0) + resources.pop(old)
            migrated = True
    if migrated:
        conn.execute("UPDATE story_state SET resources=? WHERE id=1", (json.dumps(resources),))
        conn.commit()
    options = _list_story_options(conn)
    return StoryState(epoch=epoch, mood=mood, arc=arc, resources=resources, options=options)

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
    if res.get("erfahrung", 0) >= (state.resources.get("level", 1) * 100):
        xp_cost = state.resources.get("level", 1) * 100
        opts.append(
            StoryOption(
                id=f"opt_level_{int(_story_now())}",
                label="Reflektion und Level-Aufstieg",
                rationale="Erfahrungsschwelle erreicht",
                risk=1,
                expected={"erfahrung": -xp_cost, "level": +1, "stabilitaet": +5},
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
    new_arc = _eval_arc(new_resources, state.arc)
    conn.execute("UPDATE story_state SET ts=?, epoch=?, mood=?, arc=?, resources=? WHERE id=1", (_story_now(), epoch, mood, new_arc, json.dumps(new_resources)))
    ev_id = f"sev_{int(_story_now()*1000)}"
    conn.execute("INSERT INTO story_events(ts, epoch, kind, text, mood, deltas, tags, option_ref) VALUES (?,?,?,?,?,?,?,?)", (_story_now(), epoch, "action", label, mood, json.dumps(deltas), json.dumps(["action"]), option_id))
    arc_event = _maybe_arc_shift(conn, state.arc, new_arc, epoch, mood)
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
    epoch = state.epoch + 1
    mood = state.mood
    if resources.get("energie",0) < 30:
        mood = "strained"
    new_arc = _eval_arc(resources, state.arc)
    conn.execute("UPDATE story_state SET ts=?, epoch=?, mood=?, arc=?, resources=? WHERE id=1", (_story_now(), epoch, mood, new_arc, json.dumps(resources)))
    ev_id = f"sev_{int(_story_now()*1000)}"
    text = "Zeit vergeht. Eine stille Verschiebung im inneren Raum."
    conn.execute(
        "INSERT INTO story_events(ts, epoch, kind, text, mood, deltas, tags, option_ref) VALUES (?,?,?,?,?,?,?,?)",
        (_story_now(), epoch, "tick", text, mood, json.dumps({"energie": -1}), json.dumps(["tick"]), None),
    )
    _maybe_arc_shift(conn, state.arc, new_arc, epoch, mood)
    # regenerate options occasionally
    _refresh_story_options(conn, StoryState(epoch=epoch, mood=mood, arc=state.arc, resources=resources, options=[]))
    conn.commit()
    return StoryEvent(id=ev_id, ts=_story_now(), epoch=epoch, kind="tick", text=text, mood=mood, deltas={"energie": -1}, tags=["tick"], option_ref=None)

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
# (require_auth defined later; provide lightweight forwarder for type check/import)
try:
    require_auth  # type: ignore[name-defined]
except NameError:
    async def require_auth(x_api_key: str | None = Header(None, alias="X-API-Key")) -> str:  # type: ignore
        return "default-agent"

@app.get("/story/state", response_model=StoryState)
async def story_get_state(agent: str = Depends(require_auth)) -> StoryState:  # type: ignore[override]
    rate_limit(agent)
    st = _get_story_state(db_conn)  # type: ignore[arg-type]
    return st

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
        CREATE TABLE IF NOT EXISTS thoughts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            text TEXT NOT NULL,
            kind TEXT NOT NULL,
            meta TEXT
        );
        -- Story / Narrative tables (MVP)
        CREATE TABLE IF NOT EXISTS story_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            ts REAL NOT NULL,
            epoch INTEGER NOT NULL,
            mood TEXT NOT NULL,
            arc TEXT NOT NULL,
            resources TEXT NOT NULL -- JSON serialized resource map
        );
        CREATE TABLE IF NOT EXISTS story_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            epoch INTEGER NOT NULL,
            kind TEXT NOT NULL,
            text TEXT NOT NULL,
            mood TEXT NOT NULL,
            deltas TEXT, -- JSON map of resource deltas
            tags TEXT,   -- JSON array
            option_ref TEXT
        );
        CREATE TABLE IF NOT EXISTS story_options (
            id TEXT PRIMARY KEY,
            created_at REAL NOT NULL,
            label TEXT NOT NULL,
            rationale TEXT,
            risk INTEGER NOT NULL,
            expected TEXT, -- JSON map expected resource delta
            tags TEXT, -- JSON array
            expires_at REAL
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
    if os.getenv("THOUGHT_STREAM_ENABLE", "1") == "1":
        try:
            asyncio.create_task(_thought_loop())
        except Exception:
            logger.warning("failed to start thought loop")

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
        {"id": "v_safe", "label": "Vorsichtig", "risk_level": "vorsichtig", "temperature": 0.0, "depth_limit": 3, "risk_budget": "low"},
        {"id": "v_bal", "label": "Balanciert", "risk_level": "balanciert", "temperature": 0.0, "depth_limit": 5, "risk_budget": "medium"},
        {"id": "v_bold", "label": "Mutig", "risk_level": "mutig", "temperature": 0.2, "depth_limit": 8, "risk_budget": "high"},
    ]
    # Reorder to emphasize requested risk_budget first if provided
    if rb in ("low","medium","high"):
        priority = {"low":0, "medium":1, "high":2}
        variant_specs.sort(key=lambda v: priority.get(v['risk_budget'], 99) + (0 if v['risk_budget']==rb else 10))

    variants: list[PlanVariant] = []
    for spec in variant_specs:
        summary = f"{spec['label']} Variante für '{base_intent}' mit Tiefe {spec['depth_limit']}"
        explanation = (
            "Konservativer Ansatz mit minimalem Risiko." if spec['risk_budget']=="low" else
            ("Ausgewogene Änderungen mit moderatem Umfang." if spec['risk_budget']=="medium" else "Aggressivere Variante mit erweiterten Änderungen.")
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
            },
            summary=summary,
            explanation=explanation,
            patch_preview=patch_preview,
        ))

    return PlanResponse(
        status="created",
        artifact=plan_filename,
        applied_policies=artifact['policies'],
        variants=variants,
    )

@app.get("/")
async def root() -> RedirectResponse:
    return RedirectResponse(url="/ui")

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
    evt_payload: dict[str, Any] = {"tick": idle_state["tick"], "resources": idle_state["resources"], "proposal": bool(proposal)}
    IDLE_TICKS_TOTAL.inc()
    await emit_event("idle.tick", evt_payload)
    return {"state": idle_state, "gained": gained, "proposal": proposal}

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
async def thought_stream(limit: int = 50, agent: str = Depends(require_auth)) -> ThoughtStreamResponse:  # type: ignore
    rate_limit(agent)
    items = list(_recent_thoughts)[-limit:]
    return ThoughtStreamResponse(items=items[::-1], total=len(_recent_thoughts))
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
