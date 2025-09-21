"""Core Story / Narrative logic extracted from app.py for clarity.
Minimal refactor: pure functions + dataclasses (pydantic BaseModel) reused by app.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any

from pydantic import BaseModel

# ---------------- Models ----------------
class StoryOption(BaseModel):
    id: str
    label: str
    rationale: str | None = None
    risk: int = 0
    expected: dict[str, int] | None = None
    tags: list[str] = []
    expires_at: float | None = None

class StoryEvent(BaseModel):
    id: str
    ts: float
    epoch: int
    kind: str
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
    companions: list[dict[str, Any]] = []
    buffs: list[dict[str, Any]] = []
    skills: list[dict[str, Any]] = []

# ------------- Meta Resource Models -------------
class CompanionCreate(BaseModel):
    name: str
    archetype: str | None = None
    mood: str | None = None
    stats: dict[str, Any] | None = None

class Companion(CompanionCreate):
    id: int
    acquired_at: float

class BuffCreate(BaseModel):
    label: str
    kind: str | None = None
    magnitude: int | None = None
    expires_at: float | None = None
    meta: dict[str, Any] | None = None

class Buff(BuffCreate):
    id: int

class SkillCreate(BaseModel):
    name: str
    category: str | None = None
    level: int | None = None  # allow explicit level (defaults to 1 if None)
    xp: int | None = None     # allow explicit xp (defaults to 0)

class Skill(SkillCreate):
    id: int
    updated_at: float

_ARC_ORDER = ["foundations", "exploration", "mastery"]

_STORY_RESOURCE_KEYS = [
    "energie","wissen","inspiration","ruf","stabilitaet","erfahrung","level"
]

_STORY_OLD_KEY_MAP = {"energy":"energie","knowledge":"wissen","stability":"stabilitaet","xp":"erfahrung"}

XP_BASE = float(os.getenv("STORY_XP_BASE", "80"))
XP_EXP = float(os.getenv("STORY_XP_EXP", "1.4"))
INSP_TICK_THRESHOLD = int(os.getenv("STORY_INSP_TICK_THRESHOLD", "120"))
INSP_SOFT_CAP = int(os.getenv("STORY_INSP_SOFT_CAP", "150"))
INSP_MIN_ENERGY = int(os.getenv("STORY_INSP_MIN_ENERGY", "10"))

# ------------- helpers -------------

def _story_now() -> float: return time.time()

def eval_arc(resources: dict[str,int], current: str) -> str:
    lvl = resources.get("level",1)
    if lvl >= 10: return "mastery"
    if lvl >= 3: return "exploration"
    return "foundations"

def maybe_arc_shift(conn: sqlite3.Connection, prev_arc: str, new_arc: str, epoch: int, mood: str) -> StoryEvent | None:
    if new_arc == prev_arc: return None
    ts = _story_now()
    ev_id = f"sev_{int(ts*1000)}"
    text = f"Arc-Wechsel: {prev_arc} -> {new_arc}"
    conn.execute("INSERT INTO story_events(ts, epoch, kind, text, mood, deltas, tags, option_ref) VALUES (?,?,?,?,?,?,?,?)",
                 (ts, epoch, "arc_shift", text, mood, json.dumps({}), json.dumps(["arc_shift"]), None))
    return StoryEvent(id=ev_id, ts=ts, epoch=epoch, kind="arc_shift", text=text, mood=mood, deltas={}, tags=["arc_shift"], option_ref=None)

def _load_meta(conn: sqlite3.Connection) -> tuple[list[dict[str,Any]], list[dict[str,Any]], list[dict[str,Any]]]:
    companions: list[dict[str,Any]] = []
    for row in conn.execute("SELECT id,name,archetype,mood,stats,acquired_at FROM story_companions ORDER BY id ASC").fetchall():
        cid,name,arch,mood,stats_json,acq = row
        try: stats = json.loads(stats_json) if stats_json else {}
        except Exception: stats = {}
        companions.append({"id":cid,"name":name,"archetype":arch,"mood":mood,"stats":stats,"acquired_at":acq})
    buffs: list[dict[str,Any]] = []
    for row in conn.execute("SELECT id,label,kind,magnitude,expires_at,meta FROM story_buffs ORDER BY id ASC").fetchall():
        bid,label,kind,mag,exp,meta_json = row
        try: meta = json.loads(meta_json) if meta_json else {}
        except Exception: meta={}
        buffs.append({"id":bid,"label":label,"kind":kind,"magnitude":mag,"expires_at":exp,"meta":meta})
    skills: list[dict[str,Any]] = []
    for row in conn.execute("SELECT id,name,level,xp,category,updated_at FROM story_skills ORDER BY id ASC").fetchall():
        sid,name,lvl,xp,cat,upd = row
        skills.append({"id":sid,"name":name,"level":lvl,"xp":xp,"category":cat,"updated_at":upd})
    return companions,buffs,skills

def get_story_state(conn: sqlite3.Connection) -> StoryState:
    row = conn.execute("SELECT epoch, mood, arc, resources FROM story_state WHERE id=1").fetchone()
    if not row:
        resources = {k:(0 if k!="energie" else 80) for k in _STORY_RESOURCE_KEYS}
        conn.execute("INSERT INTO story_state(id, ts, epoch, mood, arc, resources) VALUES (1, ?, ?, ?, ?, ?)", (_story_now(),0,"calm","foundations", json.dumps(resources)))
        conn.commit()
        return StoryState(epoch=0,mood="calm",arc="foundations",resources=resources,options=[])
    epoch,mood,arc,resources_json = row
    resources = json.loads(resources_json)
    migrated=False
    for old,new in _STORY_OLD_KEY_MAP.items():
        if old in resources:
            resources[new] = resources.get(new,0)+resources.pop(old)
            migrated=True
    if migrated:
        conn.execute("UPDATE story_state SET resources=? WHERE id=1", (json.dumps(resources),))
        conn.commit()
    options = list_story_options(conn)
    companions,buffs,skills = _load_meta(conn)
    return StoryState(epoch=epoch,mood=mood,arc=arc,resources=resources,options=options,companions=companions,buffs=buffs,skills=skills)

def list_story_options(conn: sqlite3.Connection) -> list[StoryOption]:
    cur = conn.execute(
        "SELECT id,label,rationale,risk,expected,tags,expires_at FROM story_options ORDER BY created_at DESC"
    )
    out: list[StoryOption] = []
    now = _story_now()
    for row in cur.fetchall():
        oid, label, rationale, risk, expected_json, tags_json, expires_at = row
        if expires_at and expires_at < now:
            continue
        expected: dict[str, int] | None = None
        if expected_json:
            try:
                raw = json.loads(expected_json)
                if isinstance(raw, dict):
                    expected = {
                        str(k): int(v)
                        for k, v in raw.items()
                        if isinstance(v, (int, float))
                    }
            except Exception:
                expected = None
        tags: list[str] = []
        if tags_json:
            try:
                raw_t = json.loads(tags_json)
                if isinstance(raw_t, list):
                    tags = [str(t) for t in raw_t]
            except Exception:
                tags = []
        out.append(
            StoryOption(
                id=oid,
                label=label,
                rationale=rationale,
                risk=int(risk),
                expected=expected,
                tags=tags,
                expires_at=expires_at,
            )
        )
    return out


def generate_story_options(state: StoryState) -> list[StoryOption]:
    opts: list[StoryOption] = []
    resources = state.resources
    if resources.get("energie", 0) < 40:
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
    if resources.get("inspiration", 0) > 10 and resources.get("wissen", 0) < 50:
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
    level = resources.get("level", 1)
    try:
        xp_threshold = int((level ** XP_EXP) * XP_BASE)
    except Exception:
        xp_threshold = int(level * XP_BASE)
    if resources.get("erfahrung", 0) >= xp_threshold:
        xp_cost = xp_threshold
        opts.append(
            StoryOption(
                id=f"opt_level_{int(_story_now())}",
                label="Reflektion und Level-Aufstieg",
                rationale="Erfahrungsschwelle erreicht",
                risk=1,
                expected={
                    "erfahrung": -xp_cost,
                    "level": +1,
                    "stabilitaet": +5,
                    "inspiration": +3,
                },
                tags=["levelup"],
            )
        )
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


def persist_options(conn: sqlite3.Connection, options: list[StoryOption]) -> None:
    now = _story_now()
    for opt in options:
        conn.execute(
            "INSERT OR REPLACE INTO story_options(id, created_at, label, rationale, risk, expected, tags, expires_at) VALUES (?,?,?,?,?,?,?,?)",
            (
                opt.id,
                now,
                opt.label,
                opt.rationale,
                opt.risk,
                json.dumps(opt.expected) if opt.expected else None,
                json.dumps(opt.tags),
                opt.expires_at,
            ),
        )
    conn.commit()


def refresh_story_options(conn: sqlite3.Connection, state: StoryState) -> list[StoryOption]:
    conn.execute("DELETE FROM story_options")
    opts = generate_story_options(state)
    try:
        cur = conn.execute("SELECT name FROM story_companions")
        names = {str(row[0]).lower() for row in cur.fetchall()}
        if any(name.startswith("mentor") for name in names):
            for opt in opts:
                if opt.risk > 0:
                    opt.risk -= 1
                if "mentor" not in opt.tags:
                    opt.tags.append("mentor")
    except Exception:
        pass
    persist_options(conn, opts)
    return opts


def apply_story_option(conn: sqlite3.Connection, state: StoryState, option_id: str) -> StoryEvent:
    cur = conn.execute(
        "SELECT id,label,rationale,risk,expected FROM story_options WHERE id=?",
        (option_id,),
    )
    row = cur.fetchone()
    if not row:
        raise LookupError("story.option_not_found")
    _, label, _rationale, risk, expected_json = row
    expected: dict[str, int] = {}
    if expected_json:
        try:
            raw = json.loads(expected_json)
            if isinstance(raw, dict):
                for key, value in raw.items():
                    if isinstance(value, (int, float)):
                        expected[str(key)] = int(value)
        except Exception:
            expected = {}
    base_risk = int(risk) if risk is not None else 0
    if "erfahrung" not in expected:
        expected["erfahrung"] = max(1, base_risk)
    new_resources = dict(state.resources)
    deltas: dict[str, int] = {}
    for key, delta in expected.items():
        before = new_resources.get(key, 0)
        after = before + int(delta)
        new_resources[key] = after
        deltas[key] = int(delta)
    if new_resources.get("energie", 0) < 0:
        new_resources["energie"] = 0
    epoch = int(getattr(state, "epoch", 0)) + 1
    mood = state.mood
    if deltas.get("energie", 0) > 0:
        mood = "calm"
    if deltas.get("inspiration", 0) > 0:
        mood = "curious"
    new_arc = eval_arc(new_resources, state.arc)
    ts = _story_now()
    conn.execute(
        "UPDATE story_state SET ts=?, epoch=?, mood=?, arc=?, resources=? WHERE id=1",
        (ts, epoch, mood, new_arc, json.dumps(new_resources)),
    )
    ev_id = f"sev_{int(ts * 1000)}"
    conn.execute(
        "INSERT INTO story_events(ts, epoch, kind, text, mood, deltas, tags, option_ref) VALUES (?,?,?,?,?,?,?,?)",
        (
            ts,
            epoch,
            "action",
            label,
            mood,
            json.dumps(deltas),
            json.dumps(["action"]),
            option_id,
        ),
    )
    maybe_arc_shift(conn, state.arc, new_arc, epoch, mood)
    conn.execute("DELETE FROM story_options WHERE id=?", (option_id,))
    conn.commit()
    return StoryEvent(
        id=ev_id,
        ts=ts,
        epoch=epoch,
        kind="action",
        text=label,
        mood=mood,
        deltas=deltas,
        tags=["action"],
        option_ref=option_id,
    )


def story_tick(conn: sqlite3.Connection) -> StoryEvent:
    state = get_story_state(conn)
    resources = dict(state.resources)
    resources["energie"] = max(0, resources.get("energie", 0) - 1)
    inspiration_before = state.resources.get("inspiration", 0)
    if (
        resources.get("inspiration", 0) < INSP_TICK_THRESHOLD
        and resources.get("energie", 0) > INSP_MIN_ENERGY
    ):
        resources["inspiration"] = resources.get("inspiration", 0) + 1
    if resources.get("inspiration", 0) > INSP_SOFT_CAP:
        resources["inspiration"] = INSP_SOFT_CAP
    epoch = state.epoch + 1
    mood = state.mood
    if resources.get("energie", 0) < 30:
        mood = "strained"
    new_arc = eval_arc(resources, state.arc)
    ts = _story_now()
    conn.execute(
        "UPDATE story_state SET ts=?, epoch=?, mood=?, arc=?, resources=? WHERE id=1",
        (ts, epoch, mood, new_arc, json.dumps(resources)),
    )
    text = "Zeit vergeht. Eine stille Verschiebung im inneren Raum."
    deltas_tick: dict[str, int] = {"energie": -1}
    inspiration_delta = resources.get("inspiration", 0) - inspiration_before
    if inspiration_delta > 0:
        deltas_tick["inspiration"] = inspiration_delta
    ev_id = f"sev_{int(ts * 1000)}"
    conn.execute(
        "INSERT INTO story_events(ts, epoch, kind, text, mood, deltas, tags, option_ref) VALUES (?,?,?,?,?,?,?,?)",
        (
            ts,
            epoch,
            "tick",
            text,
            mood,
            json.dumps(deltas_tick),
            json.dumps(["tick"]),
            None,
        ),
    )
    maybe_arc_shift(conn, state.arc, new_arc, epoch, mood)
    refresh_story_options(
        conn,
        StoryState(
            epoch=epoch,
            mood=mood,
            arc=new_arc,
            resources=resources,
            options=[],
        ),
    )
    conn.commit()
    return StoryEvent(
        id=ev_id,
        ts=ts,
        epoch=epoch,
        kind="tick",
        text=text,
        mood=mood,
        deltas=deltas_tick,
        tags=["tick"],
        option_ref=None,
    )


def fetch_events(conn: sqlite3.Connection, limit: int = 100) -> list[StoryEvent]:
    cur = conn.execute(
        "SELECT id, ts, epoch, kind, text, mood, deltas, tags, option_ref FROM story_events ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    out: list[StoryEvent] = []
    for row in cur.fetchall():
        rid, ts, epoch, kind, text, mood, deltas_json, tags_json, option_ref = row
        deltas: dict[str, int] | None
        if deltas_json:
            try:
                raw = json.loads(deltas_json)
                if isinstance(raw, dict):
                    deltas = {
                        str(k): int(v)
                        for k, v in raw.items()
                        if isinstance(v, (int, float))
                    }
                else:
                    deltas = None
            except Exception:
                deltas = None
        else:
            deltas = None
        tags: list[str] = []
        if tags_json:
            try:
                raw_tags = json.loads(tags_json)
                if isinstance(raw_tags, list):
                    tags = [str(t) for t in raw_tags]
            except Exception:
                tags = []
        out.append(
            StoryEvent(
                id=str(rid),
                ts=ts,
                epoch=epoch,
                kind=kind,
                text=text,
                mood=mood,
                deltas=deltas,
                tags=tags,
                option_ref=option_ref,
            )
        )
    return out


__all__ = [
    "StoryOption",
    "StoryEvent",
    "StoryState",
    "Companion",
    "CompanionCreate",
    "Buff",
    "BuffCreate",
    "Skill",
    "SkillCreate",
    "eval_arc",
    "maybe_arc_shift",
    "get_story_state",
    "list_story_options",
    "generate_story_options",
    "persist_options",
    "refresh_story_options",
    "apply_story_option",
    "story_tick",
    "fetch_events",
]
