"""Core Story / Narrative logic extracted from app.py for clarity.
Minimal refactor: pure functions + dataclasses (pydantic BaseModel) reused by app.
"""
from __future__ import annotations
import json, sqlite3, time
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
    cur = conn.execute("SELECT id,label,rationale,risk,expected,tags,expires_at FROM story_options ORDER BY created_at DESC")
    out: list[StoryOption] = []
    now = _story_now()
    for row in cur.fetchall():
        oid,label,rationale,risk,expected_json,tags_json,expires_at = row
        if expires_at and expires_at < now: continue
        expected: dict[str,int] | None = None
        if expected_json:
            try:
                raw = json.loads(expected_json);
                if isinstance(raw, dict):
                    expected = {str(k):int(v) for k,v in raw.items() if isinstance(v,(int,float))}
            except Exception: expected=None
        tags: list[str] = []
        if tags_json:
            try:
                raw_t=json.loads(tags_json); tags=[str(t) for t in raw_t] if isinstance(raw_t,list) else []
            except Exception: tags=[]
        out.append(StoryOption(id=oid,label=label,rationale=rationale,risk=risk,expected=expected,tags=tags,expires_at=expires_at))
    return out

def generate_story_options(state: StoryState) -> list[StoryOption]:
    opts: list[StoryOption] = []
    r=state.resources
    if r.get("energie",0) < 40:
        opts.append(StoryOption(id=f"opt_rest_{int(_story_now())}",label="Meditieren und Energie sammeln",rationale="Niedrige Energie erkannt",risk=0,expected={"energie":15,"inspiration":2},tags=["resource:energie"]))
    if r.get("inspiration",0) > 10 and r.get("wissen",0) < 50:
        opts.append(StoryOption(id=f"opt_write_{int(_story_now())}",label="Ideen schriftlich strukturieren",rationale="Inspiration in Wissen umwandeln",risk=1,expected={"inspiration":-5,"wissen":8,"erfahrung":5},tags=["convert","resource:wissen"]))
    if r.get("erfahrung",0) >= (r.get("level",1)*100):
        xp_cost = r.get("level",1)*100
        opts.append(StoryOption(id=f"opt_level_{int(_story_now())}",label="Reflektion und Level-Aufstieg",rationale="Erfahrungsschwelle erreicht",risk=1,expected={"erfahrung":-xp_cost,"level":1,"stabilitaet":5},tags=["levelup"]))
    if not opts:
        opts.append(StoryOption(id=f"opt_explore_{int(_story_now())}",label="Neuen Gedankenpfad erkunden",rationale="Kein dringendes Bedürfnis",risk=1,expected={"inspiration":5,"energie":-5,"erfahrung":3},tags=["explore"]))
    return opts

def persist_options(conn: sqlite3.Connection, options: list[StoryOption]) -> None:
    now=_story_now()
    for o in options:
        conn.execute("INSERT OR REPLACE INTO story_options(id, created_at, label, rationale, risk, expected, tags, expires_at) VALUES (?,?,?,?,?,?,?,?)",
                     (o.id, now, o.label, o.rationale, o.risk, json.dumps(o.expected) if o.expected else None, json.dumps(o.tags), o.expires_at))
    conn.commit()

def refresh_story_options(conn: sqlite3.Connection, state: StoryState) -> list[StoryOption]:
    conn.execute("DELETE FROM story_options")
    opts = generate_story_options(state)
    persist_options(conn, opts)
    return opts

def apply_story_option(conn: sqlite3.Connection, state: StoryState, option_id: str) -> StoryEvent:
    cur = conn.execute("SELECT id,label,rationale,risk,expected FROM story_options WHERE id=?", (option_id,))
    row = cur.fetchone()
    if not row:
        raise ValueError("option_not_found")
    _,label,_rationale,_risk,expected_json = row
    expected: dict[str,int] = {}
    if expected_json:
        try:
            raw = json.loads(expected_json)
            if isinstance(raw, dict):
                for k,v in raw.items():
                    if isinstance(v,(int,float)): expected[str(k)] = int(v)
        except Exception: pass
    for k,v in expected.items():
        state.resources[k] = state.resources.get(k,0) + v
    # energy decay floor
    if state.resources.get("energie",0) < 0: state.resources["energie"] = 0
    conn.execute("UPDATE story_state SET resources=?, ts=? WHERE id=1", (json.dumps(state.resources), _story_now()))
    # record event
    eid = f"ev_{int(_story_now()*1000)}"
    txt = label
    conn.execute("INSERT INTO story_events(ts, epoch, kind, text, mood, deltas, tags, option_ref) VALUES (?,?,?,?,?,?,?,?)",
                 (_story_now(), state.epoch, "action", txt, state.mood, json.dumps(expected), json.dumps(["action"]), option_id))
    conn.execute("DELETE FROM story_options WHERE id=?", (option_id,))
    conn.commit()
    return StoryEvent(id=eid, ts=_story_now(), epoch=state.epoch, kind="action", text=txt, mood=state.mood, deltas=expected, tags=["action"], option_ref=option_id)

def story_tick(conn: sqlite3.Connection) -> StoryEvent:
    st = get_story_state(conn)
    st.epoch += 1
    st.resources["energie"] = max(0, st.resources.get("energie",0)-1)
    new_arc = eval_arc(st.resources, st.arc)
    arc_evt = maybe_arc_shift(conn, st.arc, new_arc, st.epoch, st.mood)
    st.arc = new_arc
    conn.execute("UPDATE story_state SET epoch=?, resources=?, ts=?, arc=? WHERE id=1", (st.epoch, json.dumps(st.resources), _story_now(), st.arc))
    eid=f"tick_{int(_story_now()*1000)}"
    conn.execute("INSERT INTO story_events(ts, epoch, kind, text, mood, deltas, tags, option_ref) VALUES (?,?,?,?,?,?,?,?)",
                 (_story_now(), st.epoch, "tick", "Zeit vergeht", st.mood, json.dumps({}), json.dumps(["tick"]), None))
    conn.commit()
    if arc_evt: return arc_evt
    return StoryEvent(id=eid, ts=_story_now(), epoch=st.epoch, kind="tick", text="Zeit vergeht", mood=st.mood, deltas={}, tags=["tick"], option_ref=None)

def fetch_events(conn: sqlite3.Connection, limit: int = 100) -> list[StoryEvent]:
    cur = conn.execute("SELECT id, ts, epoch, kind, text, mood, deltas, tags, option_ref FROM story_events ORDER BY id DESC LIMIT ?", (limit,))
    out: list[StoryEvent] = []
    for row in cur.fetchall():
        rid,ts,epoch,kind,text,mood,deltas_json,tags_json,option_ref = row
        try: deltas = json.loads(deltas_json) if deltas_json else None
        except Exception: deltas=None
        try: tags = json.loads(tags_json) if tags_json else []
        except Exception: tags=[]
        out.append(StoryEvent(id=str(rid), ts=ts, epoch=epoch, kind=kind, text=text, mood=mood, deltas=deltas, tags=tags, option_ref=option_ref))
    return out
