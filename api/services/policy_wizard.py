"""Policy Wizard service layer.
Contains pure logic functions and Pydantic models for request/response.
"""
from __future__ import annotations
import os, re, json, time
from typing import Any, List, Dict
from fastapi import HTTPException
from pydantic import BaseModel
import yaml

POLICY_DIR = os.getenv("POLICY_DIR", "policies")
TEMPLATE_DIR = os.path.join(POLICY_DIR, "templates")

class PolicyWizardRequest(BaseModel):
    template: str
    goals: list[str] | None = []
    risk_profile: str | None = None  # low|medium|high
    overrides: dict[str, Any] | None = None
    annotate: bool | None = False

class PolicyWizardResponse(BaseModel):
    source_template: str
    policy: dict[str, Any]
    diff: dict[str, Any] | None = None
    notes: list[str] | None = None

def _load_template(name: str) -> dict[str, Any]:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "", name)
    path = os.path.join(TEMPLATE_DIR, f"{safe}.yaml")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail={"code": "template_not_found", "template": name})
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f) or {}
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
    if risk == "low":
        llm = doc.setdefault("llm", {})
        if isinstance(llm, dict):
            prev = llm.get("temperature")
            llm["temperature"] = 0.0
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

def generate(req: PolicyWizardRequest) -> PolicyWizardResponse:
    notes: list[str] = []
    base = _load_template(req.template)
    orig = json.loads(json.dumps(base))  # deep copy
    goals = [g.strip().lower() for g in (req.goals or []) if g.strip()]
    if goals:
        notes.append(f"goals: {', '.join(goals)}")
    _apply_risk_profile(base, req.risk_profile, notes)
    _merge_overrides(base, req.overrides, notes)
    if req.annotate:
        base.setdefault("_meta", {})
        if isinstance(base["_meta"], dict):
            base["_meta"].update({"goals": goals, "risk_profile": req.risk_profile, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    diff = _compute_diff(orig, base)
    return PolicyWizardResponse(source_template=req.template, policy=base, diff=diff, notes=notes)

__all__ = [
    "PolicyWizardRequest","PolicyWizardResponse","generate",
    "_load_template","_apply_risk_profile","_merge_overrides","_compute_diff"
]
