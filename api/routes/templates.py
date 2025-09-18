"""Routes for listing policy templates available to the wizard.

Provides a minimal GET endpoint to enumerate YAML templates under policies/templates.
"""
from __future__ import annotations

import os
from typing import Any, cast
from fastapi import APIRouter
from pydantic import BaseModel

def _template_dir() -> str:
    policy_dir = os.getenv("POLICY_DIR", "policies")
    return os.path.join(policy_dir, "templates")


router = APIRouter()


class TemplateInfo(BaseModel):
    name: str
    path: str
    size: int


class TemplatesList(BaseModel):
    templates: list[TemplateInfo]


@router.get("/policy/templates", response_model=TemplatesList)
async def list_policy_templates() -> TemplatesList:
    TEMPLATE_DIR = _template_dir()
    os.makedirs(TEMPLATE_DIR, exist_ok=True)
    items: list[TemplateInfo] = []
    try:
        for fname in sorted(os.listdir(TEMPLATE_DIR)):
            if not (fname.endswith(".yaml") or fname.endswith(".yml")):
                continue
            pth = os.path.join(TEMPLATE_DIR, fname)
            try:
                st = os.stat(pth)
                items.append(TemplateInfo(name=fname, path=pth, size=int(st.st_size)))
            except Exception:
                # skip unreadable files
                continue
    except FileNotFoundError:
        pass
    return TemplatesList(templates=items)


class TemplateRead(BaseModel):
    name: str
    path: str
    size: int
    content: str


@router.get("/policy/templates/{name}", response_model=TemplateRead)
async def read_policy_template(name: str) -> TemplateRead:
    # basic sanitization: disallow path traversal and only .yml/.yaml
    if "/" in name or ".." in name or "\\" in name:
        raise ValueError("invalid_template_name")
    if not (name.endswith(".yaml") or name.endswith(".yml")):
        raise FileNotFoundError("template_extension_not_allowed")
    TEMPLATE_DIR = _template_dir()
    os.makedirs(TEMPLATE_DIR, exist_ok=True)
    pth = os.path.join(TEMPLATE_DIR, name)
    if not os.path.isfile(pth):
        raise FileNotFoundError("template_not_found")
    st = os.stat(pth)
    with open(pth, "r", encoding="utf-8") as f:
        content = f.read()
    return TemplateRead(name=name, path=pth, size=int(st.st_size), content=content)


# -------- Render / Apply from template --------

class TemplateRenderRequest(BaseModel):
    name: str
    overrides: dict[str, Any] | None = None


class TemplateRenderResponse(BaseModel):
    status: str
    name: str
    content: str
    policy: dict[str, Any]


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


@router.post("/policy/render-from-template", response_model=TemplateRenderResponse)
async def render_from_template(req: TemplateRenderRequest) -> TemplateRenderResponse:
    import yaml  # lazy import
    # read template
    tpl = await read_policy_template(req.name)
    raw: Any = yaml.safe_load(tpl.content)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("template_not_object")
    base = cast(dict[str, Any], raw)
    merged = _deep_merge(base, req.overrides or {})
    # validate policy
    from policy.model import Policy as _Pol  # type: ignore
    try:
        pol = _Pol.model_validate(merged)
    except Exception as e:  # noqa: BLE001
        # bubble up as 422 by re-raising; FastAPI will format
        raise e
    content = yaml.safe_dump(pol.model_dump(), sort_keys=False, allow_unicode=True)
    return TemplateRenderResponse(status="ok", name=req.name, content=content, policy=pol.model_dump())


class TemplateApplyRequest(TemplateRenderRequest):
    dry_run: bool | None = False


class TemplateApplyResponse(BaseModel):
    status: str
    path: str | None = None
    name: str
    dry_run: bool | None = None
    policy: dict[str, Any]
    content: str


@router.post("/policy/apply-from-template", response_model=TemplateApplyResponse)
async def apply_from_template(req: TemplateApplyRequest) -> TemplateApplyResponse:
    # reuse render
    rendered = await render_from_template(TemplateRenderRequest(name=req.name, overrides=req.overrides))
    if req.dry_run:
        return TemplateApplyResponse(status="validated", path=None, name=req.name, dry_run=True, policy=rendered.policy, content=rendered.content)
    # persist to POLICY_DIR/policy.yaml
    policy_dir = os.getenv("POLICY_DIR", "policies")
    os.makedirs(policy_dir, exist_ok=True)
    target_path = os.path.join(policy_dir, "policy.yaml")
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(rendered.content if rendered.content.endswith("\n") else rendered.content + "\n")
    return TemplateApplyResponse(status="applied", path=target_path, name=req.name, dry_run=False, policy=rendered.policy, content=rendered.content)


# Note: Removed older duplicate implementations of render/apply endpoints to avoid route conflicts.
