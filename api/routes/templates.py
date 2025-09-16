"""Routes for listing policy templates available to the wizard.

Provides a minimal GET endpoint to enumerate YAML templates under policies/templates.
"""
from __future__ import annotations

import os
from typing import Any
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


# ---------- Render / Apply from Template ----------

class RenderFromTemplateRequest(BaseModel):
    name: str
    overrides: dict[str, Any] | None = None


class RenderFromTemplateResponse(BaseModel):
    status: str
    content: str
    policy: dict[str, Any]


def _deep_merge(base: Any, overrides: Any) -> Any:
    if isinstance(base, dict) and isinstance(overrides, dict):
        out = dict(base)
        for k, v in overrides.items():
            if k in out:
                out[k] = _deep_merge(out[k], v)
            else:
                out[k] = v
        return out
    return overrides if overrides is not None else base


@router.post("/policy/render-from-template", response_model=RenderFromTemplateResponse)
async def render_from_template(req: RenderFromTemplateRequest) -> RenderFromTemplateResponse:
    import yaml
    # load template
    TEMPLATE_DIR = _template_dir()
    if "/" in req.name or ".." in req.name or "\\" in req.name:
        raise ValueError("invalid_template_name")
    pth = os.path.join(TEMPLATE_DIR, req.name)
    with open(pth, "r", encoding="utf-8") as f:
        tpl = yaml.safe_load(f) or {}
    merged = _deep_merge(tpl, req.overrides or {})
    # validate
    from policy.model import Policy as _Pol
    new_pol = _Pol.model_validate(merged)
    content = yaml.safe_dump(merged, sort_keys=False, allow_unicode=True)
    return RenderFromTemplateResponse(status="ok", content=content, policy=new_pol.model_dump())


class ApplyFromTemplateRequest(BaseModel):
    name: str
    overrides: dict[str, Any] | None = None
    dry_run: bool | None = True


class ApplyFromTemplateResponse(BaseModel):
    status: str
    path: str | None = None
    policy: dict[str, Any] | None = None
    message: str | None = None


@router.post("/policy/apply-from-template", response_model=ApplyFromTemplateResponse)
async def apply_from_template(req: ApplyFromTemplateRequest) -> ApplyFromTemplateResponse:
    import yaml
    TEMPLATE_DIR = _template_dir()
    if "/" in req.name or ".." in req.name or "\\" in req.name:
        return ApplyFromTemplateResponse(status="error", message="invalid_template_name")
    pth = os.path.join(TEMPLATE_DIR, req.name)
    if not os.path.isfile(pth):
        return ApplyFromTemplateResponse(status="error", message="template_not_found")
    with open(pth, "r", encoding="utf-8") as f:
        tpl = yaml.safe_load(f) or {}
    merged = _deep_merge(tpl, req.overrides or {})
    # validate
    from policy.model import Policy as _Pol
    try:
        new_pol = _Pol.model_validate(merged)
    except Exception as e:  # noqa: BLE001
        return ApplyFromTemplateResponse(status="invalid", message=str(e))
    content = yaml.safe_dump(merged, sort_keys=False, allow_unicode=True)
    if req.dry_run:
        return ApplyFromTemplateResponse(status="validated", policy=new_pol.model_dump())
    # persist to POLICY_DIR/policy.yaml
    policy_dir = os.getenv("POLICY_DIR", "policies")
    os.makedirs(policy_dir, exist_ok=True)
    target = os.path.join(policy_dir, "policy.yaml")
    with open(target, "w", encoding="utf-8") as f:
        f.write(content if content.endswith("\n") else content + "\n")
    return ApplyFromTemplateResponse(status="applied", path=target, policy=new_pol.model_dump())
