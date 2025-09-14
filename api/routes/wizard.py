"""Wizard route exposing /policy/wizard via APIRouter.
Phase 1 extraction from monolithic app.
"""
from __future__ import annotations
from fastapi import APIRouter
from typing import Any
from ..services.policy_wizard import PolicyWizardRequest, PolicyWizardResponse, generate
from ..core.infra import emit_event  # event emission reused

router = APIRouter()

@router.post("/policy/wizard", response_model=PolicyWizardResponse)
async def policy_wizard(req: PolicyWizardRequest) -> PolicyWizardResponse:  # auth injected via app.include_router dependencies
    resp = generate(req)
    # event emission remains optional (not previously emitted, but we add for observability)
    try:
        await emit_event("wizard.generated", {"template": req.template})
    except Exception:
        pass
    return resp
