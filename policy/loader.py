from __future__ import annotations
import yaml
from fastapi import HTTPException
from pydantic import ValidationError
from .model import Policy

def load_policy(path: str) -> Policy:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return Policy.model_validate(data)
    except ValidationError as e:  # pragma: no cover
        raise HTTPException(status_code=422, detail={"code": "policy_invalid", "errors": e.errors()})
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail={"code": "policy_not_found"})
