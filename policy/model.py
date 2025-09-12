from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, DirectoryPath, Field, conint, constr, confloat

class LLMPolicy(BaseModel):
    model: constr(strip_whitespace=True) = "llama-3.3-70b-versatile"  # type: ignore[type-var]
    temperature: confloat(ge=0, le=1) = 0.0  # type: ignore[type-var]
    max_tokens: conint(ge=1, le=8192) = 2048  # type: ignore[type-var]

class Policy(BaseModel):
    version: Literal["1"]
    allowed_dirs: list[DirectoryPath] = Field(min_length=1)
    deny_globs: list[str] = []
    max_diff_lines: conint(ge=0, le=5000) = 800  # type: ignore[type-var]
    llm: LLMPolicy = LLMPolicy()
