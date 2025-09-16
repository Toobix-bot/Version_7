from __future__ import annotations
import os, json
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List

router = APIRouter()

class HelpTopic(BaseModel):
    id: str
    title: str
    path: str
    size: Optional[int] = None

class HelpTopicsResponse(BaseModel):
    topics: List[HelpTopic]

class HelpReadResponse(BaseModel):
    id: str
    title: Optional[str]
    content: str

DOCS = [
    {"id":"overview","title":"Überblick","path":"docs/OVERVIEW.md"},
    {"id":"story","title":"Story-Modul","path":"docs/STORY.md"},
    {"id":"idle","title":"Idle-Modul","path":"docs/IDLE.md"},
    {"id":"examples","title":"Beispiele","path":"docs/USAGE_EXAMPLES.md"},
]

@router.get("/help/topics", response_model=HelpTopicsResponse)
async def help_topics() -> HelpTopicsResponse:
    items: List[HelpTopic] = []
    for d in DOCS:
        size = None
        try:
            st = os.stat(d["path"])  # relative to project root
            size = st.st_size
        except Exception:
            pass
        items.append(HelpTopic(id=d["id"], title=d["title"], path=d["path"], size=size))
    return HelpTopicsResponse(topics=items)

@router.get("/help/read", response_model=HelpReadResponse)
async def help_read(id: str) -> HelpReadResponse:
    m = next((x for x in DOCS if x["id"] == id), None)
    if not m:
        return HelpReadResponse(id=id, title=None, content="")
    try:
        with open(m["path"], "r", encoding="utf-8") as f:
            return HelpReadResponse(id=id, title=m["title"], content=f.read())
    except Exception:
        return HelpReadResponse(id=id, title=m["title"], content="")
