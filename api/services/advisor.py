from __future__ import annotations
import json, os, re
from typing import Any, List, Dict
from pydantic import BaseModel

KB_PATH = os.getenv("ADVISOR_KB", "advisor_kb.json")

class AdvisorQuestion(BaseModel):
    query: str
    topic: str | None = None  # e.g., story|idle|plan|policy|events

class AdvisorAnswer(BaseModel):
    answer: str
    sources: List[str] = []
    related: List[str] = []

class AdvisorTopic(BaseModel):
    id: str
    title: str
    description: str

_DEFAULT_TOPICS = [
    AdvisorTopic(id="story", title="Story", description="Narrative Modul: Entscheidungen, Ereignisse, Skills."),
    AdvisorTopic(id="idle", title="Idle", description="Hintergrund‑Fortschritt: Quests, Auto‑Ticks."),
    AdvisorTopic(id="plan", title="Plan", description="Ideen, Aufgaben und PR‑Entwürfe."),
    AdvisorTopic(id="policy", title="Policy", description="Richtlinien, Wizard und Vorlagen."),
]

class _KB:
    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []
        self.topics: List[AdvisorTopic] = list(_DEFAULT_TOPICS)

    def load(self) -> None:
        if os.path.exists(KB_PATH):
            try:
                with open(KB_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.entries = data.get("entries", [])
                    raw_topics = data.get("topics") or []
                    if isinstance(raw_topics, list) and raw_topics:
                        self.topics = [AdvisorTopic(**t) for t in raw_topics if isinstance(t, dict)]
            except Exception:
                pass

    def search(self, q: str, topic: str | None) -> AdvisorAnswer:
        ql = q.lower().strip()
        hits: List[tuple[int, Dict[str, Any]]] = []
        for e in self.entries:
            text = str(e.get("text", ""))
            etopic = e.get("topic")
            if topic and etopic and topic != etopic:
                continue
            score = 0
            tl = text.lower()
            for term in set(re.findall(r"[a-zA-ZäöüÄÖÜß0-9_]{3,}", ql)):
                if term in tl:
                    score += 1
            if score:
                hits.append((score, e))
        hits.sort(key=lambda x: x[0], reverse=True)
        if not hits:
            return AdvisorAnswer(answer="Ich habe nichts Passendes in meinem Wissen gefunden. Schau in /help/topics vorbei.")
        best = hits[0][1]
        answer = best.get("answer") or best.get("text") or ""
        return AdvisorAnswer(answer=answer, sources=[best.get("source") or "kb"], related=[h[1].get("id","kb") for h in hits[1:4]])

KB = _KB()
KB.load()

__all__ = ["AdvisorQuestion","AdvisorAnswer","AdvisorTopic","KB"]
