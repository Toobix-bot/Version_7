from __future__ import annotations
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from ..services.advisor import AdvisorQuestion, AdvisorAnswer, AdvisorTopic, KB

router = APIRouter()

class AdvisorTopicsResponse(BaseModel):
    topics: List[AdvisorTopic]

@router.get("/advisor/topics", response_model=AdvisorTopicsResponse)
async def advisor_topics() -> AdvisorTopicsResponse:
    return AdvisorTopicsResponse(topics=KB.topics)

@router.post("/advisor/ask", response_model=AdvisorAnswer)
async def advisor_ask(q: AdvisorQuestion) -> AdvisorAnswer:
    return KB.search(q.query, q.topic)
