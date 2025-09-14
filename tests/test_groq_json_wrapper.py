import os
import pytest
from api.app import call_groq_json

@pytest.mark.asyncio
async def test_groq_json_wrapper_dummy_mode(monkeypatch):  # type: ignore
    # ensure no real key => dummy path
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    schema = {
        "title": "Untitled",
        "steps": {"a": 1, "b": 2},
        "meta": {"info": "x"},
        "flag": True
    }
    out = await call_groq_json("Erzeuge Plan Zusammenfassung", schema, timeout=0.1)
    # all top-level keys preserved
    for k in schema.keys():
        assert k in out
    # nested object filled
    assert isinstance(out["steps"], dict)
    assert "a" in out["steps"]
