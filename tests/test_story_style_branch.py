import os
import pytest
from httpx import AsyncClient, ASGITransport

from api.app import app  # type: ignore

API_KEY = os.getenv("TEST_API_KEY", "test")


@pytest.mark.asyncio
async def test_story_style_patch_and_invalid_temp():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # invalid temp
        r_bad = await ac.patch("/story/style", headers={"X-API-Key": API_KEY}, json={"temperature": 1.5})
        assert r_bad.status_code == 400
        # valid update
        r = await ac.patch("/story/style", headers={"X-API-Key": API_KEY}, json={"tone": "prägnanter Erzähler", "temperature": 0.3})
        assert r.status_code == 200
        body = r.json()
        assert body.get("status") == "ok"
        assert body.get("tone") and isinstance(body.get("tone"), str)
        assert 0.0 <= float(body.get("temperature", 0.0)) <= 1.0


@pytest.mark.asyncio
async def test_story_branch_alias_works():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Ensure state is available
        _ = await ac.get("/story/state", headers={"X-API-Key": API_KEY})
        # Use branch as free-text action
        r = await ac.post("/story/branch", headers={"X-API-Key": API_KEY}, json={"free_text": "demo aktion"})
        assert r.status_code == 200
        ev = r.json()
        assert ev.get("kind") in ("action", "tick", "arc_shift")
        assert "epoch" in ev
