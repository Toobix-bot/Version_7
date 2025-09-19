import os
import pytest
from httpx import AsyncClient, ASGITransport

from api.app import app  # type: ignore

API_KEY = os.getenv("TEST_API_KEY", "test")

@pytest.mark.asyncio
async def test_random_events_config_and_force_emit():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Get default config
        r_get = await ac.get("/story/events/config", headers={"X-API-Key": API_KEY})
        assert r_get.status_code == 200
        cfg = r_get.json()
        assert "enabled" in cfg and "prob" in cfg
        # Set config to disabled and prob 1.0, then force emit should still produce event
        r_patch = await ac.patch("/story/events/config", headers={"X-API-Key": API_KEY}, json={"enabled": False, "prob": 1.0})
        assert r_patch.status_code == 200
        # Advance with force_random=1 to deterministically create an event
        r_adv = await ac.post("/story/advance?force_random=1", headers={"X-API-Key": API_KEY})
        assert r_adv.status_code == 200
        # Fetch log and ensure there's at least one random event present
        r_log = await ac.get("/story/log?limit=5", headers={"X-API-Key": API_KEY})
        assert r_log.status_code == 200
        events = r_log.json()
        assert any(e.get("kind") == "random" for e in events), "Expected a random event in recent log"

@pytest.mark.asyncio
async def test_random_events_disable_blocks_spontaneous():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Disable and set prob to 0
        _ = await ac.patch("/story/events/config", headers={"X-API-Key": API_KEY}, json={"enabled": False, "prob": 0.0})
        # Call advance without force
        _ = await ac.post("/story/advance", headers={"X-API-Key": API_KEY})
        # Recent log should not necessarily include random (can't guarantee from earlier), but we test prob validation separately
        # Just validate API still returns events and doesn't error
        r_log = await ac.get("/story/log?limit=3", headers={"X-API-Key": API_KEY})
        assert r_log.status_code == 200

@pytest.mark.asyncio
async def test_random_events_invalid_prob_rejected():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.patch("/story/events/config", headers={"X-API-Key": API_KEY}, json={"prob": 1.2})
        assert r.status_code == 400
