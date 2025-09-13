import os, json
import pytest
from httpx import AsyncClient, ASGITransport

import api.app as appmod  # type: ignore
from api.app import app, _ensure_db  # type: ignore

API_KEY = os.getenv("TEST_API_KEY", "test")

def _db():  # helper to guarantee db init and return connection
    _ensure_db()
    return appmod.db_conn  # type: ignore

@pytest.mark.asyncio
async def test_mentor_risk_and_tag():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/story/state", headers={"X-API-Key": API_KEY})
        assert r.status_code == 200
        comps = (await ac.get("/story/meta/companions", headers={"X-API-Key": API_KEY})).json()
        if not any(c['name'].lower().startswith('mentor') for c in comps):
            await ac.post("/story/meta/companions", json={"name":"Mentor X","archetype":"weise","mood":"ruhig","stats":{}}, headers={"X-API-Key": API_KEY})
        opts = (await ac.post("/story/options/regen", headers={"X-API-Key": API_KEY})).json()
        assert isinstance(opts, list) and opts
        for o in opts:
            assert 'mentor' in o['tags']
            assert o['risk'] >= 0

@pytest.mark.asyncio
async def test_xp_curve_and_levelup_presence():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        st = (await ac.get("/story/state", headers={"X-API-Key": API_KEY})).json()
        lvl = st['resources'].get('level',1)
        st['resources']['erfahrung'] = int((lvl ** 1.4) * 80) + 5
        conn = _db()
        conn.execute("UPDATE story_state SET resources=? WHERE id=1", (json.dumps(st['resources']),))
        conn.commit()
        opts = (await ac.post("/story/options/regen", headers={"X-API-Key": API_KEY})).json()
        assert any('levelup' in o['tags'] for o in opts)

@pytest.mark.asyncio
async def test_inspiration_tick_gating():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        st = (await ac.get("/story/state", headers={"X-API-Key": API_KEY})).json()
        st['resources']['inspiration'] = 140
        st['resources']['energie'] = 50
        conn = _db()
        conn.execute("UPDATE story_state SET resources=? WHERE id=1", (json.dumps(st['resources']),))
        conn.commit()
        before = st['resources']['inspiration']
        _ = (await ac.post("/story/advance", headers={"X-API-Key": API_KEY})).json()
        st2 = (await ac.get("/story/state", headers={"X-API-Key": API_KEY})).json()
        after = st2['resources']['inspiration']
        assert after == before

@pytest.mark.asyncio
async def test_story_export_and_reset_endpoints():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/story/export", headers={"X-API-Key": API_KEY})
        assert r.status_code == 200
        data = r.json()
        assert 'state' in data and 'events' in data
        st = data['state']
        st['resources']['energie'] = 5
        conn = _db()
        conn.execute("UPDATE story_state SET resources=? WHERE id=1", (json.dumps(st['resources']),))
        conn.commit()
        rr = await ac.post("/story/reset", headers={"X-API-Key": API_KEY})
        assert rr.status_code == 200
        st_after = (await ac.get("/story/state", headers={"X-API-Key": API_KEY})).json()
        assert st_after['resources'].get('energie',0) >= 50
