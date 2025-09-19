import os
from httpx import AsyncClient, ASGITransport
import pytest

from api.app import app

@pytest.mark.asyncio
async def test_register_login_me_settings_flow():
    # Use a unique email per test run
    email = f"user_{os.getpid()}@example.com"
    password = "secret123"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # register
        r_reg = await ac.post("/auth/register", json={"email": email, "password": password})
        assert r_reg.status_code in (200, 201, 409)  # 409 if re-run
        # login
        r_login = await ac.post("/auth/login", json={"email": email, "password": password})
        assert r_login.status_code == 200
        # cookie should be set by ASGITransport response
        # /me should be accessible
        r_me = await ac.get("/me")
        assert r_me.status_code == 200
        me = r_me.json()
        assert me.get("user", {}).get("email") == email
        # read settings
        r_get = await ac.get("/me/settings")
        assert r_get.status_code == 200
        # patch settings
        r_patch = await ac.patch("/me/settings", json={"role": "beginner", "mode": "productive", "theme": "light", "density": "comfy", "toggles": {"showQuests": True}, "is_public": True})
        assert r_patch.status_code == 200
        body = r_patch.json()
        assert body.get("role") == "beginner"
        assert body.get("mode") == "productive"
        assert body.get("theme") == "light"
        assert body.get("density") == "comfy"
        assert body.get("is_public") is True
        # logout
        r_lo = await ac.post("/auth/logout")
        assert r_lo.status_code == 200
        # /me should now be 401
        r_me2 = await ac.get("/me")
        assert r_me2.status_code == 401
