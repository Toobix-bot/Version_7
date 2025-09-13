import os
import json
from fastapi.testclient import TestClient
from api.app import app, init_db

API_KEY = os.getenv("TEST_API_KEY", "test")

client = TestClient(app)

# Ensure DB initialized once
init_db()

headers = {"X-API-Key": API_KEY}

def test_story_state_roundtrip():
    r = client.get("/story/state", headers=headers)
    assert r.status_code == 200
    data = r.json()
    assert "resources" in data and isinstance(data["resources"], dict)
    assert "energie" in data["resources"]

def test_story_options_and_action_free_text():
    r_opt = client.get("/story/options", headers=headers)
    assert r_opt.status_code == 200
    options = r_opt.json()
    # free text action
    r_action = client.post("/story/action", headers=headers, json={"free_text": "kurzer test gedanke"})
    assert r_action.status_code == 200
    ev = r_action.json()
    assert ev["kind"] == "action"

def test_story_advance_tick():
    r_tick = client.post("/story/advance", headers=headers)
    assert r_tick.status_code == 200
    ev = r_tick.json()
    assert ev["kind"] == "tick"

