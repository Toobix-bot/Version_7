import os, json, sqlite3
from fastapi.testclient import TestClient
from api.app import app, init_db, DB_PATH

API_KEY = os.getenv("TEST_API_KEY", "test")
client = TestClient(app)
init_db()

headers = {"X-API-Key": API_KEY}

def test_arc_shift_on_level():
    # 1. Set initial level below threshold (foundations) then jump to exploration threshold (>=3) and then mastery (>=10)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT resources FROM story_state WHERE id=1")
    row = cur.fetchone(); assert row
    resources = json.loads(row[0])
    resources['level'] = 1
    conn.execute("UPDATE story_state SET resources=? WHERE id=1", (json.dumps(resources),))
    conn.commit()
    conn.close()

    # trigger tick to capture baseline arc
    client.post('/story/advance', headers=headers)
    log1 = client.get('/story/log', headers=headers).json()
    baseline_arc_events = [e for e in log1 if e['kind']=='arc_shift']
    # may be empty (already foundations) but should not show exploration yet

    # 2. Raise to level 3 -> expect arc shift to exploration
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT resources FROM story_state WHERE id=1")
    row = cur.fetchone(); assert row
    resources = json.loads(row[0])
    resources['level'] = 3
    conn.execute("UPDATE story_state SET resources=? WHERE id=1", (json.dumps(resources),))
    conn.commit(); conn.close()
    client.post('/story/advance', headers=headers)
    log2 = client.get('/story/log', headers=headers).json()
    arc_shift_to_exploration = [e for e in log2 if e['kind']=='arc_shift' and 'exploration' in e['text']]
    assert arc_shift_to_exploration, "Erwarteter Arc-Shift zu exploration fehlte"

    # 3. Raise to level 10 -> expect arc shift to mastery
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT resources FROM story_state WHERE id=1")
    row = cur.fetchone(); assert row
    resources = json.loads(row[0])
    resources['level'] = 10
    conn.execute("UPDATE story_state SET resources=? WHERE id=1", (json.dumps(resources),))
    conn.commit(); conn.close()
    client.post('/story/advance', headers=headers)
    log3 = client.get('/story/log', headers=headers).json()
    arc_shift_to_mastery = [e for e in log3 if e['kind']=='arc_shift' and 'mastery' in e['text']]
    assert arc_shift_to_mastery, "Erwarteter Arc-Shift zu mastery fehlte"
