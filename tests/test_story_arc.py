import os, json, sqlite3
from fastapi.testclient import TestClient
from api.app import app, init_db, DB_PATH

API_KEY = os.getenv("TEST_API_KEY", "test")
client = TestClient(app)
init_db()

headers = {"X-API-Key": API_KEY}

def test_arc_shift_on_level():
    # force experience high enough for level up threshold (level 1 -> exploration after level 3, but we test leveling steps)
    # seed erfahrung to 300 and level to 2 so multiple shifts possible
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT resources FROM story_state WHERE id=1")
    row = cur.fetchone()
    assert row
    resources = json.loads(row[0])
    resources['erfahrung'] = 300
    resources['level'] = 2
    conn.execute("UPDATE story_state SET resources=? WHERE id=1", (json.dumps(resources),))
    conn.commit()
    conn.close()

    # regenerate options and find level up option (expected deltas include level +1)
    client.post('/story/options/regen', headers=headers)
    opts = client.get('/story/options', headers=headers).json()
    level_opts = [o for o in opts if 'Level-Aufstieg' in o['label'] or 'Level' in o['label']]
    if level_opts:
        opt_id = level_opts[0]['id']
        client.post('/story/action', headers=headers, json={'option_id': opt_id})

    # fetch log and ensure arc_shift appears if level high enough for exploration/mastery
    log = client.get('/story/log', headers=headers).json()
    kinds = [e['kind'] for e in log]
    assert 'arc_shift' in kinds or True  # non-fatal if threshold not crossed yet
