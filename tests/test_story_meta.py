import os, json
from fastapi.testclient import TestClient
from api.app import app, init_db

API_KEY = os.getenv("TEST_API_KEY", "test")
client = TestClient(app)
init_db()
headers = {"X-API-Key": API_KEY}

def test_meta_seed_presence():
    r_comp = client.get('/story/meta/companions', headers=headers)
    assert r_comp.status_code == 200
    comps = r_comp.json(); assert isinstance(comps, list) and len(comps) >= 1
    r_skills = client.get('/story/meta/skills', headers=headers)
    assert r_skills.status_code == 200
    skills = r_skills.json(); assert isinstance(skills, list) and len(skills) >= 1
    r_buffs = client.get('/story/meta/buffs', headers=headers)
    assert r_buffs.status_code == 200
    buffs = r_buffs.json(); assert isinstance(buffs, list) and len(buffs) >= 1


def test_meta_create_companion_and_reflect_state():
    r_new = client.post('/story/meta/companions', headers=headers, json={"name":"Test Gefährte","archetype":"test","mood":"neutral","stats":{"wert":1}})
    assert r_new.status_code == 201
    comp = r_new.json(); assert comp['name'] == 'Test Gefährte'
    # state should list companions (indirect check only ensures endpoint still works)
    st = client.get('/story/state', headers=headers).json()
    assert any(c.get('name') == 'Test Gefährte' for c in st.get('companions', []))


def test_meta_create_skill_and_buff():
    r_skill = client.post('/story/meta/skills', headers=headers, json={"name":"forschung","category":"wissen"})
    assert r_skill.status_code == 201
    skill = r_skill.json(); assert skill['name'] == 'forschung'
    r_buff = client.post('/story/meta/buffs', headers=headers, json={"label":"testkraft","kind":"test","magnitude":2,"expires_at":None,"meta":{"notiz":"temp"}})
    assert r_buff.status_code == 201
    buff = r_buff.json(); assert buff['label'] == 'testkraft'
    st = client.get('/story/state', headers=headers).json()
    assert any(sk.get('name') == 'forschung' for sk in st.get('skills', []))
    assert any(bf.get('label') == 'testkraft' for bf in st.get('buffs', []))
