from fastapi.testclient import TestClient
from api.app import app

def test_smoke_openapi():
    c = TestClient(app)
    r = c.get('/openapi.json')
    assert r.status_code == 200
    data = r.json()
    assert 'components' in data and 'securitySchemes' in data['components']

def test_smoke_metrics():
    c = TestClient(app)
    r = c.get('/metrics', headers={'X-API-Key':'test'})
    assert r.status_code == 200
    text = r.text
    assert '# HELP' in text or '# TYPE' in text

def test_smoke_plan_created():
    c = TestClient(app)
    payload = {"intent":"sync smoke plan","context":"demo","target_paths":["plans/sync_smoke.txt"]}
    r = c.post('/plan', json=payload, headers={'X-API-Key':'test'})
    assert r.status_code == 200
    body = r.json()
    assert body.get('variants') and isinstance(body['variants'], list)
