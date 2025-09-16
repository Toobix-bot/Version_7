import pytest
from typing import Any

@pytest.mark.asyncio
async def test_help_topics_and_read(client: Any):  # type: ignore
    r = await client.get('/help/topics', headers={'X-API-Key':'test'})
    assert r.status_code == 200
    data = r.json()
    assert 'topics' in data and isinstance(data['topics'], list) and len(data['topics']) >= 1
    # read an overview
    r2 = await client.get('/help/read', params={'id':'overview'}, headers={'X-API-Key':'test'})
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2.get('id') == 'overview'
    assert isinstance(d2.get('content'), str)

@pytest.mark.asyncio
async def test_advisor_basic(client: Any):  # type: ignore
    body = {"query": "policy wizard", "topic": "policy"}
    r = await client.post('/advisor/ask', json=body, headers={'X-API-Key':'test'})
    assert r.status_code == 200
    d = r.json()
    assert 'answer' in d and isinstance(d['answer'], str)
