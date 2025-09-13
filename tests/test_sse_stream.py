import pytest
import anyio
API_H = {"X-API-Key": "test"}

@pytest.mark.asyncio
async def test_events_stream_minimum_messages(client):  # type: ignore
    await client.post('/act', headers=API_H, json={'agent_id':'sse1','action':'ping'})
    # Stream ends immediately in test mode; simple GET sufficient
    r = await client.get('/events?test=1', headers=API_H, timeout=5)
    assert r.status_code == 200
    assert b'event: ready' in r.content

@pytest.mark.asyncio
async def test_events_iter_stream(client):  # type: ignore
    await client.post('/act', headers=API_H, json={'agent_id':'idle','action':'ping'})
    r = await client.get('/events?test=1', headers=API_H, timeout=5)
    assert r.status_code == 200
    assert 'text/event-stream' in r.headers.get('content-type','')
    assert b'event: ready' in r.content
