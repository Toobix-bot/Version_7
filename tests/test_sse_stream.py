import pytest
import anyio
import httpx
from api.app import app

API_H = {"X-API-Key": "test"}

@pytest.mark.asyncio
async def test_events_stream_minimum_messages():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        await c.post('/act', headers=API_H, json={'agent_id':'sse1','action':'ping'})
        async with c.stream('GET','/events', headers=API_H) as resp:
            assert resp.status_code == 200
            saw_ready = False
            with anyio.fail_after(5):
                async for chunk in resp.aiter_text():
                    if 'event: ready' in chunk:
                        saw_ready = True
                        break
            assert saw_ready

@pytest.mark.asyncio
async def test_events_iter_stream():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        await c.post('/act', headers=API_H, json={'agent_id':'idle','action':'ping'})
        seen_ready = False
        heartbeats = 0
        # Stream with timeout guard
        async with c.stream('GET','/events', headers=API_H) as resp:
            assert resp.status_code == 200
            assert 'text/event-stream' in resp.headers.get('content-type','')
            with anyio.fail_after(5):
                async for chunk in resp.aiter_text():
                    if 'event: ready' in chunk:
                        seen_ready = True
                    if chunk.strip().startswith(':'):
                        heartbeats += 1
                    if seen_ready and heartbeats >= 1:
                        break
        assert seen_ready
        assert heartbeats >= 1
