import asyncio
import pytest
from httpx import AsyncClient
from api.app import app

@pytest.mark.asyncio
async def test_events_stream_minimum_messages():
    async with AsyncClient(app=app, base_url="http://test") as c:
        # Trigger some activity before subscribing
        await c.post('/act', headers={'X-API-Key':'test'}, json={'agent_id':'sse1','action':'ping'})
        # Consume limited chunks to avoid hanging
        r = await c.get('/events', headers={'X-API-Key':'test'}, timeout=10)
        assert r.status_code == 200

@pytest.mark.asyncio
async def test_events_iter_stream():
    from httpx import ASGITransport
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        await c.post('/act', headers={'X-API-Key':'test'}, json={'agent_id':'idle','action':'ping'})
        # Use streaming interface
        async with c.stream('GET','/events', headers={'X-API-Key':'test'}, timeout=10) as resp:
            assert resp.status_code == 200
            assert 'text/event-stream' in resp.headers.get('content-type','')
            seen_ready = False
            count = 0
            async for chunk in resp.aiter_text():
                if 'event: ready' in chunk:
                    seen_ready = True
                if chunk.strip():
                    count += 1
                if count >= 5 and seen_ready:
                    break
            assert seen_ready
            assert count >= 1
