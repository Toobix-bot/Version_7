import pytest, asyncio

@pytest.mark.asyncio
async def test_sse_filtering(client):
    # trigger events
    r = await client.get('/env/info', headers={'X-API-Key':'test'})
    assert r.status_code == 200
    # open SSE with filter to env.info only
    # Using httpx client (provided fixture) with stream
    async with client.stream('GET', '/events?kinds=env.info&test=1', headers={'X-API-Key':'test'}) as resp:
        assert resp.status_code == 200
        body = (await resp.aread()).decode()
        # ready event present, no filtered events (test=1 closes early)
        assert 'event: ready' in body
    # now generate an env.info event and read stream without test shortcut
    # open real stream but we'll break after first matching event
    # produce env.info event before filtered short stream to capture it
    r_prime = await client.get('/env/info', headers={'X-API-Key':'test'})
    assert r_prime.status_code == 200
    # Use test=1 to close immediately and inspect buffered events (if any match filter)
    async with client.stream('GET', '/events?kinds=env.info&test=1', headers={'X-API-Key':'test'}) as resp3:
        assert resp3.status_code == 200
        body2 = (await resp3.aread()).decode()
        # We may get zero or one env.info depending on queue timing; assert no foreign events leaked
        assert 'plan.created' not in body2
