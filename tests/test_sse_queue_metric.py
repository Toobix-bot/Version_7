import pytest

API_H = {"X-API-Key": "test"}

@pytest.mark.asyncio
async def test_metrics_exposes_sse_queue_dropped(client):  # type: ignore
    # prime metrics by hitting /events in test mode so app initializes SSE bits
    r = await client.get('/events?test=1', headers=API_H)
    assert r.status_code == 200
    m = await client.get('/metrics')
    assert m.status_code == 200
    body = m.text
    assert 'sse_queue_dropped_total' in body
