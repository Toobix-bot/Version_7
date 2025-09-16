import os
import anyio
import pytest

API_H = {"X-API-Key": "test"}


@pytest.mark.asyncio
async def test_sse_emits_heartbeat(client):  # type: ignore
    # Speed up heartbeats for test
    os.environ["SSE_HEARTBEAT_INTERVAL"] = "0.2"
    # Open real stream and wait briefly for a heartbeat comment
    async with client.stream('GET', '/events', headers=API_H) as resp:
        assert resp.status_code == 200
        assert 'text/event-stream' in resp.headers.get('content-type','')
        buf = b''
        with anyio.move_on_after(2.0) as scope:
            async for chunk in resp.aiter_bytes():
                buf += chunk
                if b":-hb\n\n" in buf:
                    break
        assert b":-hb\n\n" in buf, f"heartbeat not seen, got: {buf[:200]!r}"
