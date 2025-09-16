import pytest

H = {"X-API-Key": "test"}


@pytest.mark.asyncio
async def test_metrics_content_type(client):  # type: ignore
    r = await client.get('/metrics', headers=H)
    assert r.status_code == 200
    ctype = r.headers.get('content-type','')
    # Prometheus exposition content type
    assert ctype.startswith('text/plain; version=0.0.4; charset=utf-8')
    # basic sanity: HELP/TYPE lines likely present for registry
    body = r.text
    assert '# HELP' in body and '# TYPE' in body
