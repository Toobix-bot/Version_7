import pytest
from typing import Any

@pytest.mark.asyncio
async def test_policy_dry_run_valid(client: Any):  # type: ignore
    content = """version: 1
allowed_dirs: [api,plans]
llm:
  model: llama-3.3-70b-versatile
  temperature: 0.0
"""
    resp = await client.post('/policy/dry-run', json={'content': content}, headers={'X-API-Key':'test'})
    data = resp.json()
    assert data['status'] == 'ok'

@pytest.mark.asyncio
async def test_policy_dry_run_invalid_yaml(client: Any):  # type: ignore
    resp = await client.post('/policy/dry-run', json={'content': '::bad'}, headers={'X-API-Key':'test'})
    data = resp.json()
    assert data['status'] == 'error'

@pytest.mark.asyncio
async def test_policy_dry_run_invalid_schema(client: Any):  # type: ignore
    resp = await client.post('/policy/dry-run', json={'content': 'version: 1\nallowed_dirs: 1'}, headers={'X-API-Key':'test'})
    data = resp.json()
    assert data['status'] == 'invalid'
