import os, json, pytest
import pytest_asyncio

pytestmark = pytest.mark.skipif(not os.path.isdir('.git'), reason='git repo required')

@pytest.mark.asyncio
async def test_pr_from_plan_dry_run(client):
    resp = await client.post('/dev/pr-from-plan', json={'intent':'demo feature','dry_run':True}, headers={'X-API-Key':'test'})
    data = resp.json()
    assert data['status'] == 'dry-run'
    assert data['branch'].startswith('plan/')

@pytest.mark.asyncio
async def test_plan_pr_alias_dry_run(client):
    resp = await client.post('/plan/pr', json={'intent':'alias feature','dry_run':True}, headers={'X-API-Key':'test'})
    data = resp.json()
    assert data['status'] == 'dry-run'
    assert data['branch'].startswith('plan/')
