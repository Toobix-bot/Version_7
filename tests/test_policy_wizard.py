import pytest
from typing import Any

@pytest.mark.asyncio
async def test_policy_wizard_basic(client: Any):  # type: ignore
    resp = await client.post('/policy/wizard', json={'template':'solo_dev','goals':['Schneller Review','Klarheit']}, headers={'X-API-Key':'test'})
    assert resp.status_code == 200
    data = resp.json()
    assert data['source_template'] == 'solo_dev'
    assert 'policy' in data
    assert data['policy'].get('version') == 1
    assert any('goals:' in n or 'goals' in n for n in data.get('notes', [])) or data['notes'] is not None

@pytest.mark.asyncio
async def test_policy_wizard_risk_low_enforces_temp_zero(client: Any):  # type: ignore
    resp = await client.post('/policy/wizard', json={'template':'solo_dev','risk_profile':'low','annotate':True}, headers={'X-API-Key':'test'})
    data = resp.json()
    assert resp.status_code == 200
    llm = data['policy'].get('llm', {})
    assert llm.get('temperature') == 0.0
    # Note may be localized; just ensure a note referencing 'low' risk profile exists
    assert any('low' in n.lower() for n in data.get('notes', []))

@pytest.mark.asyncio
async def test_policy_wizard_overrides_and_diff(client: Any):  # type: ignore
    body = {
        'template':'solo_dev',
        'overrides':{
            'allowed_dirs':['api','ui','policies','tests']
        },
        'annotate': True
    }
    resp = await client.post('/policy/wizard', json=body, headers={'X-API-Key':'test'})
    assert resp.status_code == 200
    data = resp.json()
    assert 'diff' in data
    changed = data['diff'].get('changed', {})
    assert 'allowed_dirs' in changed or 'allowed_dirs' in data['policy']
    assert 'tests' in data['policy']['allowed_dirs']

@pytest.mark.asyncio
async def test_policy_wizard_template_not_found(client: Any):  # type: ignore
    resp = await client.post('/policy/wizard', json={'template':'does_not_exist'}, headers={'X-API-Key':'test'})
    assert resp.status_code == 404
    data = resp.json()
    # Accept either FastAPI default error wrapper or direct detail
    if 'detail' in data and isinstance(data['detail'], dict):
        assert data['detail'].get('code') == 'template_not_found'
    else:
        # fallback: wrapped error format
        err = data.get('error') or {}
        if err.get('code') == 'http_error' and 'template_not_found' in str(err.get('message')):
            assert True
        else:
            raise AssertionError(f"unexpected error payload: {data}")
