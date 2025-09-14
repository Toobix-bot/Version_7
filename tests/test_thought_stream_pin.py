import pytest

@pytest.mark.asyncio
async def test_thought_stream_filter_and_pin(client):
    # generate a thought manually
    r1 = await client.post('/thought/generate', headers={'X-API-Key':'test'})
    assert r1.status_code == 200
    t = r1.json()
    tid = t['id']
    cat = t.get('category')
    # filter by category
    r2 = await client.get(f'/thought/stream?limit=5&category={cat}', headers={'X-API-Key':'test'})
    assert r2.status_code == 200
    data = r2.json()
    assert any(item['id']==tid for item in data['items'])
    # pin it
    r3 = await client.patch(f'/thought/{tid}/pin', json={'pinned': True}, headers={'X-API-Key':'test'})
    assert r3.status_code == 200
    pinned_obj = r3.json()
    assert pinned_obj['pinned'] is True
    # fetch again without category filter
    r4 = await client.get('/thought/stream?limit=10', headers={'X-API-Key':'test'})
    assert r4.status_code == 200
    data2 = r4.json()
    # ensure pinned attribute appears
    assert any(item['id']==tid and item.get('pinned') for item in data2['items'])
