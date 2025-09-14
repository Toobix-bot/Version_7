import os
import re
import pytest
from httpx import AsyncClient

os.environ.setdefault("API_TOKEN", "test-token")
API_KEY = os.getenv("TEST_API_KEY", "test")
H = {"X-API-Key": API_KEY}

@pytest.mark.asyncio
async def test_plan_variant_metric_emitted(client: AsyncClient):  # type: ignore
    # call plan endpoint
    payload = {"intent": "metric plan", "context": "ctx", "target_paths": ["plans/metric.txt"]}
    r = await client.post("/plan", headers=H, json=payload)
    assert r.status_code == 200
    data = r.json()
    variants = data.get("variants", [])
    assert variants, "expected variants in plan response"
    # fetch metrics
    m = await client.get("/metrics", headers=H)
    assert m.status_code == 200
    body = m.text
    # each variant id should appear at least once as label
    for v in variants:
        vid = v["id"]
        assert f'variant_id="{vid}"' in body, f"metric line for variant {vid} missing"
    # ensure counter name present
    assert "plan_variant_generated_total" in body
    # basic format regex (Prometheus exposition)
    assert re.search(r'^plan_variant_generated_total\{.*variant_id="'+variants[0]['id']+r'".*\} \d+$', body, re.MULTILINE)
