import os
import pytest

H = {"X-API-Key": "test"}


@pytest.mark.asyncio
async def test_read_policy_template(client, tmp_path):  # type: ignore
    pol_dir = tmp_path / "policies" / "templates"
    pol_dir.mkdir(parents=True, exist_ok=True)
    name = "sandbox.yaml"
    content = "rules:\n  - allow: true\n"
    (pol_dir / name).write_text(content, encoding="utf-8")
    os.environ["POLICY_DIR"] = str(tmp_path / "policies")

    r = await client.get(f"/policy/templates/{name}", headers=H)
    assert r.status_code == 200
    data = r.json()
    assert data["name"] == name
    assert data["size"] >= len(content)
    assert "rules" in data["content"]
