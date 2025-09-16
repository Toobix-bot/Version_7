import os
import pytest

H = {"X-API-Key": "test"}


@pytest.mark.asyncio
async def test_list_policy_templates(client, tmp_path):  # type: ignore
    # create a temporary templates dir with samples
    pol_dir = tmp_path / "policies" / "templates"
    pol_dir.mkdir(parents=True, exist_ok=True)
    (pol_dir / "solo_dev.yaml").write_text("rules: []", encoding="utf-8")
    (pol_dir / "team.yml").write_text("rules: []", encoding="utf-8")
    os.environ["POLICY_DIR"] = str(tmp_path / "policies")
    # call endpoint
    r = await client.get("/policy/templates", headers=H)
    assert r.status_code == 200
    data = r.json()
    names = [t["name"] for t in data.get("templates", [])]
    assert "solo_dev.yaml" in names
    assert "team.yml" in names
