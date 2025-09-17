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


@pytest.mark.asyncio
async def test_read_policy_template_invalid_name_returns_400(client, tmp_path):  # type: ignore
    os.environ["POLICY_DIR"] = str(tmp_path / "policies")

    r = await client.get("/policy/templates/..secret.yaml", headers=H)
    assert r.status_code == 400
    data = r.json()
    assert data["error"]["message"] == "invalid_template_name"


@pytest.mark.asyncio
async def test_read_policy_template_forbidden_extension_returns_422(client, tmp_path):  # type: ignore
    os.environ["POLICY_DIR"] = str(tmp_path / "policies")

    r = await client.get("/policy/templates/template.txt", headers=H)
    assert r.status_code == 422
    data = r.json()
    assert data["error"]["message"] == "template_extension_not_allowed"


@pytest.mark.asyncio
async def test_read_policy_template_missing_file_returns_404(client, tmp_path):  # type: ignore
    os.environ["POLICY_DIR"] = str(tmp_path / "policies")

    r = await client.get("/policy/templates/missing.yaml", headers=H)
    assert r.status_code == 404
    data = r.json()
    assert data["error"]["message"] == "template_not_found"
