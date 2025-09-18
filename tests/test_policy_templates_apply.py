import os
import yaml
import pytest

H = {"X-API-Key": "test"}


@pytest.mark.asyncio
async def test_render_from_template_with_overrides(client, tmp_path):  # type: ignore
    pol_dir = tmp_path / "policies" / "templates"
    pol_dir.mkdir(parents=True, exist_ok=True)
    name = "solo_dev.yaml"
    base = {
        "version": "1",
        "allowed_dirs": ["api"],
        "llm": {"model": "base", "temperature": 0.2}
    }
    (pol_dir / name).write_text(yaml.safe_dump(base), encoding="utf-8")
    os.environ["POLICY_DIR"] = str(tmp_path / "policies")

    overrides = {"llm": {"temperature": 0.0}}
    r = await client.post("/policy/render-from-template", headers=H, json={"name": name, "overrides": overrides})
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    # rendered content should reflect override
    assert "temperature: 0.0" in data["content"]


@pytest.mark.asyncio
async def test_apply_from_template_persists_yaml(client, tmp_path):  # type: ignore
    pol_dir = tmp_path / "policies" / "templates"
    pol_dir.mkdir(parents=True, exist_ok=True)
    name = "team.yaml"
    base = {
        "version": "1",
        "allowed_dirs": ["api", "plans"],
        "llm": {"model": "x", "temperature": 0.4}
    }
    (pol_dir / name).write_text(yaml.safe_dump(base), encoding="utf-8")
    os.environ["POLICY_DIR"] = str(tmp_path / "policies")

    r = await client.post("/policy/apply-from-template", headers=H, json={"name": name, "overrides": {"llm": {"temperature": 0.1}}, "dry_run": False})
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "applied"
    path = data["path"]
    assert path and os.path.isfile(path)
    txt = open(path, "r", encoding="utf-8").read()
    assert "temperature: 0.1" in txt