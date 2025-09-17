import os
import pytest

H = {"X-API-Key": "test"}


@pytest.mark.asyncio
async def test_render_from_template_includes_overrides(client, tmp_path):  # type: ignore
    pol_dir = tmp_path / "policies" / "templates"
    pol_dir.mkdir(parents=True, exist_ok=True)
    (pol_dir / "solo.yaml").write_text("version: '1'\nallowed_dirs: ['api']\nmax_diff_lines: 100\n", encoding="utf-8")
    os.environ["POLICY_DIR"] = str(tmp_path / "policies")

    r = await client.post("/policy/render-from-template", headers=H, json={
        "name": "solo.yaml",
        "overrides": {"max_diff_lines": 123}
    })
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "max_diff_lines: 123" in data["content"]


@pytest.mark.asyncio
async def test_render_from_template_invalid_name_returns_400(client, tmp_path):  # type: ignore
    os.environ["POLICY_DIR"] = str(tmp_path / "policies")

    r = await client.post("/policy/render-from-template", headers=H, json={"name": "..hack.yaml"})
    assert r.status_code == 400
    data = r.json()
    assert data["error"]["message"] == "invalid_template_name"


@pytest.mark.asyncio
async def test_render_from_template_forbidden_extension_returns_422(client, tmp_path):  # type: ignore
    os.environ["POLICY_DIR"] = str(tmp_path / "policies")

    r = await client.post("/policy/render-from-template", headers=H, json={"name": "template.txt"})
    assert r.status_code == 422
    data = r.json()
    assert data["error"]["message"] == "template_extension_not_allowed"


@pytest.mark.asyncio
async def test_render_from_template_missing_file_returns_404(client, tmp_path):  # type: ignore
    os.environ["POLICY_DIR"] = str(tmp_path / "policies")

    r = await client.post("/policy/render-from-template", headers=H, json={"name": "missing.yaml"})
    assert r.status_code == 404
    data = r.json()
    assert data["error"]["message"] == "template_not_found"


@pytest.mark.asyncio
async def test_apply_from_template_persists_policy(client, tmp_path):  # type: ignore
    pol_dir = tmp_path / "policies" / "templates"
    pol_dir.mkdir(parents=True, exist_ok=True)
    (pol_dir / "team.yaml").write_text("version: '1'\nallowed_dirs: ['api','plans']\n", encoding="utf-8")
    os.environ["POLICY_DIR"] = str(tmp_path / "policies")

    # dry run first
    r_dry = await client.post("/policy/apply-from-template", headers=H, json={
        "name": "team.yaml",
        "overrides": {"deny_globs": ["**/*.secret"]},
        "dry_run": True
    })
    assert r_dry.status_code == 200
    data_dry = r_dry.json()
    assert data_dry["status"] == "validated"
    # apply
    r_apply = await client.post("/policy/apply-from-template", headers=H, json={
        "name": "team.yaml",
        "overrides": {"deny_globs": ["**/*.secret"]},
        "dry_run": False
    })
    assert r_apply.status_code == 200
    data_apply = r_apply.json()
    assert data_apply["status"] == "applied"
    path = data_apply.get("path")
    assert path and os.path.isfile(path)
    txt = open(path, "r", encoding="utf-8").read()
    assert "deny_globs" in txt
