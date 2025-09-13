#!/usr/bin/env python
"""Create a PR from a generated plan.

Dependencies:
  - gh (GitHub CLI) installed & authenticated
  - Git repo clean (no uncommitted changes) OR changes intended for commit
  - Environment:
      API_KEY (or API_TOKENS) for calling the service
      BASE_URL (default: http://127.0.0.1:8000)
Usage:
  python scripts/create_pr_from_plan.py --intent "Refactor X" --targets api/app.py policies/model.py
Options:
  --risk-budget low|medium|high (affects variant ordering)
  --variant-id <id> (choose specific variant; else first)
  --dry-run (no git/gh actions)
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, textwrap, time, pathlib
import urllib.request

DEFAULT_BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.environ.get("API_KEY") or os.environ.get("API_TOKENS", "").split(",")[0] or "test"


def fetch_plan(intent: str, context: str, targets: list[str], risk_budget: str|None) -> dict:
    payload = json.dumps({
        "intent": intent,
        "context": context,
        "target_paths": targets,
    }).encode()
    url = DEFAULT_BASE + (f"/plan?risk_budget={risk_budget}" if risk_budget else "/plan")
    req = urllib.request.Request(url, data=payload, method="POST", headers={
        "Content-Type": "application/json",
        "X-API-Key": API_KEY,
    })
    with urllib.request.urlopen(req, timeout=30) as resp:  # nosec
        return json.loads(resp.read().decode())


def ensure_clean_git():
    res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
    if res.stdout.strip():
        print("ERROR: Working tree not clean. Commit or stash first.", file=sys.stderr)
        sys.exit(2)


def make_branch(branch: str):
    subprocess.run(["git", "checkout", "-b", branch], check=True)


def write_artifact(plan: dict, branch: str) -> str:
    out_dir = pathlib.Path("plans")
    out_dir.mkdir(exist_ok=True)
    ts = int(time.time())
    fname = out_dir / f"{branch}_variant.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    return str(fname)


def commit_and_pr(branch: str, artifact_path: str, intent: str, variant: dict, dry: bool):
    title = f"Plan: {intent} ({variant.get('label')})"
    body = textwrap.dedent(f"""
    # Plan Proposal: {intent}

    **Variant**: `{variant.get('id')}` ({variant.get('label')})  
    **Risk Level**: {variant.get('risk_level')}  
    **Knobs**: `{json.dumps(variant.get('knobs', {}))}`  

    ## Summary
    {variant.get('summary')}

    ## Explanation
    {variant.get('explanation') or 'n/a'}

    ## Patch Preview
    ```diff
    {variant.get('patch_preview') or '// no preview'}
    ```

    ## Constraints
    - PR-only (no direct writes)
    - Policy whitelist enforced
    - Risk/regex gate active

    ## Checklist
    - [ ] Lint pass
    - [ ] Type check pass
    - [ ] Tests added/updated (if needed)
    - [ ] Policy unaffected or updated consciously
    - [ ] Metrics unaffected or updated consciously

    Generated at {time.ctime()}.
    """)
    if dry:
        print("[DRY RUN] Would commit & create PR:\n", body)
        return
    subprocess.run(["git", "add", artifact_path], check=True)
    subprocess.run(["git", "commit", "-m", f"Add plan artifact for {intent} ({variant.get('id')})"], check=True)
    subprocess.run(["git", "push", "--set-upstream", "origin", branch], check=True)
    subprocess.run(["gh", "pr", "create", "--title", title, "--body", body, "--label", "plan,ready-for-copilot"], check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--intent", required=True)
    p.add_argument("--context", default="")
    p.add_argument("--targets", nargs="*", default=[])
    p.add_argument("--risk-budget", choices=["low","medium","high"], dest="risk_budget")
    p.add_argument("--variant-id")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    ensure_clean_git()

    data = fetch_plan(args.intent, args.context, args.targets, args.risk_budget)
    if not data.get('variants'):
        print("ERROR: No variants returned", file=sys.stderr)
        sys.exit(1)
    variant = None
    if args.variant_id:
        variant = next((v for v in data['variants'] if v.get('id') == args.variant_id), None)
        if not variant:
            print("ERROR: variant-id not found", file=sys.stderr)
            sys.exit(1)
    else:
        variant = data['variants'][0]

    branch_slug = f"plan/{int(time.time())}_{variant.get('id')}".replace('..','.')
    make_branch(branch_slug)
    artifact_path = write_artifact(variant, branch_slug)
    commit_and_pr(branch_slug, artifact_path, args.intent, variant, args.dry_run)
    print("Created PR branch", branch_slug)

if __name__ == "__main__":
    main()
