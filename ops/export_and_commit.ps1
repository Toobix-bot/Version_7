Param(
  [string]$ServerUrl="https://echo-realm.onrender.com",
  [string]$Message="chore: update openapi spec"
)
$env:PUBLIC_SERVER_URL=$ServerUrl
Write-Host "[export] generating openapi.yaml for $ServerUrl"
python ops/export_openapi.py
if ($LASTEXITCODE -ne 0) { Write-Error "export failed"; exit 1 }
if (-not (Test-Path docs/openapi.yaml)) { Write-Error "docs/openapi.yaml missing"; exit 1 }

Write-Host "[git] staging docs/openapi.yaml"
git add docs/openapi.yaml
$changed = git diff --cached --name-only | Select-String -Pattern "docs/openapi.yaml"
if ($changed) {
  git commit -m $Message
  Write-Host "[git] committed"
} else {
  Write-Host "[git] no changes to commit"
}
