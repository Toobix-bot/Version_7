Param(
    [Parameter(Mandatory=$true)][string]$Intent,
    [string]$RiskBudget = "",
    [string]$VariantId = "",
    [switch]$DryRun
)

$baseUrl = $env:BASE_URL
if (-not $baseUrl) { $baseUrl = "http://127.0.0.1:8000" }
$apiKey = $env:API_KEY
if (-not $apiKey) { Write-Error "API_KEY env var required"; exit 2 }

$body = @{ intent = $Intent; risk_budget = $RiskBudget; variant_id = $VariantId; dry_run = $DryRun.IsPresent } | ConvertTo-Json -Depth 4
$headers = @{ 'X-API-Key' = $apiKey; 'Content-Type' = 'application/json' }

Write-Host "[info] POST /dev/pr-from-plan intent=$Intent variant=$VariantId risk=$RiskBudget dryRun=$($DryRun.IsPresent)"
try {
    $resp = Invoke-RestMethod -Uri "$baseUrl/dev/pr-from-plan" -Method Post -Headers $headers -Body $body -TimeoutSec 60
} catch {
    Write-Error $_
    exit 1
}

$resp | ConvertTo-Json -Depth 6
if ($resp.status -eq 'created') {
    Write-Host "[ok] Branch: $($resp.branch) Variant: $($resp.variant)"
} elseif ($resp.status -eq 'dry-run') {
    Write-Host "[dry-run] Would create branch $($resp.branch)"
} else {
    Write-Host "[warn] status=$($resp.status) message=$($resp.message)"
}
