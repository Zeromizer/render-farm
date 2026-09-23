# Nightly stale-workspace archive (Task Scheduler "render-farm archive", 04:00 daily).
#
# Runs tools\archive-stale-workspaces.py --execute --idle-gate --max-jobs 30. When the
# idle gate reports "busy, skipped" it retries every $RetryMinutes until $Until, then
# gives up for the day. Everything is appended to %LOCALAPPDATA%\render-farm\logs\
# archive-nightly.log; the script's own JSON receipt goes to archive-YYYYMMDD.json.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\archive-nightly.ps1 [-DryRun]
param(
    [switch]$DryRun,
    [string]$Until = "07:00",
    [int]$RetryMinutes = 15,
    [int]$MaxJobs = 30
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo ".venv\Scripts\python.exe"
$script = Join-Path $repo "tools\archive-stale-workspaces.py"
$logDir = Join-Path $env:LOCALAPPDATA "render-farm\logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$log = Join-Path $logDir "archive-nightly.log"
$deadline = [datetime]::ParseExact($Until, "HH:mm", $null)
$mode = if ($DryRun) { "--dry-run" } else { "--execute" }

function Write-Log($line) {
    $stamped = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $line
    Add-Content -Path $log -Value $stamped -Encoding utf8
    Write-Output $stamped
}

$env:PYTHONIOENCODING = "utf-8"
while ($true) {
    Write-Log "run: $mode --idle-gate --max-jobs $MaxJobs"
    $out = & $python $script $mode --idle-gate --max-jobs $MaxJobs 2>&1 | ForEach-Object { "$_" }
    $code = $LASTEXITCODE
    foreach ($l in $out) { Write-Log "  $l" }
    if ($code -ne 0) { Write-Log "exit code $code"; exit $code }
    $busy = $out | Where-Object { $_ -like "busy, skipped*" }
    if (-not $busy) { break }
    if ((Get-Date).AddMinutes($RetryMinutes) -gt $deadline) {
        Write-Log "still busy; giving up until tomorrow"
        break
    }
    Start-Sleep -Seconds ($RetryMinutes * 60)
}
