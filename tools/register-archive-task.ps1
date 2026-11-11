# Registers the nightly workspace archive in Task Scheduler for the current user:
# daily at 04:00, only while this user is logged on (the worker runs in that
# session anyway), no admin rights needed. Re-running replaces the task.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\register-archive-task.ps1 [-Unregister]
#
# A second platform gets its own task (worker/db.py projects), e.g.
#   ... register-archive-task.ps1 -Project nbd -Workspace C:\rp-workspace-nbd -Site https://<host>/jobs/ -At 05:30
# registers "render-farm archive nbd". Without -Project this is the Aimotion task, unchanged.
param(
    [switch]$Unregister,
    [string]$Project = "",
    [string]$Workspace = "",
    [string]$Site = "",
    [string]$At = "04:00"
)
$name = if ($Project) { "render-farm archive $Project" } else { "render-farm archive" }
if ($Unregister) {
    Unregister-ScheduledTask -TaskName $name -Confirm:$false
    return
}
if ($Project -and -not $Workspace) { throw "-Project needs -Workspace (its own folder, never C:\rp-workspace)" }
$wrapper = Join-Path $PSScriptRoot "archive-nightly.ps1"
$extra = ""
if ($Project) { $extra += " -Project $Project" }
if ($Workspace) { $extra += " -Workspace `"$Workspace`"" }
if ($Site) { $extra += " -Site `"$Site`"" }
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$wrapper`"$extra" `
    -WorkingDirectory (Split-Path -Parent $PSScriptRoot)
$trigger = New-ScheduledTaskTrigger -Daily -At $At
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
# 04:00-07:00 retry window plus the last run; a run missed while the PC was off is skipped.
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 4) `
    -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Principal $principal `
    -Settings $settings -Description "Archive stale $(if ($Workspace) { $Workspace } else { 'C:\rp-workspace' }) job dirs (render-farm tools\archive-nightly.ps1)" -Force
