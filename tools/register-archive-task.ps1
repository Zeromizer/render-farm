# Registers the nightly workspace archive in Task Scheduler for the current user:
# daily at 04:00, only while this user is logged on (the worker runs in that
# session anyway), no admin rights needed. Re-running replaces the task.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\register-archive-task.ps1 [-Unregister]
param([switch]$Unregister)
$name = "render-farm archive"
if ($Unregister) {
    Unregister-ScheduledTask -TaskName $name -Confirm:$false
    return
}
$wrapper = Join-Path $PSScriptRoot "archive-nightly.ps1"
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$wrapper`"" `
    -WorkingDirectory (Split-Path -Parent $PSScriptRoot)
$trigger = New-ScheduledTaskTrigger -Daily -At "04:00"
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
# 04:00-07:00 retry window plus the last run; a run missed while the PC was off is skipped.
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 4) `
    -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Principal $principal `
    -Settings $settings -Description "Archive stale C:\rp-workspace job dirs (render-farm tools\archive-nightly.ps1)" -Force
