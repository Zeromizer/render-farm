# Creates a Startup-folder shortcut that launches the render supervisor
# windowless at logon. User-space only; no admin required.
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonw = Join-Path (Split-Path -Parent $here) ".venv\Scripts\pythonw.exe"
$target = Join-Path $here "supervisor.py"
$startup = [Environment]::GetFolderPath('Startup')
$lnkPath = Join-Path $startup "render-farm-supervisor.lnk"

$shell = New-Object -ComObject WScript.Shell
$lnk = $shell.CreateShortcut($lnkPath)
$lnk.TargetPath = $pythonw
$lnk.Arguments = "`"$target`""
$lnk.WorkingDirectory = $here
$lnk.Description = "Render farm worker supervisor"
$lnk.Save()
Write-Host "Created $lnkPath"
