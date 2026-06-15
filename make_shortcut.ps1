# Creates a "privy" shortcut on the Desktop that launches the app with no console.
$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
$pythonw = Join-Path $root ".venv\Scripts\pythonw.exe"
$app = Join-Path $root "app.py"
$desktop = [Environment]::GetFolderPath('Desktop')
$lnk = Join-Path $desktop "privy.lnk"

$ws = New-Object -ComObject WScript.Shell
$sc = $ws.CreateShortcut($lnk)
$sc.TargetPath = $pythonw
$sc.Arguments = "`"$app`""
$sc.WorkingDirectory = $root
$sc.IconLocation = $pythonw
$sc.Description = "privy - local-first meeting copilot"
$sc.Save()
Write-Output "Created shortcut: $lnk"
