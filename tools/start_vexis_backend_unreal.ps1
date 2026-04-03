param(
    [int]$BridgePort = 8765
)

$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $workspace '.venv\Scripts\python.exe'
$script = Join-Path $workspace 'interface\gui_main.py'

if (!(Test-Path $python)) {
    throw "Python runtime not found at $python"
}

Start-Process -FilePath 'C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe' -ArgumentList @(
    '-NoExit',
    '-Command',
    "Set-Location '$workspace'; `$env:VEXIS_EXTERNAL_PRESENCE='1'; `$env:VEXIS_UNREAL_BRIDGE_PORT='$BridgePort'; & '$python' '$script'"
)
