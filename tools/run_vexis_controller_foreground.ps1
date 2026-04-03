$workspace = 'E:\Vexis'
$python = Join-Path $workspace '.venv\Scripts\python.exe'
$script = Join-Path $workspace 'interface\gui_main.py'

Set-Location $workspace
$env:VEXIS_EXTERNAL_PRESENCE = '1'
$env:VEXIS_UNREAL_BRIDGE_PORT = '8765'
& $python $script
