param(
    [int]$BridgePort = 8765,
    [switch]$SkipEditor
)

$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$uproject = Join-Path $workspace 'unreal\VexisPresence\VexisPresence.uproject'

$editorCandidates = @()
if ($env:VEXIS_UNREAL_EDITOR) {
    $editorCandidates += $env:VEXIS_UNREAL_EDITOR
}
$editorCandidates += @(
    'E:\UE_5.7\Engine\Binaries\Win64\UnrealEditor.exe',
    'C:\Program Files\Epic Games\UE_5.7\Engine\Binaries\Win64\UnrealEditor.exe'
)
$editor = $editorCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

& (Join-Path $workspace 'tools\start_vexis_backend_unreal.ps1') -BridgePort $BridgePort

if (-not $SkipEditor) {
    if (!$editor) {
        throw "Unreal Editor not found. Set VEXIS_UNREAL_EDITOR or install UE 5.7 in a standard location."
    }
    $args = '"{0}"' -f $uproject
    Start-Process -FilePath $editor -ArgumentList $args
}
