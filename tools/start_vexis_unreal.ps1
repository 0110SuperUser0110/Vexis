param(
    [int]$BridgePort = 8765,
    [switch]$SkipEditor
)

$workspace = 'E:\Vexis'
$uproject = 'C:\Users\Richard\Documents\Unreal Projects\VexisPresence\VexisPresence.uproject'
$editor = 'E:\UE_5.7\Engine\Binaries\Win64\UnrealEditor.exe'

& (Join-Path $workspace 'tools\start_vexis_backend_unreal.ps1') -BridgePort $BridgePort

if (-not $SkipEditor) {
    if (!(Test-Path $editor)) {
        throw "Unreal Editor not found at $editor"
    }
    $args = '"{0}"' -f $uproject
    Start-Process -FilePath $editor -ArgumentList $args
}
