param(
    [int]$BridgePort = 8765,
    [switch]$UseStagedRuntime
)

$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$uproject = Join-Path $workspace 'unreal\VexisPresence\VexisPresence.uproject'
$runtime = Join-Path $workspace 'unreal\VexisPresence\Saved\StagedBuilds\Windows\VexisPresence.exe'
$editorCandidates = @()
if ($env:VEXIS_UNREAL_EDITOR) {
    $editorCandidates += $env:VEXIS_UNREAL_EDITOR
}
$editorCandidates += @(
    'E:\UE_5.7\Engine\Binaries\Win64\UnrealEditor.exe',
    'C:\Program Files\Epic Games\UE_5.7\Engine\Binaries\Win64\UnrealEditor.exe'
)
$editor = $editorCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
$controllerLauncher = Join-Path $workspace 'tools\run_vexis_controller_foreground.ps1'
$mapArg = '/Engine/Maps/Entry?game=/Script/VexisPresence.VexisPresenceGameMode'
$windowArgs = '-windowed -noborder -d3d11 -ResX=520 -ResY=520 -WinX=70 -WinY=110 -NoSplash -NoLoadingScreen -NoScreenMessages -DefaultViewportMouseCaptureMode=NoCapture -DefaultViewportMouseLockMode=DoNotLock'

$backend = Start-Process -FilePath 'C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe' -ArgumentList @(
    '-NoExit',
    '-ExecutionPolicy',
    'Bypass',
    '-File',
    $controllerLauncher
) -WorkingDirectory $workspace -PassThru

Start-Sleep -Seconds 6

if ($UseStagedRuntime -and (Test-Path $runtime)) {
    $launchArgs = '{0} {1}' -f $mapArg, $windowArgs
    $unreal = Start-Process -FilePath $runtime -ArgumentList $launchArgs -WorkingDirectory (Split-Path $runtime) -PassThru
}
else {
    if (!$editor) {
        throw "Unreal Editor not found. Set VEXIS_UNREAL_EDITOR or install UE 5.7 in a standard location."
    }
    $launchArgs = '"{0}" {1} -game {2}' -f $uproject, $mapArg, $windowArgs
    $unreal = Start-Process -FilePath $editor -ArgumentList $launchArgs -PassThru
}

$backend.WaitForExit()
Get-Process | Where-Object { $_.ProcessName -in @('VexisPresence', 'UnrealEditor') } | Stop-Process -Force
if (Get-Process -Id $unreal.Id -ErrorAction SilentlyContinue) {
    Stop-Process -Id $unreal.Id -Force
}
