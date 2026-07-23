param(
    [switch]$CurrentStage,
    [string]$State,
    [string]$SaveName,
    [string]$LogFile,
    [string]$RecordFile
)

$repoRoot = (Resolve-Path $PSScriptRoot).Path
$condaHook = "C:\Users\Admin\miniconda3\shell\condabin\conda-hook.ps1"

if (Test-Path $condaHook) {
    & $condaHook
    conda activate ml-env
}

Set-Location $repoRoot

$argsList = @("play.py")
if ($CurrentStage -or -not $State) {
    $argsList += "--current-stage"
}
if ($State) {
    $argsList += $State
}
if ($SaveName) {
    $argsList += $SaveName
}
if ($LogFile) {
    $argsList += "--log-file"
    $argsList += $LogFile
}
if ($RecordFile) {
    $argsList += "--record-file"
    $argsList += $RecordFile
}

python @argsList
