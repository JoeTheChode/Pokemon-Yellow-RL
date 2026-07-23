# Serves pokemon_yellow_map_viewer.html (and the live_agent_positions.json
# it polls) over local HTTP -- the viewer's fetch() calls fail under file://.
param(
    [int]$Port = 8099
)

$repoWin = (Resolve-Path $PSScriptRoot).Path
$python = Join-Path $repoWin ".venv\Scripts\python.exe"
$pidFile = Join-Path $repoWin "map_viewer_server.pid"
$log = Join-Path $repoWin "map_viewer_server.log"
$errLog = Join-Path $repoWin "map_viewer_server.err.log"

if (Test-Path $pidFile) {
    $existingPid = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
    if ($existingPid -match '^\d+$' -and (Get-Process -Id $existingPid -ErrorAction SilentlyContinue)) {
        Write-Host "Map viewer server already running (PID $existingPid) at http://localhost:$Port/pokemon_yellow_map_viewer.html"
        exit 0
    }
}

$proc = Start-Process -FilePath $python `
    -ArgumentList @("-m", "http.server", "$Port", "--directory", $repoWin) `
    -WorkingDirectory $repoWin `
    -RedirectStandardOutput $log `
    -RedirectStandardError $errLog `
    -WindowStyle Hidden `
    -PassThru

Set-Content -Path $pidFile -Value $proc.Id
Start-Sleep -Seconds 1
Write-Host "Started map viewer server (PID $($proc.Id)) at http://localhost:$Port/pokemon_yellow_map_viewer.html"
