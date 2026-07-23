param(
    [string]$Distro = "Ubuntu2404GPU"
)

# Run at Windows logon (via Scheduled Task) to recover training after a
# reboot -- Windows Update auto-restarts being the concrete case that
# prompted this (2026-07-22: a WSL restart from a Windows Update killed
# training and the watchdog together, and nobody noticed for ~18 minutes).
# Only restarts if nothing is actually running, so a routine logon/logoff
# doesn't needlessly reset an already-healthy run's rolling stats.

$repoWin = $PSScriptRoot
$logFile = Join-Path $repoWin "ensure_training_running.log"
$trainCommandPattern = '^\s*\d+\s+\S*python(?:3)?(?:\s+\S+)*\s+\S*train\.py(?:\s+.*)?$'

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
    Add-Content -Path $logFile -Value "[$timestamp] $Message"
}

function Test-TrainProcessAlive {
    try {
        $raw = & wsl -d $Distro --exec /bin/sh -lc 'pgrep -af "train.py"' 2>$null
    }
    catch {
        Write-Log "wsl check failed: $($_.Exception.Message) -- assuming not running"
        return $false
    }
    if (-not $raw) {
        return $false
    }
    $match = $raw | Where-Object { $_ -match $trainCommandPattern } | Select-Object -First 1
    return [bool]$match
}

Write-Log "logon check starting (distro=$Distro)"

if (Test-TrainProcessAlive) {
    Write-Log "train.py already running -- nothing to do"
    exit 0
}

Write-Log "train.py not running -- launching start_train_wsl.ps1"
& (Join-Path $repoWin "start_train_wsl.ps1") -Distro $Distro
Write-Log "start_train_wsl.ps1 invocation complete"
