param(
    [string]$Distro = "Ubuntu2404GPU",
    [string]$Gpu = "0",
    [switch]$RestartWSL,
    [switch]$NoMonitor,
    [switch]$NoStream,
    [string]$StreamUser = "SVER-PokemonYellow",
    [string]$StreamColor = "#FFD700"
)

$repoWin = (Resolve-Path $PSScriptRoot).Path
$drive = $repoWin.Substring(0, 1).ToLower()
$rest = $repoWin.Substring(2).Replace('\', '/')
$repoWsl = "/mnt/$drive$rest"
$trainPidFile = Join-Path $repoWin "wsl_gpu_train.pid"
$hostPidFile = Join-Path $repoWin "wsl_gpu_host.pid"
$monitorPidFile = Join-Path $repoWin "wsl_gpu_monitor.pid"
$monitorScript = Join-Path $repoWin "monitor_train.ps1"
$trainLog = Join-Path $repoWin "wsl_gpu_train.log"
$trainErrLog = Join-Path $repoWin "wsl_gpu_train.err.log"
$monitorLog = Join-Path $repoWin "wsl_gpu_monitor.log"
$monitorStatusFile = Join-Path $repoWin "wsl_gpu_monitor.status.txt"
$restartAuditLog = Join-Path $repoWin "wsl_gpu_restart_history.log"
$trainCommandPattern = '^\s*\d+\s+\S*python(?:3)?(?:\s+\S+)*\s+\S*train\.py(?:\s+.*)?$'

function Write-RestartAudit {
    param(
        [string]$Event,
        [string]$Reason = "",
        [string]$Details = ""
    )

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
    $line = "[$timestamp] event=$Event reason=$Reason details=$Details"
    Add-Content -Path $restartAuditLog -Value $line
}

function Stop-Monitor {
    if (-not (Test-Path $monitorPidFile)) {
        return
    }

    $monitorPid = (Get-Content $monitorPidFile -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
    if ($monitorPid -match '^\d+$') {
        Stop-Process -Id ([int]$monitorPid) -Force -ErrorAction SilentlyContinue
    }
    Remove-Item $monitorPidFile -Force -ErrorAction SilentlyContinue
}

function Stop-HostProcess {
    if (-not (Test-Path $hostPidFile)) {
        return
    }

    $hostPid = (Get-Content $hostPidFile -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
    if ($hostPid -match '^\d+$') {
        Stop-Process -Id ([int]$hostPid) -Force -ErrorAction SilentlyContinue
    }
    Remove-Item $hostPidFile -Force -ErrorAction SilentlyContinue
}

function Get-TrainProcess {
    $raw = & wsl -d $Distro --exec /bin/sh -lc 'pgrep -af "train.py"' 2>$null
    if (-not $raw) {
        return $null
    }

    $first = $raw | Where-Object { $_ -match $trainCommandPattern } | Select-Object -First 1
    if (-not $first) {
        return $null
    }

    $parts = $first.Trim() -split '\s+', 2
    if ($parts.Length -lt 2) {
        return $null
    }

    [pscustomobject]@{
        Pid = $parts[0]
        Command = $parts[1]
    }
}

function Rotate-IfExists {
    param(
        [string]$Path,
        [string]$Timestamp,
        [string]$Label = "pre_restart"
    )

    if (-not (Test-Path $Path)) {
        return
    }

    $item = Get-Item -LiteralPath $Path
    $backupName = "$($item.Name).$Timestamp.$Label.bak"
    $backupPath = Join-Path $item.DirectoryName $backupName
    Move-Item -LiteralPath $item.FullName -Destination $backupPath -Force
}

$existingProc = Get-TrainProcess
$existingPid = if ($existingProc) { "$($existingProc.Pid)" } else { "" }
Write-RestartAudit `
    -Event "restart_requested" `
    -Reason "manual_script" `
    -Details "distro=$Distro restart_wsl=$([bool]$RestartWSL) no_monitor=$([bool]$NoMonitor) old_pid=$existingPid"

if ($RestartWSL) {
    wsl --shutdown
    Start-Sleep -Seconds 2
}

& wsl -d $Distro --exec /bin/sh -lc 'pkill -f "train.py" >/dev/null 2>&1 || true' | Out-Null

Stop-Monitor
Stop-HostProcess
Remove-Item $trainPidFile -Force -ErrorAction SilentlyContinue

$restartTimestamp = Get-Date -Format "yyyyMMdd_HHmmss"
Rotate-IfExists -Path $trainLog -Timestamp $restartTimestamp
Rotate-IfExists -Path $trainErrLog -Timestamp $restartTimestamp
Rotate-IfExists -Path $monitorLog -Timestamp $restartTimestamp
Rotate-IfExists -Path $monitorStatusFile -Timestamp $restartTimestamp

$runArgs = @("-ExecutionPolicy", "Bypass", "-File", (Join-Path $repoWin "run_train_wsl.ps1"), "-Distro", $Distro, "-Gpu", $Gpu, "-StreamUser", $StreamUser, "-StreamColor", $StreamColor)
if ($NoStream) {
    $runArgs += "-NoStream"
}
$hostWindowProc = Start-Process -FilePath "powershell.exe" `
    -ArgumentList $runArgs `
    -WorkingDirectory $repoWin `
    -RedirectStandardOutput $trainLog `
    -RedirectStandardError $trainErrLog `
    -WindowStyle Hidden `
    -PassThru

Set-Content -Path $hostPidFile -Value $hostWindowProc.Id

Start-Sleep -Seconds 1

$trainProc = $null
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    try {
        $trainProc = Get-TrainProcess
        if ($trainProc) {
            break
        }
    }
    catch {
    }

    Start-Sleep -Seconds 1
}

$trainPid = if ($trainProc) { $trainProc.Pid } else { "" }
if ($trainPid -match '^\d+$') {
    Set-Content -Path $trainPidFile -Value $trainPid
}

if (-not $NoMonitor) {
    Stop-Monitor
    Start-Process -FilePath "powershell.exe" `
        -ArgumentList @(
            "-ExecutionPolicy", "Bypass",
            "-File", $monitorScript,
            "-Distro", $Distro,
            "-PollSeconds", "60",
            "-StaleAfterSeconds", "300",
            "-AutoRestart",
            "-RestartAfterConsecutiveDown", "1",
            "-RestartAfterConsecutiveStalled", "2",
            "-RestartCooldownSeconds", "600"
        ) `
        -WorkingDirectory $repoWin `
        -WindowStyle Hidden | Out-Null
}

if ($trainPid -match '^\d+$') {
    Write-RestartAudit `
        -Event "restart_launch_complete" `
        -Reason "manual_script" `
        -Details "distro=$Distro old_pid=$existingPid new_pid=$trainPid host_pid=$($hostWindowProc.Id) monitor_started=$($NoMonitor -eq $false)"
    Write-Host "Launched WSL trainer PID $trainPid on GPU $Gpu"
    & wsl -d $Distro --exec /bin/sh -lc "ps -p $trainPid -o pid=,etime=,cmd=" 2>$null
}
else {
    Write-RestartAudit `
        -Event "restart_launch_pending" `
        -Reason "manual_script" `
        -Details "distro=$Distro old_pid=$existingPid host_pid=$($hostWindowProc.Id) monitor_started=$($NoMonitor -eq $false)"
    Write-Warning "Trainer PID was not found within 30 seconds. The WSL host is PID $($hostWindowProc.Id)."
}
