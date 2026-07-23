param(
    [string]$Distro = "Ubuntu2404GPU",
    [int]$PollSeconds = 60,
    [int]$StaleAfterSeconds = 300,
    [switch]$AutoRestart,
    [int]$RestartAfterConsecutiveDown = 2,
    [int]$RestartAfterConsecutiveStalled = 2,
    [int]$RestartCooldownSeconds = 900
)

$repoRoot = (Resolve-Path $PSScriptRoot).Path
$trainLog = Join-Path $repoRoot "wsl_gpu_train.log"
$monitorLog = Join-Path $repoRoot "wsl_gpu_monitor.log"
$statusFile = Join-Path $repoRoot "wsl_gpu_monitor.status.txt"
$pidFile = Join-Path $repoRoot "wsl_gpu_monitor.pid"
$restartAuditLog = Join-Path $repoRoot "wsl_gpu_restart_history.log"
$restartScript = Join-Path $repoRoot "start_train_wsl.ps1"
$trainCommandPattern = '^\s*\d+\s+\S*python(?:3)?(?:\s+\S+)*\s+\S*train\.py(?:\s+.*)?$'

Set-Content -Path $pidFile -Value $PID

function Write-MonitorLine {
    param([string]$Message)

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
    $line = "[$timestamp] $Message"
    Add-Content -Path $monitorLog -Value $line
    Set-Content -Path $statusFile -Value $line
}

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

function Get-TrainProcess {
    # wsl.exe can throw a terminating error if the WSL instance is mid-restart
    # (this killed the whole watchdog once already -- 2026-07-22, ~18 min of
    # undetected downtime because there was nothing left running to notice or
    # auto-restart). Treat any such failure the same as "no process found".
    try {
        $raw = & wsl -d $Distro --exec /bin/sh -lc 'pgrep -af "train.py"' 2>$null
    }
    catch {
        return $null
    }
    if (-not $raw) {
        return $null
    }

    $first = $raw | Where-Object { $_ -match $trainCommandPattern } | Select-Object -First 1
    if (-not $first) {
        return $null
    }

    $first = $first.Trim()

    $parts = $first -split "\s+", 2
    if ($parts.Length -lt 2) {
        return $null
    }

    [pscustomobject]@{
        Pid = $parts[0]
        Command = $parts[1]
    }
}

function Invoke-TrainRestart {
    param(
        [string]$Reason,
        [string]$PriorPid = "",
        [string]$PriorSessionSteps = ""
    )

    if (-not (Test-Path $restartScript)) {
        Write-MonitorLine "restart_skipped reason=$Reason missing_script=$restartScript"
        Write-RestartAudit -Event "restart_skipped" -Reason $Reason -Details "missing_script=$restartScript"
        return $false
    }

    $triggerDetails = @()
    if ($PriorPid) {
        $triggerDetails += "old_pid=$PriorPid"
    }
    if ($PriorSessionSteps) {
        $triggerDetails += "old_session_steps=$PriorSessionSteps"
    }
    $triggerDetailText = if ($triggerDetails.Count -gt 0) { " $($triggerDetails -join ' ')" } else { "" }

    Write-MonitorLine "restart_trigger reason=$Reason$triggerDetailText"
    Write-RestartAudit -Event "restart_trigger" -Reason $Reason -Details ($triggerDetails -join ' ')
    try {
        & powershell.exe -ExecutionPolicy Bypass -File $restartScript -Distro $Distro -NoMonitor | Out-Null
        $newProc = $null
        for ($attempt = 0; $attempt -lt 30; $attempt++) {
            $newProc = Get-TrainProcess
            if ($newProc) {
                break
            }
            Start-Sleep -Seconds 1
        }
        $newPid = if ($newProc) { "$($newProc.Pid)" } else { "missing" }
        $newMetrics = Get-LogTailMetrics
        $newSessionSteps = Get-SessionStepsFromMetrics -MetricsLine $newMetrics
        $completeDetails = @("new_pid=$newPid")
        if ($PriorPid) {
            $completeDetails += "old_pid=$PriorPid"
        }
        if ($PriorSessionSteps) {
            $completeDetails += "old_session_steps=$PriorSessionSteps"
        }
        if ($newSessionSteps -ne $null) {
            $completeDetails += "new_session_steps=$newSessionSteps"
        }
        $completeDetailText = $completeDetails -join ' '

        Write-MonitorLine "restart_complete reason=$Reason $completeDetailText"
        Write-RestartAudit -Event "restart_complete" -Reason $Reason -Details $completeDetailText
        return $true
    }
    catch {
        Write-MonitorLine "restart_failed reason=$Reason error=$($_.Exception.Message)"
        Write-RestartAudit -Event "restart_failed" -Reason $Reason -Details $_.Exception.Message
        return $false
    }
}

function Get-LogTailMetrics {
    if (-not (Test-Path $trainLog)) {
        return $null
    }

    $tail = @(Get-Content -Path $trainLog -Tail 120)
    for ($i = $tail.Count - 1; $i -ge 0; $i--) {
        if ($tail[$i] -notmatch "^\s*\[METRICS\]") {
            continue
        }

        $block = @($tail[$i].Trim())
        for ($j = $i + 1; $j -lt $tail.Count; $j++) {
            $line = $tail[$j]
            if ($line -match "^\s{4}(reward|hits|level|critic|value|kl|clip|entropy|ep_len)\b") {
                $block += $line.Trim()
                continue
            }
            if ($line.Trim().Length -eq 0) {
                break
            }
            if ($line -match "^\s*\[") {
                break
            }
        }

        if ($block.Count -gt 0) {
            return ($block -join " | ")
        }
    }

    $summary = $tail | Where-Object {
        $_ -match "^\s+ep_rew_mean\s+" -or
        $_ -match "^\s+goal_hits\s+" -or
        $_ -match "^\s+explained_variance\s+" -or
        $_ -match "^\|\s+fps\s+\|" -or
        $_ -match "^\|\s+iterations\s+\|" -or
        $_ -match "^\|\s+total_timesteps\s+\|" -or
        $_ -match "\[AUTO-RESET\]" -or
        $_ -match "\bMILESTONE\b"
    }

    if (-not $summary) {
        return $null
    }

    return (($summary | Select-Object -Last 4) | ForEach-Object { $_.Trim() }) -join " | "
}

function Get-SessionStepsFromMetrics {
    param([string]$MetricsLine)

    if (-not $MetricsLine) {
        return $null
    }

    $match = [regex]::Match($MetricsLine, 'session_steps=([0-9,]+)')
    if (-not $match.Success) {
        return $null
    }

    $raw = $match.Groups[1].Value.Replace(',', '')
    $value = 0L
    if ([long]::TryParse($raw, [ref]$value)) {
        return $value
    }
    return $null
}

Write-MonitorLine "monitor_started distro=$Distro poll=${PollSeconds}s stale_after=${StaleAfterSeconds}s"
Write-RestartAudit -Event "monitor_started" -Details "distro=$Distro auto_restart=$AutoRestart poll=${PollSeconds}s stale_after=${StaleAfterSeconds}s down_threshold=$RestartAfterConsecutiveDown stalled_threshold=$RestartAfterConsecutiveStalled cooldown=${RestartCooldownSeconds}s"
$lastState = ""
$consecutiveDown = 0
$consecutiveStalled = 0
$lastRestartAt = [datetime]::MinValue
$lastRunningPid = $null
$lastSessionSteps = $null

while ($true) {
  try {
    $proc = Get-TrainProcess
    $state = "down"
    $details = ""
    $currentSessionSteps = $null

    if ($proc) {
        $state = "running"
        $details = "pid=$($proc.Pid)"
    }

    $metrics = $null
    if (Test-Path $trainLog) {
        $logItem = Get-Item $trainLog
        $age = [int]((Get-Date) - $logItem.LastWriteTime).TotalSeconds
        $details = if ($details) { "$details log_age=${age}s" } else { "log_age=${age}s" }

        $metrics = Get-LogTailMetrics
        if ($metrics) {
            $details = "$details metrics=$metrics"
        }

        if ($state -eq "running" -and $age -gt $StaleAfterSeconds) {
            $state = "stalled"
        }
    }
    elseif ($details) {
        $details = "$details log_missing"
    }
    else {
        $details = "log_missing"
    }

    $snapshot = "$state $details".Trim()
    if ($snapshot -ne $lastState -or $state -ne "running") {
        Write-MonitorLine $snapshot
        $lastState = $snapshot
    }

    if ($state -ne "running") {
        $lastRunningPid = $null
        $lastSessionSteps = $null
    }
    elseif ($proc -and $metrics) {
        $currentPid = "$($proc.Pid)"
        $currentSessionSteps = Get-SessionStepsFromMetrics -MetricsLine $metrics
        if ($currentSessionSteps -ne $null) {
            if ($lastRunningPid -eq $currentPid -and $lastSessionSteps -ne $null -and $currentSessionSteps -lt $lastSessionSteps) {
                $detail = "pid=$currentPid session_steps=$currentSessionSteps prev_session_steps=$lastSessionSteps"
                Write-MonitorLine "session_steps_reset $detail"
                Write-RestartAudit -Event "session_steps_reset" -Reason "in_process_reset" -Details $detail
            }
            $lastRunningPid = $currentPid
            $lastSessionSteps = $currentSessionSteps
        }
    }

    if ($state -eq "running") {
        $consecutiveDown = 0
        $consecutiveStalled = 0
    }
    elseif ($state -eq "down") {
        $consecutiveDown += 1
        $consecutiveStalled = 0
    }
    elseif ($state -eq "stalled") {
        $consecutiveStalled += 1
        $consecutiveDown = 0
    }

    if ($AutoRestart) {
        $ageSinceRestart = (Get-Date) - $lastRestartAt
        $cooldownElapsed = $ageSinceRestart.TotalSeconds -ge $RestartCooldownSeconds
        $restartReason = $null

        if ($cooldownElapsed -and $consecutiveDown -ge $RestartAfterConsecutiveDown) {
            $restartReason = "down x$consecutiveDown"
        }
        elseif ($cooldownElapsed -and $consecutiveStalled -ge $RestartAfterConsecutiveStalled) {
            $restartReason = "stalled x$consecutiveStalled"
        }

        if ($restartReason) {
            $priorPid = if ($proc) { "$($proc.Pid)" } else { "" }
            $priorSessionSteps = if ($currentSessionSteps -ne $null) { "$currentSessionSteps" } else { "" }
            if (Invoke-TrainRestart -Reason $restartReason -PriorPid $priorPid -PriorSessionSteps $priorSessionSteps) {
                $lastRestartAt = Get-Date
                $consecutiveDown = 0
                $consecutiveStalled = 0
            }
        }
    }

    Start-Sleep -Seconds $PollSeconds
  }
  catch {
    # Defense in depth beyond the Get-TrainProcess try/catch: any other
    # unexpected error in a poll cycle (file I/O race, etc.) should not be
    # able to kill the whole watchdog silently the way the WSL-restart case
    # did on 2026-07-22.
    Write-MonitorLine "unexpected_error $($_.Exception.Message)"
    Write-RestartAudit -Event "unexpected_error" -Reason "monitor_loop" -Details $_.Exception.Message
    Start-Sleep -Seconds $PollSeconds
  }
}
