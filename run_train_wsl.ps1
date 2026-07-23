param(
    [string]$Distro = "Ubuntu2404GPU",
    [string]$Gpu = "0",
    [string]$Python = "python3",
    [switch]$NoStream,
    [string]$StreamUser = "SVER-PokemonYellow",
    [string]$StreamColor = "#FFD700"
)

$repoWin = (Resolve-Path $PSScriptRoot).Path
$drive = $repoWin.Substring(0, 1).ToLower()
$rest = $repoWin.Substring(2).Replace('\', '/')
$repoWsl = "/mnt/$drive$rest"
# Streaming defaults ON (not opt-in) specifically because automatic restart
# paths -- monitor_train.ps1's watchdog and ensure_training_running.ps1's
# logon recovery -- both call start_train_wsl.ps1 with no knowledge of a
# -Stream flag. An opt-in default meant any auto-restart silently dropped
# streaming even after a manual restart had explicitly turned it on
# (confirmed happening 2026-07-22 via /proc/<pid>/environ on the actually
# running process). Opt out with -NoStream if ever needed.
$streamEnabledValue = if ($NoStream) { "0" } else { "1" }

$command = @"
cd '$repoWsl'
chmod +x ./run_train_wsl.sh
export HIP_VISIBLE_DEVICES='$Gpu'
export POKEMON_TRAIN_DEVICE='cuda'
export POKEMON_STREAM_ENABLED='$streamEnabledValue'
export POKEMON_STREAM_USER='$StreamUser'
export POKEMON_STREAM_COLOR='$StreamColor'
./run_train_wsl.sh
"@

wsl -d $Distro --exec /bin/sh -lc $command
