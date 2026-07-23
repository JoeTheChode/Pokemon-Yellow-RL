param(
    [string]$CondaEnv = "ml-env"
)

$repoRoot = (Resolve-Path $PSScriptRoot).Path

$command = @"
conda activate $CondaEnv
cd /d `"$repoRoot`"
python watch.py
"@

Start-Process -FilePath cmd.exe -ArgumentList "/k", $command -WorkingDirectory $repoRoot
