# =========================
# ORT Doctor
# =========================
# - Detects CUDA Toolkit v12.x bin
# - Detects pip-installed NVIDIA bins inside venv
# - Appends to USER PATH safely (never overwrites)
# - Backs up USER PATH to %TEMP%
# - No registry footguns
# - Idempotent
# =========================

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# -------------------------
# Helpers
# -------------------------

function Resolve-VenvPath {
    if ($env:VIRTUAL_ENV) {
        $p = (Resolve-Path $env:VIRTUAL_ENV).Path
        if (Test-Path (Join-Path $p "Scripts\python.exe")) {
            return $p
        }
    }
    return $null
}

function Get-UserPathParts {
    $cur = [Environment]::GetEnvironmentVariable("Path", "User")
    if ([string]::IsNullOrWhiteSpace($cur)) { return @() }
    return @($cur -split ';' | Where-Object { $_ })
}

function Backup-UserPath {
    $cur = [Environment]::GetEnvironmentVariable("Path", "User")
    $ts = Get-Date -Format "yyyyMMdd-HHmmss"
    $file = Join-Path $env:TEMP "UserPathBackup-$ts.txt"
    $cur | Out-File -Encoding UTF8 $file
    Write-Host "Backed up USER PATH to: $file" -ForegroundColor DarkCyan
}

function Add-ToUserPathSafely {
    param(
        [string[]]$PathsToAdd,
        [switch]$Apply
    )

    $PathsToAdd = @($PathsToAdd) | Where-Object { $_ -and $_.Trim() -ne "" }
    if ($PathsToAdd.Length -eq 0) { return }

    $curParts = Get-UserPathParts

    $set = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($p in $curParts) { [void]$set.Add($p.TrimEnd('\')) }

    $added = @()
    foreach ($p in $PathsToAdd) {
        if (-not (Test-Path $p)) {
            Write-Host "Skipping missing path: $p" -ForegroundColor DarkYellow
            continue
        }
        $norm = $p.TrimEnd('\')
        if (-not $set.Contains($norm)) {
            $curParts += $p
            [void]$set.Add($norm)
            $added += $p
        }
    }

    if ($added.Length -eq 0) {
        Write-Host "USER PATH already contains all targets." -ForegroundColor Cyan
        return
    }

    Backup-UserPath

    if (-not $Apply) {
        Write-Host "DRY RUN — would append to USER PATH:" -ForegroundColor Green
        $added | ForEach-Object { Write-Host "  $_" }
        Write-Host "Re-run with -Apply to make changes" -ForegroundColor Cyan
        return
    }

    [Environment]::SetEnvironmentVariable("Path", ($curParts -join ';'), "User")

    Write-Host "Appended to USER PATH:" -ForegroundColor Green
    $added | ForEach-Object { Write-Host "  $_" }
    Write-Host "Open a NEW terminal to see changes." -ForegroundColor Green
}

# -------------------------
# Detection
# -------------------------

function Get-LatestCuda12Bin {
    $root = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if (-not (Test-Path $root)) { return $null }

    $dirs = @(
        Get-ChildItem $root -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^v12\.' } |
        Sort-Object { [version]($_.Name.TrimStart('v')) }
    )

    if ($dirs.Length -eq 0) { return $null }

    $bin = Join-Path $dirs[-1].FullName "bin"
    if (Test-Path $bin) { return $bin }
    return $null
}

function Get-PipNvidiaBins {
    param([string]$Venv)

    if (-not $Venv) { return @() }

    $base = Join-Path $Venv "Lib\site-packages\nvidia"
    if (-not (Test-Path $base)) { return @() }

    $bins = @()
    foreach ($d in Get-ChildItem $base -Directory -ErrorAction SilentlyContinue) {
        $b = Join-Path $d.FullName "bin"
        if (Test-Path $b) { $bins += $b }
    }
    return @($bins | Select-Object -Unique)
}

# -------------------------
# Main
# -------------------------

Write-Host "`n=== ORT Doctor ===`n" -ForegroundColor Cyan

$venv = Resolve-VenvPath
if ($venv) {
    Write-Host "Using venv: $venv" -ForegroundColor Cyan
}
else {
    Write-Host "No active venv detected." -ForegroundColor Yellow
}

$targets = @()

$cudaBin = Get-LatestCuda12Bin
if ($cudaBin) {
    Write-Host "Found CUDA Toolkit bin:" -ForegroundColor DarkCyan
    Write-Host "  $cudaBin"
    $targets += $cudaBin
}
else {
    Write-Host "CUDA Toolkit v12.x not found." -ForegroundColor DarkYellow
}

$pipBins = Get-PipNvidiaBins -Venv $venv
if ($pipBins.Length -gt 0) {
    Write-Host "Found pip NVIDIA bins:" -ForegroundColor DarkCyan
    $pipBins | ForEach-Object { Write-Host "  $_" }
    $targets += $pipBins
}
else {
    Write-Host "No pip NVIDIA bins found." -ForegroundColor DarkYellow
}

$targets = @($targets | Select-Object -Unique)

if ($targets.Length -eq 0) {
    Write-Host "Nothing to add to PATH." -ForegroundColor Yellow
    exit 0
}

Add-ToUserPathSafely -PathsToAdd $targets

Write-Host "`nReview the above. If correct, re-run with -Apply." -ForegroundColor Cyan
