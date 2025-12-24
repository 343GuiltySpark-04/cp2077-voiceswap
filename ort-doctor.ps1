<#
.SYNOPSIS
  Adds CUDA/cuDNN related bin directories to the USER PATH dynamically.

.DESCRIPTION
  - Detects active venv via $env:VIRTUAL_ENV, or you can pass -VenvPath.
  - Adds newest CUDA v12.x Toolkit bin (if installed).
  - Adds pip-provided NVIDIA bins under: <venv>\Lib\site-packages\nvidia\*\bin
    (e.g. nvidia\cudnn\bin, nvidia\cublas\bin, etc.)
  - Idempotent: won't add duplicates.
  - Broadcasts WM_SETTINGCHANGE so new processes see it.

.PARAMETER VenvPath
  Path to a Python venv root (the folder containing Scripts\python.exe).
#>

[CmdletBinding()]
param(
    [string]$VenvPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-VenvPath {
    param([string]$Candidate)
    if ($Candidate) {
        $p = (Resolve-Path -LiteralPath $Candidate).Path
        if (Test-Path -LiteralPath (Join-Path $p "Scripts\python.exe")) { return $p }
        throw "VenvPath '$Candidate' does not look like a venv (missing Scripts\python.exe)."
    }

    if ($env:VIRTUAL_ENV) {
        $p = (Resolve-Path -LiteralPath $env:VIRTUAL_ENV).Path
        if (Test-Path -LiteralPath (Join-Path $p "Scripts\python.exe")) { return $p }
    }

    return $null
}

function Get-LatestCuda12Bin {
    $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if (-not (Test-Path -LiteralPath $cudaRoot)) { return $null }

    $dirs = @(
        Get-ChildItem -LiteralPath $cudaRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^v12\.' } |
        Sort-Object {
            try { [version]($_.Name.TrimStart('v')) } catch { [version]"0.0" }
        }
    )

    if ($dirs.Length -eq 0) { return $null }

    $bin = Join-Path $dirs[-1].FullName "bin"
    if (Test-Path -LiteralPath $bin) { return $bin }
    return $null
}

function Get-PipNvidiaBinsFromVenv {
    param([string]$Venv)

    if (-not $Venv) { return @() }

    $site = Join-Path $Venv "Lib\site-packages"
    $nvidia = Join-Path $site "nvidia"
    if (-not (Test-Path -LiteralPath $nvidia)) { return @() }

    $bins = @()

    foreach ($d in (Get-ChildItem -LiteralPath $nvidia -Directory -ErrorAction SilentlyContinue)) {
        $bin = Join-Path $d.FullName "bin"
        if (Test-Path -LiteralPath $bin) { $bins += $bin }
    }

    return @($bins | Select-Object -Unique)
}

function Get-UserPath {
    $regPath = "HKCU:\Environment"
    try {
        return [string](Get-ItemProperty -LiteralPath $regPath -Name "Path" -ErrorAction Stop).Path
    }
    catch {
        return ""
    }
}

function Set-UserPath([string]$newValue) {
    $regPath = "HKCU:\Environment"
    New-Item -Path $regPath -Force | Out-Null
    Set-ItemProperty -LiteralPath $regPath -Name "Path" -Value $newValue -Type ExpandString
}

function Add-ToUserPath([string[]]$pathsToAdd) {
    $current = Get-UserPath
    $parts = @()
    if (-not [string]::IsNullOrWhiteSpace($current)) {
        $parts = $current -split ';' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    }

    # Normalize existing entries
    $existing = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($p in $parts) { [void]$existing.Add($p.TrimEnd('\')) }

    $added = @()
    foreach ($p in $pathsToAdd) {
        if ([string]::IsNullOrWhiteSpace($p)) { continue }
        $norm = $p.TrimEnd('\')
        if (-not $existing.Contains($norm)) {
            $parts += $p
            [void]$existing.Add($norm)
            $added += $p
        }
    }

    if ($added.Count -gt 0) {
        Set-UserPath (($parts) -join ';')
    }

    return , $added
}

function Broadcast-EnvChange {
    $csharp = @"
using System;
using System.Runtime.InteropServices;

public static class NativeMethods
{
    [DllImport("user32.dll", SetLastError=true, CharSet=CharSet.Auto)]
    public static extern IntPtr SendMessageTimeout(
        IntPtr hWnd,
        int Msg,
        IntPtr wParam,
        string lParam,
        int flags,
        int timeout,
        out IntPtr lpdwResult
    );
}
"@

    Add-Type -TypeDefinition $csharp -ErrorAction SilentlyContinue

    $HWND_BROADCAST = [IntPtr]0xffff
    $WM_SETTINGCHANGE = 0x001A
    $SMTO_ABORTIFHUNG = 0x0002
    $result = [IntPtr]::Zero

    [NativeMethods]::SendMessageTimeout(
        $HWND_BROADCAST,
        $WM_SETTINGCHANGE,
        [IntPtr]::Zero,
        "Environment",
        $SMTO_ABORTIFHUNG,
        5000,
        [ref]$result
    ) | Out-Null
}

# --- Main ---
$venv = Resolve-VenvPath -Candidate $VenvPath
if (-not $venv) {
    Write-Host "No venv detected. Activate a venv or pass -VenvPath." -ForegroundColor Yellow
}
else {
    Write-Host "Using venv: $venv" -ForegroundColor Cyan
}

$targets = @()

# CUDA Toolkit v12.x bin (if installed)
$cudaBin = Get-LatestCuda12Bin
if ($cudaBin) {
    $targets += $cudaBin
    Write-Host "Found CUDA Toolkit bin: $cudaBin" -ForegroundColor DarkCyan
}
else {
    Write-Host "CUDA Toolkit v12.x not found (that's OK if you rely on pip NVIDIA wheels)." -ForegroundColor DarkYellow
}

# pip NVIDIA bins inside venv
$pipBins = Get-PipNvidiaBinsFromVenv -Venv $venv
if ($pipBins.Count -gt 0) {
    Write-Host "Found pip NVIDIA bins:" -ForegroundColor DarkCyan
    $pipBins | ForEach-Object { Write-Host "  $_" }
    $targets += $pipBins
}
else {
    Write-Host "No pip NVIDIA bins found under site-packages\nvidia\*\bin." -ForegroundColor DarkYellow
}

$targets = $targets | Select-Object -Unique
if ($targets.Count -eq 0) {
    Write-Host "Nothing to add to PATH." -ForegroundColor Yellow
    exit 1
}

$added = Add-ToUserPath -pathsToAdd $targets
if ($added.Count -gt 0) {
    Broadcast-EnvChange
    Write-Host "Added to USER PATH:" -ForegroundColor Green
    $added | ForEach-Object { Write-Host "  $_" }
    Write-Host "Open a new terminal (or restart apps) to see the updated PATH." -ForegroundColor Green
}
else {
    Write-Host "All target paths already present in USER PATH." -ForegroundColor Cyan
}
