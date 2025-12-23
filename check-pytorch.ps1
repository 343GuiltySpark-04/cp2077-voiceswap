[CmdletBinding()]
param(
    [string]$Python = "python"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Fail([string]$Msg) {
    Write-Error $Msg
    exit 1
}

# Prefer local venv if present (Windows-style)
if (Test-Path ".venv\Scripts\python.exe") {
    $Python = ".venv\Scripts\python.exe"
}

if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    Fail "Python executable '$Python' not found in PATH (or .venv)."
}

Write-Host "Python:" -NoNewline
& $Python --version

$code = @'
import sys

def line(k, v):
    print(f"{k}: {v}")

try:
    import torch
except Exception as e:
    print("ERROR: failed to import torch:", e)
    sys.exit(2)

line("PyTorch", torch.__version__)

# Backends: always safe to query defensively
cuda_ok = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
mps_ok  = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
hip_ver = getattr(getattr(torch.version, "hip", None), "strip", lambda: None)()
# torch.version.hip is usually a string or None; ROCm builds often report it
hip_ok  = torch.version.hip is not None

line("CUDA available", cuda_ok)
if cuda_ok:
    line("CUDA runtime", torch.version.cuda)
    try:
        line("CUDA device", torch.cuda.get_device_name(0))
    except Exception as e:
        line("CUDA device", f"(couldn't query name: {e})")

line("MPS available", mps_ok)

# HIP/ROCm is a bit messier: presence isn't always "is_available" like CUDA
line("HIP/ROCm build", hip_ok)
if hip_ok:
    line("HIP runtime", torch.version.hip)

# Choose a "best" device without assuming a vendor
if cuda_ok:
    dev = torch.device("cuda:0")
elif mps_ok:
    dev = torch.device("mps")
else:
    dev = torch.device("cpu")

line("Selected device", dev)

# Actually do a tiny compute test on the selected device
try:
    x = torch.randn((1024, 1024), device=dev)
    y = x @ x
    # Force sync where relevant so failures show up now
    if dev.type == "cuda":
        torch.cuda.synchronize()
    line("Compute test", "OK")
except Exception as e:
    print("ERROR: compute test failed:", e)
    sys.exit(3)

sys.exit(0)
'@

try {
    & $Python -c $code
}
catch {
    Fail "PyTorch check failed to run."
}