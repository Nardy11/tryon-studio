# Start the try-on API on http://127.0.0.1:8000
# Usage:  .\server\run.ps1            (from the project root)
#         $env:TRYON_MAX_SIDE = 768; .\server\run.ps1   (faster, lighter on VRAM)
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here
# Load models only from the local cache (download them first with scripts\download_models.py).
# Keeps a flaky connection from breaking a model load halfway.
if (-not $env:HF_HUB_OFFLINE) { $env:HF_HUB_OFFLINE = '1' }
& "$here\.venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
