param(
	[string]$VenvPath = "..\AI_env",
	[int]$Port = 8000
)

$ErrorActionPreference = "Stop"

# Resolve paths relative to this script
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
$AppPath = Join-Path $RepoRoot "app\main.py"
$ActivatePs1Path = Join-Path $RepoRoot (Join-Path $VenvPath "Scripts\Activate.ps1")

if (-not (Test-Path $AppPath)) {
	Write-Error "App not found at $AppPath"
}

# Activate venv if activation script exists and we're not already in a venv
$AlreadyVenv = $null -ne $env:VIRTUAL_ENV -and $env:VIRTUAL_ENV -ne ""
if (-not $AlreadyVenv -and (Test-Path $ActivatePs1Path)) {
	Write-Host "Activating venv at $VenvPath"
	. $ActivatePs1Path
} elseif (-not $AlreadyVenv) {
	Write-Host "Virtual env not found at $VenvPath. Proceeding without activation..."
}

# Install requirements if present
$ReqPath = Join-Path $RepoRoot "requirements.txt"
if (Test-Path $ReqPath) {
	Write-Host "Installing requirements from $ReqPath"
	pip install -r $ReqPath
} else {
	Write-Host "requirements.txt not found at $ReqPath; skipping install"
}

Write-Host "Starting server on 0.0.0.0:$Port"
python -m uvicorn app.main:app --host 0.0.0.0 --port $Port
