# Local launcher for Q-SAR Sentinel 2.0 (FastAPI backend + Next.js frontend)
$ErrorActionPreference = "Stop"

$Candidates = @(
    "$PSScriptRoot\.venv\Scripts\python.exe",
    "$PSScriptRoot\backend\.venv\Scripts\python.exe",
    "$env:USERPROFILE\.venvs\qsar-sentinel\Scripts\python.exe"
)

$PythonExe = $Candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $PythonExe) {
    $Cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($Cmd) { $PythonExe = $Cmd.Source }
}
if (-not $PythonExe) {
    Write-Error "Python interpreter not found. Create a venv (.venv) and install backend/requirements.txt first."
}

Write-Host "Starting Q-SAR Sentinel backend on http://127.0.0.1:8000 ..." -ForegroundColor Cyan
$backend = Start-Process -FilePath $PythonExe -ArgumentList "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000" -PassThru

Write-Host "Starting Q-SAR Sentinel frontend on http://127.0.0.1:3000 ..." -ForegroundColor Green
$frontend = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "npm run dev" -WorkingDirectory "$PSScriptRoot\frontend" -PassThru

Write-Host ""
Write-Host "Dashboard : http://127.0.0.1:3000" -ForegroundColor Yellow
Write-Host "API Docs  : http://127.0.0.1:8000/docs" -ForegroundColor Yellow
Write-Host "Press ENTER to stop both servers..."
Read-Host | Out-Null

if ($backend -and !$backend.HasExited) { Stop-Process -Id $backend.Id -Force }
if ($frontend -and !$frontend.HasExited) { taskkill /PID $frontend.Id /T /F | Out-Null }
