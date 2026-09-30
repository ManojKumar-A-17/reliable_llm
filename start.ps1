# Reliable LLM - start backend (port 8000) and frontend (port 3000) on Windows.
# Easiest: double-click start.bat. Each part opens in its own window with logs.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$venvPy = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) { Write-Host "Run setup.bat first." -ForegroundColor Red; exit 1 }
if (-not (Test-Path "backend\.env")) { Write-Host "backend\.env is missing - run setup.bat first." -ForegroundColor Red; exit 1 }

$backend = Start-Process -PassThru -FilePath $venvPy `
    -WorkingDirectory (Join-Path $PSScriptRoot "backend") `
    -ArgumentList "-m", "uvicorn", "api.server:app", "--reload", "--reload-dir", "rllm", "--reload-dir", "api", "--port", "8000"

$frontend = Start-Process -PassThru -FilePath "cmd.exe" `
    -WorkingDirectory (Join-Path $PSScriptRoot "frontend") `
    -ArgumentList "/c", "npm run dev -- --port 3000"

Write-Host ""
Write-Host "  Reliable LLM is starting..." -ForegroundColor Cyan
Write-Host "  -> App:      http://localhost:3000"
Write-Host "  -> API docs: http://localhost:8000/docs"
Write-Host ""
Start-Sleep -Seconds 6
Start-Process "http://localhost:3000"

Read-Host "  Press Enter here to stop both servers"
foreach ($p in @($backend, $frontend)) {
    if ($p -and -not $p.HasExited) { & taskkill /T /F /PID $p.Id | Out-Null }
}
Write-Host "  Stopped."
