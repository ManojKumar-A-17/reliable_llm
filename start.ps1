# Reliable LLM - start backend (port 8000) and frontend (port 3000) on Windows.
# Easiest: double-click start.bat. Each part opens in its own window with logs.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$venvPy = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) { Write-Host "Run setup.bat first." -ForegroundColor Red; exit 1 }
if (-not (Test-Path "backend\.env")) { Write-Host "backend\.env is missing - run setup.bat first." -ForegroundColor Red; exit 1 }

# LM Studio's local server stops whenever the app restarts - start it if the app uses it.
if (Select-String -Path "backend\.env" -Pattern '^\s*LMSTUDIO_ENABLED\s*=\s*true' -Quiet) {
    $lms = Join-Path $env:USERPROFILE ".lmstudio\bin\lms.exe"
    if (Test-Path $lms) {
        Write-Host "  Starting LM Studio server..." -ForegroundColor Cyan
        & $lms server start | Out-Null
        if ($LASTEXITCODE -ne 0) { Write-Host "  Could not start the LM Studio server - open LM Studio and start it from the Developer tab." -ForegroundColor Yellow }
        # Preload the main model fully on the GPU so the first question doesn't wait for it.
        $model = (Select-String -Path "backend\.env" -Pattern '^\s*LMSTUDIO_MODEL\s*=\s*(\S+)').Matches | Select-Object -First 1
        if ($model) {
            $name = $model.Groups[1].Value
            if (-not ((& $lms ps 2>$null) -match [regex]::Escape($name))) {
                Write-Host "  Loading $name onto the GPU..." -ForegroundColor Cyan
                & $lms load $name --gpu max -c 8192 -y 2>$null | Out-Null
                if ($LASTEXITCODE -ne 0) { Write-Host "  Could not preload $name - LM Studio will load it on the first question." -ForegroundColor Yellow }
            }
        }
    } else {
        Write-Host "  LMSTUDIO_ENABLED=true but LM Studio's 'lms' tool was not found - start its server from the Developer tab." -ForegroundColor Yellow
    }
}

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
