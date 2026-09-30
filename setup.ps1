# Reliable LLM - one-time setup for Windows (PowerShell 5.1+ or 7+).
# Easiest: double-click setup.bat. Or run:
#   powershell -ExecutionPolicy Bypass -File setup.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Say($msg) { Write-Host "`n> $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "`nERROR: $msg" -ForegroundColor Red; exit 1 }

# --- Find a real Python 3.10+ (skip the Microsoft Store alias stub) ---
$pyExe = $null
foreach ($cand in @(@("py", "-3"), @("python"), @("python3"))) {
    if (-not (Get-Command $cand[0] -ErrorAction SilentlyContinue)) { continue }
    try {
        $extra = @()
        if ($cand.Count -gt 1) { $extra += $cand[1..($cand.Count - 1)] }
        $out = & $cand[0] @extra -c "import sys; print(sys.executable); print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $out.Count -ge 2) {
            $ver = [version]$out[1]
            if ($ver -ge [version]"3.10") { $pyExe = $out[0]; break }
        }
    } catch { }
}
if (-not $pyExe) {
    Fail "Python 3.10+ not found. Install it with:  winget install -e --id Python.Python.3.12`n       (or from python.org - tick 'Add python.exe to PATH'), then open a NEW terminal."
}
Say "Using Python: $pyExe"

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Fail "Node.js not found. Install it with:  winget install -e --id OpenJS.NodeJS.LTS  then open a NEW terminal."
}

# --- Python virtual environment + backend packages ---
if (-not (Test-Path "venv\Scripts\python.exe")) {
    Say "Creating virtual environment (venv\)"
    & $pyExe -m venv venv
    if ($LASTEXITCODE -ne 0) { Fail "Could not create venv" }
}
$venvPy = Join-Path $PSScriptRoot "venv\Scripts\python.exe"

Say "Installing backend packages (first time takes several minutes)"
& $venvPy -m pip install --upgrade pip
& $venvPy -m pip install -r backend\requirements.txt
if ($LASTEXITCODE -ne 0) { Fail "pip install failed - see the messages above" }

# --- Frontend packages ---
Say "Installing frontend packages"
Push-Location frontend
npm install
$npmOk = ($LASTEXITCODE -eq 0)
Pop-Location
if (-not $npmOk) { Fail "npm install failed - see the messages above" }

# --- Config file ---
if (-not (Test-Path "backend\.env")) {
    Copy-Item "backend\.env.example" "backend\.env"
    Say "Created backend\.env - open it and add an API key (or enable Ollama). See SETUP.md step 4."
} else {
    Say "backend\.env already exists - left unchanged"
}

# --- Offline tests ---
Say "Running backend tests (offline)"
Push-Location backend
& $venvPy -m pytest -q tests
$testsOk = ($LASTEXITCODE -eq 0)
Pop-Location
if (-not $testsOk) { Write-Host "Some tests failed - see above." -ForegroundColor Yellow }

Say "Setup complete. Start the app with start.bat (or: powershell -ExecutionPolicy Bypass -File start.ps1)"
