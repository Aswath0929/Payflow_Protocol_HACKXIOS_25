# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                    PAYFLOW LIVE DEPLOYMENT SCRIPT                             ║
# ║                                                                               ║
# ║  This script deploys the frontend to Vercel and exposes the AI backend       ║
# ║  via ngrok so external users can access the full system.                      ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

param(
    [string]$NgrokAuthToken
)

$ErrorActionPreference = "Continue"

Write-Host ""
Write-Host "╔═══════════════════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                    🚀 PAYFLOW LIVE DEPLOYMENT                                 ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Step 1: Configure ngrok if token provided
if ($NgrokAuthToken) {
    Write-Host "[1/5] Configuring ngrok with authtoken..." -ForegroundColor Yellow
    ngrok config add-authtoken $NgrokAuthToken
    if ($LASTEXITCODE -eq 0) {
        Write-Host "      ✅ ngrok configured successfully" -ForegroundColor Green
    } else {
        Write-Host "      ❌ Failed to configure ngrok" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[1/5] Checking ngrok configuration..." -ForegroundColor Yellow
    $ngrokCheck = ngrok config check 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "      ❌ ngrok not configured. Please run with -NgrokAuthToken parameter" -ForegroundColor Red
        Write-Host "      Get your token at: https://dashboard.ngrok.com/get-started/your-authtoken" -ForegroundColor Yellow
        exit 1
    }
    Write-Host "      ✅ ngrok already configured" -ForegroundColor Green
}

# Step 2: Check if FastAPI is running
Write-Host "[2/5] Checking FastAPI server..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 5
    Write-Host "      ✅ FastAPI running - Qwen3: $($health.qwen3_available)" -ForegroundColor Green
} catch {
    Write-Host "      ⚠️ FastAPI not running, starting it..." -ForegroundColor Yellow
    $env:PYTHONIOENCODING = "utf-8"
    Push-Location "C:\Users\sayan\Downloads\Hackxios\theblocks\packages\nextjs\services\ai"
    Start-Process python -ArgumentList "-m","uvicorn","api:app","--host","0.0.0.0","--port","8000" -WindowStyle Minimized
    Pop-Location
    Start-Sleep -Seconds 12
    try {
        $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 5
        Write-Host "      ✅ FastAPI started successfully" -ForegroundColor Green
    } catch {
        Write-Host "      ❌ Failed to start FastAPI" -ForegroundColor Red
        exit 1
    }
}

# Step 3: Start ngrok tunnel
Write-Host "[3/5] Starting ngrok tunnel on port 8000..." -ForegroundColor Yellow
# Kill any existing ngrok processes
Get-Process -Name ngrok -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# Start ngrok in background
Start-Process ngrok -ArgumentList "http","8000","--log=stdout" -WindowStyle Minimized
Start-Sleep -Seconds 5

# Get the public URL from ngrok API
try {
    $tunnels = Invoke-RestMethod -Uri "http://localhost:4040/api/tunnels" -TimeoutSec 10
    $publicUrl = $tunnels.tunnels | Where-Object { $_.proto -eq "https" } | Select-Object -First 1 -ExpandProperty public_url
    
    if ($publicUrl) {
        Write-Host "      ✅ ngrok tunnel active: $publicUrl" -ForegroundColor Green
    } else {
        Write-Host "      ❌ Could not get ngrok public URL" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "      ❌ ngrok API not responding. Check ngrok window for errors." -ForegroundColor Red
    exit 1
}

# Step 4: Deploy to Vercel with the ngrok URL
Write-Host "[4/5] Deploying frontend to Vercel..." -ForegroundColor Yellow
Push-Location "C:\Users\sayan\Downloads\Hackxios\theblocks\packages\nextjs"

# Set the environment variable for the build
$env:NEXT_PUBLIC_AI_ORACLE_URL = $publicUrl

# Deploy to production
Write-Host "      Setting AI Oracle URL to: $publicUrl" -ForegroundColor Cyan
vercel --prod --yes --env NEXT_PUBLIC_AI_ORACLE_URL=$publicUrl 2>&1 | Tee-Object -Variable vercelOutput

Pop-Location

# Extract the deployment URL
$deploymentUrl = $vercelOutput | Select-String -Pattern "https://.*\.vercel\.app" | Select-Object -First 1
if ($deploymentUrl) {
    $deploymentUrl = $deploymentUrl.Matches[0].Value
}

Write-Host ""
Write-Host "╔═══════════════════════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║                    🎉 DEPLOYMENT COMPLETE!                                    ║" -ForegroundColor Green
Write-Host "╠═══════════════════════════════════════════════════════════════════════════════╣" -ForegroundColor Green
Write-Host "║                                                                               ║" -ForegroundColor Green
Write-Host "║  LIVE URLS:                                                                   ║" -ForegroundColor Green
Write-Host "║  ─────────                                                                    ║" -ForegroundColor Green
Write-Host "║  🌐 Frontend:  $deploymentUrl" -ForegroundColor Cyan
Write-Host "║  🤖 AI Oracle: $publicUrl" -ForegroundColor Cyan
Write-Host "║  📚 API Docs:  $publicUrl/docs" -ForegroundColor Cyan
Write-Host "║                                                                               ║" -ForegroundColor Green
Write-Host "║  KEEP RUNNING:                                                                ║" -ForegroundColor Green
Write-Host "║  ─────────────                                                                ║" -ForegroundColor Green
Write-Host "║  • This terminal (ngrok tunnel)                                               ║" -ForegroundColor Yellow
Write-Host "║  • FastAPI server (minimized window)                                          ║" -ForegroundColor Yellow
Write-Host "║  • Ollama with Qwen3 (background)                                             ║" -ForegroundColor Yellow
Write-Host "║                                                                               ║" -ForegroundColor Green
Write-Host "║  ⚠️  If you close this terminal, external users won't be able to use AI!     ║" -ForegroundColor Yellow
Write-Host "║                                                                               ║" -ForegroundColor Green
Write-Host "╚═══════════════════════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""

# Keep the script running to maintain the tunnel
Write-Host "Press Ctrl+C to stop the deployment and close the tunnel..." -ForegroundColor Gray
while ($true) { Start-Sleep -Seconds 60 }
