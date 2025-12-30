@echo off
REM ═══════════════════════════════════════════════════════════════════════════════
REM                    PAYFLOW ULTIMATE FRAUD ENGINE - RUN SCRIPTS
REM                    
REM   Full GPU Acceleration Test Suite for Hackxios 2K25
REM ═══════════════════════════════════════════════════════════════════════════════

echo.
echo ╔═══════════════════════════════════════════════════════════════════════════════╗
echo ║                                                                               ║
echo ║   🛡️  PAYFLOW ULTIMATE FRAUD DETECTION ENGINE                                ║
echo ║                                                                               ║
echo ║   🚀 RTX 4070 ^| 8GB VRAM Dedicated ^| Qwen3:8B + Thinking Mode               ║
echo ║                                                                               ║
echo ╚═══════════════════════════════════════════════════════════════════════════════╝
echo.

cd /d "%~dp0"

echo [1/5] Checking Python environment...
python --version

echo.
echo [2/5] Installing required packages...
pip install httpx fastapi uvicorn colorama numpy pydantic -q

echo.
echo [3/5] Checking Ollama and Qwen3:8B...
ollama list | findstr "qwen3:8b"
if errorlevel 1 (
    echo    ⚠️  Qwen3:8B not found. Please run: ollama pull qwen3:8b
    pause
    exit /b 1
)
echo    ✅ Qwen3:8B model found

echo.
echo [4/5] Warming up GPU model...
curl -s -X POST http://localhost:11434/api/generate -d "{\"model\": \"qwen3:8b\", \"prompt\": \"Hello\", \"stream\": false, \"options\": {\"num_gpu\": 99}, \"keep_alive\": \"30m\"}" > nul
echo    ✅ Model loaded to GPU VRAM

echo.
echo [5/5] Ready to run tests!
echo.
echo ═══════════════════════════════════════════════════════════════════════════════
echo.
echo Choose an option:
echo   [1] Run Quick Benchmark (10 transactions)
echo   [2] Run Standard Benchmark (50 transactions)
echo   [3] Run Full Benchmark (200 transactions)
echo   [4] Run Stress Test (500 transactions)
echo   [5] Run Comprehensive Test Suite (15 typologies)
echo   [6] Start API Server
echo   [7] Run Single Transaction Test
echo   [8] Exit
echo.
set /p choice="Enter choice (1-8): "

if "%choice%"=="1" (
    echo.
    echo Running Quick Benchmark...
    python benchmarkRunner.py --mode quick
) else if "%choice%"=="2" (
    echo.
    echo Running Standard Benchmark...
    python benchmarkRunner.py --mode standard
) else if "%choice%"=="3" (
    echo.
    echo Running Full Benchmark...
    python benchmarkRunner.py --mode full
) else if "%choice%"=="4" (
    echo.
    echo Running Stress Test...
    python benchmarkRunner.py --mode stress
) else if "%choice%"=="5" (
    echo.
    echo Running Comprehensive Test Suite...
    python comprehensiveTestSuite.py
) else if "%choice%"=="6" (
    echo.
    echo Starting Ultimate Fraud API Server on http://localhost:8000...
    echo API Docs: http://localhost:8000/docs
    python ultimateFraudApi.py
) else if "%choice%"=="7" (
    echo.
    echo Running Single Transaction Test...
    python ultimateFraudEngine.py
) else if "%choice%"=="8" (
    echo Goodbye!
    exit /b 0
) else (
    echo Invalid choice. Please try again.
)

echo.
pause
