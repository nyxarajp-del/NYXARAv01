@echo off
REM ===========================================================================
REM  NYXARA — one-click launcher for Windows (local DistilGPT-2 brain, CPU).
REM
REM  Double-click this file. The FIRST run sets everything up automatically:
REM    * ensures Python is installed (via winget if missing),
REM    * creates a private virtual environment (.venv),
REM    * installs NYXARA + the local-model dependencies,
REM    * writes a .env pointing at the local DistilGPT-2 brain on CPU,
REM    * downloads the model weights once (~350 MB) on first launch.
REM  EVERY run after that just starts NYXARA instantly — no re-download.
REM
REM  Nothing here is destructive: it only creates .venv/.env if they are
REM  missing and never touches files you already have.
REM ===========================================================================
setlocal enableextensions
title NYXARA

REM --- run from the repository root, regardless of where this .bat lives ---
cd /d "%~dp0.."

REM --- 1. Ensure Python is available -----------------------------------------
where python >nul 2>&1
if errorlevel 1 (
    echo [setup] Python not found. Installing it via winget...
    winget install -e --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements
    echo.
    echo [setup] Python installed. Please CLOSE this window and double-click
    echo         nyxara.bat again so Windows can see the new Python.
    echo.
    pause
    exit /b 0
)

REM --- 2. First-run environment setup (only if .venv is missing) --------------
if not exist ".venv\Scripts\python.exe" (
    echo [setup] First-time setup — this happens only once, please wait...
    python -m venv .venv
    if errorlevel 1 (
        echo [error] Could not create the virtual environment. Is Python OK?
        pause
        exit /b 1
    )
    call ".venv\Scripts\activate.bat"
    python -m pip install --upgrade pip
    echo [setup] Installing NYXARA + local-model dependencies (torch is large)...
    pip install -e ".[distilgpt2,reasoning]"
    if errorlevel 1 (
        echo [error] Dependency install failed. Check your internet connection.
        pause
        exit /b 1
    )
) else (
    call ".venv\Scripts\activate.bat"
)

REM --- 3. Write .env: GLM-5 primary, her own local brains as the floor (only if missing) -----
if not exist ".env" (
    echo [setup] Writing .env: GLM-5 (airouter) primary, native own-brain floor...
    >  ".env" echo NYXARA_PROFILE=dev
    >> ".env" echo NYXARA_LLM__PROVIDER=airouter
    >> ".env" echo NYXARA_LLM__AIROUTER_MODEL=zai/glm-5
        >> ".env" echo NYXARA_FEATURES__MULTI_LLM_COUNCIL=false
    >> ".env" echo NYXARA_COUNCIL__ENABLED=false
)

REM --- 4. Launch ------------------------------------------------------------
echo.
echo [nyxara] Starting... (the very first launch downloads the model once)
echo.
python -m nyxara

echo.
echo [nyxara] Session ended.
pause
