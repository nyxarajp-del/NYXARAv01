@echo off
REM ===========================================================================
REM  NYXARA — Windows installer / bootstrapper.
REM
REM  Save this file anywhere (e.g. your Desktop) and double-click it. It:
REM    1. installs Git and Python if they are missing (via winget),
REM    2. downloads (clones) NYXARA into %USERPROFILE%\NYXARA,
REM    3. hands off to scripts\nyxara.bat, which sets up the environment and
REM       starts NYXARA on the local DistilGPT-2 brain.
REM
REM  You only run this ONCE. After that, launch NYXARA any time by double-
REM  clicking  %USERPROFILE%\NYXARA\scripts\nyxara.bat .
REM ===========================================================================
setlocal enableextensions
title NYXARA installer

set "DEST=%USERPROFILE%\NYXARA"
set "REPO=https://github.com/nyxarajp-del/NYXARAv01.git"
set "BRANCH=main"

REM --- 1. Ensure Git --------------------------------------------------------
where git >nul 2>&1
if errorlevel 1 (
    echo [setup] Git not found. Installing via winget...
    winget install -e --id Git.Git --accept-package-agreements --accept-source-agreements
    echo.
    echo [setup] Git installed. Please CLOSE this window and double-click this
    echo         installer again so Windows can see Git.
    pause
    exit /b 0
)

REM --- 2. Ensure Python -----------------------------------------------------
where python >nul 2>&1
if errorlevel 1 (
    echo [setup] Python not found. Installing via winget...
    winget install -e --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements
    echo.
    echo [setup] Python installed. Please CLOSE this window and double-click this
    echo         installer again so Windows can see Python.
    pause
    exit /b 0
)

REM --- 3. Clone (or update) NYXARA ------------------------------------------
if not exist "%DEST%\.git" (
    echo [setup] Downloading NYXARA into "%DEST%" ...
    echo         ^(A GitHub sign-in window may appear the first time.^)
    git clone --branch %BRANCH% %REPO% "%DEST%"
    if errorlevel 1 (
        echo [error] Download failed. Check your internet / GitHub sign-in.
        pause
        exit /b 1
    )
) else (
    echo [setup] NYXARA already present — updating...
    cd /d "%DEST%"
    git pull
)

REM --- 4. Hand off to the launcher ------------------------------------------
call "%DEST%\scripts\nyxara.bat"
