@echo off
title TRPG Launcher

set OLLAMA=C:\Users\Wesley\AppData\Local\Programs\Ollama\ollama.exe
set PYTHON=%~dp0gemma-env\Scripts\python.exe

set MODE=%1
if "%MODE%"=="" set MODE=web

echo ================================================
echo   TRPG LLM Engine - mode: %MODE%
echo ================================================
echo.

echo [1/3] Restarting Ollama with GGML_FLASH_ATTENTION=0 ...
taskkill /F /IM ollama.exe >nul 2>&1
timeout /t 2 /nobreak >nul

set GGML_FLASH_ATTENTION=0
start /min "Ollama" "%OLLAMA%" serve

echo [2/3] Waiting for Ollama to be ready ...
:wait_loop
timeout /t 2 /nobreak >nul
"%OLLAMA%" list >nul 2>&1
if errorlevel 1 goto wait_loop

echo [3/3] Starting game ...
echo.
cd /d "%~dp0"
"%PYTHON%" -m trpg --mode %MODE%

pause
