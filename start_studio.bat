@echo off
title ComfyUI Studio
cd /d "%~dp0"
set "CURL=%SystemRoot%\System32\curl.exe"

rem Already running? Just open the page.
"%CURL%" -s -o nul -m 2 http://127.0.0.1:7860/api/status && (start "" http://127.0.0.1:7860 & exit /b)

set "COMFY_EXE=%COMFY_DESKTOP_EXE%"
if not defined COMFY_EXE set "COMFY_EXE=%ProgramFiles%\Comfy Desktop\Comfy Desktop.exe"

"%CURL%" -s -o nul -m 2 http://127.0.0.1:8188/system_stats
if errorlevel 1 (
  if exist "%COMFY_EXE%" (
    echo Starting ComfyUI...
    start "" "%COMFY_EXE%"
  ) else (
    echo ComfyUI was not found at "%COMFY_EXE%" - please start it yourself.
  )
  echo Waiting for ComfyUI to be ready...
  for /l %%i in (1,1,90) do (
    "%CURL%" -s -o nul -m 2 http://127.0.0.1:8188/system_stats && goto ready
    "%SystemRoot%\System32\PING.EXE" -n 3 127.0.0.1 >nul
  )
  echo ComfyUI is still starting - the page will connect by itself when it is ready.
)

:ready
echo Keep this window open while you use ComfyUI Studio. Close it to stop the site.
".venv\Scripts\python.exe" -m web
pause
