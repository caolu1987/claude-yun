@echo off
rem Local Transcriber launcher. Double-click to start; the browser opens automatically.
chcp 65001 >nul
cd /d "%~dp0"
title Local Transcriber
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
"%~dp0python\python.exe" "%~dp0app\main.py" %*
if errorlevel 1 pause
