@echo off
rem Foreground launcher for debugging. Production runs via supervisor.py + Startup shortcut.
cd /d "%~dp0"
"..\\.venv\Scripts\python.exe" render_worker.py
