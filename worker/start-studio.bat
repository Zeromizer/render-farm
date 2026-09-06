@echo off
rem H3 Studio: local web UI for the video_gen engine (prompt, library, upscale, 60 fps, car turntable).
rem Binds 127.0.0.1 only. Needs ..\.env (Supabase service key) like the worker.
cd /d "%~dp0"
start "" http://127.0.0.1:8790
"..\.venv\Scripts\python.exe" studio\server.py
