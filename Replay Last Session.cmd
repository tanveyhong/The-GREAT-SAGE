@echo off
cd /d "%~dp0"
rem Replays the newest Claude session (last 10 minutes, 10x) into the running Companion.
".venv\Scripts\python.exe" -m great_sage.core.replay %*
pause
