@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m great_sage.core.progress_cues --disable
