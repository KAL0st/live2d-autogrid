@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
"C:\Users\WINDOWS\AppData\Local\Python\pythoncore-3.14-64\python.exe" "run.py" --model %*
echo.
pause
