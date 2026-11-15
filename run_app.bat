@echo off
REM Double-click to start the Livestock Counter on Windows.
cd /d "%~dp0"
if not exist venv (
  python -m venv venv
  call venv\Scripts\activate.bat
  python -m pip install --upgrade pip
  pip install -r requirements.txt
) else (
  call venv\Scripts\activate.bat
)
python app.py
pause
