@echo off
REM One-time setup for Windows.
REM Run from inside the pumpkin-detector folder:   setup_windows.bat

echo == 1/4 Checking Python ==
python --version
if errorlevel 1 (
  echo Python not found. Install it from https://www.python.org/downloads/
  echo IMPORTANT: tick "Add python.exe to PATH" in the installer, then run this again.
  exit /b 1
)

echo == 2/4 Creating virtual environment (venv) ==
python -m venv venv
call venv\Scripts\activate.bat

echo == 3/4 Installing packages (this can take 5-10 minutes) ==
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo == 4/4 Downloading AI models and testing everything ==
python smoke_test.py

echo.
echo All done! Next time, start with:   venv\Scripts\activate
