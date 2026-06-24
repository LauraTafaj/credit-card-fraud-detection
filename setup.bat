@echo off
REM ====================================================================
REM  setup.bat  -  one-time setup on Windows
REM  Double-click this file (or run it in PowerShell) to:
REM     1. create a virtual environment
REM     2. install all the required libraries
REM  You only need to run this ONCE.
REM ====================================================================

echo.
echo [1/2] Creating the virtual environment (venv)...
py -m venv venv
if errorlevel 1 (
    echo.
    echo ERROR: Could not run "py". Python may not be installed.
    echo Install it from https://www.python.org/downloads/ and tick
    echo "Add python.exe to PATH" during installation, then run this again.
    pause
    exit /b 1
)

echo.
echo [2/2] Installing the required libraries...
call venv\Scripts\activate
py -m pip install --upgrade pip
py -m pip install -r requirements.txt

echo.
echo ====================================================================
echo  Setup finished!
echo  Next: put creditcard.csv into the data\ folder, then run  run.bat
echo ====================================================================
pause
