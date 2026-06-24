@echo off
REM ====================================================================
REM  run.bat  -  train the models and open the web app on Windows
REM  Run this AFTER setup.bat has finished and after you have placed
REM  creditcard.csv inside the data\ folder.
REM ====================================================================

call venv\Scripts\activate

if not exist "data\creditcard.csv" (
    echo.
    echo ERROR: data\creditcard.csv was not found.
    echo Download it from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    echo and place it inside the data\ folder, then run this again.
    pause
    exit /b 1
)

echo.
echo [1/2] Training and evaluating the models...
py main.py

echo.
echo [2/2] Opening the web interface in your browser...
py -m streamlit run app.py

pause
