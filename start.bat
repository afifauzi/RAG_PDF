@echo off
cd /d "%~dp0"

echo Activating environment...
call ..\.venv\Scripts\activate

:: Force the app to only see the first GPU
set CUDA_VISIBLE_DEVICES=0

echo 🚀 Starting FastAPI Backend Server...
:: Launch backend.py (which now runs uvicorn) in a separate window
start "FastAPI Backend" cmd /k "call ..\.venv\Scripts\activate && python backend.py"

:: Give the FastAPI server 5 seconds to boot up and load models into memory
timeout /t 5 /nobreak >nul

echo 🚀 Launching Streamlit App...
streamlit run app.py
pause