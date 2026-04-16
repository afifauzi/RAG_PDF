@echo off
title Data Ingestion Pipeline

echo ===================================================
echo Starting Document Ingestion into ChromaDB...
echo ===================================================
echo.

:: Activate the virtual environment
call ..\.venv\Scripts\activate

:: Run the ingestion script
python ingest.py

echo.
echo ===================================================
echo Ingestion process finished.
echo ===================================================
pause