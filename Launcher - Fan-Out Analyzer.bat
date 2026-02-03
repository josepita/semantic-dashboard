@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo  Fan-Out Query Analyzer
echo ========================================
echo.

REM Verificar entorno virtual
if exist ".venv_spacy311\Scripts\activate.bat" (
    call .venv_spacy311\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo [ERROR] El entorno virtual no existe.
    echo         Ejecuta primero INSTALAR.bat
    pause
    exit /b 1
)

set STREAMLIT_SERVER_MAX_UPLOAD_SIZE=5000

echo Iniciando Fan-Out Query Analyzer...
echo.
streamlit run apps/fanout_analyzer/app.py --server.headless true

pause
