@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo  Iniciando Streamlit con limite de 5GB
echo ========================================
echo.

REM Verificar que existe el entorno virtual
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] El entorno virtual no existe.
    echo         Ejecuta primero INSTALAR.bat
    pause
    exit /b 1
)

REM Configurar variable de entorno
set STREAMLIT_SERVER_MAX_UPLOAD_SIZE=5000

echo Configuracion aplicada:
echo - Max Upload Size: 5000 MB (5 GB)
echo.

REM Activar entorno virtual e iniciar Streamlit
call .venv\Scripts\activate.bat
echo Iniciando Streamlit...
streamlit run streamlit_app.py --server.headless true

pause
