@echo off
chcp 65001 >nul

:: Cambiar al directorio del script
cd /d "%~dp0"

:: Verificar que existe el entorno virtual
if not exist ".venv\Scripts\activate.bat" (
    echo.
    echo [ERROR] El entorno virtual no existe.
    echo         Ejecuta primero INSTALAR.bat
    echo.
    pause
    exit /b 1
)

:: Activar entorno virtual y ejecutar Streamlit
echo Iniciando Embedding Dashboard...
echo.
echo La aplicacion se abrira en tu navegador.
echo Para cerrar, cierra esta ventana o pulsa Ctrl+C
echo.

call .venv\Scripts\activate.bat
streamlit run streamlit_app.py --server.headless true
