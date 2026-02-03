@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo  Embedding Insights Suite
echo ========================================
echo.

REM Verificar que existe el entorno virtual
if not exist ".venv\Scripts\activate.bat" (
    if not exist ".venv_spacy311\Scripts\activate.bat" (
        echo [ERROR] El entorno virtual no existe.
        echo         Ejecuta primero INSTALAR.bat
        pause
        exit /b 1
    )
)

REM Configurar variable de entorno
set STREAMLIT_SERVER_MAX_UPLOAD_SIZE=5000

echo Selecciona la aplicacion a ejecutar:
echo.
echo   [1] Hub Principal (Dashboard)
echo   [2] Content Analyzer
echo   [3] GSC Insights
echo   [4] Linking Optimizer
echo   [5] Fan-Out Analyzer
echo.
set /p APP_CHOICE="Introduce el numero (1-5): "

REM Activar entorno virtual
if exist ".venv_spacy311\Scripts\activate.bat" (
    call .venv_spacy311\Scripts\activate.bat
) else (
    call .venv\Scripts\activate.bat
)

echo.
echo Iniciando Streamlit...
echo.

if "%APP_CHOICE%"=="1" (
    echo Lanzando: Hub Principal
    streamlit run streamlit_app.py --server.headless true
) else if "%APP_CHOICE%"=="2" (
    echo Lanzando: Content Analyzer
    streamlit run apps/content_analyzer/app.py --server.headless true
) else if "%APP_CHOICE%"=="3" (
    echo Lanzando: GSC Insights
    streamlit run apps/gsc_insights/app.py --server.headless true
) else if "%APP_CHOICE%"=="4" (
    echo Lanzando: Linking Optimizer
    streamlit run apps/linking_optimizer/app.py --server.headless true
) else if "%APP_CHOICE%"=="5" (
    echo Lanzando: Fan-Out Analyzer
    streamlit run apps/fanout_analyzer/app.py --server.headless true
) else (
    echo Opcion no valida. Lanzando Hub Principal...
    streamlit run streamlit_app.py --server.headless true
)

pause
