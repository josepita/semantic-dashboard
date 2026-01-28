@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ═══════════════════════════════════════════════════════════════
echo           INSTALADOR - Embedding Dashboard Suite
echo ═══════════════════════════════════════════════════════════════
echo.

:: Verificar que estamos en el directorio correcto
if not exist "streamlit_app.py" (
    echo [ERROR] Este script debe ejecutarse desde la carpeta del proyecto.
    echo         Asegurate de estar en la carpeta EmbeddingDashboard
    pause
    exit /b 1
)

:: ═══════════════════════════════════════════════════════════════
:: PASO 1: Verificar Python
:: ═══════════════════════════════════════════════════════════════
echo [1/5] Verificando instalacion de Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] Python no esta instalado o no esta en el PATH.
    echo.
    echo Para instalar Python:
    echo   1. Ve a https://www.python.org/downloads/
    echo   2. Descarga Python 3.10 o superior
    echo   3. Durante la instalacion, marca "Add Python to PATH"
    echo   4. Reinicia el ordenador y ejecuta este instalador de nuevo
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo        Python encontrado: version %PYTHON_VERSION%

:: ═══════════════════════════════════════════════════════════════
:: PASO 2: Crear entorno virtual
:: ═══════════════════════════════════════════════════════════════
echo.
echo [2/5] Creando entorno virtual...

if exist ".venv" (
    echo        Entorno virtual existente encontrado, reinstalando...
    rmdir /s /q .venv 2>nul
)

python -m venv .venv
if errorlevel 1 (
    echo [ERROR] No se pudo crear el entorno virtual.
    pause
    exit /b 1
)
echo        Entorno virtual creado correctamente.

:: ═══════════════════════════════════════════════════════════════
:: PASO 3: Instalar dependencias
:: ═══════════════════════════════════════════════════════════════
echo.
echo [3/5] Instalando dependencias (esto puede tardar unos minutos)...
echo.

call .venv\Scripts\activate.bat

:: Actualizar pip
python -m pip install --upgrade pip --quiet

:: Instalar dependencias principales
echo        Instalando paquetes de requirements.txt...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Error instalando dependencias.
    pause
    exit /b 1
)

:: ═══════════════════════════════════════════════════════════════
:: PASO 4: Instalar modelo de spaCy
:: ═══════════════════════════════════════════════════════════════
echo.
echo [4/5] Instalando modelo de lenguaje spaCy (espanol)...
python -m spacy download es_core_news_md --quiet
if errorlevel 1 (
    echo [AVISO] No se pudo instalar el modelo de spaCy automaticamente.
    echo         Algunas funciones de NLP podrian no estar disponibles.
)

:: Instalar google-generativeai si no está en requirements
pip install google-generativeai --quiet 2>nul

echo        Dependencias instaladas correctamente.

:: ═══════════════════════════════════════════════════════════════
:: PASO 5: Crear archivo de configuracion
:: ═══════════════════════════════════════════════════════════════
echo.
echo [5/5] Configurando la aplicacion...

:: Crear .env si no existe
if not exist ".env" (
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        echo        Archivo .env creado desde plantilla.
        echo.
        echo [IMPORTANTE] Edita el archivo .env con tus claves API:
        echo             - GOOGLE_API_KEY para Gemini AI
        echo             - OPENAI_API_KEY para OpenAI (opcional)
    )
)

:: ═══════════════════════════════════════════════════════════════
:: FINALIZADO
:: ═══════════════════════════════════════════════════════════════
echo.
echo ═══════════════════════════════════════════════════════════════
echo                    INSTALACION COMPLETADA
echo ═══════════════════════════════════════════════════════════════
echo.
echo Para iniciar la aplicacion:
echo   - Doble clic en "EJECUTAR.bat"
echo   - O doble clic en cualquier Launcher-*.vbs
echo.
echo La aplicacion se abrira automaticamente en tu navegador.
echo.
pause
