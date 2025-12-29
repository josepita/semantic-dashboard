@echo off
REM Script para iniciar Internal Linking Optimizer
REM Uso: start_app.bat

echo 🔗 Iniciando Internal Linking Optimizer...

REM Verificar si existe entorno virtual
if exist "..\..\\.venv\" (
    echo ✅ Entorno virtual encontrado

    REM Activar entorno virtual
    call "..\..\\.venv\Scripts\activate.bat"

    REM Instalar/actualizar dependencias
    echo 📦 Verificando dependencias...
    pip install -q -r requirements.txt

    REM Ejecutar app
    echo 🚀 Iniciando aplicación...
    streamlit run app.py
) else (
    echo ❌ No se encontró entorno virtual
    echo Creando entorno virtual...

    REM Crear entorno virtual en la raíz del proyecto
    python -m venv "..\..\\.venv"

    REM Activar
    call "..\..\\.venv\Scripts\activate.bat"

    REM Instalar dependencias
    echo 📦 Instalando dependencias...
    pip install -r requirements.txt

    REM Ejecutar app
    echo 🚀 Iniciando aplicación...
    streamlit run app.py
)

pause
