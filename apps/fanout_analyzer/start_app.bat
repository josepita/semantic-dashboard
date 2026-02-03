@echo off
REM Script para iniciar Fan-Out Query Analyzer
REM Uso: start_app.bat

echo 🔍 Iniciando Fan-Out Query Analyzer...

REM Verificar si existe entorno virtual
if exist "..\..\\.venv_spacy311\" (
    echo ✅ Entorno virtual encontrado
    call "..\..\\.venv_spacy311\Scripts\activate.bat"
) else if exist "..\..\\.venv\" (
    echo ✅ Entorno virtual encontrado
    call "..\..\\.venv\Scripts\activate.bat"
) else (
    echo ❌ No se encontró entorno virtual
    echo Creando entorno virtual...
    python -m venv "..\..\\.venv"
    call "..\..\\.venv\Scripts\activate.bat"
    echo 📦 Instalando dependencias...
    pip install -r "..\..\requirements.txt"
)

REM Ejecutar app
echo 🚀 Iniciando aplicación...
streamlit run app.py

pause
