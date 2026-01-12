# Script para iniciar SEO Content Analyzer
# Uso: .\start_app.ps1

Write-Host "🎯 Iniciando SEO Content Analyzer..." -ForegroundColor Cyan

# Verificar si existe entorno virtual
$venvPath = "..\..\\.venv"
if (Test-Path $venvPath) {
    Write-Host "✅ Entorno virtual encontrado" -ForegroundColor Green

    # Activar entorno virtual
    & "$venvPath\Scripts\Activate.ps1"

    # Instalar/actualizar dependencias
    Write-Host "📦 Verificando dependencias..." -ForegroundColor Yellow
    pip install -q -r requirements.txt

    # Ejecutar app
    Write-Host "🚀 Iniciando aplicación..." -ForegroundColor Green
    streamlit run app.py
} else {
    Write-Host "❌ No se encontró entorno virtual" -ForegroundColor Red
    Write-Host "Creando entorno virtual..." -ForegroundColor Yellow

    # Crear entorno virtual
    python -m venv $venvPath

    # Activar
    & "$venvPath\Scripts\Activate.ps1"

    # Instalar dependencias
    Write-Host "📦 Instalando dependencias..." -ForegroundColor Yellow
    pip install -r requirements.txt

    # Ejecutar app
    Write-Host "🚀 Iniciando aplicación..." -ForegroundColor Green
    streamlit run app.py
}
