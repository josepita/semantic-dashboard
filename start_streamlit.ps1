Write-Host "========================================"
Write-Host " Iniciando Streamlit con limite de 5GB"
Write-Host "========================================"
Write-Host ""

# Configurar variable de entorno
$env:STREAMLIT_SERVER_MAX_UPLOAD_SIZE = "5000"

Write-Host "Configuracion aplicada:"
Write-Host "- Max Upload Size: 5000 MB (5 GB)"
Write-Host "- Config File: .streamlit\config.toml"
Write-Host ""

# Iniciar Streamlit
Write-Host "Iniciando Streamlit..."
streamlit run streamlit_app.py
