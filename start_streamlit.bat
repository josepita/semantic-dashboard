@echo off
echo ========================================
echo  Iniciando Streamlit con limite de 5GB
echo ========================================
echo.

REM Configurar variable de entorno
set STREAMLIT_SERVER_MAX_UPLOAD_SIZE=5000

echo Configuracion aplicada:
echo - Max Upload Size: 5000 MB (5 GB)
echo - Config File: .streamlit\config.toml
echo.

REM Iniciar Streamlit
echo Iniciando Streamlit...
streamlit run streamlit_app.py

pause
