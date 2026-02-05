@echo off
title Brief Generator - SEO Intelligence
cd /d "%~dp0"
cd ../..
call .venv\Scripts\activate 2>nul || echo Virtual env not found, using system Python
streamlit run apps/brief_generator/app.py --server.port 8503
pause
