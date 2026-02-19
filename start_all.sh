#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=5000

activate_env() {
  if [[ -f "$ROOT_DIR/.venv_spacy311/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$ROOT_DIR/.venv_spacy311/bin/activate"
    return
  fi

  if [[ -f "$ROOT_DIR/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$ROOT_DIR/.venv/bin/activate"
    return
  fi

  echo "[ERROR] No se encontro entorno virtual (.venv_spacy311 o .venv)."
  echo "Crea uno con:"
  echo "  python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
}

run_app() {
  local choice="$1"
  case "$choice" in
    1) echo "Lanzando: Hub Principal"; streamlit run streamlit_app.py ;;
    2) echo "Lanzando: Content Analyzer"; streamlit run apps/content_analyzer/app.py ;;
    3) echo "Lanzando: GSC Insights"; streamlit run apps/gsc_insights/app.py ;;
    4) echo "Lanzando: Linking Optimizer"; streamlit run apps/linking_optimizer/app.py ;;
    5) echo "Lanzando: Fan-Out Analyzer"; streamlit run apps/fanout_analyzer/app.py ;;
    6) echo "Lanzando: Brief Generator (puerto 8503)"; streamlit run apps/brief_generator/app.py --server.port 8503 ;;
    *) echo "Opcion no valida. Lanzando Hub Principal..."; streamlit run streamlit_app.py ;;
  esac
}

echo "========================================"
echo " Embedding Insights Suite (Linux)"
echo "========================================"
echo
echo "Selecciona la aplicacion a ejecutar:"
echo "  [1] Hub Principal (Dashboard)"
echo "  [2] Content Analyzer"
echo "  [3] GSC Insights"
echo "  [4] Linking Optimizer"
echo "  [5] Fan-Out Analyzer"
echo "  [6] Brief Generator"
echo
read -rp "Introduce el numero (1-6): " APP_CHOICE

activate_env
echo
echo "Iniciando Streamlit..."
echo
run_app "${APP_CHOICE:-1}"
