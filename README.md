# Embedding Insights Dashboard

Suite SEO con apps de Streamlit para embeddings, clustering, enlazado interno, fan-out analysis e informes.

## Features
- Upload Excel files containing pre-generated page embeddings.
- Detect embedding columns automatically or select manually.
- Compute top-N similar URLs for any page.
- Build filtered similarity matrices with thresholding and export as CSV/Excel.
- Cluster pages with K-Means, auto-name clusters from URL tokens, and visualise via t-SNE.
- Explore topic clusters through an interactive graph that links clusters and their pages.
- Build knowledge graphs from textual columns (spaCy NER + relation hints) and explore entities/relations in an interactive network.
- Upload keyword lists, fetch OpenAI embeddings, and rank relevant pages; download reports.

## Requisitos
- Python 3.10+ (recomendado 3.11)
- `pip`

## Inicialización del proyecto
1. Clona el repositorio y entra en la carpeta:
```bash
git clone <repo-url>
cd semantic-dashboard
```

2. Crea y activa el entorno virtual:

Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

3. Instala dependencias:
```bash
pip install -r requirements.txt
```

4. Configura variables de entorno:
```bash
cp .env.example .env
```
Después edita `.env` y completa tus keys (`OPENAI_API_KEY`, `GEMINI_API_KEY`, etc.).

5. (Opcional) Verifica el entorno:
```bash
python verify_setup.py
```

## Ejecutar la aplicación principal
```bash
streamlit run streamlit_app.py
```

## Ejecutar apps standalone
```bash
streamlit run apps/content_analyzer/app.py
streamlit run apps/gsc_insights/app.py
streamlit run apps/linking_optimizer/app.py
streamlit run apps/fanout_analyzer/app.py
streamlit run apps/brief_generator/app.py
```

## Ejecutar API (FastAPI)
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8001 --reload
```

## spaCy (opcional, para módulos de grafo/NER)
```bash
python -m spacy download es_core_news_sm
python -m spacy download en_core_web_sm
```

## Deployment Notes
- Configure `STREAMLIT_SERVER_HEADLESS=true` for server deployments.
- Adjust default similarity threshold and batch sizes in `streamlit_app.py` if processing very large datasets.
