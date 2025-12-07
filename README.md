# Embedding Insights Dashboard

Interactive Streamlit dashboard to explore page embeddings, compute similarity matrices, perform clustering, and analyse keyword relevance using OpenAI embeddings.

## Features
- Upload Excel files containing pre-generated page embeddings.
- Detect embedding columns automatically or select manually.
- Compute top-N similar URLs for any page.
- Build filtered similarity matrices with thresholding and export as CSV/Excel.
- Cluster pages with K-Means, auto-name clusters from URL tokens, and visualise via t-SNE.
- Explore topic clusters through an interactive graph that links clusters and their pages.
- Build knowledge graphs from textual columns (spaCy NER + relation hints) and explore entities/relations in an interactive network.
- Upload keyword lists, fetch OpenAI embeddings, and rank relevant pages; download reports.

## Running Locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
If you plan to use the knowledge-graph module, install the spaCy model you need (examples):
```bash
python -m spacy download es_core_news_sm   # Spanish
python -m spacy download en_core_web_sm    # English
```
## Instalar localmente 
cd C:\Users\jdiaz\Desktop\EmbeddingDashboard
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py



Set the environment variable `OPENAI_API_KEY` or provide the key via the UI for keyword relevance.

## Deployment Notes
- Configure `STREAMLIT_SERVER_HEADLESS=true` for server deployments.
- Adjust default similarity threshold and batch sizes in `streamlit_app.py` if processing very large datasets.
