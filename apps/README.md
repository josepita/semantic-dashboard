# 🎯 Embedding Insights - Apps Separadas

Este directorio contiene las 3 aplicaciones especializadas extraídas del dashboard unificado original.

## 📁 Estructura

```
apps/
├── content-analyzer/      # App 1: SEO Content Analyzer
├── linking-optimizer/     # App 2: Internal Linking Optimizer
└── gsc-insights/         # App 3: GSC Insights
```

## 🔗 Librería Compartida

El código común entre las apps está en `/shared/`:
- Entity filters (lemmatización, filtrado de ruido)
- Semantic tools (embeddings, similitud)
- Helpers y utilidades

## 🚀 Uso Individual

Cada app puede ejecutarse independientemente:

```bash
# App 1: Content Analyzer
cd apps/content-analyzer
streamlit run app.py

# App 2: Linking Optimizer
cd apps/linking-optimizer
streamlit run app.py

# App 3: GSC Insights
cd apps/gsc-insights
streamlit run app.py
```

## 📦 Instalación

Cada app tiene su propio `requirements.txt` optimizado:

```bash
cd apps/content-analyzer
pip install -r requirements.txt
```

## 🔄 Migración desde App Unificada

Ver `MIGRATION_GUIDE.md` para detalles sobre cómo migrar datos y configuración desde la aplicación original.
