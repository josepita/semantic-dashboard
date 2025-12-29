# 🎯 Embedding Insights Suite - Apps Especializadas

Este directorio contiene 3 aplicaciones Streamlit independientes, cada una especializada en diferentes aspectos del análisis SEO con embeddings e IA.

## 📁 Estructura de Apps

```
apps/
├── content-analyzer/      # 🎯 SEO Content Analyzer
│   ├── modules/          # 3 módulos (semantic_tools, keyword_builder, semantic_relations)
│   ├── app.py
│   ├── requirements.txt
│   └── start_app.bat
├── linking-optimizer/     # 🔗 Internal Linking Optimizer
│   ├── modules/          # 9 módulos (csv_workflow, linking_lab, knowledge_graph, etc.)
│   ├── app.py
│   ├── requirements.txt
│   └── start_app.bat
└── gsc-insights/         # 📊 GSC Insights & Reporting
    ├── modules/          # 5 módulos (positions_report, google_kg, keyword_builder, etc.)
    ├── app.py
    ├── requirements.txt
    └── start_app.bat
```

## 🚀 Inicio Rápido

Cada app incluye un script de inicio que maneja automáticamente:
- Creación del entorno virtual
- Instalación de dependencias
- Ejecución de la app

```bash
# Windows
cd apps/content-analyzer
./start_app.bat

# Linux/Mac
cd apps/content-analyzer
streamlit run app.py
```

## 📊 Aplicaciones Disponibles

### 🎯 App 1: SEO Content Analyzer
**Funcionalidades:**
- Análisis de relevancia semántica (Texto, FAQs, Competidores)
- Semantic Keyword Builder con Gemini AI
- Análisis de relaciones semánticas entre URLs
- Carga de FAQs desde Excel

**Tecnología:** Sentence Transformers, OpenAI, spaCy, Trafilatura

### 🔗 App 2: Internal Linking Optimizer
**Funcionalidades:**
- Análisis de embeddings y clustering automático
- Laboratorio de enlazado (4 modos: Básico, Avanzado, Híbrido, Estructural)
- Knowledge Graph con entidades
- Semantic Depth Score (SDS)
- Visualización t-SNE y grafos interactivos

**Tecnología:** spaCy, NetworkX, Pyvis, KMeans, PageRank

### 📊 App 3: GSC Insights & Reporting
**Funcionalidades:**
- Análisis de rank tracking y posiciones SEO
- Informes HTML con gráficos interactivos
- Agrupación de keywords por familias
- Insights con Gemini AI

**Tecnología:** Pandas, Matplotlib, Plotly, Gemini AI

## 📦 Instalación

### Opción 1: Script Automático (Recomendado)
```bash
cd apps/[nombre-app]
./start_app.bat  # Maneja todo automáticamente
```

### Opción 2: Manual
```bash
# Crear entorno virtual en la raíz del proyecto
cd ../../
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Instalar dependencias por app
cd apps/content-analyzer
pip install -r requirements.txt
```

## 🔑 Configuración de APIs

### OpenAI (Apps 1 y 2)
```bash
export OPENAI_API_KEY="tu-api-key"
```

### Gemini AI (Apps 1 y 3)
```bash
export GEMINI_API_KEY="tu-api-key"
# Obtener en: https://aistudio.google.com/app/apikey
```

### spaCy Models
```bash
python -m spacy download es_core_news_sm  # Español
python -m spacy download en_core_web_sm   # Inglés
```

## 📝 Estado de Migración

✅ **App 1: Content Analyzer** - 100% funcional (Commits: d4e6957, 1c48a9c, 464398f)
✅ **App 2: Linking Optimizer** - 100% funcional (Commit: 3df925f)
✅ **App 3: GSC Insights** - 100% funcional (Commit: 68cc0b2)

**Total:** 18 módulos migrados, 3 apps independientes, 8 commits

## 🛠️ Troubleshooting

**Problema:** Module not found
**Solución:** Verifica estar en el directorio correcto y venv activado

**Problema:** spaCy model not found
**Solución:** `python -m spacy download es_core_news_sm`

**Problema:** API key not found
**Solución:** Configura variables de entorno o ingresa en la interfaz

## 📄 Documentación Adicional

- Ver cada `app.py` para funcionalidades específicas
- Cada app tiene página de inicio con Quick Start
- Tooltips y ayuda contextual en la interfaz
