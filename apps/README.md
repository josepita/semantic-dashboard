# 🎯 Embedding Insights Suite - Apps Especializadas

Este directorio contiene 3 aplicaciones Streamlit independientes, cada una especializada en diferentes aspectos del análisis SEO con embeddings e IA.

**✨ Nuevo:** Sistema multi-proyecto con persistencia en DuckDB - gestiona múltiples clientes con datos independientes.

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

## 📁 Sistema de Proyectos (Nuevo)

Todas las apps ahora incluyen un **selector de proyectos** en el sidebar que permite:

### Características
- **Multi-proyecto:** Gestiona múltiples clientes con datos independientes
- **Persistencia automática:** Datos guardados en DuckDB por proyecto
- **Sin re-uploads:** Los datos se cargan automáticamente al abrir el proyecto
- **Estadísticas:** Visualiza URLs, embeddings, registros por proyecto
- **Switch rápido:** Cambia entre proyectos sin reiniciar la app

### Estructura de Proyecto
```
workspace/
├── .workspace_config.json      # Config global + último proyecto
└── projects/
    └── mi-cliente/
        ├── config.json          # Configuración del proyecto
        ├── database.duckdb      # Base de datos DuckDB
        ├── embeddings/          # Caché de embeddings
        │   ├── [model].faiss    # Índice FAISS (opcional)
        │   └── metadata.json
        └── oauth/               # Credenciales OAuth (gitignored)
```

### Uso Básico

**1. Crear proyecto:**
- Abre cualquier app
- Sidebar → "➕ Crear Nuevo Proyecto"
- Nombre: "Mi Cliente SEO"
- Dominio: "ejemplo.com"

**2. Trabajar con datos:**
- App 3 (GSC Insights): Sube CSV de posiciones → Se guarda en DuckDB
- App 2 (Linking Optimizer): Genera embeddings → Se guardan en caché
- App 1 (Content Analyzer): Analiza contenido → Se persiste en DB

**3. Recuperar datos:**
- Cierra la app
- Vuelve a abrir
- El proyecto se carga automáticamente
- Click "📊 Cargar datos guardados del proyecto"
- Todos los datos están disponibles sin re-subir archivos

### Componentes de Persistencia

**DataOrchestrator** ([shared/data_orchestrator.py](../shared/data_orchestrator.py))
- Gestión unificada de todos los datos en DuckDB
- Métodos para URLs, embeddings, GSC, familias, relaciones, entidades, clusters, FAQs
- Ver [USAGE.md](../shared/USAGE.md) para documentación completa

**EmbeddingCache** ([shared/embedding_cache.py](../shared/embedding_cache.py))
- Caché híbrido DuckDB + FAISS para embeddings
- Búsqueda de similitud 100-1000x más rápida con FAISS
- Sincronización automática entre DuckDB y FAISS
- Ver [USAGE.md](../shared/USAGE.md) para ejemplos de uso

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
