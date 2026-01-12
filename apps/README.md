# 🎯 Embedding Insights Suite - Apps Especializadas

Este directorio contiene 3 aplicaciones Streamlit independientes, cada una especializada en diferentes aspectos del análisis SEO con embeddings e IA.

**✨ Nuevo:** Sistema multi-proyecto con persistencia en DuckDB - gestiona múltiples clientes con datos independientes.

## 📁 Estructura de Apps

```
apps/
├── content-analyzer/      # 🎯 SEO Content Analyzer
│   ├── modules/
│   │   ├── semantic_tools.py
│   │   ├── keyword_builder.py
│   │   ├── semantic_relations.py
│   │   └── shared/           # ✨ NUEVO: Utilidades compartidas
│   │       ├── __init__.py
│   │       └── content_utils.py  # Funciones de análisis reutilizables
│   ├── app.py
│   ├── requirements.txt
│   └── start_app.bat
├── linking-optimizer/     # 🔗 Internal Linking Optimizer
│   ├── modules/
│   │   ├── __init__.py       # ✨ Package exports
│   │   ├── linking_pagerank.py      # PageRank + grafos semánticos
│   │   ├── linking_algorithms.py    # Algoritmos básico + avanzado
│   │   ├── linking_structural.py    # Enlazado estructural/taxonómico
│   │   ├── linking_hybrid.py        # Composite Link Score (CLS)
│   │   ├── linking_utils.py         # Reporting + Gemini AI
│   │   ├── csv_workflow.py
│   │   ├── knowledge_graph.py
│   │   └── ... (otros módulos)
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
  - **Modo Básico:** Semántica pura con priorización
  - **Modo Avanzado:** Semántica + silos con detección de huérfanas
  - **Modo Híbrido (CLS):** Composite Link Score (semántica 40% + PageRank 35% + entidades 25%)
  - **Modo Estructural:** Breadcrumb + hermanos + destacados por jerarquía
- Knowledge Graph con entidades
- Semantic Depth Score (SDS)
- Visualización t-SNE y grafos interactivos
- **Configuración de enlaces existentes:** Mejora PageRank con tus links actuales

**Tecnología:** spaCy, NetworkX, Pyvis, KMeans, PageRank

**✨ Arquitectura modular refactorizada:**
- `linking_pagerank.py` (259 líneas): PageRank temático + grafos semánticos
- `linking_algorithms.py` (405 líneas): Algoritmos básico y avanzado
- `linking_structural.py` (252 líneas): Enlazado estructural/taxonómico
- `linking_hybrid.py` (395 líneas): Composite Link Score (CLS)
- `linking_utils.py` (293 líneas): Reporting + interpretación con Gemini AI
- **Total:** 1604 líneas de lógica de negocio modularizada
- **UI refactorizada:** `app_sections/linking_lab.py` reducida de 2220 → 878 líneas (60% reducción)

### 📊 App 3: GSC Insights & Reporting
**Funcionalidades:**
- Análisis de rank tracking y posiciones SEO
- Parsing de múltiples formatos: CSV simple, SERP, Serprobot multi-keyword
- Asignación de familias con patrones wildcards (*pattern*)
- Informes HTML estáticos (competitivos) y dinámicos (Gemini AI)
- Agrupación de keywords por familias
- Insights con Gemini AI

**Tecnología:** Pandas, Matplotlib, Plotly, Gemini AI

**✨ Arquitectura modular refactorizada:**
- `positions_parsing.py` (499 líneas): Parsing multi-formato + normalización
- `positions_analysis.py` (164 líneas): Asignación de familias + análisis estadístico
- `positions_payload.py` (221 líneas): Construcción de payloads para reportes
- `positions_reports.py` (342 líneas): Generación HTML (estática + Gemini AI)
- **Total:** 1226 líneas de lógica de negocio modularizada
- **UI refactorizada:** `app_sections/positions_report.py` reducida de 1545 → 502 líneas (67% reducción)

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

## 🔧 Refactorización Modular (Enero 2026)

### Objetivos Alcanzados
1. ✅ **Reducción de complejidad:**
   - `linking_lab.py`: 2220 → 878 líneas (60% reducción)
   - `positions_report.py`: 1545 → 502 líneas (67% reducción)
   - **Total:** 3765 → 1380 líneas en archivos principales (63% reducción)
2. ✅ **Separación de responsabilidades:** Lógica de negocio separada de UI
3. ✅ **Reutilización de código:** Funciones compartidas entre apps
4. ✅ **Mantenibilidad:** Módulos < 600 líneas, cada uno con responsabilidad única
5. ✅ **11 módulos especializados creados** con 2830+ líneas de código bien estructurado

### Módulos Creados

#### `apps/content-analyzer/modules/shared/content_utils.py` (573 líneas)
**Funciones migradas desde linking_lab.py:**
- `detect_embedding_columns()` - Detecta columnas de embeddings
- `detect_url_columns()` - Detecta columnas de URLs
- `detect_page_type_columns()` - Detecta columnas de tipo de página
- `preprocess_embeddings()` - Valida y procesa embeddings
- `extract_url_silo()` - Extrae silo desde URL
- `extract_url_hierarchy()` - Extrae jerarquía de URL
- `calculate_weighted_entity_overlap()` - Similitud Jaccard ponderada
- `generate_contextual_anchor()` - Genera anchor text contextual
- `parse_entity_payload()` - Parsea payload de entidades JSON
- Y más... (13 funciones totales)

**Uso:** Compartidas entre content-analyzer y linking-optimizer

#### `apps/linking-optimizer/modules/linking_pagerank.py` (259 líneas)
**Algoritmos de grafos:**
- `build_similarity_edges()` - Construye aristas semánticas
- `calculate_topical_pagerank()` - PageRank temático con personalización
- `detect_orphan_pages()` - Detecta páginas huérfanas (money pages sin inlinks)
- `calculate_graph_metrics()` - Métricas de centralidad

**Características:**
- Boost 2x para enlaces existentes reales
- Personalización por tipo de página (money pages)
- Top-K vecinos semánticos configurable

#### `apps/linking-optimizer/modules/linking_algorithms.py` (405 líneas)
**Algoritmos semánticos:**
- `semantic_link_recommendations()` - Enlazado básico con priorización
- `advanced_semantic_linking()` - Avanzado con silos y detección de huérfanas

**Características:**
- Sistema de prioridades (primary/secondary/fallback)
- Boost para mismo silo configurable
- Detección automática de silo desde URL

#### `apps/linking-optimizer/modules/linking_structural.py` (252 líneas)
**Enlazado estructural:**
- `structural_taxonomy_linking()` - 3 estrategias:
  - **Ascendente (Breadcrumb):** Hijo → Padre
  - **Horizontal (Siblings):** Entre hermanos del mismo nivel
  - **Descendente (Featured):** Padre → Top N hijos

**Características:**
- Priorización semántica opcional para hermanos
- Extracción de jerarquía desde URL o columna custom
- Pesos configurables para PageRank posterior

#### `apps/linking-optimizer/modules/linking_hybrid.py` (395 líneas)
**Composite Link Score (CLS):**
- `hybrid_semantic_linking()` - Combina 3 señales:
  - Semántica (40%): Similitud coseno entre embeddings
  - Autoridad (35%): PageRank temático
  - Entidades (25%): Overlap de Knowledge Graph

**Características:**
- Decay factor para evitar concentración de inlinks
- Filtrado de enlaces existentes para evitar duplicados
- Detección de huérfanas y retorno de scores PageRank

#### `apps/linking-optimizer/modules/linking_utils.py` (293 líneas)
**Utilidades:**
- `guess_default_type()` - Autodetección de tipos de página
- `build_entity_payload_from_doc_relations()` - Convierte relaciones doc-entidad
- `build_linking_reports_payload()` - Agrega reportes de todos los modos
- `interpret_linking_reports_with_gemini()` - Análisis estratégico con Gemini AI

---

### Refactorización de `positions_report.py` (App 3: GSC Insights)

#### `apps/gsc-insights/modules/positions_parsing.py` (499 líneas)
**Parsing y normalización:**
- `normalize_domain()` - Normaliza dominios (elimina protocolo, www, subdominios)
- `parse_position_tracking_csv()` - Parser universal multi-formato:
  - Formato simple: Keyword, Position, URL
  - Formato SERP: Keyword, Position 1, Position 2, ..., Position 10
  - Formato Serprobot multi-keyword: Secciones con headers "Keyword: xxx"
- `parse_search_volume_file()` - Parser de archivos de volumen de búsqueda

**Características del parser:**
- Detección automática de encoding (utf-8, latin-1, iso-8859-1, cp1252)
- Maneja múltiples delimitadores (coma, punto y coma, tabulador, pipe)
- Procesa metadata de Serprobot (filas de encabezado)
- Extracción de dominios desde URLs completas

#### `apps/gsc-insights/modules/positions_analysis.py` (164 líneas)
**Análisis de datos:**
- `assign_keyword_families()` - Asignación con patrones:
  - Coincidencia exacta: "keyword"
  - Coincidencia parcial inicio: "patron*"
  - Coincidencia parcial fin: "*patron"
  - Coincidencia parcial central: "*patron*"
- `summarize_positions_overview()` - Resumen estadístico:
  - Keywords en top 10
  - Posición media de la marca
  - Competidores más frecuentes

#### `apps/gsc-insights/modules/positions_payload.py` (221 líneas)
**Construcción de payloads:**
- `build_family_payload()` - Agrega métricas por familia:
  - Keywords totales
  - Posición media
  - Keywords de marca en top 10
  - Volumen total y medio (si disponible)
- `build_competitive_family_payload()` - Payload competitivo:
  - Posiciones de todos los dominios por keyword
  - Comparativa marca vs competidores
  - Métricas agregadas por familia

#### `apps/gsc-insights/modules/positions_reports.py` (342 líneas)
**Generación de reportes HTML:**
- `generate_competitive_html_report()` - HTML estático con:
  - Tablas comparativas keyword-by-keyword
  - Colores según posición (verde=top1, rojo=no encontrado)
  - CSS inline moderno
- `generate_position_report_html()` - HTML dinámico con Gemini AI:
  - Resumen ejecutivo con insights
  - Recomendaciones priorizadas por volumen
  - Placeholders para gráficos
  - Análisis estratégico automático

### Archivos Actualizados

**Imports actualizados en:**
1. `app_sections/csv_workflow.py` - Ahora importa desde `content_utils`
2. `streamlit_app.py` - Imports segregados por módulo
3. `app_sections/linking_lab.py` - Refactorizada para usar módulos de linking-optimizer
4. `app_sections/positions_report.py` - Refactorizada para usar módulos de gsc-insights

### Beneficios de la Refactorización

**Para Desarrolladores:**
- 📦 Módulos autocontenidos con responsabilidad única
- 🔍 Más fácil de navegar y debuggear
- ✅ Testeable (sin dependencias de Streamlit en lógica de negocio)
- 📝 Documentación completa con ejemplos en docstrings

**Para el Proyecto:**
- 🔄 Código reutilizable entre apps
- 🧹 Reducción de duplicación (63% menos código en archivos principales)
- 🚀 Base escalable para nuevas funcionalidades
- 📊 **Métricas combinadas:**
  - **linking_lab.py**: 2220 → 878 líneas UI + 1604 líneas en 6 módulos
  - **positions_report.py**: 1545 → 502 líneas UI + 1226 líneas en 4 módulos
  - **Total**: 3765 líneas monolíticas → 1380 líneas UI + 2830 líneas modularizadas
  - **11 módulos especializados** con responsabilidad única

**Para Usuarios:**
- ⚡ Mismo rendimiento y funcionalidades
- 🎨 Interfaz idéntica (sin cambios visuales)
- ✨ Nuevas features más fáciles de añadir en el futuro

## 📄 Documentación Adicional

- Ver cada `app.py` para funcionalidades específicas
- Cada app tiene página de inicio con Quick Start
- Tooltips y ayuda contextual en la interfaz
- **Módulos documentados:** Cada función incluye docstring con ejemplos
