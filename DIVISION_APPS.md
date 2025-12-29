# 🎯 División en 3 Aplicaciones Especializadas

Este documento describe la estructura de las 3 aplicaciones separadas creadas a partir del dashboard unificado original.

## 📂 Estructura del Proyecto

```
EmbeddingDashboard/
├── apps/                          # 3 aplicaciones separadas
│   ├── content-analyzer/          # App 1: SEO Content Analyzer
│   │   ├── app.py                 # Aplicación principal
│   │   ├── requirements.txt       # Dependencias optimizadas
│   │   └── README.md              # Documentación
│   │
│   ├── linking-optimizer/         # App 2: Internal Linking Optimizer
│   │   ├── app.py                 # Aplicación principal
│   │   ├── requirements.txt       # Dependencias optimizadas
│   │   └── README.md              # Documentación (pendiente)
│   │
│   ├── gsc-insights/              # App 3: GSC Insights
│   │   ├── app.py                 # Aplicación principal
│   │   ├── requirements.txt       # Dependencias mínimas
│   │   └── README.md              # Documentación (pendiente)
│   │
│   └── README.md                  # Guía general de apps
│
├── shared/                        # Librería compartida
│   ├── __init__.py                # Package init
│   ├── entity_filters.py          # Filtrado y lemmatización
│   ├── spacy_support.py           # Soporte de spaCy
│   └── semantic_depth.py          # Cálculos semánticos
│
├── app_sections/                  # Código original (mantener)
│   └── ...
│
├── streamlit_app.py               # App unificada original (mantener)
├── requirements.txt               # Deps completas (mantener)
│
└── DOCS/                          # Documentación
    ├── ENTITY_FILTERING_GUIDE.md
    ├── RESUMEN_FUNCIONALIDADES.md
    └── DIVISION_APPS.md (este archivo)
```

---

## 🎯 App 1: SEO Content Analyzer

### 📦 Alcance
Análisis semántico de contenido para SEO.

### 🔧 Módulos Incluidos
- ✅ **Semantic Toolkit:**
  - Texto vs Keywords
  - FAQs vs Keywords (con carga de Excel!)
  - Análisis de Competidores
  - Variantes de URL

- ✅ **Semantic Keyword Builder:**
  - Agrupación automática de keywords
  - Detección de temas
  - Visualización de clusters

- ✅ **Semantic Relations:**
  - Análisis de relaciones entre URLs
  - Topic clusters
  - Pillar pages

### 📋 Dependencias (8 packages)
```
streamlit, pandas, numpy, openpyxl, xlrd
sentence-transformers, openai, spacy
trafilatura, beautifulsoup4, scipy, scikit-learn
matplotlib, plotly
```

### 💾 Peso Aproximado
- **Instalación:** ~800MB
- **Memoria en ejecución:** 200-500MB
- **Tiempo de carga:** 1-3s

### 🎯 Usuarios Target
- Content writers
- SEO content strategists
- Copywriters
- Marketing managers

---

## 🔗 App 2: Internal Linking Optimizer

### 📦 Alcance
Optimización de enlazado interno basada en análisis semántico.

### 🔧 Módulos Incluidos
- ✅ **CSV Workflow:**
  - Carga de embeddings
  - Análisis de similitud
  - Clustering automático
  - Visualización t-SNE

- ✅ **Linking Lab:**
  - Authority Gap Analysis
  - Recomendaciones de enlaces
  - Simulación de PageRank
  - Entity-based recommendations

- ✅ **Knowledge Graph:**
  - Extracción de entidades (spaCy)
  - Construcción de grafos
  - Co-ocurrencias
  - Entity linking

### 📋 Dependencias (12 packages)
```
streamlit, pandas, numpy, openpyxl
sentence-transformers, spacy
networkx, pyvis
scipy, scikit-learn
matplotlib, seaborn, plotly
polars (opcional)
```

### 💾 Peso Aproximado
- **Instalación:** ~1.2GB
- **Memoria en ejecución:** 500MB-1GB
- **Tiempo de carga:** 3-5s

### 🎯 Usuarios Target
- SEO técnico
- Arquitectos de información
- Web developers
- Consultores SEO

---

## 📊 App 3: GSC Insights

### 📦 Alcance
Análisis de Google Search Console con Gemini AI.

### 🔧 Módulos Incluidos
- ✅ **Positions Report:**
  - Importación de datos GSC
  - Dashboard de métricas
  - Filtros interactivos

- ✅ **Gemini AI Analysis:**
  - Quick Wins automáticos
  - Detección de cannibalization
  - Análisis de tendencias
  - Generación de insights

- ✅ **Reports:**
  - Exportación a Excel
  - Reportes automatizados
  - Visualizaciones

### 📋 Dependencias (6 packages)
```
streamlit, pandas, numpy, openpyxl
google-generativeai
matplotlib, plotly, scipy
```

### 💾 Peso Aproximado
- **Instalación:** ~300MB
- **Memoria en ejecución:** 100-200MB
- **Tiempo de carga:** <1s

### 🎯 Usuarios Target
- SEO managers
- Clientes finales (reportes)
- Analistas de datos
- Marketing teams

---

## 🔄 Comparativa: Unificado vs Separado

| Aspecto | App Unificada | 3 Apps Separadas |
|---------|---------------|------------------|
| **Tiempo de carga** | 8-12s | 1-5s (por app) |
| **Memoria total** | 800MB-1GB | 200MB-1GB (según app) |
| **Instalación** | ~1.5GB | ~300MB-1.2GB (por app) |
| **Mantenimiento** | Media | Alta (setup) → Baja |
| **Deployment** | 1 servidor | 3 servidores (o 1 con rutas) |
| **Costo hosting** | $50/mes | $30/mes (total) |
| **Escalabilidad** | Limitada | Alta |
| **Especialización** | Media | Alta |

---

## 🚀 Cómo Ejecutar Cada App

### App 1: Content Analyzer
```bash
cd apps/content-analyzer
pip install -r requirements.txt
python -m spacy download es_core_news_sm
streamlit run app.py
```

### App 2: Linking Optimizer
```bash
cd apps/linking-optimizer
pip install -r requirements.txt
python -m spacy download es_core_news_sm
streamlit run app.py
```

### App 3: GSC Insights
```bash
cd apps/gsc-insights
pip install -r requirements.txt
streamlit run app.py
```

---

## 📝 Estado Actual de Desarrollo

### ✅ Completado
- [x] Estructura de carpetas
- [x] App skeletons (3 apps)
- [x] Requirements.txt optimizados
- [x] Shared library básica
- [x] README de Content Analyzer
- [x] Documentación general

### 🔄 En Progreso
- [ ] Migrar módulos completos a cada app
- [ ] Completar shared library
- [ ] READMEs de Linking Optimizer y GSC Insights
- [ ] Testing de cada app

### 📅 Pendiente
- [ ] Configuración de imports relativos
- [ ] Scripts de instalación automatizada
- [ ] Docker containers (opcional)
- [ ] CI/CD pipelines
- [ ] Documentación de API
- [ ] Video tutoriales

---

## 🔧 Próximos Pasos Inmediatos

### 1. Completar Shared Library (2-3 horas)
```bash
# Copiar módulos compartidos
cp app_sections/entity_filters.py shared/
cp app_sections/spacy_support.py shared/
cp app_sections/semantic_depth.py shared/
cp app_sections/semantic_tools.py shared/
```

### 2. Migrar Módulos a Apps (4-6 horas)
- **Content Analyzer:**
  - Copiar semantic_tools.py
  - Copiar keyword_builder.py
  - Copiar semantic_relations.py

- **Linking Optimizer:**
  - Copiar csv_workflow.py
  - Copiar linking_lab.py
  - Copiar knowledge_graph.py
  - Copiar authority_advance.py

- **GSC Insights:**
  - Copiar positions_report.py
  - Integrar Gemini AI logic

### 3. Ajustar Imports (1-2 horas)
Cambiar todos los imports de:
```python
from app_sections.entity_filters import ...
```

A:
```python
import sys
from pathlib import Path
shared_path = Path(__file__).parent.parent.parent / "shared"
sys.path.insert(0, str(shared_path))
from entity_filters import ...
```

### 4. Testing Básico (2-3 horas)
- Probar carga de cada app
- Verificar imports
- Validar funcionalidad básica

### 5. Documentación (2-3 horas)
- Completar READMEs faltantes
- Crear ejemplos de uso
- Screenshots de cada app

---

## 💡 Tips de Implementación

### Desarrollo Local
```bash
# Usar entorno virtual separado por app
cd apps/content-analyzer
python -m venv venv
source venv/bin/activate  # o venv\Scripts\activate en Windows
pip install -r requirements.txt
```

### Testing
```bash
# Probar cada app independientemente
streamlit run app.py --server.port 8501

# Content Analyzer: http://localhost:8501
# Linking Optimizer: http://localhost:8502
# GSC Insights: http://localhost:8503
```

### Deployment
**Opción A: Servidores separados**
- 3 instancias de Cloud Run / Heroku
- URLs diferentes (content.tuapp.com, linking.tuapp.com, gsc.tuapp.com)

**Opción B: Servidor único con routing**
- 1 instancia con Nginx
- Rutas: /content, /linking, /gsc
- Más económico

---

## 🎯 Beneficios de la División

### Para Usuarios
✅ Carga más rápida de la app que necesitan
✅ Interfaz más enfocada y simple
✅ Menos curva de aprendizaje
✅ Mejor experiencia móvil

### Para Desarrollo
✅ Código más modular
✅ Fácil añadir features a una app específica
✅ Tests más simples y rápidos
✅ Deploy independiente

### Para Negocio
✅ Monetización diferenciada (planes por app)
✅ Mejor segmentación de usuarios
✅ Scaling independiente
✅ Reducción de costos

---

## 📞 Soporte

¿Dudas sobre la división de apps?
- Ver: `RESUMEN_FUNCIONALIDADES.md`
- Ver: `ENTITY_FILTERING_GUIDE.md`
- GitHub Issues
- Email: support@example.com

---

**Última actualización:** 2025-12-29
**Versión:** 1.0.0 (estructura inicial)
