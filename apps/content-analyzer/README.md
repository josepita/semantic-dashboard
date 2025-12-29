# 🎯 SEO Content Analyzer

Herramienta especializada para análisis semántico de contenido SEO.

## 🚀 Funcionalidades

### 📝 **Texto vs Keywords**
Evalúa la relevancia semántica de cualquier texto frente a tus keywords objetivo.

**Casos de uso:**
- Optimizar meta descriptions
- Evaluar párrafos de contenido
- Validar títulos H1/H2
- Análisis de snippets

### ❓ **FAQs vs Keywords**
Analiza preguntas frecuentes y su relevancia para keywords específicas.

**Características:**
- ✅ Carga de Excel/CSV con preguntas y respuestas
- ✅ Selector de columnas intuitivo
- ✅ Top N FAQs por keyword
- ✅ Exportación a Excel

### 🔍 **Análisis de Competidores**
Extrae y compara contenido de URLs competidoras.

**Funcionalidades:**
- Extracción automática de contenido
- Comparación semántica con tus keywords
- Detección de gaps de contenido
- Análisis de meta descriptions

### 🧠 **Semantic Keyword Builder**
Agrupa keywords automáticamente por similitud semántica.

**Beneficios:**
- Detectar temas y clusters
- Planificar arquitectura de sitio
- Identificar keywords principales
- Optimizar keyword research

### 🔗 **Relaciones Semánticas**
Analiza relaciones entre URLs de tu sitio.

**Casos de uso:**
- Identificar pillar pages
- Detectar supporting content
- Visualizar topic clusters
- Mapear arquitectura de contenido

---

## 📦 Instalación

### Requisitos
- Python 3.9+
- pip

### Pasos

1. **Clonar o copiar esta carpeta**
```bash
cd apps/content-analyzer
```

2. **Crear entorno virtual (recomendado)**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Descargar modelo spaCy**
```bash
# Español
python -m spacy download es_core_news_sm

# Inglés
python -m spacy download en_core_web_sm
```

5. **Ejecutar aplicación**
```bash
streamlit run app.py
```

La aplicación se abrirá en `http://localhost:8501`

---

## 🔑 Configuración de API Keys

### OpenAI (opcional)
Para usar análisis avanzados:

**Opción 1: Variables de entorno**
```bash
export OPENAI_API_KEY="sk-..."
```

**Opción 2: Archivo .streamlit/secrets.toml**
```toml
OPENAI_API_KEY = "sk-..."
```

**Opción 3: Interfaz de usuario**
- Configurar directamente en la sidebar

---

## 📚 Uso Rápido

### Ejemplo 1: Analizar Texto

1. Ir a **📝 Texto vs Keywords**
2. Pegar tu texto
3. Introducir keywords (una por línea)
4. Click en "Calcular relevancia"
5. Ver resultados y exportar a Excel

### Ejemplo 2: Analizar FAQs

1. Ir a **❓ FAQs vs Keywords**
2. Seleccionar "📁 Cargar archivo Excel/CSV"
3. Subir Excel con columnas de preguntas/respuestas
4. Seleccionar columnas correctas
5. Introducir keywords
6. Obtener top FAQs por keyword

### Ejemplo 3: Agrupar Keywords

1. Ir a **🧠 Semantic Keyword Builder**
2. Subir CSV con keywords
3. Configurar parámetros de clustering
4. Ver agrupación automática
5. Exportar mapeo

---

## 🎯 Casos de Uso Reales

### Content Strategist
**Objetivo:** Planificar contenido para blog

1. Usar **Keyword Builder** para agrupar keywords
2. Identificar temas principales
3. Crear pillar pages por tema
4. Usar **Relaciones Semánticas** para conectar artículos

### SEO Copywriter
**Objetivo:** Optimizar meta descriptions

1. Usar **Texto vs Keywords** para cada meta
2. Ajustar hasta alcanzar relevancia >70%
3. Exportar mejores versiones
4. Implementar en sitio

### SEO Manager
**Objetivo:** Análisis competitivo

1. Recopilar URLs de top 10 de Google
2. Usar **Análisis de Competidores**
3. Detectar gaps de contenido
4. Crear plan de contenido basado en gaps

---

## 🛠️ Tecnología

- **Streamlit:** Framework UI
- **Sentence Transformers:** Embeddings semánticos
- **spaCy:** NLP y lemmatización
- **OpenAI API:** Análisis avanzados (opcional)
- **Pandas:** Procesamiento de datos
- **Plotly:** Visualizaciones

---

## 📊 Performance

**Tiempos aproximados:**
- Texto vs Keywords (1 texto): <1s
- FAQs vs Keywords (50 FAQs): 3-5s
- Keyword Builder (500 keywords): 10-15s
- Análisis Competidores (10 URLs): 30-60s

**Memoria:**
- Mínima: 200MB
- Típica: 500MB
- Con modelos cargados: 1GB

---

## 🐛 Solución de Problemas

### Error: "No module named 'spacy'"
```bash
pip install spacy
python -m spacy download es_core_news_sm
```

### Error: "No se pudo cargar el modelo"
Descargar modelo específico:
```bash
python -m spacy download es_core_news_sm  # Español
python -m spacy download en_core_web_sm   # Inglés
```

### Error: "Out of memory"
Reducir tamaño de batch o usar modelo más pequeño

---

## 📝 Changelog

### v1.0.0 (2025-01-XX)
- ✅ Release inicial
- ✅ Análisis de texto vs keywords
- ✅ Análisis de FAQs con carga de Excel
- ✅ Semantic Keyword Builder
- ✅ Análisis de competidores
- ✅ Relaciones semánticas

---

## 🤝 Soporte

¿Problemas o sugerencias?
- GitHub Issues
- Email: support@example.com

---

## 📄 Licencia

MIT License - Ver LICENSE file
