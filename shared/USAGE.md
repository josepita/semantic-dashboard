# 📚 Guía de Uso - DataOrchestrator y EmbeddingCache

Esta guía explica cómo usar los componentes de persistencia en las aplicaciones de Embedding Insights Suite.

---

## 📊 DataOrchestrator

Gestor centralizado para todas las operaciones de datos en DuckDB.

### Inicialización

```python
from data_orchestrator import DataOrchestrator

# Opción 1: Desde path directo
orchestrator = DataOrchestrator("workspace/projects/mi-cliente/database.duckdb")

# Opción 2: Desde configuración de proyecto
from data_orchestrator import get_data_orchestrator

project_config = st.session_state.get("project_config")
orchestrator = get_data_orchestrator(project_config)
```

### Gestión de URLs

```python
# Guardar URLs con metadatos
urls_data = [
    {
        "url": "https://ejemplo.com/page1",
        "title": "Título de la página",
        "content": "Contenido completo...",
        "meta_description": "Descripción meta",
        "word_count": 500
    },
    {
        "url": "https://ejemplo.com/page2",
        "title": "Otra página",
        "content": "Más contenido...",
        "word_count": 300
    }
]

count = orchestrator.save_urls(urls_data)
print(f"Guardadas {count} URLs")

# Obtener URLs
df = orchestrator.get_urls(limit=100)
print(df.head())

# Filtrar por estado de embedding
pending_df = orchestrator.get_urls(embedding_status="pending")
```

### Gestión de Embeddings

```python
import numpy as np

# Guardar un embedding
url = "https://ejemplo.com/page1"
embedding = np.random.rand(384)  # Vector de 384 dimensiones
model = "paraphrase-multilingual-MiniLM-L12-v2"

orchestrator.save_embeddings(url, embedding, model)

# Obtener embeddings como DataFrame
df = orchestrator.get_embeddings(model)
print(df[['url', 'dimension', 'created_at']])

# Obtener como matriz numpy
urls, embeddings_matrix = orchestrator.get_embedding_vectors(model)
print(f"URLs: {len(urls)}")
print(f"Shape: {embeddings_matrix.shape}")

# Filtrar URLs específicas
specific_urls = ["https://ejemplo.com/page1", "https://ejemplo.com/page2"]
urls, embeddings = orchestrator.get_embedding_vectors(model, urls=specific_urls)
```

### Datos de Google Search Console

```python
import pandas as pd
from datetime import date

# Preparar datos
gsc_data = pd.DataFrame({
    "keyword": ["ejemplo keyword", "otra keyword"],
    "url": ["https://ejemplo.com/page1", "https://ejemplo.com/page2"],
    "position": [3.5, 8.2],
    "impressions": [1000, 500],
    "clicks": [100, 25],
    "ctr": [0.10, 0.05],
    "date": [date.today(), date.today()]
})

# Guardar (reemplaza datos existentes por defecto)
count = orchestrator.save_gsc_data(gsc_data, replace=True)

# Obtener datos con filtros
df = orchestrator.get_gsc_data(
    start_date="2025-01-01",
    end_date="2025-12-31",
    keywords=["ejemplo keyword"]
)
```

### Familias de Keywords

```python
# Guardar familias
families = {
    "Productos": ["comprar producto", "precio producto", "producto barato"],
    "Servicios": ["contratar servicio", "costo servicio", "servicio premium"],
    "Info": ["qué es", "cómo funciona", "tutorial"]
}

count = orchestrator.save_keyword_families(families)

# Obtener familias
families_dict = orchestrator.get_keyword_families()
print(families_dict)
```

### Relaciones Semánticas

```python
# Guardar relaciones para linking interno
relations = [
    {
        "source_url": "https://ejemplo.com/page1",
        "target_url": "https://ejemplo.com/page2",
        "similarity_score": 0.85,
        "relation_type": "semantic",
        "anchor_suggestion": "más información sobre..."
    },
    {
        "source_url": "https://ejemplo.com/page1",
        "target_url": "https://ejemplo.com/page3",
        "similarity_score": 0.72,
        "relation_type": "semantic",
        "anchor_suggestion": "ver también"
    }
]

count = orchestrator.save_semantic_relations(relations, replace=True)

# Obtener relaciones de una URL
df = orchestrator.get_semantic_relations(
    source_url="https://ejemplo.com/page1",
    min_score=0.7
)
print(df[['target_url', 'similarity_score', 'anchor_suggestion']])
```

### Entidades (Knowledge Graph)

```python
# Guardar entidades extraídas
entities = [
    {
        "url": "https://ejemplo.com/page1",
        "entity_text": "Madrid",
        "entity_type": "LOC",
        "frequency": 5,
        "canonical_form": "madrid"
    },
    {
        "url": "https://ejemplo.com/page1",
        "entity_text": "Google",
        "entity_type": "ORG",
        "frequency": 3,
        "canonical_form": "google"
    }
]

count = orchestrator.save_entities(entities, replace=False)

# Obtener entidades
df = orchestrator.get_entities(
    url="https://ejemplo.com/page1",
    entity_type="LOC",
    min_frequency=2
)
```

### Clusters

```python
# Guardar resultados de clustering
clusters_df = pd.DataFrame({
    "url": ["url1", "url2", "url3"],
    "cluster_id": [0, 0, 1],
    "cluster_label": ["Grupo A", "Grupo A", "Grupo B"],
    "distance_to_centroid": [0.1, 0.15, 0.08]
})

count = orchestrator.save_clusters(
    clusters_df,
    model="paraphrase-multilingual-MiniLM-L12-v2",
    replace=True
)

# Obtener clusters
df = orchestrator.get_clusters(model="paraphrase-multilingual-MiniLM-L12-v2")
```

### Análisis de FAQs

```python
# Guardar análisis de FAQs
faqs = [
    {
        "question": "¿Qué es el SEO?",
        "answer": "El SEO es...",
        "url": "https://ejemplo.com/seo",
        "similarity_score": 0.92,
        "keywords": ["seo", "optimización", "búsqueda"]
    }
]

count = orchestrator.save_faq_analysis(faqs, replace=True)

# Obtener FAQs relevantes
df = orchestrator.get_faq_analysis(min_score=0.8)
```

### Estadísticas

```python
# Obtener estadísticas generales
stats = orchestrator.get_stats()
print(stats)
# {'urls': 10, 'embeddings': 10, 'gsc_positions': 500,
#  'keyword_families': 3, 'semantic_relations': 25, ...}
```

---

## 🧠 EmbeddingCache

Sistema híbrido DuckDB + FAISS para caché de embeddings con búsqueda rápida.

### Inicialización

```python
from embedding_cache import EmbeddingCache

# Crear caché para un proyecto
cache = EmbeddingCache(
    project_path="workspace/projects/mi-cliente",
    model_name="paraphrase-multilingual-MiniLM-L12-v2",
    use_faiss=True  # Usar FAISS si está disponible
)
```

### Añadir Embeddings

```python
import numpy as np

# Añadir un embedding
url = "https://ejemplo.com/page1"
embedding = np.random.rand(384)

cache.add_embedding(url, embedding, rebuild_index=False)

# Añadir múltiples embeddings (batch)
urls = ["url1.com", "url2.com", "url3.com"]
embeddings = np.random.rand(3, 384)

count = cache.add_embeddings_batch(urls, embeddings)
print(f"Añadidos {count} embeddings")
# El índice FAISS se reconstruye automáticamente
```

### Obtener Embeddings

```python
# Obtener un embedding específico
embedding = cache.get_embedding("https://ejemplo.com/page1")
if embedding is not None:
    print(f"Dimensión: {len(embedding)}")

# Verificar si existe
exists = cache.has_embedding("https://ejemplo.com/page1")

# Obtener todos los embeddings
urls, embeddings_matrix = cache.get_all_embeddings()
print(f"Total URLs: {len(urls)}")
print(f"Shape: {embeddings_matrix.shape}")
```

### Búsqueda de Similitud

```python
# Buscar los 10 más similares
query_embedding = np.random.rand(384)

results = cache.search_similar(
    query_embedding,
    top_k=10,
    exclude_urls=["https://ejemplo.com/self"]  # Excluir URLs específicas
)

for result in results:
    print(f"URL: {result['url']}")
    print(f"Similitud: {result['similarity']:.3f}")
    print(f"Distancia: {result['distance']:.3f}")
    print("---")

# Si FAISS está disponible, la búsqueda es 100-1000x más rápida
# Si no, usa búsqueda lineal con numpy (fallback automático)
```

### Helper: Get or Compute

```python
from embedding_cache import get_or_compute_embedding

def compute_embedding_fn(url, model):
    # Tu lógica para computar embedding
    # Por ejemplo, con sentence-transformers
    from sentence_transformers import SentenceTransformer
    model_obj = SentenceTransformer(model)
    # Extraer texto de la URL y computar
    text = extract_text(url)
    return model_obj.encode(text)

# Obtiene del caché o computa si no existe
embedding = get_or_compute_embedding(
    cache,
    url="https://ejemplo.com/page1",
    compute_fn=compute_embedding_fn,
    model="paraphrase-multilingual-MiniLM-L12-v2"
)
```

### Sincronización y Mantenimiento

```python
# Sincronizar índice FAISS con DuckDB
cache.sync_from_db()

# Obtener estadísticas del caché
stats = cache.get_cache_stats()
print(stats)
# {
#   'model': 'paraphrase-multilingual-MiniLM-L12-v2',
#   'count': 100,
#   'dimension': 384,
#   'faiss_enabled': True,
#   'faiss_index_exists': True,
#   'faiss_size_mb': 0.5
# }

# Limpiar caché (solo FAISS, no DuckDB)
cache.clear_cache(confirm=True)
```

---

## 🔗 Integración en Streamlit

### Ejemplo completo con project selector

```python
import streamlit as st
from data_orchestrator import get_data_orchestrator
from embedding_cache import EmbeddingCache

# Obtener proyecto actual
project_config = st.session_state.get("project_config")

if not project_config:
    st.warning("Selecciona un proyecto en el sidebar")
    st.stop()

# Inicializar componentes
orchestrator = get_data_orchestrator(project_config)
cache = EmbeddingCache(
    project_path=project_config["path"],
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

# Usar en tu lógica
if st.button("Procesar URLs"):
    urls = ["url1.com", "url2.com"]

    # Guardar URLs
    orchestrator.save_urls([
        {"url": url, "title": f"Título {url}"}
        for url in urls
    ])

    # Computar y guardar embeddings
    embeddings = compute_embeddings(urls)
    cache.add_embeddings_batch(urls, embeddings)

    st.success(f"Procesadas {len(urls)} URLs")

# Mostrar estadísticas
col1, col2 = st.columns(2)
with col1:
    db_stats = orchestrator.get_stats()
    st.metric("URLs en DB", db_stats['urls'])

with col2:
    cache_stats = cache.get_cache_stats()
    st.metric("Embeddings", cache_stats['count'])
```

---

## 💡 Mejores Prácticas

### DataOrchestrator

1. **Usa get_data_orchestrator()** en lugar de instanciar directamente
2. **Guarda en batch** cuando sea posible para mejor rendimiento
3. **Usa filtros** en get_* para reducir memoria
4. **Cierra conexiones** (se hace automáticamente, pero ten en cuenta)

### EmbeddingCache

1. **Usa batch inserts** para múltiples embeddings
2. **Habilita FAISS** para datasets grandes (>1000 embeddings)
3. **No rebuilds frecuentes** - solo después de batch inserts
4. **Sincroniza después de cambios** externos en DuckDB
5. **Normaliza embeddings** antes de búsqueda si es necesario

### General

1. **Verifica proyecto activo** antes de usar orchestrator/cache
2. **Maneja errores** - ambas clases pueden lanzar excepciones
3. **Monitorea tamaño** de DB con get_stats()
4. **Backups periódicos** de database.duckdb

---

## 🐛 Troubleshooting

**Error: DuckDB no está instalado**
```bash
pip install duckdb
```

**Error: FAISS no está disponible**
```bash
# FAISS es opcional, la búsqueda funcionará sin él (más lento)
pip install faiss-cpu  # O faiss-gpu si tienes CUDA
```

**Error: Embedding dimension mismatch**
- Verifica que todos los embeddings tengan la misma dimensión
- No mezcles modelos diferentes en el mismo caché

**Performance lento en búsqueda**
- Activa FAISS: `use_faiss=True`
- Verifica que el índice FAISS se haya construido: `cache.sync_from_db()`

**Database locked**
- DuckDB no permite múltiples escritores
- Cierra otras conexiones antes de escribir
- Usa `read_only=True` para lecturas paralelas

---

## 📖 Documentación Adicional

- [ROADMAP.md](../ROADMAP.md) - Plan de desarrollo completo
- [shared/db_schema.py](db_schema.py) - Definición del schema DuckDB
- [shared/project_manager.py](project_manager.py) - Gestión de proyectos

**Última actualización:** 2025-12-30
**Versión:** 1.0.0
