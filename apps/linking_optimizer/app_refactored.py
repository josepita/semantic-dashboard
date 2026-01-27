"""
Internal Linking Optimizer - Refactored
========================================

Versión refactorizada usando módulos compartidos para eliminar duplicación.

Autor: Embedding Insights
Versión: 1.0.1
"""

import streamlit as st
import sys
from pathlib import Path

# Añadir paths al sistema
current_dir = Path(__file__).parent.resolve()
shared_path = (current_dir.parent.parent / "shared").resolve()
modules_path = (current_dir / "modules").resolve()

if str(shared_path) not in sys.path:
    sys.path.insert(0, str(shared_path))
if str(modules_path) not in sys.path:
    sys.path.insert(0, str(modules_path))

# Importar módulos compartidos
from app_common import (
    setup_page_config,
    apply_global_styles,
    render_app_sidebar,
)

# Importar módulos específicos de la app
from modules.csv_workflow import render_csv_workflow
from modules.linking_lab import render_linking_lab

# Configurar página
setup_page_config(
    title="Internal Linking Optimizer",
    icon="🔗",
    layout="wide"
)


def render_home():
    """Renderiza la página de inicio."""
    st.header("👋 Bienvenido a Internal Linking Optimizer")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 ¿Qué puedes hacer?")
        st.markdown("""
        **Análisis de Arquitectura:**
        - ✅ Subir CSV con embeddings de URLs
        - ✅ Detectar similitud semántica entre páginas
        - ✅ Clustering automático de contenido
        - ✅ Visualización t-SNE en 2D

        **Optimización de Enlaces:**
        - ✅ Authority Gap Analysis
        - ✅ Recomendaciones basadas en:
          - Similitud semántica
          - Entidades compartidas
          - Page types y silos
        - ✅ Simulación de PageRank

        **Knowledge Graph:**
        - ✅ Extracción de entidades (spaCy)
        - ✅ Análisis de co-ocurrencias
        - ✅ Entity linking con Wikidata
        - ✅ Visualización de grafos
        """)

    with col2:
        st.markdown("### 🚀 Quick Start")
        st.markdown("""
        **1. Preparar Datos**
        - Exporta embeddings de tus URLs
        - Formato CSV con columnas: `url`, `embedding`
        - Opcional: `page_type`, `title`, `meta_description`

        **2. Subir y Analizar**
        - Carga CSV en "📂 Cargar Embeddings"
        - Ejecuta análisis de similitud
        - Visualiza clusters en t-SNE

        **3. Optimizar Enlaces**
        - Ve a "📊 Authority Gap"
        - Identifica páginas infraenlazadas
        - Aplica recomendaciones IA

        **4. Exportar Resultados**
        - Descarga recomendaciones en Excel
        - Implementa enlaces sugeridos
        - Monitorea impacto
        """)

    # Estadísticas
    st.markdown("---")
    st.markdown("### 📊 Tecnología")

    col_tech1, col_tech2, col_tech3, col_tech4 = st.columns(4)

    with col_tech1:
        st.metric("Algoritmos", "5", help="KMeans, t-SNE, PageRank, etc.")
    with col_tech2:
        st.metric("NLP", "spaCy", help="Extracción de entidades")
    with col_tech3:
        st.metric("Grafos", "NetworkX", help="Knowledge graphs")
    with col_tech4:
        st.metric("Viz", "Plotly+Pyvis", help="Visualizaciones interactivas")

    # Características destacadas
    st.markdown("---")
    st.markdown("### ✨ Características Destacadas")

    col_feat1, col_feat2, col_feat3 = st.columns(3)

    with col_feat1:
        st.markdown("#### 🔍 Análisis Semántico")
        st.markdown("""
        - Similitud coseno entre URLs
        - Clustering automático (KMeans)
        - Visualización t-SNE 2D/3D
        - Detección de topic clusters
        """)

    with col_feat2:
        st.markdown("#### 🔗 Laboratorio de Enlaces")
        st.markdown("""
        - **4 modos de análisis** ⭐
        - Básico (similitud)
        - Avanzado (silos)
        - Híbrido (CLS + PageRank)
        - Estructural (taxonomía)
        """)

    with col_feat3:
        st.markdown("#### 🕸️ Knowledge Graph")
        st.markdown("""
        - Extracción de entidades NER
        - Entity linking Wikidata
        - Grafos interactivos
        - Semantic Depth Score
        """)

    # Tips
    st.markdown("---")
    with st.expander("💡 Tips de Uso", expanded=False):
        st.markdown("""
        **Embeddings:**
        - Usa OpenAI text-embedding-3-small o similar
        - Sentence Transformers también funciona
        - Normaliza vectores antes de subir

        **Clustering:**
        - Más URLs = mejor detección de silos
        - Prueba diferentes valores de K
        - Revisa silhouette score

        **Authority Gap:**
        - Prioriza páginas con alto tráfico/conversión
        - Combina con datos de GSC
        - Implementa gradualmente

        **Knowledge Graph:**
        - Requiere contenido en HTML o texto
        - spaCy detecta entidades automáticamente
        - Filtra entidades irrelevantes

        **Performance:**
        - >1000 URLs puede tardar varios minutos
        - Usa caché de embeddings
        - Procesa en batches si es necesario
        """)

    # Casos de uso
    st.markdown("---")
    st.markdown("### 🎯 Casos de Uso")

    tab1, tab2, tab3 = st.tabs(["SEO Técnico", "Content Strategy", "Enterprise"])

    with tab1:
        st.markdown("""
        **Para SEO Técnico:**

        1. **Auditoría de Arquitectura**
           - Analiza estructura de silos
           - Detecta páginas huérfanas
           - Identifica oportunidades de enlazado

        2. **Optimización de PageRank**
           - Simula distribución de autoridad
           - Identifica páginas infraenlazadas
           - Implementa enlaces estratégicos

        3. **Topic Clustering**
           - Agrupa contenido por tema
           - Define pillar pages
           - Planifica supporting content
        """)

    with tab2:
        st.markdown("""
        **Para Content Strategy:**

        1. **Planificación de Contenido**
           - Identifica gaps temáticos
           - Detecta contenido relacionado
           - Crea calendario editorial

        2. **Optimización de Enlaces**
           - Enlaces contextuales relevantes
           - Distribución de autoridad
           - Mejora experiencia usuario

        3. **Knowledge Graph**
           - Mapea entidades clave
           - Identifica relaciones
           - Crea contenido conectado
        """)

    with tab3:
        st.markdown("""
        **Para Enterprise:**

        1. **Escalabilidad**
           - Procesa miles de URLs
           - Automatiza recomendaciones
           - Integra con CMS

        2. **Multi-sitio**
           - Analiza múltiples dominios
           - Compara arquitecturas
           - Replica best practices

        3. **Reporting**
           - Exporta a Excel/CSV
           - Visualizaciones para stakeholders
           - Tracking de implementación
        """)


def main():
    """Main application entry point."""
    apply_global_styles()

    # Título y descripción
    st.title("🔗 Internal Linking Optimizer")
    st.markdown(
        "Optimiza tu enlazado interno con análisis semántico, "
        "clustering y recomendaciones basadas en IA."
    )

    # Sidebar con navegación
    tools = [
        "🏠 Inicio",
        "📂 Análisis de Embeddings",
        "🔗 Laboratorio de Enlazado",
    ]
    
    tool = render_app_sidebar("Internal Linking Optimizer", tools)

    # Renderizar herramienta seleccionada
    if tool == "🏠 Inicio":
        render_home()
    elif tool == "📂 Análisis de Embeddings":
        render_csv_workflow()
    elif tool == "🔗 Laboratorio de Enlazado":
        render_linking_lab()


if __name__ == "__main__":
    main()
