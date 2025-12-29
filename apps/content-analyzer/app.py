"""
SEO Content Analyzer
====================

Herramienta especializada para análisis semántico de contenido SEO.

Funcionalidades:
- Análisis de texto vs keywords
- Análisis de FAQs (con carga de Excel)
- Comparación con competidores
- Semantic Keyword Builder
- Análisis de relaciones semánticas

Autor: Embedding Insights
Versión: 1.0.0
"""

import streamlit as st
import sys
from pathlib import Path

# Añadir shared library al path
shared_path = Path(__file__).parent.parent.parent / "shared"
sys.path.insert(0, str(shared_path))

# Importar módulos compartidos
from entity_filters import clean_entities_advanced, lemmatize_text

st.set_page_config(
    page_title="SEO Content Analyzer",
    layout="wide",
    page_icon="🎯",
)


def main():
    """Main application entry point."""

    # Título y descripción
    st.title("🎯 SEO Content Analyzer")
    st.markdown(
        "Analiza contenido SEO con herramientas semánticas avanzadas: "
        "keywords, FAQs, competidores y más."
    )

    # Sidebar - Navegación
    with st.sidebar:
        st.header("🧭 Herramientas")

        tool = st.radio(
            "Selecciona una herramienta:",
            options=[
                "🏠 Inicio",
                "📝 Texto vs Keywords",
                "❓ FAQs vs Keywords",
                "🔍 Análisis de Competidores",
                "🧠 Semantic Keyword Builder",
                "🔗 Relaciones Semánticas",
            ],
            key="tool_selector"
        )

    # Renderizar herramienta seleccionada
    if tool == "🏠 Inicio":
        render_home()
    elif tool == "📝 Texto vs Keywords":
        render_text_analysis()
    elif tool == "❓ FAQs vs Keywords":
        render_faq_analysis()
    elif tool == "🔍 Análisis de Competidores":
        render_competitor_analysis()
    elif tool == "🧠 Semantic Keyword Builder":
        render_keyword_builder()
    elif tool == "🔗 Relaciones Semánticas":
        render_semantic_relations()


def render_home():
    """Renderiza la página de inicio."""
    st.header("👋 Bienvenido a SEO Content Analyzer")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 ¿Qué puedes hacer?")
        st.markdown("""
        **Análisis de Contenido:**
        - ✅ Evaluar relevancia semántica de textos
        - ✅ Optimizar FAQs para keywords específicas
        - ✅ Comparar tu contenido vs competidores

        **Keyword Research:**
        - ✅ Agrupar keywords por similitud semántica
        - ✅ Detectar temas y clusters
        - ✅ Identificar keywords principales

        **Análisis Avanzado:**
        - ✅ Relaciones semánticas entre URLs
        - ✅ Topic clusters y pillar pages
        - ✅ Visualización de grafos de relaciones
        """)

    with col2:
        st.markdown("### 🚀 Quick Start")
        st.markdown("""
        **1. Texto vs Keywords**
        - Pega tu texto y tus keywords
        - Obtén score de relevancia semántica
        - Exporta resultados a Excel

        **2. FAQs vs Keywords**
        - Carga Excel con preguntas/respuestas
        - Analiza relevancia por keyword
        - Identifica mejores FAQs

        **3. Keyword Builder**
        - Sube lista de keywords
        - Obtén agrupación automática
        - Planifica arquitectura de contenido
        """)

    # Estadísticas y métricas
    st.markdown("---")
    st.markdown("### 📊 Tecnología")

    col_tech1, col_tech2, col_tech3, col_tech4 = st.columns(4)

    with col_tech1:
        st.metric("Modelos AI", "3", help="Sentence Transformers, OpenAI, Gemini")
    with col_tech2:
        st.metric("Idiomas", "ES/EN", help="Soporte para español e inglés")
    with col_tech3:
        st.metric("Formatos", "CSV/Excel", help="Carga y exportación")
    with col_tech4:
        st.metric("NLP", "spaCy", help="Lemmatización y entidades")

    # Tips
    with st.expander("💡 Tips de Uso"):
        st.markdown("""
        - **Performance:** Los modelos se cachean automáticamente
        - **Calidad:** Mayor relevancia = mejor optimización SEO
        - **FAQs:** Usa Excel para análisis masivos (>50 FAQs)
        - **Keywords:** Agrupa keywords antes de crear contenido
        - **Exportar:** Todos los análisis se pueden exportar a Excel
        """)


def render_text_analysis():
    """Renderiza análisis de texto vs keywords."""
    st.header("📝 Análisis: Texto vs Keywords")

    st.info("⚙️ Módulo en desarrollo - próximamente disponible")
    st.markdown("""
    **Funcionalidad:**
    - Pega cualquier texto (meta description, párrafo, contenido)
    - Introduce tus keywords target
    - Obtén score de relevancia semántica
    - Identifica keywords más relevantes para el texto
    """)


def render_faq_analysis():
    """Renderiza análisis de FAQs vs keywords."""
    st.header("❓ Análisis: FAQs vs Keywords")

    st.info("⚙️ Módulo en desarrollo - próximamente disponible")
    st.markdown("""
    **Funcionalidad:**
    - Carga Excel con columnas de preguntas y respuestas
    - Introduce keywords a analizar
    - Obtén relevancia de cada FAQ
    - Exporta top FAQs por keyword
    """)


def render_competitor_analysis():
    """Renderiza análisis de competidores."""
    st.header("🔍 Análisis de Competidores")

    st.info("⚙️ Módulo en desarrollo - próximamente disponible")
    st.markdown("""
    **Funcionalidad:**
    - Introduce URLs de competidores
    - Extrae contenido automáticamente
    - Compara con tus keywords target
    - Detecta gaps de contenido
    """)


def render_keyword_builder():
    """Renderiza Semantic Keyword Builder."""
    st.header("🧠 Semantic Keyword Builder")

    st.info("⚙️ Módulo en desarrollo - próximamente disponible")
    st.markdown("""
    **Funcionalidad:**
    - Sube CSV con keywords
    - Agrupación automática por similitud
    - Detección de temas principales
    - Exporta mapeo keyword → cluster
    """)


def render_semantic_relations():
    """Renderiza análisis de relaciones semánticas."""
    st.header("🔗 Relaciones Semánticas")

    st.info("⚙️ Módulo en desarrollo - próximamente disponible")
    st.markdown("""
    **Funcionalidad:**
    - Analiza relaciones entre URLs
    - Visualiza grafos de contenido
    - Identifica pillar pages
    - Detecta topic clusters
    """)


if __name__ == "__main__":
    main()
