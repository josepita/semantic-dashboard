"""
SEO Content Analyzer - Refactored
==================================

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
from modules.semantic_tools import render_semantic_toolkit_section
from modules.keyword_builder import render_semantic_keyword_builder
from modules.semantic_relations import render_semantic_relations

# Configurar página
setup_page_config(
    title="SEO Content Analyzer",
    icon="🎯",
    layout="wide"
)


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
        **1. Herramientas Semánticas**
        - Ve a "🧰 Herramientas Semánticas"
        - Elige: Texto, FAQs, Competidores o Variantes
        - Introduce tus datos y keywords
        - Obtén análisis de relevancia

        **2. Keyword Builder**
        - Ve a "🧠 Semantic Keyword Builder"
        - Sube CSV o pega keywords
        - Obtén agrupación automática
        - Exporta clusters a Excel

        **3. Relaciones Semánticas**
        - Ve a "🔗 Relaciones Semánticas"
        - Sube CSV con URLs
        - Visualiza relaciones
        - Identifica topic clusters
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

    # Características destacadas
    st.markdown("---")
    st.markdown("### ✨ Características Destacadas")

    col_feat1, col_feat2, col_feat3 = st.columns(3)

    with col_feat1:
        st.markdown("#### 📝 Análisis de Texto")
        st.markdown("""
        - Relevancia semántica en tiempo real
        - Comparación con múltiples keywords
        - Scoring de 0-100%
        - Exportación a Excel
        """)

    with col_feat2:
        st.markdown("#### ❓ FAQs Inteligentes")
        st.markdown("""
        - **Carga de Excel/CSV** ⭐ NUEVO
        - Selector de columnas
        - Top N por keyword
        - Análisis masivo
        """)

    with col_feat3:
        st.markdown("#### 🔍 Competidores")
        st.markdown("""
        - Extracción automática de contenido
        - Análisis de gap
        - Meta descriptions
        - Exportación de insights
        """)

    # Tips de uso
    st.markdown("---")
    with st.expander("💡 Tips de Uso", expanded=False):
        st.markdown("""
        **Performance:**
        - Los modelos se cachean automáticamente
        - Primera carga: ~10s, siguientes: instantáneas

        **Calidad:**
        - Relevancia >70% = Bien optimizado
        - Relevancia >85% = Excelente
        - Relevancia <50% = Necesita mejora

        **FAQs:**
        - Usa Excel para análisis masivos (>50 FAQs)
        - Formato: 2 columnas (pregunta, respuesta)
        - Soporta CSV, XLSX, XLS

        **Keywords:**
        - Agrupa keywords ANTES de crear contenido
        - Identifica temas principales primero
        - Exporta mapeo para planificación

        **Exportar:**
        - Todos los análisis se pueden exportar
        - Formato Excel para fácil lectura
        - Incluye métricas y scores
        """)

    # Casos de uso
    st.markdown("---")
    st.markdown("### 🎯 Casos de Uso")

    tab1, tab2, tab3 = st.tabs(["Content Writer", "SEO Strategist", "Manager"])

    with tab1:
        st.markdown("""
        **Para Content Writers:**

        1. **Optimizar Meta Descriptions**
           - Pega tu meta en "Texto vs Keywords"
           - Analiza relevancia
           - Ajusta hasta alcanzar >70%

        2. **Validar Contenido**
           - Analiza párrafos importantes
           - Verifica relevancia para keywords target
           - Mejora donde sea necesario

        3. **Crear FAQs**
           - Analiza FAQs existentes
           - Identifica gaps de keywords
           - Crea nuevas FAQs relevantes
        """)

    with tab2:
        st.markdown("""
        **Para SEO Strategists:**

        1. **Keyword Research**
           - Usa Keyword Builder para agrupar
           - Identifica temas principales
           - Planifica arquitectura de contenido

        2. **Gap Analysis**
           - Analiza competidores
           - Detecta keywords que cubren
           - Crea plan de contenido

        3. **Topic Clusters**
           - Usa Relaciones Semánticas
           - Identifica pillar pages
           - Mapea supporting content
        """)

    with tab3:
        st.markdown("""
        **Para Managers:**

        1. **Reportes de Calidad**
           - Análisis masivo de FAQs
           - Exporta scores a Excel
           - Presenta a stakeholders

        2. **Priorización**
           - Identifica contenido con bajo score
           - Prioriza optimizaciones
           - Trackea mejoras

        3. **Planificación**
           - Usa clusters de keywords
           - Planifica calendario editorial
           - Asigna temas a writers
        """)


def main():
    """Main application entry point."""
    apply_global_styles()

    # Título y descripción
    st.title("🎯 SEO Content Analyzer")
    st.markdown(
        "Analiza contenido SEO con herramientas semánticas avanzadas: "
        "keywords, FAQs, competidores y más."
    )

    # Sidebar con navegación
    tools = [
        "🏠 Inicio",
        "🧰 Herramientas Semánticas",
        "🧠 Semantic Keyword Builder",
        "🔗 Relaciones Semánticas",
    ]
    
    tool = render_app_sidebar("SEO Content Analyzer", tools)

    # Renderizar herramienta seleccionada
    if tool == "🏠 Inicio":
        render_home()
    elif tool == "🧰 Herramientas Semánticas":
        render_semantic_toolkit_section()
    elif tool == "🧠 Semantic Keyword Builder":
        render_semantic_keyword_builder()
    elif tool == "🔗 Relaciones Semánticas":
        render_semantic_relations()


if __name__ == "__main__":
    main()
