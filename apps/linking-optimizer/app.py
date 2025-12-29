"""
Internal Linking Optimizer
===========================

Herramienta especializada para optimización de enlazado interno basada en
análisis semántico y embeddings.

Funcionalidades:
- Análisis de similitud entre URLs
- Clustering automático de contenido
- Authority Gap Analysis
- Recomendaciones de enlaces inteligentes
- Knowledge Graph con entidades
- PageRank simulation

Autor: Embedding Insights
Versión: 1.0.0
"""

import streamlit as st
import sys
from pathlib import Path

# Añadir paths al sistema
current_dir = Path(__file__).parent
shared_path = current_dir.parent.parent / "shared"
modules_path = current_dir / "modules"

sys.path.insert(0, str(shared_path))
sys.path.insert(0, str(modules_path))

# Importar módulos
from modules.csv_workflow import render_csv_workflow
from modules.linking_lab import render_linking_lab

st.set_page_config(
    page_title="Internal Linking Optimizer",
    layout="wide",
    page_icon="🔗",
)


def apply_global_styles():
    """Aplicar estilos globales."""
    st.markdown("""
    <style>
    .main {
        padding: 1rem;
    }
    .stButton button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)


def main():
    """Main application entry point."""
    apply_global_styles()

    # Título y descripción
    st.title("🔗 Internal Linking Optimizer")
    st.markdown(
        "Optimiza tu enlazado interno con análisis semántico, "
        "clustering y recomendaciones basadas en IA."
    )

    # Sidebar - Navegación
    with st.sidebar:
        st.header("🧭 Navegación")

        tool = st.radio(
            "Selecciona una herramienta:",
            options=[
                "🏠 Inicio",
                "📂 Análisis de Embeddings",
                "🔗 Laboratorio de Enlazado",
            ],
            key="tool_selector"
        )

        st.markdown("---")
        st.markdown("### ℹ️ Acerca de")
        st.caption("Internal Linking Optimizer v1.0.0")
        st.caption("Parte de Embedding Insights Suite")

    # Renderizar herramienta seleccionada
    if tool == "🏠 Inicio":
        render_home()
    elif tool == "📂 Análisis de Embeddings":
        render_csv_workflow()
    elif tool == "🔗 Laboratorio de Enlazado":
        render_linking_lab()


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

    # Tips
    with st.expander("💡 Tips de Uso"):
        st.markdown("""
        - **Embeddings:** Usa OpenAI text-embedding-3-small o similar
        - **Clustering:** Más URLs = mejor detección de silos
        - **Authority Gap:** Prioriza páginas con alto tráfico/conversión
        - **Knowledge Graph:** Requiere contenido en HTML o texto
        - **Performance:** >1000 URLs puede tardar varios minutos
        """)


if __name__ == "__main__":
    main()
