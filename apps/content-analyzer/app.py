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

# Añadir paths al sistema
current_dir = Path(__file__).parent
shared_path = current_dir.parent.parent / "shared"
modules_path = current_dir / "modules"

# Añadir paths ANTES de cualquier import
if str(shared_path) not in sys.path:
    sys.path.insert(0, str(shared_path))
if str(modules_path) not in sys.path:
    sys.path.insert(0, str(modules_path))

# Importar módulos
from modules.semantic_tools import render_semantic_toolkit_section
from modules.keyword_builder import render_semantic_keyword_builder
from modules.semantic_relations import render_semantic_relations

# Import con manejo de errores
try:
    from project_manager import get_project_manager
except ImportError:
    # Fallback: importar directamente desde shared
    import importlib.util
    spec = importlib.util.spec_from_file_location("project_manager", shared_path / "project_manager.py")
    project_manager = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(project_manager)
    get_project_manager = project_manager.get_project_manager

st.set_page_config(
    page_title="SEO Content Analyzer",
    layout="wide",
    page_icon="🎯",
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


def render_project_selector():
    """Renderiza el selector de proyectos en el sidebar."""
    st.sidebar.header("📁 Proyecto Actual")

    pm = get_project_manager()

    # Inicializar session state
    if "current_project" not in st.session_state:
        last_project = pm.get_last_project()
        st.session_state.current_project = last_project

    # Listar proyectos disponibles
    projects = pm.list_projects()
    project_names = [p["name"] for p in projects]

    if not projects:
        st.sidebar.warning("⚠️ No hay proyectos creados")
        st.session_state.current_project = None
    else:
        # Selector de proyecto
        current_index = 0
        if st.session_state.current_project:
            try:
                current_index = project_names.index(st.session_state.current_project)
            except ValueError:
                current_index = 0

        selected_project = st.sidebar.selectbox(
            "Selecciona un proyecto:",
            options=project_names,
            index=current_index,
            key="project_selector"
        )

        # Actualizar si cambió
        if selected_project != st.session_state.current_project:
            st.session_state.current_project = selected_project
            pm.set_last_project(selected_project)
            st.rerun()

        # Cargar configuración del proyecto
        if st.session_state.current_project:
            try:
                project_config = pm.load_project(st.session_state.current_project)
                st.session_state.project_config = project_config

                # Mostrar info del proyecto
                st.sidebar.success(f"✅ {project_config['domain']}")

                # Stats del proyecto
                with st.sidebar.expander("📊 Estadísticas", expanded=False):
                    stats = pm.get_project_stats(st.session_state.current_project)
                    st.metric("URLs", stats.get("urls_count", 0))
                    st.metric("Embeddings", stats.get("embeddings_count", 0))
                    st.metric("Tamaño", f"{stats.get('size_mb', 0)} MB")
            except Exception as e:
                st.sidebar.error(f"Error al cargar proyecto: {e}")

    # Botón para crear nuevo proyecto
    st.sidebar.markdown("---")
    with st.sidebar.expander("➕ Crear Nuevo Proyecto", expanded=False):
        with st.form("new_project_form"):
            new_name = st.text_input("Nombre del proyecto", placeholder="Mi Cliente SEO")
            new_domain = st.text_input("Dominio principal", placeholder="ejemplo.com")
            new_desc = st.text_area("Descripción (opcional)", placeholder="Proyecto de optimización SEO...")

            submit = st.form_submit_button("Crear Proyecto")

            if submit:
                if not new_name or not new_domain:
                    st.error("Nombre y dominio son obligatorios")
                else:
                    try:
                        project_path = pm.create_project(new_name, new_domain, new_desc)
                        safe_name = Path(project_path).name
                        st.session_state.current_project = safe_name
                        pm.set_last_project(safe_name)
                        st.success(f"✅ Proyecto '{new_name}' creado")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")


def main():
    """Main application entry point."""
    apply_global_styles()

    # Título y descripción
    st.title("🎯 SEO Content Analyzer")
    st.markdown(
        "Analiza contenido SEO con herramientas semánticas avanzadas: "
        "keywords, FAQs, competidores y más."
    )

    # Sidebar - Project Selector
    render_project_selector()

    # Sidebar - Navegación
    with st.sidebar:
        st.markdown("---")
        st.header("🧭 Navegación")

        tool = st.radio(
            "Selecciona una herramienta:",
            options=[
                "🏠 Inicio",
                "🧰 Herramientas Semánticas",
                "🧠 Semantic Keyword Builder",
                "🔗 Relaciones Semánticas",
            ],
            key="tool_selector"
        )

        st.markdown("---")
        st.markdown("### ℹ️ Acerca de")
        st.caption("SEO Content Analyzer v1.0.0")
        st.caption("Parte de Embedding Insights Suite")

    # Renderizar herramienta seleccionada
    if tool == "🏠 Inicio":
        render_home()
    elif tool == "🧰 Herramientas Semánticas":
        render_semantic_toolkit_section()
    elif tool == "🧠 Semantic Keyword Builder":
        render_semantic_keyword_builder()
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


if __name__ == "__main__":
    main()
