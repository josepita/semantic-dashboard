"""
GSC Insights & Reporting
========================

Herramienta especializada para análisis de posiciones SEO con datos de
rank tracking y análisis competitivo avanzado.

Funcionalidades:
- Importación de datos de rank tracking
- Análisis competitivo por familias de keywords
- Generación de informes HTML con gráficos
- Agrupación inteligente de keywords
- Insights con Gemini AI

Autor: Embedding Insights
Versión: 1.0.0
"""

import streamlit as st
import sys
from pathlib import Path

# Añadir paths al sistema (resolver a paths absolutos)
current_dir = Path(__file__).parent.resolve()
shared_path = (current_dir.parent.parent / "shared").resolve()
modules_path = (current_dir / "modules").resolve()

# Añadir paths ANTES de cualquier import
if str(shared_path) not in sys.path:
    sys.path.insert(0, str(shared_path))
if str(modules_path) not in sys.path:
    sys.path.insert(0, str(modules_path))

# Importar módulos
from modules.positions_report import render_positions_report

# Import con manejo de errores
try:
    from project_manager import get_project_manager
except ImportError:
    # Fallback: importar directamente desde shared
    import importlib.util
    pm_path = shared_path / "project_manager.py"
    if not pm_path.exists():
        raise ImportError(f"No se encuentra project_manager.py en {shared_path}")
    spec = importlib.util.spec_from_file_location("project_manager", str(pm_path))
    project_manager = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(project_manager)
    get_project_manager = project_manager.get_project_manager

st.set_page_config(
    page_title="GSC Insights & Reporting",
    layout="wide",
    page_icon="📊",
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
                    st.metric("Registros GSC", stats.get("gsc_records", 0))
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
    st.title("📊 GSC Insights & Reporting")
    st.markdown(
        "Genera informes avanzados de posiciones SEO con análisis competitivo "
        "y agrupación inteligente de keywords."
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
                "📈 Informe de Posiciones",
            ],
            key="tool_selector"
        )

        st.markdown("---")
        st.markdown("### ℹ️ Acerca de")
        st.caption("GSC Insights & Reporting v1.0.0")
        st.caption("Parte de Embedding Insights Suite")

    # Renderizar herramienta seleccionada
    if tool == "🏠 Inicio":
        render_home()
    elif tool == "📈 Informe de Posiciones":
        render_positions_report()


def render_home():
    """Renderiza la página de inicio."""
    st.header("👋 Bienvenido a GSC Insights & Reporting")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 ¿Qué puedes hacer?")
        st.markdown("""
        **Análisis de Posiciones:**
        - ✅ Importar datos de rank tracking
        - ✅ Análisis competitivo automático
        - ✅ Agrupación de keywords por familias
        - ✅ Generación de informes HTML

        **Insights Avanzados:**
        - ✅ Heatmaps de presencia competitiva
        - ✅ Gráficos radar por familia
        - ✅ Análisis de volumen de búsqueda
        - ✅ Recomendaciones con Gemini AI

        **Exportación:**
        - ✅ Informes HTML interactivos
        - ✅ Datos en Excel
        - ✅ Gráficos descargables
        - ✅ Análisis por competidor
        """)

    with col2:
        st.markdown("### 🚀 Quick Start")
        st.markdown("""
        **1. Preparar Datos**
        - Exporta rank tracking desde tu herramienta SEO
        - Formato CSV con columnas: keyword, position, URL
        - Opcional: volumen de búsqueda, fecha

        **2. Cargar y Configurar**
        - Ve a "📈 Informe de Posiciones"
        - Sube CSV de rank tracking
        - Define tu dominio principal
        - Opcional: Agrega familias de keywords

        **3. Generar Informe**
        - Selecciona tipos de gráficos
        - Configura Gemini API (opcional)
        - Genera informe HTML
        - Descarga y comparte

        **4. Análisis Avanzado**
        - Revisa competidores principales
        - Identifica oportunidades por familia
        - Exporta datos para optimización
        - Implementa mejoras
        """)

    # Estadísticas
    st.markdown("---")
    st.markdown("### 📊 Tecnología")

    col_tech1, col_tech2, col_tech3, col_tech4 = st.columns(4)

    with col_tech1:
        st.metric("IA", "Gemini 2.0", help="Análisis automático")
    with col_tech2:
        st.metric("Formatos", "CSV/Excel", help="Import/Export")
    with col_tech3:
        st.metric("Gráficos", "Interactivos", help="HTML embebido")
    with col_tech4:
        st.metric("Familias", "Auto", help="Agrupación automática")

    # Características destacadas
    st.markdown("---")
    st.markdown("### ✨ Características Destacadas")

    col_feat1, col_feat2, col_feat3 = st.columns(3)

    with col_feat1:
        st.markdown("#### 📈 Análisis de Posiciones")
        st.markdown("""
        - Carga CSV de rank tracking
        - Detección automática de dominios
        - Normalización de URLs
        - Vista por keyword y por URL
        """)

    with col_feat2:
        st.markdown("#### 🎯 Familias de Keywords")
        st.markdown("""
        - **Agrupación automática** ⭐
        - Definición manual o CSV
        - Análisis por familia
        - Volumen agregado
        """)

    with col_feat3:
        st.markdown("#### 📊 Informes HTML")
        st.markdown("""
        - Gráficos interactivos
        - Heatmaps competitivos
        - Análisis con Gemini AI
        - Descarga y compartir
        """)

    # Tips de uso
    st.markdown("---")
    with st.expander("💡 Tips de Uso", expanded=False):
        st.markdown("""
        **Preparación de Datos:**
        - Exporta al menos 100 keywords para mejores insights
        - Incluye volumen de búsqueda si es posible
        - Mantén formato consistente en URLs

        **Familias de Keywords:**
        - Define 5-10 familias principales
        - Usa clustering automático con embeddings
        - Revisa y ajusta manualmente

        **Análisis Competitivo:**
        - Identifica competidores recurrentes
        - Analiza por familia, no general
        - Busca gaps de contenido

        **Gemini AI:**
        - Obtén API key gratis en Google AI Studio
        - Usa para insights automáticos
        - Revisa recomendaciones antes de implementar

        **Exportación:**
        - Genera HTML para presentaciones
        - Exporta Excel para análisis detallado
        - Comparte con equipo técnico
        """)

    # Casos de uso
    st.markdown("---")
    st.markdown("### 🎯 Casos de Uso")

    tab1, tab2, tab3 = st.tabs(["SEO Manager", "Content Strategist", "Agency"])

    with tab1:
        st.markdown("""
        **Para SEO Managers:**

        1. **Monitoreo Mensual**
           - Sube datos de rank tracker
           - Genera informe HTML
           - Presenta a stakeholders

        2. **Análisis Competitivo**
           - Identifica competidores por familia
           - Analiza gaps de contenido
           - Planifica estrategia

        3. **Priorización**
           - Revisa quick wins (posiciones 4-10)
           - Identifica keywords de alto volumen
           - Crea plan de acción
        """)

    with tab2:
        st.markdown("""
        **Para Content Strategists:**

        1. **Planificación de Contenido**
           - Analiza familias de keywords
           - Identifica temas sin cubrir
           - Crea calendario editorial

        2. **Optimización Existente**
           - Detecta contenido infraoptimizado
           - Revisa competencia por tema
           - Actualiza contenido

        3. **Clustering de Topics**
           - Agrupa keywords semánticamente
           - Define pillar pages
           - Planifica supporting content
        """)

    with tab3:
        st.markdown("""
        **Para Agencias:**

        1. **Reportes Cliente**
           - Genera informes HTML branded
           - Exporta datos para análisis
           - Presenta evolución mensual

        2. **Análisis Multi-Cliente**
           - Compara rendimiento
           - Identifica best practices
           - Escala estrategias exitosas

        3. **Automatización**
           - Integra con rank trackers
           - Programa generación mensual
           - Entrega automática
        """)


if __name__ == "__main__":
    main()
