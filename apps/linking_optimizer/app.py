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

# Añadir paths al sistema (resolver a paths absolutos)
current_dir = Path(__file__).parent.resolve()
project_root = (current_dir.parent.parent).resolve()  # EmbeddingDashboard/
shared_path = (project_root / "shared").resolve()
modules_path = (current_dir / "modules").resolve()

# Añadir paths ANTES de cualquier import
# IMPORTANTE: Añadir project_root para que funcionen imports como "from apps.content_analyzer..."
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(shared_path) not in sys.path:
    sys.path.insert(0, str(shared_path))
if str(modules_path) not in sys.path:
    sys.path.insert(0, str(modules_path))

# Importar módulos
from modules.csv_workflow import render_csv_workflow
from app_sections.linking_lab import render_linking_lab
from shared.env_utils import bootstrap_api_session_state

# License management - TEMPORAL: licencias desactivadas
# TODO: Restaurar verificación de licencias cuando esté listo
def check_license_or_block(): return True
def render_license_status_sidebar(): pass
def require_feature(f, n=""): return True

# Import con manejo de errores
try:
    from project_manager import get_project_manager
    from oauth_manager import get_oauth_manager
    from project_ui import render_export_import_sidebar
except ImportError:
    # Fallback: importar directamente desde shared
    import importlib.util
    pm_path = shared_path / "project_manager.py"
    oauth_path = shared_path / "oauth_manager.py"
    project_ui_path = shared_path / "project_ui.py"

    if not pm_path.exists():
        raise ImportError(f"No se encuentra project_manager.py en {shared_path}")

    # Cargar project_manager
    spec = importlib.util.spec_from_file_location("project_manager", str(pm_path))
    project_manager = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(project_manager)
    get_project_manager = project_manager.get_project_manager

    # Cargar oauth_manager
    if oauth_path.exists():
        spec = importlib.util.spec_from_file_location("oauth_manager", str(oauth_path))
        oauth_manager_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(oauth_manager_module)
        get_oauth_manager = oauth_manager_module.get_oauth_manager
    else:
        # Fallback si no existe
        get_oauth_manager = lambda x: None

    # Cargar project_ui
    if project_ui_path.exists():
        spec = importlib.util.spec_from_file_location("project_ui", str(project_ui_path))
        project_ui_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(project_ui_module)
        render_export_import_sidebar = project_ui_module.render_export_import_sidebar
    else:
        # Fallback si no existe
        render_export_import_sidebar = lambda x: None

st.set_page_config(
    page_title="Internal Linking Optimizer",
    layout="wide",
    page_icon="🔗",
)
# Marcar que page_config ya fue configurado (para license_ui)
st.session_state["_page_config_set"] = True


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

                # Auto-cargar credenciales OAuth (Fase 3)
                oauth_manager = get_oauth_manager(project_config)
                if oauth_manager:
                    st.session_state.oauth_manager = oauth_manager

                    # Cargar API keys si existen
                    openai_key = oauth_manager.load_api_key('openai', 'OPENAI_API_KEY')
                    if openai_key:
                        st.session_state.openai_api_key = openai_key

                # Mostrar info del proyecto
                st.sidebar.success(f"✅ {project_config['domain']}")

                # Stats del proyecto
                with st.sidebar.expander("📊 Estadísticas", expanded=False):
                    stats = pm.get_project_stats(st.session_state.current_project)
                    st.metric("URLs", stats.get("urls_count", 0))
                    st.metric("Embeddings", stats.get("embeddings_count", 0))
                    st.metric("Entidades", stats.get("entities", 0))
                    st.metric("Tamaño", f"{stats.get('size_mb', 0)} MB")

                # Estado de autenticación (Fase 3)
                if oauth_manager:
                    with st.sidebar.expander("🔐 Credenciales", expanded=False):
                        auth_status = oauth_manager.get_auth_status()

                        # API Keys
                        api_keys = auth_status.get('api_keys', [])
                        if api_keys:
                            st.success(f"✅ API Keys: {', '.join(api_keys)}")
                        else:
                            st.info("ℹ️ No hay API keys")

            except FileNotFoundError as e:
                st.sidebar.error(f"📁 Proyecto no encontrado: {e}")
            except (ValueError, KeyError) as e:
                st.sidebar.error(f"⚠️ Configuración de proyecto inválida: {e}")
            except PermissionError as e:
                st.sidebar.error(f"🔒 Sin permisos para acceder al proyecto: {e}")
            except Exception as e:
                st.sidebar.error(f"❌ Error inesperado al cargar proyecto: {e}")

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
                    except FileExistsError as e:
                        st.error(f"⚠️ Ya existe un proyecto con ese nombre")
                    except PermissionError as e:
                        st.error(f"🔒 Sin permisos para crear proyecto: {e}")
                    except ValueError as e:
                        st.error(f"❌ Datos inválidos: {e}")
                    except Exception as e:
                        st.error(f"❌ Error inesperado al crear proyecto: {e}")


def main():
    """Main application entry point."""
    # Verificar licencia - bloquea si no hay licencia válida o trial
    if not check_license_or_block():
        return  # No continuar si no hay licencia

    bootstrap_api_session_state()

    apply_global_styles()

    # Título y descripción
    st.title("🔗 Internal Linking Optimizer")
    st.markdown(
        "Optimiza tu enlazado interno con análisis semántico, "
        "clustering y recomendaciones basadas en IA."
    )

    # Sidebar - Project Selector
    render_project_selector()

    # Sidebar - Export/Import (Fase 4)
    pm = get_project_manager()
    render_export_import_sidebar(pm)

    # License status
    render_license_status_sidebar()

    # Sidebar - Navegación
    with st.sidebar:
        st.markdown("---")
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
    # Linking Lab es PRO, CSV workflow tiene límite de 100 filas en trial
    if tool == "🏠 Inicio":
        render_home()
    elif tool == "📂 Análisis de Embeddings":
        # CSV disponible en trial pero limitado a 100 filas
        render_csv_workflow()
    elif tool == "🔗 Laboratorio de Enlazado":
        # Requiere licencia (feature: linking)
        if require_feature("linking", "Laboratorio de Enlazado"):
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
