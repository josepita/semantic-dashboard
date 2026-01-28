"""
Módulo común para funcionalidades compartidas entre las apps.

Este módulo contiene funciones y configuraciones comunes que se usan
en las 3 aplicaciones (content-analyzer, gsc-insights, linking-optimizer).
"""

import streamlit as st
from pathlib import Path
from typing import Optional


def setup_page_config(title: str, icon: str, layout: str = "wide") -> None:
    """
    Configura la página de Streamlit con parámetros estándar.
    
    Args:
        title: Título de la página
        icon: Emoji o icono para la página
        layout: Layout de la página (default: "wide")
    """
    st.set_page_config(
        page_title=title,
        layout=layout,
        page_icon=icon,
    )


def apply_global_styles() -> None:
    """
    Aplica estilos CSS globales consistentes para todas las apps.
    """
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


def render_project_selector() -> None:
    """
    Renderiza el selector de proyectos en el sidebar.
    
    Esta función maneja:
    - Listado de proyectos disponibles
    - Selector de proyecto actual
    - Carga de configuración del proyecto
    - Carga automática de credenciales OAuth
    - Estadísticas del proyecto
    - Creación de nuevos proyectos
    """
    from project_manager import get_project_manager
    from oauth_manager import get_oauth_manager
    
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

                    anthropic_key = oauth_manager.load_api_key('anthropic', 'ANTHROPIC_API_KEY')
                    if anthropic_key:
                        st.session_state.anthropic_api_key = anthropic_key

                # Mostrar info del proyecto
                st.sidebar.success(f"✅ {project_config['domain']}")

                # Stats del proyecto
                with st.sidebar.expander("📊 Estadísticas", expanded=False):
                    stats = pm.get_project_stats(st.session_state.current_project)
                    st.metric("URLs", stats.get("urls_count", 0))
                    st.metric("Embeddings", stats.get("embeddings_count", 0))
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


def render_app_sidebar(app_name: str, tools: list) -> str:
    """
    Renderiza el sidebar común con navegación y información.
    
    Args:
        app_name: Nombre de la aplicación
        tools: Lista de herramientas disponibles
        
    Returns:
        Herramienta seleccionada
    """
    from project_manager import get_project_manager
    from project_ui import render_export_import_sidebar
    
    # Project Selector
    render_project_selector()
    
    # Export/Import (Fase 4)
    pm = get_project_manager()
    render_export_import_sidebar(pm)
    
    # Navegación
    with st.sidebar:
        st.markdown("---")
        st.header("🧭 Navegación")

        tool = st.radio(
            "Selecciona una herramienta:",
            options=tools,
            key="tool_selector"
        )

        st.markdown("---")
        st.markdown("### ℹ️ Acerca de")
        st.caption(f"{app_name} v1.0.0")
        st.caption("Parte de Embedding Insights Suite")
    
    return tool


def safe_import_from_shared(module_name: str, function_name: str):
    """
    Importa una función desde /shared con manejo de errores y fallback.
    
    Args:
        module_name: Nombre del módulo (ej: "project_manager")
        function_name: Nombre de la función a importar
        
    Returns:
        La función importada o un fallback
    """
    import importlib.util
    import sys
    from pathlib import Path
    
    # Intentar import directo primero
    try:
        module = __import__(module_name)
        return getattr(module, function_name)
    except ImportError:
        pass
    
    # Fallback: cargar desde shared/
    current_dir = Path(__file__).parent.resolve()
    shared_path = (current_dir.parent / "shared").resolve()
    module_path = shared_path / f"{module_name}.py"
    
    if not module_path.exists():
        # Retornar función dummy si no existe
        if function_name == "render_export_import_sidebar":
            return lambda x: None
        elif function_name == "get_oauth_manager":
            return lambda x: None
        else:
            raise ImportError(f"No se encuentra {module_name}.py en {shared_path}")
    
    # Cargar módulo dinámicamente
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    loaded_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded_module)
    
    return getattr(loaded_module, function_name)


def get_app_version() -> str:
    """Retorna la versión de la aplicación."""
    return "1.0.0"


def get_suite_name() -> str:
    """Retorna el nombre de la suite."""
    return "Embedding Insights Suite"
