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
from modules.semantic_tools import render_semantic_toolkit_section
from modules.keyword_builder import render_semantic_keyword_builder
from modules.semantic_relations import render_semantic_relations
from modules.content_plan import render_content_plan

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
    page_title="SEO Content Analyzer",
    layout="wide",
    page_icon="🎯",
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

    apply_global_styles()

    # Título y descripción
    st.title("🎯 SEO Content Analyzer")
    st.markdown(
        "Analiza contenido SEO con herramientas semánticas avanzadas: "
        "keywords, FAQs, competidores y más."
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
                "🔑 Configuración API",
                "🧰 Herramientas Semánticas",
                "🧠 Semantic Keyword Builder",
                "🔗 Relaciones Semánticas",
                "📋 Content Plan Generator",
            ],
            key="tool_selector"
        )

        st.markdown("---")
        st.markdown("### ℹ️ Acerca de")
        st.caption("SEO Content Analyzer v1.0.0")
        st.caption("Parte de Embedding Insights Suite")

    # Renderizar herramienta seleccionada
    # Features disponibles en trial: hub, semantic_tools
    # Features PRO: keywords, relations, content_plan
    if tool == "🏠 Inicio":
        render_home()
    elif tool == "🔑 Configuración API":
        render_api_settings()
    elif tool == "🧰 Herramientas Semánticas":
        # Disponible en trial
        render_semantic_toolkit_section()
    elif tool == "🧠 Semantic Keyword Builder":
        # Requiere licencia (feature: keywords)
        if require_feature("keywords", "Semantic Keyword Builder"):
            render_semantic_keyword_builder()
    elif tool == "🔗 Relaciones Semánticas":
        # Requiere licencia (feature: relations)
        if require_feature("relations", "Relaciones Semánticas"):
            render_semantic_relations()
    elif tool == "📋 Content Plan Generator":
        # Requiere licencia (feature: content_plan)
        if require_feature("content_plan", "Content Plan Generator"):
            render_content_plan()


def render_api_settings():
    """Renderiza la página de configuración de APIs."""
    st.header("🔑 Configuración de APIs")
    st.markdown("Configura aquí tus claves de API. Se guardarán en la sesión actual.")

    GEMINI_MODELS = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ]
    OPENAI_MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
    ]

    col1, col2 = st.columns(2)

    # ── OpenAI ──
    with col1:
        st.subheader("OpenAI (GPT)")
        openai_key = st.text_input(
            "API Key",
            value=st.session_state.get("openai_api_key", ""),
            type="password",
            key="settings_openai_key",
            help="https://platform.openai.com/api-keys",
        )
        openai_model = st.selectbox(
            "Modelo",
            options=OPENAI_MODELS,
            index=OPENAI_MODELS.index(st.session_state.get("openai_model", "gpt-4o-mini"))
            if st.session_state.get("openai_model", "gpt-4o-mini") in OPENAI_MODELS
            else 1,
            key="settings_openai_model",
        )
        if openai_key:
            st.session_state["openai_api_key"] = openai_key
            st.session_state["openai_model"] = openai_model
            st.success("OpenAI configurado")
        else:
            st.info("Introduce tu API key de OpenAI")

    # ── Gemini ──
    with col2:
        st.subheader("Google Gemini")
        gemini_key = st.text_input(
            "API Key",
            value=st.session_state.get("gemini_api_key", ""),
            type="password",
            key="settings_gemini_key",
            help="https://aistudio.google.com/app/apikey",
        )
        gemini_model = st.selectbox(
            "Modelo",
            options=GEMINI_MODELS,
            index=GEMINI_MODELS.index(st.session_state.get("gemini_model_name", "gemini-2.5-flash"))
            if st.session_state.get("gemini_model_name", "gemini-2.5-flash") in GEMINI_MODELS
            else 0,
            key="settings_gemini_model",
        )
        if gemini_key:
            st.session_state["gemini_api_key"] = gemini_key
            st.session_state["gemini_model_name"] = gemini_model
            st.success("Gemini configurado")
        else:
            st.info("Introduce tu API key de Gemini")

    # ── Summary ──
    st.markdown("---")
    st.subheader("Estado actual")
    c1, c2 = st.columns(2)
    c1.metric("OpenAI", "Configurado" if st.session_state.get("openai_api_key") else "No configurado")
    c2.metric("Gemini", "Configurado" if st.session_state.get("gemini_api_key") else "No configurado")


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
