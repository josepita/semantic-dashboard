"""
Fan-Out Query Analyzer
======================

Herramienta para extraer fan-out queries de IAs (Gemini, ChatGPT) y
analizar la cobertura de dominio usando similitud semántica con embeddings.

Funcionalidades:
- Extracción de fan-out queries via Gemini con Google Search grounding
- Importación de fan-out queries desde ChatGPT (CSV del bookmarklet)
- Análisis de cobertura de dominio con embeddings
- Visualización y exportación de resultados

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
app_sections_path = (project_root / "app_sections").resolve()

# Añadir paths ANTES de cualquier import
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(shared_path) not in sys.path:
    sys.path.insert(0, str(shared_path))
if str(modules_path) not in sys.path:
    sys.path.insert(0, str(modules_path))
if str(app_sections_path) not in sys.path:
    sys.path.insert(0, str(app_sections_path))

# Importar módulos locales
from modules.fanout_extraction import extract_fanout_queries, fanout_results_to_dataframe
from modules.chatgpt_import import parse_chatgpt_fanout_csv
from modules.domain_coverage import (
    analyze_domain_coverage,
    DEFAULT_THRESHOLDS,
    CLASSIFICATION_LABELS,
)
from modules.serp_extraction import (
    batch_extract_serp,
    serp_results_to_dataframe,
    merge_serp_with_fanout,
    get_serp_summary,
)

# License management - TEMPORAL: licencias desactivadas
# TODO: Restaurar verificación de licencias cuando esté listo
def check_license_or_block():
    print(">>> LICENCIAS DESACTIVADAS - check_license_or_block")
    return True
def render_license_status_sidebar():
    print(">>> LICENCIAS DESACTIVADAS - render_license_status_sidebar")
    pass  # No mostrar nada
def require_feature(f, n=""):
    print(f">>> LICENCIAS DESACTIVADAS - require_feature({f})")
    return True

# Import con manejo de errores
try:
    from project_manager import get_project_manager
    from oauth_manager import get_oauth_manager
    from project_ui import render_export_import_sidebar
except ImportError:
    import importlib.util
    pm_path = shared_path / "project_manager.py"
    oauth_path = shared_path / "oauth_manager.py"
    project_ui_path = shared_path / "project_ui.py"

    if not pm_path.exists():
        raise ImportError(f"No se encuentra project_manager.py en {shared_path}")

    spec = importlib.util.spec_from_file_location("project_manager", str(pm_path))
    project_manager = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(project_manager)
    get_project_manager = project_manager.get_project_manager

    if oauth_path.exists():
        spec = importlib.util.spec_from_file_location("oauth_manager", str(oauth_path))
        oauth_manager_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(oauth_manager_module)
        get_oauth_manager = oauth_manager_module.get_oauth_manager
    else:
        get_oauth_manager = lambda x: None

    if project_ui_path.exists():
        spec = importlib.util.spec_from_file_location("project_ui", str(project_ui_path))
        project_ui_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(project_ui_module)
        render_export_import_sidebar = project_ui_module.render_export_import_sidebar
    else:
        render_export_import_sidebar = lambda x: None

st.set_page_config(
    page_title="Fan-Out Query Analyzer",
    layout="wide",
    page_icon="🔍",
)
# Marcar que page_config ya fue configurado (para license_ui)
st.session_state["_page_config_set"] = True


def apply_global_styles():
    """Aplicar estilos globales."""
    st.markdown("""
    <style>
    .main { padding: 1rem; }
    .stButton button { width: 100%; }
    </style>
    """, unsafe_allow_html=True)


def render_project_selector():
    """Renderiza el selector de proyectos en el sidebar."""
    st.sidebar.header("📁 Proyecto Actual")

    pm = get_project_manager()

    if "current_project" not in st.session_state:
        last_project = pm.get_last_project()
        st.session_state.current_project = last_project

    projects = pm.list_projects()
    project_names = [p["name"] for p in projects]

    if not projects:
        st.sidebar.warning("⚠️ No hay proyectos creados")
        st.session_state.current_project = None
    else:
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
            key="fanout_project_selector"
        )

        if selected_project != st.session_state.current_project:
            st.session_state.current_project = selected_project
            pm.set_last_project(selected_project)
            st.rerun()

        if st.session_state.current_project:
            try:
                project_config = pm.load_project(st.session_state.current_project)
                st.session_state.project_config = project_config
                st.sidebar.success(f"✅ {project_config['domain']}")
            except Exception as e:
                st.sidebar.error(f"❌ Error: {e}")


def main():
    """Main application entry point."""
    # Verificar licencia - bloquea si no hay licencia válida o trial
    if not check_license_or_block():
        return  # No continuar si no hay licencia

    apply_global_styles()

    st.title("🔍 Fan-Out Query Analyzer")
    st.markdown(
        "Extrae fan-out queries de Gemini y ChatGPT, y analiza la cobertura "
        "de tu dominio usando similitud semántica con embeddings."
    )

    render_project_selector()
    pm = get_project_manager()
    render_export_import_sidebar(pm)

    # License status
    render_license_status_sidebar()

    with st.sidebar:
        st.markdown("---")
        st.header("🧭 Navegación")

        tool = st.radio(
            "Selecciona una sección:",
            options=[
                "🏠 Inicio",
                "🔑 Configuración API",
                "🌐 Extracción Gemini",
                "💬 Importar ChatGPT",
                "🔎 Extracción SERP",
                "📊 Análisis Cobertura",
            ],
            key="fanout_tool_selector"
        )

        st.markdown("---")
        st.markdown("### ℹ️ Acerca de")
        st.caption("Fan-Out Query Analyzer v1.0.0")
        st.caption("Parte de Embedding Insights Suite")

    # Fan-Out es una feature PRO - bloquear funcionalidades principales
    if tool == "🏠 Inicio":
        render_home()
    elif tool == "🔑 Configuración API":
        render_api_settings()
    elif tool == "🌐 Extracción Gemini":
        # Requiere licencia PRO (feature: fanout)
        if require_feature("fanout", "Extracción Fan-Out Gemini"):
            render_gemini_extraction()
    elif tool == "💬 Importar ChatGPT":
        # Requiere licencia PRO (feature: fanout)
        if require_feature("fanout", "Importar Fan-Out ChatGPT"):
            render_chatgpt_import()
    elif tool == "🔎 Extracción SERP":
        # Requiere licencia PRO (feature: fanout)
        if require_feature("fanout", "Extracción SERP"):
            render_serp_extraction()
    elif tool == "📊 Análisis Cobertura":
        # Requiere licencia PRO (feature: fanout)
        if require_feature("fanout", "Análisis de Cobertura"):
            render_coverage_analysis()


def render_api_settings():
    """Renderiza la página de configuración de APIs."""
    st.header("🔑 Configuración de APIs")
    st.markdown("""
    Configura aquí tu clave API de Gemini. Se usará para extraer fan-out queries
    con Google Search grounding.

    **Nota importante:** Esta funcionalidad requiere el SDK `google-genai` (no `google-generativeai`).
    Si aún no lo tienes instalado, ejecuta: `pip install google-genai`
    """)

    GEMINI_MODELS = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Google Gemini")
        gemini_key = st.text_input(
            "API Key",
            value=st.session_state.get("gemini_api_key", ""),
            type="password",
            key="fanout_gemini_key",
            help="https://aistudio.google.com/app/apikey",
        )
        gemini_model = st.selectbox(
            "Modelo",
            options=GEMINI_MODELS,
            index=GEMINI_MODELS.index(st.session_state.get("gemini_model_name", "gemini-2.5-flash"))
            if st.session_state.get("gemini_model_name", "gemini-2.5-flash") in GEMINI_MODELS
            else 0,
            key="fanout_gemini_model",
        )
        if gemini_key:
            st.session_state["gemini_api_key"] = gemini_key
            st.session_state["gemini_model_name"] = gemini_model
            st.success("Gemini configurado")
        else:
            st.info("Introduce tu API key de Gemini")

    with col2:
        st.subheader("Estado de Dependencias")
        try:
            from google import genai
            st.success("✅ google-genai instalado")
        except ImportError:
            st.error("❌ google-genai NO instalado")
            st.code("pip install google-genai", language="bash")

        try:
            from sentence_transformers import SentenceTransformer
            st.success("✅ sentence-transformers instalado")
        except ImportError:
            st.error("❌ sentence-transformers NO instalado")

    st.markdown("---")
    st.subheader("Estado actual")
    c1, c2 = st.columns(2)
    c1.metric("Gemini", "Configurado" if st.session_state.get("gemini_api_key") else "No configurado")
    c2.metric("Modelo", st.session_state.get("gemini_model_name", "No seleccionado"))


def render_home():
    """Renderiza la página de inicio."""
    st.header("👋 Bienvenido a Fan-Out Query Analyzer")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 ¿Qué son las Fan-Out Queries?")
        st.markdown("""
        Cuando las IAs como Gemini o ChatGPT responden a preguntas complejas,
        internamente generan **múltiples consultas de búsqueda** para obtener información.

        Estas "fan-out queries" revelan:
        - **Qué busca realmente la IA** para responder
        - **Subtemas** que considera relevantes
        - **Gaps de contenido** en tu dominio

        **Ejemplo:** Si preguntas "¿Cuál es el mejor CRM para pequeñas empresas?",
        la IA puede generar internamente:
        - "mejores CRM pequeñas empresas 2026"
        - "CRM precio bajo características"
        - "Salesforce vs HubSpot comparación"
        - "CRM fácil de usar reseñas"
        """)

    with col2:
        st.markdown("### 🚀 Flujo de Trabajo")
        st.markdown("""
        **1. Extracción de Fan-Out Queries**
        - **Gemini**: Envía prompts → extrae webSearchQueries automáticamente
        - **ChatGPT**: Usa el bookmarklet → importa el CSV exportado

        **2. Preparar tu Sitio**
        - Sube un CSV con tus URLs y embeddings
        - (El mismo formato que usas en las otras apps)

        **3. Análisis de Cobertura**
        - Comparamos cada fan-out query con tus páginas
        - Clasificamos: Perfect Coverage → No Coverage
        - Identificas gaps de contenido

        **4. Exportar Resultados**
        - Excel con detalle y resumen
        - Prioriza qué contenido crear
        """)

    st.markdown("---")
    st.markdown("### 📊 Clasificaciones de Cobertura")

    cols = st.columns(5)
    classifications = [
        ("🟢", "Perfect Coverage", "≥80%", "Tu contenido responde perfectamente"),
        ("🔵", "Aligned", "65-80%", "Contenido relacionado, puede mejorarse"),
        ("🟡", "Related Gap", "45-65%", "Tema tangencial, oportunidad"),
        ("🟠", "Clear Gap", "25-45%", "Gap claro, crear contenido"),
        ("🔴", "No Coverage", "<25%", "Sin cobertura, evaluar relevancia"),
    ]
    for col, (icon, label, threshold, desc) in zip(cols, classifications):
        with col:
            st.markdown(f"**{icon} {label}**")
            st.caption(f"Similitud: {threshold}")
            st.caption(desc)


def render_gemini_extraction():
    """Renderiza la sección de extracción de fan-out queries via Gemini."""
    st.header("🌐 Extracción de Fan-Out Queries (Gemini)")

    api_key = st.session_state.get("gemini_api_key", "")
    model_id = st.session_state.get("gemini_model_name", "gemini-2.5-flash")

    if not api_key:
        st.warning("⚠️ Configura tu API key de Gemini en la sección de Configuración API")
        return

    st.markdown(f"**Modelo seleccionado:** `{model_id}`")

    tab1, tab2, tab3 = st.tabs(["📝 Introducir Prompts", "📁 Subir Prompts", "📥 Importar Extracción Previa"])

    with tab1:
        prompts_text = st.text_area(
            "Introduce los prompts (uno por línea):",
            height=200,
            placeholder="¿Cuál es el mejor CRM para pequeñas empresas?\n¿Cómo elegir un software de contabilidad?\n...",
            key="fanout_prompts_text",
        )
        prompts_from_text = [p.strip() for p in prompts_text.split("\n") if p.strip()]

    with tab2:
        uploaded_prompts = st.file_uploader(
            "Sube un archivo .txt o .csv con prompts",
            type=["txt", "csv"],
            key="fanout_prompts_file",
        )
        prompts_from_file = []
        if uploaded_prompts:
            content = uploaded_prompts.read().decode("utf-8", errors="ignore")
            prompts_from_file = [p.strip() for p in content.split("\n") if p.strip()]
            st.info(f"Archivo cargado: {len(prompts_from_file)} prompts")

    with tab3:
        st.markdown("""
        **Importa un CSV con fan-out queries ya extraídas previamente.**

        El archivo debe tener al menos las columnas `prompt` y `web_search_query`.
        Opcionalmente puede incluir `query_index`, `source` y `error`.
        """)

        uploaded_prev = st.file_uploader(
            "CSV con extracción previa",
            type=["csv", "xlsx", "xls"],
            key="fanout_import_prev",
        )

        if uploaded_prev:
            try:
                filename = uploaded_prev.name.lower()
                if filename.endswith(".csv"):
                    import_df = pd.read_csv(uploaded_prev)
                else:
                    import_df = pd.read_excel(uploaded_prev)

                st.success(f"✅ Cargado: {len(import_df)} filas")

                with st.expander("📋 Preview", expanded=False):
                    st.dataframe(import_df.head(10), use_container_width=True)

                cols = import_df.columns.tolist()
                cols_lower = {c.lower().strip(): c for c in cols}

                # Auto-detectar columnas
                prompt_col = None
                query_col = None
                for cand in ["prompt", "user query", "user_query"]:
                    if cand in cols_lower:
                        prompt_col = cols_lower[cand]
                        break
                for cand in ["web_search_query", "search queries", "query", "queries"]:
                    if cand in cols_lower:
                        query_col = cols_lower[cand]
                        break

                st.markdown("**Mapeo de columnas:**")
                c1, c2 = st.columns(2)
                with c1:
                    prompt_col = st.selectbox(
                        "Columna de Prompt",
                        options=cols,
                        index=cols.index(prompt_col) if prompt_col and prompt_col in cols else 0,
                        key="fanout_import_prompt_col",
                    )
                with c2:
                    query_col = st.selectbox(
                        "Columna de Query",
                        options=cols,
                        index=cols.index(query_col) if query_col and query_col in cols else (1 if len(cols) > 1 else 0),
                        key="fanout_import_query_col",
                    )

                if st.button("➕ Importar a sesión", key="fanout_import_prev_btn"):
                    # Normalizar a formato estándar
                    rows = []
                    for _, row in import_df.iterrows():
                        p = str(row.get(prompt_col, "")).strip()
                        q = str(row.get(query_col, "")).strip()
                        if p and q and p != "nan" and q != "nan":
                            rows.append({
                                "prompt": p,
                                "web_search_query": q,
                                "source": "gemini_import",
                                "error": "",
                            })

                    if rows:
                        new_df = pd.DataFrame(rows)
                        new_df["query_index"] = new_df.groupby("prompt").cumcount()

                        if "fanout_queries_df" in st.session_state and st.session_state["fanout_queries_df"] is not None:
                            existing = st.session_state["fanout_queries_df"]
                            new_df = pd.concat([existing, new_df], ignore_index=True)

                        st.session_state["fanout_queries_df"] = new_df
                        st.success(f"✅ Importadas {len(rows)} queries")
                        st.rerun()
                    else:
                        st.warning("No se encontraron datos válidos")

            except Exception as e:
                st.error(f"Error al importar: {e}")

    prompts = prompts_from_text or prompts_from_file

    col1, col2 = st.columns([3, 1])
    with col1:
        delay = st.slider(
            "Delay entre requests (segundos)",
            min_value=0.1,
            max_value=3.0,
            value=0.5,
            step=0.1,
            help="Aumenta si recibes errores de rate limiting",
        )
    with col2:
        st.metric("Prompts a procesar", len(prompts))

    if st.button("🚀 Extraer Fan-Out Queries", disabled=len(prompts) == 0, key="fanout_extract_btn"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(current, total, prompt):
            progress_bar.progress((current + 1) / total)
            status_text.text(f"Procesando {current + 1}/{total}: {prompt[:50]}...")

        with st.spinner("Extrayendo fan-out queries..."):
            results = extract_fanout_queries(
                prompts=prompts,
                api_key=api_key,
                model_id=model_id,
                delay=delay,
                progress_callback=progress_callback,
            )

        progress_bar.empty()
        status_text.empty()

        df = fanout_results_to_dataframe(results)

        if "fanout_queries_df" in st.session_state and st.session_state["fanout_queries_df"] is not None:
            existing = st.session_state["fanout_queries_df"]
            df = pd.concat([existing, df], ignore_index=True)

        st.session_state["fanout_queries_df"] = df

        errors = df[df["error"] != ""]
        successes = df[df["error"] == ""]

        col1, col2, col3 = st.columns(3)
        col1.metric("Queries extraídas", len(successes))
        col2.metric("Prompts con error", len(errors["prompt"].unique()))
        col3.metric("Total en sesión", len(df))

        st.success("Extracción completada")

    if "fanout_queries_df" in st.session_state and st.session_state["fanout_queries_df"] is not None:
        df = st.session_state["fanout_queries_df"]
        st.markdown("---")
        st.subheader("📋 Queries Extraídas")

        gemini_df = df[df["source"] == "gemini"]
        if not gemini_df.empty:
            st.dataframe(gemini_df, use_container_width=True)

            csv = gemini_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Descargar CSV (Gemini)",
                csv,
                "fanout_queries_gemini.csv",
                "text/csv",
                key="fanout_download_gemini",
            )


def render_chatgpt_import():
    """Renderiza la sección de importación de fan-out queries desde ChatGPT."""
    st.header("💬 Importar Fan-Out Queries (ChatGPT)")

    with st.expander("📖 Instrucciones para usar el Bookmarklet", expanded=True):
        st.markdown("""
        **Paso 1:** Copia el bookmarklet de ChatGPT Conversations Analyzer

        **Paso 2:** Abre una conversación en ChatGPT (chat.openai.com)

        **Paso 3:** Ejecuta el bookmarklet (click en el marcador)

        **Paso 4:** En la ventana que se abre:
        - Marca la columna "Queries" en el selector de columnas
        - Click en "Export Selected"
        - Se descargará un archivo `Queries_Report.csv`

        **Paso 5:** Sube ese CSV aquí abajo
        """)

    uploaded_file = st.file_uploader(
        "Sube el CSV exportado del bookmarklet",
        type=["csv"],
        key="fanout_chatgpt_file",
    )

    if uploaded_file:
        try:
            df = parse_chatgpt_fanout_csv(uploaded_file)

            st.success(f"✅ Importado: {len(df)} queries de {df['prompt'].nunique()} prompts")

            st.dataframe(df.head(20), use_container_width=True)

            if st.button("➕ Añadir a la sesión", key="fanout_add_chatgpt"):
                if "fanout_queries_df" in st.session_state and st.session_state["fanout_queries_df"] is not None:
                    existing = st.session_state["fanout_queries_df"]
                    df = pd.concat([existing, df], ignore_index=True)

                st.session_state["fanout_queries_df"] = df
                st.success("Queries añadidas a la sesión")
                st.rerun()

        except Exception as e:
            st.error(f"Error al parsear CSV: {e}")

    if "fanout_queries_df" in st.session_state and st.session_state["fanout_queries_df"] is not None:
        df = st.session_state["fanout_queries_df"]
        st.markdown("---")
        st.subheader("📋 Todas las Queries en Sesión")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total queries", len(df))
        col2.metric("De Gemini", len(df[df["source"] == "gemini"]))
        col3.metric("De ChatGPT", len(df[df["source"] == "chatgpt"]))

        if st.button("🗑️ Limpiar todas las queries", key="fanout_clear_all"):
            st.session_state["fanout_queries_df"] = None
            st.rerun()


def render_serp_extraction():
    """Renderiza la sección de extracción de títulos SERP."""
    st.header("🔎 Extracción de Títulos SERP")
    st.markdown(
        "Extrae los títulos de los resultados de búsqueda de Google (SERP) "
        "para cada fan-out query. Útil para analizar qué contenido posiciona "
        "actualmente para cada consulta."
    )

    queries_df = st.session_state.get("fanout_queries_df")

    if queries_df is None or queries_df.empty:
        st.warning("⚠️ Primero extrae o importa fan-out queries en las secciones anteriores")
        return

    # Filtrar queries válidas (sin errores)
    valid_queries = queries_df[queries_df["error"] == ""] if "error" in queries_df.columns else queries_df
    unique_queries = valid_queries["web_search_query"].unique().tolist()

    st.info(f"📝 {len(unique_queries)} queries únicas disponibles para extraer SERP")

    # ══════════════════════════════════════════════════════════
    # MÉTODO DE EXTRACCIÓN
    # ══════════════════════════════════════════════════════════
    st.subheader("🔧 Método de Extracción")

    method = st.radio(
        "Selecciona el método:",
        options=["scraping", "api"],
        format_func=lambda x: {
            "scraping": "🕷️ Scraping directo (gratis, menos fiable)",
            "api": "🔑 Google Custom Search API (100 queries/día gratis, más fiable)",
        }[x],
        horizontal=True,
        key="serp_method",
    )

    # Configuración de API si se selecciona
    api_key = None
    cx = None

    if method == "api":
        with st.expander("🔑 Configurar Google Custom Search API", expanded=True):
            st.markdown("""
            **Para usar la API necesitas:**
            1. Crear una API Key en [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
            2. Habilitar la API "Custom Search API"
            3. Crear un buscador en [Programmable Search Engine](https://programmablesearchengine.google.com/)
            4. Obtener el CX (Search Engine ID) del buscador

            **Límites:** 100 queries/día gratis, después $5 por 1000 queries.
            """)

            col1, col2 = st.columns(2)
            with col1:
                api_key = st.text_input(
                    "API Key",
                    type="password",
                    key="serp_api_key",
                    help="Tu API Key de Google Cloud",
                )
            with col2:
                cx = st.text_input(
                    "Search Engine ID (CX)",
                    key="serp_cx",
                    help="El ID de tu Custom Search Engine (formato: xxx:yyy)",
                )

            if not api_key or not cx:
                st.warning("⚠️ Introduce API Key y CX para usar este método")

    # ══════════════════════════════════════════════════════════
    # CONFIGURACIÓN GENERAL
    # ══════════════════════════════════════════════════════════
    st.subheader("⚙️ Configuración")

    col1, col2, col3 = st.columns(3)

    with col1:
        num_results = st.slider(
            "Resultados por query",
            min_value=5,
            max_value=20,
            value=10,
            help="Número de títulos a extraer por cada query",
            key="serp_num_results",
        )

    with col2:
        lang = st.selectbox(
            "Idioma",
            options=["es", "en", "fr", "de", "it", "pt"],
            index=0,
            key="serp_lang",
        )

    with col3:
        country = st.selectbox(
            "País",
            options=["es", "us", "mx", "ar", "co", "uk", "fr", "de"],
            index=0,
            key="serp_country",
        )

    # Delays solo para scraping
    delay_min = 3.0
    delay_max = 6.0

    if method == "scraping":
        col1, col2 = st.columns(2)

        with col1:
            delay_min = st.slider(
                "Delay mínimo (segundos)",
                min_value=2.0,
                max_value=10.0,
                value=3.0,
                step=0.5,
                help="Tiempo mínimo de espera entre peticiones",
                key="serp_delay_min",
            )

        with col2:
            delay_max = st.slider(
                "Delay máximo (segundos)",
                min_value=3.0,
                max_value=15.0,
                value=6.0,
                step=0.5,
                help="Tiempo máximo de espera entre peticiones",
                key="serp_delay_max",
            )

        st.warning(
            "⚠️ **Scraping:** Google puede bloquear peticiones frecuentes. "
            "Usa delays de 3-6 segundos mínimo. Si hay errores, espera unos minutos."
        )
    else:
        st.success(
            "✅ **API:** Método más fiable. 100 queries/día gratis. "
            "No necesita delays largos."
        )

    # Selección de queries a procesar
    st.markdown("---")
    st.subheader("📝 Queries a procesar")

    process_mode = st.radio(
        "¿Qué queries procesar?",
        options=["todas", "seleccion", "nuevas"],
        format_func=lambda x: {
            "todas": f"Todas las queries ({len(unique_queries)})",
            "seleccion": "Selección manual",
            "nuevas": "Solo queries sin SERP extraída",
        }[x],
        horizontal=True,
        key="serp_process_mode",
    )

    queries_to_process = unique_queries

    if process_mode == "seleccion":
        queries_to_process = st.multiselect(
            "Selecciona queries:",
            options=unique_queries,
            default=unique_queries[:min(10, len(unique_queries))],
            key="serp_selected_queries",
        )
    elif process_mode == "nuevas":
        # Filtrar queries que ya tienen SERP
        existing_serp = st.session_state.get("fanout_serp_df")
        if existing_serp is not None and not existing_serp.empty:
            processed_queries = set(existing_serp["query"].unique())
            queries_to_process = [q for q in unique_queries if q not in processed_queries]
            st.info(f"📝 {len(queries_to_process)} queries nuevas (sin SERP extraída)")
        else:
            st.info("No hay extracciones previas, se procesarán todas las queries")

    st.metric("Queries a procesar", len(queries_to_process))

    # Estimación de tiempo
    if queries_to_process:
        avg_delay = (delay_min + delay_max) / 2
        est_time = len(queries_to_process) * avg_delay
        est_minutes = int(est_time // 60)
        est_seconds = int(est_time % 60)
        st.caption(f"⏱️ Tiempo estimado: ~{est_minutes}m {est_seconds}s")

    # Botón de extracción
    st.markdown("---")

    # Validar que si se usa API, hay credenciales
    can_extract = len(queries_to_process) > 0
    if method == "api" and (not api_key or not cx):
        can_extract = False

    if st.button(
        "🚀 Extraer Títulos SERP",
        disabled=not can_extract,
        type="primary",
        key="serp_extract_btn",
    ):
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.empty()

        def progress_callback(current, total, query):
            progress_bar.progress((current + 1) / total)
            status_text.text(f"Procesando {current + 1}/{total}: {query[:60]}...")

        with st.spinner("Extrayendo títulos SERP..."):
            results = batch_extract_serp(
                queries=queries_to_process,
                num_results=num_results,
                lang=lang,
                country=country,
                delay_min=delay_min,
                delay_max=delay_max,
                method=method,
                api_key=api_key,
                cx=cx,
                progress_callback=progress_callback,
            )

        progress_bar.empty()
        status_text.empty()

        # Convertir a DataFrame
        serp_df = serp_results_to_dataframe(results)

        # Combinar con resultados previos si existen
        existing_serp = st.session_state.get("fanout_serp_df")
        if existing_serp is not None and not existing_serp.empty:
            serp_df = pd.concat([existing_serp, serp_df], ignore_index=True)
            # Eliminar duplicados (misma query + posición)
            serp_df = serp_df.drop_duplicates(subset=["query", "position"], keep="last")

        st.session_state["fanout_serp_df"] = serp_df

        # Mostrar resumen
        summary = get_serp_summary(serp_df)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Queries procesadas", summary["total_queries"])
        col2.metric("Exitosas", summary["successful_queries"])
        col3.metric("Con errores", summary["failed_queries"])
        col4.metric("Total resultados", summary["total_results"])

        st.success("✅ Extracción SERP completada")

        # ── Mostrar resultados inmediatamente ──
        st.markdown("### 📋 Resultados Extraídos")

        # Tabla de resultados (solo exitosos)
        results_df = serp_df[serp_df["error"] == ""]
        st.dataframe(
            results_df[["query", "position", "title", "url"]],
            use_container_width=True,
            hide_index=True,
            height=400,
        )

        # Descargas inmediatas
        st.markdown("##### 📥 Descargar ahora")
        dl_col1, dl_col2, dl_col3 = st.columns(3)

        with dl_col1:
            csv_serp = serp_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 CSV (Solo SERP)",
                csv_serp,
                "serp_titles.csv",
                "text/csv",
                key="serp_download_immediate_csv",
            )

        with dl_col2:
            if queries_df is not None:
                merged_df = merge_serp_with_fanout(queries_df, serp_df)
                csv_merged = merged_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 CSV (Fan-Out + SERP)",
                    csv_merged,
                    "fanout_with_serp.csv",
                    "text/csv",
                    key="serp_download_immediate_merged",
                )

        with dl_col3:
            from io import BytesIO
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                results_df.to_excel(writer, sheet_name="SERP Títulos", index=False)
                if queries_df is not None:
                    merged_df = merge_serp_with_fanout(queries_df, serp_df)
                    merged_df.to_excel(writer, sheet_name="Fan-Out + SERP", index=False)
            st.download_button(
                "📥 Excel Completo",
                buffer.getvalue(),
                "serp_extraction.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="serp_download_immediate_excel",
            )

    # Mostrar resultados existentes
    serp_df = st.session_state.get("fanout_serp_df")

    if serp_df is not None and not serp_df.empty:
        st.markdown("---")
        st.subheader("📋 Resultados SERP Extraídos")

        # Resumen
        summary = get_serp_summary(serp_df)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total queries", summary["total_queries"])
        col2.metric("Exitosas", summary["successful_queries"])
        col3.metric("Resultados totales", summary["total_results"])
        col4.metric("Promedio/query", summary["avg_results_per_query"])

        # Filtrar por query
        st.markdown("##### Filtrar resultados")
        filter_query = st.selectbox(
            "Filtrar por query:",
            options=["(Todas)"] + sorted(serp_df["query"].unique().tolist()),
            key="serp_filter_query",
        )

        display_df = serp_df if filter_query == "(Todas)" else serp_df[serp_df["query"] == filter_query]
        display_df = display_df[display_df["error"] == ""]

        st.dataframe(
            display_df[["query", "position", "title", "url"]],
            use_container_width=True,
            hide_index=True,
        )

        # Errores
        errors_df = serp_df[serp_df["error"] != ""]
        if not errors_df.empty:
            with st.expander(f"⚠️ {len(errors_df)} errores"):
                st.dataframe(errors_df[["query", "error"]], use_container_width=True, hide_index=True)

        # Exportación
        st.markdown("---")
        st.subheader("📥 Exportar")

        col1, col2, col3 = st.columns(3)

        with col1:
            # CSV de SERP
            csv_serp = serp_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Descargar SERP (CSV)",
                csv_serp,
                "serp_titles.csv",
                "text/csv",
                key="serp_download_csv",
            )

        with col2:
            # Merge con fan-out queries
            if queries_df is not None:
                merged_df = merge_serp_with_fanout(queries_df, serp_df)
                csv_merged = merged_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Fan-Out + SERP (CSV)",
                    csv_merged,
                    "fanout_with_serp.csv",
                    "text/csv",
                    key="serp_download_merged",
                )

        with col3:
            # Excel completo
            from io import BytesIO
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                serp_df[serp_df["error"] == ""].to_excel(writer, sheet_name="SERP Títulos", index=False)
                if queries_df is not None:
                    merged_df = merge_serp_with_fanout(queries_df, serp_df)
                    merged_df.to_excel(writer, sheet_name="Fan-Out + SERP", index=False)
                errors_df.to_excel(writer, sheet_name="Errores", index=False)

            st.download_button(
                "📥 Excel Completo",
                buffer.getvalue(),
                "serp_extraction_report.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="serp_download_excel",
            )

        # Limpiar resultados
        if st.button("🗑️ Limpiar resultados SERP", key="serp_clear"):
            st.session_state["fanout_serp_df"] = None
            st.rerun()


def render_coverage_analysis():
    """Renderiza la sección de análisis de cobertura de dominio."""
    st.header("📊 Análisis de Cobertura de Dominio")

    queries_df = st.session_state.get("fanout_queries_df")

    if queries_df is None or queries_df.empty:
        st.warning("⚠️ Primero extrae o importa fan-out queries en las secciones anteriores")
        return

    valid_queries = queries_df[queries_df["error"] == ""] if "error" in queries_df.columns else queries_df
    st.info(f"📝 {len(valid_queries)} queries disponibles para analizar")

    st.subheader("1. Sube el archivo de tu sitio")
    st.markdown("Sube un CSV o Excel con las URLs de tu sitio y sus embeddings.")

    uploaded_site = st.file_uploader(
        "Archivo con URLs y Embeddings",
        type=["csv", "xlsx", "xls"],
        key="fanout_site_file",
    )

    if uploaded_site:
        import pandas as pd
        import numpy as np
        import ast

        try:
            # Leer según extensión
            filename = uploaded_site.name.lower()
            if filename.endswith(".csv"):
                site_df = pd.read_csv(uploaded_site)
            elif filename.endswith((".xlsx", ".xls")):
                site_df = pd.read_excel(uploaded_site)
            else:
                st.error("Formato no soportado. Usa CSV o Excel.")
                return

            st.success(f"✅ Cargado: {len(site_df)} filas, {len(site_df.columns)} columnas")

            # Mostrar preview
            with st.expander("📋 Preview del archivo", expanded=False):
                st.dataframe(site_df.head(5), use_container_width=True)

            cols = site_df.columns.tolist()

            # Auto-detectar columnas candidatas
            url_candidates = [c for c in cols if any(x in c.lower() for x in ["url", "address", "link", "page", "direccion"])]
            emb_candidates = [c for c in cols if any(x in c.lower() for x in ["embed", "vector", "emb"])]
            title_candidates = [c for c in cols if any(x in c.lower() for x in ["title", "titulo", "h1", "nombre"])]

            st.markdown("**Selecciona las columnas:**")
            col1, col2, col3 = st.columns(3)

            with col1:
                url_col = st.selectbox(
                    "Columna de URLs *",
                    options=cols,
                    index=cols.index(url_candidates[0]) if url_candidates else 0,
                    key="fanout_url_col",
                    help="Columna que contiene las URLs de las páginas",
                )

            with col2:
                emb_col = st.selectbox(
                    "Columna de Embeddings *",
                    options=cols,
                    index=cols.index(emb_candidates[0]) if emb_candidates else 0,
                    key="fanout_emb_col",
                    help="Columna que contiene los vectores de embeddings",
                )

            with col3:
                title_options = ["(No usar)"] + cols
                title_default = 0
                if title_candidates:
                    try:
                        title_default = title_options.index(title_candidates[0])
                    except ValueError:
                        title_default = 0
                title_col = st.selectbox(
                    "Columna de Título (opcional)",
                    options=title_options,
                    index=title_default,
                    key="fanout_title_col",
                    help="Columna con el título de la página (para contexto en resultados)",
                )

            st.session_state["fanout_url_column"] = url_col
            if title_col != "(No usar)":
                st.session_state["fanout_title_column"] = title_col
            else:
                st.session_state["fanout_title_column"] = None

            # Preprocesar embeddings si son strings
            if site_df[emb_col].dtype == object:
                def parse_emb(x):
                    if isinstance(x, str):
                        try:
                            return np.array(ast.literal_eval(x))
                        except:
                            return None
                    return x
                site_df["EmbeddingsFloat"] = site_df[emb_col].apply(parse_emb)
                site_df = site_df.dropna(subset=["EmbeddingsFloat"])
            else:
                site_df["EmbeddingsFloat"] = site_df[emb_col]

            st.session_state["fanout_site_df"] = site_df
            st.info(f"URLs con embeddings válidos: {len(site_df)}")

        except Exception as e:
            st.error(f"Error al cargar archivo: {e}")
            import traceback
            st.code(traceback.format_exc())
            return

    site_df = st.session_state.get("fanout_site_df")
    url_col = st.session_state.get("fanout_url_column")

    if site_df is None:
        return

    st.markdown("---")
    st.subheader("2. Configurar Umbrales")

    with st.expander("⚙️ Ajustar umbrales de clasificación", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            t_perfect = st.slider("Perfect Coverage", 0.5, 1.0, DEFAULT_THRESHOLDS["perfect_coverage"], 0.05, key="fanout_t_perfect")
        with col2:
            t_aligned = st.slider("Aligned", 0.4, 0.9, DEFAULT_THRESHOLDS["aligned"], 0.05, key="fanout_t_aligned")
        with col3:
            t_related = st.slider("Related Gap", 0.2, 0.7, DEFAULT_THRESHOLDS["related_gap"], 0.05, key="fanout_t_related")
        with col4:
            t_clear = st.slider("Clear Gap", 0.1, 0.5, DEFAULT_THRESHOLDS["clear_gap"], 0.05, key="fanout_t_clear")

    thresholds = {
        "perfect_coverage": st.session_state.get("fanout_t_perfect", DEFAULT_THRESHOLDS["perfect_coverage"]),
        "aligned": st.session_state.get("fanout_t_aligned", DEFAULT_THRESHOLDS["aligned"]),
        "related_gap": st.session_state.get("fanout_t_related", DEFAULT_THRESHOLDS["related_gap"]),
        "clear_gap": st.session_state.get("fanout_t_clear", DEFAULT_THRESHOLDS["clear_gap"]),
    }

    st.markdown("---")
    st.subheader("3. Ejecutar Análisis")

    if st.button("🔍 Analizar Cobertura", key="fanout_analyze_btn"):
        with st.spinner("Analizando cobertura..."):
            try:
                detail_df, summary_df = analyze_domain_coverage(
                    queries_df=valid_queries,
                    site_df=site_df,
                    url_column=url_col,
                    embedding_col="EmbeddingsFloat",
                    thresholds=thresholds,
                )

                st.session_state["fanout_coverage_detail"] = detail_df
                st.session_state["fanout_coverage_summary"] = summary_df

                st.success("Análisis completado")

            except Exception as e:
                st.error(f"Error en análisis: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    detail_df = st.session_state.get("fanout_coverage_detail")
    summary_df = st.session_state.get("fanout_coverage_summary")

    if detail_df is not None and summary_df is not None:
        st.markdown("---")
        st.subheader("📈 Resultados")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("**Resumen de Cobertura**")
            for _, row in summary_df.iterrows():
                icon_map = {
                    "perfect_coverage": "🟢",
                    "aligned": "🔵",
                    "related_gap": "🟡",
                    "clear_gap": "🟠",
                    "no_coverage": "🔴",
                }
                icon = icon_map.get(row["classification"], "⚪")
                st.metric(
                    f"{icon} {row['label']}",
                    f"{row['count']} ({row['percentage']}%)",
                )

        with col2:
            import plotly.express as px

            fig = px.pie(
                summary_df,
                values="count",
                names="label",
                title="Distribución de Cobertura",
                color="classification",
                color_discrete_map={
                    "perfect_coverage": "#22c55e",
                    "aligned": "#3b82f6",
                    "related_gap": "#eab308",
                    "clear_gap": "#f97316",
                    "no_coverage": "#ef4444",
                },
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("📋 Detalle por Query")

        filter_class = st.multiselect(
            "Filtrar por clasificación:",
            options=list(CLASSIFICATION_LABELS.values()),
            default=list(CLASSIFICATION_LABELS.values()),
            key="fanout_filter_class",
        )

        filtered = detail_df[detail_df["classification_label"].isin(filter_class)]
        st.dataframe(
            filtered[["prompt", "web_search_query", "best_url", "similarity", "classification_label"]],
            use_container_width=True,
        )

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            csv_detail = detail_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Descargar Detalle (CSV)",
                csv_detail,
                "fanout_coverage_detail.csv",
                "text/csv",
            )

        with col2:
            from io import BytesIO
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                summary_df.to_excel(writer, sheet_name="Resumen", index=False)
                detail_df.to_excel(writer, sheet_name="Detalle", index=False)
            st.download_button(
                "📥 Descargar Excel Completo",
                buffer.getvalue(),
                "fanout_coverage_report.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


# Importar pandas para usar en las funciones
import pandas as pd


if __name__ == "__main__":
    main()
