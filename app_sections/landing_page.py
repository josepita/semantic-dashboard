from __future__ import annotations

import os
import streamlit as st


def get_gemini_api_key_from_context() -> str:
    candidates = [
        st.session_state.get("gemini_api_key"),
        os.environ.get("GEMINI_API_KEY"),
        os.environ.get("GOOGLE_API_KEY"),
        os.environ.get("GOOGLE_GENAI_KEY"),
    ]
    for candidate in candidates:
        if candidate:
            return candidate.strip()
    return ""


def get_gemini_model_from_context(default: str = "gemini-2.5-flash") -> str:
    candidate = (
        st.session_state.get("gemini_model_name")
        or os.environ.get("GEMINI_MODEL")
        or default
    )
    return candidate.strip()


def render_api_settings_panel() -> None:
    with st.sidebar:
        st.markdown("### ⚙️ Configuración de Gemini")
        st.caption(
            "Guarda tu API key una vez y la reutilizaremos en el laboratorio, el builder semántico "
            "y el informe de posiciones. También puedes definir `GOOGLE_API_KEY` o `GEMINI_API_KEY` "
            "como variable de entorno o en `.streamlit/secrets.toml`."
        )
        if "sidebar_gemini_api_value" not in st.session_state:
            st.session_state["sidebar_gemini_api_value"] = get_gemini_api_key_from_context()
        if "sidebar_gemini_model_value" not in st.session_state:
            st.session_state["sidebar_gemini_model_value"] = get_gemini_model_from_context()

        sidebar_key = st.text_input(
            "Gemini API Key",
            type="password",
            key="sidebar_gemini_api_value",
            help="Introduce la clave de https://aistudio.google.com/app/apikey",
        )
        sidebar_model = st.text_input(
            "Modelo Gemini preferido",
            key="sidebar_gemini_model_value",
            help="Ejemplo: gemini-2.5-flash o gemini-1.5-pro",
        )
        if st.button("Guardar clave en esta sesión", key="sidebar_save_gemini"):
            cleaned_key = (sidebar_key or "").strip()
            cleaned_model = (sidebar_model or "").strip() or "gemini-2.5-flash"
            if cleaned_key:
                st.session_state["gemini_api_key"] = cleaned_key
                st.session_state["gemini_model_name"] = cleaned_model
                st.success("API key almacenada en la sesión actual.")
            else:
                st.warning("Introduce una API key válida antes de guardar.")


def render_sidebar_navigation() -> None:
    """
    Renderiza un menú de navegación permanente en el sidebar con acceso directo a todas las funcionalidades.
    """
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🧭 Navegación Rápida")
        st.caption("Acceso directo a todas las funcionalidades")
        
        # Obtener la vista actual
        current_view = st.session_state.get("app_view", "landing")
        
        # Definir las opciones de navegación (mismas que en landing page)
        nav_options = [
            {"icon": "🏠", "label": "Inicio", "view": "landing"},
            {"icon": "📂", "label": "Archivo CSV", "view": "csv"},
            {"icon": "🧰", "label": "Herramientas", "view": "tools"},
            {"icon": "🧠", "label": "Semantic Keyword", "view": "keywords"},
            {"icon": "🔗", "label": "Laboratorio Enlazado", "view": "linking"},
            {"icon": "📊", "label": "Informe Posiciones", "view": "positions"},
            {"icon": "🔍", "label": "Relaciones Semánticas", "view": "relations"},
        ]
        
        # Crear botones de navegación
        for option in nav_options:
            # Marcar el botón actual con un estilo diferente
            is_current = current_view == option["view"]
            button_label = f"{option['icon']} **{option['label']}**" if is_current else f"{option['icon']} {option['label']}"
            
            if st.button(
                button_label,
                key=f"nav_{option['view']}",
                use_container_width=True,
                type="primary" if is_current else "secondary"
            ):
                if option["view"] != current_view:
                    set_app_view(option["view"])


def apply_global_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-primary: #0f111a;
            --bg-secondary: #16192a;
            --bg-card: #1d2136;
            --border-color: rgba(255,255,255,0.08);
            --accent: #5c6bff;
            --accent-soft: rgba(92,107,255,0.15);
            --text-primary: #f5f7ff;
            --text-secondary: #a0a8c3;
        }

        /* Fondo general */
        body {
            background-color: var(--bg-primary);
            color: var(--text-primary);
        }
        .main {
            background-color: var(--bg-primary);
        }
        .stApp {
            background-color: var(--bg-primary);
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #14172b;
        }
        section[data-testid="stSidebar"] * {
            color: var(--text-primary) !important;
        }

        /* Textos y headings */
        .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
            color: var(--text-primary) !important;
        }
        h1, h2, h3, h4, h5, h6, p, span, label {
            color: var(--text-primary) !important;
        }

        /* DataFrames y tablas */
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            background-color: var(--bg-card);
        }
        .dataframe {
            color: var(--text-primary) !important;
            background-color: var(--bg-card) !important;
        }
        .dataframe thead th {
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
        }
        .dataframe tbody tr {
            background-color: var(--bg-card) !important;
        }
        .dataframe tbody tr:hover {
            background-color: var(--bg-secondary) !important;
        }

        /* Inputs, selectbox, multiselect */
        .stTextInput > div > div > input,
        .stSelectbox > div > div > div,
        .stMultiSelect > div > div > div,
        .stTextArea > div > div > textarea {
            background-color: var(--bg-card) !important;
            color: var(--text-primary) !important;
            border-color: var(--border-color) !important;
        }

        /* Metrics */
        [data-testid="stMetricValue"] {
            color: var(--text-primary) !important;
        }
        [data-testid="stMetricLabel"] {
            color: var(--text-secondary) !important;
        }

        /* Expander */
        .streamlit-expanderHeader {
            background-color: var(--bg-card) !important;
            color: var(--text-primary) !important;
        }
        .streamlit-expanderContent {
            background-color: var(--bg-secondary) !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: var(--bg-card);
            color: var(--text-secondary);
            border-radius: 8px 8px 0 0;
        }
        .stTabs [aria-selected="true"] {
            background-color: var(--bg-secondary);
            color: var(--text-primary) !important;
        }

        /* Buttons */
        .stButton > button {
            background-color: var(--accent) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
        }
        .stButton > button:hover {
            background-color: #4a5cef !important;
        }

        /* Download button */
        .stDownloadButton > button {
            background-color: var(--bg-card) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
        }
        .card-panel {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 18px;
            padding: 1.5rem 1.75rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 20px 45px rgba(5, 7, 12, 0.45);
        }
        .card-panel h3 {
            margin-top: 0.2rem;
            margin-bottom: 0.35rem;
            font-weight: 600;
        }
        div[data-testid="stFileUploader"] {
            background: #1b1f30;
            border: 1px dashed rgba(255,255,255,0.2);
            border-radius: 18px;
            padding: 1.5rem;
        }
        .action-card {
            background: #1b1f32;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 20px;
            padding: 1.75rem;
            box-shadow: 0 25px 45px rgba(8, 10, 20, 0.4);
            transition: transform 0.25s ease, border-color 0.25s ease, background 0.25s ease;
        }
        .action-card.primary {
            background: linear-gradient(180deg, rgba(92,107,255,0.18), rgba(92,107,255,0.08));
            border-color: rgba(92,107,255,0.4);
        }
        .action-card.secondary {
            background: linear-gradient(180deg, rgba(210,186,255,0.25), rgba(210,186,255,0.1));
            border-color: rgba(210,186,255,0.4);
        }
        .action-card:hover {
            transform: translateY(-6px);
            border-color: var(--accent);
        }
        .action-card .icon {
            font-size: 2.6rem;
            margin-bottom: 0.8rem;
        }
        .cta-button button {
            border-radius: 999px !important;
            width: 100%;
            padding: 0.55rem 1.2rem !important;
            font-weight: 600 !important;
            border: 1px solid rgba(255,255,255,0.2) !important;
            background: rgba(15,17,26,0.4) !important;
        }
        .cta-button button:hover {
            border-color: var(--accent) !important;
            color: #fff !important;
        }
        .back-link button {
            background: transparent !important;
            border: 1px solid rgba(255,255,255,0.18) !important;
            color: var(--text-secondary) !important;
            border-radius: 999px !important;
            padding: 0.35rem 0.9rem !important;
            font-size: 0.9rem !important;
        }
        .back-link button:hover {
            border-color: var(--accent) !important;
            color: #fff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def bordered_container():
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()


def set_app_view(view: str) -> None:
    st.session_state["app_view"] = view
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


def render_back_to_landing() -> None:
    with st.container():
        st.markdown("")
        back_key = f"back_to_landing_{st.session_state.get('app_view', 'landing')}"
        with st.container():
            st.markdown('<div class="back-link">', unsafe_allow_html=True)
            if st.button("⬅️ Volver a selección", key=back_key):
                set_app_view("landing")
            st.markdown("</div>", unsafe_allow_html=True)


def render_landing_view() -> None:
    st.subheader("Selección de funcionalidad")
    st.caption(
        "Escoge si quieres cargar un dataset, usar herramientas rápidas, el builder semántico o el nuevo laboratorio de enlazado."
    )
    cards = [
        {
            "icon": "📂",
            "title": "Trabajar mediante archivo CSV",
            "body": "Sube tu dataset para desbloquear los análisis principales (similitud, clustering, grafo).",
            "button": "Ir a carga CSV",
            "view": "csv",
            "style": "primary",
            "key": "cta_csv",
        },
        {
            "icon": "🧰",
            "title": "Trabajar sin archivo CSV",
            "body": "Usa las herramientas adicionales (texto vs keywords, FAQs, competidores, URLs enriquecidas).",
            "button": "Ir a herramientas adicionales",
            "view": "tools",
            "style": "secondary",
            "key": "cta_tools",
        },
        {
            "icon": "🧠",
            "title": "Semantic Keyword Builder",
            "body": "Genera un universo EAV de keywords con intención, volumen cualitativo y clusters usando Gemini.",
            "button": "Ir a Semantic Keyword",
            "view": "keywords",
            "style": "secondary",
            "key": "cta_keywords",
        },
        {
            "icon": "🔗",
            "title": "Laboratorio de enlazado interno",
            "body": "Accede a los modos básico, avanzado, híbrido (CLS) y estructural para optimizar tu internal linking.",
            "button": "Ir al laboratorio de enlazado",
            "view": "linking",
            "style": "secondary",
            "key": "cta_linking",
        },
        {
            "icon": "📊",
            "title": "Informe de posiciones",
            "body": "Convierte un CSV de rankings en un informe HTML con insights competitivos y sugerencias gráficas.",
            "button": "Ir a informe de posiciones",
            "view": "positions",
            "style": "secondary",
            "key": "cta_positions",
        },
        {
            "icon": "🔍",
            "title": "Relaciones Semánticas",
            "body": "Visualiza las relaciones semánticas entre palabras clave con matrices de similitud, grafos de red y mapas 2D interactivos.",
            "button": "Analizar relaciones",
            "view": "relations",
            "style": "primary",
            "key": "cta_relations",
        },
    ]

    row_cols = None
    for idx, card in enumerate(cards):
        if idx % 2 == 0:
            row_cols = st.columns(2, gap="large")
        col = row_cols[idx % 2]
        with col:
            card_class = "primary" if card["style"] == "primary" else "secondary"
            st.markdown(
                f"""
                <div class="action-card {card_class}">
                    <div class="icon">{card['icon']}</div>
                    <h4>{card['title']}</h4>
                    <p>{card['body']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown('<div class="cta-button">', unsafe_allow_html=True)
            if st.button(card["button"], key=card["key"]):
                set_app_view(card["view"])
            st.markdown("</div>", unsafe_allow_html=True)


__all__ = [
    "apply_global_styles",
    "bordered_container",
    "get_gemini_api_key_from_context",
    "get_gemini_model_from_context",
    "render_api_settings_panel",
    "render_back_to_landing",
    "render_landing_view",
    "render_sidebar_navigation",  # ← Nueva función
    "set_app_view",
]
