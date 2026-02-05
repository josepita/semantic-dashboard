"""
Brief Generator
===============
Generador de Briefs SEO con análisis SERP y propuestas de contenido.

Dos modos de entrada:
1. Desde Excel: Subir Excel con títulos/keywords → Generar HNs → SERP
2. Brief Directo: Input keyword → SERP inmediato → Keywords + Propuestas

Autor: Embedding Insights
Versión: 1.0.0
"""

import sys
from pathlib import Path

# ══════════════════════════════════════════════════════════
# PATH SETUP
# ══════════════════════════════════════════════════════════

current_dir = Path(__file__).parent.resolve()
project_root = (current_dir.parent.parent).resolve()
shared_path = (project_root / "shared").resolve()
modules_path = (current_dir / "modules").resolve()

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(shared_path) not in sys.path:
    sys.path.insert(0, str(shared_path))
if str(modules_path) not in sys.path:
    sys.path.insert(0, str(modules_path))

# ══════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
from datetime import datetime

from modules.brief_storage import (
    init_database,
    create_folder,
    get_folders,
    delete_folder,
    rename_folder,
    create_brief,
    get_briefs,
    get_brief,
    update_brief,
    delete_brief,
    move_brief_to_folder,
    get_brief_stats,
)
from modules.serp_scraper import (
    scrape_serp,
    serp_to_dict,
    get_google_suggestions,
    COUNTRY_FLAGS,
    GOOGLE_DOMAINS,
)
from modules.brief_ai import (
    is_llm_configured,
    render_llm_config_sidebar,
    generate_title_proposals,
    generate_meta_proposals,
    generate_hn_structure,
    generate_full_brief,
)

# License management - TEMPORAL: licencias desactivadas
# TODO: Restaurar verificación de licencias cuando esté listo
def init_license_check(): pass
def render_license_status_sidebar(): pass

# ══════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════

SS = "bg_"  # Session state prefix
COUNTRIES = list(COUNTRY_FLAGS.keys())

STATUS_LABELS = {
    "pending": ("Pendiente", "🟡"),
    "serp_done": ("SERP Listo", "🔵"),
    "keywords_done": ("Keywords Listo", "🟣"),
    "completed": ("Completado", "🟢"),
}


# ══════════════════════════════════════════════════════════
# INITIALIZATION
# ══════════════════════════════════════════════════════════

def init_app():
    """Inicializa la aplicación y la base de datos."""
    if f"{SS}initialized" not in st.session_state:
        db_path = init_database()
        st.session_state[f"{SS}db_path"] = db_path
        st.session_state[f"{SS}initialized"] = True
        st.session_state[f"{SS}selected_folder"] = None
        st.session_state[f"{SS}selected_brief"] = None
        st.session_state[f"{SS}view_mode"] = "list"  # list, detail


def get_db():
    """Obtiene la ruta de la DB."""
    return st.session_state.get(f"{SS}db_path")


# ══════════════════════════════════════════════════════════
# SIDEBAR - FOLDERS
# ══════════════════════════════════════════════════════════

def render_sidebar():
    """Renderiza el sidebar con carpetas y navegación."""
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=Brief+Generator", width=150)
        st.markdown("### SEO Intelligence")
        st.markdown("---")

        # Botón nuevo brief
        if st.button("➕ Nuevo Brief", use_container_width=True, type="primary"):
            st.session_state[f"{SS}view_mode"] = "new_brief"
            st.session_state[f"{SS}selected_brief"] = None

        # Todos los briefs
        if st.button("📋 Todos los Briefs", use_container_width=True):
            st.session_state[f"{SS}selected_folder"] = None
            st.session_state[f"{SS}view_mode"] = "list"

        st.markdown("---")
        st.markdown("### 📁 Mis Carpetas")

        # Lista de carpetas
        folders = get_folders(get_db())

        for folder in folders:
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(
                    f"📂 {folder['name']} ({folder['brief_count']})",
                    key=f"folder_{folder['id']}",
                    use_container_width=True
                ):
                    st.session_state[f"{SS}selected_folder"] = folder['id']
                    st.session_state[f"{SS}view_mode"] = "list"
            with col2:
                if st.button("🗑️", key=f"del_folder_{folder['id']}"):
                    delete_folder(folder['id'], get_db())
                    st.rerun()

        # Crear nueva carpeta
        with st.expander("➕ Nueva Carpeta"):
            new_folder_name = st.text_input("Nombre", key=f"{SS}new_folder_name")
            if st.button("Crear", key=f"{SS}create_folder_btn"):
                if new_folder_name:
                    result = create_folder(new_folder_name, get_db())
                    if result:
                        st.success(f"Carpeta '{new_folder_name}' creada")
                        st.rerun()
                    else:
                        st.error("Ya existe una carpeta con ese nombre")

        # Stats
        st.markdown("---")
        stats = get_brief_stats(get_db())
        st.markdown("### 📊 Estadísticas")
        col1, col2 = st.columns(2)
        col1.metric("Total", stats['total'])
        col2.metric("Completados", stats['completed'])

        # Config IA
        st.markdown("---")
        render_llm_config_sidebar()


# ══════════════════════════════════════════════════════════
# BRIEF LIST VIEW
# ══════════════════════════════════════════════════════════

def render_brief_list():
    """Renderiza la lista de briefs."""
    selected_folder = st.session_state.get(f"{SS}selected_folder")
    folders = get_folders(get_db())

    # Header
    if selected_folder:
        folder_name = next((f['name'] for f in folders if f['id'] == selected_folder), "Carpeta")
        st.header(f"📂 {folder_name}")
    else:
        st.header("📋 Todos los Briefs")

    # Filtros
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        filter_status = st.selectbox(
            "Estado",
            ["Todos"] + list(STATUS_LABELS.keys()),
            format_func=lambda x: STATUS_LABELS.get(x, ("Todos", ""))[0] if x != "Todos" else "Todos"
        )
    with col3:
        if st.button("🔄 Actualizar"):
            st.rerun()

    # Obtener briefs
    briefs = get_briefs(
        folder_id=selected_folder,
        status=filter_status if filter_status != "Todos" else None,
        db_path=get_db()
    )

    if not briefs:
        st.info("No hay briefs. Crea uno nuevo con el botón '+ Nuevo Brief'")
        return

    # Tabla de briefs
    for brief in briefs:
        status_label, status_icon = STATUS_LABELS.get(brief['status'], ("", ""))
        country_flag = COUNTRY_FLAGS.get(brief['country'], "🌐")

        col1, col2, col3, col4, col5, col6 = st.columns([3, 1, 2, 2, 1, 1])

        with col1:
            if st.button(f"📄 {brief['keyword']}", key=f"brief_{brief['id']}"):
                st.session_state[f"{SS}selected_brief"] = brief['id']
                st.session_state[f"{SS}view_mode"] = "detail"
                st.rerun()

        with col2:
            st.write(country_flag)

        with col3:
            # Selector de carpeta
            folder_options = [{"id": None, "name": "Sin carpeta"}] + folders
            current_idx = next(
                (i for i, f in enumerate(folder_options) if f['id'] == brief['folder_id']),
                0
            )
            new_folder = st.selectbox(
                "Carpeta",
                folder_options,
                index=current_idx,
                format_func=lambda x: x['name'],
                key=f"move_{brief['id']}",
                label_visibility="collapsed"
            )
            if new_folder['id'] != brief['folder_id']:
                move_brief_to_folder(brief['id'], new_folder['id'], get_db())
                st.rerun()

        with col4:
            st.write(f"{status_icon} {status_label}")

        with col5:
            created = brief['created_at'][:10] if brief['created_at'] else ""
            st.write(created)

        with col6:
            if st.button("🗑️", key=f"del_brief_{brief['id']}"):
                delete_brief(brief['id'], get_db())
                st.rerun()

        st.markdown("---")


# ══════════════════════════════════════════════════════════
# NEW BRIEF VIEW
# ══════════════════════════════════════════════════════════

def render_new_brief():
    """Renderiza el formulario de nuevo brief."""
    st.header("✨ Nuevo Brief SEO")

    tab1, tab2 = st.tabs(["📝 Brief Directo", "📊 Desde Excel"])

    # ── Tab 1: Brief Directo ──
    with tab1:
        st.markdown("Ingresa una palabra clave para generar un brief con datos de SERP.")

        col1, col2 = st.columns([3, 1])
        with col1:
            keyword = st.text_input("Palabra clave", placeholder="ej: auditor financiero")
        with col2:
            country = st.selectbox(
                "País",
                COUNTRIES,
                format_func=lambda x: f"{COUNTRY_FLAGS.get(x, '')} {x}"
            )

        # Selector de carpeta
        folders = get_folders(get_db())
        folder_options = [{"id": None, "name": "Sin carpeta"}] + folders
        selected_folder = st.selectbox(
            "Guardar en carpeta",
            folder_options,
            format_func=lambda x: f"📂 {x['name']}" if x['id'] else x['name']
        )

        if st.button("🚀 Generar Brief", type="primary", disabled=not keyword):
            with st.spinner("Analizando SERP..."):
                # Crear brief
                brief_id = create_brief(
                    keyword=keyword,
                    country=country,
                    source="direct",
                    folder_id=selected_folder['id'],
                    db_path=get_db()
                )

                # Scrape SERP
                serp_data = scrape_serp(keyword, country)

                if serp_data:
                    # Guardar resultados
                    update_brief(
                        brief_id,
                        db_path=get_db(),
                        serp_results=serp_to_dict(serp_data),
                        status="serp_done"
                    )
                    st.success(f"Brief creado con {len(serp_data.organic_results)} resultados SERP")

                    # Ir al detalle
                    st.session_state[f"{SS}selected_brief"] = brief_id
                    st.session_state[f"{SS}view_mode"] = "detail"
                    st.rerun()
                else:
                    st.warning("No se pudieron obtener resultados SERP. El brief se creó como pendiente.")
                    update_brief(brief_id, db_path=get_db(), status="pending")

    # ── Tab 2: Desde Excel ──
    with tab2:
        st.markdown("Sube un Excel con múltiples keywords para generar briefs en lote.")
        st.caption("El archivo debe tener una columna llamada **keyword** o **kw**")

        uploaded = st.file_uploader("Archivo Excel", type=["xlsx", "xls", "csv"])

        if uploaded:
            try:
                if uploaded.name.endswith('.csv'):
                    df = pd.read_csv(uploaded)
                else:
                    df = pd.read_excel(uploaded)

                # Normalizar nombres de columnas
                df.columns = df.columns.str.lower().str.strip()

                # Buscar columna de keyword
                kw_col = None
                for col in ['keyword', 'kw', 'keywords', 'palabra clave']:
                    if col in df.columns:
                        kw_col = col
                        break

                if not kw_col:
                    st.error("No se encontró columna de keywords. Usa 'keyword' o 'kw'")
                else:
                    st.success(f"✅ {len(df)} keywords encontradas")
                    st.dataframe(df.head(10))

                    col1, col2 = st.columns(2)
                    with col1:
                        country = st.selectbox(
                            "País para todos",
                            COUNTRIES,
                            format_func=lambda x: f"{COUNTRY_FLAGS.get(x, '')} {x}",
                            key="excel_country"
                        )
                    with col2:
                        folder_options = [{"id": None, "name": "Sin carpeta"}] + folders
                        selected_folder = st.selectbox(
                            "Carpeta destino",
                            folder_options,
                            format_func=lambda x: f"📂 {x['name']}" if x['id'] else x['name'],
                            key="excel_folder"
                        )

                    if st.button("📥 Importar Briefs", type="primary"):
                        progress = st.progress(0)
                        status_text = st.empty()

                        keywords = df[kw_col].dropna().unique().tolist()
                        total = len(keywords)

                        for i, kw in enumerate(keywords):
                            status_text.text(f"Procesando: {kw}")

                            # Crear brief
                            brief_id = create_brief(
                                keyword=str(kw).strip(),
                                country=country,
                                source="excel",
                                folder_id=selected_folder['id'],
                                db_path=get_db()
                            )

                            progress.progress((i + 1) / total)

                        st.success(f"✅ {total} briefs importados")
                        st.session_state[f"{SS}view_mode"] = "list"
                        st.rerun()

            except Exception as e:
                st.error(f"Error al procesar archivo: {e}")


# ══════════════════════════════════════════════════════════
# BRIEF DETAIL VIEW
# ══════════════════════════════════════════════════════════

def render_brief_detail():
    """Renderiza el detalle de un brief."""
    brief_id = st.session_state.get(f"{SS}selected_brief")
    if not brief_id:
        st.warning("No hay brief seleccionado")
        return

    brief = get_brief(brief_id, get_db())
    if not brief:
        st.error("Brief no encontrado")
        return

    # Header
    col1, col2, col3 = st.columns([1, 6, 2])
    with col1:
        if st.button("← Volver"):
            st.session_state[f"{SS}view_mode"] = "list"
            st.rerun()
    with col2:
        country_flag = COUNTRY_FLAGS.get(brief['country'], "🌐")
        st.header(f"{country_flag} {brief['keyword']}")
    with col3:
        status_label, status_icon = STATUS_LABELS.get(brief['status'], ("", ""))
        st.markdown(f"### {status_icon} {status_label}")

    st.markdown("---")

    # Tabs de contenido
    tab_serp, tab_keywords, tab_proposals = st.tabs([
        "🔍 Competencia SERP",
        "🔑 Oportunidades Keywords",
        "💡 Propuestas SEO"
    ])

    # ── Tab SERP ──
    with tab_serp:
        serp_data = brief.get('serp_results')

        if not serp_data:
            st.info("No hay datos SERP. Haz clic en 'Analizar SERP' para obtenerlos.")
            if st.button("🔍 Analizar SERP", type="primary"):
                with st.spinner("Analizando SERP..."):
                    result = scrape_serp(brief['keyword'], brief['country'])
                    if result:
                        update_brief(
                            brief_id,
                            db_path=get_db(),
                            serp_results=serp_to_dict(result),
                            status="serp_done"
                        )
                        st.success("SERP analizado correctamente")
                        st.rerun()
                    else:
                        st.error("No se pudieron obtener resultados")
        else:
            st.subheader(f"Top {len(serp_data.get('organic_results', []))} Competencia Orgánica")

            for result in serp_data.get('organic_results', []):
                with st.container():
                    col1, col2 = st.columns([1, 20])
                    with col1:
                        st.markdown(f"**{result['position']}**")
                    with col2:
                        st.markdown(f"**{result['domain']}**")
                        st.markdown(f"[{result['title']}]({result['url']})")
                        st.caption(result.get('snippet', '')[:200])
                    st.markdown("---")

            # People Also Ask
            paa = serp_data.get('people_also_ask', [])
            if paa:
                st.subheader("❓ Preguntas Relacionadas (PAA)")
                for q in paa:
                    st.markdown(f"- {q}")

            # Related Searches
            related = serp_data.get('related_searches', [])
            if related:
                st.subheader("🔗 Búsquedas Relacionadas")
                cols = st.columns(3)
                for i, search in enumerate(related):
                    cols[i % 3].markdown(f"• {search}")

    # ── Tab Keywords ──
    with tab_keywords:
        kw_data = brief.get('keyword_opportunities')

        if not kw_data:
            st.info("Obtén sugerencias de keywords relacionadas.")
            if st.button("🔑 Obtener Keywords", type="primary"):
                with st.spinner("Buscando oportunidades..."):
                    suggestions = get_google_suggestions(brief['keyword'], brief['country'])

                    # También de PAA y related si existen
                    serp = brief.get('serp_results', {})
                    paa = serp.get('people_also_ask', [])
                    related = serp.get('related_searches', [])

                    kw_opportunities = {
                        "suggestions": suggestions,
                        "questions": paa,
                        "related": related
                    }

                    update_brief(
                        brief_id,
                        db_path=get_db(),
                        keyword_opportunities=kw_opportunities,
                        status="keywords_done" if brief['status'] != 'completed' else 'completed'
                    )
                    st.success("Keywords obtenidas")
                    st.rerun()
        else:
            st.subheader("💡 Sugerencias de Google")
            for kw in kw_data.get('suggestions', []):
                st.markdown(f"• {kw}")

            st.subheader("❓ Preguntas")
            for q in kw_data.get('questions', []):
                st.markdown(f"• {q}")

            st.subheader("🔗 Relacionadas")
            for r in kw_data.get('related', []):
                st.markdown(f"• {r}")

    # ── Tab Propuestas ──
    with tab_proposals:
        serp_data = brief.get('serp_results', {})
        title_proposals = brief.get('title_proposals')
        meta_proposals = brief.get('meta_proposals')
        hn_structure = brief.get('hn_structure')

        # Verificar si hay datos SERP
        if not serp_data:
            st.warning("Primero debes analizar el SERP en la pestaña 'Competencia SERP'")
        elif not is_llm_configured():
            st.warning("Configura una API de IA (Gemini/OpenAI) en el panel lateral para generar propuestas")
        else:
            # Botón para generar todo
            if not title_proposals:
                st.info("Genera propuestas de titulo, meta description y estructura de encabezados con IA.")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🚀 Generar Todo", type="primary"):
                        with st.spinner("Generando propuestas con IA..."):
                            result = generate_full_brief(
                                keyword=brief['keyword'],
                                serp_data=serp_data,
                                country=brief['country']
                            )

                            # Guardar resultados
                            update_brief(
                                brief_id,
                                db_path=get_db(),
                                title_proposals=result['titles'],
                                meta_proposals=result['metas'],
                                hn_structure=result['hn_structure'],
                                status="completed"
                            )

                            if result['errors']:
                                for err in result['errors']:
                                    st.warning(err)
                            else:
                                st.success("Propuestas generadas correctamente")
                            st.rerun()

                with col2:
                    if st.button("📝 Solo Titulos"):
                        with st.spinner("Generando titulos..."):
                            titles, error = generate_title_proposals(
                                brief['keyword'], serp_data, brief['country']
                            )
                            if titles:
                                update_brief(brief_id, db_path=get_db(), title_proposals=titles)
                                st.success(f"{len(titles)} titulos generados")
                                st.rerun()
                            else:
                                st.error(error or "Error al generar titulos")

            # Mostrar propuestas existentes
            if title_proposals:
                st.subheader("📝 Propuestas de Titulo")
                selected_title = brief.get('selected_title', title_proposals[0] if title_proposals else '')

                for i, title in enumerate(title_proposals):
                    col1, col2 = st.columns([1, 10])
                    with col1:
                        is_selected = title == selected_title
                        if st.checkbox("", value=is_selected, key=f"title_{i}"):
                            if not is_selected:
                                update_brief(brief_id, db_path=get_db(), selected_title=title)
                                st.rerun()
                    with col2:
                        char_count = len(title)
                        color = "green" if char_count <= 60 else "orange"
                        st.markdown(f"**{title}** :{color}[({char_count} chars)]")

                # Generar metas basadas en titulo seleccionado
                if not meta_proposals:
                    if st.button("📋 Generar Meta Descriptions"):
                        with st.spinner("Generando metas..."):
                            metas, error = generate_meta_proposals(
                                brief['keyword'], selected_title, serp_data, brief['country']
                            )
                            if metas:
                                update_brief(brief_id, db_path=get_db(), meta_proposals=metas)
                                st.success(f"{len(metas)} meta descriptions generadas")
                                st.rerun()
                            else:
                                st.error(error or "Error al generar metas")

            if meta_proposals:
                st.subheader("📋 Propuestas de Meta Description")
                selected_meta = brief.get('selected_meta', meta_proposals[0] if meta_proposals else '')

                for i, meta in enumerate(meta_proposals):
                    col1, col2 = st.columns([1, 10])
                    with col1:
                        is_selected = meta == selected_meta
                        if st.checkbox("", value=is_selected, key=f"meta_{i}"):
                            if not is_selected:
                                update_brief(brief_id, db_path=get_db(), selected_meta=meta)
                                st.rerun()
                    with col2:
                        char_count = len(meta)
                        color = "green" if char_count <= 155 else "orange"
                        st.markdown(f"{meta} :{color}[({char_count} chars)]")

            if hn_structure:
                st.subheader("🏗️ Estructura de Encabezados")
                st.markdown(f"### {hn_structure.get('h1', '')}")

                for section in hn_structure.get('sections', []):
                    st.markdown(f"**{section.get('h2', '')}**")
                    for h3 in section.get('h3', []):
                        st.markdown(f"  - {h3}")

            # Botón regenerar
            if title_proposals or meta_proposals:
                st.markdown("---")
                if st.button("🔄 Regenerar Propuestas"):
                    update_brief(
                        brief_id,
                        db_path=get_db(),
                        title_proposals=None,
                        meta_proposals=None,
                        hn_structure=None
                    )
                    st.rerun()

    # Marcar como completado
    st.markdown("---")
    if brief['status'] != 'completed':
        if st.button("✅ Marcar como Completado", type="primary"):
            update_brief(brief_id, db_path=get_db(), status="completed")
            st.success("Brief marcado como completado")
            st.rerun()


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    """Función principal de la aplicación."""
    st.set_page_config(
        page_title="Brief Generator",
        page_icon="📋",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Estilos CSS
    st.markdown("""
        <style>
        .stButton button {
            width: 100%;
        }
        div[data-testid="stHorizontalBlock"] > div {
            align-items: center;
        }
        </style>
    """, unsafe_allow_html=True)

    # Verificar licencia
    init_license_check()

    # Inicializar
    init_app()

    # Sidebar
    render_sidebar()
    render_license_status_sidebar()

    # Main content
    view_mode = st.session_state.get(f"{SS}view_mode", "list")

    if view_mode == "new_brief":
        render_new_brief()
    elif view_mode == "detail":
        render_brief_detail()
    else:
        render_brief_list()


if __name__ == "__main__":
    main()
