from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import seaborn as sns
import streamlit as st

from app_sections.authority_advance import (
    AuthorityGapResult,
    run_authority_gap_from_embeddings,
    run_authority_gap_simulation,
)
from app_sections.linking_lab import render_linking_lab

# Funciones migradas a módulos compartidos
from apps.content_analyzer.modules.shared.content_utils import (
    detect_embedding_columns,
    preprocess_embeddings,
    detect_url_columns,
    detect_page_type_columns,
)
from apps.linking_optimizer.modules.linking_utils import (
    build_entity_payload_from_doc_relations,
)
from app_sections.keyword_builder import (
    render_semantic_keyword_builder,
    group_keywords_with_semantic_builder,
)
from app_sections.semantic_tools import (
    best_similarity_per_url_keyword,
    build_url_variant_entries,
    compute_faq_keyword_similarity,
    compute_text_keyword_similarity,
    compute_url_variant_keyword_similarity,
    download_dataframe_button,
    fetch_url_content,
    fetch_url_text_variants,
    get_sentence_transformer,
    keyword_relevance,
    parse_faq_blocks,
    parse_line_input,
    render_semantic_toolkit_section,
    top_n_by_group,
)
from app_sections.csv_workflow import render_csv_workflow
from app_sections.positions_report import render_positions_report
from app_sections.semantic_relations import render_semantic_relations
from app_sections.fanout_report import render_fanout_report
from app_sections.landing_page import (
    render_api_settings_panel,
    apply_global_styles,
    render_back_to_landing,
    render_landing_view,
    render_sidebar_navigation,
)
from shared.env_utils import bootstrap_api_session_state
from shared.license_ui import (
    init_license_check,
    render_license_status_sidebar,
    require_feature,
)

if TYPE_CHECKING:  # pragma: no cover - solo para anotaciones
    import spacy  # noqa: F401

try:
    import google.generativeai as genai
except ModuleNotFoundError:
    genai = None

COREFEREE_MODULE = None
COREFEREE_IMPORT_ERROR: Optional[Exception] = None
ENTITY_LINKER_MODULE = None
ENTITY_LINKER_IMPORT_ERROR: Optional[Exception] = None

SPACY_MODULE = None
SPACY_DOWNLOAD_FN = None
SPACY_IMPORT_ERROR: Optional[Exception] = None
SUPPORTED_COREF_LANGS = {"en", "de", "fr", "pl"}
from shared.config import ENTITY_PROFILE_PRESETS  # noqa: E402

st.set_page_config(
    page_title="Embedding Insights Dashboard",
    layout="wide",
    page_icon=":bar_chart:",
)

sns.set_theme(style="whitegrid")



def main():
    bootstrap_api_session_state()

    # Inicializar verificación de licencia
    init_license_check()

    apply_global_styles()
<<<<<<< HEAD
    st.title("📈 Embedding Insights Dashboard")
=======
    st.title("Embedding Insights Dashboard")
>>>>>>> 7e807037dbf03248488e6303b1d6335fb21f7514
    st.markdown(
        "Sube tus datos de embeddings para descubrir similitudes entre URLs, agruparlas en clusters "
        "y analizar la relevancia frente a palabras clave."
    )

    if "entity_payload_by_url" not in st.session_state:
        st.session_state["entity_payload_by_url"] = None
    if "processed_df" not in st.session_state:
        st.session_state["processed_df"] = None
    if "raw_df" not in st.session_state:
        st.session_state["raw_df"] = None
    if "url_column" not in st.session_state:
        st.session_state["url_column"] = None
    if "cluster_result" not in st.session_state:
        st.session_state["cluster_result"] = None
    if "tsne_figure" not in st.session_state:
        st.session_state["tsne_figure"] = None
    if "topic_graph_html" not in st.session_state:
        st.session_state["topic_graph_html"] = None
    if "knowledge_graph_html" not in st.session_state:
        st.session_state["knowledge_graph_html"] = None
    if "knowledge_entities" not in st.session_state:
        st.session_state["knowledge_entities"] = None
    if "knowledge_doc_relations" not in st.session_state:
        st.session_state["knowledge_doc_relations"] = None
    if "knowledge_spo_relations" not in st.session_state:
        st.session_state["knowledge_spo_relations"] = None
    if "auto_cluster_scores" not in st.session_state:
        st.session_state["auto_cluster_scores"] = None
    if "auto_cluster_best_k" not in st.session_state:
        st.session_state["auto_cluster_best_k"] = None
    if "page_type_column" not in st.session_state:
        st.session_state["page_type_column"] = None
    if "semantic_links_report" not in st.session_state:
        st.session_state["semantic_links_report"] = None
    if "semantic_links_adv_report" not in st.session_state:
        st.session_state["semantic_links_adv_report"] = None
    if "semantic_links_adv_orphans" not in st.session_state:
        st.session_state["semantic_links_adv_orphans"] = None

    if "app_view" not in st.session_state:
        st.session_state["app_view"] = "landing"

    render_api_settings_panel()
    render_sidebar_navigation()
    render_license_status_sidebar()  # Estado de licencia en sidebar

    app_view = st.session_state["app_view"]

    if app_view == "csv":
        with st.sidebar:
            st.header("Pasos")
            st.markdown("1. Subir archivo con embeddings.")
            st.markdown("2. Ejecutar anÃ¡lisis de similitud.")
            st.markdown("3. Encontrar enlaces internos semÃ¡nticos.")
            st.markdown("4. Aplicar enlazado avanzado por silos (opcional).")
            st.markdown("5. Ejecutar clustering (opcional).")
            st.markdown("6. Analizar relevancia por palabras clave.")
            st.markdown("7. Construir grafo de conocimiento (opcional).")

    if app_view == "landing":
        render_landing_view()
        return

    render_back_to_landing()

    if app_view == "csv":
        render_csv_workflow()
    elif app_view == "tools":
        render_semantic_toolkit_section()
    elif app_view == "keywords":
        if require_feature("keywords", "Keyword Builder"):
            render_semantic_keyword_builder()
    elif app_view == "linking":
        if require_feature("linking", "Linking Lab"):
            render_linking_lab()
    elif app_view == "positions":
        if require_feature("positions", "Positions Report"):
            render_positions_report()
    elif app_view == "relations":
        if require_feature("relations", "Relaciones Semánticas"):
            render_semantic_relations()
    elif app_view == "fanout":
        if require_feature("fanout", "Fan-Out Analyzer"):
            render_fanout_report()




if __name__ == "__main__":
    main()
