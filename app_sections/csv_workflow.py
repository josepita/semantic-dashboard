from __future__ import annotations

import io
import math
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from app_sections.knowledge_graph import (
    build_entity_payload_from_doc_relations,
    generate_knowledge_graph_html_v2,
)
# Funciones migradas a módulos compartidos
from apps.content_analyzer.modules.shared.content_utils import (
    detect_embedding_columns,
    detect_url_columns,
    preprocess_embeddings,
)
from app_sections.semantic_tools import (
    download_dataframe_button,
    keyword_relevance,
    parse_line_input,
)
from shared.license_ui import get_csv_limit_message, is_licensed

# Constants that need to be imported or defined
ENTITY_PROFILE_PRESETS: Dict[str, List[str]] = {
    "Clinica / Salud": [
        "ORG",
        "PERSON",
        "PRODUCT",
        "GPE",
        "LOC",
        "FAC",
        "EVENT",
        "LAW",
        "NORP",
        "LANGUAGE",
        "WORK_OF_ART",
        "DISEASE",
        "SYMPTOM",
        "MEDICATION",
    ],
    "Editorial / Libros": [
        "ORG",
        "PERSON",
        "WORK_OF_ART",
        "PRODUCT",
        "EVENT",
        "GPE",
        "LOC",
        "LANGUAGE",
        "LAW",
        "NORP",
        "FAC",
    ],
    "Ecommerce / Retail": [
        "ORG",
        "PRODUCT",
        "PERSON",
        "GPE",
        "LOC",
        "FAC",
        "EVENT",
        "WORK_OF_ART",
        "LAW",
        "NORP",
    ],
}


from shared.ui_components import bordered_container  # noqa: E402
from shared.data_orchestrator import DataOrchestrator  # noqa: E402

_CSV_MODEL_ID = "csv_upload"

# Screaming Frog column mapping: SF name → canonical name
_SF_COLUMN_MAP = {
    "address": "url",
    "status code": "status_code",
    "indexability": "indexability",
    "word count": "word_count",
    "title 1": "title",
    "meta description 1": "meta_description",
    "h1-1": "h1",
    "canonical link element 1": "canonical_url",
    "content type": "content_type",
}

_SF_SIGNATURE_COLS = {"address", "status code", "indexability"}


def _detect_screaming_frog(df: pd.DataFrame) -> bool:
    """Return True if the DataFrame looks like a Screaming Frog export."""
    lower_cols = {c.lower().strip() for c in df.columns}
    return _SF_SIGNATURE_COLS.issubset(lower_cols)


def _normalize_screaming_frog(df: pd.DataFrame) -> pd.DataFrame:
    """Rename Screaming Frog columns to canonical names, keeping originals."""
    rename_map = {}
    for col in df.columns:
        key = col.lower().strip()
        if key in _SF_COLUMN_MAP:
            rename_map[col] = _SF_COLUMN_MAP[key]
    return df.rename(columns=rename_map)


def is_spacy_available() -> bool:
    """Checks if spaCy is available in the environment."""
    try:
        import importlib
        importlib.import_module("spacy")
        return True
    except Exception:
        return False


def compute_similar_pages(
    df: pd.DataFrame,
    target_url: str,
    url_column: str,
    top_n: int,
    embedding_col: str = "EmbeddingsFloat",
) -> List[Tuple[str, float]]:
    """Compute most similar pages to a target URL."""
    if target_url not in df[url_column].values:
        return []

    embeddings = np.vstack(df[embedding_col].values)
    embeddings_norm = normalize(embeddings)
    target_idx = df[df[url_column] == target_url].index[0]
    target_embedding = embeddings_norm[target_idx]

    dot_products = embeddings_norm @ target_embedding
    similarities = (dot_products * 100).tolist()

    results = []
    for idx, url in enumerate(df[url_column]):
        if url == target_url:
            continue
        results.append((url, similarities[idx]))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]


@st.cache_data(show_spinner=False)
def build_similarity_matrix(
    df: pd.DataFrame,
    url_column: str,
    similarity_threshold: Optional[float],
    max_results: Optional[int],
    embedding_col: str = "EmbeddingsFloat",
) -> pd.DataFrame:
    """Build a pairwise similarity matrix for all URLs."""
    urls = df[url_column].tolist()
    embeddings = np.vstack(df[embedding_col].values)
    embeddings_norm = normalize(embeddings)
    sim_matrix = embeddings_norm @ embeddings_norm.T * 100

    # Extract upper triangle indices efficiently with numpy
    i_idx, j_idx = np.triu_indices(len(urls), k=1)
    scores = sim_matrix[i_idx, j_idx]

    # Apply threshold filter vectorized
    if similarity_threshold is not None:
        mask = scores >= similarity_threshold
        i_idx, j_idx, scores = i_idx[mask], j_idx[mask], scores[mask]

    # Sort by score descending
    sort_order = np.argsort(scores)[::-1]
    if max_results is not None:
        sort_order = sort_order[:max_results]
    i_idx, j_idx, scores = i_idx[sort_order], j_idx[sort_order], scores[sort_order]

    url_arr = np.array(urls)
    result_df = pd.DataFrame({
        "URL1": url_arr[i_idx],
        "URL2": url_arr[j_idx],
        "Similitud": scores,
    })
    return result_df


def extract_words_from_url(url: str) -> List[str]:
    """Extract meaningful words from a URL for cluster naming."""
    clean = url.split("://")[-1]
    clean = clean.replace("www.", "")
    segments = [segment for segment in clean.split("/") if segment and not segment.startswith("?")]
    words: List[str] = []
    for segment in segments:
        parts = segment.replace("-", " ").replace("_", " ").split()
        for part in parts:
            tokens = Counter()
            tokens.update(
                word.lower()
                for word in part.replace(".", " ").replace("=", " ").split()
                if word.isalpha() and len(word) > 2
            )
            words.extend(tokens.elements())
    return words


@st.cache_data(show_spinner=False)
def auto_select_cluster_count(
    df: pd.DataFrame,
    min_clusters: int,
    max_clusters: int,
    embedding_col: str = "EmbeddingsFloat",
) -> Tuple[int, Dict[int, float]]:
    """Automatically select the optimal number of clusters using silhouette score."""
    embeddings = np.vstack(df[embedding_col].values)
    embeddings_norm = normalize(embeddings)
    n_samples = embeddings_norm.shape[0]

    upper_bound = min(max_clusters, n_samples - 1)
    if upper_bound < 2 or upper_bound < min_clusters:
        raise ValueError("No hay suficientes páginas para evaluar el rango de clusters indicado.")

    best_k: Optional[int] = None
    best_score = -1.0
    scores: Dict[int, float] = {}

    for n_clusters in range(min_clusters, upper_bound + 1):
        if n_clusters < 2:
            continue
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(embeddings_norm)
        unique_labels = np.unique(labels)
        if unique_labels.size < 2:
            score = -1.0
        else:
            score = float(silhouette_score(embeddings_norm, labels, metric="cosine"))
        scores[n_clusters] = score
        if score > best_score:
            best_score = score
            best_k = n_clusters

    if best_k is None:
        raise ValueError("No se pudo determinar un número óptimo de clusters con el rango indicado.")

    return best_k, scores


def cluster_pages(
    df: pd.DataFrame,
    n_clusters: int,
    url_column: str,
    embedding_col: str = "EmbeddingsFloat",
) -> Tuple[pd.DataFrame, Dict[int, str], KMeans, np.ndarray]:
    """Cluster pages using K-means and assign descriptive names to clusters."""
    embeddings = np.vstack(df[embedding_col].values)
    embeddings_norm = normalize(embeddings)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = model.fit_predict(embeddings_norm)

    df_clustered = df.copy()
    df_clustered["Cluster"] = cluster_labels

    centroids = model.cluster_centers_
    cluster_names: Dict[int, str] = {}
    general_themes = [
        "Información General",
        "Servicios",
        "Productos",
        "Recursos",
        "Documentación",
        "Ayuda",
        "Categoría Principal",
        "Sección Técnica",
        "Contenido Especializado",
        "Información Legal",
        "Área de Usuario",
        "Novedades",
    ]
    stopwords = {
        "com",
        "org",
        "net",
        "html",
        "php",
        "index",
        "default",
        "es",
        "en",
        "ca",
        "mx",
        "cl",
        "ar",
        "co",
    }

    for cluster_id in range(n_clusters):
        indices = np.where(cluster_labels == cluster_id)[0]
        if len(indices) == 0:
            cluster_names[cluster_id] = f"Cluster vacío {cluster_id}"
            continue

        distances = cdist([centroids[cluster_id]], embeddings_norm[indices], "cosine")[0]
        top_indices_local = indices[np.argsort(distances)[: min(5, len(indices))]]

        words: Counter = Counter()
        for idx in top_indices_local:
            words.update(word for word in extract_words_from_url(df.iloc[idx][url_column]) if word not in stopwords)

        if words:
            top_words = [word.capitalize() for word, _ in words.most_common(3)]
            cluster_names[cluster_id] = " ".join(top_words)
        else:
            cluster_names[cluster_id] = f"{general_themes[cluster_id % len(general_themes)]} {cluster_id}"

    df_clustered["Nombre_Cluster"] = df_clustered["Cluster"].map(cluster_names)
    return df_clustered, cluster_names, model, embeddings_norm


def tsne_visualisation(df: pd.DataFrame, embeddings_norm: np.ndarray, cluster_names: Dict[int, str]) -> plt.Figure:
    """Create a t-SNE visualization of the clustered pages."""
    n_samples = embeddings_norm.shape[0]
    if n_samples < 2:
        raise ValueError("Se requieren al menos dos páginas para visualizar t-SNE.")

    perplexity = min(30, max(5, n_samples // 2))
    if perplexity >= n_samples:
        perplexity = max(2, n_samples - 1)
    if perplexity < 2:
        raise ValueError("Añade más páginas para generar la visualización t-SNE (se requieren al menos 3).")

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings_norm)

    viz_df = pd.DataFrame(
        {
            "x": embeddings_2d[:, 0],
            "y": embeddings_2d[:, 1],
            "cluster": df["Cluster"],
            "ClusterName": df["Nombre_Cluster"],
            "URL": df[st.session_state["url_column"]],
        }
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    palette = sns.color_palette("husl", len(cluster_names))
    sns.scatterplot(
        data=viz_df,
        x="x",
        y="y",
        hue="cluster",
        palette=palette,
        s=120,
        alpha=0.8,
        ax=ax,
        legend=False,
    )

    for cluster_id, name in cluster_names.items():
        cluster_points = viz_df[viz_df["cluster"] == cluster_id]
        if cluster_points.empty:
            continue
        centroid_x = cluster_points["x"].mean()
        centroid_y = cluster_points["y"].mean()
        ax.text(
            centroid_x,
            centroid_y,
            name,
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.75),
        )

    ax.set_title("Clusters de páginas (t-SNE)", fontsize=16)
    ax.set_xlabel("Dimensión t-SNE 1")
    ax.set_ylabel("Dimensión t-SNE 2")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def generate_topic_cluster_graph(
    df: pd.DataFrame,
    cluster_names: Dict[int, str],
    model: KMeans,
    url_column: str,
    min_cluster_similarity: float = 70.0,
) -> str:
    """Generate an interactive graph visualization of topic clusters."""
    graph = nx.Graph()
    cluster_counts = df["Cluster"].value_counts()

    for cluster_id, name in cluster_names.items():
        size = int(cluster_counts.get(cluster_id, 0))
        graph.add_node(
            f"cluster_{cluster_id}",
            label=name,
            title=f"{name}<br/>Cluster {cluster_id}<br/>{size} páginas",
            size=25 + size * 2,
            group=f"cluster_{cluster_id}",
        )

    for idx, row in df.iterrows():
        node_id = f"page_{idx}"
        label = row[url_column]
        short_label = label if len(label) <= 42 else f"{label[:39]}…"
        graph.add_node(
            node_id,
            label=short_label,
            title=label,
            size=10,
            group=f"cluster_{row['Cluster']}",
        )
        graph.add_edge(f"cluster_{row['Cluster']}", node_id, weight=1)

    centroids = model.cluster_centers_
    centroid_similarity = cosine_similarity(centroids) * 100
    n_clusters = len(centroids)
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            score = centroid_similarity[i, j]
            if score >= min_cluster_similarity:
                graph.add_edge(
                    f"cluster_{i}",
                    f"cluster_{j}",
                    weight=max(score / 10, 1),
                    title=f"Similitud clusters: {score:.1f}%",
                )

    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#333333")
    net.from_nx(graph)
    net.repulsion(node_distance=200, spring_length=200, damping=0.85)
    net.toggle_physics(True)
    return net.generate_html(notebook=False)


def ensure_openai_key(input_key: str) -> str:
    """Ensure OpenAI API key is available."""
    if input_key:
        return input_key.strip()
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    raise ValueError("No se proporcionó la clave de OpenAI. Ingrésala en el campo de texto o establece la variable OPENAI_API_KEY.")


def render_csv_workflow():
    """Main rendering function for the CSV workflow section."""
    card = bordered_container()
    with card:
        st.markdown("### Carga de Datos")
        st.caption("Sube tu dataset con embeddings y define la columna vectorial antes de habilitar los análisis avanzados.")
        uploaded_file = st.file_uploader(
            "Archivo CSV o Excel con embeddings",
            type=["csv", "xlsx", "xls"],
            label_visibility="collapsed",
        )
        st.caption("Arrastra y suelta tu archivo en el área o presiona el botón para explorarlo (max. 200 MB).")
        extra_cols = st.columns(3)
        extra_cols[0].markdown("✅ **Formato sugerido:** columnas con URL, tipo, embeddings.")
        extra_cols[1].markdown("📄 **Ejemplo de archivo:** `embeddings_site.xlsx`.")
        extra_cols[2].markdown("🧭 **Tip:** asegúrate de que todos los embeddings tengan el mismo tamaño.")

    if uploaded_file:
        try:
            with st.spinner("Leyendo archivo..."):
                file_name = (uploaded_file.name or "").lower()
                if file_name.endswith(".csv"):
                    site_df = pd.read_csv(uploaded_file)
                else:
                    site_df = pd.read_excel(uploaded_file)
        except Exception as exc:
            st.error(f"No se pudo leer el archivo proporcionado: {exc}")
            return

        # ── Verificar límite de filas en modo trial ──
        original_row_count = len(site_df)
        limit_msg = get_csv_limit_message(original_row_count)
        if limit_msg:
            st.warning(limit_msg)
            site_df = site_df.head(100)
            st.info(f"📊 Se procesarán las primeras 100 filas de {original_row_count} totales.")

        # ── Auto-detección de formato Screaming Frog ──
        if _detect_screaming_frog(site_df):
            st.info(
                "Formato Screaming Frog detectado. "
                "Columnas renombradas automaticamente (Address → url, Word Count → word_count, etc.)."
            )
            site_df = _normalize_screaming_frog(site_df)

        st.markdown("#### Vista previa de datos")
        st.dataframe(site_df.head(), use_container_width=True)

        candidate_embedding_cols = detect_embedding_columns(site_df)
        embedding_col = st.selectbox(
            "Selecciona la columna de embeddings",
            options=candidate_embedding_cols or site_df.columns.tolist(),
        )

        # ── Detección de columna URL (antes del procesamiento para permitir incrementalidad) ──
        url_cols = detect_url_columns(site_df)
        url_column = st.selectbox("Selecciona la columna de URL", options=url_cols, index=0)

        force_reprocess = st.checkbox("Forzar reprocesamiento completo", value=False)

        # ── Obtener DataOrchestrator si hay proyecto activo ──
        orchestrator = None
        proj_cfg = st.session_state.get("project_config")
        if proj_cfg and proj_cfg.get("db_path"):
            try:
                orchestrator = DataOrchestrator(proj_cfg["db_path"])
            except Exception:
                orchestrator = None

        # ── Procesamiento incremental de embeddings ──
        # Prioridad: 1) DB persistente, 2) session_state, 3) reprocesar todo
        current_urls = set(site_df[url_column].astype(str).str.strip())

        # Check DB cache
        db_cached_urls: set = set()
        if orchestrator and not force_reprocess:
            try:
                db_cached_urls = orchestrator.get_cached_urls(_CSV_MODEL_ID) & current_urls
            except Exception:
                db_cached_urls = set()

        # Check session_state cache
        prev_df = st.session_state.get("processed_df")
        prev_url_col = st.session_state.get("url_column")
        ss_cached_urls: set = set()
        if (
            not force_reprocess
            and prev_df is not None
            and prev_url_col == url_column
            and "EmbeddingsFloat" in prev_df.columns
        ):
            ss_cached_urls = set(prev_df[prev_url_col].astype(str).str.strip()) & current_urls

        all_cached_urls = db_cached_urls | ss_cached_urls
        new_urls = current_urls - all_cached_urls

        if force_reprocess and orchestrator:
            try:
                orchestrator.delete_embeddings_by_model(_CSV_MODEL_ID)
            except Exception:
                pass

        try:
            info_messages: list = []
            parts: list = []

            # 1) Restore from DB (URLs not in session but in DB)
            db_only_urls = db_cached_urls - ss_cached_urls
            if db_only_urls and orchestrator:
                db_url_list = list(db_only_urls)
                db_urls_ret, db_vectors = orchestrator.get_embedding_vectors(_CSV_MODEL_ID, db_url_list)
                if len(db_urls_ret) > 0:
                    db_rows = site_df[site_df[url_column].astype(str).str.strip().isin(set(db_urls_ret))].copy()
                    url_to_vec = dict(zip(db_urls_ret, db_vectors))
                    db_rows["EmbeddingsFloat"] = db_rows[url_column].astype(str).str.strip().map(url_to_vec)
                    db_rows = db_rows[db_rows["EmbeddingsFloat"].notna()]
                    parts.append(db_rows)

            # 2) Reuse from session_state
            if ss_cached_urls:
                ss_rows = prev_df[prev_df[prev_url_col].astype(str).str.strip().isin(ss_cached_urls)]
                parts.append(ss_rows)

            # 3) Process new URLs
            if new_urls:
                new_rows = site_df[site_df[url_column].astype(str).str.strip().isin(new_urls)]
                new_processed, info_messages = preprocess_embeddings(new_rows, embedding_col)
                parts.append(new_processed)

                # Persist new embeddings to DB
                if orchestrator and not new_processed.empty and "EmbeddingsFloat" in new_processed.columns:
                    for _, row in new_processed.iterrows():
                        vec = row.get("EmbeddingsFloat")
                        url_val = str(row[url_column]).strip()
                        if vec is not None and url_val:
                            try:
                                orchestrator.save_embeddings(url_val, np.array(vec, dtype=np.float32), _CSV_MODEL_ID)
                            except Exception:
                                pass

            if parts:
                processed_df = pd.concat(parts, ignore_index=True)
            else:
                processed_df, info_messages = preprocess_embeddings(site_df, embedding_col)
                # Persist all to DB
                if orchestrator and "EmbeddingsFloat" in processed_df.columns:
                    for _, row in processed_df.iterrows():
                        vec = row.get("EmbeddingsFloat")
                        url_val = str(row[url_column]).strip()
                        if vec is not None and url_val:
                            try:
                                orchestrator.save_embeddings(url_val, np.array(vec, dtype=np.float32), _CSV_MODEL_ID)
                            except Exception:
                                pass

            n_from_db = len(db_only_urls & current_urls) if db_only_urls else 0
            n_from_ss = len(ss_cached_urls)
            n_new = len(new_urls)
            if n_from_db > 0 or n_from_ss > 0:
                st.success(
                    f"Incremental: {n_from_db + n_from_ss} URLs reutilizadas "
                    f"({n_from_db} de BD, {n_from_ss} de sesion), "
                    f"{n_new} nuevas. {len(processed_df)} filas listas."
                )
            else:
                st.success(f"Embeddings procesados correctamente. {len(processed_df)} filas listas.")
            for msg in info_messages:
                st.info(msg)
        except ValueError as exc:
            st.error(str(exc))
            return

        processed_df[url_column] = processed_df[url_column].astype(str).str.strip()

        st.session_state["raw_df"] = site_df
        st.session_state["processed_df"] = processed_df
        st.session_state["url_column"] = url_column
        st.session_state["cluster_result"] = None
        st.session_state["tsne_figure"] = None
        st.session_state["topic_graph_html"] = None
        st.session_state["knowledge_graph_html"] = None
        st.session_state["knowledge_entities"] = None
        st.session_state["knowledge_doc_relations"] = None
        st.session_state["knowledge_spo_relations"] = None
        st.session_state["knowledge_sds"] = None
        st.session_state["auto_cluster_scores"] = None
        st.session_state["auto_cluster_best_k"] = None
        st.session_state["page_type_column"] = None
        st.session_state["semantic_links_report"] = None
        st.session_state["semantic_links_adv_report"] = None
        st.session_state["semantic_links_adv_orphans"] = None

    processed_df = st.session_state.get("processed_df")
    url_column = st.session_state.get("url_column")

    if processed_df is None or url_column is None:
        st.warning("Sube y procesa un archivo para habilitar el resto de funcionalidades.")
        return

    st.divider()
    st.subheader("2️⃣ Análisis de similitud entre páginas")
    st.caption("Compara cada URL con el resto para encontrar sus páginas gemelas y generar matrices filtradas por similitud.")
    col_select, col_topn = st.columns([3, 1])
    with col_select:
        target_url = st.selectbox("Selecciona la URL objetivo", options=processed_df[url_column].tolist())
    with col_topn:
        top_n = st.number_input("Top N", min_value=3, max_value=50, value=10, step=1)

    if st.button("Calcular páginas más similares", type="primary"):
        with st.spinner("Calculando similitudes..."):
            results = compute_similar_pages(processed_df, target_url, url_column, top_n)
        if results:
            df_similar = pd.DataFrame(results, columns=["URL", "Similitud (%)"])
            st.dataframe(df_similar)
            download_dataframe_button(df_similar, "similares.csv", "Descargar resultados CSV")
        else:
            st.info("No se encontraron páginas similares o la URL no existe.")

    with st.expander("Opcional: generar matriz completa de similitud"):
        threshold = st.slider("Umbral mínimo de similitud (%)", min_value=0, max_value=100, value=70, step=1)
        max_pairs = st.number_input(
            "Máximo de pares a conservar (0 para ilimitado)",
            min_value=0,
            value=100000,
            step=1000,
        )
        if st.button("Construir matriz filtrada"):
            with st.spinner("Calculando matriz de similitud..."):
                matrix_df = build_similarity_matrix(
                    processed_df,
                    url_column,
                    similarity_threshold=threshold,
                    max_results=None if max_pairs == 0 else int(max_pairs),
                )
            if matrix_df.empty:
                st.info("No hay pares que cumplan el umbral establecido.")
            else:
                st.dataframe(matrix_df.head(100))
                st.caption(f"{len(matrix_df)} pares generados.")
                download_dataframe_button(matrix_df, "matriz_similitud.xlsx", "Descargar matriz Excel")

    st.info(
        "¿Buscas recomendaciones de enlazado interno? Abre el **Laboratorio de enlazado** desde la pantalla principal "
        "para acceder a los modos básico, avanzado, híbrido CLS y estructural."
    )

    st.divider()
    st.subheader("3️⃣ Clustering de páginas")
    st.caption("Agrupa las URLs en clusters temáticos, evalúa silhouette y genera visualizaciones t-SNE y grafos interactivos.")

    # Sistema de ayuda contextual
    from app_sections.help_ui import show_help_section
    show_help_section("clustering", expanded=False)

    num_pages = len(processed_df)
    if num_pages < 3:
        st.info("Se necesitan al menos 3 páginas para realizar clustering.")
    else:
        max_clusters_allowed = min(25, num_pages)
        cluster_mode = st.radio(
            "Modo de selección de clusters",
            options=["Automática", "Manual"],
            index=0,
            horizontal=True,
            key="cluster_mode_selector",
        )

        if cluster_mode == "Manual":
            default_clusters = min(10, max(3, int(math.sqrt(num_pages))))
            default_clusters = min(default_clusters, max_clusters_allowed)
            n_clusters = st.slider(
                "Número de clusters",
                min_value=2,
                max_value=max_clusters_allowed,
                value=default_clusters,
                key="manual_cluster_slider",
            )
            if st.button("Ejecutar clustering", key="manual_cluster_button"):
                with st.spinner("Agrupando páginas..."):
                    df_clustered, cluster_names, model, embeddings_norm = cluster_pages(
                        processed_df, int(n_clusters), url_column
                    )
                st.session_state["cluster_result"] = (df_clustered, cluster_names, model, embeddings_norm)
                st.session_state["auto_cluster_scores"] = None
                st.session_state["auto_cluster_best_k"] = None
                st.success("Clustering completado.")
        else:
            auto_upper_bound = min(15, max_clusters_allowed, num_pages - 1)
            if auto_upper_bound <= 2:
                st.info("Con las páginas disponibles solo es posible evaluar 2 clusters.")
                auto_max_clusters = 2
            else:
                auto_max_clusters = st.slider(
                    "Máximo de clusters a evaluar",
                    min_value=2,
                    max_value=auto_upper_bound,
                    value=min(8, auto_upper_bound),
                    help="Se usa la métrica silhouette para elegir el mejor número entre 2 y el máximo indicado.",
                    key="auto_cluster_slider",
                )
            if st.button("Calcular y agrupar automáticamente", key="auto_cluster_button"):
                try:
                    best_k, score_map = auto_select_cluster_count(
                        processed_df,
                        min_clusters=2,
                        max_clusters=int(auto_max_clusters),
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    scores_df = (
                        pd.DataFrame(
                            {"Clusters": list(score_map.keys()), "Silhouette": list(score_map.values())}
                        )
                        .sort_values("Clusters")
                        .reset_index(drop=True)
                    )
                    scores_df["Silhouette"] = scores_df["Silhouette"].round(4)
                    scores_df["Mejor"] = np.where(scores_df["Clusters"] == best_k, "★", "")
                    st.session_state["auto_cluster_scores"] = scores_df
                    st.session_state["auto_cluster_best_k"] = best_k
                    best_score = score_map.get(best_k, float("nan"))
                    with st.spinner(f"Agrupando páginas con {best_k} clusters..."):
                        df_clustered, cluster_names, model, embeddings_norm = cluster_pages(
                            processed_df, best_k, url_column
                        )
                    st.session_state["cluster_result"] = (df_clustered, cluster_names, model, embeddings_norm)
                    st.success(
                        f"Clustering completado con {best_k} clusters."
                        + (f" (silhouette={best_score:.3f})" if not np.isnan(best_score) else "")
                    )

        if st.session_state["cluster_result"]:
            df_clustered, cluster_names, model, embeddings_norm = st.session_state["cluster_result"]
            st.dataframe(
                df_clustered[[url_column, "Cluster", "Nombre_Cluster"]].sort_values("Cluster"), use_container_width=True
            )

            auto_scores_df = st.session_state.get("auto_cluster_scores")
            if isinstance(auto_scores_df, pd.DataFrame) and not auto_scores_df.empty:
                st.markdown("**Evaluación silhouette por número de clusters**")
                st.dataframe(auto_scores_df, use_container_width=True)

            if st.button("Generar visualización t-SNE"):
                try:
                    fig = tsne_visualisation(df_clustered, embeddings_norm, cluster_names)
                    st.pyplot(fig, clear_figure=True)
                    st.session_state["tsne_figure"] = fig
                except ValueError as exc:
                    st.error(str(exc))

            if st.button("Generar grafo de topic clusters"):
                with st.spinner("Construyendo grafo interactivo..."):
                    graph_html = generate_topic_cluster_graph(
                        df_clustered,
                        cluster_names,
                        model,
                        url_column=url_column,
                    )
                st.session_state["topic_graph_html"] = graph_html

            download_dataframe_button(df_clustered, "paginas_clusterizadas.xlsx", "Descargar clustering Excel")

            if st.session_state["tsne_figure"]:
                buffer = io.BytesIO()
                st.session_state["tsne_figure"].savefig(buffer, format="png", dpi=300, bbox_inches="tight")
                buffer.seek(0)
                st.download_button(
                    "Descargar visualización PNG",
                    data=buffer,
                    file_name="clusters_tsne.png",
                    mime="image/png",
                )

            if st.session_state["topic_graph_html"]:
                components.html(st.session_state["topic_graph_html"], height=620, scrolling=True)
                st.download_button(
                    "Descargar grafo HTML",
                    data=st.session_state["topic_graph_html"].encode("utf-8"),
                    file_name="topic_clusters_graph.html",
                    mime="text/html",
                )

    # ══════════════════════════════════════════════════════════════
    # Section: Thin / Duplicate Content Detection
    # ══════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("4️⃣ Detección de contenido duplicado / thin")
    st.caption(
        "Identifica páginas con contenido casi idéntico (duplicado) o demasiado corto (thin) "
        "usando la similitud de embeddings."
    )

    with st.expander("⚙️ Configuración de umbrales", expanded=False):
        col_dup, col_near, col_thin = st.columns(3)
        dup_threshold = col_dup.slider("Duplicado (% similitud)", 80, 100, 90, key="thin_dup_th")
        near_threshold = col_near.slider("Near-duplicate (% similitud)", 60, 95, 80, key="thin_near_th")
        thin_word_limit = col_thin.number_input("Thin: max palabras", min_value=50, max_value=1000, value=300, key="thin_wc")

    if st.button("🔍 Analizar contenido duplicado / thin", key="thin_run", type="primary"):
        with st.spinner("Calculando matriz de similitud..."):
            sim_df = build_similarity_matrix(
                processed_df, url_column,
                similarity_threshold=near_threshold,
                max_results=None,
            )

        duplicates = sim_df[sim_df["Similitud"] >= dup_threshold].copy()
        near_dups = sim_df[
            (sim_df["Similitud"] >= near_threshold) & (sim_df["Similitud"] < dup_threshold)
        ].copy()

        st.session_state["thin_duplicates"] = duplicates
        st.session_state["thin_near_dups"] = near_dups

        # Thin content detection (by word count if column available)
        wc_candidates = [c for c in processed_df.columns if "word" in c.lower() and "count" in c.lower()]
        thin_pages = pd.DataFrame()
        if wc_candidates:
            wc_col = wc_candidates[0]
            thin_pages = processed_df[processed_df[wc_col].fillna(0).astype(float) < thin_word_limit][
                [url_column, wc_col]
            ].copy()
            thin_pages = thin_pages.sort_values(wc_col)
        st.session_state["thin_pages"] = thin_pages
        st.session_state["thin_wc_col"] = wc_candidates[0] if wc_candidates else None

    # Display results
    duplicates = st.session_state.get("thin_duplicates")
    near_dups = st.session_state.get("thin_near_dups")
    thin_pages = st.session_state.get("thin_pages")

    if duplicates is not None:
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Pares duplicados", len(duplicates))
        col_m2.metric("Near-duplicates", len(near_dups) if near_dups is not None else 0)
        col_m3.metric(
            "Páginas thin",
            len(thin_pages) if thin_pages is not None and not thin_pages.empty else 0,
        )

        if not duplicates.empty:
            st.markdown("##### Pares duplicados (>= {}%)".format(
                st.session_state.get("thin_dup_th", 90)
            ))
            st.dataframe(duplicates, use_container_width=True, hide_index=True)

        if near_dups is not None and not near_dups.empty:
            st.markdown("##### Near-duplicates ({}-{}%)".format(
                st.session_state.get("thin_near_th", 80),
                st.session_state.get("thin_dup_th", 90),
            ))
            st.dataframe(near_dups, use_container_width=True, hide_index=True)

        if thin_pages is not None and not thin_pages.empty:
            wc_col = st.session_state.get("thin_wc_col", "Word Count")
            st.markdown(f"##### Páginas thin (< {st.session_state.get('thin_wc', 300)} palabras)")
            st.dataframe(thin_pages, use_container_width=True, hide_index=True)

        # Export
        export_parts = []
        if not duplicates.empty:
            export_parts.append(("Duplicados", duplicates))
        if near_dups is not None and not near_dups.empty:
            export_parts.append(("Near-duplicates", near_dups))
        if thin_pages is not None and not thin_pages.empty:
            export_parts.append(("Thin content", thin_pages))

        if export_parts:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                for sheet_name, sheet_df in export_parts:
                    sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
            st.download_button(
                "📥 Descargar informe Excel",
                data=buf.getvalue(),
                file_name="thin_duplicate_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="thin_download",
            )

    st.divider()
    st.subheader("5\ufe0f\u20e3 Relevancia de palabras clave (OpenAI)")
    st.caption("Cruza tus keywords con embeddings para determinar qué URLs son más relevantes por query (requiere `OPENAI_API_KEY`).")
    api_key_input = st.text_input("OpenAI API Key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
    keywords_file = st.file_uploader("Archivo Excel con palabras clave", type=["xlsx", "xls"], key="keywords_upload")

    if keywords_file:
        keywords_df = pd.read_excel(keywords_file)
        st.dataframe(keywords_df.head())
        keyword_columns = keywords_df.columns.tolist()
        keyword_col = st.selectbox("Columna de palabras clave", options=keyword_columns)
        keywords = keywords_df[keyword_col].dropna().astype(str).str.strip().unique().tolist()
        st.write(f"{len(keywords)} palabras clave detectadas.")

        top_n_keywords = st.number_input("Top N URLs por palabra clave", min_value=1, max_value=50, value=5)
        min_score = st.slider("Umbral mínimo de relevancia (%)", min_value=0, max_value=100, value=50, step=1)

        if st.button("Analizar relevancia"):
            try:
                api_key = ensure_openai_key(api_key_input)
            except ValueError as exc:
                st.error(str(exc))
            else:
                with st.spinner("Calculando relevancia..."):
                    relevance_df = keyword_relevance(
                        processed_df,
                        keywords,
                        api_key,
                        embedding_col="EmbeddingsFloat",
                        url_column=url_column,
                        top_n=top_n_keywords,
                        min_score=min_score,
                    )
                if "Error" in relevance_df.columns:
                    st.warning("Algunas palabras clave generaron errores. Revisa la tabla.")
                st.dataframe(relevance_df)
                download_dataframe_button(relevance_df, "relevancia_palabras_clave.xlsx", "Descargar relevancia Excel")


    st.divider()
    st.subheader("6️⃣ Grafo de conocimiento y entidades")
    st.caption(
        "Audita tus textos con canonicalización de entidades, QIDs de Wikidata y tripletes SPO para evaluar E-E-A-T y Topic Authority."
    )
    with st.expander("¿Qué incluye este análisis avanzado?"):
        st.markdown(
            "- **Canonicalización**: agrupa alias y pronombres mediante coreferencia opcional.\n"
            "- **Entity linking**: vincula entidades al Knowledge Graph (Wikidata) para enriquecer metadata.\n"
            "- **Prominence score**: pondera las menciones por su rol sintáctico (nsubj, dobj, etc.).\n"
            "- **Tripletes SPO**: detecta relaciones Sujeto-Predicado-Objeto para construir un grafo semántico interno."
        )

    # Sistema de ayuda contextual
    from app_sections.help_ui import show_help_section
    show_help_section("knowledge_graph", expanded=False)

    raw_df = st.session_state.get("raw_df")
    if raw_df is None or raw_df.empty:
        st.info("Sube y prepara un archivo con embeddings para habilitar esta sección.")
    else:
        text_columns = raw_df.select_dtypes(include=["object"]).columns.tolist()
        if not text_columns:
            st.info("No se detectaron columnas de texto en el archivo original.")
        else:
            text_column = st.selectbox("Columna con contenido textual", options=text_columns)
            use_existing_url = st.checkbox(
                "Usar la misma columna de URL seleccionada anteriormente",
                value=st.session_state.get("url_column") in raw_df.columns,
            )
            if use_existing_url and st.session_state.get("url_column") in raw_df.columns:
                knowledge_url_col = st.session_state.get("url_column")
            else:
                candidate_url_cols = ["(ninguna)"] + detect_url_columns(raw_df)
                selected_url_option = st.selectbox("Columna de URL (opcional)", options=candidate_url_cols)
                knowledge_url_col = None if selected_url_option == "(ninguna)" else selected_url_option

            max_rows = min(len(raw_df), 200)
            row_limit = st.slider(
                "Número máximo de filas a procesar",
                min_value=1,
                max_value=max_rows,
                value=min(50, max_rows),
            )
            max_entities = st.slider(
                "Número máximo de entidades en el grafo",
                min_value=10,
                max_value=300,
                value=80,
                step=5,
            )
            include_pages = st.checkbox("Incluir nodos de página en el grafo", value=True)
            max_pages = 0
            if include_pages:
                max_pages = st.slider(
                    "Número máximo de páginas a conectar",
                    min_value=5,
                    max_value=100,
                    value=25,
                    step=5,
                )

            model_name = st.text_input(
                "Modelo de spaCy",
                value="es_core_news_sm",
                help="Ejemplos: 'es_core_news_sm' para español, 'en_core_web_sm' para inglés.",
            )
            st.caption(
                "El modelo debe estar instalado en el entorno. Si aparece un error, ejecuta "
                "'python -m spacy download nombre_del_modelo' antes de volver a intentarlo."
            )

            base_entity_labels = [
                "ORG",
                "PERSON",
                "PRODUCT",
                "GPE",
                "LOC",
                "FAC",
                "EVENT",
                "WORK_OF_ART",
                "LAW",
                "NORP",
                "LANGUAGE",
            ]
            extra_labels = sorted(
                {
                    label
                    for labels in ENTITY_PROFILE_PRESETS.values()
                    for label in labels
                    if label not in base_entity_labels
                }
            )
            entity_label_options = base_entity_labels + extra_labels
            profile_options = ["(Sin plantilla)"] + list(ENTITY_PROFILE_PRESETS.keys())
            stored_profile = st.session_state.get("knowledge_entity_profile_choice", "(Sin plantilla)")
            profile_index = profile_options.index(stored_profile) if stored_profile in profile_options else 0
            profile_choice = st.selectbox(
                "Plantilla de entidades según el sector",
                options=profile_options,
                index=profile_index,
                help="Aplica una lista de entidades sugeridas según el tipo de cliente (clínica, editorial, ecommerce, etc.).",
                key="knowledge_entity_profile_choice",
            )
            preset_labels = ENTITY_PROFILE_PRESETS.get(profile_choice, [])
            if profile_choice != st.session_state.get("knowledge_last_profile_choice"):
                if preset_labels:
                    st.session_state["knowledge_graph_entity_labels"] = preset_labels
                st.session_state["knowledge_last_profile_choice"] = profile_choice
            custom_extra = st.text_input(
                "Etiquetas adicionales (separadas por coma)",
                value=st.session_state.get("knowledge_entity_extra_labels", ""),
                help="Añade etiquetas específicas (por ejemplo DISEASE, MEDICATION, BOOK_SERIES).",
                key="knowledge_entity_extra_labels",
            )
            default_selection = st.session_state.get(
                "knowledge_graph_entity_labels",
                preset_labels or ["ORG", "PRODUCT", "PERSON", "GPE"],
            )
            selected_labels = st.multiselect(
                "Tipos de entidad a incluir",
                options=entity_label_options,
                default=default_selection,
                help="Filtra los tipos de entidades que se mostrarán en el grafo. Si no seleccionas ninguna, se incluirán todas.",
                key="knowledge_graph_entity_labels",
            )
            custom_label_set = {
                label.strip().upper()
                for label in (custom_extra or "").split(",")
                if label and label.strip()
            }
            combined_labels = set(selected_labels)
            if custom_label_set:
                combined_labels.update(custom_label_set)

            min_entity_freq = st.slider(
                "Frecuencia mínima de una entidad para incluirla",
                min_value=1,
                max_value=10,
                value=2,
                help="Una entidad debe aparecer al menos este número de veces para entrar en el grafo.",
            )

            st.markdown("#### 🧹 Filtrado automático de ruido")
            st.info(
                "El sistema filtra automáticamente:\n"
                "- Palabras comunes (gracias, hola, etc, etc.)\n"
                "- Nombres genéricos sin apellido (Juan, María, Alicia)\n"
                "- Signos de puntuación y texto muy corto\n"
                "- Entidades de baja calidad (fechas cortas, números sueltos)"
            )

            col_pipeline = st.columns(2)
            with col_pipeline[0]:
                enable_coref = st.checkbox(
                    "Activar coreferencia (coreferee)",
                    value=False,
                    help="Agrupa pronombres y aliases bajo la misma entidad. Requiere instalar `coreferee`.",
                )
            with col_pipeline[1]:
                enable_linking = st.checkbox(
                    "Activar entity linking (Wikidata)",
                    value=False,
                    help="Necesita `spacy-entity-linker` y la descarga de su KB (~1.3GB).",
                )
            col_perf = st.columns(2)
            with col_perf[0]:
                n_process = st.number_input(
                    "Procesos paralelos (spaCy n_process)",
                    min_value=1,
                    max_value=8,
                    value=1,
                    step=1,
                )
            with col_perf[1]:
                batch_size = st.number_input(
                    "Batch size spaCy",
                    min_value=10,
                    max_value=2000,
                    value=200,
                    step=10,
                )

            # Control manual de entidades (whitelist y blacklist)
            st.markdown("#### 🎯 Control manual de entidades")
            st.caption(
                "Afina los resultados añadiendo entidades prioritarias o excluyendo ruido. "
                "Las entidades prioritarias recibirán un boost de prominence (3x)."
            )

            col_manual_1, col_manual_2 = st.columns(2)

            with col_manual_1:
                manual_entities_input = st.text_area(
                    "Entidades prioritarias (una por línea)",
                    value="",
                    height=120,
                    help="Lista de entidades que DEBEN aparecer con alta prominence. "
                         "Ejemplo: tratamientos médicos, productos específicos, etc.",
                    placeholder="fibromialgia\nartritis reumatoide\nozono terapia\nacupuntura",
                    key="knowledge_manual_entities"
                )

            with col_manual_2:
                default_blacklist = [
                    "página", "sitio web", "artículo", "post", "blog",
                    "usuario", "cliente", "persona",
                    "ver más", "leer más", "más información",
                    "consultas", "centros", "centros encontrados",
                    "pedir", "solicitar", "contactar"
                ]
                blacklist_input = st.text_area(
                    "Entidades a excluir (una por línea)",
                    value="\n".join(default_blacklist),
                    height=120,
                    help="Lista de entidades que serán ignoradas. "
                         "Útil para filtrar ruido como elementos de navegación.",
                    key="knowledge_blacklist_entities"
                )

            # Parsear inputs
            manual_entities_list = [
                line.strip()
                for line in (manual_entities_input or "").split("\n")
                if line.strip()
            ] if manual_entities_input else None

            blacklist_entities_list = [
                line.strip()
                for line in (blacklist_input or "").split("\n")
                if line.strip()
            ] if blacklist_input else None

            spacy_ready = is_spacy_available()
            if not spacy_ready:
                st.warning(
                    "spaCy no está instalado o tiene conflictos binarios. "
                    "Instálalo/ajústalo (por ejemplo `pip install spacy==3.5.4`) antes de ejecutar el análisis."
                )

            if st.button("Generar grafo de conocimiento"):
                try:
                    graph_html, entities_df, doc_relations_df, spo_df, sds_df = generate_knowledge_graph_html_v2(
                        raw_df,
                        text_column=text_column,
                        url_column=knowledge_url_col,
                        model_name=model_name.strip(),
                        row_limit=int(row_limit),
                        max_entities=int(max_entities),
                        min_entity_frequency=int(min_entity_freq),
                        include_pages=include_pages,
                        max_pages=int(max_pages),
                        allowed_entity_labels=combined_labels if combined_labels else None,
                        enable_coref=enable_coref,
                        enable_linking=enable_linking,
                        n_process=int(n_process),
                        batch_size=int(batch_size),
                        manual_entities=manual_entities_list,
                        blacklist_entities=blacklist_entities_list,
                    )
                except RuntimeError as exc:
                    st.error(str(exc))
                except ValueError as exc:
                    st.warning(str(exc))
                else:
                    st.success("Grafo de conocimiento generado correctamente.")
                    st.session_state["knowledge_graph_html"] = graph_html
                    st.session_state["knowledge_entities"] = entities_df
                    st.session_state["knowledge_doc_relations"] = doc_relations_df
                    st.session_state["knowledge_spo_relations"] = spo_df
                    st.session_state["knowledge_sds"] = sds_df
                    try:
                        entity_payload = build_entity_payload_from_doc_relations(doc_relations_df)
                    except Exception:
                        entity_payload = {}
                    if entity_payload:
                        st.session_state["entity_payload_by_url"] = entity_payload
                        processed_df = st.session_state.get("processed_df")
                        url_column_session = st.session_state.get("url_column")
                        helper_col = "EntityPayloadFromGraph"
                        if (
                            isinstance(processed_df, pd.DataFrame)
                            and url_column_session in processed_df.columns
                        ):
                            processed_df[helper_col] = processed_df[url_column_session].astype(str).map(
                                lambda url: entity_payload.get(url.strip()) or {}
                            )
                            st.session_state["processed_df"] = processed_df
                        st.info(
                            "Los resultados de entidades se guardaron y ahora puedes usarlos en el laboratorio "
                            "de enlazado hibrido."
                        )

            if st.session_state.get("knowledge_graph_html"):
                components.html(st.session_state["knowledge_graph_html"], height=640, scrolling=True)
                st.download_button(
                    "Descargar grafo de conocimiento (HTML)",
                    data=st.session_state["knowledge_graph_html"].encode("utf-8"),
                    file_name="knowledge_graph.html",
                    mime="text/html",
                )

            entities_summary = st.session_state.get("knowledge_entities")
            if isinstance(entities_summary, pd.DataFrame) and not entities_summary.empty:
                st.markdown("**Entidades más relevantes**")
                st.dataframe(entities_summary, use_container_width=True)
                download_dataframe_button(
                    entities_summary,
                    "entidades_grafo.xlsx",
                    "Descargar listado de entidades",
                )
                st.markdown("**Enriquecer entidades con Google Enterprise Knowledge Graph**")

                from app_sections.google_kg import ensure_google_kg_api_key, query_google_enterprise_kg

                google_kg_api_key_input = st.text_input(
                    "API key de Google Enterprise KG",
                    value=st.session_state.get("google_kg_api_key", ""),
                    type="password",
                    help="Necesitas habilitar la Enterprise Knowledge Graph Search API en Google Cloud.",
                )
                if google_kg_api_key_input:
                    st.session_state["google_kg_api_key"] = google_kg_api_key_input
                available_entities = (
                    entities_summary["Entidad"].dropna().astype(str).str.strip().unique().tolist()
                )
                default_selection = available_entities[: min(15, len(available_entities))]
                selected_entities = st.multiselect(
                    "Selecciona las entidades detectadas para consultarlas en el KG",
                    options=available_entities,
                    default=default_selection,
                    help="Limita la consulta al subconjunto más relevante de tu grafo.",
                )
                manual_mentions_raw = st.text_area(
                    "Entidades adicionales (una por línea o separadas por ';')",
                    value="",
                    help="Añade consultas personalizadas para cruzar información con Google KG.",
                )
                manual_mentions = parse_line_input(manual_mentions_raw, separators=("\n", ";", ","))
                google_kg_languages_value = st.text_input(
                    "Idiomas preferidos (códigos ISO separados por comas)",
                    value="es,en",
                )
                google_kg_types_value = st.text_input(
                    "Filtrar por tipos (opcional, separados por comas)",
                    value="",
                    help="Ejemplo: Person,Organization,Product. Deja vacío para permitir cualquier tipo.",
                )
                google_kg_limit = st.slider(
                    "Resultados por entidad (Google KG)",
                    min_value=1,
                    max_value=10,
                    value=3,
                )
                merged_mentions: List[str] = []
                seen_mentions: set = set()
                for entry in selected_entities + manual_mentions:
                    cleaned = (entry or "").strip()
                    if not cleaned:
                        continue
                    lowered = cleaned.lower()
                    if lowered in seen_mentions:
                        continue
                    seen_mentions.add(lowered)
                    merged_mentions.append(cleaned)
                st.caption(f"Se enviarán {len(merged_mentions)} entidades (límite interno de 50 por llamada).")
                language_tokens = [token.strip() for token in google_kg_languages_value.split(",") if token.strip()]
                type_tokens = [token.strip() for token in google_kg_types_value.split(",") if token.strip()]
                if st.button(
                    "Consultar Google Knowledge Graph",
                    disabled=not merged_mentions,
                ):
                    try:
                        google_kg_key = ensure_google_kg_api_key(google_kg_api_key_input)
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        with st.spinner("Consultando Enterprise Knowledge Graph..."):
                            google_kg_df = query_google_enterprise_kg(
                                merged_mentions,
                                api_key=google_kg_key,
                                limit=int(google_kg_limit),
                                languages=language_tokens or None,
                                types=type_tokens or None,
                            )
                        st.session_state["google_kg_results"] = google_kg_df
                        if google_kg_df.empty:
                            st.warning("Google KG no devolvió resultados para las consultas enviadas.")
                        else:
                            st.success(
                                f"Se recuperaron {len(google_kg_df)} filas enriquecidas desde Google KG."
                            )
                google_kg_results = st.session_state.get("google_kg_results")
                if isinstance(google_kg_results, pd.DataFrame) and not google_kg_results.empty:
                    st.dataframe(google_kg_results, use_container_width=True)
                    download_dataframe_button(
                        google_kg_results,
                        "google_enterprise_kg.xlsx",
                        "Descargar resultados de Google KG",
                    )
                elif isinstance(google_kg_results, pd.DataFrame):
                    st.info("No hay resultados previos de Google KG para mostrar.")

            doc_relations_summary = st.session_state.get("knowledge_doc_relations")
            if isinstance(doc_relations_summary, pd.DataFrame) and not doc_relations_summary.empty:
                st.markdown("**Cobertura por página y Prominence Score**")
                st.dataframe(doc_relations_summary, use_container_width=True)
                download_dataframe_button(
                    doc_relations_summary,
                    "relaciones_documentos.xlsx",
                    "Descargar cobertura por página",
                )

            spo_summary = st.session_state.get("knowledge_spo_relations")
            if isinstance(spo_summary, pd.DataFrame) and not spo_summary.empty:
                st.markdown("**Tripletes SPO detectados (Sujeto-Predicado-Objeto)**")
                st.dataframe(spo_summary, use_container_width=True)
                download_dataframe_button(
                    spo_summary,
                    "tripletes_spo.xlsx",
                    "Descargar relaciones SPO",
                )

            sds_summary = st.session_state.get("knowledge_sds")
            if isinstance(sds_summary, pd.DataFrame) and not sds_summary.empty:
                st.markdown("---")
                st.markdown("### 📊 Semantic Depth Score (SDS) por Documento")

                # Sistema de ayuda contextual para SDS
                from app_sections.help_ui import show_help_section
                show_help_section("semantic_depth_score", expanded=False)

                st.markdown("""
                El **Semantic Depth Score** mide la calidad del contenido mediante:
                - **Score ER (Entity Relevance)**: Densidad y relevancia de entidades
                - **Score CV (Vector Cohesion)**: Cohesión narrativa del discurso

                **Clasificación:**
                - 🔴 0-33: Contenido Thin o Irrelevante
                - 🟡 34-66: Calidad Decente o Sesgada
                - 🟢 67-100: Calidad Óptima y Relevante
                """)

                # Mostrar DataFrame con estilo
                st.dataframe(
                    sds_summary.style.background_gradient(
                        subset=["SDS (Semantic Depth Score)"],
                        cmap="RdYlGn",
                        vmin=0,
                        vmax=100
                    ),
                    use_container_width=True
                )

                download_dataframe_button(
                    sds_summary,
                    "semantic_depth_scores.xlsx",
                    "Descargar Semantic Depth Scores",
                )

                # Mostrar estadísticas resumidas
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_sds = sds_summary["SDS (Semantic Depth Score)"].mean()
                    st.metric("SDS Promedio", f"{avg_sds:.2f}")
                with col2:
                    avg_er = sds_summary["Score ER (Entity Relevance)"].mean()
                    st.metric("Entity Relevance Promedio", f"{avg_er:.4f}")
                with col3:
                    avg_cv = sds_summary["Score CV (Vector Cohesion)"].mean()
                    st.metric("Vector Cohesion Promedio", f"{avg_cv:.4f}")

                # Distribución de clasificaciones
                if "Clasificación" in sds_summary.columns:
                    st.markdown("**Distribución de Clasificaciones:**")
                    classification_counts = sds_summary["Clasificación"].value_counts()
                    for classification, count in classification_counts.items():
                        percentage = (count / len(sds_summary)) * 100
                        emoji = "🔴" if "Thin" in classification else ("🟡" if "Decente" in classification else "🟢")
                        st.write(f"{emoji} {classification}: {count} documentos ({percentage:.1f}%)")
