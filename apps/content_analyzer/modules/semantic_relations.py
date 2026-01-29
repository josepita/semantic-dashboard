"""
Módulo de Análisis de Relaciones Semánticas entre Keywords.

Permite analizar y visualizar las relaciones semánticas entre múltiples palabras clave
mediante embeddings y diferentes tipos de gráficos interactivos.
"""

from typing import List, Tuple, Dict, Optional
import hashlib
import os
import re
import tempfile

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import networkx as nx
from pyvis.network import Network

# UMAP es opcional (requiere umap-learn)
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def normalize_keywords(keywords_list: List[str]) -> Tuple[List[str], int]:
    """
    Normaliza y limpia keywords eliminando duplicados.
    
    Args:
        keywords_list: Lista de keywords raw
        
    Returns:
        Tupla (keywords_normalizadas, num_duplicados_eliminados)
    """
    normalized = []
    seen = set()
    duplicates = 0
    
    for kw in keywords_list:
        # Convertir a minúsculas para comparación
        kw_lower = kw.lower().strip()
        
        # Eliminar espacios múltiples
        kw_clean = re.sub(r'\s+', ' ', kw_lower)
        
        # Evitar duplicados exactos (case-insensitive)
        if kw_clean not in seen:
            normalized.append(kw.strip())  # Guardar original con capitalización
            seen.add(kw_clean)
        else:
            duplicates += 1
    
    return normalized, duplicates


def suggest_threshold(similarity_df: pd.DataFrame) -> Dict[str, float]:
    """
    Sugiere valores de threshold basados en la distribución de similitudes.
    
    Args:
        similarity_df: DataFrame con matriz de similitud
        
    Returns:
        Diccionario con thresholds sugeridos
    """
    # Obtener triángulo superior (sin diagonal)
    mask = np.triu(np.ones_like(similarity_df), k=1).astype(bool)
    similarities = similarity_df.values[mask]
    
    # Calcular percentiles
    p50 = float(np.percentile(similarities, 50))
    p75 = float(np.percentile(similarities, 75))
    p90 = float(np.percentile(similarities, 90))
    
    return {
        "permisivo": round(p50, 2),
        "balanceado": round(p75, 2),
        "restrictivo": round(p90, 2)
    }



def find_optimal_clusters(
    embeddings: np.ndarray,
    max_clusters: int = 10
) -> Tuple[int, List[Tuple[int, float]]]:
    """
    Encuentra el número óptimo de clusters usando Silhouette Score.

    Args:
        embeddings: Array de embeddings
        max_clusters: Máximo de clusters a probar

    Returns:
        Tupla (n_clusters_óptimo, lista de (n, score))
    """
    max_k = min(max_clusters, len(embeddings) - 1)
    if max_k < 2:
        return 2, [(2, 0.0)]

    scores = []
    for n in range(2, max_k + 1):
        clustering = AgglomerativeClustering(n_clusters=n, linkage='ward')
        labels = clustering.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        scores.append((n, float(score)))

    optimal_n = max(scores, key=lambda x: x[1])[0]
    return optimal_n, scores


def detect_outliers(
    similarity_df: pd.DataFrame,
    threshold: float = 0.3
) -> pd.Series:
    """
    Detecta keywords con baja similitud promedio respecto al resto.

    Args:
        similarity_df: DataFrame con matriz de similitud
        threshold: Umbral por debajo del cual se considera outlier

    Returns:
        Series con similitud promedio de los outliers
    """
    # Excluir la diagonal (similitud consigo misma = 1.0)
    mask = ~np.eye(len(similarity_df), dtype=bool)
    masked_values = similarity_df.values * mask
    row_sums = masked_values.sum(axis=1)
    counts = mask.sum(axis=1)
    avg_similarities = pd.Series(
        row_sums / counts,
        index=similarity_df.index,
        name="Similitud promedio"
    )
    return avg_similarities[avg_similarities < threshold]


@st.cache_data(ttl=3600, show_spinner=False)
def get_embeddings_cached(
    keywords_tuple: Tuple[str, ...],
    model_name: str
) -> np.ndarray:
    """
    Calcula y cachea embeddings para evitar recálculos.

    Args:
        keywords_tuple: Tupla de keywords (hashable para cache)
        model_name: Nombre del modelo

    Returns:
        Array de embeddings
    """
    from semantic_tools import get_sentence_transformer
    model = get_sentence_transformer(model_name)
    return model.encode(list(keywords_tuple), show_progress_bar=False)


def calculate_keyword_similarities(
    keywords: List[str],
    model
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Calcula la matriz de similitud entre keywords.

    Args:
        keywords: Lista de palabras clave
        model: Modelo de embeddings (SentenceTransformer)

    Returns:
        Tupla (embeddings_array, similarity_df)
    """
    # Generar embeddings
    embeddings = model.encode(keywords, show_progress_bar=False)

    # Calcular matriz de similitud coseno
    similarity_matrix = cosine_similarity(embeddings)

    # Crear DataFrame con nombres de keywords
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=keywords,
        columns=keywords
    )

    return embeddings, similarity_df


def render_similarity_heatmap(
    similarity_df: pd.DataFrame,
    title: str = "Matriz de Similitud Semántica"
) -> go.Figure:
    """
    Genera un heatmap de la matriz de similitud.

    Args:
        similarity_df: DataFrame con matriz de similitud
        title: Título del gráfico

    Returns:
        Figura de Plotly
    """
    fig = go.Figure(data=go.Heatmap(
        z=similarity_df.values,
        x=similarity_df.columns,
        y=similarity_df.index,
        colorscale='RdYlGn',
        zmid=0.5,
        text=similarity_df.values.round(3),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(
            title=dict(text="Similitud", side="right")
        ),
        hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>Similitud: %{z:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Keywords",
        yaxis_title="Keywords",
        height=600,
        width=800,
        xaxis={'side': 'bottom'},
        yaxis={'autorange': 'reversed'}
    )

    return fig


def render_network_graph(
    similarity_df: pd.DataFrame,
    threshold: float = 0.5,
    height: str = "600px"
) -> str:
    """
    Genera un grafo de red interactivo con PyVis.

    Args:
        similarity_df: DataFrame con matriz de similitud
        threshold: Umbral mínimo de similitud para mostrar conexión
        height: Altura del grafo

    Returns:
        Path del archivo HTML temporal
    """
    # Crear grafo de NetworkX
    G = nx.Graph()

    # Añadir nodos (keywords)
    keywords = similarity_df.index.tolist()
    for kw in keywords:
        G.add_node(kw)

    # Añadir aristas (similitudes > threshold)
    for i, kw1 in enumerate(keywords):
        for j, kw2 in enumerate(keywords):
            if i < j:  # Evitar duplicados
                similarity = float(similarity_df.loc[kw1, kw2])  # Convertir a Python float
                if similarity >= threshold:
                    G.add_edge(
                        kw1,
                        kw2,
                        weight=similarity,
                        title=f"Similitud: {similarity:.3f}"
                    )

    # Crear visualización con PyVis
    net = Network(height=height, width="100%", bgcolor="#ffffff", font_color="black")
    net.from_nx(G)

    # Configurar física del grafo
    net.set_options("""
    {
      "nodes": {
        "font": {
          "size": 16,
          "face": "arial"
        },
        "size": 25,
        "color": {
          "border": "#2B7CE9",
          "background": "#97C2FC",
          "highlight": {
            "border": "#2B7CE9",
            "background": "#D2E5FF"
          }
        }
      },
      "edges": {
        "color": {
          "inherit": false,
          "color": "#848484",
          "highlight": "#2B7CE9"
        },
        "smooth": {
          "type": "continuous"
        }
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": {
          "iterations": 150
        }
      }
    }
    """)

    # Guardar en archivo temporal
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8')
    net.save_graph(tmpfile.name)
    tmpfile.close()

    return tmpfile.name


def render_2d_visualization(
    embeddings: np.ndarray,
    keywords: List[str],
    method: str = "tsne",
    perplexity: int = 5,
    max_iter: int = 1000,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    sizes: Optional[List[float]] = None,
) -> go.Figure:
    """
    Visualización 2D de embeddings con T-SNE, PCA o UMAP.

    Args:
        embeddings: Array de embeddings
        keywords: Lista de keywords correspondientes
        method: 'tsne', 'pca' o 'umap'
        perplexity: Perplexity para T-SNE
        max_iter: Iteraciones máximas para T-SNE
        umap_n_neighbors: Neighbors para UMAP
        umap_min_dist: Min distance para UMAP
        sizes: Tamaños opcionales por punto (ej. volumen de búsqueda)

    Returns:
        Figura de Plotly
    """
    n_samples = len(keywords)

    # Reducción de dimensionalidad
    if method == "umap" and UMAP_AVAILABLE:
        n_neighbors_adj = min(umap_n_neighbors, n_samples - 1)
        reducer = UMAP(
            n_components=2,
            n_neighbors=max(2, n_neighbors_adj),
            min_dist=umap_min_dist,
            random_state=42,
        )
        coords_2d = reducer.fit_transform(embeddings)
        method_name = "UMAP"
    elif method == "tsne":
        perplexity = min(perplexity, n_samples - 1)
        if n_samples < 4:
            st.warning("T-SNE requiere al menos 4 palabras. Usando PCA en su lugar.")
            method = "pca"
        else:
            reducer = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=42,
                max_iter=max_iter,
            )
            coords_2d = reducer.fit_transform(embeddings)
            method_name = "T-SNE"

    if method == "pca":
        reducer = PCA(n_components=2, random_state=42)
        coords_2d = reducer.fit_transform(embeddings)
        method_name = "PCA"

    # Crear DataFrame para plotly
    df_plot = pd.DataFrame({
        'x': coords_2d[:, 0],
        'y': coords_2d[:, 1],
        'keyword': keywords,
    })

    scatter_kwargs = dict(
        x='x', y='y', text='keyword',
        title=f'Relaciones Semánticas ({method_name})',
        labels={'x': f'{method_name} Dimensión 1', 'y': f'{method_name} Dimensión 2'},
    )

    if sizes is not None:
        df_plot['size'] = sizes
        scatter_kwargs['size'] = 'size'
        scatter_kwargs['size_max'] = 40

    fig = px.scatter(df_plot, **scatter_kwargs)

    marker_opts = dict(color='#97C2FC', line=dict(width=2, color='#2B7CE9'))
    if sizes is None:
        marker_opts['size'] = 15

    fig.update_traces(
        textposition='top center',
        marker=marker_opts,
        textfont=dict(size=12, color='black'),
    )

    fig.update_layout(height=600, width=800, showlegend=False, hovermode='closest')

    return fig


def perform_clustering(
    embeddings: np.ndarray,
    keywords: List[str],
    n_clusters: int = 3
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Agrupa keywords por similitud semántica.

    Args:
        embeddings: Array de embeddings
        keywords: Lista de keywords
        n_clusters: Número de clusters

    Returns:
        Tupla (cluster_labels, results_df)
    """
    # Clustering jerárquico
    clustering = AgglomerativeClustering(
        n_clusters=min(n_clusters, len(keywords)),
        linkage='ward'
    )
    labels = clustering.fit_predict(embeddings)

    # Crear DataFrame de resultados
    results_df = pd.DataFrame({
        'Keyword': keywords,
        'Cluster': labels,
        'Cluster_Name': [f"Grupo {i+1}" for i in labels]
    })

    return labels, results_df


def find_most_similar(
    similarity_df: pd.DataFrame,
    keyword: str,
    top_n: int = 5
) -> pd.DataFrame:
    """
    Encuentra las keywords más similares a una dada.

    Args:
        similarity_df: DataFrame con matriz de similitud
        keyword: Keyword de referencia
        top_n: Número de resultados a retornar

    Returns:
        DataFrame con keywords más similares
    """
    if keyword not in similarity_df.index:
        return pd.DataFrame()

    # Obtener similitudes
    similarities = similarity_df[keyword].copy()

    # Excluir la misma keyword
    similarities = similarities[similarities.index != keyword]

    # Ordenar y tomar top N
    top_similar = similarities.nlargest(top_n)

    # Crear DataFrame de resultados
    results_df = pd.DataFrame({
        'Keyword': top_similar.index,
        'Similitud': top_similar.values
    })

    return results_df


def render_semantic_relations():
    """
    Renderiza la interfaz completa de análisis de relaciones semánticas.
    """
    st.title("🔗 Análisis de Relaciones Semánticas")
    st.markdown("""
    Analiza las relaciones semánticas entre múltiples palabras clave mediante embeddings.
    Visualiza similitudes, conexiones y agrupaciones de forma interactiva.
    """)

    # Imports locales
    from semantic_tools import AVAILABLE_MODELS, MODEL_DESCRIPTIONS

    # Sección de entrada de keywords
    st.header("📝 Configuración")

    col1, col2 = st.columns([2, 1])

    with col1:
        keywords_input = st.text_area(
            "Introduce palabras clave (una por línea)",
            height=150,
            placeholder="marketing digital\nSEO\nSEM\npublicidad online\nredes sociales\ncontent marketing\nemail marketing",
            help="Introduce al menos 3 palabras clave para obtener resultados significativos"
        )

        # Fase 3 - Mejora 6: Cargar CSV con metadata
        with st.expander("📁 Cargar keywords desde CSV con metadata (opcional)", expanded=False):
            st.caption(
                "Sube un CSV/Excel con keywords y datos como volumen de búsqueda o CPC. "
                "Los datos se usarán para ajustar el tamaño de los nodos en las visualizaciones."
            )
            metadata_file = st.file_uploader(
                "Archivo de keywords con metadata",
                type=["csv", "xlsx", "xls"],
                key="sr_metadata_uploader",
                label_visibility="collapsed",
            )
            metadata_kw_col = None
            metadata_size_col = None
            metadata_df = None

            if metadata_file:
                try:
                    if metadata_file.name.lower().endswith(".csv"):
                        metadata_df = pd.read_csv(metadata_file)
                    else:
                        metadata_df = pd.read_excel(metadata_file)

                    st.dataframe(metadata_df.head(), use_container_width=True)

                    m_col1, m_col2 = st.columns(2)
                    with m_col1:
                        metadata_kw_col = st.selectbox(
                            "Columna de keywords",
                            options=list(metadata_df.columns),
                            key="sr_metadata_kw_col",
                        )
                    with m_col2:
                        numeric_cols = ["(Ninguna)"] + [
                            c for c in metadata_df.columns
                            if pd.api.types.is_numeric_dtype(metadata_df[c])
                        ]
                        metadata_size_col_raw = st.selectbox(
                            "Columna de volumen/tamaño (opcional)",
                            options=numeric_cols,
                            key="sr_metadata_size_col",
                        )
                        metadata_size_col = (
                            metadata_size_col_raw if metadata_size_col_raw != "(Ninguna)" else None
                        )

                    if st.button("Usar keywords del archivo", key="sr_apply_metadata"):
                        kws_from_file = metadata_df[metadata_kw_col].dropna().astype(str).tolist()
                        st.session_state["sr_keywords_from_file"] = kws_from_file
                        st.session_state["sr_metadata_df"] = metadata_df
                        st.session_state["sr_metadata_kw_col"] = metadata_kw_col
                        st.session_state["sr_metadata_size_col"] = metadata_size_col
                        st.success(f"{len(kws_from_file)} keywords cargadas desde archivo")
                        st.rerun()
                except Exception as exc:
                    st.error(f"Error al leer archivo: {exc}")

    with col2:
        st.markdown("### ⚙️ Opciones")

        # Selector de modelo
        selected_model_key = st.selectbox(
            "Modelo de embeddings",
            options=list(AVAILABLE_MODELS.keys()),
            index=0,
            format_func=lambda x: MODEL_DESCRIPTIONS[x],
            help="Modelos más grandes = mejor calidad pero más lentos",
        )

        selected_model_name = AVAILABLE_MODELS[selected_model_key]

        similarity_threshold = st.slider(
            "Umbral de similitud (grafo)",
            min_value=0.0, max_value=1.0, value=0.5, step=0.05,
            help="Umbral mínimo para mostrar conexiones en el grafo de red",
        )

        # Fase 3 - Mejora 2: Selector de método con UMAP
        viz_options = ["tsne", "pca"]
        viz_labels = {"tsne": "T-SNE (recomendado)", "pca": "PCA (más rápido)"}
        if UMAP_AVAILABLE:
            viz_options.append("umap")
            viz_labels["umap"] = "UMAP (estructura global)"

        visualization_method = st.selectbox(
            "Método de visualización 2D",
            options=viz_options,
            format_func=lambda x: viz_labels[x],
            help="T-SNE: relaciones locales | PCA: rápido | UMAP: estructura global",
        )

        n_clusters = st.number_input(
            "Número de clusters",
            min_value=2, max_value=10, value=3,
            help="Número de grupos para agrupar keywords similares",
        )

        # Fase 3 - Mejora 2: Parámetros avanzados
        with st.expander("⚙️ Parámetros avanzados de visualización", expanded=False):
            if visualization_method == "tsne":
                adv_perplexity = st.slider("Perplexity", 2, 50, 5, key="sr_perplexity")
                adv_max_iter = st.slider("Iteraciones máx.", 250, 5000, 1000, step=250, key="sr_max_iter")
            elif visualization_method == "umap" and UMAP_AVAILABLE:
                adv_umap_neighbors = st.slider("Neighbors", 2, 50, 15, key="sr_umap_neighbors")
                adv_umap_min_dist = st.slider("Min distance", 0.0, 1.0, 0.1, step=0.05, key="sr_umap_min_dist")

    # Resolver keywords (del text_area o del archivo cargado)
    keywords_from_file = st.session_state.get("sr_keywords_from_file")
    if keywords_from_file:
        keywords_raw = keywords_from_file
    elif keywords_input:
        keywords_raw = [kw.strip() for kw in keywords_input.split('\n') if kw.strip()]
    else:
        keywords_raw = []

    # Procesar keywords
    if keywords_raw:
        # Normalizar y eliminar duplicados
        keywords, duplicates_removed = normalize_keywords(keywords_raw)

        if len(keywords) < 2:
            st.warning("⚠️ Introduce al menos 2 palabras clave para analizar.")
            return

        # Mostrar información de normalización
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.success(f"✅ {len(keywords)} palabras clave únicas detectadas")
        with col_info2:
            if duplicates_removed > 0:
                st.warning(f"⚠️ {duplicates_removed} duplicado(s) eliminado(s)")

        # Fase 2 - Mejora 8: Cache de embeddings
        with st.spinner(f"Calculando embeddings ({selected_model_key})..."):
            embeddings = get_embeddings_cached(tuple(keywords), selected_model_name)

        similarity_matrix = cosine_similarity(embeddings)
        similarity_df = pd.DataFrame(similarity_matrix, index=keywords, columns=keywords)

        # Resolver tamaños desde metadata (volumen de búsqueda)
        sizes = None
        meta_df_session = st.session_state.get("sr_metadata_df")
        meta_kw_col_session = st.session_state.get("sr_metadata_kw_col")
        meta_size_col_session = st.session_state.get("sr_metadata_size_col")

        if meta_df_session is not None and meta_kw_col_session and meta_size_col_session:
            size_map = dict(zip(
                meta_df_session[meta_kw_col_session].astype(str).str.strip().str.lower(),
                meta_df_session[meta_size_col_session],
            ))
            sizes = [float(size_map.get(kw.lower(), 10)) for kw in keywords]
            # Normalizar a rango razonable para visualización
            max_s = max(sizes) if max(sizes) > 0 else 1
            sizes = [max(5, (s / max_s) * 40) for s in sizes]

        # Mostrar sugerencias de threshold
        thresholds = suggest_threshold(similarity_df)
        with st.expander("💡 Thresholds sugeridos según tu dataset", expanded=False):
            th_col1, th_col2, th_col3 = st.columns(3)
            with th_col1:
                st.metric("Permisivo", f"{thresholds['permisivo']:.2f}",
                          help="Muestra más conexiones (percentil 50)")
            with th_col2:
                st.metric("Balanceado", f"{thresholds['balanceado']:.2f}",
                          help="Equilibrio entre conexiones y claridad (percentil 75)")
            with th_col3:
                st.metric("Restrictivo", f"{thresholds['restrictivo']:.2f}",
                          help="Solo conexiones fuertes (percentil 90)")
            st.caption(
                f"Tu umbral actual es **{similarity_threshold:.2f}**. "
                "Ajústalo arriba en ⚙️ Opciones según el nivel de detalle que necesites."
            )

        # Fase 2 - Mejora 7: Detección de outliers
        outliers = detect_outliers(similarity_df, threshold=0.3)
        if len(outliers) > 0:
            with st.expander(f"🔍 {len(outliers)} keyword(s) con baja similitud general (posibles outliers)", expanded=False):
                st.caption("Estas keywords tienen una similitud promedio baja con el resto. "
                           "Podrían no pertenecer al mismo tema o ser demasiado genéricas/específicas.")
                outlier_df = outliers.reset_index()
                outlier_df.columns = ["Keyword", "Similitud promedio"]
                st.dataframe(outlier_df, use_container_width=True, hide_index=True)

        # Tabs para diferentes visualizaciones
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Matriz de Similitud",
            "🕸️ Grafo de Red",
            "📍 Mapa 2D",
            "🗂️ Clusters",
            "🔍 Búsqueda",
            "📋 Tabla de Datos"
        ])

        # Tab 1: Heatmap
        with tab1:
            st.subheader("Matriz de Similitud Semántica")
            st.markdown("""
            **Cómo interpretar:**
            - 🟢 **Verde (≥0.7):** Alta similitud - Conceptos muy relacionados
            - 🟡 **Amarillo (0.4-0.7):** Similitud media - Relacionados moderadamente
            - 🔴 **Rojo (<0.4):** Baja similitud - Conceptos distantes
            """)

            fig_heatmap = render_similarity_heatmap(similarity_df)
            st.plotly_chart(fig_heatmap, use_container_width=True)

            # Estadísticas
            with st.expander("📈 Estadísticas de Similitud"):
                col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)

                # Obtener triángulo superior (sin diagonal)
                mask = np.triu(np.ones_like(similarity_df), k=1).astype(bool)
                similarities = similarity_df.values[mask]

                col_stats1.metric("Similitud Promedio", f"{similarities.mean():.3f}")
                col_stats2.metric("Similitud Máxima", f"{similarities.max():.3f}")
                col_stats3.metric("Similitud Mínima", f"{similarities.min():.3f}")
                col_stats4.metric("Desviación Estándar", f"{similarities.std():.3f}")

                # Distribución
                st.markdown("**Distribución de Similitudes:**")
                fig_dist = px.histogram(
                    x=similarities,
                    nbins=20,
                    labels={'x': 'Similitud', 'y': 'Frecuencia'},
                    title='Distribución de Similitudes entre Keywords'
                )
                st.plotly_chart(fig_dist, use_container_width=True)

        # Tab 2: Grafo de Red
        with tab2:
            st.subheader("Grafo de Red de Relaciones")
            st.markdown(f"""
            **Visualización interactiva de conexiones semánticas**
            - Los nodos representan keywords
            - Las aristas muestran similitud ≥ {similarity_threshold:.2f}
            - Arrastra los nodos para reorganizar el grafo
            """)

            with st.spinner("Generando grafo de red..."):
                try:
                    graph_file = render_network_graph(
                        similarity_df,
                        threshold=similarity_threshold
                    )

                    # Leer y mostrar el HTML
                    with open(graph_file, 'r', encoding='utf-8') as f:
                        html_content = f.read()

                    st.components.v1.html(html_content, height=650)

                    # Limpiar archivo temporal
                    try:
                        os.unlink(graph_file)
                    except:
                        pass

                except Exception as e:
                    st.error(f"Error al generar el grafo: {str(e)}")
                    st.info("Prueba a reducir el número de keywords o ajustar el umbral de similitud.")

        # Tab 3: Visualización 2D
        with tab3:
            st.subheader("Mapa 2D de Relaciones Semánticas")
            st.markdown("""
            **Visualización espacial de similitudes**
            - Keywords cercanas son semánticamente similares
            - La distancia representa diferencia semántica
            """)

            viz_kwargs = dict(
                embeddings=embeddings,
                keywords=keywords,
                method=visualization_method,
                sizes=sizes,
            )
            if visualization_method == "tsne":
                viz_kwargs["perplexity"] = st.session_state.get("sr_perplexity", 5)
                viz_kwargs["max_iter"] = st.session_state.get("sr_max_iter", 1000)
            elif visualization_method == "umap" and UMAP_AVAILABLE:
                viz_kwargs["umap_n_neighbors"] = st.session_state.get("sr_umap_neighbors", 15)
                viz_kwargs["umap_min_dist"] = st.session_state.get("sr_umap_min_dist", 0.1)

            with st.spinner(f"Calculando proyección {visualization_method.upper()}..."):
                fig_2d = render_2d_visualization(**viz_kwargs)

            st.plotly_chart(fig_2d, use_container_width=True)

            methods_info = {
                "tsne": "**T-SNE:** Preserva mejor las relaciones locales (recomendado para visualización)",
                "pca": "**PCA:** Más rápido, preserva mejor la varianza global",
                "umap": "**UMAP:** Preserva estructura local y global, bueno para datasets grandes",
            }
            st.info(f"**Método usado:** {visualization_method.upper()}\n\n"
                    + "\n\n".join(f"- {v}" for v in methods_info.values()))

        # Tab 4: Clustering
        with tab4:
            st.subheader("Agrupación por Similitud Semántica")

            # Fase 2 - Mejora 4: Auto-detección de clusters
            auto_detect = st.checkbox(
                "Auto-detectar número óptimo de clusters",
                value=False,
                key="sr_auto_clusters",
            )

            effective_n_clusters = n_clusters
            if auto_detect and len(keywords) >= 4:
                with st.spinner("Buscando número óptimo de clusters..."):
                    optimal_n, cluster_scores = find_optimal_clusters(embeddings, max_clusters=min(10, len(keywords) - 1))

                effective_n_clusters = optimal_n
                st.success(f"Número óptimo detectado: **{optimal_n} clusters** (Silhouette Score más alto)")

                # Gráfico de scores
                fig_scores = px.line(
                    x=[s[0] for s in cluster_scores],
                    y=[s[1] for s in cluster_scores],
                    labels={'x': 'Número de clusters', 'y': 'Silhouette Score'},
                    title='Calidad del Clustering por Número de Grupos',
                    markers=True,
                )
                # Marcar el óptimo
                fig_scores.add_vline(x=optimal_n, line_dash="dash", line_color="green",
                                     annotation_text=f"Óptimo: {optimal_n}")
                st.plotly_chart(fig_scores, use_container_width=True)
            elif auto_detect:
                st.warning("Se necesitan al menos 4 keywords para la auto-detección.")

            with st.spinner("Agrupando keywords..."):
                labels, cluster_df = perform_clustering(embeddings, keywords, effective_n_clusters)

            # Silhouette score del clustering actual
            if len(set(labels)) > 1:
                current_silhouette = silhouette_score(embeddings, labels)
                st.metric("Silhouette Score (calidad)", f"{current_silhouette:.3f}",
                          help="Valores cercanos a 1 = clusters bien separados. "
                               "Valores cercanos a 0 = clusters solapados.")

            # Mostrar tabla de clusters
            st.dataframe(
                cluster_df.sort_values('Cluster'),
                use_container_width=True,
                hide_index=True
            )

            # Visualización de clusters en 2D
            st.markdown("### Visualización de Clusters")

            # Reducir dimensionalidad
            if len(keywords) >= 4 and visualization_method == "tsne":
                perp = min(5, len(keywords) - 1)
                reducer = TSNE(n_components=2, perplexity=perp, random_state=42)
                coords_2d = reducer.fit_transform(embeddings)
            elif visualization_method == "umap" and UMAP_AVAILABLE and len(keywords) >= 4:
                n_neigh = min(15, len(keywords) - 1)
                reducer = UMAP(n_components=2, n_neighbors=max(2, n_neigh), random_state=42)
                coords_2d = reducer.fit_transform(embeddings)
            else:
                reducer = PCA(n_components=2, random_state=42)
                coords_2d = reducer.fit_transform(embeddings)

            # Crear scatter plot con clusters
            df_cluster_plot = pd.DataFrame({
                'x': coords_2d[:, 0],
                'y': coords_2d[:, 1],
                'keyword': keywords,
                'cluster': [f"Grupo {i+1}" for i in labels]
            })

            fig_clusters = px.scatter(
                df_cluster_plot,
                x='x',
                y='y',
                color='cluster',
                text='keyword',
                title='Keywords Agrupadas por Similitud Semántica',
                color_discrete_sequence=px.colors.qualitative.Set2
            )

            fig_clusters.update_traces(
                textposition='top center',
                marker=dict(size=15, line=dict(width=2, color='white')),
                textfont=dict(size=11)
            )

            fig_clusters.update_layout(
                height=600,
                showlegend=True,
                plot_bgcolor='#F8F9FA',  # Gris muy claro
                paper_bgcolor='#FFFFFF',  # Blanco para el papel
                xaxis=dict(
                    showgrid=True,
                    gridcolor='#E0E0E0',  # Grid un poco más visible
                    zeroline=True,
                    zerolinecolor='#BDBDBD'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='#E0E0E0',
                    zeroline=True,
                    zerolinecolor='#BDBDBD'
                )
            )
            st.plotly_chart(fig_clusters, use_container_width=True)

            # Resumen por cluster
            with st.expander("📋 Resumen por Cluster"):
                for cluster_id in sorted(cluster_df['Cluster'].unique()):
                    cluster_kws = cluster_df[cluster_df['Cluster'] == cluster_id]['Keyword'].tolist()
                    st.markdown(f"**Grupo {cluster_id + 1}** ({len(cluster_kws)} keywords):")
                    st.write(", ".join(cluster_kws))

        # Tab 5: Búsqueda de similares
        with tab5:
            st.subheader("🔍 Buscar Keywords Similares")

            selected_keyword = st.selectbox(
                "Selecciona una keyword:",
                options=keywords,
                help="Encuentra las keywords más similares a la seleccionada"
            )

            top_n = st.slider(
                "Número de resultados",
                min_value=1,
                max_value=min(10, len(keywords) - 1),
                value=min(5, len(keywords) - 1)
            )

            if selected_keyword:
                similar_df = find_most_similar(similarity_df, selected_keyword, top_n)

                if not similar_df.empty:
                    st.markdown(f"### Keywords más similares a **'{selected_keyword}'**:")

                    # Crear gráfico de barras
                    fig_similar = px.bar(
                        similar_df,
                        x='Similitud',
                        y='Keyword',
                        orientation='h',
                        title=f'Top {top_n} Keywords Similares',
                        color='Similitud',
                        color_continuous_scale='RdYlGn',
                        range_color=[0, 1]
                    )

                    fig_similar.update_layout(
                        height=400,
                        yaxis={'categoryorder': 'total ascending'},
                        showlegend=False
                    )

                    st.plotly_chart(fig_similar, use_container_width=True)

                    # Tabla de resultados
                    st.dataframe(
                        similar_df.style.background_gradient(
                            subset=['Similitud'],
                            cmap='RdYlGn',
                            vmin=0,
                            vmax=1
                        ),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.warning("No se encontraron resultados.")

        # Tab 6: Tabla de Datos
        with tab6:
            st.subheader("📋 Matriz de Similitud - Vista de Tabla")
            st.markdown("""
            Visualiza la matriz de similitud completa en formato tabla.
            Los valores representan la similitud coseno entre cada par de keywords (0 = sin relación, 1 = idénticas).
            """)

            # Mostrar matriz completa
            st.markdown("### Matriz de Similitud Completa")

            # Formatear valores como porcentajes para mejor lectura
            similarity_display = similarity_df.copy()

            # Aplicar estilo con gradiente de color
            styled_df = similarity_display.style.background_gradient(
                cmap='RdYlGn',
                vmin=0,
                vmax=1,
                axis=None
            ).format("{:.3f}")

            st.dataframe(
                styled_df,
                use_container_width=True,
                height=600
            )

            # Tabla de pares con mayor similitud
            st.markdown("### Top Pares más Similares")

            # Extraer pares únicos (triángulo superior sin diagonal)
            pairs_data = []
            for i, kw1 in enumerate(keywords):
                for j, kw2 in enumerate(keywords):
                    if i < j:
                        sim_value = float(similarity_df.loc[kw1, kw2])
                        pairs_data.append({
                            'Keyword 1': kw1,
                            'Keyword 2': kw2,
                            'Similitud': sim_value,
                            'Similitud (%)': f"{sim_value * 100:.1f}%"
                        })

            pairs_df = pd.DataFrame(pairs_data).sort_values('Similitud', ascending=False)

            # Mostrar top 20
            st.dataframe(
                pairs_df.head(20).style.background_gradient(
                    subset=['Similitud'],
                    cmap='RdYlGn',
                    vmin=0,
                    vmax=1
                ),
                use_container_width=True,
                hide_index=True
            )

            # Opción de ver todos los pares
            with st.expander("Ver todos los pares de similitud"):
                st.dataframe(
                    pairs_df.style.background_gradient(
                        subset=['Similitud'],
                        cmap='RdYlGn',
                        vmin=0,
                        vmax=1
                    ),
                    use_container_width=True,
                    hide_index=True
                )

        # Opción de descarga
        st.divider()
        st.subheader("💾 Exportar Resultados")

        col_download1, col_download2 = st.columns(2)

        with col_download1:
            # Exportar matriz de similitud
            csv_similarity = similarity_df.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Descargar Matriz de Similitud (CSV)",
                data=csv_similarity,
                file_name="matriz_similitud_keywords.csv",
                mime="text/csv"
            )

        with col_download2:
            # Exportar clusters
            if 'cluster_df' in locals():
                csv_clusters = cluster_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Descargar Clusters (CSV)",
                    data=csv_clusters,
                    file_name="clusters_keywords.csv",
                    mime="text/csv"
                )

    else:
        # Placeholder cuando no hay input
        st.info("👆 Introduce palabras clave arriba para comenzar el análisis")

        # Ejemplo
        with st.expander("💡 Ver Ejemplo"):
            st.markdown("""
            **Ejemplo de keywords para probar:**
            ```
            marketing digital
            SEO
            SEM
            publicidad online
            redes sociales
            content marketing
            email marketing
            estrategia digital
            analítica web
            conversión
            ```

            **Qué puedes analizar:**
            - ✅ Similitud semántica entre conceptos
            - ✅ Agrupaciones naturales de temas
            - ✅ Relaciones visuales entre keywords
            - ✅ Encontrar keywords relacionadas
            """)


if __name__ == "__main__":
    render_semantic_relations()
