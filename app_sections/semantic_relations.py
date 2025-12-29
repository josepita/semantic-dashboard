"""
Módulo de Análisis de Relaciones Semánticas entre Keywords.

Permite analizar y visualizar las relaciones semánticas entre múltiples palabras clave
mediante embeddings y diferentes tipos de gráficos interactivos.
"""

from typing import List, Tuple, Dict, Optional
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import networkx as nx
from pyvis.network import Network
import tempfile
import os


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
    perplexity: int = 5
) -> go.Figure:
    """
    Visualización 2D de embeddings con T-SNE o PCA.

    Args:
        embeddings: Array de embeddings
        keywords: Lista de keywords correspondientes
        method: 'tsne' o 'pca'
        perplexity: Perplexity para T-SNE (ignorado si method='pca')

    Returns:
        Figura de Plotly
    """
    # Reducción de dimensionalidad
    if method == "tsne":
        # Ajustar perplexity si hay pocas keywords
        n_samples = len(keywords)
        perplexity = min(perplexity, n_samples - 1)

        if n_samples < 4:
            st.warning(f"T-SNE requiere al menos 4 palabras. Usando PCA en su lugar.")
            method = "pca"
        else:
            reducer = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=42,
                max_iter=1000
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
        'keyword': keywords
    })

    # Crear scatter plot
    fig = px.scatter(
        df_plot,
        x='x',
        y='y',
        text='keyword',
        title=f'Relaciones Semánticas ({method_name})',
        labels={'x': f'{method_name} Dimensión 1', 'y': f'{method_name} Dimensión 2'}
    )

    # Personalizar puntos y etiquetas
    fig.update_traces(
        textposition='top center',
        marker=dict(
            size=15,
            color='#97C2FC',
            line=dict(width=2, color='#2B7CE9')
        ),
        textfont=dict(size=12, color='black')
    )

    fig.update_layout(
        height=600,
        width=800,
        showlegend=False,
        hovermode='closest'
    )

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

    # Cargar modelo
    from app_sections.semantic_tools import get_sentence_transformer, DEFAULT_SENTENCE_MODEL

    with st.spinner("Cargando modelo de embeddings..."):
        model = get_sentence_transformer(DEFAULT_SENTENCE_MODEL)

    # Sección de entrada de keywords
    st.header("📝 Configuración")

    col1, col2 = st.columns([2, 1])

    with col1:
        keywords_input = st.text_area(
            "Introduce palabras clave (una por línea)",
            height=200,
            placeholder="marketing digital\nSEO\nSEM\npublicidad online\nredes sociales\ncontent marketing\nemail marketing",
            help="Introduce al menos 3 palabras clave para obtener resultados significativos"
        )

    with col2:
        st.markdown("### ⚙️ Opciones")

        similarity_threshold = st.slider(
            "Umbral de similitud (grafo)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Umbral mínimo para mostrar conexiones en el grafo de red"
        )

        visualization_method = st.selectbox(
            "Método de visualización 2D",
            options=["tsne", "pca"],
            format_func=lambda x: "T-SNE (recomendado)" if x == "tsne" else "PCA (más rápido)",
            help="T-SNE preserva mejor las relaciones locales, PCA es más rápido"
        )

        n_clusters = st.number_input(
            "Número de clusters",
            min_value=2,
            max_value=10,
            value=3,
            help="Número de grupos para agrupar keywords similares"
        )

    # Procesar keywords
    if keywords_input:
        keywords = [kw.strip() for kw in keywords_input.split('\n') if kw.strip()]

        if len(keywords) < 2:
            st.warning("⚠️ Introduce al menos 2 palabras clave para analizar.")
            return

        st.success(f"✅ {len(keywords)} palabras clave detectadas")

        # Calcular similitudes
        with st.spinner("Calculando similitudes semánticas..."):
            embeddings, similarity_df = calculate_keyword_similarities(keywords, model)

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

            with st.spinner(f"Calculando proyección {visualization_method.upper()}..."):
                fig_2d = render_2d_visualization(
                    embeddings,
                    keywords,
                    method=visualization_method
                )

            st.plotly_chart(fig_2d, use_container_width=True)

            st.info(f"""
            **Método usado:** {visualization_method.upper()}
            - **T-SNE:** Preserva mejor las relaciones locales (recomendado para visualización)
            - **PCA:** Más rápido, preserva mejor la varianza global
            """)

        # Tab 4: Clustering
        with tab4:
            st.subheader("Agrupación por Similitud Semántica")

            with st.spinner("Agrupando keywords..."):
                labels, cluster_df = perform_clustering(embeddings, keywords, n_clusters)

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
                perplexity = min(5, len(keywords) - 1)
                reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
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
