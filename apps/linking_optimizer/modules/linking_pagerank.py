"""
PageRank & Graph Algorithms
============================

Algoritmos de PageRank temático y construcción de grafos semánticos
para el enlazado interno.

Autor: Embedding Insights
Versión: 1.0.0
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd


# ============================================================================
# CONSTRUCCIÓN DE GRAFOS SEMÁNTICOS
# ============================================================================

def build_similarity_edges(
    embeddings_norm: np.ndarray,
    urls: Sequence[str],
    min_threshold: float,
    top_k: int = 5,
) -> List[Tuple[str, str, float]]:
    """
    Construye aristas basadas en similitud coseno entre embeddings.

    Selecciona top-K candidatos más similares por encima de umbral.
    Embeddings deben estar L2-normalizados.

    Args:
        embeddings_norm: Matriz de embeddings L2-normalizados (N x D)
        urls: Lista de URLs correspondientes a cada embedding
        min_threshold: Similitud mínima para crear arista
        top_k: Máximo de vecinos a conectar por nodo

    Returns:
        Lista de tuplas (source_url, target_url, similarity_score)

    Example:
        >>> from sklearn.preprocessing import normalize
        >>> embs = np.array([[0.5, 0.5], [0.6, 0.4], [0.1, 0.9]])
        >>> embs_norm = normalize(embs, norm='l2')
        >>> urls = ['page1', 'page2', 'page3']
        >>> edges = build_similarity_edges(embs_norm, urls, min_threshold=0.5, top_k=2)
        >>> len(edges) <= len(urls) * 2
        True
    """
    edges: List[Tuple[str, str, float]] = []
    min_threshold = float(min_threshold)
    n = len(urls)

    for idx in range(n):
        # Similitud coseno: producto punto con embeddings normalizados
        sims = embeddings_norm @ embeddings_norm[idx]
        # Top-K+1 eficiente con argpartition (incluye self que luego excluimos)
        k_fetch = min(top_k + 1, n)
        top_indices = np.argpartition(sims, -k_fetch)[-k_fetch:]
        # Ordenar solo los top-K por score descendente
        top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

        added = 0
        for candidate_idx in top_indices:
            if candidate_idx == idx:
                continue

            score = float(sims[candidate_idx])

            if score < min_threshold:
                continue

            edges.append((urls[idx], urls[candidate_idx], max(score, 1e-6)))

            added += 1
            if added >= top_k:
                break

    return edges


# ============================================================================
# PAGERANK TEMÁTICO
# ============================================================================

def calculate_topical_pagerank(
    df: pd.DataFrame,
    url_column: str,
    type_column: str,
    primary_target_types: Sequence[str],
    graph_edges: Sequence[Tuple[str, str, float]],
    alpha: float = 0.85,
    existing_edges: Optional[Sequence[Tuple[str, str]]] = None,
) -> Dict[str, float]:
    """
    Calcula PageRank temático combinando similitud semántica con enlaces reales.

    Combina dos fuentes de autoridad:
    1. **Aristas semánticas**: Enlaces sugeridos por similitud de embeddings
    2. **Aristas existentes**: Enlaces reales del sitio (con boost de peso 2x)

    Personalización: Da más peso inicial a páginas objetivo prioritarias.

    Args:
        df: DataFrame con URLs y tipos de página
        url_column: Nombre de la columna con las URLs
        type_column: Nombre de la columna con los tipos de página
        primary_target_types: Tipos de página considerados prioritarios
        graph_edges: Aristas semánticas (source, target, weight basado en similitud)
        alpha: Factor de damping para PageRank (default 0.85, típico en literatura)
        existing_edges: Tuplas (source, target) o (source, target, weight) de enlaces reales existentes

    Returns:
        Diccionario {url: pagerank_score} normalizado

    Raises:
        nx.NetworkXException: Si el grafo tiene problemas (manejado internamente con fallback uniforme)

    Example:
        >>> df = pd.DataFrame({
        ...     'url': ['page1', 'page2', 'page3'],
        ...     'tipo': ['blog', 'servicio', 'blog']
        ... })
        >>> edges = [('page1', 'page2', 0.8), ('page2', 'page3', 0.6)]
        >>> pr = calculate_topical_pagerank(
        ...     df, 'url', 'tipo', ['servicio'], edges
        ... )
        >>> 'page1' in pr and 'page2' in pr
        True
        >>> pr['page2'] > pr['page1']  # 'servicio' debería tener más PageRank
        True
    """
    urls = df[url_column].astype(str).str.strip().tolist()
    url_set = set(urls)
    graph = nx.DiGraph()
    graph.add_nodes_from(urls)

    # 1. Añadir aristas semánticas (basadas en similitud de embeddings)
    for source, target, weight in graph_edges:
        if source not in url_set or target not in url_set:
            continue
        graph.add_edge(source, target, weight=max(float(weight), 1e-6))

    # 2. Añadir enlaces explícitos existentes con boost de autoridad
    if existing_edges:
        for edge_tuple in existing_edges:
            # Soportar tanto tuplas de 2 (source, target) como de 3 (source, target, weight_multiplier)
            if len(edge_tuple) == 3:
                source, target, weight_multiplier = edge_tuple
                weight_multiplier = float(weight_multiplier)
            elif len(edge_tuple) == 2:
                source, target = edge_tuple
                weight_multiplier = 1.0
            else:
                continue

            # Normalizar URLs para matching
            source_clean = str(source).strip()
            target_clean = str(target).strip()

            if source_clean not in url_set or target_clean not in url_set:
                continue

            # Aplicar peso base (2.0) multiplicado por el factor configurado
            # Enlaces reales son señal fuerte de relevancia
            base_weight = 2.0
            final_weight = base_weight * weight_multiplier

            if graph.has_edge(source_clean, target_clean):
                # Si ya existe arista semántica, añadir boost adicional
                # Enlace real = evidencia fuerte de relevancia
                graph[source_clean][target_clean]['weight'] += final_weight
            else:
                # Enlace real sin similitud semántica alta
                # Aún así es valioso por la estructura del sitio
                graph.add_edge(source_clean, target_clean, weight=final_weight)

    # 3. Personalización: dar más peso a páginas objetivo prioritarias
    personalization: Dict[str, float] = {}
    primary_set = {str(t).strip() for t in primary_target_types}
    primary_weight = 0.5  # 50% del peso total para páginas prioritarias
    other_weight = 0.05   # 5% para otras páginas

    for url_value, page_type in zip(urls, df[type_column].astype(str).str.strip()):
        personalization[url_value] = primary_weight if page_type in primary_set else other_weight

    # Normalizar distribución de personalización
    total = sum(personalization.values())
    if not total:
        # Fallback uniforme si no hay personalización válida
        personalization = {url: 1.0 / len(urls) for url in urls} if urls else {}
    else:
        personalization = {url: weight / total for url, weight in personalization.items()}

    # 4. Calcular PageRank con personalización
    try:
        pr_scores = nx.pagerank(
            graph,
            alpha=alpha,
            personalization=personalization if personalization else None,
            weight="weight",
        )
    except nx.NetworkXException:
        # Fallback: distribución uniforme si el grafo tiene problemas
        pr_scores = {url: 1.0 / len(graph) for url in graph} if graph else {}

    return pr_scores


# ============================================================================
# UTILIDADES DE GRAFO
# ============================================================================

def detect_orphan_pages(
    all_urls: Sequence[str],
    links_df: pd.DataFrame,
    target_column: str = "target_url",
) -> List[str]:
    """
    Detecta páginas huérfanas (sin inlinks).

    Args:
        all_urls: Lista de todas las URLs del sitio
        links_df: DataFrame con enlaces (debe tener columna target_url)
        target_column: Nombre de la columna con URLs destino

    Returns:
        Lista de URLs sin inlinks

    Example:
        >>> all_urls = ['page1', 'page2', 'page3']
        >>> links = pd.DataFrame({'target_url': ['page1', 'page1']})
        >>> orphans = detect_orphan_pages(all_urls, links)
        >>> set(orphans)
        {'page2', 'page3'}
    """
    if links_df.empty or target_column not in links_df.columns:
        return list(all_urls)

    linked_urls = set(links_df[target_column].astype(str).str.strip())
    orphans = [url for url in all_urls if str(url).strip() not in linked_urls]

    return orphans


def calculate_graph_metrics(
    graph: nx.DiGraph,
) -> Dict[str, Dict[str, float]]:
    """
    Calcula métricas de centralidad del grafo.

    Args:
        graph: Grafo dirigido de NetworkX

    Returns:
        Diccionario con métricas por URL:
            - in_degree: Número de inlinks
            - out_degree: Número de outlinks
            - betweenness: Centralidad de intermediación
            - closeness: Centralidad de cercanía

    Example:
        >>> G = nx.DiGraph()
        >>> G.add_edges_from([('A', 'B'), ('B', 'C'), ('A', 'C')])
        >>> metrics = calculate_graph_metrics(G)
        >>> 'in_degree' in metrics and 'A' in metrics['in_degree']
        True
    """
    metrics = {
        'in_degree': dict(graph.in_degree()),
        'out_degree': dict(graph.out_degree()),
    }

    # Métricas de centralidad (pueden ser costosas en grafos grandes)
    try:
        metrics['betweenness'] = nx.betweenness_centrality(graph)
        metrics['closeness'] = nx.closeness_centrality(graph)
    except (nx.NetworkXException, ZeroDivisionError):
        # Fallback si el grafo no es conexo
        metrics['betweenness'] = {node: 0.0 for node in graph.nodes()}
        metrics['closeness'] = {node: 0.0 for node in graph.nodes()}

    return metrics


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "build_similarity_edges",
    "calculate_topical_pagerank",
    "detect_orphan_pages",
    "calculate_graph_metrics",
]
