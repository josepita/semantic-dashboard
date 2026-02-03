"""
Análisis de cobertura de dominio usando similitud semántica con embeddings.

Compara fan-out queries contra los embeddings de las páginas del sitio
para clasificar la cobertura temática.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_THRESHOLDS = {
    "perfect_coverage": 0.80,
    "aligned": 0.65,
    "related_gap": 0.45,
    "clear_gap": 0.25,
}

CLASSIFICATION_LABELS = {
    "perfect_coverage": "Perfect Coverage",
    "aligned": "Aligned",
    "related_gap": "Related Gap",
    "clear_gap": "Clear Gap",
    "no_coverage": "No Coverage",
}

CLASSIFICATION_ORDER = [
    "perfect_coverage",
    "aligned",
    "related_gap",
    "clear_gap",
    "no_coverage",
]


def _classify_similarity(sim: float, thresholds: Dict[str, float]) -> str:
    if sim >= thresholds.get("perfect_coverage", 0.80):
        return "perfect_coverage"
    if sim >= thresholds.get("aligned", 0.65):
        return "aligned"
    if sim >= thresholds.get("related_gap", 0.45):
        return "related_gap"
    if sim >= thresholds.get("clear_gap", 0.25):
        return "clear_gap"
    return "no_coverage"


def analyze_domain_coverage(
    queries_df: pd.DataFrame,
    site_df: pd.DataFrame,
    url_column: str,
    embedding_col: str = "EmbeddingsFloat",
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Analiza cobertura de dominio comparando queries con embeddings del sitio.

    Args:
        queries_df: DataFrame con columnas [prompt, web_search_query].
        site_df: DataFrame del sitio con URLs y embeddings.
        url_column: Nombre de la columna de URLs.
        embedding_col: Nombre de la columna de embeddings.
        model_id: Modelo sentence-transformers para generar embeddings de queries.
        thresholds: Umbrales de clasificación (override defaults).

    Returns:
        (detail_df, summary_df)
        - detail_df: query, best_url, similarity, classification, prompt
        - summary_df: classification, count, percentage
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS.copy()

    # Extraer embeddings del sitio
    site_embeddings = np.stack(site_df[embedding_col].values)
    site_urls = site_df[url_column].astype(str).values

    # Generar embeddings para las queries
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_id)

    unique_queries = queries_df["web_search_query"].dropna().unique().tolist()
    unique_queries = [q for q in unique_queries if q.strip()]

    if not unique_queries:
        raise ValueError("No hay queries válidas para analizar.")

    query_embeddings = model.encode(unique_queries, show_progress_bar=False)

    # Calcular similitud coseno: queries × sitio
    sim_matrix = cosine_similarity(query_embeddings, site_embeddings)

    # Para cada query, encontrar mejor URL
    query_to_best = {}
    for i, query in enumerate(unique_queries):
        best_idx = int(np.argmax(sim_matrix[i]))
        best_sim = float(sim_matrix[i, best_idx])
        best_url = site_urls[best_idx]
        classification = _classify_similarity(best_sim, thresholds)
        query_to_best[query] = {
            "best_url": best_url,
            "similarity": round(best_sim, 4),
            "classification": classification,
            "classification_label": CLASSIFICATION_LABELS[classification],
        }

    # Construir detail_df uniendo con el DataFrame original de queries
    detail_rows = []
    for _, row in queries_df.iterrows():
        q = str(row.get("web_search_query", "")).strip()
        if not q or q not in query_to_best:
            continue
        match = query_to_best[q]
        detail_rows.append({
            "prompt": row.get("prompt", ""),
            "web_search_query": q,
            "source": row.get("source", ""),
            "best_url": match["best_url"],
            "similarity": match["similarity"],
            "classification": match["classification"],
            "classification_label": match["classification_label"],
        })

    detail_df = pd.DataFrame(detail_rows)

    # Construir summary_df
    if not detail_df.empty:
        summary = detail_df["classification"].value_counts()
        total = len(detail_df)
        summary_rows = []
        for cls in CLASSIFICATION_ORDER:
            count = int(summary.get(cls, 0))
            summary_rows.append({
                "classification": cls,
                "label": CLASSIFICATION_LABELS[cls],
                "count": count,
                "percentage": round(count / total * 100, 1) if total > 0 else 0,
            })
        summary_df = pd.DataFrame(summary_rows)
    else:
        summary_df = pd.DataFrame(columns=["classification", "label", "count", "percentage"])

    return detail_df, summary_df


__all__ = [
    "CLASSIFICATION_LABELS",
    "CLASSIFICATION_ORDER",
    "DEFAULT_THRESHOLDS",
    "analyze_domain_coverage",
]
