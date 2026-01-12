"""
Internal Linking Algorithms
============================

Algoritmos de enlazado interno basados en similitud semántica,
arquitectura de información y señales híbridas.

Autor: Embedding Insights
Versión: 1.0.0
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

# Import funciones compartidas de content_utils
# Estas deben estar disponibles en el path cuando se ejecute
from apps.content_analyzer.modules.shared.content_utils import (
    extract_url_silo,
    extract_url_hierarchy,
    generate_contextual_anchor,
    parse_entity_payload,
    calculate_weighted_entity_overlap,
)

from .linking_pagerank import build_similarity_edges, calculate_topical_pagerank


# ============================================================================
# ALGORITMO BÁSICO: ENLAZADO SEMÁNTICO
# ============================================================================

def semantic_link_recommendations(
    df: pd.DataFrame,
    url_column: str,
    type_column: str,
    source_types: Sequence[str],
    primary_target_types: Sequence[str],
    secondary_target_types: Optional[Sequence[str]],
    similarity_threshold: float,
    max_links_per_source: int,
    max_primary: int,
    max_secondary: int,
    source_limit: Optional[int] = None,
    selected_source_urls: Optional[Sequence[str]] = None,
    embedding_col: str = "EmbeddingsFloat",
) -> pd.DataFrame:
    """
    Genera recomendaciones de enlazado interno respetando prioridades de tipos de destino.

    Algoritmo básico que utiliza similitud coseno entre embeddings para identificar
    páginas semánticamente relacionadas, priorizando objetivos business (money pages).

    Args:
        df: DataFrame con URLs, tipos y embeddings
        url_column: Nombre de columna con URLs
        type_column: Nombre de columna con tipos de página
        source_types: Tipos de página que generarán enlaces (ej: ["blog", "post"])
        primary_target_types: Tipos objetivo prioritarios (ej: ["servicio", "producto"])
        secondary_target_types: Tipos objetivo secundarios (ej: ["categoría"])
        similarity_threshold: Similitud mínima para considerar enlace (0.0-1.0)
        max_links_per_source: Máximo de enlaces totales por página origen
        max_primary: Máximo de enlaces a targets prioritarios
        max_secondary: Máximo de enlaces a targets secundarios
        source_limit: Limitar número de páginas origen a procesar
        selected_source_urls: Filtrar a URLs origen específicas
        embedding_col: Nombre de columna con embeddings

    Returns:
        DataFrame con columnas:
        - Origen URL, Origen Tipo
        - Destino Sugerido URL, Destino Tipo
        - Score Similitud (%)
        - Acción SEO (ej: "Objetivo prioritario")

    Raises:
        ValueError: Si faltan columnas necesarias o configuración inválida

    Example:
        >>> df = pd.DataFrame({
        ...     'url': ['blog1', 'blog2', 'servicio1'],
        ...     'tipo': ['blog', 'blog', 'servicio'],
        ...     'EmbeddingsFloat': [emb1, emb2, emb3]
        ... })
        >>> links = semantic_link_recommendations(
        ...     df, 'url', 'tipo',
        ...     source_types=['blog'],
        ...     primary_target_types=['servicio'],
        ...     secondary_target_types=None,
        ...     similarity_threshold=0.6,
        ...     max_links_per_source=3,
        ...     max_primary=2,
        ...     max_secondary=1
        ... )
    """
    # Validaciones
    required_columns = {url_column, type_column, embedding_col}
    if not required_columns.issubset(df.columns):
        missing = ", ".join(required_columns - set(df.columns))
        raise ValueError(f"Hacen falta las columnas: {missing}")
    if not source_types:
        raise ValueError("Selecciona al menos un tipo de página de origen.")

    # Preparar datos
    df_local = df.copy()
    df_local[url_column] = df_local[url_column].astype(str).str.strip()
    df_local[type_column] = df_local[type_column].astype(str).str.strip()

    # Embeddings normalizados
    embeddings = np.vstack(df_local[embedding_col].values)
    embeddings_norm = normalize(embeddings)

    urls = df_local[url_column].tolist()
    page_types = df_local[type_column].tolist()

    # Sets para filtrado eficiente
    source_set = {str(value).strip() for value in source_types}
    primary_set = {str(value).strip() for value in primary_target_types or []}
    secondary_set = (
        {str(value).strip() for value in secondary_target_types} if secondary_target_types is not None else set()
    )

    # Identificar páginas origen
    source_indices = [idx for idx, page_type in enumerate(page_types) if page_type in source_set]

    # Filtros opcionales
    if selected_source_urls:
        selected_set = {str(url).strip() for url in selected_source_urls}
        source_indices = [idx for idx in source_indices if urls[idx] in selected_set]

    if source_limit is not None and source_limit > 0:
        source_indices = source_indices[: int(source_limit)]

    columns = [
        "Origen URL",
        "Origen Tipo",
        "Destino Sugerido URL",
        "Destino Tipo",
        "Score Similitud (%)",
        "Acción SEO",
    ]

    if not source_indices:
        return pd.DataFrame(columns=columns)

    recommendations: List[Dict[str, str]] = []
    total_rows = len(df_local)

    # Generar recomendaciones por cada página origen
    for src_idx in source_indices:
        # Calcular similitud con todas las páginas
        similarities = embeddings_norm @ embeddings_norm[src_idx]

        # Filtrar candidatos válidos
        candidate_indices = [
            idx
            for idx in range(total_rows)
            if idx != src_idx
            and similarities[idx] >= similarity_threshold
            and urls[idx] != urls[src_idx]
        ]

        if not candidate_indices:
            continue

        # Ordenar por similitud descendente
        candidate_indices.sort(key=lambda idx: float(similarities[idx]), reverse=True)

        # Segmentar por prioridad
        primary_candidates = [idx for idx in candidate_indices if page_types[idx] in primary_set] if primary_set else []
        primary_idx_set = set(primary_candidates)

        secondary_candidates = (
            [idx for idx in candidate_indices if idx not in primary_idx_set and page_types[idx] in secondary_set]
            if secondary_set
            else []
        )
        secondary_idx_set = set(secondary_candidates)

        fallback_candidates = [
            idx for idx in candidate_indices if idx not in primary_idx_set and idx not in secondary_idx_set
        ]

        # Seleccionar enlaces respetando límites
        selected_pairs: List[Tuple[int, str]] = []
        used_indices: set[int] = set()

        def extend(indices: Sequence[int], limit: Optional[int], label: str) -> None:
            """Helper para añadir enlaces respetando límite."""
            if limit is not None and limit <= 0:
                return
            taken = 0
            for idx in indices:
                if len(selected_pairs) >= max_links_per_source:
                    break
                if idx in used_indices:
                    continue
                selected_pairs.append((idx, label))
                used_indices.add(idx)
                taken += 1
                if limit is not None and taken >= limit:
                    break

        # Prioridad 1: Objetivos prioritarios (money pages)
        if primary_candidates:
            extend(primary_candidates, int(max_primary), "Objetivo prioritario")

        # Prioridad 2: Objetivos secundarios
        if secondary_candidates:
            extend(secondary_candidates, int(max_secondary), "Cluster complementario")

        # Prioridad 3: Fallback (exploración semántica)
        if len(selected_pairs) < max_links_per_source:
            extend(fallback_candidates, max_links_per_source - len(selected_pairs), "Exploración")

        if not selected_pairs:
            continue

        # Crear registros de recomendaciones
        for candidate_idx, action_label in selected_pairs:
            score = float(similarities[candidate_idx]) * 100.0
            recommendations.append(
                {
                    "Origen URL": urls[src_idx],
                    "Origen Tipo": page_types[src_idx],
                    "Destino Sugerido URL": urls[candidate_idx],
                    "Destino Tipo": page_types[candidate_idx],
                    "Score Similitud (%)": round(score, 2),
                    "Acción SEO": action_label,
                }
            )

    if not recommendations:
        return pd.DataFrame(columns=columns)

    return (
        pd.DataFrame(recommendations)
        .sort_values(["Origen URL", "Score Similitud (%)"], ascending=[True, False])
        .reset_index(drop=True)
    )


# ============================================================================
# ALGORITMO AVANZADO: ENLAZADO SEMÁNTICO + SILOS
# ============================================================================

def advanced_semantic_linking(
    df: pd.DataFrame,
    url_column: str,
    type_column: str,
    source_types: Sequence[str],
    primary_target_types: Sequence[str],
    secondary_target_types: Optional[Sequence[str]],
    similarity_threshold: float,
    max_links_per_source: int,
    max_primary: int,
    max_secondary: int,
    silo_depth: int,
    silo_boost: float,
    embedding_col: str = "EmbeddingsFloat",
    source_limit: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Variante avanzada que añade señal de arquitectura (silos) y reporte de páginas huérfanas.

    Combina similitud semántica con estructura de silos para priorizar enlaces
    dentro del mismo tema. Identifica páginas objetivo sin inlinks (huérfanas).

    Args:
        silo_depth: Profundidad URL para extraer silo (1=primera carpeta, 2=segunda)
        silo_boost: Boost adicional a similitud para páginas del mismo silo

    Returns:
        Tupla (DataFrame de recomendaciones, Lista de URLs huérfanas)

    El DataFrame contiene:
        - Origen URL, Tipo, Silo
        - Destino URL, Tipo, Silo
        - Anchor Text Sugerido
        - Score Base (%), Score Final (%)
        - Boost Aplicado, Estrategia

    Example:
        >>> links_df, orphans = advanced_semantic_linking(
        ...     df, 'url', 'tipo',
        ...     source_types=['blog'],
        ...     primary_target_types=['servicio'],
        ...     secondary_target_types=['categoria'],
        ...     similarity_threshold=0.5,
        ...     max_links_per_source=5,
        ...     max_primary=3,
        ...     max_secondary=2,
        ...     silo_depth=2,
        ...     silo_boost=0.15
        ... )
        >>> len(orphans)  # Servicios sin inlinks
        2
    """
    # Validaciones
    required_columns = {url_column, type_column, embedding_col}
    if not required_columns.issubset(df.columns):
        missing = ", ".join(required_columns - set(df.columns))
        raise ValueError(f"Faltan columnas necesarias: {missing}")
    if not source_types:
        raise ValueError("Debes seleccionar tipos de origen.")
    if not primary_target_types:
        raise ValueError("Debes seleccionar al menos un tipo destino prioritario.")

    # Preparar datos con silos y anchors
    df_local = df.copy()
    df_local[url_column] = df_local[url_column].astype(str).str.strip()
    df_local[type_column] = df_local[type_column].astype(str).str.strip()
    df_local["Silo"] = df_local[url_column].apply(lambda url: extract_url_silo(url, depth=silo_depth))
    df_local["SuggestedAnchor"] = df_local.apply(
        lambda row: generate_contextual_anchor(row[type_column], row["Silo"], row[url_column]),
        axis=1,
    )

    # Embeddings normalizados
    embeddings = np.vstack(df_local[embedding_col].values)
    embeddings_norm = normalize(embeddings)

    urls = df_local[url_column].tolist()
    page_types = df_local[type_column].tolist()
    silos = df_local["Silo"].tolist()
    anchors = df_local["SuggestedAnchor"].tolist()
    total_rows = len(df_local)

    url_type_map: Dict[str, str] = {}
    for url_value, page_type in zip(urls, page_types):
        url_type_map.setdefault(url_value, page_type)

    # Sets para filtrado
    source_set = {str(t).strip() for t in source_types}
    primary_set = {str(t).strip() for t in primary_target_types}
    secondary_set = (
        {str(t).strip() for t in secondary_target_types} if secondary_target_types is not None else set()
    )
    allowed_target_types = primary_set.union(secondary_set) if secondary_set else set(primary_set)

    # Identificar páginas origen
    source_indices = [idx for idx, page_type in enumerate(page_types) if page_type in source_set]
    if source_limit is not None and source_limit > 0:
        source_indices = source_indices[: int(source_limit)]

    # Ajustar límites
    max_primary = min(max_primary, max_links_per_source)
    max_secondary = min(max_secondary, max_links_per_source)

    recommendations: List[Dict[str, object]] = []
    target_counts: Dict[str, int] = {url: 0 for url in urls}

    # Generar recomendaciones
    for src_idx in source_indices:
        source_url = urls[src_idx]
        source_type = page_types[src_idx]
        source_silo = silos[src_idx]

        # Similitud base
        similarities = embeddings_norm @ embeddings_norm[src_idx]

        # Boost para mismo silo
        same_silo_mask = np.array([1.0 if silo == source_silo else 0.0 for silo in silos])
        boosted = similarities + same_silo_mask * float(silo_boost)

        # Filtrar candidatos
        candidate_indices = [
            idx
            for idx in range(total_rows)
            if idx != src_idx
            and boosted[idx] >= similarity_threshold
            and urls[idx] != source_url
            and page_types[idx] in allowed_target_types
        ]

        if not candidate_indices:
            continue

        # Ordenar por score boosted
        candidate_indices.sort(key=lambda i: float(boosted[i]), reverse=True)

        # Segmentar por prioridad
        primary_candidates = [idx for idx in candidate_indices if page_types[idx] in primary_set]
        primary_idx_set = set(primary_candidates)
        secondary_candidates = (
            [idx for idx in candidate_indices if idx not in primary_idx_set and page_types[idx] in secondary_set]
            if secondary_set
            else []
        )
        secondary_idx_set = set(secondary_candidates)
        fallback_candidates = [
            idx for idx in candidate_indices if idx not in primary_idx_set and idx not in secondary_idx_set
        ]

        selected_pairs: List[Tuple[int, str]] = []
        used_indices: set[int] = set()

        def extend(indices: Sequence[int], limit: Optional[int], label: str) -> None:
            if limit is not None and limit <= 0:
                return
            taken = 0
            for idx in indices:
                if len(selected_pairs) >= max_links_per_source:
                    break
                if idx in used_indices:
                    continue
                selected_pairs.append((idx, label))
                used_indices.add(idx)
                taken += 1
                if limit is not None and taken >= limit:
                    break

        extend(primary_candidates, int(max_primary), "Silo vertical / Money page")
        if len(selected_pairs) < max_links_per_source and secondary_candidates:
            extend(secondary_candidates, int(max_secondary), "Cluster relacionado")
        if len(selected_pairs) < max_links_per_source and fallback_candidates:
            extend(fallback_candidates, max_links_per_source - len(selected_pairs), "Exploración semántica")

        if not selected_pairs:
            continue

        # Crear registros
        for candidate_idx, action_label in selected_pairs:
            target_counts[urls[candidate_idx]] += 1
            final_score = float(boosted[candidate_idx]) * 100.0
            base_score = float(similarities[candidate_idx]) * 100.0
            recommendations.append(
                {
                    "Origen URL": source_url,
                    "Origen Tipo": source_type,
                    "Origen Silo": source_silo,
                    "Destino URL": urls[candidate_idx],
                    "Destino Tipo": page_types[candidate_idx],
                    "Destino Silo": silos[candidate_idx],
                    "Anchor Text Sugerido": anchors[candidate_idx],
                    "Score Base (%)": round(base_score, 2),
                    "Score Final (%)": round(final_score, 2),
                    "Boost Aplicado": round(float(final_score - base_score), 2),
                    "Estrategia": action_label
                    if source_silo != silos[candidate_idx]
                    else ("Silo reforzado" if "Money" in action_label else action_label),
                }
            )

    if recommendations:
        report_df = (
            pd.DataFrame(recommendations)
            .sort_values(["Origen URL", "Score Final (%)"], ascending=[True, False])
            .reset_index(drop=True)
        )
    else:
        report_df = pd.DataFrame(
            columns=[
                "Origen URL",
                "Origen Tipo",
                "Origen Silo",
                "Destino URL",
                "Destino Tipo",
                "Destino Silo",
                "Anchor Text Sugerido",
                "Score Base (%)",
                "Score Final (%)",
                "Boost Aplicado",
                "Estrategia",
            ]
        )

    # Detectar páginas huérfanas (money pages sin inlinks)
    orphan_urls = [
        url for url, count in target_counts.items() if count == 0 and url_type_map.get(url) in primary_set
    ]

    return report_df, orphan_urls


# ============================================================================
# Continúa en siguiente mensaje (archivo demasiado grande)
# ============================================================================
