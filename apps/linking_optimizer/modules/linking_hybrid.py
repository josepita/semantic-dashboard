"""
Hybrid Linking Algorithm
=========================

Algoritmo híbrido que combina 3 señales:
- Similitud semántica
- Autoridad (PageRank)
- Solapamiento de entidades

Autor: Embedding Insights
Versión: 1.0.0
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from apps.content_analyzer.modules.shared.content_utils import (
    extract_url_silo,
    generate_contextual_anchor,
    parse_entity_payload,
    calculate_weighted_entity_overlap,
)

from .linking_pagerank import build_similarity_edges, calculate_topical_pagerank
from .linking_algorithms import normalize_url  # Import shared normalize_url


# ============================================================================
# ALGORITMO HÍBRIDO: SEMÁNTICA + AUTORIDAD + ENTIDADES
# ============================================================================

def hybrid_semantic_linking(
    df: pd.DataFrame,
    url_column: str,
    type_column: str,
    entity_column: str,
    source_types: Sequence[str],
    primary_target_types: Sequence[str],
    similarity_threshold: float,
    max_links_per_source: int,
    max_primary: int,
    decay_factor: float,
    weights: Optional[Dict[str, float]] = None,
    embedding_col: str = "EmbeddingsFloat",
    source_limit: Optional[int] = None,
    top_k_edges: int = 5,
    existing_edges: Optional[Sequence[Tuple[str, str]]] = None,
    exclude_as_source_mask: Optional[pd.Series] = None,
    exclude_as_target_mask: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, List[str], Dict[str, float]]:
    """
    Genera recomendaciones de enlaces internos combinando similitud semántica,
    autoridad (PageRank) y solapamiento de entidades.

    **Composite Link Score (CLS)** combina 3 señales:
    - **Semántica** (40%): Similitud coseno entre embeddings
    - **Autoridad** (35%): PageRank de la página destino
    - **Entidades** (25%): Solapamiento de entidades nombradas

    El decay factor penaliza páginas destino que ya tienen muchos inlinks
    para distribuir mejor la autoridad (evitar concentración).

    Args:
        df: DataFrame con URLs, tipos, embeddings y entidades
        url_column: Nombre de columna con URLs
        type_column: Nombre de columna con tipos de página
        entity_column: Nombre de columna con payload de entidades
        source_types: Tipos de página que generarán enlaces
        primary_target_types: Tipos objetivo prioritarios
        similarity_threshold: Similitud mínima para considerar enlace
        max_links_per_source: Máximo de enlaces por página origen
        max_primary: Máximo de enlaces a targets prioritarios
        decay_factor: Factor de penalización por concentración de inlinks (0.0-1.0)
        weights: Pesos custom para las 3 señales {'semantic': 0.4, 'authority': 0.35, 'entity_overlap': 0.25}
        embedding_col: Nombre de columna con embeddings
        source_limit: Limitar número de páginas origen a procesar
        top_k_edges: Top-K aristas semánticas para grafo de PageRank
        existing_edges: Enlaces existentes (source, target) o (source, target, weight) para:
                       - Filtrar y evitar recomendarlos nuevamente
                       - Mejorar cálculo de PageRank con enlaces reales

    Returns:
        Tupla (DataFrame de recomendaciones, Lista de URLs huérfanas, Dict de PageRank scores)

    El DataFrame contiene:
        - Origen URL, Tipo
        - Destino URL, Tipo
        - Anchor Text Sugerido
        - Score Semántico (%)
        - Score Entidades (%)
        - Score Autoridad (PR)
        - CLS Ajustado (%)
        - CLS Base (%)
        - Estrategia

    Example:
        >>> df = pd.DataFrame({
        ...     'url': ['blog1', 'blog2', 'servicio1'],
        ...     'tipo': ['blog', 'blog', 'servicio'],
        ...     'EmbeddingsFloat': [emb1, emb2, emb3],
        ...     'entities': [
        ...         '{"entity1": 0.8, "entity2": 0.5}',
        ...         '{"entity1": 0.6, "entity3": 0.7}',
        ...         '{"entity1": 0.9}'
        ...     ]
        ... })
        >>> links, orphans, pr = hybrid_semantic_linking(
        ...     df, 'url', 'tipo', 'entities',
        ...     source_types=['blog'],
        ...     primary_target_types=['servicio'],
        ...     similarity_threshold=0.5,
        ...     max_links_per_source=3,
        ...     max_primary=2,
        ...     decay_factor=0.1
        ... )
    """
    # Validaciones
    required_columns = {url_column, type_column, embedding_col, entity_column}
    if not required_columns.issubset(df.columns):
        missing = ", ".join(required_columns - set(df.columns))
        raise ValueError(f"Faltan columnas necesarias: {missing}")

    # Pesos por defecto: semántica (40%), autoridad (35%), entidades (25%)
    weights = weights or {"semantic": 0.4, "authority": 0.35, "entity_overlap": 0.25}
    weight_sem = max(0.0, float(weights.get("semantic", 0.4)))
    weight_auth = max(0.0, float(weights.get("authority", 0.35)))
    weight_ent = max(0.0, float(weights.get("entity_overlap", 0.25)))

    # Normalizar pesos
    total_weights = weight_sem + weight_auth + weight_ent
    if total_weights == 0:
        weight_sem, weight_auth, weight_ent = 0.4, 0.35, 0.25
        total_weights = 1.0
    weight_sem /= total_weights
    weight_auth /= total_weights
    weight_ent /= total_weights

    # Preparar datos
    df_local = df.copy()
    df_local[url_column] = df_local[url_column].astype(str).str.strip().apply(normalize_url)
    df_local[type_column] = df_local[type_column].astype(str).str.strip()

    # Parsear entidades
    entity_maps: List[Dict[str, float]] = df_local[entity_column].apply(parse_entity_payload).tolist()

    # Embeddings normalizados
    embeddings = np.vstack(df_local[embedding_col].values)
    embeddings_norm = normalize(embeddings)

    urls = df_local[url_column].tolist()
    page_types = df_local[type_column].tolist()
    total_rows = len(df_local)

    # ========================================================================
    # CALCULAR PAGERANK TEMÁTICO
    # ========================================================================
    # Construir grafo semántico
    similarity_edges = build_similarity_edges(
        embeddings_norm=embeddings_norm,
        urls=urls,
        min_threshold=similarity_threshold,
        top_k=top_k_edges,
    )

    # Calcular PageRank (incorporando enlaces existentes para mayor precisión)
    pagerank_scores = calculate_topical_pagerank(
        df_local,
        url_column=url_column,
        type_column=type_column,
        primary_target_types=primary_target_types,
        graph_edges=similarity_edges,
        alpha=0.85,
        existing_edges=existing_edges,  # Incluir enlaces reales
    )

    # Normalizar PageRank a [0, 1]
    max_pr = max(pagerank_scores.values()) if pagerank_scores else 1.0
    pr_norm = {url: score / max_pr for url, score in pagerank_scores.items()} if max_pr else pagerank_scores

    # ========================================================================
    # PREPARAR ORIGEN Y TARGETS
    # ========================================================================
    source_set = {str(t).strip() for t in source_types}
    primary_set = {str(t).strip() for t in primary_target_types}

    source_indices = [idx for idx, page_type in enumerate(page_types) if page_type in source_set]

    # Aplicar filtro de exclusión como origen
    if exclude_as_source_mask is not None:
        exclude_source_list = exclude_as_source_mask.tolist()
        source_indices = [idx for idx in source_indices if not exclude_source_list[idx]]

    if source_limit is not None and source_limit > 0:
        source_indices = source_indices[: int(source_limit)]

    # ========================================================================
    # FILTRAR ENLACES EXISTENTES
    # ========================================================================
    # Crear set de enlaces existentes para evitar recomendar duplicados
    existing_links_set: Set[Tuple[str, str]] = set()
    if existing_edges:
        for edge_tuple in existing_edges:
            # Soportar (source, target) o (source, target, weight)
            if len(edge_tuple) == 3:
                src, tgt, _ = edge_tuple
            elif len(edge_tuple) == 2:
                src, tgt = edge_tuple
            else:
                continue
            existing_links_set.add((str(src).strip(), str(tgt).strip()))

    # ========================================================================
    # GENERAR RECOMENDACIONES
    # ========================================================================
    recommendations: List[Dict[str, object]] = []
    target_link_counts: Dict[str, int] = {url: 0 for url in urls}

    # Preparar lista de exclusión como destino
    exclude_target_list = exclude_as_target_mask.tolist() if exclude_as_target_mask is not None else None

    for src_idx in source_indices:
        source_url = urls[src_idx]
        source_entities = entity_maps[src_idx]

        # Calcular similitud semántica
        similarities = embeddings_norm @ embeddings_norm[src_idx]

        # Evaluar candidatos
        candidate_scores: List[Tuple[int, float, float, float]] = []

        for cand_idx in range(total_rows):
            if cand_idx == src_idx:
                continue

            target_url = urls[cand_idx]

            # Validación de auto-enlace mejorada (URLs ya normalizadas)
            if target_url == source_url:
                continue

            # Filtrar enlaces que ya existen
            if (source_url, target_url) in existing_links_set:
                continue

            # Filtrar páginas excluidas como destino
            if exclude_target_list is not None and exclude_target_list[cand_idx]:
                continue

            # Score semántico (ya normalizado en [0, 1] por sklearn.normalize)
            semantic_score = float(similarities[cand_idx])

            # Verificar threshold
            if semantic_score < similarity_threshold:
                continue

            # Score de autoridad (PageRank normalizado)
            authority_score = pr_norm.get(target_url, 0.0)

            # Score de entidades (solapamiento ponderado)
            target_entities = entity_maps[cand_idx]
            entity_overlap = calculate_weighted_entity_overlap(source_entities, target_entities)

            # Composite Link Score (CLS)
            composite_score = (
                weight_sem * semantic_score + weight_auth * authority_score + weight_ent * entity_overlap
            )

            candidate_scores.append((cand_idx, composite_score, semantic_score, entity_overlap))

        if not candidate_scores:
            continue

        # Ordenar por CLS descendente
        candidate_scores.sort(key=lambda item: item[1], reverse=True)

        # ====================================================================
        # SELECCIONAR ENLACES CON DECAY FACTOR
        # ====================================================================
        selected_pairs: List[Tuple[int, float, float, float, float, str]] = []
        used_indices: Set[int] = set()

        def apply_selection(
            candidates: List[Tuple[int, float, float, float]],
            limit: Optional[int],
            label: str,
        ) -> None:
            """Helper para seleccionar enlaces aplicando decay por concentración."""
            if limit is not None and limit <= 0:
                return

            taken = 0
            for cand_idx, cls_raw, semantic_value, entity_value in candidates:
                if len(selected_pairs) >= max_links_per_source:
                    break
                if cand_idx in used_indices:
                    continue

                target_url = urls[cand_idx]
                current_count = target_link_counts.get(target_url, 0)

                # Penalización por concentración de inlinks (decay factor)
                decay_penalty = math.log(1 + max(decay_factor, 0.0) * current_count) if decay_factor > 0 else 0.0
                adjusted_cls = max(0.0, cls_raw - decay_penalty)

                if adjusted_cls <= 0:
                    continue

                selected_pairs.append((cand_idx, adjusted_cls, cls_raw, semantic_value, entity_value, label))
                used_indices.add(cand_idx)
                target_link_counts[target_url] = current_count + 1

                taken += 1
                if limit is not None and taken >= limit:
                    break

        # Segmentar candidatos por prioridad
        primary_candidates = [item for item in candidate_scores if page_types[item[0]] in primary_set]
        secondary_candidates = [item for item in candidate_scores if page_types[item[0]] not in primary_set]

        # Aplicar selección priorizada
        apply_selection(primary_candidates, int(max_primary), "Objetivo prioritario (CLS)")

        remaining_limit = max_links_per_source - len(selected_pairs)
        apply_selection(secondary_candidates, remaining_limit, "Exploración semántica (CLS)")

        # Crear registros de recomendaciones
        for cand_idx, adjusted_cls, cls_raw, semantic_val, entity_val, label in selected_pairs:
            target_url = urls[cand_idx]
            recommendations.append(
                {
                    "Origen URL": source_url,
                    "Origen Tipo": page_types[src_idx],
                    "Destino URL": target_url,
                    "Destino Tipo": page_types[cand_idx],
                    "Anchor Text Sugerido": generate_contextual_anchor(
                        page_types[cand_idx],
                        extract_url_silo(target_url),
                        target_url,
                    ),
                    "Score Semántico (%)": round(semantic_val * 100.0, 2),
                    "Score Entidades (%)": round(entity_val * 100.0, 2),
                    "Score Autoridad (PR)": round(pr_norm.get(target_url, 0.0), 4),
                    "CLS Ajustado (%)": round(adjusted_cls * 100.0, 2),
                    "CLS Base (%)": round(cls_raw * 100.0, 2),
                    "Estrategia": label,
                }
            )

    # ========================================================================
    # PREPARAR RESULTADOS
    # ========================================================================
    if recommendations:
        report_df = (
            pd.DataFrame(recommendations).sort_values("CLS Ajustado (%)", ascending=False).reset_index(drop=True)
        )
    else:
        report_df = pd.DataFrame(
            columns=[
                "Origen URL",
                "Origen Tipo",
                "Destino URL",
                "Destino Tipo",
                "Anchor Text Sugerido",
                "Score Semántico (%)",
                "Score Entidades (%)",
                "Score Autoridad (PR)",
                "CLS Ajustado (%)",
                "CLS Base (%)",
                "Estrategia",
            ]
        )

    # Detectar páginas huérfanas (money pages sin inlinks)
    orphan_urls: List[str] = []
    for url, count in target_link_counts.items():
        if count != 0:
            continue
        type_values = df_local.loc[df_local[url_column] == url, type_column]
        if type_values.empty:
            continue
        if type_values.iloc[0] in primary_set:
            orphan_urls.append(url)

    return report_df, orphan_urls, pagerank_scores


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = ["hybrid_semantic_linking"]
