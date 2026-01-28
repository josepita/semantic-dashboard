"""
Structural Linking Algorithms
==============================

Algoritmos de enlazado basados en jerarquía de URLs y taxonomía del sitio.

Autor: Embedding Insights
Versión: 1.0.0
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from apps.content_analyzer.modules.shared.content_utils import (
    extract_url_hierarchy,
    generate_contextual_anchor,
)


# ============================================================================
# ALGORITMO ESTRUCTURAL: TAXONOMÍA Y JERARQUÍA
# ============================================================================

def structural_taxonomy_linking(
    df: pd.DataFrame,
    url_column: str,
    hierarchy_column: Optional[str],
    depth: int,
    max_links_per_parent: int,
    include_horizontal: bool,
    link_weight: float,
    use_semantic_priority: bool = False,
    embedding_col: str = "EmbeddingsFloat",
    exclude_as_source_mask: Optional[pd.Series] = None,
    exclude_as_target_mask: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Genera recomendaciones de enlaces basadas en la estructura jerárquica de URLs.

    Implementa 3 estrategias de enlazado estructural:
    1. **Ascendente (breadcrumb)**: Hijos → Padres
    2. **Horizontal**: Enlaces entre hermanos del mismo nivel
    3. **Descendente**: Padres → Hijos destacados

    Args:
        df: DataFrame con URLs
        url_column: Nombre de columna con URLs
        hierarchy_column: Columna con jerarquía custom (opcional)
        depth: Profundidad para extraer jerarquía de URLs
        max_links_per_parent: Máximo de enlaces entre hermanos o de padre a hijos
        include_horizontal: Si True, incluye enlaces entre hermanos
        link_weight: Peso base de los enlaces (para PageRank posterior)
        use_semantic_priority: Si True, ordena hermanos por similitud semántica
        embedding_col: Columna con embeddings (necesaria si use_semantic_priority=True)

    Returns:
        DataFrame con columnas:
        - Origen URL, Destino URL
        - Estrategia (ascendente/horizontal/descendente)
        - Anchor Text Sugerido
        - Link Weight

    Example:
        >>> df = pd.DataFrame({
        ...     'url': [
        ...         'https://example.com/blog',
        ...         'https://example.com/blog/python',
        ...         'https://example.com/blog/python/tutorial1',
        ...         'https://example.com/blog/python/tutorial2',
        ...     ]
        ... })
        >>> links = structural_taxonomy_linking(
        ...     df, 'url', None,
        ...     depth=2,
        ...     max_links_per_parent=2,
        ...     include_horizontal=True,
        ...     link_weight=1.0
        ... )
        >>> # Resultado incluye:
        >>> # - tutorial1 → python (breadcrumb)
        >>> # - tutorial1 ↔ tutorial2 (hermanos)
        >>> # - python → [tutorial1, tutorial2] (destacados)
    """
    df_local = df.copy()
    df_local[url_column] = df_local[url_column].astype(str).str.strip()

    # Crear sets de URLs excluidas para búsqueda eficiente
    urls_list = df_local[url_column].tolist()
    excluded_as_source_urls: set = set()
    excluded_as_target_urls: set = set()

    if exclude_as_source_mask is not None:
        exclude_source_list = exclude_as_source_mask.tolist()
        excluded_as_source_urls = {urls_list[i] for i in range(len(urls_list)) if exclude_source_list[i]}

    if exclude_as_target_mask is not None:
        exclude_target_list = exclude_as_target_mask.tolist()
        excluded_as_target_urls = {urls_list[i] for i in range(len(urls_list)) if exclude_target_list[i]}

    # Extraer jerarquía (custom o desde URL)
    if hierarchy_column and hierarchy_column in df_local.columns:
        df_local["HierarchyPath"] = (
            df_local[hierarchy_column].astype(str).str.strip().replace("", "root").fillna("root")
        )
    else:
        df_local["HierarchyPath"] = df_local[url_column].apply(
            lambda url: extract_url_hierarchy(url, depth=depth)
        )

    df_local["HierarchyPath"] = df_local["HierarchyPath"].replace("", "root")

    # Calcular jerarquía padre
    df_local["ParentPath"] = df_local["HierarchyPath"].apply(
        lambda path: "/".join(path.split("/")[:-1]) if path and "/" in path else "root"
    )

    # Mapeos jerárquicos
    path_to_urls: Dict[str, List[str]] = (
        df_local.groupby("HierarchyPath")[url_column].apply(list).to_dict()
        if not df_local.empty
        else {}
    )

    parent_to_children: Dict[str, List[str]] = (
        df_local.groupby("ParentPath")[url_column].apply(list).to_dict()
        if not df_local.empty
        else {}
    )

    recommendations: List[Dict[str, object]] = []

    # ========================================================================
    # ESTRATEGIA 1: ENLACES ASCENDENTES (Breadcrumb)
    # ========================================================================
    # Cada página enlaza a su padre jerárquico
    for _, row in df_local.iterrows():
        source_url = row[url_column]

        # Filtrar si está excluido como origen
        if source_url in excluded_as_source_urls:
            continue

        parent_path = row["ParentPath"]

        if parent_path != "root":
            parent_candidates = path_to_urls.get(parent_path, [])
            # Filtrar candidatos excluidos como destino
            parent_candidates = [p for p in parent_candidates if p not in excluded_as_target_urls]
            parent_url = parent_candidates[0] if parent_candidates else None

            if parent_url:
                recommendations.append(
                    {
                        "Origen URL": source_url,
                        "Destino URL": parent_url,
                        "Estrategia": "Estructural ascendente (breadcrumb)",
                        "Anchor Text Sugerido": f"Volver a {parent_path.split('/')[-1].replace('-', ' ').title()}",
                        "Link Weight": float(link_weight),
                    }
                )

        # ====================================================================
        # ESTRATEGIA 2: ENLACES HORIZONTALES (Hermanos)
        # ====================================================================
        if include_horizontal and source_url not in excluded_as_source_urls:
            siblings = parent_to_children.get(parent_path, [])
            # Filtrar el propio source_url y URLs excluidas como destino
            siblings_filtered = [
                sib for sib in siblings
                if sib != source_url and sib not in excluded_as_target_urls
            ]

            # Priorización semántica opcional
            if use_semantic_priority and embedding_col in df_local.columns and len(siblings_filtered) > 0:
                try:
                    # Obtener embedding de la página origen
                    source_emb_row = df_local.loc[df_local[url_column] == source_url, embedding_col]

                    if not source_emb_row.empty:
                        source_emb = source_emb_row.iloc[0]

                        # Calcular similitud con cada hermano
                        sibling_similarities = []
                        for sib_url in siblings_filtered:
                            sib_emb_row = df_local.loc[df_local[url_column] == sib_url, embedding_col]
                            if not sib_emb_row.empty:
                                sib_emb = sib_emb_row.iloc[0]
                                # Calcular similitud coseno
                                sim = float(
                                    np.dot(source_emb, sib_emb)
                                    / (np.linalg.norm(source_emb) * np.linalg.norm(sib_emb) + 1e-10)
                                )
                                sibling_similarities.append((sib_url, sim))

                        # Ordenar por similitud descendente
                        sibling_similarities.sort(key=lambda x: x[1], reverse=True)
                        # Tomar top-k más similares
                        siblings_to_link = [url for url, _ in sibling_similarities[:max_links_per_parent]]
                    else:
                        # Si no hay embedding para source, usar orden original
                        siblings_to_link = siblings_filtered[:max_links_per_parent]

                except Exception:
                    # En caso de error, usar orden original sin priorización
                    siblings_to_link = siblings_filtered[:max_links_per_parent]
            else:
                # Sin priorización semántica, tomar los primeros N
                siblings_to_link = siblings_filtered[:max_links_per_parent]

            # Crear enlaces a hermanos
            for sibling_url in siblings_to_link:
                recommendations.append(
                    {
                        "Origen URL": source_url,
                        "Destino URL": sibling_url,
                        "Estrategia": "Estructural horizontal (hermanos)",
                        "Anchor Text Sugerido": generate_contextual_anchor("", parent_path, sibling_url),
                        "Link Weight": float(link_weight) * 0.9,  # Peso ligeramente menor
                    }
                )

    # ========================================================================
    # ESTRATEGIA 3: ENLACES DESCENDENTES (Destacados)
    # ========================================================================
    # Cada padre enlaza a sus hijos más importantes
    for parent_path, children_urls in parent_to_children.items():
        parent_candidates = path_to_urls.get(parent_path, [])
        # Filtrar candidatos excluidos como origen
        parent_candidates = [p for p in parent_candidates if p not in excluded_as_source_urls]
        parent_url = parent_candidates[0] if parent_candidates else None

        if not parent_url:
            continue

        # Filtrar hijos excluidos como destino y limitar
        children_filtered = [c for c in children_urls if c not in excluded_as_target_urls]
        limited_children = children_filtered[:max_links_per_parent]

        for child_url in limited_children:
            recommendations.append(
                {
                    "Origen URL": parent_url,
                    "Destino URL": child_url,
                    "Estrategia": "Estructural descendente (destacados)",
                    "Anchor Text Sugerido": generate_contextual_anchor(
                        "", child_url.split("/")[-2] if "/" in child_url else "", child_url
                    ),
                    "Link Weight": float(link_weight) * 0.85,  # Peso ligeramente menor
                }
            )

    if not recommendations:
        return pd.DataFrame(
            columns=[
                "Origen URL",
                "Destino URL",
                "Estrategia",
                "Anchor Text Sugerido",
                "Link Weight",
            ]
        )

    # Eliminar duplicados
    return pd.DataFrame(recommendations).drop_duplicates(subset=["Origen URL", "Destino URL", "Estrategia"])


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = ["structural_taxonomy_linking"]
