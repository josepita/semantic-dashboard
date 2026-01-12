"""
Módulo para construcción de payloads para informes de posiciones SEO.

Este módulo contiene funciones para construir estructuras de datos agregadas
para los informes HTML y análisis con IA.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from .positions_parsing import normalize_domain


def build_family_payload(
    df: pd.DataFrame,
    brand_domain: str,
    max_keywords_per_family: int = 20,
    competitor_domains: Optional[List[str]] = None
) -> List[Dict]:
    """
    Construye el payload de familias para el informe.

    Agrega métricas por familia incluyendo posición media, keywords en top 10,
    y volumen de búsqueda si está disponible.

    Args:
        df: DataFrame con columnas Keyword, Position, Domain, Familia, SearchVolume (opcional)
        brand_domain: Dominio de la marca principal
        max_keywords_per_family: Máximo de keywords a incluir por familia
        competitor_domains: Lista de dominios competidores (opcional)

    Returns:
        Lista de diccionarios con información agregada por familia:
        - nombre: Nombre de la familia
        - total_keywords: Total de keywords en la familia
        - keywords: Lista de keywords
        - posicion_media: Posición media de todas las keywords
        - keywords_marca_top10: Keywords de la marca en top 10
        - volumen_total: Volumen total (si SearchVolume disponible)
        - volumen_medio: Volumen medio (si SearchVolume disponible)
        - volumen_marca: Volumen de la marca (si SearchVolume disponible)

    Examples:
        >>> df = pd.DataFrame({
        ...     "Keyword": ["kw1", "kw2", "kw3"],
        ...     "Position": [3, 7, 15],
        ...     "Domain": ["example.com", "example.com", "example.com"],
        ...     "Familia": ["Familia1", "Familia1", "Familia2"],
        ...     "SearchVolume": [1000, 500, 300]
        ... })
        >>> payload = build_family_payload(df, "example.com")
        >>> len(payload)
        2
    """
    if "Familia" not in df.columns:
        return []

    brand_domain_normalized = normalize_domain(brand_domain) if brand_domain else ""
    families_payload = []

    for family_name in df["Familia"].unique():
        if family_name == "Sin familia":
            continue

        family_df = df[df["Familia"] == family_name].copy()

        # Limitar keywords por familia
        family_df = family_df.head(max_keywords_per_family)

        # Calcular métricas
        family_info = {
            "nombre": family_name,
            "total_keywords": len(family_df),
            "keywords": family_df["Keyword"].tolist(),
        }

        if "Position" in family_df.columns:
            family_info["posicion_media"] = round(family_df["Position"].mean(), 1)

        if "Domain" in family_df.columns and brand_domain_normalized:
            brand_in_family = family_df[family_df["Domain"] == brand_domain_normalized]
            family_info["keywords_marca_top10"] = len(
                brand_in_family[brand_in_family["Position"] <= 10]
            )

        # Agregar métricas de volumen de búsqueda si están disponibles
        if "SearchVolume" in family_df.columns:
            total_volume = family_df["SearchVolume"].sum()
            avg_volume = family_df["SearchVolume"].mean()
            family_info["volumen_total"] = int(total_volume)
            family_info["volumen_medio"] = round(avg_volume, 0)

            # Si hay datos de dominio, calcular volumen potencial de la marca
            if "Domain" in family_df.columns and brand_domain_normalized:
                brand_in_family = family_df[family_df["Domain"] == brand_domain_normalized]
                if not brand_in_family.empty:
                    family_info["volumen_marca"] = int(brand_in_family["SearchVolume"].sum())

        families_payload.append(family_info)

    return families_payload


def build_competitive_family_payload(
    df: pd.DataFrame,
    brand_domain: str,
    competitor_domains: Optional[List[str]] = None
) -> List[Dict]:
    """
    Construye payload competitivo con posiciones de todos los dominios por keyword.

    Este payload es usado para generar tablas comparativas keyword-by-keyword
    mostrando las posiciones de la marca vs competidores.

    Args:
        df: DataFrame con columnas Keyword, Position, Domain, SearchVolume (opcional), Familia
        brand_domain: Dominio de la marca principal
        competitor_domains: Lista de dominios competidores a incluir

    Returns:
        Lista de diccionarios con estructura competitiva por familia:
        - nombre: Nombre de la familia
        - total_keywords: Total de keywords
        - keywords_data: Lista de keywords con:
            - keyword: Nombre de la keyword
            - volume: Volumen de búsqueda
            - positions: Dict con posiciones por dominio {domain: position}
        - domains: Lista de todos los dominios
        - brand_domain: Dominio de la marca normalizado
        - volumen_total: Volumen total de la familia
        - volumen_medio: Volumen medio por keyword
        - posicion_media_marca: Posición media de la marca en esta familia

    Examples:
        >>> df = pd.DataFrame({
        ...     "Keyword": ["kw1", "kw1", "kw2"],
        ...     "Position": [3, 5, 7],
        ...     "Domain": ["example.com", "competitor.com", "example.com"],
        ...     "Familia": ["Familia1", "Familia1", "Familia1"],
        ...     "SearchVolume": [1000, 1000, 500]
        ... })
        >>> payload = build_competitive_family_payload(df, "example.com", ["competitor.com"])
        >>> payload[0]["keywords_data"][0]["positions"]["example.com"]
        3
    """
    if "Familia" not in df.columns:
        return []

    brand_domain_normalized = normalize_domain(brand_domain) if brand_domain else ""

    # Normalizar dominios competidores
    all_domains = [brand_domain_normalized]
    if competitor_domains:
        all_domains.extend([normalize_domain(d) for d in competitor_domains if d])

    families_payload = []

    for family_name in df["Familia"].unique():
        if family_name == "Sin familia":
            continue

        family_df = df[df["Familia"] == family_name].copy()

        # Obtener keywords únicas en esta familia
        unique_keywords = family_df["Keyword"].unique()

        keywords_data = []
        for kw in unique_keywords:
            kw_rows = family_df[family_df["Keyword"] == kw]

            # Obtener volumen (debería ser el mismo para todas las filas de la keyword)
            volume = 0
            if "SearchVolume" in kw_rows.columns:
                volume = kw_rows["SearchVolume"].iloc[0] if not kw_rows.empty else 0

            # Construir diccionario de posiciones por dominio
            positions = {}
            for domain in all_domains:
                domain_rows = kw_rows[kw_rows["Domain"] == domain]
                if not domain_rows.empty:
                    # Tomar la mejor posición si hay múltiples
                    positions[domain] = int(domain_rows["Position"].min())
                else:
                    positions[domain] = None  # No encontrado

            keywords_data.append({
                "keyword": kw,
                "volume": int(volume),
                "positions": positions
            })

        # Calcular métricas agregadas
        total_volume = sum(k["volume"] for k in keywords_data)
        avg_volume = total_volume / len(keywords_data) if keywords_data else 0

        # Posición media de la marca
        brand_positions = [k["positions"].get(brand_domain_normalized) for k in keywords_data
                          if k["positions"].get(brand_domain_normalized) is not None]
        avg_position = sum(brand_positions) / len(brand_positions) if brand_positions else None

        families_payload.append({
            "nombre": family_name,
            "total_keywords": len(keywords_data),
            "keywords_data": keywords_data,
            "domains": all_domains,
            "brand_domain": brand_domain_normalized,
            "volumen_total": total_volume,
            "volumen_medio": round(avg_volume, 0),
            "posicion_media_marca": round(avg_position, 1) if avg_position else None
        })

    return families_payload


__all__ = [
    "build_family_payload",
    "build_competitive_family_payload",
]
