"""
Módulo de análisis de datos para informes de posiciones SEO.

Este módulo contiene funciones para asignar familias a keywords y generar resúmenes estadísticos.
"""

from __future__ import annotations

import re
from typing import Dict, List

import pandas as pd

from .positions_parsing import normalize_domain


def assign_keyword_families(df: pd.DataFrame, families_text: str) -> pd.DataFrame:
    """
    Asigna familias a las keywords basándose en reglas textuales.

    Formatos de patrones soportados:
    - Coincidencia exacta: "keyword"
    - Coincidencia parcial inicio: "patron*"
    - Coincidencia parcial fin: "*patron"
    - Coincidencia parcial central: "*patron*"

    Args:
        df: DataFrame con columna "Keyword"
        families_text: Texto con definiciones de familias formato "Familia: keyword1, keyword2, *patron*"

    Returns:
        DataFrame con columna "Familia" añadida

    Examples:
        >>> df = pd.DataFrame({"Keyword": ["comprar zapatos", "zapatos rojos", "botas"]})
        >>> families_text = "Zapatos: *zapatos*\\nBotas: botas"
        >>> result = assign_keyword_families(df, families_text)
        >>> result["Familia"].tolist()
        ['Zapatos', 'Zapatos', 'Botas']
    """
    df = df.copy()
    df["Familia"] = "Sin familia"

    # Parsear definiciones de familias
    families_rules: Dict[str, List[str]] = {}
    for line in families_text.strip().split("\n"):
        if ":" not in line:
            continue
        family_name, patterns_str = line.split(":", 1)
        family_name = family_name.strip()
        # Separar por comas o punto y coma
        patterns = re.split(r"[,;]", patterns_str)
        families_rules[family_name] = [p.strip() for p in patterns if p.strip()]

    # Asignar familias
    for idx, row in df.iterrows():
        keyword_lower = str(row["Keyword"]).lower()
        for family_name, patterns in families_rules.items():
            for pattern in patterns:
                pattern_lower = pattern.lower()
                # Si tiene asterisco, es coincidencia parcial
                if pattern.startswith("*") and pattern.endswith("*"):
                    pattern_clean = pattern_lower.strip("*")
                    if pattern_clean in keyword_lower:
                        df.at[idx, "Familia"] = family_name
                        break
                elif pattern.startswith("*"):
                    pattern_clean = pattern_lower.strip("*")
                    if keyword_lower.endswith(pattern_clean):
                        df.at[idx, "Familia"] = family_name
                        break
                elif pattern.endswith("*"):
                    pattern_clean = pattern_lower.strip("*")
                    if keyword_lower.startswith(pattern_clean):
                        df.at[idx, "Familia"] = family_name
                        break
                else:
                    # Coincidencia exacta de palabra
                    if pattern_lower == keyword_lower or pattern_lower in keyword_lower.split():
                        df.at[idx, "Familia"] = family_name
                        break
            else:
                continue
            break

    return df


def summarize_positions_overview(
    df: pd.DataFrame,
    brand_domain: str,
    competitor_domains: List[str]
) -> Dict:
    """
    Genera un resumen estadístico general de las posiciones.

    Args:
        df: DataFrame con columnas Keyword, Position, Domain, Familia
        brand_domain: Dominio de la marca principal
        competitor_domains: Lista de dominios competidores

    Returns:
        Diccionario con métricas clave:
        - total_keywords: Total de keywords analizadas
        - brand_keywords_in_top10: Keywords de la marca en top 10
        - brand_average_position: Posición media de la marca
        - top_competitors_by_presence: Lista de competidores más frecuentes en top 10

    Examples:
        >>> df = pd.DataFrame({
        ...     "Keyword": ["kw1", "kw2", "kw3"],
        ...     "Position": [3, 7, 15],
        ...     "Domain": ["example.com", "example.com", "example.com"]
        ... })
        >>> summary = summarize_positions_overview(df, "example.com", [])
        >>> summary["brand_keywords_in_top10"]
        2
    """
    brand_domain_normalized = normalize_domain(brand_domain) if brand_domain else ""

    summary = {
        "total_keywords": len(df),
        "brand_keywords_in_top10": 0,
        "brand_average_position": None,
        "top_competitors_by_presence": []
    }

    if "Domain" not in df.columns:
        return summary

    # Filtrar keywords de la marca
    brand_df = df[df["Domain"] == brand_domain_normalized] if brand_domain_normalized else pd.DataFrame()

    if not brand_df.empty:
        # Keywords en top 10
        summary["brand_keywords_in_top10"] = len(brand_df[brand_df["Position"] <= 10])

        # Posición media
        summary["brand_average_position"] = round(brand_df["Position"].mean(), 1)

    # Competidores más frecuentes en top 10
    top10_df = df[df["Position"] <= 10]
    if not top10_df.empty and "Domain" in top10_df.columns:
        competitor_counts = (
            top10_df["Domain"]
            .value_counts()
            .head(10)
            .reset_index()
        )
        competitor_counts.columns = ["Domain", "Count"]
        # Filtrar el dominio de la marca
        if brand_domain_normalized:
            competitor_counts = competitor_counts[
                competitor_counts["Domain"] != brand_domain_normalized
            ]
        summary["top_competitors_by_presence"] = competitor_counts.values.tolist()

    return summary


__all__ = [
    "assign_keyword_families",
    "summarize_positions_overview",
]
