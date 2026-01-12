"""
Linking Utilities
=================

Utilidades para reporting, interpretación con Gemini AI y helpers generales
para el laboratorio de enlazado interno.

Autor: Embedding Insights
Versión: 1.0.0
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Sequence

import pandas as pd
import streamlit as st

try:
    import google.generativeai as genai
except ModuleNotFoundError:
    genai = None


# ============================================================================
# HELPERS GENERALES
# ============================================================================

def guess_default_type(values: Sequence[str], keywords: Sequence[str]) -> Optional[str]:
    """
    Busca el primer valor que contenga alguno de los keywords indicados.

    Útil para auto-detectar tipos de página en selectboxes.

    Args:
        values: Lista de valores posibles (ej: tipos de página únicos)
        keywords: Palabras clave a buscar (ej: ['servicio', 'product'])

    Returns:
        Primer valor que coincida o None

    Example:
        >>> types = ['blog', 'servicio', 'categoria', 'producto']
        >>> guess_default_type(types, ['servicio', 'product'])
        'servicio'
        >>> guess_default_type(types, ['landing', 'home'])
        None
    """
    for keyword in keywords:
        for value in values:
            if keyword in value.lower():
                return value
    return None


# ============================================================================
# CONSTRUCCIÓN DE PAYLOADS PARA REPORTING
# ============================================================================

def build_entity_payload_from_doc_relations(doc_relations_df: Optional[pd.DataFrame]) -> Dict[str, Dict[str, float]]:
    """
    Convierte DataFrame de relaciones página-entidad a payload para enlazado híbrido.

    Estructura de entrada (doc_relations_df):
    - URL: URL de la página
    - QID o Entidad: ID de la entidad
    - Prominence documento: Score de prominencia
    - Frecuencia documento: Fallback si no hay prominence

    Args:
        doc_relations_df: DataFrame con relaciones página-entidad

    Returns:
        Diccionario {url: {entidad_id: prominence}}

    Example:
        >>> df = pd.DataFrame({
        ...     'URL': ['page1', 'page1', 'page2'],
        ...     'QID': ['Q123', 'Q456', 'Q123'],
        ...     'Prominence documento': [0.8, 0.5, 0.9]
        ... })
        >>> build_entity_payload_from_doc_relations(df)
        {'page1': {'Q123': 0.8, 'Q456': 0.5}, 'page2': {'Q123': 0.9}}
    """
    payload: Dict[str, Dict[str, float]] = {}

    if doc_relations_df is None or doc_relations_df.empty:
        return payload

    for _, row in doc_relations_df.iterrows():
        # Extraer URL
        url_value = str(row.get("URL", "")).strip()
        if not url_value:
            continue

        # Extraer entity ID (QID preferido, fallback a Entidad)
        entity_id = str(row.get("QID") or "").strip()
        if not entity_id or entity_id in {"—", "-", "None", "nan"}:
            entity_id = str(row.get("Entidad") or "").strip()
        if not entity_id:
            continue

        # Extraer prominence (con fallback a frecuencia)
        prominence = row.get("Prominence documento")
        try:
            prominence_value = float(prominence)
        except (TypeError, ValueError):
            freq_value = row.get("Frecuencia documento", 0)
            try:
                prominence_value = float(freq_value)
            except (TypeError, ValueError):
                prominence_value = 0.0

        if prominence_value <= 0:
            continue

        # Acumular prominence por URL y entidad
        url_bucket = payload.setdefault(url_value, {})
        url_bucket[entity_id] = url_bucket.get(entity_id, 0.0) + prominence_value

    return payload


def build_linking_reports_payload(max_rows: int = 40) -> Dict[str, object]:
    """
    Recopila resultados del laboratorio de enlazado desde session_state para análisis con Gemini.

    Limita el número de filas para no saturar el prompt.

    Args:
        max_rows: Máximo de filas a incluir por reporte

    Returns:
        Diccionario con structure:
        {
            'basic': {
                'total_rows': int,
                'columns': List[str],
                'sample': List[Dict] (hasta max_rows)
            },
            'advanced': {..., 'orphans': List[str]},
            'hybrid_cls': {..., 'orphans': List[str], 'pagerank': List[Tuple[str, float]]},
            'structural': {...}
        }

    Example:
        >>> # En Streamlit app después de generar reportes
        >>> payload = build_linking_reports_payload(max_rows=20)
        >>> 'basic' in payload
        True
        >>> len(payload['basic']['sample']) <= 20
        True
    """
    payload: Dict[str, object] = {}

    def _add(tag: str, df: Optional[pd.DataFrame]) -> None:
        """Helper para añadir DataFrame al payload."""
        if isinstance(df, pd.DataFrame) and not df.empty:
            payload[tag] = {
                "total_rows": int(len(df)),
                "columns": list(df.columns),
                "sample": df.head(max_rows).to_dict("records"),
            }

    # Añadir reportes básicos
    _add("basic", st.session_state.get("linking_basic_report"))
    _add("advanced", st.session_state.get("linking_adv_report"))
    _add("hybrid_cls", st.session_state.get("linking_hybrid_report"))
    _add("structural", st.session_state.get("linking_structural_report"))

    # Añadir orphans del modo avanzado
    adv_orphans = st.session_state.get("linking_adv_orphans") or []
    if adv_orphans:
        payload.setdefault("advanced", {}).update({"orphans": adv_orphans[:max_rows]})

    # Añadir orphans y PageRank del modo híbrido
    hybrid_orphans = st.session_state.get("linking_hybrid_orphans") or []
    if hybrid_orphans:
        payload.setdefault("hybrid_cls", {}).update({"orphans": hybrid_orphans[:max_rows]})

    pagerank_scores = st.session_state.get("linking_hybrid_pagerank") or {}
    if pagerank_scores:
        sorted_scores = sorted(pagerank_scores.items(), key=lambda item: item[1], reverse=True)[:max_rows]
        payload.setdefault("hybrid_cls", {}).update({"pagerank": sorted_scores})

    return payload


# ============================================================================
# INTERPRETACIÓN CON GEMINI AI
# ============================================================================

def interpret_linking_reports_with_gemini(
    api_key: str,
    model_name: str,
    payload: Dict[str, object],
    extra_notes: str = "",
) -> str:
    """
    Envía resultados del laboratorio a Gemini para obtener conclusiones estratégicas.

    Genera análisis de alto nivel con:
    - Insights generales por cada modo evaluado
    - Riesgos detectados (gaps, páginas sin cobertura, etc.)
    - Acciones recomendadas

    Args:
        api_key: API key de Google Gemini
        model_name: Modelo a usar (ej: 'gemini-2.0-flash-exp')
        payload: Payload de reportes construido con build_linking_reports_payload()
        extra_notes: Notas adicionales del usuario para contexto

    Returns:
        Texto con análisis y recomendaciones en español

    Raises:
        RuntimeError: Si google-generativeai no está instalado
        ValueError: Si falta API key o no hay resultados

    Example:
        >>> payload = build_linking_reports_payload()
        >>> analysis = interpret_linking_reports_with_gemini(
        ...     api_key='AIza...',
        ...     model_name='gemini-2.0-flash-exp',
        ...     payload=payload,
        ...     extra_notes='Priorizar conversión en servicios'
        ... )
        >>> 'Insight' in analysis
        True
    """
    if genai is None:
        raise RuntimeError("Instala `google-generativeai` para usar la interpretación con Gemini.")

    cleaned_key = (api_key or "").strip()
    if not cleaned_key:
        raise ValueError("Introduce una API key válida de Gemini (panel lateral).")

    cleaned_model = (model_name or "gemini-2.0-flash-exp").strip()

    if not payload:
        raise ValueError("No hay resultados que interpretar todavía.")

    # Configurar Gemini
    genai.configure(api_key=cleaned_key)
    model = genai.GenerativeModel(cleaned_model)

    # Construir prompt
    payload_json = json.dumps(payload, ensure_ascii=False)

    prompt = f"""
Eres un consultor SEO especializado en enlazado interno. Resume los hallazgos del laboratorio
de enlazado y genera conclusiones accionables.

Resultados (JSON recortado):
{payload_json}

Notas adicionales del usuario: {extra_notes or "Sin observaciones"}.

Redacta en español:
- Insight general por cada modo evaluado (básico, avanzado, híbrido, estructural) si hay datos.
- Riesgos detectados (por ejemplo, falta de enlaces, tipos sin cobertura, competidores fuertes).
- Acciones recomendadas para el siguiente sprint.
"""

    # Generar respuesta
    response = model.generate_content(prompt)

    # Extraer texto
    text = getattr(response, "text", "") or ""
    if not text and response.candidates:
        text = "".join(
            getattr(part, "text", "") for part in response.candidates[0].content.parts
        ).strip()

    text = text.strip()

    # Limpiar markdown code blocks si los hay
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*", "", text).strip()
        if text.endswith("```"):
            text = text[:-3].strip()

    return text


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "guess_default_type",
    "build_entity_payload_from_doc_relations",
    "build_linking_reports_payload",
    "interpret_linking_reports_with_gemini",
]
