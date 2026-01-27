"""
Módulo para calcular el Semantic Depth Score (SDS).

El SDS es una métrica avanzada que cuantifica la calidad del contenido mediante:
1. Score_ER: Relevancia y densidad de entidades (Entity Relevance)
2. Score_CV: Cohesión vectorial del discurso (Vector Cohesion)

Basado en la investigación sobre profundidad semántica y la patente US9183499B1
de Google sobre Entity Quality Score.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity


# Pesos por tipo de entidad según dominio
ENTITY_TYPE_WEIGHTS = {
    "ORG": 0.30,
    "PERSON": 0.25,
    "PRODUCT": 0.35,
    "GPE": 0.15,
    "LOC": 0.15,
    "FAC": 0.20,
    "EVENT": 0.25,
    "LAW": 0.20,
    "NORP": 0.10,
    "LANGUAGE": 0.10,
    "WORK_OF_ART": 0.25,
    "DATE": 0.05,
    "TIME": 0.05,
    "PERCENT": 0.08,
    "MONEY": 0.08,
    "QUANTITY": 0.10,
    "ORDINAL": 0.05,
    "CARDINAL": 0.05,
    # Tipos biomédicos específicos
    "DISEASE": 0.40,
    "SYMPTOM": 0.35,
    "MEDICATION": 0.40,
    "TREATMENT": 0.35,
    "ANATOMY": 0.30,
}


def normalize_min_max(value: float, min_val: float, max_val: float) -> float:
    """
    Normaliza un valor al rango [0, 1] usando Min-Max normalization.

    Args:
        value: Valor a normalizar
        min_val: Valor mínimo del rango
        max_val: Valor máximo del rango

    Returns:
        Valor normalizado entre 0 y 1
    """
    if max_val == min_val:
        return 0.0
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


def calculate_entity_density(entities: List[Dict], text_length: int) -> float:
    """
    Calcula la Densidad Bruta de Entidades (DBE).

    DBE = Entidades Únicas / Longitud del Texto

    Args:
        entities: Lista de entidades identificadas con formato:
                  [{"text": str, "label": str, "start": int, "end": int}, ...]
        text_length: Longitud del texto en tokens o caracteres

    Returns:
        Densidad bruta de entidades
    """
    if text_length == 0:
        return 0.0

    # Contar entidades únicas (por texto normalizado)
    unique_entities = set()
    for entity in entities:
        entity_text = entity.get("text", "").strip().lower()
        if entity_text:
            unique_entities.add(entity_text)

    return len(unique_entities) / text_length


def calculate_entity_co_occurrence(
    entities: List[Dict],
    window_size: int = 5
) -> float:
    """
    Calcula la puntuación de co-ocurrencia de entidades dentro de ventanas de proximidad.

    Mide cuántas entidades aparecen cerca unas de otras, lo cual indica
    profundidad temática según la patente US9183499B1 (Quality Smear).

    Args:
        entities: Lista de entidades con posiciones
        window_size: Tamaño de la ventana de proximidad (en número de entidades)

    Returns:
        Puntuación de co-ocurrencia normalizada
    """
    if len(entities) < 2:
        return 0.0

    # Ordenar entidades por posición de inicio
    sorted_entities = sorted(entities, key=lambda x: x.get("start", 0))

    # Contar co-ocurrencias dentro de ventanas
    co_occurrence_count = 0
    total_windows = 0

    for i in range(len(sorted_entities)):
        # Ventana: entidades desde i hasta i+window_size
        window_end = min(i + window_size, len(sorted_entities))
        window_entities = sorted_entities[i:window_end]

        if len(window_entities) > 1:
            # Contar pares únicos en esta ventana
            entity_texts = set(e.get("text", "").strip().lower() for e in window_entities)
            co_occurrence_count += len(entity_texts) - 1
            total_windows += 1

    if total_windows == 0:
        return 0.0

    return co_occurrence_count / total_windows


def calculate_weighted_entity_relevance(
    entities: List[Dict],
    entity_weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Calcula la Relevancia Contextual Ponderada (RCP) de las entidades.

    Pondera las entidades según su tipo e importancia para el dominio.

    Args:
        entities: Lista de entidades identificadas
        entity_weights: Diccionario de pesos por tipo de entidad

    Returns:
        Puntuación de relevancia ponderada
    """
    if not entities:
        return 0.0

    if entity_weights is None:
        entity_weights = ENTITY_TYPE_WEIGHTS

    total_weight = 0.0
    for entity in entities:
        entity_type = entity.get("label", "")
        weight = entity_weights.get(entity_type, 0.10)  # Peso por defecto: 0.10
        total_weight += weight

    # Normalizar por el número de entidades
    return total_weight / len(entities)


def calculate_score_er(
    entities: List[Dict],
    text_length: int,
    entity_weights: Optional[Dict[str, float]] = None,
    reference_min: float = 0.0,
    reference_max: float = 1.0
) -> float:
    """
    Calcula el Score_ER (Entity Relevance Score).

    Score_ER = Normalizar(DBE × ∑ Ponderación(Tipo) × Co-ocurrencia)

    Args:
        entities: Lista de entidades identificadas
        text_length: Longitud del texto en tokens
        entity_weights: Pesos personalizados por tipo de entidad
        reference_min: Valor mínimo de referencia para normalización
        reference_max: Valor máximo de referencia para normalización

    Returns:
        Puntuación normalizada de relevancia de entidades [0, 1]
    """
    if not entities or text_length == 0:
        return 0.0

    # Componente 1: Densidad Bruta de Entidades (DBE)
    dbe = calculate_entity_density(entities, text_length)

    # Componente 2: Co-ocurrencia de entidades
    co_occurrence = calculate_entity_co_occurrence(entities, window_size=5)

    # Componente 3: Relevancia ponderada por tipo
    weighted_relevance = calculate_weighted_entity_relevance(entities, entity_weights)

    # Score bruto: DBE × (Co-ocurrencia + Relevancia ponderada)
    raw_score = dbe * (co_occurrence + weighted_relevance)

    # Normalizar al rango [0, 1]
    normalized_score = normalize_min_max(raw_score, reference_min, reference_max)

    return normalized_score


def calculate_vector_cohesion(embeddings: np.ndarray) -> float:
    """
    Calcula la cohesión vectorial del discurso mediante distancia al centroide.

    Mide qué tan cerca están todos los vectores (frases/párrafos) del
    vector centroide (significado promedio del documento).

    D = Promedio(Distancia Coseno(v_i, C_avg))
    Score_CV = 1 - D

    Args:
        embeddings: Matriz de embeddings de forma (N, d) donde:
                    N = número de unidades de texto (frases/párrafos)
                    d = dimensionalidad del embedding

    Returns:
        Puntuación de cohesión vectorial [0, 1]
        Valores cercanos a 1 indican alta cohesión (baja dispersión)
    """
    if embeddings is None or len(embeddings) == 0:
        return 0.0

    # Si solo hay un embedding, cohesión perfecta
    if len(embeddings) == 1:
        return 1.0

    # Calcular el centroide (vector promedio)
    centroid = np.mean(embeddings, axis=0)

    # Calcular la similitud coseno de cada vector con el centroide
    # Reshape para asegurar dimensiones correctas
    centroid_2d = centroid.reshape(1, -1)

    # Similitudes de todos los vectores con el centroide
    similarities = cosine_similarity(embeddings, centroid_2d).flatten()

    # Convertir similitudes a distancias (1 - similitud)
    distances = 1 - similarities

    # Distancia promedio
    avg_distance = np.mean(distances)

    # Score de cohesión: inversa de la distancia
    # Valores cercanos a 0 de distancia → cohesión cercana a 1
    cohesion_score = 1 - avg_distance

    return max(0.0, min(1.0, cohesion_score))


def calculate_score_cv(
    embeddings: np.ndarray,
    reference_min: float = 0.0,
    reference_max: float = 1.0
) -> float:
    """
    Calcula el Score_CV (Vector Cohesion Score) normalizado.

    Args:
        embeddings: Matriz de embeddings de unidades de texto
        reference_min: Valor mínimo de referencia para normalización
        reference_max: Valor máximo de referencia para normalización

    Returns:
        Puntuación normalizada de cohesión vectorial [0, 1]
    """
    raw_cohesion = calculate_vector_cohesion(embeddings)

    # Normalizar al rango de referencia
    normalized_score = normalize_min_max(raw_cohesion, reference_min, reference_max)

    return normalized_score


def calculate_semantic_depth_score(
    entities: List[Dict],
    text_length: int,
    embeddings: Optional[np.ndarray] = None,
    w_er: float = 0.5,
    w_cv: float = 0.5,
    entity_weights: Optional[Dict[str, float]] = None,
    reference_params: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, float]:
    """
    Calcula el Semantic Depth Score (SDS) completo.

    SDS = (W_ER × Score_ER) + (W_CV × Score_CV)

    Args:
        entities: Lista de entidades identificadas en el texto
        text_length: Longitud del texto en tokens
        embeddings: Matriz de embeddings de frases/párrafos (opcional)
        w_er: Peso para Score_ER (por defecto 0.5)
        w_cv: Peso para Score_CV (por defecto 0.5)
        entity_weights: Pesos personalizados por tipo de entidad
        reference_params: Parámetros de referencia para normalización:
                         {"er": (min, max), "cv": (min, max)}

    Returns:
        Diccionario con:
        - "sds": Semantic Depth Score final [0, 100]
        - "score_er": Entity Relevance Score [0, 1]
        - "score_cv": Vector Cohesion Score [0, 1]
        - "classification": Clasificación de calidad
        - "entity_density": Densidad bruta de entidades
        - "entity_count": Número de entidades únicas
        - "cohesion_raw": Cohesión vectorial sin normalizar
    """
    # Validar pesos
    if not np.isclose(w_er + w_cv, 1.0):
        raise ValueError(f"Los pesos deben sumar 1.0. Actual: w_er={w_er}, w_cv={w_cv}")

    # Parámetros de referencia por defecto
    if reference_params is None:
        reference_params = {
            "er": (0.0, 2.0),  # Rango típico observado para Score_ER bruto
            "cv": (0.0, 1.0),  # Score_CV ya está en [0, 1]
        }

    # Calcular Score_ER
    er_min, er_max = reference_params.get("er", (0.0, 2.0))
    score_er = calculate_score_er(
        entities=entities,
        text_length=text_length,
        entity_weights=entity_weights,
        reference_min=er_min,
        reference_max=er_max
    )

    # Calcular Score_CV
    score_cv = 0.0
    cohesion_raw = 0.0
    if embeddings is not None and len(embeddings) > 0:
        cv_min, cv_max = reference_params.get("cv", (0.0, 1.0))
        cohesion_raw = calculate_vector_cohesion(embeddings)
        score_cv = normalize_min_max(cohesion_raw, cv_min, cv_max)

    # Calcular SDS final
    sds_normalized = (w_er * score_er) + (w_cv * score_cv)

    # Escalar a rango [0, 100]
    sds_final = sds_normalized * 100

    # Clasificación de calidad según rangos
    if sds_final < 34:
        classification = "Contenido Thin o Irrelevante"
    elif sds_final < 67:
        classification = "Calidad Decente o Sesgada"
    else:
        classification = "Calidad Óptima y Relevante"

    # Calcular métricas adicionales
    unique_entities = set(e.get("text", "").strip().lower() for e in entities if e.get("text"))
    entity_density = len(unique_entities) / text_length if text_length > 0 else 0.0

    return {
        "sds": round(sds_final, 2),
        "score_er": round(score_er, 4),
        "score_cv": round(score_cv, 4),
        "classification": classification,
        "entity_density": round(entity_density, 4),
        "entity_count": len(unique_entities),
        "cohesion_raw": round(cohesion_raw, 4),
        "w_er": w_er,
        "w_cv": w_cv,
    }


def analyze_document_sds(
    text: str,
    entities: List[Dict],
    embeddings: Optional[np.ndarray] = None,
    **kwargs
) -> Dict[str, float]:
    """
    Función de conveniencia para analizar un documento completo.

    Args:
        text: Texto completo del documento
        entities: Entidades identificadas por NER
        embeddings: Embeddings de frases/párrafos del documento
        **kwargs: Parámetros adicionales para calculate_semantic_depth_score

    Returns:
        Resultado del SDS con todas las métricas
    """
    # Calcular longitud en tokens (aproximación simple)
    text_length = len(text.split())

    return calculate_semantic_depth_score(
        entities=entities,
        text_length=text_length,
        embeddings=embeddings,
        **kwargs
    )


def batch_analyze_sds(
    documents: List[Dict[str, any]],
    w_er: float = 0.5,
    w_cv: float = 0.5
) -> pd.DataFrame:
    """
    Analiza múltiples documentos en batch y devuelve un DataFrame.

    Args:
        documents: Lista de diccionarios con formato:
                   [{"text": str, "entities": List[Dict], "embeddings": np.ndarray}, ...]
        w_er: Peso para Score_ER
        w_cv: Peso para Score_CV

    Returns:
        DataFrame con resultados SDS para cada documento
    """
    results = []

    for i, doc in enumerate(documents):
        text = doc.get("text", "")
        entities = doc.get("entities", [])
        embeddings = doc.get("embeddings", None)

        try:
            sds_result = analyze_document_sds(
                text=text,
                entities=entities,
                embeddings=embeddings,
                w_er=w_er,
                w_cv=w_cv
            )
            sds_result["doc_id"] = i
            sds_result["success"] = True
        except Exception as e:
            sds_result = {
                "doc_id": i,
                "sds": 0.0,
                "score_er": 0.0,
                "score_cv": 0.0,
                "classification": "Error",
                "error": str(e),
                "success": False
            }

        results.append(sds_result)

    return pd.DataFrame(results)


__all__ = [
    "calculate_semantic_depth_score",
    "calculate_score_er",
    "calculate_score_cv",
    "calculate_entity_density",
    "calculate_entity_co_occurrence",
    "calculate_weighted_entity_relevance",
    "calculate_vector_cohesion",
    "analyze_document_sds",
    "batch_analyze_sds",
    "ENTITY_TYPE_WEIGHTS",
]
