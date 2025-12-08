"""
Utilidades para filtrado y validación de entidades extraídas por NER.

Este módulo proporciona funciones para eliminar ruido y mejorar la calidad
de las entidades identificadas, filtrando:
- Palabras comunes y stopwords
- Signos de puntuación
- Entidades demasiado cortas
- Nombres genéricos sin valor semántico
"""

from __future__ import annotations

import re
from typing import Set


# Lista de palabras comunes en español que no son entidades significativas
SPANISH_STOPWORDS = {
    "gracias", "hola", "adiós", "adios", "buenos", "buenas", "días", "dias",
    "tardes", "noches", "señor", "senor", "señora", "senora", "don", "doña", "dona",
    "etc", "etcétera", "etcetera", "ejemplo", "ejemplos", "si", "sí", "no",
    "ver", "aquí", "aqui", "allí", "alli", "ahí", "ahi", "más", "mas", "menos",
    "muy", "mucho", "poco", "bastante", "demasiado", "algo", "nada", "todo",
    "todos", "algunas", "algunos", "varias", "varios", "otras", "otros",
    "puede", "pueden", "debe", "deben", "hacer", "hace", "hacen",
    "como", "cómo", "cuando", "cuándo", "donde", "dónde", "porque", "porqué",
    "para", "por", "con", "sin", "sobre", "bajo", "ante", "tras",
    "durante", "mediante", "según", "segun", "entre", "hacia", "hasta",
    "desde", "contra", "dentro", "fuera", "encima", "debajo",
    "sr", "sra", "srta", "dr", "dra", "ing", "lic",
}

# Lista de palabras comunes en inglés
ENGLISH_STOPWORDS = {
    "thanks", "thank", "hello", "hi", "bye", "goodbye", "mr", "mrs", "miss",
    "dr", "prof", "sir", "madam", "yes", "no", "ok", "okay",
    "etc", "example", "examples", "see", "here", "there", "more", "less",
    "very", "much", "little", "something", "nothing", "everything",
    "some", "any", "all", "several", "other", "others",
    "can", "could", "should", "must", "may", "might", "will", "would",
    "how", "when", "where", "why", "what", "which", "who",
    "for", "with", "without", "about", "under", "over", "before", "after",
    "during", "through", "between", "among", "against", "within",
}

# Combinar stopwords de ambos idiomas
STOPWORDS = SPANISH_STOPWORDS | ENGLISH_STOPWORDS

# Nombres muy comunes que generan ruido (sin apellido son poco informativos)
COMMON_NAMES = {
    # Español
    "juan", "maría", "maria", "josé", "jose", "antonio", "manuel", "francisco",
    "david", "daniel", "carlos", "miguel", "alejandro", "pedro", "javier",
    "luis", "sergio", "jorge", "alberto", "pablo", "rafael", "fernando",
    "ana", "carmen", "laura", "isabel", "pilar", "elena", "rosa", "mercedes",
    "alicia", "marta", "cristina", "teresa", "beatriz", "patricia", "silvia",
    # Inglés
    "john", "mary", "james", "robert", "michael", "william", "david", "richard",
    "joseph", "thomas", "charles", "christopher", "daniel", "matthew", "mark",
    "patricia", "jennifer", "linda", "elizabeth", "barbara", "susan", "jessica",
    "sarah", "karen", "nancy", "lisa", "betty", "margaret", "sandra",
}

# Patrones de texto que indican ruido
NOISE_PATTERNS = [
    r"^[^\w\s]+$",  # Solo signos de puntuación
    r"^[\d\s\-\.\,\/]+$",  # Solo números y separadores
    r"^[ivxlcdm]+$",  # Solo números romanos en minúsculas
    r"^\d{1,2}[a-z]?$",  # Números sueltos (1, 2a, 3b, etc.)
    r"^https?://",  # URLs
    r"^www\.",  # URLs sin protocolo
    r"^[@#]\w+",  # Hashtags o menciones
]


def is_noise_pattern(text: str) -> bool:
    """
    Verifica si el texto coincide con algún patrón de ruido.

    Args:
        text: Texto a verificar

    Returns:
        True si es ruido, False si es válido
    """
    text_lower = text.lower().strip()

    for pattern in NOISE_PATTERNS:
        if re.match(pattern, text_lower):
            return True

    return False


def is_valid_entity(
    text: str,
    entity_type: str,
    min_length: int = 2,
    allow_common_names: bool = False,
    custom_stopwords: Set[str] | None = None
) -> bool:
    """
    Valida si una entidad es significativa y no es ruido.

    Args:
        text: Texto de la entidad
        entity_type: Tipo de entidad (PERSON, ORG, etc.)
        min_length: Longitud mínima en caracteres
        allow_common_names: Si True, permite nombres comunes sin apellido
        custom_stopwords: Stopwords personalizadas adicionales

    Returns:
        True si la entidad es válida, False si es ruido
    """
    if not text or not isinstance(text, str):
        return False

    text_clean = text.strip()

    # Filtro 1: Longitud mínima
    if len(text_clean) < min_length:
        return False

    # Filtro 2: Patrones de ruido
    if is_noise_pattern(text_clean):
        return False

    text_lower = text_clean.lower()

    # Filtro 3: Stopwords
    stopwords_to_check = STOPWORDS.copy()
    if custom_stopwords:
        stopwords_to_check.update(custom_stopwords)

    if text_lower in stopwords_to_check:
        return False

    # Filtro 4: Nombres comunes sin apellido (solo para PERSON)
    if entity_type == "PERSON" and not allow_common_names:
        # Si es un nombre de una sola palabra y está en la lista de nombres comunes
        words = text_clean.split()
        if len(words) == 1 and text_lower in COMMON_NAMES:
            return False

    # Filtro 5: Solo puntuación o caracteres especiales
    if not any(c.isalnum() for c in text_clean):
        return False

    # Filtro 6: Demasiados caracteres especiales (más del 50%)
    special_char_count = sum(1 for c in text_clean if not c.isalnum() and not c.isspace())
    if len(text_clean) > 0 and (special_char_count / len(text_clean)) > 0.5:
        return False

    # Filtro 7: Entidades de tipo DATE, TIME, PERCENT, etc. con longitud mínima mayor
    low_value_types = {"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"}
    if entity_type in low_value_types and len(text_clean) < 4:
        return False

    return True


def filter_entities(
    entities: list[dict],
    min_length: int = 2,
    min_frequency: int = 1,
    allow_common_names: bool = False,
    custom_stopwords: Set[str] | None = None,
    excluded_types: Set[str] | None = None
) -> list[dict]:
    """
    Filtra una lista de entidades eliminando ruido.

    Args:
        entities: Lista de entidades con formato:
                  [{"text": str, "label": str, ...}, ...]
        min_length: Longitud mínima en caracteres
        min_frequency: Frecuencia mínima para mantener la entidad
        allow_common_names: Si True, permite nombres comunes
        custom_stopwords: Stopwords personalizadas adicionales
        excluded_types: Tipos de entidad a excluir completamente

    Returns:
        Lista filtrada de entidades
    """
    if not entities:
        return []

    # Contar frecuencias
    entity_frequency = {}
    for entity in entities:
        text = entity.get("text", "").strip().lower()
        if text:
            entity_frequency[text] = entity_frequency.get(text, 0) + 1

    # Filtrar
    filtered = []
    for entity in entities:
        text = entity.get("text", "")
        entity_type = entity.get("label", "")

        # Verificar tipo excluido
        if excluded_types and entity_type in excluded_types:
            continue

        # Validar entidad
        if not is_valid_entity(
            text=text,
            entity_type=entity_type,
            min_length=min_length,
            allow_common_names=allow_common_names,
            custom_stopwords=custom_stopwords
        ):
            continue

        # Verificar frecuencia mínima
        text_lower = text.strip().lower()
        if entity_frequency.get(text_lower, 0) < min_frequency:
            continue

        filtered.append(entity)

    return filtered


def normalize_entity_text(text: str) -> str:
    """
    Normaliza el texto de una entidad eliminando artefactos comunes.

    Args:
        text: Texto de la entidad

    Returns:
        Texto normalizado
    """
    if not text:
        return ""

    # Eliminar espacios múltiples
    normalized = re.sub(r'\s+', ' ', text)

    # Eliminar comillas alrededor (simples, dobles y curvadas)
    # Comillas rectas
    normalized = normalized.strip('"').strip("'")
    # Comillas curvadas usando códigos Unicode
    normalized = normalized.strip('\u201c\u201d\u2018\u2019')

    # Eliminar puntuación al final (pero no dentro)
    normalized = normalized.rstrip('.,;:!?¡¿')

    # Eliminar espacios
    normalized = normalized.strip()

    return normalized


def get_entity_quality_score(entity: dict) -> float:
    """
    Calcula un score de calidad para una entidad (0-1).

    Score más alto = entidad más valiosa/significativa.

    Args:
        entity: Diccionario de entidad con "text" y "label"

    Returns:
        Score de calidad entre 0 y 1
    """
    text = entity.get("text", "").strip()
    entity_type = entity.get("label", "")

    if not text:
        return 0.0

    score = 1.0

    # Penalizar entidades muy cortas
    if len(text) < 3:
        score *= 0.3
    elif len(text) < 5:
        score *= 0.6

    # Penalizar nombres comunes sin apellido
    if entity_type == "PERSON":
        words = text.split()
        if len(words) == 1 and text.lower() in COMMON_NAMES:
            score *= 0.2
        elif len(words) >= 2:
            score *= 1.2  # Bonus para nombres completos

    # Bonus para entidades de alto valor
    high_value_types = {"ORG", "PRODUCT", "EVENT", "FAC", "WORK_OF_ART", "LAW"}
    if entity_type in high_value_types:
        score *= 1.3

    # Penalizar entidades de bajo valor
    low_value_types = {"DATE", "TIME", "PERCENT", "MONEY", "CARDINAL", "ORDINAL"}
    if entity_type in low_value_types:
        score *= 0.5

    # Penalizar si está en stopwords
    if text.lower() in STOPWORDS:
        score *= 0.1

    # Bonus por longitud razonable (no demasiado larga)
    if 5 <= len(text) <= 50:
        score *= 1.1
    elif len(text) > 100:
        score *= 0.7

    # Normalizar al rango [0, 1]
    return min(1.0, max(0.0, score))


__all__ = [
    "is_valid_entity",
    "is_noise_pattern",
    "filter_entities",
    "normalize_entity_text",
    "get_entity_quality_score",
    "STOPWORDS",
    "COMMON_NAMES",
]
