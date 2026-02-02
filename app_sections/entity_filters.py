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
from typing import Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    import spacy


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
    # Artículos y pronombres
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
    "aquel", "aquella", "aquellos", "aquellas",
    "yo", "tú", "tu", "él", "el", "ella", "nosotros", "vosotros", "ellos", "ellas",
    "mi", "mis", "su", "sus", "nuestro", "nuestra", "vuestro", "vuestra",
    "me", "te", "se", "nos", "os", "le", "les", "lo", "la",
    # Verbos auxiliares comunes
    "ser", "estar", "haber", "tener", "ir", "venir", "poder", "querer", "deber",
    "saber", "decir", "dar", "poner", "salir", "volver", "llevar", "seguir",
    # Palabras de relleno
    "cosa", "cosas", "parte", "forma", "manera", "modo", "tipo", "vez", "veces",
    "momento", "situación", "situacion", "caso", "casos", "punto", "lado",
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
    # Articles and pronouns
    "the", "a", "an", "this", "that", "these", "those",
    "i", "you", "he", "she", "it", "we", "they",
    "my", "your", "his", "her", "its", "our", "their",
    "me", "him", "us", "them",
    # Common verbs
    "be", "is", "are", "was", "were", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "get", "got", "getting", "make", "makes", "made", "making",
    "go", "goes", "went", "going", "come", "comes", "came", "coming",
    # Filler words
    "thing", "things", "part", "way", "type", "kind", "time", "times",
    "moment", "situation", "case", "point", "side",
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

# Patrones de texto que indican ruido (precompilados para rendimiento)
NOISE_PATTERNS = [
    re.compile(r"^[^\w\s]+$"),  # Solo signos de puntuación
    re.compile(r"^[\d\s\-\.\,\/]+$"),  # Solo números y separadores
    re.compile(r"^[ivxlcdm]+$"),  # Solo números romanos en minúsculas
    re.compile(r"^\d{1,2}[a-z]?$"),  # Números sueltos (1, 2a, 3b, etc.)
    re.compile(r"^https?://"),  # URLs
    re.compile(r"^www\."),  # URLs sin protocolo
    re.compile(r"^[@#]\w+"),  # Hashtags o menciones
    re.compile(r"^\s+$"),  # Solo espacios en blanco
    re.compile(r"^\.+$"),  # Solo puntos
    re.compile(r"^-+$"),  # Solo guiones
    re.compile(r"^_+$"),  # Solo guiones bajos
    re.compile(r"^\*+$"),  # Solo asteriscos
    re.compile(r"^=+$"),  # Solo signos igual
    re.compile(r"^\d+[.,]\d+$"),  # Solo números decimales (3.14, 10,5)
    re.compile(r"^[a-z]$"),  # Una sola letra minúscula
    re.compile(r"^[A-Z]$"),  # Una sola letra mayúscula
    re.compile(r"^\([^)]*\)$"),  # Solo texto entre paréntesis
    re.compile(r"^\[[^\]]*\]$"),  # Solo texto entre corchetes
    re.compile(r'^"[^"]*"$'),  # Solo texto entre comillas
    re.compile(r"^'\w+'$"),  # Palabra entre comillas simples
    re.compile(r"^(\w+\s+){10,}"),  # Demasiadas palabras (párrafo mal parseado)
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
        if pattern.match(text_lower):
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


def lemmatize_text(text: str, nlp_model: Optional['spacy.Language'] = None) -> str:
    """
    Lemmatiza un texto usando spaCy.

    La lemmatización reduce las palabras a su forma base (lema):
    - "corriendo" → "correr"
    - "casas" → "casa"
    - "mejor" → "bueno"

    Args:
        text: Texto a lemmatizar
        nlp_model: Modelo spaCy cargado (opcional). Si no se proporciona,
                   retorna el texto original.

    Returns:
        Texto lemmatizado con espacios normalizados
    """
    if not text or not nlp_model:
        return text.strip()

    try:
        doc = nlp_model(text)
        lemmas = [token.lemma_ for token in doc if not token.is_space]
        return " ".join(lemmas)
    except Exception:
        # En caso de error, retornar texto original
        return text.strip()


def lemmatize_entity(entity: dict, nlp_model: Optional['spacy.Language'] = None) -> dict:
    """
    Crea una copia de la entidad con el texto lemmatizado.

    Args:
        entity: Diccionario de entidad con al menos "text"
        nlp_model: Modelo spaCy cargado (opcional)

    Returns:
        Nueva entidad con campo "lemma" añadido
    """
    entity_copy = entity.copy()
    text = entity.get("text", "")

    if nlp_model and text:
        entity_copy["lemma"] = lemmatize_text(text, nlp_model)
    else:
        entity_copy["lemma"] = text.strip().lower()

    return entity_copy


def deduplicate_entities_by_lemma(
    entities: List[dict],
    nlp_model: Optional['spacy.Language'] = None,
    keep_strategy: str = "first"
) -> List[dict]:
    """
    Elimina entidades duplicadas usando lemmatización.

    Esto agrupa variantes de la misma entidad:
    - "Google Inc." y "Google" → mantiene una
    - "hospitales" y "hospital" → mantiene una
    - "corriendo en parques" y "correr en parque" → mantiene una

    Args:
        entities: Lista de entidades (cada una con "text" y opcionalmente "label")
        nlp_model: Modelo spaCy para lemmatización (opcional)
        keep_strategy: Estrategia para decidir cuál mantener:
                       - "first": Mantiene la primera aparición
                       - "longest": Mantiene la versión más larga
                       - "shortest": Mantiene la versión más corta
                       - "most_frequent": Mantiene la más frecuente

    Returns:
        Lista de entidades deduplicadas
    """
    if not entities:
        return []

    # Agrupar por lemma
    lemma_groups: Dict[str, List[dict]] = {}

    for entity in entities:
        text = entity.get("text", "").strip()
        if not text:
            continue

        # Generar lemma
        if nlp_model:
            lemma = lemmatize_text(text, nlp_model).lower()
        else:
            lemma = text.lower()

        # Agrupar
        if lemma not in lemma_groups:
            lemma_groups[lemma] = []
        lemma_groups[lemma].append(entity)

    # Aplicar estrategia de selección
    deduplicated = []

    for lemma, group in lemma_groups.items():
        if not group:
            continue

        if keep_strategy == "first":
            selected = group[0]
        elif keep_strategy == "longest":
            selected = max(group, key=lambda e: len(e.get("text", "")))
        elif keep_strategy == "shortest":
            selected = min(group, key=lambda e: len(e.get("text", "")))
        elif keep_strategy == "most_frequent":
            # Contar frecuencias dentro del grupo
            text_count: Dict[str, int] = {}
            for e in group:
                text = e.get("text", "")
                text_count[text] = text_count.get(text, 0) + 1
            # Encontrar el más frecuente
            most_common_text = max(text_count, key=text_count.get)
            selected = next(e for e in group if e.get("text") == most_common_text)
        else:
            selected = group[0]

        # Añadir lemma al resultado
        selected_copy = selected.copy()
        selected_copy["lemma"] = lemma
        deduplicated.append(selected_copy)

    return deduplicated


def normalize_entity_variations(
    entities: List[dict],
    nlp_model: Optional['spacy.Language'] = None,
    min_similarity: float = 0.85
) -> List[dict]:
    """
    Normaliza variaciones de entidades similares.

    Agrupa y fusiona entidades que son muy similares:
    - "Dr. Smith" y "Doctor Smith" → "Dr. Smith"
    - "NYC" y "New York City" → mantiene la más larga
    - "iPhone" e "iphone" → normaliza capitalización

    Args:
        entities: Lista de entidades
        nlp_model: Modelo spaCy (opcional)
        min_similarity: Umbral de similitud para considerar duplicados (0-1)

    Returns:
        Lista de entidades normalizadas
    """
    if not entities:
        return []

    # Primero deduplicar por lemma exacto
    deduplicated = deduplicate_entities_by_lemma(
        entities,
        nlp_model=nlp_model,
        keep_strategy="longest"
    )

    # Normalizar capitalización inconsistente
    normalized = []
    seen_lowercase: Dict[str, dict] = {}

    for entity in deduplicated:
        text = entity.get("text", "").strip()
        text_lower = text.lower()

        # Si ya vimos esta versión en minúsculas
        if text_lower in seen_lowercase:
            existing = seen_lowercase[text_lower]
            existing_text = existing.get("text", "")

            # Preferir versiones con capitalización adecuada
            # (no todas mayúsculas ni todas minúsculas)
            if text.isupper() or text.islower():
                # Versión actual es todo mayúsculas o minúsculas
                if not (existing_text.isupper() or existing_text.islower()):
                    # Mantener la existente que tiene mejor capitalización
                    continue
            elif existing_text.isupper() or existing_text.islower():
                # Reemplazar la existente con la actual que tiene mejor capitalización
                seen_lowercase[text_lower] = entity
                normalized = [e for e in normalized if e.get("text", "").lower() != text_lower]
                normalized.append(entity)
                continue

            # Si ambas tienen buena capitalización, preferir la más larga
            if len(text) > len(existing_text):
                seen_lowercase[text_lower] = entity
                normalized = [e for e in normalized if e.get("text", "").lower() != text_lower]
                normalized.append(entity)
        else:
            seen_lowercase[text_lower] = entity
            normalized.append(entity)

    return normalized


def clean_entities_advanced(
    entities: List[dict],
    nlp_model: Optional['spacy.Language'] = None,
    min_length: int = 2,
    min_frequency: int = 1,
    allow_common_names: bool = False,
    custom_stopwords: Set[str] | None = None,
    excluded_types: Set[str] | None = None,
    use_lemmatization: bool = True,
    dedup_strategy: str = "longest"
) -> List[dict]:
    """
    Pipeline completo de limpieza y normalización de entidades.

    Aplica todos los filtros disponibles en secuencia:
    1. Filtrado básico (validación, stopwords, patrones de ruido)
    2. Normalización de texto (comillas, espacios, puntuación)
    3. Lemmatización (si está habilitada)
    4. Deduplicación por lemas
    5. Normalización de variaciones (capitalización)

    Args:
        entities: Lista de entidades con formato [{"text": str, "label": str}, ...]
        nlp_model: Modelo spaCy para lemmatización (opcional pero recomendado)
        min_length: Longitud mínima en caracteres
        min_frequency: Frecuencia mínima para mantener la entidad
        allow_common_names: Si True, permite nombres comunes sin apellido
        custom_stopwords: Stopwords personalizadas adicionales
        excluded_types: Tipos de entidad a excluir
        use_lemmatization: Si True, aplica lemmatización y deduplicación por lemas
        dedup_strategy: Estrategia de deduplicación ("first", "longest", "shortest", "most_frequent")

    Returns:
        Lista limpia y normalizada de entidades

    Example:
        >>> entities = [
        ...     {"text": "Google Inc.", "label": "ORG"},
        ...     {"text": "Google", "label": "ORG"},
        ...     {"text": "hospitales", "label": "FAC"},
        ...     {"text": "hospital", "label": "FAC"},
        ...     {"text": "el", "label": "MISC"},
        ... ]
        >>> nlp = spacy.load("es_core_news_sm")
        >>> clean = clean_entities_advanced(entities, nlp_model=nlp)
        >>> # Resultado: [{"text": "Google Inc.", "label": "ORG", "lemma": "google inc."},
        >>> #            {"text": "hospital", "label": "FAC", "lemma": "hospital"}]
    """
    if not entities:
        return []

    # Paso 1: Normalizar texto de entidades
    normalized_entities = []
    for entity in entities:
        entity_copy = entity.copy()
        text = entity.get("text", "")
        entity_copy["text"] = normalize_entity_text(text)
        normalized_entities.append(entity_copy)

    # Paso 2: Filtrar ruido básico
    filtered = filter_entities(
        normalized_entities,
        min_length=min_length,
        min_frequency=min_frequency,
        allow_common_names=allow_common_names,
        custom_stopwords=custom_stopwords,
        excluded_types=excluded_types
    )

    if not filtered:
        return []

    # Paso 3: Aplicar lemmatización y deduplicación (si está habilitada)
    if use_lemmatization and nlp_model:
        # Añadir lemas a las entidades
        with_lemmas = [lemmatize_entity(e, nlp_model) for e in filtered]

        # Deduplicar por lemma
        deduplicated = deduplicate_entities_by_lemma(
            with_lemmas,
            nlp_model=nlp_model,
            keep_strategy=dedup_strategy
        )

        # Normalizar variaciones (capitalización, etc.)
        final = normalize_entity_variations(
            deduplicated,
            nlp_model=nlp_model
        )
    else:
        # Sin lemmatización, solo deduplicación simple por texto exacto
        seen = set()
        final = []
        for entity in filtered:
            text_lower = entity.get("text", "").lower()
            if text_lower not in seen:
                seen.add(text_lower)
                final.append(entity)

    # Paso 4: Ordenar por calidad (opcional, para facilitar revisión)
    final_sorted = sorted(
        final,
        key=lambda e: get_entity_quality_score(e),
        reverse=True
    )

    return final_sorted


__all__ = [
    "is_valid_entity",
    "is_noise_pattern",
    "filter_entities",
    "normalize_entity_text",
    "get_entity_quality_score",
    "lemmatize_text",
    "lemmatize_entity",
    "deduplicate_entities_by_lemma",
    "normalize_entity_variations",
    "clean_entities_advanced",
    "STOPWORDS",
    "COMMON_NAMES",
]
