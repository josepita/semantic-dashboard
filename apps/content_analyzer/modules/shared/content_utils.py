"""
Content Analysis Utilities
===========================

Funciones compartidas para análisis de contenido, detección de columnas,
procesamiento de embeddings y extracción de estructura de URLs.

Estas funciones son reutilizables en content-analyzer y linking-optimizer.

Autor: Embedding Insights
Versión: 1.0.0
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# DETECCIÓN DE COLUMNAS
# ============================================================================

def detect_embedding_columns(df: pd.DataFrame) -> List[str]:
    """
    Detecta columnas que parecen contener embeddings.

    Busca columnas donde los valores tienen formato de lista separada por comas
    con más de 50 elementos (típico de embeddings).

    Args:
        df: DataFrame a analizar

    Returns:
        Lista de nombres de columnas candidatas con embeddings

    Example:
        >>> df = pd.DataFrame({
        ...     'text': ['hello'],
        ...     'embedding': ['0.1,0.2,0.3,...']  # 384 valores
        ... })
        >>> detect_embedding_columns(df)
        ['embedding']
    """
    candidate_cols = []
    for col in df.columns:
        sample_values = df[col].dropna().astype(str).head(3)
        if sample_values.empty:
            continue
        first_val = sample_values.iloc[0]
        # Embeddings típicos tienen 50+ dimensiones
        if "," in first_val and len(first_val.split(",")) > 50:
            candidate_cols.append(col)
    return candidate_cols


def detect_url_columns(df: pd.DataFrame) -> List[str]:
    """
    Detecta columnas que probablemente contengan URLs.

    Busca columnas con nombres que contengan palabras clave relacionadas con URLs.
    Si no encuentra ninguna, devuelve todas las columnas.

    Args:
        df: DataFrame a analizar

    Returns:
        Lista de nombres de columnas candidatas con URLs

    Example:
        >>> df = pd.DataFrame({
        ...     'page_url': ['https://example.com'],
        ...     'title': ['Example']
        ... })
        >>> detect_url_columns(df)
        ['page_url']
    """
    patterns = ("url", "address", "dirección", "link", "href")
    matches = [col for col in df.columns if any(pat in col.lower() for pat in patterns)]
    return matches or df.columns.tolist()


def detect_page_type_columns(df: pd.DataFrame, max_unique_values: int = 40) -> List[str]:
    """
    Identifica columnas categóricas candidatas a representar el tipo de página.

    Busca columnas con:
    - 2-40 valores únicos (excluye IDs y valores demasiado granulares)
    - No son identificadores únicos (1 valor por fila)
    - Priorizadas por keywords en el nombre ('tipo', 'category', etc.)

    Args:
        df: DataFrame a analizar
        max_unique_values: Máximo de valores únicos permitidos

    Returns:
        Lista de nombres de columnas ordenadas por relevancia

    Example:
        >>> df = pd.DataFrame({
        ...     'page_type': ['blog', 'product', 'blog'],
        ...     'id': [1, 2, 3]
        ... })
        >>> detect_page_type_columns(df)
        ['page_type']
    """
    candidates: List[Tuple[int, str]] = []
    keywords = ("tipo", "type", "categoria", "category", "segment", "seccion", "page_type", "familia")

    for col in df.columns:
        # Excluir columna de embeddings
        if col == "EmbeddingsFloat":
            continue

        series = df[col].dropna()
        if series.empty:
            continue

        unique_values = series.astype(str).str.strip().unique()
        unique_count = len(unique_values)

        # Validaciones
        if unique_count <= 1 or unique_count > max_unique_values:
            continue
        if unique_count == len(df):
            # Probablemente es un identificador único, no una categoría
            continue

        # Score: 0 = mejor (tiene keywords), 1 = sin keywords
        score = 1
        lower_name = col.lower()
        if any(keyword in lower_name for keyword in keywords):
            score = 0

        candidates.append((score, col))

    # Ordenar por score (0 primero) y alfabético
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [col for _, col in candidates]


# ============================================================================
# PROCESAMIENTO DE EMBEDDINGS
# ============================================================================

def convert_embedding_cell(value: str) -> Optional[np.ndarray]:
    """
    Convierte string de embedding a array NumPy.

    Acepta formatos:
    - "[0.1, 0.2, 0.3]"
    - "0.1, 0.2, 0.3"
    - "0.1,0.2,0.3"

    Args:
        value: String con valores separados por comas

    Returns:
        Array NumPy normalizado o None si es inválido

    Example:
        >>> convert_embedding_cell("[0.5, 0.5]")
        array([0.5, 0.5])
        >>> convert_embedding_cell("[0.0, 0.0]")  # Vector cero
        None
    """
    try:
        clean = str(value).replace("[", "").replace("]", "")
        vector = np.array([float(x) for x in clean.split(",")])

        if vector.size == 0:
            return None
        if np.linalg.norm(vector) == 0:
            return None

        return vector
    except Exception:
        return None


def preprocess_embeddings(df: pd.DataFrame, embedding_col: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Procesa y valida embeddings de un DataFrame.

    Realiza:
    1. Conversión de strings a arrays NumPy
    2. Filtrado de embeddings inválidos
    3. Validación de dimensionalidad consistente
    4. Reporte de filas descartadas

    Args:
        df: DataFrame con columna de embeddings
        embedding_col: Nombre de la columna con embeddings

    Returns:
        Tupla (DataFrame procesado, lista de mensajes de log)

    Raises:
        ValueError: Si no quedan embeddings válidos

    Example:
        >>> df = pd.DataFrame({
        ...     'text': ['a', 'b', 'c'],
        ...     'emb': ['[0.1,0.2]', '[0.3,0.4]', '[0.5]']  # c tiene dim diferente
        ... })
        >>> processed_df, messages = preprocess_embeddings(df, 'emb')
        >>> len(processed_df)
        2
        >>> 'EmbeddingsFloat' in processed_df.columns
        True
    """
    messages: List[str] = []
    df_local = df.copy()

    # Convertir embeddings
    df_local["EmbeddingsFloat"] = df_local[embedding_col].apply(convert_embedding_cell)

    # Filtrar nulos
    before_drop = len(df_local)
    df_local = df_local[df_local["EmbeddingsFloat"].notna()].copy()
    dropped = before_drop - len(df_local)
    if dropped:
        messages.append(f"Se descartaron {dropped} filas por embeddings inválidos.")

    if df_local.empty:
        raise ValueError("No quedan filas con embeddings válidos.")

    # Validar dimensionalidad consistente
    lengths = df_local["EmbeddingsFloat"].apply(len)
    if lengths.nunique() > 1:
        mode_length = lengths.mode().iloc[0]
        df_local = df_local[lengths == mode_length].copy()
        messages.append(
            "Los embeddings tenían longitudes distintas. "
            f"Se conservaron {len(df_local)} filas con longitud {mode_length}."
        )

    df_local.reset_index(drop=True, inplace=True)
    return df_local, messages


# ============================================================================
# EXTRACCIÓN DE ESTRUCTURA DE URLs
# ============================================================================

def extract_url_silo(url: str, depth: int = 2, default: str = "general") -> str:
    """
    Extrae un segmento de la URL para usarlo como 'silo' o tema.

    Útil para identificar temas principales en arquitectura de información.

    Args:
        url: URL completa
        depth: Nivel de profundidad (1=primero, 2=segundo, etc.)
        default: Valor por defecto si no se puede extraer

    Returns:
        Segmento de URL normalizado (lowercase, sin espacios)

    Example:
        >>> extract_url_silo("https://example.com/blog/python/tutorial", depth=2)
        'python'
        >>> extract_url_silo("https://example.com/blog/python/tutorial", depth=1)
        'blog'
        >>> extract_url_silo("https://example.com", depth=2)
        'general'
    """
    try:
        # Remover query strings
        cleaned = str(url).split("?")[0]
        segments = [part for part in cleaned.split("/") if part]

        if not segments:
            return default

        # depth=1 -> index 0, depth=2 -> index 1
        index = min(max(depth - 1, 0), len(segments) - 1)
        candidate = segments[index].strip().lower()

        return candidate or default
    except Exception:
        return default


def extract_url_hierarchy(url: str, depth: int = 2) -> str:
    """
    Extrae jerarquía completa de URL hasta cierta profundidad.

    Args:
        url: URL completa
        depth: Profundidad máxima a extraer

    Returns:
        Ruta jerárquica (ej: "blog/python")

    Example:
        >>> extract_url_hierarchy("https://example.com/blog/python/tutorial", depth=2)
        'blog/python'
        >>> extract_url_hierarchy("https://example.com/blog/python/tutorial", depth=1)
        'blog'
        >>> extract_url_hierarchy("https://example.com", depth=2)
        'root'
    """
    cleaned = str(url).strip()
    if not cleaned:
        return "root"

    # Remover protocolo y dominio
    cleaned = re.sub(r"https?://[^/]+", "", cleaned)
    segments = [segment for segment in cleaned.split("/") if segment]

    if not segments:
        return "root"

    depth = max(1, int(depth))
    return "/".join(segments[:depth])


# ============================================================================
# ANÁLISIS DE ENTIDADES
# ============================================================================

def parse_entity_payload(value: object) -> Dict[str, float]:
    """
    Parsea payload de entidades desde múltiples formatos.

    Soporta:
    - Dict: {"entity1": 0.8, "entity2": 0.5}
    - JSON string: '{"entity1": 0.8}'
    - Lista con estructura: [{"id": "entity1", "score": 0.8}]

    Args:
        value: Payload en cualquier formato soportado

    Returns:
        Diccionario {entity_id: prominence_score}

    Example:
        >>> parse_entity_payload({"entity1": 0.8})
        {'entity1': 0.8}
        >>> parse_entity_payload('[{"id": "ent1", "score": 0.9}]')
        {'ent1': 0.9}
    """
    # Ya es dict
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items() if isinstance(v, (int, float))}

    # Es string JSON
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}

        # JSON dict
        if isinstance(parsed, dict):
            return {str(k): float(v) for k, v in parsed.items() if isinstance(v, (int, float))}

        # JSON lista con estructura [{"id": X, "score": Y}]
        if isinstance(parsed, list):
            result: Dict[str, float] = {}
            for item in parsed:
                if isinstance(item, dict) and "id" in item and "score" in item:
                    try:
                        result[str(item["id"])] = float(item["score"])
                    except (TypeError, ValueError):
                        continue
            return result

    return {}


def calculate_weighted_entity_overlap(
    source_entities: Dict[str, float],
    target_entities: Dict[str, float],
) -> float:
    """
    Calcula Jaccard ponderado entre dos conjuntos de entidades.

    Usa scores de prominencia para calcular overlap:
    - Intersección: min(score1, score2) por cada entidad común
    - Unión: max(score1, score2) por cada entidad en ambos sets

    Args:
        source_entities: {entity_id: prominence} de la página origen
        target_entities: {entity_id: prominence} de la página destino

    Returns:
        Score de overlap entre 0.0 y 1.0

    Example:
        >>> src = {"ent1": 0.8, "ent2": 0.5}
        >>> tgt = {"ent1": 0.6, "ent3": 0.7}
        >>> calculate_weighted_entity_overlap(src, tgt)
        0.3  # min(0.8,0.6) / (max(0.8,0.6) + 0.5 + 0.7)
    """
    if not source_entities or not target_entities:
        return 0.0

    source_keys = set(source_entities.keys())
    target_keys = set(target_entities.keys())
    union_keys = source_keys.union(target_keys)

    if not union_keys:
        return 0.0

    intersection_keys = source_keys.intersection(target_keys)

    # Weighted intersection: suma de min scores
    weighted_intersection = sum(
        min(source_entities.get(key, 0.0), target_entities.get(key, 0.0))
        for key in intersection_keys
    )

    # Weighted union: suma de max scores
    weighted_union = sum(
        max(source_entities.get(key, 0.0), target_entities.get(key, 0.0))
        for key in union_keys
    )

    if weighted_union == 0.0:
        return 0.0

    return float(weighted_intersection / weighted_union)


def build_entity_payload_from_doc_relations(
    doc_entities_df: pd.DataFrame,
    url_column: str,
    entity_column: str,
    prominence_column: str,
) -> Dict[str, Dict[str, float]]:
    """
    Construye payload de entidades desde DataFrame relacional.

    Transforma tabla larga (una fila por relación documento-entidad) a estructura
    anidada {url: {entity_id: prominence}}.

    Args:
        doc_entities_df: DataFrame con relaciones
        url_column: Nombre de columna con URLs
        entity_column: Nombre de columna con entity IDs
        prominence_column: Nombre de columna con scores de prominencia

    Returns:
        Diccionario {url: {entity_id: prominence}}

    Example:
        >>> df = pd.DataFrame({
        ...     'url': ['page1', 'page1', 'page2'],
        ...     'entity': ['ent1', 'ent2', 'ent1'],
        ...     'score': [0.8, 0.5, 0.9]
        ... })
        >>> build_entity_payload_from_doc_relations(df, 'url', 'entity', 'score')
        {'page1': {'ent1': 0.8, 'ent2': 0.5}, 'page2': {'ent1': 0.9}}
    """
    payload: Dict[str, Dict[str, float]] = {}

    for _, row in doc_entities_df.iterrows():
        try:
            url = str(row[url_column]).strip()
            entity_id = str(row[entity_column]).strip()
            prominence = float(row[prominence_column])

            if url not in payload:
                payload[url] = {}

            payload[url][entity_id] = prominence

        except (KeyError, ValueError, TypeError):
            continue

    return payload


# ============================================================================
# UTILIDADES DE ANCHOR TEXT
# ============================================================================

def suggest_anchor_from_url(url: str) -> str:
    """
    Genera anchor text legible desde URL.

    Extrae el último segmento de la URL, limpia guiones/underscores y
    convierte a Title Case.

    Args:
        url: URL completa

    Returns:
        Anchor text sugerido

    Example:
        >>> suggest_anchor_from_url("https://example.com/blog/python-tutorial")
        'Python Tutorial'
        >>> suggest_anchor_from_url("https://example.com/products/seo_tools")
        'Seo Tools'
        >>> suggest_anchor_from_url("https://example.com/")
        'Leer Más'
    """
    try:
        # Limpiar query y fragment
        clean_url = str(url).split("?")[0].split("#")[0]
        segments = [segment for segment in clean_url.split("/") if segment]

        if not segments:
            slug = clean_url
        else:
            slug = segments[-1]
            # Si último segmento vacío, usar penúltimo
            if not slug.strip() and len(segments) >= 2:
                slug = segments[-2]

        # Remover extensión de archivo
        slug = slug.split(".")[0]

        # Convertir separadores a espacios
        slug = re.sub(r"[-_]+", " ", slug)
        slug = re.sub(r"\s+", " ", slug).strip()

        return slug.title() if slug else "Leer Más"
    except Exception:
        return "Leer Más"


def format_topic_label(raw_value: Optional[str]) -> Optional[str]:
    """
    Limpia y formatea valor de silo/tema para anchor text.

    Args:
        raw_value: Valor crudo del silo

    Returns:
        Valor formateado en Title Case o None si inválido

    Example:
        >>> format_topic_label("python-tutorials")
        'Python Tutorials'
        >>> format_topic_label("general")
        None
        >>> format_topic_label("  ")
        None
    """
    if not raw_value:
        return None

    lowered = str(raw_value).strip().lower()

    # Excluir valores genéricos
    if lowered in {"", "general", "none", "null"}:
        return None

    # Limpiar separadores
    cleaned = re.sub(r"[-_]+", " ", lowered)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned.title() if cleaned else None


def generate_contextual_anchor(page_type: str, silo_value: str, url: str) -> str:
    """
    Genera anchor text contextual usando tipo de página y silo.

    Crea anchors naturales adaptados al tipo de página de destino.

    Args:
        page_type: Tipo de página destino (ej: "servicio", "blog")
        silo_value: Silo o tema de la página
        url: URL de la página destino

    Returns:
        Anchor text contextual

    Example:
        >>> generate_contextual_anchor("servicio", "python", "https://example.com/servicios/python")
        'Ver servicio de Python'
        >>> generate_contextual_anchor("blog", "seo", "https://example.com/blog/seo")
        'Leer guía sobre Seo'
        >>> generate_contextual_anchor("", "", "https://example.com/page")
        'Page'
    """
    topic_label = format_topic_label(silo_value)
    page_type_clean = (page_type or "").strip().lower()

    # Con tema definido
    if topic_label:
        if page_type_clean in {"tratamiento", "servicio"}:
            return f"Ver {page_type_clean} de {topic_label}"
        if page_type_clean in {"categoria", "categoría"}:
            return f"Explorar categoría {topic_label}"
        if page_type_clean in {"blog", "post", "artículo"}:
            return f"Leer guía sobre {topic_label}"
        if page_type_clean in {"producto", "servicios"}:
            return f"Descubrir {page_type_clean} de {topic_label}"
        if page_type_clean:
            return f"Más sobre {topic_label} ({page_type_clean})"
        return f"Aprender más sobre {topic_label}"

    # Sin tema, solo tipo
    if page_type_clean:
        return f"Ver {page_type_clean}"

    # Fallback: extraer desde URL
    return suggest_anchor_from_url(url)


# ============================================================================
# DETECCIÓN DE PÁGINAS NO ENLAZABLES
# ============================================================================

# Patrones de URL que indican páginas de navegación/listado (no deberían recibir enlaces)
NON_LINKABLE_URL_PATTERNS = [
    # Categorías y taxonomías
    r'/categor[iy]a?/',
    r'/category/',
    r'/tag/',
    r'/etiqueta/',
    r'/tags/',
    r'/archivo/',
    r'/archive/',
    # Paginación
    r'/page/\d+',
    r'/pagina/\d+',
    r'\?page=',
    r'\?paged=',
    r'/p/\d+$',
    # Tienda/ecommerce - listados
    r'/shop/$',
    r'/tienda/$',
    r'/store/$',
    r'/productos/$',
    r'/products/$',
    r'/collections/',
    r'/colecciones/',
    # Blog - índices
    r'/blog/$',
    r'/noticias/$',
    r'/news/$',
    r'/articles/$',
    r'/articulos/$',
    # Búsqueda y filtros
    r'/search',
    r'/buscar',
    r'/busqueda',
    r'\?s=',
    r'\?q=',
    r'\?filter',
    r'\?ordenar',
    r'\?sort',
    # Autor/fecha
    r'/author/',
    r'/autor/',
    r'/\d{4}/\d{2}/$',  # /2024/01/ (archivos por fecha)
    # Legal/utilidad
    r'/politica-',
    r'/privacy',
    r'/legal/',
    r'/terminos',
    r'/terms',
    r'/cookie',
    r'/aviso-legal',
    # Otros
    r'/feed/',
    r'/rss/',
    r'/sitemap',
    r'/wp-content/',
    r'/wp-admin/',
]

# Tipos de página que típicamente no deberían recibir enlaces
NON_LINKABLE_PAGE_TYPES = [
    'categoria', 'categoría', 'category',
    'tag', 'etiqueta',
    'archivo', 'archive',
    'indice', 'índice', 'index',
    'listado', 'listing',
    'paginacion', 'paginación', 'pagination',
    'busqueda', 'búsqueda', 'search',
    'legal', 'privacidad', 'privacy',
    'navegacion', 'navegación', 'navigation',
]


def detect_non_linkable_pages(
    df: pd.DataFrame,
    url_column: str,
    type_column: Optional[str] = None,
    custom_patterns: Optional[List[str]] = None,
    custom_types: Optional[List[str]] = None,
) -> pd.Series:
    """
    Detecta páginas que no deberían recibir enlaces internos.

    Analiza URLs y tipos de página para identificar:
    - Páginas de categoría/listado
    - Paginaciones
    - Páginas de búsqueda
    - Páginas legales/utilidad

    Args:
        df: DataFrame con los datos
        url_column: Nombre de la columna de URLs
        type_column: Nombre de la columna de tipo de página (opcional)
        custom_patterns: Patrones de URL adicionales a excluir
        custom_types: Tipos de página adicionales a excluir

    Returns:
        Serie booleana donde True = página NO enlazable

    Example:
        >>> df['no_enlazar'] = detect_non_linkable_pages(df, 'URL', 'Tipo')
        >>> df_enlazables = df[~df['no_enlazar']]
    """
    # Combinar patrones
    patterns = NON_LINKABLE_URL_PATTERNS.copy()
    if custom_patterns:
        patterns.extend(custom_patterns)

    # Crear regex combinada
    combined_pattern = '|'.join(f'({p})' for p in patterns)

    # Detectar por URL
    urls = df[url_column].astype(str).str.lower()
    is_non_linkable = urls.str.contains(combined_pattern, regex=True, na=False)

    # Detectar por tipo de página si existe la columna
    if type_column and type_column in df.columns:
        types_to_exclude = [t.lower() for t in NON_LINKABLE_PAGE_TYPES]
        if custom_types:
            types_to_exclude.extend([t.lower() for t in custom_types])

        page_types = df[type_column].astype(str).str.lower().str.strip()
        is_non_linkable_type = page_types.isin(types_to_exclude)

        # Combinar ambos criterios (OR)
        is_non_linkable = is_non_linkable | is_non_linkable_type

    return is_non_linkable


def get_linkable_page_stats(
    df: pd.DataFrame,
    url_column: str,
    type_column: Optional[str] = None,
) -> Dict[str, any]:
    """
    Obtiene estadísticas sobre páginas enlazables vs no enlazables.

    Returns:
        Diccionario con estadísticas
    """
    non_linkable = detect_non_linkable_pages(df, url_column, type_column)

    total = len(df)
    non_linkable_count = non_linkable.sum()
    linkable_count = total - non_linkable_count

    # Analizar razones de exclusión
    urls = df[url_column].astype(str).str.lower()
    reasons = {}

    reason_patterns = {
        'Categorías/Tags': r'/categor|/tag/|/etiqueta/',
        'Paginación': r'/page/|/pagina/|\?page',
        'Tienda/Listados': r'/shop/|/tienda/|/products/|/collections/',
        'Blog/Índices': r'/blog/$|/noticias/$|/news/$',
        'Búsqueda': r'/search|/buscar|\?s=|\?q=',
        'Legal/Utilidad': r'/privacy|/legal|/cookie|/terms',
        'Archivos/Fechas': r'/archive|/\d{4}/\d{2}/$',
    }

    for reason, pattern in reason_patterns.items():
        count = urls.str.contains(pattern, regex=True, na=False).sum()
        if count > 0:
            reasons[reason] = count

    return {
        'total_pages': total,
        'linkable_pages': linkable_count,
        'non_linkable_pages': non_linkable_count,
        'linkable_percentage': round(linkable_count / total * 100, 1) if total > 0 else 0,
        'exclusion_reasons': reasons,
    }


# ============================================================================
# CARGA DE DATOS DE CRAWL (SCREAMING FROG)
# ============================================================================

# Mapeo de columnas conocidas de Screaming Frog a nombres estándar
SCREAMING_FROG_COLUMN_MAPPINGS = {
    # Columnas de origen
    'source': 'source_url',
    'from': 'source_url',
    'source url': 'source_url',
    'url origen': 'source_url',
    'origen': 'source_url',
    # Columnas de destino
    'destination': 'target_url',
    'dest': 'target_url',
    'target': 'target_url',
    'to': 'target_url',
    'destination url': 'target_url',
    'url destino': 'target_url',
    'destino': 'target_url',
    'href': 'target_url',
    'link': 'target_url',
    # Anchor text
    'anchor': 'anchor_text',
    'anchor text': 'anchor_text',
    'text': 'anchor_text',
    'link text': 'anchor_text',
    'texto ancla': 'anchor_text',
    # Tipo de enlace
    'type': 'link_type',
    'link type': 'link_type',
    'tipo': 'link_type',
    'tipo de enlace': 'link_type',
    # Status
    'status code': 'status_code',
    'status': 'status_code',
    'código estado': 'status_code',
}


def load_screaming_frog_internal_links(
    file_path_or_df: any,
    source_column: Optional[str] = None,
    target_column: Optional[str] = None,
    anchor_column: Optional[str] = None,
    filter_follow_only: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Carga enlaces internos desde exportación de Screaming Frog.

    Soporta múltiples formatos de exportación:
    - internal_all.csv (reporte de enlaces internos)
    - all_inlinks.csv (reporte de inlinks)
    - Cualquier CSV con columnas de origen y destino

    Args:
        file_path_or_df: Ruta al archivo CSV o DataFrame ya cargado
        source_column: Columna de URL origen (auto-detectada si None)
        target_column: Columna de URL destino (auto-detectada si None)
        anchor_column: Columna de anchor text (auto-detectada si None)
        filter_follow_only: Si True, filtra solo enlaces follow (excluye nofollow)

    Returns:
        Tupla (DataFrame normalizado, lista de mensajes/warnings)

    El DataFrame resultante tiene columnas:
        - source_url: URL de la página que contiene el enlace
        - target_url: URL de la página enlazada
        - anchor_text: Texto del enlace (si disponible)

    Example:
        >>> links_df, messages = load_screaming_frog_internal_links("internal_all.csv")
        >>> existing_links = set(zip(links_df['source_url'], links_df['target_url']))
    """
    messages: List[str] = []

    # Cargar datos
    if isinstance(file_path_or_df, pd.DataFrame):
        df = file_path_or_df.copy()
    else:
        try:
            # Intentar diferentes encodings comunes de Screaming Frog
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(file_path_or_df, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError(f"No se pudo leer el archivo con ningún encoding común")
        except Exception as e:
            raise ValueError(f"Error al cargar archivo: {str(e)}")

    if df.empty:
        return pd.DataFrame(columns=['source_url', 'target_url', 'anchor_text']), ["El archivo está vacío"]

    # Normalizar nombres de columnas a minúsculas
    df.columns = [str(col).lower().strip() for col in df.columns]
    original_columns = df.columns.tolist()

    # Auto-detectar columnas si no se especifican
    def find_column(target_name: str, explicit_col: Optional[str]) -> Optional[str]:
        """Busca columna por nombre explícito o por mapeo."""
        if explicit_col:
            explicit_lower = explicit_col.lower().strip()
            if explicit_lower in df.columns:
                return explicit_lower
            messages.append(f"Columna '{explicit_col}' no encontrada")

        # Buscar por mapeo
        for original, mapped in SCREAMING_FROG_COLUMN_MAPPINGS.items():
            if mapped == target_name and original in df.columns:
                return original

        return None

    source_col = find_column('source_url', source_column)
    target_col = find_column('target_url', target_column)
    anchor_col = find_column('anchor_text', anchor_column)

    # Validar columnas requeridas
    if not source_col:
        # Intentar usar primera columna como source
        source_col = df.columns[0] if len(df.columns) > 0 else None
        if source_col:
            messages.append(f"Usando '{source_col}' como columna de origen (auto-detectada)")

    if not target_col:
        # Intentar usar segunda columna como target
        target_col = df.columns[1] if len(df.columns) > 1 else None
        if target_col:
            messages.append(f"Usando '{target_col}' como columna de destino (auto-detectada)")

    if not source_col or not target_col:
        available = ", ".join(original_columns[:10])
        raise ValueError(
            f"No se encontraron columnas de origen/destino. "
            f"Columnas disponibles: {available}"
        )

    # Crear DataFrame normalizado
    result_df = pd.DataFrame({
        'source_url': df[source_col].astype(str).str.strip(),
        'target_url': df[target_col].astype(str).str.strip(),
    })

    # Añadir anchor si existe
    if anchor_col:
        result_df['anchor_text'] = df[anchor_col].astype(str).str.strip()
    else:
        result_df['anchor_text'] = ''

    # Filtrar enlaces nofollow si se especifica
    if filter_follow_only:
        # Buscar columnas que indiquen tipo de enlace
        type_cols = [col for col in df.columns if 'follow' in col or 'type' in col]
        for type_col in type_cols:
            type_values = df[type_col].astype(str).str.lower()
            nofollow_mask = type_values.str.contains('nofollow', na=False)
            if nofollow_mask.any():
                before_count = len(result_df)
                result_df = result_df[~nofollow_mask]
                filtered_count = before_count - len(result_df)
                if filtered_count > 0:
                    messages.append(f"Filtrados {filtered_count} enlaces nofollow")
                break

    # Eliminar duplicados
    before_dedup = len(result_df)
    result_df = result_df.drop_duplicates(subset=['source_url', 'target_url'])
    after_dedup = len(result_df)
    if before_dedup != after_dedup:
        messages.append(f"Eliminados {before_dedup - after_dedup} enlaces duplicados")

    # Eliminar self-links
    self_links = result_df['source_url'] == result_df['target_url']
    if self_links.any():
        result_df = result_df[~self_links]
        messages.append(f"Eliminados {self_links.sum()} auto-enlaces")

    messages.insert(0, f"Cargados {len(result_df)} enlaces internos únicos")

    return result_df.reset_index(drop=True), messages


def get_existing_links_set(
    links_df: pd.DataFrame,
    normalize_urls: bool = True,
) -> set:
    """
    Convierte DataFrame de enlaces a set para búsqueda rápida.

    Args:
        links_df: DataFrame con columnas source_url y target_url
        normalize_urls: Si True, normaliza URLs removiendo trailing slash

    Returns:
        Set de tuplas (source_url, target_url)

    Example:
        >>> links_df, _ = load_screaming_frog_internal_links("internal_all.csv")
        >>> existing = get_existing_links_set(links_df)
        >>> ("https://example.com/blog", "https://example.com/service") in existing
        True
    """
    if links_df.empty:
        return set()

    source_urls = links_df['source_url'].astype(str)
    target_urls = links_df['target_url'].astype(str)

    if normalize_urls:
        source_urls = source_urls.str.rstrip('/')
        target_urls = target_urls.str.rstrip('/')

    return set(zip(source_urls, target_urls))


def filter_new_link_recommendations(
    recommendations_df: pd.DataFrame,
    existing_links: set,
    source_col: str = 'Origen URL',
    target_col: str = 'Destino URL',
    normalize_urls: bool = True,
) -> Tuple[pd.DataFrame, int]:
    """
    Filtra recomendaciones para excluir enlaces que ya existen.

    Args:
        recommendations_df: DataFrame con recomendaciones de enlaces
        existing_links: Set de (source, target) de enlaces existentes
        source_col: Nombre de columna origen en recommendations_df
        target_col: Nombre de columna destino en recommendations_df
        normalize_urls: Si True, normaliza URLs para comparación

    Returns:
        Tupla (DataFrame filtrado, número de enlaces excluidos)

    Example:
        >>> recs = semantic_link_recommendations(df, ...)
        >>> existing = get_existing_links_set(crawl_df)
        >>> new_recs, excluded = filter_new_link_recommendations(recs, existing)
    """
    if recommendations_df.empty or not existing_links:
        return recommendations_df, 0

    source_urls = recommendations_df[source_col].astype(str)
    target_urls = recommendations_df[target_col].astype(str)

    if normalize_urls:
        source_urls = source_urls.str.rstrip('/')
        target_urls = target_urls.str.rstrip('/')

    # Marcar enlaces que ya existen
    exists_mask = [
        (src, tgt) in existing_links
        for src, tgt in zip(source_urls, target_urls)
    ]

    filtered_df = recommendations_df[~pd.Series(exists_mask, index=recommendations_df.index)]
    excluded_count = sum(exists_mask)

    return filtered_df.reset_index(drop=True), excluded_count


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Detección
    "detect_embedding_columns",
    "detect_url_columns",
    "detect_page_type_columns",
    # Embeddings
    "convert_embedding_cell",
    "preprocess_embeddings",
    # URLs
    "extract_url_silo",
    "extract_url_hierarchy",
    # Entidades
    "parse_entity_payload",
    "calculate_weighted_entity_overlap",
    "build_entity_payload_from_doc_relations",
    # Anchors
    "suggest_anchor_from_url",
    "format_topic_label",
    "generate_contextual_anchor",
    # Páginas no enlazables
    "detect_non_linkable_pages",
    "get_linkable_page_stats",
    "NON_LINKABLE_URL_PATTERNS",
    "NON_LINKABLE_PAGE_TYPES",
    # Carga de crawl (Screaming Frog)
    "load_screaming_frog_internal_links",
    "get_existing_links_set",
    "filter_new_link_recommendations",
    "SCREAMING_FROG_COLUMN_MAPPINGS",
]
