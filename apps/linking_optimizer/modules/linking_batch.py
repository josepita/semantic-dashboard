"""
Batch Processing for Large Datasets
=====================================

Utilidades para procesamiento por lotes de datasets grandes (+10k URLs).
Divide el trabajo en chunks para evitar problemas de memoria.

Autor: Embedding Insights
Versión: 1.0.0
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


# ============================================================================
# CONSTANTES Y CONFIGURACIÓN
# ============================================================================

# Tamaño de chunk por defecto (URLs origen por lote)
DEFAULT_CHUNK_SIZE = 1000

# Umbral para activar procesamiento por lotes automáticamente
AUTO_BATCH_THRESHOLD = 5000

# Tamaño máximo recomendado para matriz de similitud en memoria
MAX_SIMILARITY_MATRIX_SIZE = 50_000_000  # ~400MB para float64


def estimate_memory_usage(n_rows: int, n_dims: int = 384) -> Dict[str, float]:
    """
    Estima el uso de memoria para operaciones de similitud.

    Args:
        n_rows: Número de filas (URLs)
        n_dims: Dimensiones del embedding

    Returns:
        Diccionario con estimaciones en MB
    """
    # Matriz de embeddings: n_rows x n_dims x 8 bytes (float64)
    embeddings_mb = (n_rows * n_dims * 8) / (1024 * 1024)

    # Matriz de similitud completa: n_rows x n_rows x 8 bytes
    similarity_mb = (n_rows * n_rows * 8) / (1024 * 1024)

    # Memoria total estimada (embeddings + similitud + overhead)
    total_mb = embeddings_mb + similarity_mb + (embeddings_mb * 0.5)

    return {
        'embeddings_mb': round(embeddings_mb, 2),
        'similarity_matrix_mb': round(similarity_mb, 2),
        'total_estimated_mb': round(total_mb, 2),
        'recommended_batch': n_rows > AUTO_BATCH_THRESHOLD,
    }


def get_optimal_chunk_size(
    total_rows: int,
    max_memory_mb: float = 500,
    embedding_dims: int = 384,
) -> int:
    """
    Calcula el tamaño óptimo de chunk basado en memoria disponible.

    Args:
        total_rows: Total de URLs a procesar
        max_memory_mb: Memoria máxima a usar (MB)
        embedding_dims: Dimensiones del embedding

    Returns:
        Tamaño de chunk recomendado
    """
    # Memoria por fila para matriz de similitud parcial
    # chunk_size x total_rows x 8 bytes
    bytes_per_source = total_rows * 8

    # Cuántos sources podemos procesar con max_memory
    max_chunk = int((max_memory_mb * 1024 * 1024) / bytes_per_source)

    # Limitar entre 100 y DEFAULT_CHUNK_SIZE
    optimal = max(100, min(max_chunk, DEFAULT_CHUNK_SIZE))

    return optimal


# ============================================================================
# GENERADORES DE CHUNKS
# ============================================================================

def chunk_indices(
    indices: List[int],
    chunk_size: int,
) -> Generator[List[int], None, None]:
    """
    Divide una lista de índices en chunks.

    Args:
        indices: Lista de índices a dividir
        chunk_size: Tamaño de cada chunk

    Yields:
        Listas de índices de tamaño chunk_size o menor
    """
    for i in range(0, len(indices), chunk_size):
        yield indices[i:i + chunk_size]


def chunk_dataframe(
    df: pd.DataFrame,
    chunk_size: int,
) -> Generator[Tuple[int, pd.DataFrame], None, None]:
    """
    Divide un DataFrame en chunks.

    Args:
        df: DataFrame a dividir
        chunk_size: Tamaño de cada chunk

    Yields:
        Tuplas (índice_chunk, DataFrame_chunk)
    """
    n_chunks = math.ceil(len(df) / chunk_size)
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, len(df))
        yield i, df.iloc[start_idx:end_idx]


# ============================================================================
# CÁLCULO DE SIMILITUD POR LOTES
# ============================================================================

def compute_similarity_batch(
    source_embeddings: np.ndarray,
    all_embeddings: np.ndarray,
    normalize_embeddings: bool = True,
) -> np.ndarray:
    """
    Calcula similitud coseno entre embeddings de origen y todos los embeddings.

    Args:
        source_embeddings: Embeddings de las URLs origen (n_sources x dims)
        all_embeddings: Todos los embeddings (n_total x dims)
        normalize_embeddings: Si True, normaliza antes de calcular

    Returns:
        Matriz de similitud (n_sources x n_total)
    """
    if normalize_embeddings:
        source_norm = normalize(source_embeddings)
        all_norm = normalize(all_embeddings)
    else:
        source_norm = source_embeddings
        all_norm = all_embeddings

    # Producto punto = similitud coseno si están normalizados
    return source_norm @ all_norm.T


def find_top_k_similar_batch(
    source_embeddings: np.ndarray,
    all_embeddings: np.ndarray,
    k: int,
    threshold: float = 0.0,
    exclude_indices: Optional[List[int]] = None,
) -> List[List[Tuple[int, float]]]:
    """
    Encuentra los top-k más similares para cada embedding de origen.

    Args:
        source_embeddings: Embeddings origen
        all_embeddings: Todos los embeddings
        k: Número de similares a retornar
        threshold: Umbral mínimo de similitud
        exclude_indices: Índices a excluir (ej: páginas no enlazables)

    Returns:
        Lista de listas de tuplas (índice, similitud) para cada source
    """
    similarities = compute_similarity_batch(source_embeddings, all_embeddings)

    results = []
    exclude_set = set(exclude_indices) if exclude_indices else set()

    for i, row in enumerate(similarities):
        # Filtrar por umbral y exclusiones
        valid_indices = [
            (idx, float(row[idx]))
            for idx in range(len(row))
            if row[idx] >= threshold and idx not in exclude_set
        ]

        # Ordenar por similitud descendente y tomar top-k
        valid_indices.sort(key=lambda x: x[1], reverse=True)
        results.append(valid_indices[:k])

    return results


# ============================================================================
# PROCESAMIENTO POR LOTES PARA ALGORITMOS
# ============================================================================

class BatchProcessor:
    """
    Procesador por lotes para algoritmos de enlazado interno.

    Divide el trabajo en chunks y proporciona callbacks de progreso.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ):
        """
        Args:
            chunk_size: Tamaño de cada lote
            progress_callback: Función para reportar progreso (current, total, message)
        """
        self.chunk_size = chunk_size
        self.progress_callback = progress_callback
        self._cancelled = False

    def cancel(self):
        """Cancela el procesamiento actual."""
        self._cancelled = True

    def _report_progress(self, current: int, total: int, message: str = ""):
        """Reporta progreso si hay callback."""
        if self.progress_callback:
            self.progress_callback(current, total, message)

    def process_in_batches(
        self,
        source_indices: List[int],
        process_func: Callable[[List[int]], List[Dict]],
    ) -> List[Dict]:
        """
        Procesa índices de origen en lotes.

        Args:
            source_indices: Índices de URLs origen
            process_func: Función que procesa un chunk y retorna recomendaciones

        Returns:
            Lista agregada de todas las recomendaciones
        """
        all_results = []
        chunks = list(chunk_indices(source_indices, self.chunk_size))
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            if self._cancelled:
                break

            self._report_progress(
                i + 1,
                total_chunks,
                f"Procesando lote {i + 1}/{total_chunks} ({len(chunk)} URLs)"
            )

            chunk_results = process_func(chunk)
            all_results.extend(chunk_results)

        return all_results


def should_use_batch_processing(
    n_source: int,
    n_total: int,
    threshold: int = AUTO_BATCH_THRESHOLD,
) -> bool:
    """
    Determina si se debe usar procesamiento por lotes.

    Args:
        n_source: Número de URLs origen
        n_total: Total de URLs
        threshold: Umbral para activar batch

    Returns:
        True si se recomienda usar batch
    """
    # Si hay muchas URLs origen o el total es grande
    return n_source > threshold or n_total > threshold


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Constantes
    "DEFAULT_CHUNK_SIZE",
    "AUTO_BATCH_THRESHOLD",
    # Utilidades
    "estimate_memory_usage",
    "get_optimal_chunk_size",
    "chunk_indices",
    "chunk_dataframe",
    # Cálculo de similitud
    "compute_similarity_batch",
    "find_top_k_similar_batch",
    # Procesador
    "BatchProcessor",
    "should_use_batch_processing",
]
