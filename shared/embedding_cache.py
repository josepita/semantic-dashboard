"""
Embedding Cache - Sistema de caché persistente para embeddings
===============================================================

Sistema de caché que combina DuckDB para persistencia y opcionalmente
FAISS para búsquedas rápidas de similitud.

Autor: Embedding Insights
Versión: 1.0.0
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from data_orchestrator import DataOrchestrator


class EmbeddingCache:
    """Sistema de caché de embeddings con DuckDB y FAISS."""

    def __init__(
        self,
        project_path: str,
        model_name: str,
        use_faiss: bool = True
    ):
        """
        Inicializa el caché de embeddings.

        Args:
            project_path: Ruta al proyecto (donde está database.duckdb)
            model_name: Nombre del modelo de embeddings
            use_faiss: Si True, usa FAISS para búsquedas rápidas
        """
        self.project_path = Path(project_path)
        self.model_name = model_name
        self.use_faiss = use_faiss and FAISS_AVAILABLE

        # Paths
        self.db_path = self.project_path / "database.duckdb"
        self.embeddings_dir = self.project_path / "embeddings"
        self.embeddings_dir.mkdir(exist_ok=True)

        # FAISS index
        self.faiss_index_path = self.embeddings_dir / f"{model_name}.faiss"
        self.faiss_metadata_path = self.embeddings_dir / f"{model_name}_metadata.json"

        # Inicializar DataOrchestrator
        self.orchestrator = DataOrchestrator(str(self.db_path))

        # FAISS index (se cargará bajo demanda)
        self._faiss_index = None
        self._faiss_urls = []

    def add_embedding(
        self,
        url: str,
        embedding: np.ndarray,
        rebuild_index: bool = False
    ) -> bool:
        """
        Añade un embedding al caché.

        Args:
            url: URL asociada
            embedding: Vector de embedding
            rebuild_index: Si True, reconstruye el índice FAISS

        Returns:
            True si se añadió correctamente
        """
        # Guardar en DuckDB
        success = self.orchestrator.save_embeddings(url, embedding, self.model_name)

        if success and self.use_faiss and rebuild_index:
            # Reconstruir índice FAISS
            self._rebuild_faiss_index()

        return success

    def add_embeddings_batch(
        self,
        urls: List[str],
        embeddings: np.ndarray
    ) -> int:
        """
        Añade múltiples embeddings de forma batch.

        Args:
            urls: Lista de URLs
            embeddings: Matriz de embeddings (n_samples x dimension)

        Returns:
            Número de embeddings añadidos
        """
        count = 0
        for url, embedding in zip(urls, embeddings):
            if self.orchestrator.save_embeddings(url, embedding, self.model_name):
                count += 1

        # Reconstruir índice FAISS si está habilitado
        if self.use_faiss:
            self._rebuild_faiss_index()

        return count

    def get_embedding(self, url: str) -> Optional[np.ndarray]:
        """
        Obtiene el embedding de una URL.

        Args:
            url: URL a buscar

        Returns:
            Embedding como numpy array, o None si no existe
        """
        urls_list, embeddings = self.orchestrator.get_embedding_vectors(
            self.model_name,
            urls=[url]
        )

        if len(embeddings) == 0:
            return None

        return embeddings[0]

    def get_all_embeddings(self) -> Tuple[List[str], np.ndarray]:
        """
        Obtiene todos los embeddings del caché.

        Returns:
            Tupla de (lista de URLs, matriz de embeddings)
        """
        return self.orchestrator.get_embedding_vectors(self.model_name)

    def has_embedding(self, url: str) -> bool:
        """
        Verifica si existe un embedding para una URL.

        Args:
            url: URL a verificar

        Returns:
            True si existe el embedding
        """
        return self.get_embedding(url) is not None

    def search_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        exclude_urls: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Busca embeddings similares usando FAISS o búsqueda lineal.

        Args:
            query_embedding: Vector de query
            top_k: Número de resultados a retornar
            exclude_urls: URLs a excluir de los resultados

        Returns:
            Lista de diccionarios con 'url', 'similarity', 'distance'
        """
        if self.use_faiss:
            return self._search_with_faiss(query_embedding, top_k, exclude_urls)
        else:
            return self._search_linear(query_embedding, top_k, exclude_urls)

    def _search_with_faiss(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        exclude_urls: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Búsqueda usando FAISS."""
        # Cargar índice si no está cargado
        if self._faiss_index is None:
            self._load_faiss_index()

        if self._faiss_index is None:
            # Fallback a búsqueda lineal si no hay índice
            return self._search_linear(query_embedding, top_k, exclude_urls)

        # Normalizar query
        query_normalized = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_normalized)

        # Buscar
        distances, indices = self._faiss_index.search(query_normalized, top_k * 2)

        # Construir resultados
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= len(self._faiss_urls):
                continue

            url = self._faiss_urls[idx]

            # Excluir URLs especificadas
            if exclude_urls and url in exclude_urls:
                continue

            # Convertir distancia L2 a similitud coseno
            # distance = 2 - 2*similarity (para vectores normalizados)
            similarity = 1 - (dist / 2)

            results.append({
                "url": url,
                "similarity": float(similarity),
                "distance": float(dist)
            })

            if len(results) >= top_k:
                break

        return results

    def _search_linear(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        exclude_urls: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Búsqueda lineal (sin FAISS)."""
        urls, embeddings = self.get_all_embeddings()

        if len(embeddings) == 0:
            return []

        # Normalizar
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Calcular similitudes coseno
        similarities = np.dot(embeddings_norm, query_norm)

        # Ordenar por similitud
        indices = np.argsort(similarities)[::-1]

        # Construir resultados
        results = []
        for idx in indices:
            url = urls[idx]

            # Excluir URLs especificadas
            if exclude_urls and url in exclude_urls:
                continue

            results.append({
                "url": url,
                "similarity": float(similarities[idx]),
                "distance": float(1 - similarities[idx])
            })

            if len(results) >= top_k:
                break

        return results

    def _rebuild_faiss_index(self):
        """Reconstruye el índice FAISS desde DuckDB."""
        if not self.use_faiss:
            return

        urls, embeddings = self.get_all_embeddings()

        if len(embeddings) == 0:
            return

        # Crear índice FAISS
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner Product (para coseno con vectores normalizados)

        # Normalizar embeddings
        embeddings_normalized = embeddings.astype('float32')
        faiss.normalize_L2(embeddings_normalized)

        # Añadir al índice
        index.add(embeddings_normalized)

        # Guardar índice
        faiss.write_index(index, str(self.faiss_index_path))

        # Guardar metadata (mapeo índice -> URL)
        metadata = {
            "model": self.model_name,
            "dimension": dimension,
            "count": len(urls),
            "urls": urls
        }

        with open(self.faiss_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Actualizar caché en memoria
        self._faiss_index = index
        self._faiss_urls = urls

    def _load_faiss_index(self):
        """Carga el índice FAISS desde disco."""
        if not self.use_faiss:
            return

        if not self.faiss_index_path.exists():
            # No hay índice, crearlo
            self._rebuild_faiss_index()
            return

        try:
            # Cargar índice
            self._faiss_index = faiss.read_index(str(self.faiss_index_path))

            # Cargar metadata
            with open(self.faiss_metadata_path, 'r') as f:
                metadata = json.load(f)

            self._faiss_urls = metadata.get("urls", [])

        except Exception as e:
            print(f"Error al cargar índice FAISS: {e}")
            # Reconstruir si falla
            self._rebuild_faiss_index()

    def sync_from_db(self):
        """Sincroniza el caché FAISS con los datos en DuckDB."""
        if self.use_faiss:
            self._rebuild_faiss_index()

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del caché.

        Returns:
            Diccionario con estadísticas
        """
        urls, embeddings = self.get_all_embeddings()

        stats = {
            "model": self.model_name,
            "count": len(urls),
            "dimension": embeddings.shape[1] if len(embeddings) > 0 else 0,
            "faiss_enabled": self.use_faiss,
            "faiss_index_exists": self.faiss_index_path.exists() if self.use_faiss else False
        }

        # Tamaño del índice FAISS
        if self.use_faiss and self.faiss_index_path.exists():
            stats["faiss_size_mb"] = round(
                self.faiss_index_path.stat().st_size / (1024 * 1024), 2
            )

        return stats

    def clear_cache(self, confirm: bool = False):
        """
        Limpia el caché completamente.

        Args:
            confirm: Debe ser True para confirmar

        Raises:
            ValueError: Si confirm no es True
        """
        if not confirm:
            raise ValueError("Debes confirmar con confirm=True")

        # Eliminar archivos FAISS
        if self.faiss_index_path.exists():
            self.faiss_index_path.unlink()

        if self.faiss_metadata_path.exists():
            self.faiss_metadata_path.unlink()

        # Limpiar caché en memoria
        self._faiss_index = None
        self._faiss_urls = []

        # Nota: No limpiamos DuckDB aquí, eso lo maneja DataOrchestrator


def get_or_compute_embedding(
    cache: EmbeddingCache,
    url: str,
    compute_fn,
    **compute_kwargs
) -> np.ndarray:
    """
    Helper para obtener embedding del caché o computarlo si no existe.

    Args:
        cache: Instancia de EmbeddingCache
        url: URL a procesar
        compute_fn: Función que computa el embedding (debe retornar np.ndarray)
        **compute_kwargs: Argumentos adicionales para compute_fn

    Returns:
        Embedding como numpy array
    """
    # Intentar obtener del caché
    embedding = cache.get_embedding(url)

    if embedding is not None:
        return embedding

    # Computar si no existe
    embedding = compute_fn(url, **compute_kwargs)

    # Guardar en caché
    cache.add_embedding(url, embedding, rebuild_index=False)

    return embedding
