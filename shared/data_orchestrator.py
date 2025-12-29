"""
DataOrchestrator - Gestión unificada de datos en DuckDB
=======================================================

Orquestador centralizado para todas las operaciones de persistencia de datos
en las aplicaciones de Embedding Insights Suite.

Autor: Embedding Insights
Versión: 1.0.0
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

try:
    import duckdb
except ImportError:
    duckdb = None


class DataOrchestrator:
    """Gestor centralizado de datos con DuckDB."""

    def __init__(self, db_path: str):
        """
        Inicializa el DataOrchestrator.

        Args:
            db_path: Ruta a la base de datos DuckDB del proyecto
        """
        self.db_path = Path(db_path)

        if duckdb is None:
            raise ImportError(
                "DuckDB no está instalado. "
                "Instala con: pip install duckdb"
            )

    def _get_connection(self, read_only: bool = False):
        """
        Obtiene una conexión a la base de datos.

        Args:
            read_only: Si True, abre en modo solo lectura

        Returns:
            Conexión a DuckDB
        """
        return duckdb.connect(str(self.db_path), read_only=read_only)

    # ============================================================
    # URLs Management
    # ============================================================

    def save_urls(self, urls: List[Dict[str, Any]]) -> int:
        """
        Guarda URLs en la base de datos.

        Args:
            urls: Lista de diccionarios con datos de URL:
                - url (str): URL completa
                - title (str, opcional): Título de la página
                - content (str, opcional): Contenido extraído
                - meta_description (str, opcional): Meta descripción
                - word_count (int, opcional): Conteo de palabras

        Returns:
            Número de URLs guardadas
        """
        if not urls:
            return 0

        conn = self._get_connection()

        inserted = 0
        for url_data in urls:
            url = url_data.get("url")
            if not url:
                continue

            # Verificar si ya existe
            existing = conn.execute(
                "SELECT id FROM urls WHERE url = ?", (url,)
            ).fetchone()

            if existing:
                # Actualizar
                conn.execute(
                    """UPDATE urls SET
                       title = ?,
                       content = ?,
                       meta_description = ?,
                       word_count = ?,
                       updated_at = CURRENT_TIMESTAMP
                       WHERE url = ?""",
                    (
                        url_data.get("title"),
                        url_data.get("content"),
                        url_data.get("meta_description"),
                        url_data.get("word_count"),
                        url
                    )
                )
            else:
                # Insertar
                conn.execute(
                    """INSERT INTO urls (url, title, content, meta_description, word_count, scraped_at)
                       VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                    (
                        url,
                        url_data.get("title"),
                        url_data.get("content"),
                        url_data.get("meta_description"),
                        url_data.get("word_count")
                    )
                )
                inserted += 1

        conn.close()
        return inserted

    def get_urls(
        self,
        limit: Optional[int] = None,
        embedding_status: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Obtiene URLs de la base de datos.

        Args:
            limit: Número máximo de URLs a retornar
            embedding_status: Filtrar por estado de embedding ('pending', 'completed', etc.)

        Returns:
            DataFrame con las URLs
        """
        conn = self._get_connection(read_only=True)

        query = "SELECT * FROM urls"
        params = []

        if embedding_status:
            query += " WHERE embedding_status = ?"
            params.append(embedding_status)

        query += " ORDER BY created_at DESC"

        if limit:
            query += f" LIMIT {limit}"

        df = conn.execute(query, params).fetch_df()
        conn.close()

        return df

    # ============================================================
    # Embeddings Management
    # ============================================================

    def save_embeddings(
        self,
        url: str,
        embedding: np.ndarray,
        model: str
    ) -> bool:
        """
        Guarda un embedding para una URL.

        Args:
            url: URL asociada al embedding
            embedding: Vector de embedding (numpy array)
            model: Nombre del modelo usado

        Returns:
            True si se guardó correctamente
        """
        conn = self._get_connection()

        try:
            # Convertir numpy array a bytes
            embedding_bytes = embedding.tobytes()
            dimension = len(embedding)

            # Verificar si ya existe
            existing = conn.execute(
                "SELECT id FROM embeddings WHERE url = ? AND model = ?",
                (url, model)
            ).fetchone()

            if existing:
                # Actualizar
                conn.execute(
                    """UPDATE embeddings SET
                       embedding = ?,
                       dimension = ?,
                       created_at = CURRENT_TIMESTAMP
                       WHERE url = ? AND model = ?""",
                    (embedding_bytes, dimension, url, model)
                )
            else:
                # Insertar
                conn.execute(
                    """INSERT INTO embeddings (url, model, embedding, dimension)
                       VALUES (?, ?, ?, ?)""",
                    (url, model, embedding_bytes, dimension)
                )

            # Actualizar estado en tabla urls
            conn.execute(
                "UPDATE urls SET embedding_status = 'completed' WHERE url = ?",
                (url,)
            )

            conn.close()
            return True

        except Exception as e:
            conn.close()
            raise Exception(f"Error al guardar embedding: {e}")

    def get_embeddings(
        self,
        model: str,
        urls: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Obtiene embeddings de la base de datos.

        Args:
            model: Nombre del modelo
            urls: Lista opcional de URLs específicas

        Returns:
            DataFrame con url, model, embedding (como bytes), dimension
        """
        conn = self._get_connection(read_only=True)

        query = "SELECT url, model, embedding, dimension, created_at FROM embeddings WHERE model = ?"
        params = [model]

        if urls:
            placeholders = ",".join("?" * len(urls))
            query += f" AND url IN ({placeholders})"
            params.extend(urls)

        df = conn.execute(query, params).fetch_df()
        conn.close()

        return df

    def get_embedding_vectors(
        self,
        model: str,
        urls: Optional[List[str]] = None
    ) -> Tuple[List[str], np.ndarray]:
        """
        Obtiene embeddings como matriz numpy.

        Args:
            model: Nombre del modelo
            urls: Lista opcional de URLs específicas

        Returns:
            Tupla de (lista de URLs, matriz numpy de embeddings)
        """
        df = self.get_embeddings(model, urls)

        if df.empty:
            return [], np.array([])

        # Convertir bytes a numpy arrays
        embeddings = []
        urls_list = []

        for _, row in df.iterrows():
            embedding_bytes = row["embedding"]
            dimension = row["dimension"]

            # Reconstruir numpy array desde bytes
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)

            embeddings.append(embedding)
            urls_list.append(row["url"])

        return urls_list, np.array(embeddings)

    # ============================================================
    # GSC Positions Management
    # ============================================================

    def save_gsc_data(
        self,
        df: pd.DataFrame,
        replace: bool = True
    ) -> int:
        """
        Guarda datos de Google Search Console.

        Args:
            df: DataFrame con columnas: keyword, url, position, impressions, clicks, ctr, date
            replace: Si True, reemplaza todos los datos existentes

        Returns:
            Número de registros guardados
        """
        conn = self._get_connection()

        if replace:
            conn.execute("DELETE FROM gsc_positions")

        inserted = 0
        for _, row in df.iterrows():
            conn.execute(
                """INSERT INTO gsc_positions (keyword, url, position, impressions, clicks, ctr, date, country, device)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    row.get("keyword"),
                    row.get("url", ""),
                    row.get("position"),
                    row.get("impressions", 0),
                    row.get("clicks", 0),
                    row.get("ctr", 0.0),
                    row.get("date", datetime.now().date()),
                    row.get("country", "global"),
                    row.get("device", "all")
                )
            )
            inserted += 1

        conn.close()
        return inserted

    def get_gsc_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        keywords: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Obtiene datos de GSC.

        Args:
            start_date: Fecha de inicio (formato YYYY-MM-DD)
            end_date: Fecha de fin (formato YYYY-MM-DD)
            keywords: Lista opcional de keywords específicas

        Returns:
            DataFrame con datos de GSC
        """
        conn = self._get_connection(read_only=True)

        query = "SELECT * FROM gsc_positions WHERE 1=1"
        params = []

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        if keywords:
            placeholders = ",".join("?" * len(keywords))
            query += f" AND keyword IN ({placeholders})"
            params.extend(keywords)

        query += " ORDER BY date DESC, keyword"

        df = conn.execute(query, params).fetch_df()
        conn.close()

        return df

    # ============================================================
    # Keyword Families Management
    # ============================================================

    def save_keyword_families(self, families: Dict[str, List[str]]) -> int:
        """
        Guarda familias de keywords.

        Args:
            families: Diccionario {nombre_familia: [keywords]}

        Returns:
            Número de familias guardadas
        """
        conn = self._get_connection()

        # Limpiar familias existentes
        conn.execute("DELETE FROM keyword_families")

        inserted = 0
        for family_name, keywords_list in families.items():
            conn.execute(
                """INSERT INTO keyword_families (family_name, keywords, description)
                   VALUES (?, ?, ?)""",
                (family_name, json.dumps(keywords_list), "")
            )
            inserted += 1

        conn.close()
        return inserted

    def get_keyword_families(self) -> Dict[str, List[str]]:
        """
        Obtiene todas las familias de keywords.

        Returns:
            Diccionario {nombre_familia: [keywords]}
        """
        conn = self._get_connection(read_only=True)

        df = conn.execute("SELECT family_name, keywords FROM keyword_families").fetch_df()
        conn.close()

        families = {}
        for _, row in df.iterrows():
            family_name = row["family_name"]
            keywords_list = json.loads(row["keywords"])
            families[family_name] = keywords_list

        return families

    # ============================================================
    # Semantic Relations Management
    # ============================================================

    def save_semantic_relations(
        self,
        relations: List[Dict[str, Any]],
        replace: bool = True
    ) -> int:
        """
        Guarda relaciones semánticas entre URLs.

        Args:
            relations: Lista de diccionarios con:
                - source_url: URL origen
                - target_url: URL destino
                - similarity_score: Score de similitud
                - relation_type: Tipo de relación
                - anchor_suggestion: Sugerencia de anchor text
            replace: Si True, reemplaza todas las relaciones

        Returns:
            Número de relaciones guardadas
        """
        conn = self._get_connection()

        if replace:
            conn.execute("DELETE FROM semantic_relations")

        inserted = 0
        for rel in relations:
            conn.execute(
                """INSERT INTO semantic_relations
                   (source_url, target_url, similarity_score, relation_type, anchor_suggestion)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    rel.get("source_url"),
                    rel.get("target_url"),
                    rel.get("similarity_score"),
                    rel.get("relation_type", "semantic"),
                    rel.get("anchor_suggestion", "")
                )
            )
            inserted += 1

        conn.close()
        return inserted

    def get_semantic_relations(
        self,
        source_url: Optional[str] = None,
        min_score: float = 0.0
    ) -> pd.DataFrame:
        """
        Obtiene relaciones semánticas.

        Args:
            source_url: Filtrar por URL origen específica
            min_score: Score mínimo de similitud

        Returns:
            DataFrame con relaciones
        """
        conn = self._get_connection(read_only=True)

        query = "SELECT * FROM semantic_relations WHERE similarity_score >= ?"
        params = [min_score]

        if source_url:
            query += " AND source_url = ?"
            params.append(source_url)

        query += " ORDER BY similarity_score DESC"

        df = conn.execute(query, params).fetch_df()
        conn.close()

        return df

    # ============================================================
    # Entities Management (Knowledge Graph)
    # ============================================================

    def save_entities(
        self,
        entities: List[Dict[str, Any]],
        replace: bool = False
    ) -> int:
        """
        Guarda entidades extraídas del contenido.

        Args:
            entities: Lista de diccionarios con:
                - url: URL de origen
                - entity_text: Texto de la entidad
                - entity_type: Tipo (PERSON, ORG, LOC, etc.)
                - frequency: Frecuencia de aparición
                - canonical_form: Forma lemmatizada
            replace: Si True, reemplaza todas las entidades

        Returns:
            Número de entidades guardadas
        """
        conn = self._get_connection()

        if replace:
            conn.execute("DELETE FROM entities")

        inserted = 0
        for ent in entities:
            conn.execute(
                """INSERT INTO entities (url, entity_text, entity_type, frequency, canonical_form)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    ent.get("url"),
                    ent.get("entity_text"),
                    ent.get("entity_type"),
                    ent.get("frequency", 1),
                    ent.get("canonical_form", ent.get("entity_text"))
                )
            )
            inserted += 1

        conn.close()
        return inserted

    def get_entities(
        self,
        url: Optional[str] = None,
        entity_type: Optional[str] = None,
        min_frequency: int = 1
    ) -> pd.DataFrame:
        """
        Obtiene entidades.

        Args:
            url: Filtrar por URL específica
            entity_type: Filtrar por tipo de entidad
            min_frequency: Frecuencia mínima

        Returns:
            DataFrame con entidades
        """
        conn = self._get_connection(read_only=True)

        query = "SELECT * FROM entities WHERE frequency >= ?"
        params = [min_frequency]

        if url:
            query += " AND url = ?"
            params.append(url)

        if entity_type:
            query += " AND entity_type = ?"
            params.append(entity_type)

        query += " ORDER BY frequency DESC"

        df = conn.execute(query, params).fetch_df()
        conn.close()

        return df

    # ============================================================
    # Clusters Management
    # ============================================================

    def save_clusters(
        self,
        clusters: pd.DataFrame,
        model: str,
        replace: bool = True
    ) -> int:
        """
        Guarda resultados de clustering.

        Args:
            clusters: DataFrame con url, cluster_id, cluster_label, distance_to_centroid
            model: Nombre del modelo usado
            replace: Si True, reemplaza clusters del mismo modelo

        Returns:
            Número de registros guardados
        """
        conn = self._get_connection()

        if replace:
            conn.execute("DELETE FROM clusters WHERE model = ?", (model,))

        inserted = 0
        for _, row in clusters.iterrows():
            conn.execute(
                """INSERT INTO clusters (url, cluster_id, cluster_label, distance_to_centroid, model)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    row.get("url"),
                    row.get("cluster_id"),
                    row.get("cluster_label"),
                    row.get("distance_to_centroid"),
                    model
                )
            )
            inserted += 1

        conn.close()
        return inserted

    def get_clusters(self, model: str) -> pd.DataFrame:
        """
        Obtiene resultados de clustering.

        Args:
            model: Nombre del modelo

        Returns:
            DataFrame con clusters
        """
        conn = self._get_connection(read_only=True)

        df = conn.execute(
            "SELECT * FROM clusters WHERE model = ? ORDER BY cluster_id, url",
            (model,)
        ).fetch_df()
        conn.close()

        return df

    # ============================================================
    # FAQ Analysis Management
    # ============================================================

    def save_faq_analysis(
        self,
        faqs: List[Dict[str, Any]],
        replace: bool = True
    ) -> int:
        """
        Guarda análisis de FAQs.

        Args:
            faqs: Lista de diccionarios con:
                - question: Pregunta
                - answer: Respuesta
                - url: URL relacionada
                - similarity_score: Score de similitud
                - keywords: Lista de keywords (se convertirá a JSON)
            replace: Si True, reemplaza todos los FAQs

        Returns:
            Número de FAQs guardados
        """
        conn = self._get_connection()

        if replace:
            conn.execute("DELETE FROM faq_analysis")

        inserted = 0
        for faq in faqs:
            keywords_json = json.dumps(faq.get("keywords", []))

            conn.execute(
                """INSERT INTO faq_analysis (question, answer, url, similarity_score, keywords)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    faq.get("question"),
                    faq.get("answer"),
                    faq.get("url"),
                    faq.get("similarity_score"),
                    keywords_json
                )
            )
            inserted += 1

        conn.close()
        return inserted

    def get_faq_analysis(
        self,
        url: Optional[str] = None,
        min_score: float = 0.0
    ) -> pd.DataFrame:
        """
        Obtiene análisis de FAQs.

        Args:
            url: Filtrar por URL específica
            min_score: Score mínimo de similitud

        Returns:
            DataFrame con FAQs
        """
        conn = self._get_connection(read_only=True)

        query = "SELECT * FROM faq_analysis WHERE similarity_score >= ?"
        params = [min_score]

        if url:
            query += " AND url = ?"
            params.append(url)

        query += " ORDER BY similarity_score DESC"

        df = conn.execute(query, params).fetch_df()
        conn.close()

        return df

    # ============================================================
    # Utility Methods
    # ============================================================

    def get_stats(self) -> Dict[str, int]:
        """
        Obtiene estadísticas generales de la base de datos.

        Returns:
            Diccionario con conteos de cada tabla
        """
        conn = self._get_connection(read_only=True)

        stats = {}
        tables = [
            "urls", "gsc_positions", "embeddings", "keyword_families",
            "semantic_relations", "entities", "clusters", "faq_analysis"
        ]

        for table in tables:
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                stats[table] = count
            except Exception:
                stats[table] = 0

        conn.close()
        return stats

    def clear_all_data(self, confirm: bool = False):
        """
        Limpia todos los datos de todas las tablas.

        Args:
            confirm: Debe ser True para confirmar la operación

        Raises:
            ValueError: Si confirm no es True
        """
        if not confirm:
            raise ValueError("Debes confirmar la limpieza con confirm=True")

        conn = self._get_connection()

        tables = [
            "urls", "gsc_positions", "embeddings", "keyword_families",
            "semantic_relations", "entities", "clusters", "faq_analysis"
        ]

        for table in tables:
            conn.execute(f"DELETE FROM {table}")

        conn.close()


# Función helper para obtener instancia
def get_data_orchestrator(project_config: Dict[str, Any]) -> DataOrchestrator:
    """
    Obtiene una instancia de DataOrchestrator para un proyecto.

    Args:
        project_config: Configuración del proyecto (debe incluir 'db_path')

    Returns:
        Instancia de DataOrchestrator
    """
    db_path = project_config.get("db_path")
    if not db_path:
        raise ValueError("project_config debe incluir 'db_path'")

    return DataOrchestrator(db_path)
