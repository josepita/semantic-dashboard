"""
DuckDB Schema Definition
========================

Schema inicial para la base de datos DuckDB de cada proyecto.
Define las tablas para URLs, embeddings, posiciones GSC, y familias de keywords.

Autor: Embedding Insights
Versión: 1.0.0
"""


def get_initial_schema() -> str:
    """
    Retorna el SQL para crear el schema inicial de DuckDB.

    Returns:
        String con comandos SQL para inicializar la base de datos
    """
    return """
    -- ============================================================
    -- TABLA: urls
    -- Almacena todas las URLs del proyecto con su contenido
    -- ============================================================
    CREATE TABLE IF NOT EXISTS urls (
        id INTEGER PRIMARY KEY,
        url TEXT UNIQUE NOT NULL,
        title TEXT,
        content TEXT,
        meta_description TEXT,
        scraped_at TIMESTAMP,
        embedding_status TEXT DEFAULT 'pending',
        word_count INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_urls_url ON urls(url);
    CREATE INDEX IF NOT EXISTS idx_urls_embedding_status ON urls(embedding_status);


    -- ============================================================
    -- TABLA: gsc_positions
    -- Datos de Google Search Console (rank tracking)
    -- ============================================================
    CREATE TABLE IF NOT EXISTS gsc_positions (
        id INTEGER PRIMARY KEY,
        keyword TEXT NOT NULL,
        url TEXT NOT NULL,
        position REAL,
        impressions INTEGER,
        clicks INTEGER,
        ctr REAL,
        date DATE,
        country TEXT DEFAULT 'global',
        device TEXT DEFAULT 'all',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_gsc_keyword ON gsc_positions(keyword);
    CREATE INDEX IF NOT EXISTS idx_gsc_url ON gsc_positions(url);
    CREATE INDEX IF NOT EXISTS idx_gsc_date ON gsc_positions(date);
    CREATE INDEX IF NOT EXISTS idx_gsc_position ON gsc_positions(position);


    -- ============================================================
    -- TABLA: embeddings
    -- Almacena embeddings generados para cada URL
    -- ============================================================
    CREATE TABLE IF NOT EXISTS embeddings (
        id INTEGER PRIMARY KEY,
        url TEXT NOT NULL,
        model TEXT NOT NULL,
        embedding BLOB NOT NULL,
        dimension INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_embeddings_url ON embeddings(url);
    CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_embeddings_url_model ON embeddings(url, model);


    -- ============================================================
    -- TABLA: keyword_families
    -- Familias de keywords para agrupación semántica
    -- ============================================================
    CREATE TABLE IF NOT EXISTS keyword_families (
        id INTEGER PRIMARY KEY,
        family_name TEXT NOT NULL UNIQUE,
        keywords TEXT NOT NULL,  -- JSON array de keywords
        description TEXT,
        color TEXT,  -- Para visualización
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_families_name ON keyword_families(family_name);


    -- ============================================================
    -- TABLA: semantic_relations
    -- Relaciones semánticas entre URLs (para linking)
    -- ============================================================
    CREATE TABLE IF NOT EXISTS semantic_relations (
        id INTEGER PRIMARY KEY,
        source_url TEXT NOT NULL,
        target_url TEXT NOT NULL,
        similarity_score REAL NOT NULL,
        relation_type TEXT,  -- 'semantic', 'structural', 'hybrid'
        anchor_suggestion TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_relations_source ON semantic_relations(source_url);
    CREATE INDEX IF NOT EXISTS idx_relations_target ON semantic_relations(target_url);
    CREATE INDEX IF NOT EXISTS idx_relations_score ON semantic_relations(similarity_score);


    -- ============================================================
    -- TABLA: entities
    -- Entidades extraídas del contenido (Knowledge Graph)
    -- ============================================================
    CREATE TABLE IF NOT EXISTS entities (
        id INTEGER PRIMARY KEY,
        url TEXT NOT NULL,
        entity_text TEXT NOT NULL,
        entity_type TEXT,  -- PERSON, ORG, LOC, etc.
        frequency INTEGER DEFAULT 1,
        canonical_form TEXT,  -- Forma lemmatizada
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_entities_url ON entities(url);
    CREATE INDEX IF NOT EXISTS idx_entities_text ON entities(entity_text);
    CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
    CREATE INDEX IF NOT EXISTS idx_entities_canonical ON entities(canonical_form);


    -- ============================================================
    -- TABLA: clusters
    -- Resultados de clustering de URLs
    -- ============================================================
    CREATE TABLE IF NOT EXISTS clusters (
        id INTEGER PRIMARY KEY,
        url TEXT NOT NULL,
        cluster_id INTEGER NOT NULL,
        cluster_label TEXT,
        distance_to_centroid REAL,
        model TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_clusters_url ON clusters(url);
    CREATE INDEX IF NOT EXISTS idx_clusters_id ON clusters(cluster_id);
    CREATE INDEX IF NOT EXISTS idx_clusters_model ON clusters(model);


    -- ============================================================
    -- TABLA: faq_analysis
    -- Análisis de FAQs con relevancia semántica
    -- ============================================================
    CREATE TABLE IF NOT EXISTS faq_analysis (
        id INTEGER PRIMARY KEY,
        question TEXT NOT NULL,
        answer TEXT,
        url TEXT,  -- URL relacionada (si aplica)
        similarity_score REAL,
        keywords TEXT,  -- JSON array
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_faq_url ON faq_analysis(url);
    CREATE INDEX IF NOT EXISTS idx_faq_score ON faq_analysis(similarity_score);


    -- ============================================================
    -- TABLA: project_metadata
    -- Metadatos generales del proyecto
    -- ============================================================
    CREATE TABLE IF NOT EXISTS project_metadata (
        key TEXT PRIMARY KEY,
        value TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Insertar versión del schema
    INSERT OR IGNORE INTO project_metadata (key, value)
    VALUES ('schema_version', '1.0.0');

    INSERT OR IGNORE INTO project_metadata (key, value)
    VALUES ('created_at', CURRENT_TIMESTAMP::TEXT);
    """


def get_migration_v1_to_v2() -> str:
    """
    Migración de schema v1.0.0 a v2.0.0 (ejemplo para futuras migraciones).

    Returns:
        String con comandos SQL para la migración
    """
    return """
    -- Ejemplo de migración futura
    -- ALTER TABLE urls ADD COLUMN canonical_url TEXT;
    -- UPDATE project_metadata SET value = '2.0.0' WHERE key = 'schema_version';
    """


def get_schema_version(conn) -> str:
    """
    Obtiene la versión actual del schema.

    Args:
        conn: Conexión a DuckDB

    Returns:
        Versión del schema (ej: '1.0.0')
    """
    try:
        result = conn.execute(
            "SELECT value FROM project_metadata WHERE key = 'schema_version'"
        ).fetchone()
        if result:
            return result[0]
        return "unknown"
    except Exception:
        return "unknown"


def needs_migration(conn) -> bool:
    """
    Verifica si el schema necesita migración.

    Args:
        conn: Conexión a DuckDB

    Returns:
        True si necesita migración
    """
    current_version = get_schema_version(conn)
    target_version = "1.0.0"
    return current_version != target_version


def apply_migrations(conn):
    """
    Aplica migraciones necesarias al schema.

    Args:
        conn: Conexión a DuckDB
    """
    current_version = get_schema_version(conn)

    # Backup antes de migrar (futuro)
    # ...

    # Aplicar migraciones en orden
    if current_version == "unknown":
        # Crear schema desde cero
        conn.execute(get_initial_schema())
    elif current_version == "1.0.0":
        # Ya está actualizado
        pass
    # Futuras migraciones:
    # elif current_version == "1.0.0":
    #     conn.execute(get_migration_v1_to_v2())

    print(f"✅ Schema actualizado a versión 1.0.0")
