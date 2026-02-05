"""
Brief Storage Module
====================
Gestiona la persistencia de carpetas y briefs en SQLite.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ══════════════════════════════════════════════════════════
# DATABASE SETUP
# ══════════════════════════════════════════════════════════

def get_db_path(project_path: Optional[str] = None) -> str:
    """Obtiene la ruta de la base de datos."""
    if project_path:
        db_dir = Path(project_path)
    else:
        db_dir = Path(__file__).parent.parent / "data"

    db_dir.mkdir(parents=True, exist_ok=True)
    return str(db_dir / "briefs.db")


def init_database(db_path: Optional[str] = None) -> str:
    """Inicializa la base de datos con las tablas necesarias."""
    if db_path is None:
        db_path = get_db_path()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Tabla de carpetas
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS folders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Tabla de briefs
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS briefs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            folder_id INTEGER,
            keyword TEXT NOT NULL,
            country TEXT DEFAULT 'ES',
            source TEXT CHECK(source IN ('excel', 'direct')) DEFAULT 'direct',
            status TEXT CHECK(status IN ('pending', 'serp_done', 'keywords_done', 'completed')) DEFAULT 'pending',

            -- Datos generados (JSON)
            hn_structure TEXT,
            serp_results TEXT,
            keyword_opportunities TEXT,
            title_proposals TEXT,
            meta_proposals TEXT,
            selected_title TEXT,
            selected_meta TEXT,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP,

            FOREIGN KEY (folder_id) REFERENCES folders(id) ON DELETE SET NULL
        )
    """)

    conn.commit()
    conn.close()

    return db_path


# ══════════════════════════════════════════════════════════
# FOLDER OPERATIONS
# ══════════════════════════════════════════════════════════

def create_folder(name: str, db_path: Optional[str] = None) -> Optional[int]:
    """Crea una nueva carpeta. Retorna el ID o None si ya existe."""
    if db_path is None:
        db_path = get_db_path()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("INSERT INTO folders (name) VALUES (?)", (name,))
        conn.commit()
        folder_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        folder_id = None
    finally:
        conn.close()

    return folder_id


def get_folders(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Obtiene todas las carpetas con conteo de briefs."""
    if db_path is None:
        db_path = get_db_path()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            f.id,
            f.name,
            f.created_at,
            COUNT(b.id) as brief_count
        FROM folders f
        LEFT JOIN briefs b ON f.id = b.folder_id
        GROUP BY f.id
        ORDER BY f.created_at DESC
    """)

    folders = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return folders


def delete_folder(folder_id: int, db_path: Optional[str] = None) -> bool:
    """Elimina una carpeta (los briefs quedan sin carpeta)."""
    if db_path is None:
        db_path = get_db_path()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM folders WHERE id = ?", (folder_id,))
    deleted = cursor.rowcount > 0

    conn.commit()
    conn.close()

    return deleted


def rename_folder(folder_id: int, new_name: str, db_path: Optional[str] = None) -> bool:
    """Renombra una carpeta."""
    if db_path is None:
        db_path = get_db_path()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("UPDATE folders SET name = ? WHERE id = ?", (new_name, folder_id))
        conn.commit()
        success = cursor.rowcount > 0
    except sqlite3.IntegrityError:
        success = False
    finally:
        conn.close()

    return success


# ══════════════════════════════════════════════════════════
# BRIEF OPERATIONS
# ══════════════════════════════════════════════════════════

def create_brief(
    keyword: str,
    country: str = "ES",
    source: str = "direct",
    folder_id: Optional[int] = None,
    hn_structure: Optional[Dict] = None,
    db_path: Optional[str] = None
) -> int:
    """Crea un nuevo brief. Retorna el ID."""
    if db_path is None:
        db_path = get_db_path()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    hn_json = json.dumps(hn_structure, ensure_ascii=False) if hn_structure else None

    cursor.execute("""
        INSERT INTO briefs (keyword, country, source, folder_id, hn_structure, status)
        VALUES (?, ?, ?, ?, ?, 'pending')
    """, (keyword, country, source, folder_id, hn_json))

    conn.commit()
    brief_id = cursor.lastrowid
    conn.close()

    return brief_id


def get_briefs(
    folder_id: Optional[int] = None,
    status: Optional[str] = None,
    db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Obtiene briefs filtrados por carpeta y/o estado."""
    if db_path is None:
        db_path = get_db_path()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = """
        SELECT
            b.*,
            f.name as folder_name
        FROM briefs b
        LEFT JOIN folders f ON b.folder_id = f.id
        WHERE 1=1
    """
    params = []

    if folder_id is not None:
        query += " AND b.folder_id = ?"
        params.append(folder_id)

    if status is not None:
        query += " AND b.status = ?"
        params.append(status)

    query += " ORDER BY b.created_at DESC"

    cursor.execute(query, params)
    briefs = [dict(row) for row in cursor.fetchall()]
    conn.close()

    # Parse JSON fields
    json_fields = ['hn_structure', 'serp_results', 'keyword_opportunities', 'title_proposals', 'meta_proposals']
    for brief in briefs:
        for field in json_fields:
            if brief.get(field):
                try:
                    brief[field] = json.loads(brief[field])
                except json.JSONDecodeError:
                    pass

    return briefs


def get_brief(brief_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Obtiene un brief por ID."""
    if db_path is None:
        db_path = get_db_path()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            b.*,
            f.name as folder_name
        FROM briefs b
        LEFT JOIN folders f ON b.folder_id = f.id
        WHERE b.id = ?
    """, (brief_id,))

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    brief = dict(row)

    # Parse JSON fields
    json_fields = ['hn_structure', 'serp_results', 'keyword_opportunities', 'title_proposals', 'meta_proposals']
    for field in json_fields:
        if brief.get(field):
            try:
                brief[field] = json.loads(brief[field])
            except json.JSONDecodeError:
                pass

    return brief


def update_brief(
    brief_id: int,
    db_path: Optional[str] = None,
    **kwargs
) -> bool:
    """Actualiza campos de un brief."""
    if db_path is None:
        db_path = get_db_path()

    if not kwargs:
        return False

    # Convertir campos JSON
    json_fields = ['hn_structure', 'serp_results', 'keyword_opportunities', 'title_proposals', 'meta_proposals']
    for field in json_fields:
        if field in kwargs and kwargs[field] is not None:
            kwargs[field] = json.dumps(kwargs[field], ensure_ascii=False)

    # Añadir timestamp de actualización
    kwargs['updated_at'] = datetime.now().isoformat()

    # Construir query
    set_clause = ", ".join([f"{k} = ?" for k in kwargs.keys()])
    values = list(kwargs.values()) + [brief_id]

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"UPDATE briefs SET {set_clause} WHERE id = ?", values)
    conn.commit()
    success = cursor.rowcount > 0
    conn.close()

    return success


def delete_brief(brief_id: int, db_path: Optional[str] = None) -> bool:
    """Elimina un brief."""
    if db_path is None:
        db_path = get_db_path()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM briefs WHERE id = ?", (brief_id,))
    deleted = cursor.rowcount > 0

    conn.commit()
    conn.close()

    return deleted


def move_brief_to_folder(brief_id: int, folder_id: Optional[int], db_path: Optional[str] = None) -> bool:
    """Mueve un brief a otra carpeta (o a ninguna si folder_id es None)."""
    return update_brief(brief_id, db_path=db_path, folder_id=folder_id)


def get_brief_stats(db_path: Optional[str] = None) -> Dict[str, int]:
    """Obtiene estadísticas de briefs."""
    if db_path is None:
        db_path = get_db_path()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
            SUM(CASE WHEN status = 'serp_done' THEN 1 ELSE 0 END) as serp_done,
            SUM(CASE WHEN status = 'keywords_done' THEN 1 ELSE 0 END) as keywords_done,
            SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed
        FROM briefs
    """)

    row = cursor.fetchone()
    conn.close()

    return {
        'total': row[0] or 0,
        'pending': row[1] or 0,
        'serp_done': row[2] or 0,
        'keywords_done': row[3] or 0,
        'completed': row[4] or 0
    }
