"""
ProjectManager - Gestión de proyectos multi-cliente
===================================================

Sistema de gestión de proyectos para las aplicaciones de Embedding Insights Suite.
Permite crear, cargar, listar y eliminar proyectos con estructura independiente.

Autor: Embedding Insights
Versión: 1.0.0
"""

import json
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import os


class ProjectManager:
    """Gestor de proyectos con workspace independiente."""

    def __init__(self, workspace_root: Optional[Path] = None):
        """
        Inicializa el ProjectManager.

        Args:
            workspace_root: Ruta al directorio workspace. Si es None, usa workspace/
                          en la raíz del proyecto.
        """
        if workspace_root is None:
            # Buscar la raíz del proyecto (donde está workspace/)
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent
            workspace_root = project_root / "workspace"

        self.workspace_root = Path(workspace_root)
        self.projects_dir = self.workspace_root / "projects"
        self.config_file = self.workspace_root / ".workspace_config.json"

        # Asegurar que existen los directorios
        self.workspace_root.mkdir(exist_ok=True)
        self.projects_dir.mkdir(exist_ok=True)

        # Crear config si no existe
        if not self.config_file.exists():
            self._init_workspace_config()

    def _init_workspace_config(self):
        """Inicializa el archivo de configuración del workspace."""
        config = {
            "version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "last_project": None,
            "settings": {
                "auto_backup": True,
                "max_embedding_cache_mb": 500,
                "default_embedding_model": "paraphrase-multilingual-MiniLM-L12-v2"
            }
        }
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def _load_workspace_config(self) -> Dict[str, Any]:
        """Carga la configuración del workspace."""
        if not self.config_file.exists():
            self._init_workspace_config()

        with open(self.config_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_workspace_config(self, config: Dict[str, Any]):
        """Guarda la configuración del workspace."""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def list_projects(self) -> List[Dict[str, Any]]:
        """
        Lista todos los proyectos disponibles.

        Returns:
            Lista de diccionarios con información de cada proyecto:
            - name: Nombre del proyecto
            - domain: Dominio principal
            - created_at: Fecha de creación
            - path: Ruta al proyecto
            - size_mb: Tamaño aproximado en MB
        """
        projects = []

        if not self.projects_dir.exists():
            return projects

        for project_dir in self.projects_dir.iterdir():
            if not project_dir.is_dir():
                continue

            config_path = project_dir / "config.json"
            if not config_path.exists():
                continue

            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                # Calcular tamaño aproximado
                size_bytes = sum(
                    f.stat().st_size
                    for f in project_dir.rglob('*')
                    if f.is_file()
                )
                size_mb = round(size_bytes / (1024 * 1024), 2)

                projects.append({
                    "name": project_dir.name,
                    "domain": config.get("domain", ""),
                    "created_at": config.get("created_at", ""),
                    "path": str(project_dir),
                    "size_mb": size_mb,
                    "description": config.get("description", "")
                })
            except Exception as e:
                print(f"Error al cargar proyecto {project_dir.name}: {e}")
                continue

        # Ordenar por fecha de creación (más reciente primero)
        projects.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return projects

    def create_project(
        self,
        name: str,
        domain: str,
        description: str = ""
    ) -> str:
        """
        Crea un nuevo proyecto con estructura completa.

        Args:
            name: Nombre del proyecto (usado para directorio)
            domain: Dominio principal del proyecto
            description: Descripción opcional del proyecto

        Returns:
            Ruta absoluta al proyecto creado

        Raises:
            ValueError: Si el proyecto ya existe o el nombre es inválido
        """
        # Validar nombre
        if not name or not name.strip():
            raise ValueError("El nombre del proyecto no puede estar vacío")

        # Sanitizar nombre para usar como directorio
        safe_name = "".join(
            c if c.isalnum() or c in ('-', '_') else '_'
            for c in name.lower().strip()
        )

        project_path = self.projects_dir / safe_name

        # Verificar si ya existe
        if project_path.exists():
            raise ValueError(f"El proyecto '{safe_name}' ya existe")

        # Crear estructura de directorios
        project_path.mkdir(parents=True)
        (project_path / "embeddings").mkdir()
        (project_path / "oauth").mkdir()

        # Crear configuración del proyecto
        config = {
            "name": name,
            "safe_name": safe_name,
            "domain": domain,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "schema_version": "1.0.0",
            "settings": {
                "openai_api_key": None,
                "gemini_api_key": None,
                "default_embedding_model": "paraphrase-multilingual-MiniLM-L12-v2"
            }
        }

        config_path = project_path / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        # Crear archivo .gitignore para oauth
        gitignore_path = project_path / "oauth" / ".gitignore"
        with open(gitignore_path, 'w') as f:
            f.write("# Ignorar todas las credenciales OAuth\n")
            f.write("*\n")
            f.write("!.gitignore\n")

        # Crear metadatos vacíos para embeddings
        embeddings_meta = {
            "model": config["settings"]["default_embedding_model"],
            "count": 0,
            "created_at": datetime.now().isoformat()
        }
        meta_path = project_path / "embeddings" / "metadata.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(embeddings_meta, f, indent=2)

        # Inicializar base de datos DuckDB
        self._init_database(project_path)

        print(f"✅ Proyecto '{name}' creado en: {project_path}")

        return str(project_path)

    def _init_database(self, project_path: Path):
        """
        Inicializa la base de datos DuckDB con el schema.

        Args:
            project_path: Ruta al directorio del proyecto
        """
        try:
            import duckdb
        except ImportError:
            print("⚠️  DuckDB no instalado. La base de datos se creará cuando se use.")
            return

        db_path = project_path / "database.duckdb"

        # Importar schema desde db_schema.py
        try:
            from shared.db_schema import get_initial_schema
            schema_sql = get_initial_schema()
        except ImportError:
            # Si db_schema.py aún no existe, usar schema básico
            schema_sql = """
            CREATE TABLE IF NOT EXISTS urls (
                id INTEGER PRIMARY KEY,
                url TEXT UNIQUE NOT NULL,
                title TEXT,
                content TEXT,
                scraped_at TIMESTAMP,
                embedding_status TEXT DEFAULT 'pending'
            );

            CREATE TABLE IF NOT EXISTS gsc_positions (
                id INTEGER PRIMARY KEY,
                keyword TEXT NOT NULL,
                url TEXT NOT NULL,
                position INTEGER,
                impressions INTEGER,
                clicks INTEGER,
                ctr REAL,
                date DATE
            );

            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY,
                url TEXT NOT NULL,
                model TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS keyword_families (
                id INTEGER PRIMARY KEY,
                family_name TEXT NOT NULL,
                keywords TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """

        # Crear base de datos y ejecutar schema
        conn = duckdb.connect(str(db_path))
        conn.execute(schema_sql)
        conn.close()

        print(f"✅ Base de datos inicializada: {db_path}")

    def load_project(self, project_name: str) -> Dict[str, Any]:
        """
        Carga la configuración de un proyecto existente.

        Args:
            project_name: Nombre del proyecto a cargar

        Returns:
            Diccionario con la configuración del proyecto

        Raises:
            FileNotFoundError: Si el proyecto no existe
        """
        project_path = self.projects_dir / project_name

        if not project_path.exists():
            raise FileNotFoundError(f"El proyecto '{project_name}' no existe")

        config_path = project_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuración no encontrada para '{project_name}'"
            )

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Añadir ruta absoluta
        config["path"] = str(project_path)
        config["db_path"] = str(project_path / "database.duckdb")

        return config

    def get_last_project(self) -> Optional[str]:
        """
        Obtiene el nombre del último proyecto usado.

        Returns:
            Nombre del último proyecto o None si no hay ninguno
        """
        config = self._load_workspace_config()
        return config.get("last_project")

    def set_last_project(self, project_name: str):
        """
        Guarda el último proyecto usado.

        Args:
            project_name: Nombre del proyecto a guardar
        """
        config = self._load_workspace_config()
        config["last_project"] = project_name
        self._save_workspace_config(config)

    def delete_project(self, project_name: str, confirm: bool = False):
        """
        Elimina un proyecto completamente.

        Args:
            project_name: Nombre del proyecto a eliminar
            confirm: Debe ser True para confirmar la eliminación

        Raises:
            ValueError: Si confirm no es True
            FileNotFoundError: Si el proyecto no existe
        """
        if not confirm:
            raise ValueError(
                "Debe confirmar la eliminación estableciendo confirm=True"
            )

        project_path = self.projects_dir / project_name

        if not project_path.exists():
            raise FileNotFoundError(f"El proyecto '{project_name}' no existe")

        # Eliminar directorio completo
        shutil.rmtree(project_path)

        # Si era el último proyecto, limpiar configuración
        if self.get_last_project() == project_name:
            self.set_last_project(None)

        print(f"✅ Proyecto '{project_name}' eliminado")

    def update_project_config(
        self,
        project_name: str,
        updates: Dict[str, Any]
    ):
        """
        Actualiza la configuración de un proyecto.

        Args:
            project_name: Nombre del proyecto
            updates: Diccionario con los campos a actualizar
        """
        config = self.load_project(project_name)
        project_path = Path(config["path"])

        # Actualizar campos
        for key, value in updates.items():
            if key == "settings":
                # Mergear settings en lugar de reemplazar
                config.setdefault("settings", {})
                config["settings"].update(value)
            else:
                config[key] = value

        # Actualizar timestamp
        config["updated_at"] = datetime.now().isoformat()

        # Guardar
        config_path = project_path / "config.json"

        # Remover campos añadidos por load_project
        config.pop("path", None)
        config.pop("db_path", None)

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def get_project_stats(self, project_name: str) -> Dict[str, Any]:
        """
        Obtiene estadísticas de un proyecto.

        Args:
            project_name: Nombre del proyecto

        Returns:
            Diccionario con estadísticas del proyecto
        """
        config = self.load_project(project_name)
        project_path = Path(config["path"])
        db_path = project_path / "database.duckdb"

        stats = {
            "name": config["name"],
            "domain": config["domain"],
            "created_at": config["created_at"],
            "size_mb": 0,
            "db_exists": db_path.exists(),
            "db_size_mb": 0,
            "embeddings_count": 0,
            "urls_count": 0,
            "gsc_records": 0
        }

        # Calcular tamaño total
        if project_path.exists():
            size_bytes = sum(
                f.stat().st_size
                for f in project_path.rglob('*')
                if f.is_file()
            )
            stats["size_mb"] = round(size_bytes / (1024 * 1024), 2)

        # Tamaño de DB
        if db_path.exists():
            stats["db_size_mb"] = round(
                db_path.stat().st_size / (1024 * 1024), 2
            )

        # Contar registros en DuckDB
        if db_path.exists():
            try:
                import duckdb
                conn = duckdb.connect(str(db_path), read_only=True)

                # Contar URLs
                result = conn.execute("SELECT COUNT(*) FROM urls").fetchone()
                if result:
                    stats["urls_count"] = result[0]

                # Contar embeddings
                result = conn.execute(
                    "SELECT COUNT(*) FROM embeddings"
                ).fetchone()
                if result:
                    stats["embeddings_count"] = result[0]

                # Contar registros GSC
                result = conn.execute(
                    "SELECT COUNT(*) FROM gsc_positions"
                ).fetchone()
                if result:
                    stats["gsc_records"] = result[0]

                conn.close()
            except Exception as e:
                print(f"Error al obtener stats de DB: {e}")

        return stats

    def export_project(
        self,
        project_name: str,
        output_path: Optional[str] = None,
        include_oauth: bool = False
    ) -> str:
        """
        Exporta un proyecto completo a un archivo ZIP.

        Args:
            project_name: Nombre del proyecto a exportar
            output_path: Ruta de salida del ZIP (opcional, default: workspace/exports/)
            include_oauth: Si True, incluye credenciales OAuth (NO RECOMENDADO)

        Returns:
            Ruta al archivo ZIP creado

        Raises:
            FileNotFoundError: Si el proyecto no existe
        """
        import zipfile
        from datetime import datetime

        config = self.load_project(project_name)
        project_path = Path(config["path"])

        if not project_path.exists():
            raise FileNotFoundError(f"El proyecto '{project_name}' no existe")

        # Crear directorio de exports si no existe
        if output_path is None:
            exports_dir = self.workspace_root / "exports"
            exports_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = exports_dir / f"{project_name}_{timestamp}.zip"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # Crear ZIP
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Recorrer todos los archivos del proyecto
            for file_path in project_path.rglob('*'):
                if not file_path.is_file():
                    continue

                # Excluir oauth/ por defecto (seguridad)
                relative_path = file_path.relative_to(project_path)

                if not include_oauth and str(relative_path).startswith('oauth'):
                    continue

                # Excluir archivos temporales
                if file_path.suffix in ['.tmp', '.bak', '.lock']:
                    continue

                # Excluir DuckDB WAL files
                if file_path.suffix in ['.duckdb-wal', '.duckdb-shm']:
                    continue

                # Añadir al ZIP
                arcname = f"{project_name}/{relative_path}"
                zipf.write(file_path, arcname=arcname)

        # Calcular tamaño del ZIP
        zip_size_mb = round(output_path.stat().st_size / (1024 * 1024), 2)

        print(f"✅ Proyecto '{project_name}' exportado a: {output_path}")
        print(f"   Tamaño: {zip_size_mb} MB")
        if not include_oauth:
            print(f"   ⚠️  Credenciales OAuth NO incluidas (por seguridad)")

        return str(output_path)

    def import_project(
        self,
        zip_path: str,
        project_name: Optional[str] = None,
        overwrite: bool = False
    ) -> str:
        """
        Importa un proyecto desde un archivo ZIP.

        Args:
            zip_path: Ruta al archivo ZIP a importar
            project_name: Nombre para el proyecto importado (opcional)
            overwrite: Si True, sobrescribe proyecto existente

        Returns:
            Nombre del proyecto importado

        Raises:
            FileNotFoundError: Si el ZIP no existe
            ValueError: Si el ZIP no es válido o el proyecto ya existe
        """
        import zipfile
        import tempfile

        zip_path = Path(zip_path)

        if not zip_path.exists():
            raise FileNotFoundError(f"Archivo ZIP no encontrado: {zip_path}")

        # Validar que sea un ZIP válido
        if not zipfile.is_zipfile(zip_path):
            raise ValueError(f"El archivo no es un ZIP válido: {zip_path}")

        # Extraer a directorio temporal para validar
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(temp_path)

            # Encontrar directorio del proyecto dentro del ZIP
            # Buscar config.json en el primer nivel
            project_dirs = [
                d for d in temp_path.iterdir()
                if d.is_dir() and (d / "config.json").exists()
            ]

            if not project_dirs:
                raise ValueError(
                    "ZIP inválido: no se encontró config.json en el proyecto"
                )

            if len(project_dirs) > 1:
                raise ValueError(
                    "ZIP inválido: múltiples proyectos encontrados"
                )

            extracted_project_dir = project_dirs[0]

            # Cargar config del proyecto
            with open(extracted_project_dir / "config.json", 'r', encoding='utf-8') as f:
                imported_config = json.load(f)

            # Determinar nombre del proyecto
            if project_name is None:
                project_name = imported_config.get("safe_name", extracted_project_dir.name)

            # Sanitizar nombre
            safe_name = "".join(
                c if c.isalnum() or c in ('-', '_') else '_'
                for c in project_name.lower().strip()
            )

            target_path = self.projects_dir / safe_name

            # Verificar si existe
            if target_path.exists():
                if not overwrite:
                    raise ValueError(
                        f"El proyecto '{safe_name}' ya existe. "
                        f"Usa overwrite=True para sobrescribir."
                    )
                # Eliminar proyecto existente
                shutil.rmtree(target_path)

            # Copiar proyecto importado
            shutil.copytree(extracted_project_dir, target_path)

            # Actualizar config con nuevo nombre si cambió
            config_path = target_path / "config.json"
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            config["safe_name"] = safe_name
            config["imported_at"] = datetime.now().isoformat()
            config["imported_from"] = str(zip_path)

            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            # Verificar estructura del proyecto
            required_items = ["config.json", "database.duckdb"]
            missing = []
            for item in required_items:
                if not (target_path / item).exists():
                    missing.append(item)

            if missing and "database.duckdb" in missing:
                # Crear DB vacía si no existe
                self._init_database(target_path)

            # Validar y actualizar schema si es necesario
            self._validate_and_migrate_schema(target_path)

        print(f"✅ Proyecto importado como '{safe_name}'")
        print(f"   Origen: {zip_path}")
        if not (target_path / "oauth").exists() or not list((target_path / "oauth").iterdir()):
            print(f"   ⚠️  Deberás reconfigurar las credenciales OAuth")

        return safe_name

    def _validate_and_migrate_schema(self, project_path: Path):
        """
        Valida el schema de la base de datos y migra si es necesario.

        Args:
            project_path: Ruta al proyecto
        """
        db_path = project_path / "database.duckdb"

        if not db_path.exists():
            return

        try:
            import duckdb

            conn = duckdb.connect(str(db_path))

            # Obtener tablas existentes
            tables = conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main'"
            ).fetchall()

            table_names = [t[0] for t in tables]

            # Tablas requeridas
            required_tables = ['urls', 'gsc_positions', 'embeddings', 'keyword_families']

            # Verificar que existen las tablas básicas
            missing_tables = [t for t in required_tables if t not in table_names]

            if missing_tables:
                print(f"   ⚠️  Tablas faltantes detectadas: {missing_tables}")
                print(f"   Creando tablas faltantes...")

                # Crear tablas faltantes (schema básico)
                from shared.db_schema import get_initial_schema
                schema_sql = get_initial_schema()
                conn.execute(schema_sql)

                print(f"   ✅ Schema actualizado")

            conn.close()

        except Exception as e:
            print(f"   ⚠️  Error al validar schema: {e}")


# Función helper para obtener instancia global
_project_manager_instance = None

def get_project_manager() -> ProjectManager:
    """Obtiene la instancia global del ProjectManager."""
    global _project_manager_instance
    if _project_manager_instance is None:
        _project_manager_instance = ProjectManager()
    return _project_manager_instance
