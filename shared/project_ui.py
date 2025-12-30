"""
Project UI Components
======================

Componentes de interfaz de usuario para gestión de proyectos en Streamlit.

Funcionalidades:
- UI para exportar proyectos
- UI para importar proyectos
- Gestión de backups
- Indicadores de estado

Autor: Embedding Insights
Versión: 1.0.0
"""

import streamlit as st
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def render_export_import_sidebar(project_manager=None):
    """
    Renderiza sección de export/import en el sidebar.

    Args:
        project_manager: Instancia de ProjectManager
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📦 Export / Import")

    if not project_manager:
        st.sidebar.warning("No hay ProjectManager disponible")
        return

    # Verificar si hay proyecto activo
    current_project = st.session_state.get("current_project")

    # Export
    with st.sidebar.expander("📤 Exportar Proyecto", expanded=False):
        if not current_project:
            st.warning("⚠️ Selecciona un proyecto primero")
        else:
            st.markdown(f"**Proyecto:** {current_project}")

            include_oauth = st.checkbox(
                "Incluir credenciales OAuth",
                value=False,
                help="⚠️ NO recomendado por seguridad",
                key="export_include_oauth"
            )

            if include_oauth:
                st.warning("⚠️ Las credenciales OAuth son sensibles")

            if st.button("📤 Exportar", key="btn_export", use_container_width=True):
                try:
                    with st.spinner("Exportando proyecto..."):
                        zip_path = project_manager.export_project(
                            current_project,
                            include_oauth=include_oauth
                        )

                    st.success(f"✅ Proyecto exportado")

                    # Mostrar info del archivo
                    zip_file = Path(zip_path)
                    size_mb = round(zip_file.stat().st_size / (1024 * 1024), 2)

                    st.info(f"📁 {zip_file.name}\n\n💾 {size_mb} MB")

                    # Botón para descargar
                    with open(zip_path, 'rb') as f:
                        st.download_button(
                            label="⬇️ Descargar ZIP",
                            data=f,
                            file_name=zip_file.name,
                            mime="application/zip",
                            use_container_width=True
                        )

                except Exception as e:
                    st.error(f"❌ Error al exportar: {e}")
                    logger.error(f"Error en export: {e}", exc_info=True)

    # Import
    with st.sidebar.expander("📥 Importar Proyecto", expanded=False):
        uploaded_file = st.file_uploader(
            "Selecciona archivo ZIP:",
            type=['zip'],
            key="import_zip_uploader"
        )

        if uploaded_file is not None:
            # Opciones de importación
            import_name = st.text_input(
                "Nombre del proyecto (opcional):",
                placeholder="Déjalo vacío para usar el nombre original",
                key="import_project_name"
            )

            overwrite = st.checkbox(
                "Sobrescribir si existe",
                value=False,
                help="⚠️ Eliminará el proyecto existente",
                key="import_overwrite"
            )

            if overwrite:
                st.warning("⚠️ Se sobrescribirá el proyecto existente")

            if st.button("📥 Importar", key="btn_import", use_container_width=True):
                try:
                    # Guardar archivo temporal
                    import tempfile

                    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name

                    with st.spinner("Importando proyecto..."):
                        imported_name = project_manager.import_project(
                            tmp_path,
                            project_name=import_name if import_name else None,
                            overwrite=overwrite
                        )

                    # Limpiar archivo temporal
                    Path(tmp_path).unlink()

                    st.success(f"✅ Proyecto importado como '{imported_name}'")
                    st.info("💡 Cambia al proyecto en el selector para usarlo")

                    # Opcional: cambiar automáticamente al proyecto importado
                    if st.button("🔄 Cambiar a proyecto importado", key="switch_imported"):
                        st.session_state.current_project = imported_name
                        project_manager.set_last_project(imported_name)
                        st.rerun()

                except Exception as e:
                    st.error(f"❌ Error al importar: {e}")
                    logger.error(f"Error en import: {e}", exc_info=True)


def render_project_backups_ui(project_manager=None):
    """
    Renderiza UI para gestión de backups.

    Args:
        project_manager: Instancia de ProjectManager
    """
    st.header("🗄️ Gestión de Backups")

    if not project_manager:
        st.warning("No hay ProjectManager disponible")
        return

    st.markdown("""
    Los backups son copias de seguridad de tus proyectos.
    Se guardan en `workspace/exports/` como archivos ZIP.
    """)

    # Listar backups existentes
    exports_dir = Path(project_manager.workspace_root) / "exports"

    if not exports_dir.exists() or not list(exports_dir.glob("*.zip")):
        st.info("📭 No hay backups disponibles")
    else:
        st.subheader("📦 Backups Disponibles")

        # Listar ZIPs
        backups = sorted(
            exports_dir.glob("*.zip"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        for backup_file in backups:
            with st.expander(f"📁 {backup_file.name}", expanded=False):
                # Info del archivo
                size_mb = round(backup_file.stat().st_size / (1024 * 1024), 2)
                from datetime import datetime
                mtime = datetime.fromtimestamp(backup_file.stat().st_mtime)

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Tamaño", f"{size_mb} MB")

                with col2:
                    st.metric("Fecha", mtime.strftime("%Y-%m-%d"))

                # Acciones
                col_download, col_delete = st.columns(2)

                with col_download:
                    with open(backup_file, 'rb') as f:
                        st.download_button(
                            label="⬇️ Descargar",
                            data=f,
                            file_name=backup_file.name,
                            mime="application/zip",
                            use_container_width=True,
                            key=f"download_{backup_file.name}"
                        )

                with col_delete:
                    if st.button(
                        "🗑️ Eliminar",
                        use_container_width=True,
                        key=f"delete_{backup_file.name}"
                    ):
                        try:
                            backup_file.unlink()
                            st.success("✅ Backup eliminado")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")


def render_auto_backup_settings(project_manager=None):
    """
    Renderiza configuración de auto-backup.

    Args:
        project_manager: Instancia de ProjectManager
    """
    st.subheader("⚙️ Configuración de Auto-Backup")

    if not project_manager:
        st.warning("No hay ProjectManager disponible")
        return

    # Cargar configuración
    config = project_manager._load_workspace_config()
    settings = config.get("settings", {})

    auto_backup = st.checkbox(
        "Activar auto-backup",
        value=settings.get("auto_backup", True),
        help="Crea backups automáticos al hacer cambios importantes"
    )

    backup_frequency = st.selectbox(
        "Frecuencia de backup:",
        options=["daily", "weekly", "monthly"],
        index=0,
        format_func=lambda x: {
            "daily": "Diario",
            "weekly": "Semanal",
            "monthly": "Mensual"
        }[x]
    )

    max_backups = st.number_input(
        "Máximo de backups a mantener:",
        min_value=1,
        max_value=100,
        value=settings.get("max_backups", 10),
        help="Backups más antiguos se eliminarán automáticamente"
    )

    if st.button("💾 Guardar Configuración", use_container_width=True):
        # Actualizar configuración
        settings["auto_backup"] = auto_backup
        settings["backup_frequency"] = backup_frequency
        settings["max_backups"] = max_backups

        config["settings"] = settings
        project_manager._save_workspace_config(config)

        st.success("✅ Configuración guardada")


def render_project_info_card(project_config: dict, project_manager=None):
    """
    Renderiza tarjeta con información del proyecto.

    Args:
        project_config: Configuración del proyecto
        project_manager: Instancia de ProjectManager
    """
    st.markdown("### 📊 Información del Proyecto")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Nombre:** {project_config.get('name', 'N/A')}")
        st.markdown(f"**Dominio:** {project_config.get('domain', 'N/A')}")
        st.markdown(f"**Schema:** v{project_config.get('schema_version', '1.0.0')}")

    with col2:
        from datetime import datetime
        created = project_config.get('created_at', '')
        if created:
            created_dt = datetime.fromisoformat(created)
            st.markdown(f"**Creado:** {created_dt.strftime('%Y-%m-%d')}")

        updated = project_config.get('updated_at', '')
        if updated:
            updated_dt = datetime.fromisoformat(updated)
            st.markdown(f"**Actualizado:** {updated_dt.strftime('%Y-%m-%d')}")

    # Descripción
    description = project_config.get('description', '')
    if description:
        st.markdown(f"**Descripción:**\n\n{description}")

    # Estadísticas si está disponible
    if project_manager:
        with st.expander("📈 Estadísticas Detalladas", expanded=False):
            try:
                stats = project_manager.get_project_stats(
                    project_config.get('safe_name', '')
                )

                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

                with col_stat1:
                    st.metric("URLs", stats.get('urls_count', 0))

                with col_stat2:
                    st.metric("Embeddings", stats.get('embeddings_count', 0))

                with col_stat3:
                    st.metric("GSC Records", stats.get('gsc_records', 0))

                with col_stat4:
                    st.metric("DB Size", f"{stats.get('db_size_mb', 0)} MB")

            except Exception as e:
                st.error(f"Error al cargar estadísticas: {e}")


def create_project_backup_on_change(
    project_name: str,
    project_manager,
    action: str = "manual"
):
    """
    Crea un backup automático del proyecto.

    Args:
        project_name: Nombre del proyecto
        project_manager: Instancia de ProjectManager
        action: Tipo de acción que triggerea el backup
    """
    try:
        # Verificar si auto-backup está activado
        config = project_manager._load_workspace_config()
        if not config.get("settings", {}).get("auto_backup", True):
            return

        # Crear backup
        from datetime import datetime

        exports_dir = Path(project_manager.workspace_root) / "exports"
        exports_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = exports_dir / f"{project_name}_auto_{action}_{timestamp}.zip"

        project_manager.export_project(
            project_name,
            output_path=str(output_path),
            include_oauth=False
        )

        # Limpiar backups antiguos
        _cleanup_old_backups(project_manager)

        logger.info(f"Auto-backup creado: {output_path}")

    except Exception as e:
        logger.error(f"Error en auto-backup: {e}", exc_info=True)


def _cleanup_old_backups(project_manager):
    """
    Elimina backups antiguos según configuración.

    Args:
        project_manager: Instancia de ProjectManager
    """
    try:
        config = project_manager._load_workspace_config()
        max_backups = config.get("settings", {}).get("max_backups", 10)

        exports_dir = Path(project_manager.workspace_root) / "exports"

        if not exports_dir.exists():
            return

        # Listar backups ordenados por fecha
        backups = sorted(
            exports_dir.glob("*.zip"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        # Eliminar backups excedentes
        for backup in backups[max_backups:]:
            backup.unlink()
            logger.info(f"Backup antiguo eliminado: {backup.name}")

    except Exception as e:
        logger.error(f"Error al limpiar backups: {e}", exc_info=True)


if __name__ == "__main__":
    print("Project UI Components")
    print("=" * 50)
    print("Este módulo contiene componentes de UI para Streamlit")
    print("Ejecuta desde una app Streamlit para ver los componentes")
