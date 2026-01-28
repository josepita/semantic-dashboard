# Changelog

Todos los cambios notables de este proyecto se documentan en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/),
y este proyecto sigue [Semantic Versioning](https://semver.org/lang/es/).

## [1.0.0] - 2025-01-28

### Added
- Instalador automático (`INSTALAR.bat`) para Windows
- Lanzador simple (`EJECUTAR.bat`) sin necesidad de consola
- Procesamiento por lotes para datasets grandes (+10k URLs)
- Carga de enlaces existentes desde Screaming Frog
- Filtros de páginas no enlazables en todos los algoritmos de linking
- Módulo `linking_batch.py` para gestión de memoria en datasets grandes

### Changed
- Reorganización de estructura de carpetas (eliminados directorios duplicados)
- Actualizado `.gitignore` para excluir datos de usuario

### Fixed
- Limpieza de archivos temporales y no utilizados

---

## Tipos de cambios

- **Added**: Nuevas funcionalidades
- **Changed**: Cambios en funcionalidades existentes
- **Deprecated**: Funcionalidades que serán eliminadas próximamente
- **Removed**: Funcionalidades eliminadas
- **Fixed**: Corrección de errores
- **Security**: Correcciones de vulnerabilidades
