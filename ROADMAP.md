# 🗺️ Roadmap: Sistema de Gestión de Proyectos

## Visión General

Transformar las 3 aplicaciones actuales en un sistema multi-proyecto con gestión de credenciales, persistencia de datos y sincronización automática.

**Objetivo:** Permitir gestionar múltiples clientes/proyectos de forma independiente con sus propias credenciales OAuth, datos y configuraciones.

---

## 📋 Fase 1: Foundation (Fundación)
**Duración estimada:** 2-3 días
**Prioridad:** CRÍTICA
**Estado:** 🔄 En progreso

### Objetivos
- Crear estructura base de workspace y proyectos
- Implementar ProjectManager para gestión de proyectos
- Migrar App 3 (GSC Insights) a DuckDB
- Guardar último proyecto usado

### Tareas

#### 1.1 Estructura de Workspace
- [x] Crear `workspace/` en raíz del proyecto
- [x] Crear `workspace/projects/` para almacenar proyectos
- [x] Crear `workspace/.workspace_config.json` para configuración global
```json
{
  "last_project": "proyecto-ejemplo",
  "version": "1.0.0",
  "created_at": "2025-12-29"
}
```

#### 1.2 ProjectManager
- [ ] Crear `shared/project_manager.py`
- [ ] Implementar clase `ProjectManager` con métodos:
  - `list_projects() -> List[dict]` - Listar todos los proyectos
  - `create_project(name: str, domain: str) -> str` - Crear nuevo proyecto
  - `load_project(project_name: str) -> dict` - Cargar proyecto existente
  - `get_last_project() -> Optional[str]` - Obtener último proyecto usado
  - `set_last_project(project_name: str)` - Guardar último proyecto
  - `delete_project(project_name: str)` - Eliminar proyecto

#### 1.3 Estructura de Proyecto Individual
Cada proyecto tendrá:
```
workspace/projects/proyecto-ejemplo/
├── config.json              # Configuración del proyecto
├── database.duckdb          # Base de datos DuckDB
├── embeddings/              # Caché de embeddings
│   ├── urls.faiss           # Índice FAISS
│   └── metadata.json        # Metadatos de embeddings
└── oauth/                   # Credenciales OAuth (gitignored)
    ├── gsc_token.json
    └── analytics_token.json
```

#### 1.4 Schema DuckDB Inicial
- [ ] Crear `shared/db_schema.py` con esquema inicial:
```sql
-- Tabla de URLs del proyecto
CREATE TABLE urls (
    id INTEGER PRIMARY KEY,
    url TEXT UNIQUE NOT NULL,
    title TEXT,
    content TEXT,
    scraped_at TIMESTAMP,
    embedding_status TEXT DEFAULT 'pending'
);

-- Tabla de posiciones GSC
CREATE TABLE gsc_positions (
    id INTEGER PRIMARY KEY,
    keyword TEXT NOT NULL,
    url TEXT NOT NULL,
    position INTEGER,
    impressions INTEGER,
    clicks INTEGER,
    ctr REAL,
    date DATE,
    FOREIGN KEY (url) REFERENCES urls(url)
);

-- Tabla de embeddings
CREATE TABLE embeddings (
    id INTEGER PRIMARY KEY,
    url TEXT NOT NULL,
    model TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (url) REFERENCES urls(url)
);

-- Tabla de familias de keywords
CREATE TABLE keyword_families (
    id INTEGER PRIMARY KEY,
    family_name TEXT NOT NULL,
    keywords TEXT NOT NULL,  -- JSON array
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 1.5 UI: Project Selector
- [ ] Añadir selector de proyectos en sidebar de App 3 (GSC Insights)
- [ ] Componente para crear nuevo proyecto
- [ ] Mostrar proyecto actual en header
- [ ] Guardar último proyecto en session_state

#### 1.6 Migración App 3 a DuckDB
- [ ] Modificar `apps/gsc-insights/modules/positions_report.py`:
  - Reemplazar carga de CSV por queries DuckDB
  - Implementar `save_gsc_data_to_db(df, project_path)`
  - Implementar `load_gsc_data_from_db(project_path) -> pd.DataFrame`
- [ ] Mantener opción de importar CSV (migra a DuckDB)

---

## 📦 Fase 2: Persistence (Persistencia)
**Duración estimada:** 2-3 días
**Prioridad:** ALTA
**Estado:** ✅ Completada

### Objetivos
- Implementar DataOrchestrator para gestión de datos
- Migrar todas las apps a usar DuckDB
- Implementar caché de embeddings persistente

### Tareas

#### 2.1 DataOrchestrator
- [x] Crear `shared/data_orchestrator.py`
- [x] Implementar clase `DataOrchestrator`:
  - `save_urls(urls: List[str])` - Guardar URLs en DB
  - `get_urls() -> List[dict]` - Obtener URLs del proyecto
  - `save_embeddings(url: str, embedding: np.ndarray, model: str)`
  - `get_embeddings(model: str) -> pd.DataFrame`
  - `save_gsc_data(df: pd.DataFrame)`
  - `get_gsc_data(start_date, end_date) -> pd.DataFrame`
  - `save_keyword_families(families: dict)`
  - `get_keyword_families() -> dict`

#### 2.2 Migración App 1 (Content Analyzer)
- [x] Añadir project selector a Content Analyzer
- [x] Añadir DuckDB a requirements.txt
- [x] Preparar módulos para usar DataOrchestrator

#### 2.3 Migración App 2 (Linking Optimizer)
- [x] Añadir project selector a Linking Optimizer
- [x] Añadir DuckDB a requirements.txt
- [x] Preparar módulos para usar EmbeddingCache

#### 2.4 Caché de Embeddings
- [x] Implementar `shared/embedding_cache.py`
- [x] Guardar embeddings en formato FAISS para búsqueda rápida
- [x] Implementar `get_or_compute_embedding(text, model)`
- [x] Sincronizar FAISS con DuckDB
- [x] Búsqueda de similitud con FAISS
- [x] Fallback a búsqueda lineal sin FAISS

#### 2.5 App 3 (GSC Insights)
- [x] Integración completa con DuckDB
- [x] save_gsc_data_to_db() y load_gsc_data_from_db()
- [x] Auto-save al procesar CSV
- [x] Botón para cargar datos guardados

---

## 🔐 Fase 3: OAuth & Credentials (Credenciales)
**Duración estimada:** 2-3 días
**Prioridad:** MEDIA
**Estado:** ✅ Completada

### Objetivos
- Almacenar credenciales OAuth por proyecto ✅
- Auto-switch de credenciales al cambiar proyecto ✅
- Gestión segura de API keys ✅

### Tareas

#### 3.1 OAuth Storage
- [x] Crear `shared/oauth_manager.py`
- [x] Implementar `OAuthManager`:
  - `save_gsc_credentials(project_name, credentials)`
  - `load_gsc_credentials(project_name) -> Credentials`
  - `save_analytics_credentials(project_name, credentials)`
  - `is_authenticated(project_name, service) -> bool`
  - `save_api_key(service, api_key)` - Guardar API keys encriptadas
  - `load_api_key(service, fallback_env)` - Cargar con fallback
  - `delete_api_key(service)` - Eliminar API key
  - `list_api_keys()` - Listar servicios configurados

#### 3.2 Credential Switching
- [x] Auto-cargar credenciales al cambiar de proyecto
- [x] Actualizar `st.session_state` con credenciales del proyecto
- [x] Indicador visual de estado de autenticación en sidebar
- [x] Expander "🔐 Credenciales" en todas las apps

#### 3.3 API Keys por Proyecto
- [x] Almacenar API keys encriptadas en `oauth/api_keys.encrypted.json`
- [x] UI completa para configurar (`shared/credentials_ui.py`):
  - OpenAI API Key
  - Gemini API Key
  - Anthropic (Claude) API Key
  - Serprobot API Key
- [x] Fallback a variables de entorno si no hay key en proyecto
- [x] Encriptación con Fernet (AES-128)

#### 3.4 Security
- [x] Añadir `workspace/projects/*/oauth/` a `.gitignore`
- [x] Añadir `workspace/projects/*/.encryption_key` a `.gitignore`
- [x] Implementar encriptación básica de API keys con Fernet
- [x] Clave de encriptación única por proyecto
- [x] Dependencias añadidas: `cryptography>=42.0.0`, `google-auth>=2.27.0`

---

## 🔄 Fase 4: Auto-Sync & Export (Sincronización y Exportación)
**Duración estimada:** 1-2 días
**Prioridad:** BAJA
**Estado:** ✅ Completada (Core Features)

### Objetivos
- ~~Auto-save de datos en DuckDB~~ (Implícito en DataOrchestrator) ✅
- Exportar/importar proyectos completos ✅
- ~~Sincronización de estado entre sesiones~~ (Future enhancement)

### Tareas

#### 4.1 StateManager
- [~] ~~Crear `shared/state_manager.py`~~ (No necesario - DuckDB auto-save)
- [x] Auto-save implícito en DataOrchestrator
- [~] ~~Detectar cambios en DataFrames y guardar automáticamente~~ (Future)
- [~] ~~Indicador de "Guardando..." en UI~~ (Future)

#### 4.2 Project Export
- [x] Implementar `ProjectManager.export_project(project_name, output_path)`
- [x] Crear archivo ZIP con:
  - `database.duckdb` ✅
  - `embeddings/` ✅
  - `config.json` ✅
  - EXCLUIR `oauth/` por seguridad ✅
  - Excluir temporales (.duckdb-wal, .duckdb-shm) ✅
- [x] UI para exportar proyecto desde sidebar (en las 3 apps) ✅
- [x] Botón de descarga directa del ZIP ✅
- [x] Info de tamaño y fecha del export ✅

#### 4.3 Project Import
- [x] Implementar `ProjectManager.import_project(zip_path)`
- [x] Validar estructura del ZIP ✅
- [x] Migrar schema si es necesario con `_validate_and_migrate_schema()` ✅
- [x] UI para importar proyecto (en las 3 apps) ✅
- [x] File uploader en sidebar ✅
- [x] Opción de sobrescribir proyecto existente ✅
- [x] Auto-switch al proyecto importado ✅

#### 4.4 Session Recovery
- [~] ~~Guardar estado de tabs activos en `session_state.json`~~ (Future enhancement)
- [x] Restaurar último proyecto usado (ya implementado en Fase 1) ✅
- [~] ~~Recuperar uploads en progreso~~ (Not feasible with Streamlit)

---

## 🎯 Consideraciones Técnicas

### Database Locks (DuckDB)
**Problema:** DuckDB no soporta múltiples escritores simultáneos
**Solución:**
- Usar un solo proceso de escritura por proyecto
- Implementar cola de escritura si es necesario
- Advertir al usuario si intenta abrir el mismo proyecto en 2 apps

### Schema Migrations
**Problema:** Proyectos antiguos pueden tener schema desactualizado
**Solución:**
- Guardar `schema_version` en `config.json`
- Implementar migraciones incrementales en `shared/migrations/`
- Backup automático antes de upgrade

### Embedding Cache
**Problema:** Embeddings ocupan mucho espacio
**Solución:**
- Usar FAISS para compresión (PQ, IVF)
- Implementar límite de tamaño de caché
- Permitir borrar caché antiguo

### Portabilidad
**Objetivo:** Proyectos portables entre máquinas
**Solución:**
- Usar rutas relativas dentro del proyecto
- Excluir credenciales OAuth del export
- Documentar proceso de re-autenticación

---

## 📊 Métricas de Éxito

### Fase 1 ✅
- [x] Estructura de workspace creada
- [x] ProjectManager funcional con CRUD completo
- [x] App 3 usando DuckDB en lugar de CSV
- [x] Selector de proyectos en UI (en las 3 apps)

### Fase 2 ✅
- [x] 3 apps guardando datos en DuckDB
- [x] Caché de embeddings persistente (híbrido DuckDB + FAISS)
- [x] Migraciones de schema automáticas

### Fase 3 ✅
- [x] OAuth storage implementado (preparado para GSC/Analytics)
- [x] Auto-switch de credenciales al cambiar proyecto
- [x] API keys encriptadas con Fernet

### Fase 4 ✅
- [x] Auto-save implícito en DataOrchestrator
- [x] Export/import de proyectos completo con validación
- [x] Session recovery básico (último proyecto usado)

---

## 🚀 Quick Start (Después de Fase 1)

```bash
# 1. Crear nuevo proyecto
cd apps/gsc-insights
streamlit run app.py

# 2. En sidebar: "Crear Nuevo Proyecto"
# - Nombre: mi-cliente
# - Dominio: miclientedominio.com

# 3. Importar datos CSV
# - Se guardarán automáticamente en DuckDB

# 4. Cambiar de proyecto
# - Selector en sidebar
# - Datos y credenciales se cargan automáticamente
```

---

## 📝 Notas de Implementación

### Prioridades
1. **Fase 1** - Crítica: Base para todo el sistema
2. **Fase 2** - Alta: Persistencia es core feature
3. **Fase 3** - Media: Mejora UX pero no bloquea funcionalidad
4. **Fase 4** - Baja: Nice to have, no esencial

### Testing
- Crear proyecto de test en cada fase
- Verificar migración de schema
- Probar export/import con datos reales
- Validar seguridad de OAuth storage

### Documentación
- Actualizar README.md con nueva estructura
- Documentar schema DuckDB
- Guía de migración para usuarios existentes

---

**Última actualización:** 2025-12-29
**Versión del roadmap:** 1.0.0
