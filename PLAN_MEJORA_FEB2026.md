# Plan de Mejora — Febrero 2026

## Fase 1 — Bugs Críticos ✅

- [x] `knowledge_graph.py:257` — Añadido `"all-MiniLM-L6-v2"` como argumento a `get_sentence_transformer()`.
- [x] `positions_report.py:19-35` — Corregido `apps.gsc-insights.modules` → `apps.gsc_insights.modules` (4 imports).
- [x] `db_schema.py:188` — Reemplazado `INSERT OR IGNORE INTO` por `INSERT INTO ... ON CONFLICT DO NOTHING`.
- [x] `semantic_tools.py:447-458` — Reescrito bucle de `keyword_relevance`: `top_n` ahora es límite duro, `min_score` filtra adicionalmente.
- [x] `linking_lab.py:67-69,136` — Se recomputa `dataset_ready` después del bloque de upload in-page.

## Fase 2 — Quick Wins ✅

- [x] Eliminados ~28 imports muertos de `streamlit_app.py` (csv, importlib, io, math, re, numpy, pandas, matplotlib, etc.).
- [x] Añadido `@st.cache_data` a `build_similarity_matrix` y `auto_select_cluster_count` en `csv_workflow.py`.
- [x] Precompilados 21 regex en `entity_filters.py` con `re.compile()` a nivel de módulo.
- [x] Añadido `import functools` y `@functools.wraps(func)` al decorador `handle_errors` en `exception_utils.py`.
- [x] Eliminadas copias duplicadas de `semantic_depth.py` en `app_sections/` y `apps/linking_optimizer/modules/`. Canónica: `shared/semantic_depth.py`.
- [x] Creado `shared/config.py` con `DEFAULT_GEMINI_MODEL`, `GEMINI_MODELS` y `ENTITY_PROFILE_PRESETS`. `streamlit_app.py` importa de ahí.
- [x] `INSERT OR IGNORE` ya corregido en Fase 1.

## Fase 3 — Seguridad (pendiente)

- [ ] `project_manager.py:587` — Validar path traversal en `zipf.extractall()` (zip-slip).
- [ ] `oauth_manager.py:83` — Mover clave de cifrado fuera del directorio de datos cifrados.
- [ ] `oauth_manager.py:313-319` — Mostrar warning visible al usuario cuando no hay cifrado disponible.
- [ ] `linking_lab.py:1054-1055` — No exponer `traceback.format_exc()` al usuario. Loguear internamente.
- [ ] `google_kg.py:88` — Considerar pasar API key via header en vez de URL.
- [ ] `oauth_manager.py:26` — Mover `logging.basicConfig()` fuera del import time.

## Fase 4 — Performance ✅ (1 pendiente)

- [x] `csv_workflow.py:148-159` — Reemplazado bucle O(n²) Python por operaciones vectorizadas con `np.triu_indices`, mask numpy y `argsort` compacto.
- [x] `data_orchestrator.py` — Bulk insert de `gsc_positions` via `INSERT INTO ... SELECT FROM insert_df` (DuckDB registra DataFrame directamente).
- [x] `semantic_relations.py:498-537` — Coordenadas 2D (T-SNE/PCA) cacheadas en `st.session_state`. `render_2d_visualization` guarda, Tab 4 reutiliza.
- [x] `entity_filters.py:130-136` — Precompilados regex (hecho en Fase 2).
- [x] `linking_pagerank.py:63` — Reemplazado `np.argsort` por `np.argpartition` + sort parcial para top-K.
- [ ] Estandarizar uso de `embedding_cache.py` en todos los módulos (diferido a Fase 5).

## Fase 5 — Arquitectura y Deduplicación ✅ (2 diferidos)

- [x] Creado `shared/ui_components.py` con `bordered_container()`. Eliminadas 5 copias en csv_workflow, landing_page, gsc/keyword_builder, linking_optimizer/csv_workflow, content_analyzer/keyword_builder.
- [x] Creado `shared/config.py` con constantes centralizadas (`DEFAULT_GEMINI_MODEL`, `GEMINI_MODELS`, `ENTITY_PROFILE_PRESETS`).
- [x] Añadido `@contextmanager _connection()` en `data_orchestrator.py` para garantizar cierre de conexiones DuckDB.
- [ ] Consolidar `semantic_tools.py` duplicado en 3 sub-apps → importar de shared (diferido — requiere testing cross-app extenso).
- [x] Unificadas `get_gemini_api_key_from_context` en `gsc_insights/keyword_builder.py` y `content_analyzer/keyword_builder.py` → importan de `shared/gemini_utils.py`. Corregido default `gemini-2.0-flash-exp` → `gemini-2.5-flash`.
- [x] Extraída lógica de non-linkable mask (repetida 4 veces) a `_get_exclusion_masks()` en `linking_lab.py`.
- [ ] Resolver duplicación `build_entity_payload_from_doc_relations` (4 copias — diferido, requiere análisis de imports cruzados).

## Fase 6 — Nuevas Features (pendiente)

- [ ] Procesamiento incremental de embeddings (detectar URLs ya procesadas).
- [ ] Export unificado PDF/CSV/Excel con sistema centralizado.
- [ ] Histórico de análisis con snapshots en DuckDB + comparación temporal.
- [ ] Batch processing con cola de tareas y `st.status` para datasets >5.000 URLs.
- [ ] Dashboard resumen de proyecto (métricas clave, estado general).
- [ ] API REST (FastAPI) para automatización e integración externa.
- [ ] Detección automática de contenido thin/duplicado via embeddings.
- [ ] Integración con exports de crawlers (Screaming Frog, Sitebulb).

## Errores Silenciosos a Resolver (parcialmente resuelto)

- [x] `knowledge_graph.py:257-258` — Resuelto en Fase 1 (añadido `model_name`).
- [ ] `semantic_relations.py:481-482` — bare `except:` oculta errores en limpieza de temp file.
- [ ] `google_kg.py:94-96` — HTTP 429 hace `continue` sin retry, resultado perdido.
- [x] `data_orchestrator.py:196-210` — Context manager `_connection()` añadido para prevenir leaks (Fase 5).
- [ ] `csv_workflow.py:472-488` — Reset de 17 keys puede limpiar resultados válidos.
- [ ] `content_plan.py:81` — Import relativo `from semantic_tools import ...` falla fuera del directorio correcto.

## Cambios adicionales realizados (fuera del plan original)

- [x] Content Plan Generator: añadido preview antes de procesamiento completo.
- [x] Content Plan Generator: prompt editable directamente (sustituye caja de matices).
- [x] Content Plan Generator: botón de retry solo para filas fallidas.
- [x] Content Plan Generator: selector de modelo de embeddings en pestaña Análisis.
- [x] Corregido error `gemini-pro` 404 → eliminado de listas de modelos, defaults actualizados a `gemini-2.5-flash`.
- [x] Validación de conexión API antes de vista previa con mensajes específicos por tipo de error.
