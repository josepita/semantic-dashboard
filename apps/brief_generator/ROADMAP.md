# Brief Generator - Roadmap de Desarrollo

## Estado Actual

**Fecha:** 04/02/2026
**Version:** 1.0.0-beta

---

## Fases Completadas

### Fase 1: Estructura Base
- [x] Crear estructura `apps/brief_generator/`
- [x] Modulo `brief_storage.py` - SQLite para carpetas y briefs
- [x] Modulo `serp_scraper.py` - Scraping Google con BeautifulSoup
- [x] UI basica con Streamlit
- [x] Sistema de carpetas
- [x] Panel de control con estados
- [x] Import desde Excel
- [x] Brief directo (1 keyword)

### Fase 2: Integracion IA
- [x] Modulo `brief_ai.py` - Gemini/OpenAI
- [x] Generacion de titulos SEO (5 propuestas)
- [x] Generacion de meta descriptions (5 propuestas)
- [x] Generacion de estructura HN (H1/H2/H3)
- [x] Selector de proveedor IA en sidebar
- [x] Seleccion de titulo/meta preferido con checkbox

---

## Fases Pendientes

### Fase 3: Exportacion
- [ ] Exportar brief a Word (.docx)
- [ ] Exportar brief a PDF
- [ ] Exportar lista de briefs a Excel
- [ ] Template personalizable para exportacion

**Archivos a crear/modificar:**
```
modules/export_docx.py  (nuevo)
app.py                  (añadir tab/boton exportar)
```

**Dependencias:**
```
python-docx>=0.8.11
```

---

### Fase 4: Mejoras SERP
- [ ] Cachear resultados SERP en SQLite (evitar re-scraping)
- [ ] Proxy rotation para evitar bloqueos
- [ ] Extraer mas datos: featured snippets, videos, imagenes
- [ ] Analisis de contenido de competidores (scrape paginas)

**Archivos a modificar:**
```
modules/serp_scraper.py
modules/brief_storage.py (añadir tabla cache)
```

---

### Fase 5: Volumenes de Keywords
- [ ] Integracion con API de DataForSEO (volumenes reales)
- [ ] Alternativa: estimaciones con Google Trends
- [ ] Mostrar volumen junto a cada keyword
- [ ] Ordenar keywords por volumen

**Archivos a crear:**
```
modules/keyword_volume.py  (nuevo)
```

**Configuracion necesaria:**
```python
# En shared/config.py
DATAFORSEO_LOGIN = ""
DATAFORSEO_PASSWORD = ""
```

---

### Fase 6: Flujo Excel + HNs (Fusion con Content Plan)
- [ ] Tab "Desde Excel" genera HNs automaticamente
- [ ] Reutilizar logica de `content_plan.py` para HNs
- [ ] Proceso batch: Excel -> HNs -> SERP -> Propuestas
- [ ] Progress bar para procesamiento masivo

**Dependencias:**
```
Reutilizar: apps/content_analyzer/modules/content_plan.py
```

---

### Fase 7: UI/UX Avanzada
- [ ] Vista cards (como en las capturas de referencia)
- [ ] Drag & drop para mover briefs entre carpetas
- [ ] Filtros avanzados (fecha, fuente, pais)
- [ ] Busqueda de briefs
- [ ] Dark mode

---

### Fase 8: Automatizacion
- [ ] Programar scraping periodico de keywords
- [ ] Notificaciones cuando cambia el SERP
- [ ] API REST para integracion externa
- [ ] Webhooks

---

## Estructura de Archivos Actual

```
apps/brief_generator/
├── app.py                          # UI principal Streamlit
├── data/
│   └── briefs.db                   # SQLite (auto-generada)
├── modules/
│   ├── __init__.py
│   ├── brief_storage.py            # CRUD carpetas + briefs
│   ├── brief_ai.py                 # Generacion IA (Gemini/OpenAI)
│   └── serp_scraper.py             # Scraping Google
├── start_app.bat                   # Launcher interno
└── ROADMAP.md                      # Este archivo

Launcher - Brief Generator.bat      # En raiz del proyecto
```

---

## Base de Datos (SQLite)

### Tabla: folders
| Campo | Tipo | Descripcion |
|-------|------|-------------|
| id | INTEGER | PK autoincrement |
| name | TEXT | Nombre unico |
| created_at | TIMESTAMP | Fecha creacion |

### Tabla: briefs
| Campo | Tipo | Descripcion |
|-------|------|-------------|
| id | INTEGER | PK autoincrement |
| folder_id | INTEGER | FK a folders |
| keyword | TEXT | Keyword principal |
| country | TEXT | Codigo pais (ES, MX...) |
| source | TEXT | 'excel' o 'direct' |
| status | TEXT | pending/serp_done/keywords_done/completed |
| hn_structure | JSON | Estructura H1/H2/H3 |
| serp_results | JSON | Top 10 SERP |
| keyword_opportunities | JSON | Keywords relacionadas |
| title_proposals | JSON | 5 propuestas titulo |
| meta_proposals | JSON | 5 propuestas meta |
| selected_title | TEXT | Titulo elegido |
| selected_meta | TEXT | Meta elegida |
| created_at | TIMESTAMP | Fecha creacion |
| updated_at | TIMESTAMP | Ultima actualizacion |

---

## Comandos Utiles

```bash
# Arrancar la app
streamlit run apps/brief_generator/app.py --server.port 8503

# O usar el launcher
"Launcher - Brief Generator.bat"

# Test de imports
cd apps/brief_generator
python -c "from modules.brief_storage import init_database; print('OK')"
python -c "from modules.brief_ai import is_llm_configured; print('OK')"
```

---

## Notas Tecnicas

1. **API Keys**: Se leen de `st.session_state` (configuradas en la app principal de EmbeddingDashboard)

2. **Scraping Google**: Usar con moderacion para evitar bloqueos. Implementar delays entre requests.

3. **Prompts IA**: Los prompts estan en `brief_ai.py`. Ajustar segun necesidades.

4. **Persistencia**: SQLite local en `data/briefs.db`. Para produccion considerar PostgreSQL.
