"""
Módulo para el informe de posiciones SEO.

Este módulo gestiona la interfaz de usuario para carga, procesamiento y generación
de informes HTML a partir de datos de rank tracking exportados desde herramientas SEO.

Todas las funciones de análisis, parsing y generación han sido modularizadas en:
- apps/gsc-insights/modules/positions_parsing.py
- apps/gsc-insights/modules/positions_analysis.py
- apps/gsc-insights/modules/positions_payload.py
- apps/gsc-insights/modules/positions_reports.py
"""

from typing import List, Tuple

import pandas as pd
import streamlit as st

try:
    import google.generativeai as genai
except ModuleNotFoundError:
    genai = None

# Import módulos especializados
from apps.gsc_insights.modules.positions_parsing import (
    ParseResult,
    normalize_domain,
    parse_position_tracking_csv,
    parse_search_volume_file,
)
from apps.gsc_insights.modules.positions_analysis import (
    assign_keyword_families,
    summarize_positions_overview,
)
from apps.gsc_insights.modules.positions_payload import (
    build_family_payload,
    build_competitive_family_payload,
)
from apps.gsc_insights.modules.positions_reports import (
    generate_competitive_html_report,
    generate_position_report_html,
)

# Import otras dependencias internas
from app_sections.keyword_builder import group_keywords_with_semantic_builder
from app_sections.semantic_tools import download_dataframe_button

# Constantes para configuración de gráficos
POSITION_CHART_PRESETS: List[Tuple[str, str, str]] = [
    (
        "heatmap",
        "Heatmap por familia que compare la presencia relativa de cada dominio en las posiciones 1-10.",
        "🗺️",
    ),
    (
        "competitors",
        "Grafico de barras con la frecuencia total de los principales competidores en el Top 10 para todas las keywords.",
        "📊",
    ),
    (
        "radar",
        "Grafico radar que muestre la posicion media del dominio de la marca frente a competidores por familia.",
        "🎯",
    ),
    (
        "trendline",
        "Linea temporal con la evolucion de la posicion media por familia (si el CSV incluye fechas).",
        "📈",
    ),
    (
        "stacked",
        "Barras apiladas por familia indicando que dominio ocupa cada posicion del 1 al 5.",
        "📉",
    ),
]
DEFAULT_CHART_KEYS = ["heatmap", "competitors", "radar"]

def render_positions_report() -> None:
    """
    Renderiza la sección de informe de posiciones SEO en Streamlit.

    Permite cargar un CSV de rank tracking, procesarlo, agrupar keywords
    en familias (manual o automáticamente con Gemini), y generar un informe
    HTML profesional con visualizaciones y análisis.
    """
    from app_sections.landing_page import get_gemini_api_key_from_context, get_gemini_model_from_context

    st.subheader("📊 Informe de posiciones SEO")
    st.caption(
        "Procesa un CSV exportado de tu herramienta de rank tracking, añade volumen de búsqueda y genera informes HTML con Gemini."
    )

    st.session_state.setdefault("positions_raw_df", None)
    st.session_state.setdefault("positions_volume_df", None)
    st.session_state.setdefault("positions_report_html", None)
    st.session_state.setdefault("positions_competitors", [])
    st.session_state.setdefault("positions_semantic_groups", None)
    st.session_state.setdefault("positions_semantic_groups_raw", None)
    st.session_state.setdefault("positions_semantic_language", "es")
    st.session_state.setdefault("positions_semantic_country", "Spain")
    st.session_state.setdefault("positions_semantic_niche", "Proyecto SEO")
    st.session_state.setdefault("positions_auto_classify", True)
    if "positions_gemini_key" not in st.session_state or not st.session_state["positions_gemini_key"]:
        st.session_state["positions_gemini_key"] = get_gemini_api_key_from_context()
    if "positions_gemini_model" not in st.session_state or not st.session_state["positions_gemini_model"]:
        st.session_state["positions_gemini_model"] = get_gemini_model_from_context()

    st.markdown("### 📥 Carga de archivos")

    col_upload1, col_upload2 = st.columns(2)
    with col_upload1:
        st.markdown("**CSV/Excel de posiciones** (requerido)")
        uploaded_csv = st.file_uploader(
            "Sube tu archivo",
            type=["csv", "xlsx", "xls"],
            key="positions_csv_uploader",
            help="Formato SERP con columnas: Keyword, Position 1, Position 2, ..., Position 10"
        )

    with col_upload2:
        st.markdown("**Excel con volumen de búsqueda** (opcional)")
        uploaded_volume = st.file_uploader(
            "Sube tu archivo",
            type=["xlsx", "xls", "csv"],
            key="positions_volume_uploader",
            help="Puede ser cualquier formato - seleccionarás las columnas después"
        )

    # --- Selector de columnas para CSV de posiciones ---
    col_mapping = None
    _detected_se_ranking_format = None
    if uploaded_csv:
        from apps.gsc_insights.modules.positions_parsing import _read_file_lines, _detect_export_format

        # Detectar formato SE Ranking antes de mostrar el selector de columnas
        try:
            _preview_lines = _read_file_lines(uploaded_csv)
            _detected_se_ranking_format = _detect_export_format(_preview_lines)
            uploaded_csv.seek(0)
        except Exception:
            _detected_se_ranking_format = "unknown"

        if _detected_se_ranking_format in ("brief", "full", "serp_multi"):
            _format_labels = {
                "brief": "Brief Export (una fila por keyword con volumen y competidores)",
                "full": "Full Export (historial por keyword con volumen y competidores)",
                "serp_multi": "SERP Export (historial de posiciones 1-10 por keyword)",
            }
            st.success(
                f"📋 Formato detectado: **{_format_labels.get(_detected_se_ranking_format, _detected_se_ranking_format)}**\n\n"
                f"El archivo se procesará automáticamente al pulsar **Procesar CSV**. No necesitas configurar columnas."
            )
            col_mapping = None  # No se necesita mapeo manual
        else:
            st.markdown("#### 🔧 Configurar columnas del CSV de posiciones")
            try:
                uploaded_csv.seek(0)
                csv_ext = uploaded_csv.name.split('.')[-1].lower()
                if csv_ext in ['xlsx', 'xls']:
                    csv_preview = pd.read_excel(uploaded_csv, engine='openpyxl' if csv_ext == 'xlsx' else None, nrows=5)
                else:
                    uploaded_csv.seek(0)
                    csv_preview = None
                    for _delim in [',', ';', '\t', '|']:
                        for _enc in ['utf-8', 'latin-1', 'cp1252']:
                            try:
                                uploaded_csv.seek(0)
                                csv_preview = pd.read_csv(uploaded_csv, delimiter=_delim, encoding=_enc, nrows=5, on_bad_lines='skip')
                                if len(csv_preview.columns) >= 2:
                                    break
                                csv_preview = None
                            except Exception:
                                csv_preview = None
                        if csv_preview is not None and len(csv_preview.columns) >= 2:
                            break
                    if csv_preview is None:
                        uploaded_csv.seek(0)
                        csv_preview = pd.read_csv(uploaded_csv, sep=None, engine='python', nrows=5, on_bad_lines='skip')
                uploaded_csv.seek(0)

                all_cols = list(csv_preview.columns)
                no_selection = "— No usar —"
                options_with_none = [no_selection] + all_cols

                # Auto-detectar mejores candidatos
                def _guess(candidates, cols):
                    for c in cols:
                        cl = str(c).lower().strip()
                        for term in candidates:
                            if term in cl:
                                return cols.index(c)
                    return 0

                kw_terms = ["keyword", "query", "palabra clave", "consulta"]
                pos_terms = ["position", "rank", "posicion", "ranking"]
                url_terms = ["url", "landing page", "página", "pagina"]
                domain_terms = ["domain", "dominio", "site"]

                # Detectar si es formato SERP (Position 1, Position 2...)
                import re as _re
                serp_cols = [c for c in all_cols if _re.match(r'^Position\s+\d+$', str(c).strip(), _re.IGNORECASE)]
                is_serp_format = len(serp_cols) >= 2

                if is_serp_format:
                    st.info(f"📋 Formato SERP detectado ({len(serp_cols)} columnas de posición). Solo necesitas indicar la columna de Keyword.")
                    col_m1, = st.columns(1)
                    with col_m1:
                        kw_idx = _guess(kw_terms, all_cols)
                        sel_keyword = st.selectbox("Columna de Keyword", options=all_cols, index=kw_idx, key="pos_map_keyword")
                    col_mapping = {"keyword": sel_keyword}
                else:
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    with col_m1:
                        kw_idx = _guess(kw_terms, all_cols)
                        sel_keyword = st.selectbox("Columna de Keyword", options=all_cols, index=kw_idx, key="pos_map_keyword")
                    with col_m2:
                        pos_idx = _guess(pos_terms, all_cols)
                        sel_position = st.selectbox("Columna de Position", options=all_cols, index=pos_idx, key="pos_map_position")
                    with col_m3:
                        url_idx_guess = _guess(url_terms, all_cols)
                        sel_url = st.selectbox("Columna de URL", options=options_with_none, index=url_idx_guess + 1 if url_idx_guess > 0 else 0, key="pos_map_url")
                    with col_m4:
                        dom_idx_guess = _guess(domain_terms, all_cols)
                        sel_domain = st.selectbox("Columna de Dominio", options=options_with_none, index=dom_idx_guess + 1 if dom_idx_guess > 0 else 0, key="pos_map_domain")

                    col_mapping = {"keyword": sel_keyword, "position": sel_position}
                    if sel_url != no_selection:
                        col_mapping["url"] = sel_url
                    if sel_domain != no_selection:
                        col_mapping["domain"] = sel_domain

                with st.expander("👁️ Vista previa del CSV de posiciones"):
                    st.dataframe(csv_preview.head(5), use_container_width=True)

            except Exception as e:
                st.warning(f"No se pudo leer el preview del CSV: {e}")
                col_mapping = None

    # Si se subió archivo de volumen, permitir selección manual de columnas
    if uploaded_volume:
        st.markdown("#### 🔧 Configurar archivo de volumen")
        try:
            uploaded_volume.seek(0)
            file_ext = uploaded_volume.name.split('.')[-1].lower()
            if file_ext in ['xlsx', 'xls']:
                preview_df = pd.read_excel(uploaded_volume, engine='openpyxl' if file_ext == 'xlsx' else None, nrows=5)
            else:
                preview_df = pd.read_csv(uploaded_volume, encoding='utf-8-sig', nrows=5)
            uploaded_volume.seek(0)

            col_sel1, col_sel2 = st.columns(2)
            with col_sel1:
                keyword_column = st.selectbox(
                    "Columna de Keywords",
                    options=list(preview_df.columns),
                    key="volume_keyword_col",
                    help="Selecciona la columna que contiene las keywords"
                )
            with col_sel2:
                volume_column = st.selectbox(
                    "Columna de Volumen",
                    options=list(preview_df.columns),
                    key="volume_volume_col",
                    help="Selecciona la columna que contiene el volumen de búsqueda"
                )

            # Guardar selecciones en session_state
            st.session_state["selected_keyword_col"] = keyword_column
            st.session_state["selected_volume_col"] = volume_column

            # Mostrar preview
            with st.expander("👁️ Vista previa del archivo de volumen"):
                st.dataframe(preview_df[[keyword_column, volume_column]].head(5))
        except Exception as e:
            st.warning(f"No se pudo leer el archivo para preview: {e}")
    else:
        # Limpiar selecciones manuales si no hay archivo de volumen
        if "selected_keyword_col" in st.session_state:
            del st.session_state["selected_keyword_col"]
        if "selected_volume_col" in st.session_state:
            del st.session_state["selected_volume_col"]

    max_keywords = st.slider("Keywords por familia", min_value=5, max_value=40, value=20, step=5)

    col_a, col_b = st.columns(2)
    with col_a:
        brand_domain = st.text_input(
            "Dominio principal",
            value=st.session_state.get("positions_brand", ""),
            help="Se usará para resaltar la marca en las tablas.",
        )
    with col_b:
        report_title = st.text_input(
            "Título del informe",
            value=st.session_state.get("positions_report_title", "Informe de posiciones orgánicas"),
        )

    competitor_domains_raw = st.text_input(
        "Dominios competidores (separa por coma)",
        value=", ".join(st.session_state.get("positions_competitors", [])),
        help="Añade los dominios raíz que quieres vigilar en el informe.",
    )
    competitor_domains = [
        normalize_domain(domain.strip())
        for domain in competitor_domains_raw.split(",")
        if domain.strip()
    ]
    st.session_state["positions_competitors"] = [domain for domain in competitor_domains if domain]

    families_instructions = st.text_area(
        "Definición de familias (formato: Familia: keyword1, keyword2, *fragmento*)",
        value=st.session_state.get(
            "positions_family_text",
            "Anillos: anillo, alianzas\nPendientes: pendiente, aro\nCollares: collar, colgante, gargantilla",
        ),
        height=140,
        help="Usa comas o punto y coma para separar keywords/patrones. El carácter * permite coincidencias parciales.",
    )

    st.markdown("**Selecciona los gráficos que quieres incluir en el informe**")
    chart_columns = st.columns(3)
    default_chart_keys = st.session_state.get("positions_chart_selection") or DEFAULT_CHART_KEYS
    selected_chart_keys: List[str] = []
    cards_per_row = len(chart_columns)
    for idx, (chart_key, chart_label, chart_icon) in enumerate(POSITION_CHART_PRESETS):
        widget_key = f"positions_chart_{chart_key}"
        col = chart_columns[idx % cards_per_row]
        col.markdown(
            f"""
            <div style='border:2px solid rgba(255,255,255,0.5); border-radius:12px; padding:0.8rem; text-align:center; background-color: rgba(255,255,255,0.05);'>
                <div style='font-size:1.6rem'>{chart_icon}</div>
                <div style='font-size:0.9rem; color: #e0e0e0;'>{chart_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        checked = col.checkbox(
            f"{chart_icon} {chart_key.title()}",
            key=widget_key,
            value=chart_key in default_chart_keys,
            help=chart_label,
        )
        if checked:
            selected_chart_keys.append(chart_key)
    custom_chart_note = st.text_area(
        "Notas adicionales para visualizaciones (opcional)",
        value=st.session_state.get("positions_chart_custom", ""),
        height=80,
        key="positions_chart_custom_input",
    )
    st.session_state["positions_chart_custom"] = custom_chart_note
    st.session_state["positions_chart_selection"] = selected_chart_keys
    chart_descriptions = [desc for key, desc, _ in POSITION_CHART_PRESETS if key in selected_chart_keys]
    if custom_chart_note.strip():
        chart_descriptions.append(custom_chart_note.strip())
    if not chart_descriptions:
        st.info("Selecciona al menos un gráfico o añade una nota personalizada para guiar al informe.")
    chart_notes = "\n".join(f"- {desc}" for desc in chart_descriptions).strip()

    # Configuración de clasificación automática
    with st.expander("⚙️ Configuración de clasificación automática", expanded=False):
        auto_classify = st.checkbox(
            "Clasificar automáticamente al procesar CSV",
            value=st.session_state.get("positions_auto_classify", True),
            help="Usa Gemini para agrupar keywords automáticamente en familias semánticas al subir el CSV.",
        )
        st.session_state["positions_auto_classify"] = auto_classify

        cfg_cols = st.columns(3)
        with cfg_cols[0]:
            semantic_language = st.text_input(
                "Idioma de las keywords",
                value=st.session_state.get("positions_semantic_language", "es"),
                key="positions_semantic_language",
            )
        with cfg_cols[1]:
            semantic_country = st.text_input(
                "País / mercado",
                value=st.session_state.get("positions_semantic_country", "Spain"),
                key="positions_semantic_country",
            )
        with cfg_cols[2]:
            semantic_niche = st.text_input(
                "Nicho o proyecto",
                value=st.session_state.get("positions_semantic_niche", "Proyecto SEO"),
                key="positions_semantic_niche",
            )

        if not auto_classify:
            st.info("💡 Usa las reglas manuales abajo para definir familias, o activa la clasificación automática.")

    col_api1, col_api2 = st.columns(2)
    with col_api1:
        gemini_api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=st.session_state.get("positions_gemini_key", ""),
        )
    with col_api2:
        gemini_model = st.text_input(
            "Modelo Gemini",
            value=st.session_state.get("positions_gemini_model", get_gemini_model_from_context()),
        )
    gemini_available = genai is not None
    if not gemini_available:
        st.warning(
            "Instala la librería `google-generativeai` en tu entorno para habilitar el informe "
            "(pip install google-generativeai)."
        )

    if uploaded_csv and st.button("Procesar CSV", type="primary"):
        try:
            parse_result = parse_position_tracking_csv(uploaded_csv, column_mapping=col_mapping)
        except ValueError as exc:
            st.error(str(exc))
        else:
            # Manejar tanto ParseResult como DataFrame legacy
            if isinstance(parse_result, ParseResult):
                parsed_df = parse_result.df
                auto_volume_df = parse_result.volume_df
                auto_competitors = parse_result.detected_competitors
                auto_brand = parse_result.detected_brand_domain
                detected_format = parse_result.export_format
            else:
                parsed_df = parse_result
                auto_volume_df = None
                auto_competitors = []
                auto_brand = ""
                detected_format = "standard"

            st.session_state["positions_raw_df"] = parsed_df

            # Auto-detectar dominio de marca si vino del CSV
            if auto_brand and not brand_domain:
                st.session_state["positions_brand"] = auto_brand
                brand_domain = auto_brand
                st.info(f"🔍 Dominio detectado del CSV: **{auto_brand}**")
            else:
                st.session_state["positions_brand"] = brand_domain

            # Auto-detectar competidores si vinieron del CSV
            if auto_competitors:
                existing = st.session_state.get("positions_competitors", [])
                merged = list(dict.fromkeys(existing + auto_competitors))
                st.session_state["positions_competitors"] = merged
                competitor_domains[:] = merged
                st.info(f"🔍 Competidores detectados del CSV: {', '.join(auto_competitors)}")

            if detected_format != "standard":
                st.success(f"📋 Formato detectado: **{detected_format}** — {len(parsed_df)} filas procesadas.")

            # Procesar volumen: priorizar el auto-extraído del brief/full
            if auto_volume_df is not None and not auto_volume_df.empty:
                st.session_state["positions_volume_df"] = auto_volume_df
                st.success(f"Se extrajeron {len(parsed_df)} filas de posiciones y {len(auto_volume_df)} registros de volumen del CSV.")
            elif uploaded_volume:
                try:
                    volume_df = parse_search_volume_file(uploaded_volume)
                    st.session_state["positions_volume_df"] = volume_df
                    st.success(f"Se procesaron {len(parsed_df)} keywords y {len(volume_df)} registros de volumen.")
                except ValueError as vol_exc:
                    st.error(f"Error al procesar archivo de volumen: {vol_exc}")
                    st.session_state["positions_volume_df"] = None
            else:
                st.session_state["positions_volume_df"] = None
                if detected_format == "standard":
                    st.success(f"Se procesaron {len(parsed_df)} keywords.")

            # Clasificación automática si está habilitada y hay API key
            if st.session_state.get("positions_auto_classify", True) and gemini_api_key.strip():
                with st.spinner("🤖 Clasificando keywords automáticamente con Gemini..."):
                    try:
                        unique_keywords = parsed_df["Keyword"].dropna().astype(str).unique().tolist()
                        st.info(f"🔍 Enviando {len(unique_keywords)} keywords únicas a Gemini para clasificación...")

                        mapping, raw_groups = group_keywords_with_semantic_builder(
                            api_key=gemini_api_key.strip(),
                            model_name=gemini_model.strip() or get_gemini_model_from_context(),
                            keywords=unique_keywords,
                            language=st.session_state.get("positions_semantic_language", "es"),
                            country=st.session_state.get("positions_semantic_country", "Spain"),
                            niche=st.session_state.get("positions_semantic_niche", "Proyecto SEO"),
                            brand_domain=brand_domain,
                            competitors=st.session_state.get("positions_competitors", []),
                        )
                        st.session_state["positions_semantic_groups"] = mapping
                        st.session_state["positions_semantic_groups_raw"] = raw_groups
                        total_families = len({fam for fam in mapping.values() if fam})
                        keywords_classified = len([k for k in mapping.values() if k and k != "Sin familia"])
                        st.success(f"✅ Se clasificaron {keywords_classified} keywords en {total_families} familias semánticas.")

                        # Debug: mostrar una muestra del mapping
                        if mapping:
                            sample_items = list(mapping.items())[:3]
                            st.caption(f"📋 Ejemplo de clasificación: {dict(sample_items)}")
                    except Exception as exc:
                        st.error(f"❌ Error en clasificación automática: {exc}")
                        import traceback
                        st.code(traceback.format_exc())
                        st.session_state["positions_semantic_groups"] = None

    raw_df = st.session_state.get("positions_raw_df")
    if raw_df is None:
        st.info("Carga un CSV de posiciones para comenzar.")
        return

    # Merge con datos de volumen si están disponibles
    volume_df = st.session_state.get("positions_volume_df")
    if volume_df is not None and not volume_df.empty:
        # Hacer merge manteniendo todas las keywords de posiciones
        raw_df_with_volume = raw_df.merge(
            volume_df,
            on="Keyword",
            how="left"
        )
        # Rellenar valores faltantes con 0
        raw_df_with_volume["SearchVolume"] = raw_df_with_volume["SearchVolume"].fillna(0).astype(int)
    else:
        raw_df_with_volume = raw_df.copy()
        raw_df_with_volume["SearchVolume"] = 0

    # Leer las clasificaciones automáticas DESPUÉS de procesar el CSV
    semantic_grouping_map = st.session_state.get("positions_semantic_groups")
    use_semantic_builder = semantic_grouping_map is not None

    if use_semantic_builder and semantic_grouping_map:
        st.info(f"📊 Aplicando clasificación automática con {len(semantic_grouping_map)} keywords en el mapping")
        enriched_df = raw_df_with_volume.copy()
        enriched_df["Familia"] = (
            enriched_df["Keyword"]
            .astype(str)
            .apply(
                lambda kw: semantic_grouping_map.get(kw)
                or semantic_grouping_map.get(kw.lower())
                or "Sin familia"
            )
        )
        # Mostrar estadísticas de clasificación
        familias_asignadas = enriched_df[enriched_df["Familia"] != "Sin familia"]["Familia"].nunique()
        keywords_clasificadas = (enriched_df["Familia"] != "Sin familia").sum()
        st.caption(f"✅ {keywords_clasificadas} filas clasificadas en {familias_asignadas} familias diferentes")
    else:
        # Usar clasificación manual basada en reglas
        enriched_df = assign_keyword_families(raw_df_with_volume, families_instructions)
    st.session_state["positions_family_text"] = families_instructions
    st.session_state["positions_report_title"] = report_title

    st.write("### Vista previa de datos")
    st.dataframe(enriched_df.head(50), use_container_width=True)
    download_dataframe_button(enriched_df, "tabla_dominios.xlsx", "Descargar tabla procesada (Excel)")

    chart_notes_payload = chart_notes or "El analista definira los graficos adecuados."
    summary = summarize_positions_overview(enriched_df, brand_domain, competitor_domains)
    st.session_state["positions_summary"] = summary
    family_payload = build_family_payload(
        enriched_df,
        brand_domain,
        max_keywords_per_family=max_keywords,
        competitor_domains=competitor_domains,
    )
    st.session_state["positions_payload"] = family_payload

    col_metrics = st.columns(3)
    col_metrics[0].metric("Keywords analizadas", summary.get("total_keywords", 0))
    col_metrics[1].metric(
        "Keywords con la marca en Top10",
        summary.get("brand_keywords_in_top10", 0),
    )
    avg_pos = summary.get("brand_average_position")
    col_metrics[2].metric("Posicion media de la marca", avg_pos if avg_pos is not None else "No disponible")

    # Métricas de volumen si están disponibles
    if "SearchVolume" in enriched_df.columns and enriched_df["SearchVolume"].sum() > 0:
        col_volume = st.columns(3)
        total_volume = int(enriched_df["SearchVolume"].sum())
        col_volume[0].metric("Volumen total de búsqueda", f"{total_volume:,}")

        # Volumen de la marca si hay datos de dominio
        brand_domain_normalized = normalize_domain(brand_domain) if brand_domain else ""
        if "Domain" in enriched_df.columns and brand_domain_normalized:
            brand_keywords = enriched_df[enriched_df["Domain"] == brand_domain_normalized]
            brand_volume = int(brand_keywords["SearchVolume"].sum())
            col_volume[1].metric("Volumen capturado por la marca", f"{brand_volume:,}")

            # Calcular potencial (keywords en top 10 pero no posición 1)
            brand_top10 = brand_keywords[brand_keywords["Position"] <= 10]
            potential_volume = int(brand_top10["SearchVolume"].sum())
            col_volume[2].metric("Volumen en Top 10", f"{potential_volume:,}")

    if competitor_domains:
        st.caption(f"Competidores definidos manualmente: {', '.join(competitor_domains)}")

    with st.expander("Competidores más frecuentes"):
        competitor_counts = summary.get("top_competitors_by_presence", [])
        if competitor_counts:
            competitor_df = pd.DataFrame(competitor_counts, columns=["Dominio", "Frecuencia"])
            st.dataframe(competitor_df, use_container_width=True)
        else:
            st.info("Sin datos de competidores en el Top 10.")

    st.session_state["positions_chart_notes"] = chart_notes_payload
    st.session_state["positions_gemini_key"] = gemini_api_key
    st.session_state["positions_gemini_model"] = gemini_model
    st.session_state["gemini_api_key"] = gemini_api_key.strip()
    st.session_state["gemini_model_name"] = gemini_model.strip() or get_gemini_model_from_context()

    if not family_payload:
        st.warning("No hay datos suficientes para generar el informe. Ajusta las familias o revisa el CSV.")
        return

    if st.button(
        "Generar informe HTML competitivo",
        type="primary",
        disabled=not gemini_available,
    ):
        with st.spinner("⚙️ Generando informe competitivo..."):
            try:
                # Construir payload competitivo con posiciones de todos los dominios
                competitive_payload = build_competitive_family_payload(
                    df=enriched_df,
                    brand_domain=brand_domain or "Sin dominio especificado",
                    competitor_domains=competitor_domains
                )

                if not competitive_payload:
                    st.warning("No hay familias con datos para generar el informe")
                    return

                # Generar HTML competitivo
                html_report = generate_competitive_html_report(
                    report_title=report_title,
                    brand_domain=brand_domain or "Sin dominio especificado",
                    competitive_payload=competitive_payload,
                    overview=summary,
                    competitor_domains=competitor_domains,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"No se pudo generar el informe: {exc}")
                import traceback
                st.code(traceback.format_exc())
            else:
                st.session_state["positions_report_html"] = html_report
                st.success("Informe competitivo generado correctamente.")

    if st.session_state.get("positions_report_html"):
        st.write("### Vista previa del informe")
        components.html(st.session_state["positions_report_html"], height=700, scrolling=True)
        st.download_button(
            label="Descargar informe HTML",
            data=st.session_state["positions_report_html"].encode("utf-8"),
            file_name="informe_posiciones.html",
            mime="text/html",
        )

__all__ = ['render_positions_report']
