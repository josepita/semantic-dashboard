from __future__ import annotations

import pandas as pd
import streamlit as st

# Import from landing page (for Gemini config)
from app_sections.landing_page import (
    get_gemini_api_key_from_context,
    get_gemini_model_from_context,
)
from app_sections.semantic_tools import download_dataframe_button

# Import shared content analysis utilities
from apps.content_analyzer.modules.shared.content_utils import (
    detect_embedding_columns,
    detect_url_columns,
    detect_page_type_columns,
    preprocess_embeddings,
    detect_non_linkable_pages,
    get_linkable_page_stats,
    load_screaming_frog_internal_links,
    get_existing_links_set,
    filter_new_link_recommendations,
)

# Import linking algorithms and utilities
from apps.linking_optimizer.modules import (
    # Algorithms
    semantic_link_recommendations,
    advanced_semantic_linking,
    structural_taxonomy_linking,
    hybrid_semantic_linking,
    # Utilities
    guess_default_type,
    build_entity_payload_from_doc_relations,
    build_linking_reports_payload,
    interpret_linking_reports_with_gemini,
    # Batch processing
    estimate_memory_usage,
    AUTO_BATCH_THRESHOLD,
    DEFAULT_CHUNK_SIZE,
)

# Import GSC integration (optional - may not be installed)
try:
    from shared.gsc_ui import (
        render_gsc_connection_panel,
        render_gsc_site_selector,
        render_gsc_data_loader,
        render_gsc_linking_integration,
        render_opportunity_pages,
        GSC_DATA_KEY,
    )
    from shared.gsc_client import GOOGLE_API_AVAILABLE
    GSC_AVAILABLE = GOOGLE_API_AVAILABLE
except ImportError:
    GSC_AVAILABLE = False

def render_linking_lab() -> None:
    """Pantalla dedicada al laboratorio de enlazado interno."""
    st.subheader("🔗 Laboratorio de enlazado interno")
    st.caption(
        "Este laboratorio combina teoría de PageRank, semántica vectorial y arquitectura de la información para diseñar "
        "estrategias de enlazado interno basadas en relevancia, autoridad y estructura."
    )

    processed_df = st.session_state.get("processed_df")
    url_column = st.session_state.get("url_column")
    dataset_ready = processed_df is not None and url_column is not None
    with st.expander(
        "Cargar dataset directamente para este laboratorio",
        expanded=not dataset_ready,
    ):
        st.caption(
            "Sube un archivo con embeddings (CSV o Excel) para trabajar aquí sin pasar por la Caja 1. "
            "El procesamiento es el mismo: selecciona la columna vectorial y define la columna de URL."
        )
        linking_upload = st.file_uploader(
            "Dataset para enlazado interno",
            type=["csv", "xlsx", "xls"],
            key="linking_lab_dataset_uploader",
            label_visibility="collapsed",
        )
        if linking_upload:
            try:
                if linking_upload.name.lower().endswith(".csv"):
                    lab_raw_df = pd.read_csv(linking_upload)
                else:
                    lab_raw_df = pd.read_excel(linking_upload)
            except Exception as exc:  # noqa: BLE001
                st.error(f"No se pudo leer el archivo: {exc}")
            else:
                st.dataframe(lab_raw_df.head(), use_container_width=True)
                candidate_embedding_cols = detect_embedding_columns(lab_raw_df)
                if not candidate_embedding_cols:
                    st.warning(
                        "No se detectaron columnas de embeddings en el archivo. Asegúrate de incluir una columna con los vectores."
                    )
                else:
                    lab_processed_df = None
                    selected_embedding = st.selectbox(
                        "Columna de embeddings",
                        options=candidate_embedding_cols,
                        key="linking_lab_embedding_column",
                    )
                    try:
                        lab_processed_df, lab_messages = preprocess_embeddings(lab_raw_df, selected_embedding)
                    except ValueError as exc:  # noqa: BLE001
                        st.error(str(exc))
                        lab_processed_df = None
                    else:
                        for msg in lab_messages:
                            st.info(msg)
                    if lab_processed_df is not None:
                        url_candidates = detect_url_columns(lab_processed_df)
                        if not url_candidates:
                            st.error(
                                "No se detectaron columnas de URL tras procesar los embeddings. "
                                "Incluye al menos una columna con URLs absolutas."
                            )
                        else:
                            selected_url = st.selectbox(
                                "Columna de URL",
                                options=url_candidates,
                                key="linking_lab_url_column",
                            )
                            if st.button("Usar dataset en el laboratorio", key="linking_lab_apply_button"):
                                lab_processed_df[selected_url] = lab_processed_df[selected_url].astype(str).str.strip()
                                st.session_state["raw_df"] = lab_raw_df
                                st.session_state["processed_df"] = lab_processed_df
                                st.session_state["url_column"] = selected_url
                                st.success("Dataset cargado correctamente para el laboratorio de enlazado.")
                                processed_df = lab_processed_df
                                url_column = selected_url

    if not dataset_ready:
        st.info("Carga un dataset con embeddings para empezar a trabajar en el laboratorio.")
        return

    st.markdown("---")
    st.caption(f"Dataset activo: **{len(processed_df)}** filas | Columna URL: **{url_column}**")

    # Advertencia para datasets grandes
    if len(processed_df) > AUTO_BATCH_THRESHOLD:
        memory_info = estimate_memory_usage(len(processed_df))
        with st.expander("⚠️ Dataset grande detectado - Ver opciones de rendimiento", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("URLs totales", f"{len(processed_df):,}")
            with col2:
                st.metric("Memoria estimada", f"{memory_info['total_estimated_mb']:.0f} MB")
            with col3:
                st.metric("Procesamiento", "Por lotes")

            st.info(
                "📊 **Modo de procesamiento por lotes activado automáticamente.** "
                "Los algoritmos procesarán las URLs en grupos para evitar problemas de memoria."
            )

            batch_size = st.slider(
                "Tamaño del lote (URLs por iteración)",
                min_value=100,
                max_value=5000,
                value=DEFAULT_CHUNK_SIZE,
                step=100,
                key="global_batch_size",
                help="Lotes más pequeños = menos memoria pero más lento. Lotes más grandes = más rápido pero más memoria."
            )
            st.session_state["linking_batch_size"] = batch_size
    else:
        st.session_state["linking_batch_size"] = None  # No usar batch para datasets pequeños

    # Sección de configuración de enlaces existentes (inlinks)
    with st.expander("⚙️ Cargar enlaces existentes (Screaming Frog / CSV)", expanded=False):
        st.markdown("""
        **Carga tus enlaces internos existentes** para:
        - Evitar recomendar enlaces que ya tienes
        - Mejorar el cálculo de PageRank con datos reales

        **Formatos soportados:**
        - Exportación de Screaming Frog (`internal_all.csv`, `all_inlinks.csv`)
        - Cualquier CSV/Excel con columnas de origen y destino
        """)

        inlinks_upload = st.file_uploader(
            "Archivo de enlaces existentes",
            type=["csv", "xlsx", "xls"],
            key="linking_lab_inlinks_uploader",
            label_visibility="collapsed",
        )
        if inlinks_upload:
            try:
                # Cargar archivo
                if inlinks_upload.name.lower().endswith(".csv"):
                    raw_df = pd.read_csv(inlinks_upload)
                else:
                    raw_df = pd.read_excel(inlinks_upload)

                st.info(f"📄 Archivo cargado: {len(raw_df)} filas, {len(raw_df.columns)} columnas")

                # Mostrar columnas disponibles
                with st.expander("Ver columnas del archivo"):
                    st.write(list(raw_df.columns))

                # Opción de especificar columnas manualmente o auto-detectar
                use_auto_detect = st.checkbox(
                    "Auto-detectar columnas (Screaming Frog)",
                    value=True,
                    key="inlinks_auto_detect"
                )

                source_col_input = None
                target_col_input = None

                if not use_auto_detect:
                    col1, col2 = st.columns(2)
                    with col1:
                        source_col_input = st.selectbox(
                            "Columna de URL origen",
                            options=list(raw_df.columns),
                            key="inlinks_source_col"
                        )
                    with col2:
                        target_col_input = st.selectbox(
                            "Columna de URL destino",
                            options=list(raw_df.columns),
                            key="inlinks_target_col"
                        )

                filter_nofollow = st.checkbox(
                    "Filtrar enlaces nofollow",
                    value=True,
                    key="inlinks_filter_nofollow"
                )

                if st.button("📥 Procesar enlaces", key="process_inlinks"):
                    try:
                        inlinks_df, messages = load_screaming_frog_internal_links(
                            raw_df,
                            source_column=source_col_input,
                            target_column=target_col_input,
                            filter_follow_only=filter_nofollow,
                        )

                        for msg in messages:
                            st.info(msg)

                        if not inlinks_df.empty:
                            # Guardar DataFrame y set de enlaces
                            st.session_state["linking_existing_links_df"] = inlinks_df
                            st.session_state["linking_existing_links_set"] = get_existing_links_set(inlinks_df)

                            # También crear formato de edges para PageRank
                            st.session_state["linking_existing_edges"] = [
                                (row["source_url"], row["target_url"], 1.0)
                                for _, row in inlinks_df.iterrows()
                            ]

                            st.success(f"✅ {len(inlinks_df)} enlaces internos procesados correctamente")
                            st.dataframe(inlinks_df.head(15), use_container_width=True)
                        else:
                            st.warning("No se encontraron enlaces válidos.")

                    except Exception as exc:
                        st.error(f"Error al procesar enlaces: {exc}")

            except Exception as exc:  # noqa: BLE001
                st.error(f"Error al leer el archivo: {exc}")

        # Mostrar estado y opción para limpiar
        if st.session_state.get("linking_existing_links_set"):
            num_links = len(st.session_state["linking_existing_links_set"])
            st.success(f"📊 **{num_links} enlaces existentes en memoria** - se excluirán de las recomendaciones")

            # Estadísticas de enlaces
            if st.session_state.get("linking_existing_links_df") is not None:
                links_df = st.session_state["linking_existing_links_df"]
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("URLs origen únicas", links_df['source_url'].nunique())
                with col2:
                    st.metric("URLs destino únicas", links_df['target_url'].nunique())

            if st.button("🗑️ Limpiar enlaces existentes", key="linking_clear_inlinks"):
                st.session_state["linking_existing_links_df"] = None
                st.session_state["linking_existing_links_set"] = None
                st.session_state["linking_existing_edges"] = None
                st.success("Enlaces existentes eliminados.")
                st.rerun()

    # ========================================================================
    # SECCIÓN: GOOGLE SEARCH CONSOLE
    # ========================================================================
    if GSC_AVAILABLE:
        with st.expander("📊 Conectar Google Search Console (datos de rendimiento)", expanded=False):
            st.markdown("""
            **Conecta tu cuenta de Search Console** para:
            - Obtener datos reales de clics, impresiones, CTR y posición
            - Identificar páginas con alta visibilidad pero bajo CTR (oportunidades)
            - Priorizar enlaces internos hacia páginas que necesitan autoridad
            """)

            # Panel de conexión
            gsc_client = render_gsc_connection_panel()

            if gsc_client:
                # Selector de sitio
                selected_site = render_gsc_site_selector(gsc_client)

                if selected_site:
                    # Cargar datos
                    gsc_df = render_gsc_data_loader(gsc_client, selected_site)

                    if gsc_df is not None and not gsc_df.empty:
                        st.markdown("---")

                        # Integración con embeddings
                        if processed_df is not None and url_column is not None:
                            merged_df = render_gsc_linking_integration(
                                gsc_df=gsc_df,
                                embeddings_df=processed_df,
                                url_column=url_column,
                            )

                            if merged_df is not None:
                                # Actualizar el dataframe procesado con datos de GSC
                                st.session_state["processed_df"] = merged_df
                                st.session_state["gsc_enriched"] = True
                                st.info("✅ Dataset enriquecido con datos de GSC. Las columnas `clicks`, `impressions`, `ctr` y `position` están disponibles.")

                        # Análisis de oportunidades
                        st.markdown("---")
                        opportunities = render_opportunity_pages(gsc_df)

                        if opportunities is not None and not opportunities.empty:
                            # Guardar URLs de oportunidad para priorizar en algoritmos
                            opportunity_urls = set(opportunities['page'].tolist()) if 'page' in opportunities.columns else set()
                            st.session_state["gsc_opportunity_urls"] = opportunity_urls
                            st.info(f"💡 {len(opportunity_urls)} URLs de oportunidad identificadas. Los algoritmos de enlazado las priorizarán automáticamente.")
    else:
        with st.expander("📊 Google Search Console (no disponible)", expanded=False):
            st.warning(
                "Las librerías de Google API no están instaladas. "
                "Para habilitar la conexión con Search Console, ejecuta:"
            )
            st.code("pip install google-api-python-client google-auth-oauthlib")

    st.markdown("---")

    # ========================================================================
    # SECCIÓN: FILTRADO DE PÁGINAS NO ENLAZABLES
    # ========================================================================
    with st.expander("🚫 Filtrar páginas no enlazables (categorías, paginación, etc.)", expanded=False):
        st.caption(
            "Detecta automáticamente páginas que **no deberían recibir enlaces internos**: "
            "categorías, tags, paginación, páginas de búsqueda, listados de tienda, etc."
        )

        type_column_for_filter = st.session_state.get("page_type_column")

        col1, col2 = st.columns([2, 1])
        with col1:
            # Patrones personalizados adicionales
            custom_patterns_input = st.text_input(
                "Patrones de URL adicionales a excluir (regex, separados por coma)",
                placeholder="/mi-categoria/, /ofertas/, /promociones/",
                help="Añade patrones de URL específicos de tu sitio que no deban recibir enlaces"
            )
        with col2:
            # Tipos personalizados adicionales
            custom_types_input = st.text_input(
                "Tipos de página adicionales a excluir",
                placeholder="landing, oferta, promo",
                help="Tipos de página específicos a excluir"
            )

        # Parsear inputs
        custom_patterns = [p.strip() for p in custom_patterns_input.split(',') if p.strip()] if custom_patterns_input else None
        custom_types = [t.strip() for t in custom_types_input.split(',') if t.strip()] if custom_types_input else None

        if st.button("🔍 Analizar páginas no enlazables", key="analyze_non_linkable"):
            with st.spinner("Analizando URLs..."):
                stats = get_linkable_page_stats(processed_df, url_column, type_column_for_filter)

                # Mostrar estadísticas
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("Total páginas", stats['total_pages'])
                with col_stat2:
                    st.metric("✅ Enlazables", stats['linkable_pages'],
                              delta=f"{stats['linkable_percentage']}%")
                with col_stat3:
                    st.metric("🚫 No enlazables", stats['non_linkable_pages'])

                # Mostrar razones de exclusión
                if stats['exclusion_reasons']:
                    st.markdown("**Razones de exclusión detectadas:**")
                    for reason, count in sorted(stats['exclusion_reasons'].items(), key=lambda x: -x[1]):
                        st.write(f"- {reason}: {count} páginas")

                # Guardar en session state para filtrar
                non_linkable_mask = detect_non_linkable_pages(
                    processed_df, url_column, type_column_for_filter, custom_patterns, custom_types
                )
                st.session_state["non_linkable_mask"] = non_linkable_mask

        # Mostrar opción de filtrar si ya se analizó
        if st.session_state.get("non_linkable_mask") is not None:
            non_linkable_mask = st.session_state["non_linkable_mask"]
            non_linkable_count = non_linkable_mask.sum()

            if non_linkable_count > 0:
                st.warning(f"⚠️ Se detectaron **{non_linkable_count}** páginas no enlazables")

                # Mostrar ejemplos
                with st.expander("Ver ejemplos de páginas detectadas"):
                    examples = processed_df[non_linkable_mask][url_column].head(20).tolist()
                    for url in examples:
                        st.text(f"🚫 {url}")

                # Opciones de filtro
                col_filter1, col_filter2 = st.columns(2)
                with col_filter1:
                    exclude_as_target = st.checkbox(
                        "🚫 Excluir como DESTINO de enlaces",
                        value=st.session_state.get("exclude_non_linkable_as_target", True),
                        key="exclude_non_linkable_target_cb",
                        help="Las páginas detectadas no recibirán enlaces"
                    )
                    st.session_state["exclude_non_linkable_as_target"] = exclude_as_target

                with col_filter2:
                    exclude_as_source = st.checkbox(
                        "🚫 Excluir como ORIGEN de enlaces",
                        value=st.session_state.get("exclude_non_linkable_as_source", True),
                        key="exclude_non_linkable_source_cb",
                        help="Las páginas detectadas no generarán enlaces a otras"
                    )
                    st.session_state["exclude_non_linkable_as_source"] = exclude_as_source

                if exclude_as_target or exclude_as_source:
                    msg_parts = []
                    if exclude_as_target:
                        msg_parts.append("destino")
                    if exclude_as_source:
                        msg_parts.append("origen")
                    st.success(f"✅ Filtro activo: {non_linkable_count} páginas excluidas como {' y '.join(msg_parts)}")
            else:
                st.success("✅ No se detectaron páginas no enlazables con los criterios actuales")

    st.markdown("---")

    # Tabs para diferentes modos de enlazado
    tab_basic, tab_advanced, tab_hybrid, tab_structural = st.tabs([
        "🔹 Modo Básico",
        "🔸 Modo Avanzado",
        "💎 Modo Híbrido (CLS)",
        "🏛️ Modo Estructural"
    ])

    # ========================================================================
    # TAB 1: MODO BÁSICO
    # ========================================================================
    with tab_basic:
        st.markdown("### Enlazado semántico básico")
        st.caption(
            "Estrategia simple basada únicamente en similitud semántica. "
            "Ideal para experimentar con diferentes umbrales y prioridades."
        )

        type_candidates = detect_page_type_columns(processed_df)
        if not type_candidates:
            st.warning("No se detectaron columnas de tipo de página. Este modo requiere una columna de tipo.")
        else:
            type_column = st.selectbox(
                "Columna de tipo de página",
                options=type_candidates,
                key="linking_basic_type_column"
            )
            unique_types = sorted(processed_df[type_column].dropna().astype(str).unique())

            col1, col2 = st.columns(2)
            with col1:
                source_types = st.multiselect(
                    "Tipos de página ORIGEN (que generarán enlaces)",
                    options=unique_types,
                    key="linking_basic_source_types"
                )
            with col2:
                primary_targets = st.multiselect(
                    "Tipos objetivo PRIORITARIOS",
                    options=unique_types,
                    default=guess_default_type(unique_types, ["servicio", "product", "landing"]),
                    key="linking_basic_primary_targets"
                )

            secondary_targets = st.multiselect(
                "Tipos objetivo SECUNDARIOS (opcional)",
                options=unique_types,
                key="linking_basic_secondary_targets"
            )

            st.markdown("**Parámetros de enlazado**")
            param_col1, param_col2, param_col3 = st.columns(3)
            with param_col1:
                similarity_threshold = st.slider(
                    "Umbral de similitud",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    key="linking_basic_similarity_threshold"
                )
            with param_col2:
                max_links_total = st.number_input(
                    "Máximo enlaces por página origen",
                    min_value=1,
                    max_value=20,
                    value=3,
                    key="linking_basic_max_links_total"
                )
            with param_col3:
                max_primary = st.number_input(
                    "Máximo a targets prioritarios",
                    min_value=0,
                    max_value=10,
                    value=2,
                    key="linking_basic_max_primary"
                )

            max_secondary = st.number_input(
                "Máximo a targets secundarios",
                min_value=0,
                max_value=10,
                value=1,
                key="linking_basic_max_secondary"
            )

            source_limit = st.number_input(
                "Limitar número de páginas origen a procesar (0 = todas)",
                min_value=0,
                max_value=10000,
                value=0,
                key="linking_basic_source_limit"
            )

            if st.button("🚀 Generar recomendaciones básicas", key="linking_basic_run"):
                if not source_types:
                    st.error("Selecciona al menos un tipo de página origen.")
                elif not primary_targets and not secondary_targets:
                    st.error("Selecciona al menos un tipo de página objetivo (prioritario o secundario).")
                else:
                    # Crear placeholder para progreso
                    progress_placeholder = st.empty()
                    status_placeholder = st.empty()

                    def update_progress(current, total, message):
                        progress_placeholder.progress(current / total if total > 0 else 0)
                        status_placeholder.caption(message)

                    with st.spinner("Calculando recomendaciones semánticas básicas..."):
                        try:
                            # Preparar máscaras de exclusión
                            exclude_source_mask = None
                            exclude_target_mask = None
                            non_linkable = st.session_state.get("non_linkable_mask")

                            if non_linkable is not None:
                                if st.session_state.get("exclude_non_linkable_as_source", False):
                                    exclude_source_mask = non_linkable
                                if st.session_state.get("exclude_non_linkable_as_target", False):
                                    exclude_target_mask = non_linkable

                            # Configuración de batch
                            batch_size = st.session_state.get("linking_batch_size")
                            use_batch = batch_size is not None

                            report_df = semantic_link_recommendations(
                                df=processed_df,
                                url_column=url_column,
                                type_column=type_column,
                                source_types=source_types,
                                primary_target_types=primary_targets,
                                secondary_target_types=secondary_targets if secondary_targets else None,
                                similarity_threshold=similarity_threshold,
                                max_links_per_source=int(max_links_total),
                                max_primary=int(max_primary),
                                max_secondary=int(max_secondary),
                                source_limit=int(source_limit) if source_limit > 0 else None,
                                exclude_as_source_mask=exclude_source_mask,
                                exclude_as_target_mask=exclude_target_mask,
                                use_batch=use_batch,
                                batch_size=batch_size or DEFAULT_CHUNK_SIZE,
                                progress_callback=update_progress if use_batch else None,
                            )

                            # Limpiar placeholders de progreso
                            progress_placeholder.empty()
                            status_placeholder.empty()

                            # Filtrar enlaces existentes si están cargados
                            existing_links = st.session_state.get("linking_existing_links_set")
                            if existing_links and len(report_df) > 0:
                                report_df, excluded_count = filter_new_link_recommendations(
                                    report_df, existing_links, "Origen URL", "Destino Sugerido URL"
                                )
                                if excluded_count > 0:
                                    st.info(f"ℹ️ Se excluyeron {excluded_count} enlaces que ya existen")

                            st.session_state["linking_basic_report"] = report_df
                            st.success(f"✅ {len(report_df)} recomendaciones generadas.")
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"Error al generar recomendaciones: {exc}")

            if st.session_state.get("linking_basic_report") is not None:
                report_df = st.session_state["linking_basic_report"]
                st.dataframe(report_df, use_container_width=True)
                download_dataframe_button(
                    report_df,
                    filename="linking_basic_recommendations.xlsx",
                    label="📥 Descargar recomendaciones básicas"
                )

    # ========================================================================
    # TAB 2: MODO AVANZADO
    # ========================================================================
    with tab_advanced:
        st.markdown("### Enlazado semántico avanzado con silos")
        st.caption(
            "Incorpora detección de silos por estructura de URL y boost de misma categoría. "
            "Detecta páginas huérfanas (money pages sin inlinks)."
        )

        type_candidates_adv = detect_page_type_columns(processed_df)
        if not type_candidates_adv:
            st.warning("No se detectaron columnas de tipo de página. Este modo requiere una columna de tipo.")
        else:
            type_column_adv = st.selectbox(
                "Columna de tipo de página",
                options=type_candidates_adv,
                key="linking_adv_type_column"
            )
            unique_types_adv = sorted(processed_df[type_column_adv].dropna().astype(str).unique())

            col1, col2 = st.columns(2)
            with col1:
                source_types_adv = st.multiselect(
                    "Tipos de página ORIGEN",
                    options=unique_types_adv,
                    key="linking_adv_source_types"
                )
            with col2:
                primary_targets_adv = st.multiselect(
                    "Tipos objetivo PRIORITARIOS (money pages)",
                    options=unique_types_adv,
                    default=guess_default_type(unique_types_adv, ["servicio", "product", "landing"]),
                    key="linking_adv_primary_targets"
                )

            secondary_targets_adv = st.multiselect(
                "Tipos objetivo SECUNDARIOS (opcional)",
                options=unique_types_adv,
                key="linking_adv_secondary_targets"
            )

            st.markdown("**Parámetros de enlazado**")
            param_col1, param_col2, param_col3, param_col4 = st.columns(4)
            with param_col1:
                similarity_threshold_adv = st.slider(
                    "Umbral de similitud",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    key="linking_adv_similarity_threshold"
                )
            with param_col2:
                max_links_total_adv = st.number_input(
                    "Máximo enlaces por origen",
                    min_value=1,
                    max_value=20,
                    value=3,
                    key="linking_adv_max_links_total"
                )
            with param_col3:
                max_primary_adv = st.number_input(
                    "Máximo a prioritarios",
                    min_value=0,
                    max_value=10,
                    value=2,
                    key="linking_adv_max_primary"
                )
            with param_col4:
                silo_boost = st.slider(
                    "Boost mismo silo",
                    min_value=0.0,
                    max_value=0.5,
                    value=0.1,
                    step=0.05,
                    key="linking_adv_silo_boost"
                )

            max_secondary_adv = st.number_input(
                "Máximo a secundarios",
                min_value=0,
                max_value=10,
                value=1,
                key="linking_adv_max_secondary"
            )

            silo_depth = st.number_input(
                "Profundidad de silo (segmentos de URL a considerar)",
                min_value=1,
                max_value=5,
                value=2,
                key="linking_adv_silo_depth"
            )

            source_limit_adv = st.number_input(
                "Limitar número de páginas origen a procesar (0 = todas)",
                min_value=0,
                max_value=10000,
                value=0,
                key="linking_adv_source_limit"
            )

            if st.button("🚀 Generar recomendaciones avanzadas", key="linking_adv_run"):
                if not source_types_adv:
                    st.error("Selecciona al menos un tipo de página origen.")
                elif not primary_targets_adv and not secondary_targets_adv:
                    st.error("Selecciona al menos un tipo de página objetivo.")
                else:
                    with st.spinner("Calculando recomendaciones semánticas avanzadas con silos..."):
                        try:
                            # Preparar máscaras de exclusión
                            exclude_source_mask = None
                            exclude_target_mask = None
                            non_linkable = st.session_state.get("non_linkable_mask")

                            if non_linkable is not None:
                                if st.session_state.get("exclude_non_linkable_as_source", False):
                                    exclude_source_mask = non_linkable
                                if st.session_state.get("exclude_non_linkable_as_target", False):
                                    exclude_target_mask = non_linkable

                            report_df, orphan_urls = advanced_semantic_linking(
                                df=processed_df,
                                url_column=url_column,
                                type_column=type_column_adv,
                                source_types=source_types_adv,
                                primary_target_types=primary_targets_adv,
                                secondary_target_types=secondary_targets_adv if secondary_targets_adv else None,
                                similarity_threshold=similarity_threshold_adv,
                                max_links_per_source=int(max_links_total_adv),
                                max_primary=int(max_primary_adv),
                                max_secondary=int(max_secondary_adv),
                                silo_depth=int(silo_depth),
                                silo_boost=float(silo_boost),
                                source_limit=int(source_limit_adv) if source_limit_adv > 0 else None,
                                exclude_as_source_mask=exclude_source_mask,
                                exclude_as_target_mask=exclude_target_mask,
                            )

                            # Filtrar enlaces existentes si están cargados
                            existing_links = st.session_state.get("linking_existing_links_set")
                            if existing_links and len(report_df) > 0:
                                report_df, excluded_count = filter_new_link_recommendations(
                                    report_df, existing_links, "Origen URL", "Destino URL"
                                )
                                if excluded_count > 0:
                                    st.info(f"ℹ️ Se excluyeron {excluded_count} enlaces que ya existen")

                            st.session_state["linking_adv_report"] = report_df
                            st.session_state["linking_adv_orphans"] = orphan_urls
                            st.success(f"✅ {len(report_df)} recomendaciones generadas.")
                            if orphan_urls:
                                st.warning(f"⚠️ {len(orphan_urls)} páginas prioritarias huérfanas detectadas (sin inlinks).")
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"Error al generar recomendaciones: {exc}")

            if st.session_state.get("linking_adv_report") is not None:
                report_df = st.session_state["linking_adv_report"]
                st.dataframe(report_df, use_container_width=True)
                download_dataframe_button(
                    report_df,
                    filename="linking_advanced_recommendations.xlsx",
                    label="📥 Descargar recomendaciones avanzadas"
                )

                # Mostrar páginas huérfanas
                orphan_urls = st.session_state.get("linking_adv_orphans", [])
                if orphan_urls:
                    with st.expander(f"⚠️ Páginas huérfanas detectadas ({len(orphan_urls)})", expanded=True):
                        st.caption(
                            "Estas páginas prioritarias (money pages) no recibieron ningún enlace interno. "
                            "Considera crear contenido puente o ajustar los parámetros de enlazado."
                        )
                        for orphan_url in orphan_urls:
                            st.markdown(f"- {orphan_url}")

    # ========================================================================
    # TAB 3: MODO HÍBRIDO (CLS)
    # ========================================================================
    with tab_hybrid:
        st.markdown("### Algoritmo híbrido: Composite Link Score (CLS)")
        st.caption(
            "Combina 3 señales: **semántica** (40%), **autoridad PageRank** (35%), **entidades** (25%). "
            "Incorpora decay factor para distribuir mejor la autoridad y evitar concentración de inlinks."
        )

        type_candidates_hybrid = detect_page_type_columns(processed_df)
        if not type_candidates_hybrid:
            st.warning("No se detectaron columnas de tipo de página. Este modo requiere una columna de tipo.")
        else:
            type_column_hybrid = st.selectbox(
                "Columna de tipo de página",
                options=type_candidates_hybrid,
                key="linking_hybrid_type_column"
            )
            unique_types_hybrid = sorted(processed_df[type_column_hybrid].dropna().astype(str).unique())

            # Selección de columna de entidades
            st.markdown("**Columna de entidades (Knowledge Graph)**")
            st.caption(
                "Selecciona la columna que contiene el payload de entidades en formato JSON. "
                "Si no tienes datos de entidades, el algoritmo funcionará solo con semántica + autoridad."
            )
            entity_column_options = ["(Sin entidades - solo semántica + PageRank)"] + list(processed_df.columns)
            entity_column_hybrid = st.selectbox(
                "Columna de entidades",
                options=entity_column_options,
                key="linking_hybrid_entity_column"
            )

            # Opción para cargar relaciones doc-entidad desde archivo
            with st.expander("🔄 Cargar relaciones página-entidad desde archivo", expanded=False):
                st.caption(
                    "Si tienes un archivo CSV/Excel con relaciones página-entidad (ej: desde GSC Knowledge Graph), "
                    "puedes cargarlo aquí. Formato requerido: columnas 'URL', 'QID' o 'Entidad', "
                    "'Prominence documento' o 'Frecuencia documento'."
                )
                doc_relations_upload = st.file_uploader(
                    "Archivo de relaciones página-entidad",
                    type=["csv", "xlsx", "xls"],
                    key="linking_hybrid_doc_relations_uploader",
                    label_visibility="collapsed",
                )
                if doc_relations_upload:
                    try:
                        if doc_relations_upload.name.lower().endswith(".csv"):
                            doc_relations_df = pd.read_csv(doc_relations_upload)
                        else:
                            doc_relations_df = pd.read_excel(doc_relations_upload)

                        # Construir payload de entidades
                        entity_payload = build_entity_payload_from_doc_relations(doc_relations_df)

                        if not entity_payload:
                            st.warning("No se pudieron extraer relaciones página-entidad del archivo.")
                        else:
                            # Añadir columna al DataFrame
                            import json
                            processed_df["EntitiesFromRelations"] = processed_df[url_column].apply(
                                lambda url: json.dumps(entity_payload.get(url, {}))
                            )
                            st.session_state["processed_df"] = processed_df
                            st.success(
                                f"✅ Relaciones página-entidad cargadas. Nueva columna: 'EntitiesFromRelations' "
                                f"({len(entity_payload)} URLs con entidades)."
                            )
                            st.dataframe(doc_relations_df.head(10), use_container_width=True)

                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Error al cargar relaciones: {exc}")

            col1, col2 = st.columns(2)
            with col1:
                source_types_hybrid = st.multiselect(
                    "Tipos de página ORIGEN",
                    options=unique_types_hybrid,
                    key="linking_hybrid_source_types"
                )
            with col2:
                primary_targets_hybrid = st.multiselect(
                    "Tipos objetivo PRIORITARIOS (money pages para PageRank)",
                    options=unique_types_hybrid,
                    default=guess_default_type(unique_types_hybrid, ["servicio", "product", "landing"]),
                    key="linking_hybrid_primary_targets"
                )

            st.markdown("**Parámetros del algoritmo**")
            param_col1, param_col2, param_col3, param_col4 = st.columns(4)
            with param_col1:
                similarity_threshold_hybrid = st.slider(
                    "Umbral de similitud",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    key="linking_hybrid_similarity_threshold"
                )
            with param_col2:
                max_links_total_hybrid = st.number_input(
                    "Máximo enlaces por origen",
                    min_value=1,
                    max_value=20,
                    value=3,
                    key="linking_hybrid_max_links_total"
                )
            with param_col3:
                max_primary_hybrid = st.number_input(
                    "Máximo a prioritarios",
                    min_value=0,
                    max_value=10,
                    value=2,
                    key="linking_hybrid_max_primary"
                )
            with param_col4:
                decay_factor = st.slider(
                    "Decay factor (concentración)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1,
                    step=0.05,
                    key="linking_hybrid_decay_factor",
                    help="Penaliza páginas que ya tienen muchos inlinks para distribuir mejor la autoridad"
                )

            st.markdown("**Pesos del Composite Link Score**")
            st.caption("Ajusta la importancia de cada señal (deben sumar ~1.0)")
            weight_col1, weight_col2, weight_col3 = st.columns(3)
            with weight_col1:
                weight_semantic = st.slider(
                    "Peso semántica",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.4,
                    step=0.05,
                    key="linking_hybrid_weight_semantic"
                )
            with weight_col2:
                weight_authority = st.slider(
                    "Peso autoridad (PageRank)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.35,
                    step=0.05,
                    key="linking_hybrid_weight_authority"
                )
            with weight_col3:
                weight_entity = st.slider(
                    "Peso entidades",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.25,
                    step=0.05,
                    key="linking_hybrid_weight_entity"
                )

            top_k_edges = st.number_input(
                "Top-K aristas semánticas para grafo de PageRank",
                min_value=1,
                max_value=20,
                value=5,
                key="linking_hybrid_top_k_edges",
                help="Número de vecinos más similares para construir el grafo semántico"
            )

            source_limit_hybrid = st.number_input(
                "Limitar número de páginas origen a procesar (0 = todas)",
                min_value=0,
                max_value=10000,
                value=0,
                key="linking_hybrid_source_limit"
            )

            if st.button("🚀 Generar recomendaciones híbridas (CLS)", key="linking_hybrid_run"):
                if not source_types_hybrid:
                    st.error("Selecciona al menos un tipo de página origen.")
                elif not primary_targets_hybrid:
                    st.error("Selecciona al menos un tipo de página objetivo prioritario.")
                else:
                    # Validar columna de entidades
                    effective_entity_column = None
                    if entity_column_hybrid != "(Sin entidades - solo semántica + PageRank)":
                        if entity_column_hybrid not in processed_df.columns:
                            st.error(f"La columna de entidades '{entity_column_hybrid}' no existe en el dataset.")
                        else:
                            effective_entity_column = entity_column_hybrid

                    if effective_entity_column is None:
                        # Crear columna temporal vacía para entidades
                        import json
                        processed_df["_empty_entities"] = processed_df[url_column].apply(lambda x: json.dumps({}))
                        effective_entity_column = "_empty_entities"
                        st.info("ℹ️ Ejecutando sin datos de entidades (solo semántica + PageRank)")

                    with st.spinner("Calculando Composite Link Score (CLS) con PageRank y entidades..."):
                        try:
                            # Preparar máscaras de exclusión
                            exclude_source_mask = None
                            exclude_target_mask = None
                            non_linkable = st.session_state.get("non_linkable_mask")

                            if non_linkable is not None:
                                if st.session_state.get("exclude_non_linkable_as_source", False):
                                    exclude_source_mask = non_linkable
                                if st.session_state.get("exclude_non_linkable_as_target", False):
                                    exclude_target_mask = non_linkable

                            # Obtener enlaces existentes si están configurados
                            existing_edges = st.session_state.get("linking_existing_edges")

                            report_df, orphan_urls, pagerank_scores = hybrid_semantic_linking(
                                df=processed_df,
                                url_column=url_column,
                                type_column=type_column_hybrid,
                                entity_column=effective_entity_column,
                                source_types=source_types_hybrid,
                                primary_target_types=primary_targets_hybrid,
                                similarity_threshold=similarity_threshold_hybrid,
                                max_links_per_source=int(max_links_total_hybrid),
                                max_primary=int(max_primary_hybrid),
                                decay_factor=float(decay_factor),
                                weights={
                                    "semantic": float(weight_semantic),
                                    "authority": float(weight_authority),
                                    "entity_overlap": float(weight_entity),
                                },
                                top_k_edges=int(top_k_edges),
                                source_limit=int(source_limit_hybrid) if source_limit_hybrid > 0 else None,
                                existing_edges=existing_edges,
                                exclude_as_source_mask=exclude_source_mask,
                                exclude_as_target_mask=exclude_target_mask,
                            )

                            # Filtrar enlaces existentes adicionales si están cargados
                            existing_links = st.session_state.get("linking_existing_links_set")
                            if existing_links and len(report_df) > 0:
                                report_df, excluded_count = filter_new_link_recommendations(
                                    report_df, existing_links, "Origen URL", "Destino URL"
                                )
                                if excluded_count > 0:
                                    st.info(f"ℹ️ Se excluyeron {excluded_count} enlaces que ya existen")

                            st.session_state["linking_hybrid_report"] = report_df
                            st.session_state["linking_hybrid_orphans"] = orphan_urls
                            st.session_state["linking_hybrid_pagerank"] = pagerank_scores
                            st.success(f"✅ {len(report_df)} recomendaciones generadas.")
                            if orphan_urls:
                                st.warning(f"⚠️ {len(orphan_urls)} páginas prioritarias huérfanas detectadas.")
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"Error al generar recomendaciones: {exc}")
                            import traceback
                            st.code(traceback.format_exc())

            if st.session_state.get("linking_hybrid_report") is not None:
                report_df = st.session_state["linking_hybrid_report"]
                st.dataframe(report_df, use_container_width=True)
                download_dataframe_button(
                    report_df,
                    filename="linking_hybrid_cls_recommendations.xlsx",
                    label="📥 Descargar recomendaciones híbridas (CLS)"
                )

                # Mostrar páginas huérfanas
                orphan_urls = st.session_state.get("linking_hybrid_orphans", [])
                if orphan_urls:
                    with st.expander(f"⚠️ Páginas huérfanas detectadas ({len(orphan_urls)})", expanded=False):
                        st.caption("Money pages sin inlinks recomendados.")
                        for orphan_url in orphan_urls:
                            st.markdown(f"- {orphan_url}")

                # Mostrar top PageRank scores
                pagerank_scores = st.session_state.get("linking_hybrid_pagerank", {})
                if pagerank_scores:
                    with st.expander("📊 Top 20 páginas por PageRank temático", expanded=False):
                        sorted_pr = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:20]
                        pr_df = pd.DataFrame(sorted_pr, columns=["URL", "PageRank Score"])
                        st.dataframe(pr_df, use_container_width=True)

    # ========================================================================
    # TAB 4: MODO ESTRUCTURAL
    # ========================================================================
    with tab_structural:
        st.markdown("### Enlazado estructural/taxonómico")
        st.caption(
            "Genera enlaces basados en la jerarquía de URLs (breadcrumb, hermanos, destacados). "
            "No requiere embeddings, solo estructura de URLs o columna de jerarquía custom."
        )

        # ====================================================================
        # SECCIÓN DE AYUDA
        # ====================================================================
        with st.expander("📚 **Ayuda: Cómo usar el modo estructural**", expanded=False):
            st.markdown("""
            ### 🏛️ ¿Qué es el enlazado estructural?

            El enlazado estructural crea enlaces basados en la **jerarquía/taxonomía** de tu sitio:
            - **Breadcrumb**: Enlaces de hijos a padres (ej: `/blog/seo/guia-seo` → `/blog/seo/`)
            - **Hermanos**: Enlaces entre páginas del mismo nivel (ej: `/servicios/seo/` ↔ `/servicios/sem/`)
            - **Destacados**: Enlaces de categorías a sus mejores hijos

            ---

            ### 📊 Opciones de jerarquía

            **Opción 1: Extraer de URLs (automático)**
            - El sistema analiza la estructura de tus URLs
            - Ejemplo: `/blog/marketing/guia-email/` → Jerarquía = `blog > marketing`

            **Opción 2: Columna en tu CSV**
            - Si tu CSV ya tiene una columna de categoría/silo
            - Ejemplo: columna "Categoría" con valores como "Blog", "Servicios", "Productos"

            **Opción 3: Cargar Excel de jerarquía** (abajo)
            - Archivo separado con la estructura jerárquica completa
            - Útil cuando la jerarquía no coincide con la URL

            ---

            ### 📝 Ejemplo de Excel de jerarquía

            Tu Excel debe tener estas columnas (la URL es la clave):

            | URL | Nivel1 | Nivel2 | Nivel3 | Padre |
            |-----|--------|--------|--------|-------|
            | https://ejemplo.com/ | Home | | | |
            | https://ejemplo.com/blog/ | Blog | | | https://ejemplo.com/ |
            | https://ejemplo.com/blog/seo/ | Blog | SEO | | https://ejemplo.com/blog/ |
            | https://ejemplo.com/blog/seo/guia-basica/ | Blog | SEO | Guías | https://ejemplo.com/blog/seo/ |
            | https://ejemplo.com/servicios/ | Servicios | | | https://ejemplo.com/ |
            | https://ejemplo.com/servicios/consultoria/ | Servicios | Consultoría | | https://ejemplo.com/servicios/ |

            **Columnas requeridas:**
            - `URL`: La URL completa (clave primaria, debe coincidir con tu CSV de embeddings)
            - `Nivel1`, `Nivel2`, `Nivel3`: Categorías jerárquicas (opcional)
            - `Padre`: URL de la página padre (opcional pero recomendado)

            ---

            ### ⚙️ Parámetros explicados

            - **Profundidad de URL**: Cuántos segmentos de URL usar para la jerarquía (ej: 2 = `/blog/seo/`)
            - **Máx. enlaces por padre**: Límite de enlaces entre hermanos o de padre a hijos
            - **Peso de enlaces**: Para cálculos de PageRank posteriores
            - **Enlaces horizontales**: Activar para enlazar hermanos entre sí
            - **Prioridad semántica**: Ordenar hermanos por similitud (requiere embeddings)
            """)

        st.markdown("---")

        # ====================================================================
        # CARGAR EXCEL DE JERARQUÍA (NUEVO)
        # ====================================================================
        with st.expander("📁 **Cargar Excel de jerarquía (opcional)**", expanded=False):
            st.markdown("""
            Sube un Excel con la estructura jerárquica de tu sitio.
            La columna **URL** será la clave para unir con tus datos de embeddings.
            """)

            hierarchy_file = st.file_uploader(
                "Subir Excel de jerarquía",
                type=["xlsx", "xls", "csv"],
                key="structural_hierarchy_file"
            )

            if hierarchy_file is not None:
                try:
                    if hierarchy_file.name.endswith('.csv'):
                        hierarchy_df = pd.read_csv(hierarchy_file)
                    else:
                        hierarchy_df = pd.read_excel(hierarchy_file)

                    st.success(f"✅ Cargado: {len(hierarchy_df)} filas")

                    # Mostrar preview
                    st.dataframe(hierarchy_df.head(10))

                    # Detectar columnas
                    url_col_options = [c for c in hierarchy_df.columns if 'url' in c.lower()]
                    if not url_col_options:
                        url_col_options = list(hierarchy_df.columns)

                    hierarchy_url_col = st.selectbox(
                        "Columna de URL en el archivo de jerarquía",
                        options=url_col_options,
                        key="structural_hierarchy_url_col"
                    )

                    # Detectar columnas de jerarquía
                    hierarchy_level_cols = st.multiselect(
                        "Columnas de niveles jerárquicos (en orden)",
                        options=[c for c in hierarchy_df.columns if c != hierarchy_url_col],
                        default=[c for c in hierarchy_df.columns if 'nivel' in c.lower() or 'level' in c.lower() or 'categoria' in c.lower()],
                        key="structural_hierarchy_level_cols"
                    )

                    parent_col_options = ['(Ninguna)'] + [c for c in hierarchy_df.columns if c != hierarchy_url_col]
                    parent_col = st.selectbox(
                        "Columna de URL padre (opcional)",
                        options=parent_col_options,
                        key="structural_hierarchy_parent_col"
                    )

                    if st.button("✅ Aplicar jerarquía", key="apply_hierarchy"):
                        # Crear columna de jerarquía combinada
                        if hierarchy_level_cols:
                            hierarchy_df['_jerarquia_combinada'] = hierarchy_df[hierarchy_level_cols].fillna('').agg(' > '.join, axis=1)
                            hierarchy_df['_jerarquia_combinada'] = hierarchy_df['_jerarquia_combinada'].str.strip(' > ')

                        # Guardar en session state
                        st.session_state["structural_hierarchy_df"] = hierarchy_df
                        st.session_state["structural_hierarchy_url_col"] = hierarchy_url_col
                        st.session_state["structural_hierarchy_level_cols"] = hierarchy_level_cols
                        st.session_state["structural_hierarchy_parent_col"] = parent_col if parent_col != '(Ninguna)' else None

                        st.success("✅ Jerarquía aplicada. Ahora puedes generar enlaces estructurales.")
                        st.rerun()

                except Exception as e:
                    st.error(f"Error al cargar archivo: {e}")

            # Mostrar estado si ya hay jerarquía cargada
            if st.session_state.get("structural_hierarchy_df") is not None:
                st.info(f"📊 Jerarquía cargada: {len(st.session_state['structural_hierarchy_df'])} registros")
                if st.button("🗑️ Limpiar jerarquía cargada", key="clear_hierarchy"):
                    st.session_state["structural_hierarchy_df"] = None
                    st.rerun()

        st.markdown("---")

        # ====================================================================
        # CONFIGURACIÓN DE JERARQUÍA
        # ====================================================================
        st.markdown("**Fuente de jerarquía**")

        hierarchy_source = st.radio(
            "¿De dónde obtener la jerarquía?",
            options=[
                "Extraer de URLs automáticamente",
                "Usar columna del CSV principal",
                "Usar Excel de jerarquía cargado"
            ],
            key="linking_structural_hierarchy_source",
            horizontal=True
        )

        hierarchy_column = None
        use_loaded_hierarchy = False

        if hierarchy_source == "Usar columna del CSV principal":
            hierarchy_candidates = list(processed_df.columns)
            hierarchy_column = st.selectbox(
                "Columna de jerarquía/categoría",
                options=hierarchy_candidates,
                key="linking_structural_hierarchy_column"
            )
        elif hierarchy_source == "Usar Excel de jerarquía cargado":
            if st.session_state.get("structural_hierarchy_df") is None:
                st.warning("⚠️ Primero carga un Excel de jerarquía en la sección de arriba")
            else:
                use_loaded_hierarchy = True
                st.success("✅ Usando jerarquía cargada")

        st.markdown("**Parámetros de enlazado estructural**")
        param_col1, param_col2, param_col3 = st.columns(3)
        with param_col1:
            url_depth = st.number_input(
                "Profundidad de URL para jerarquía (si no hay columna custom)",
                min_value=1,
                max_value=5,
                value=2,
                key="linking_structural_url_depth",
                help="Número de segmentos de URL a considerar para extraer jerarquía"
            )
        with param_col2:
            max_links_per_parent = st.number_input(
                "Máximo enlaces entre hermanos o de padre a hijos",
                min_value=1,
                max_value=20,
                value=3,
                key="linking_structural_max_links_per_parent"
            )
        with param_col3:
            link_weight = st.slider(
                "Peso de enlaces (para PageRank posterior)",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1,
                key="linking_structural_link_weight"
            )

        include_horizontal = st.checkbox(
            "Incluir enlaces horizontales (entre hermanos del mismo nivel)",
            value=True,
            key="linking_structural_include_horizontal"
        )

        use_semantic_priority = st.checkbox(
            "Priorizar hermanos por similitud semántica (requiere embeddings)",
            value=False,
            key="linking_structural_use_semantic_priority",
            help="Ordena enlaces entre hermanos por similitud semántica en lugar de orden alfabético"
        )

        if st.button("🚀 Generar recomendaciones estructurales", key="linking_structural_run"):
            with st.spinner("Calculando enlaces estructurales basados en jerarquía..."):
                try:
                    # Preparar DataFrame con jerarquía si se cargó un Excel
                    df_to_use = processed_df.copy()

                    if use_loaded_hierarchy and st.session_state.get("structural_hierarchy_df") is not None:
                        # Unir con el Excel de jerarquía cargado
                        hierarchy_df = st.session_state["structural_hierarchy_df"]
                        hierarchy_url_col = st.session_state.get("structural_hierarchy_url_col", "URL")
                        hierarchy_level_cols = st.session_state.get("structural_hierarchy_level_cols", [])

                        if '_jerarquia_combinada' in hierarchy_df.columns:
                            # Unir por URL
                            df_to_use = df_to_use.merge(
                                hierarchy_df[[hierarchy_url_col, '_jerarquia_combinada']],
                                left_on=url_column,
                                right_on=hierarchy_url_col,
                                how='left'
                            )
                            hierarchy_column = '_jerarquia_combinada'
                            st.info(f"📊 Jerarquía aplicada a {df_to_use['_jerarquia_combinada'].notna().sum()} de {len(df_to_use)} URLs")

                    # Preparar máscaras de exclusión
                    exclude_source_mask = None
                    exclude_target_mask = None
                    non_linkable = st.session_state.get("non_linkable_mask")

                    if non_linkable is not None:
                        if st.session_state.get("exclude_non_linkable_as_source", False):
                            exclude_source_mask = non_linkable
                        if st.session_state.get("exclude_non_linkable_as_target", False):
                            exclude_target_mask = non_linkable

                    report_df = structural_taxonomy_linking(
                        df=df_to_use,
                        url_column=url_column,
                        hierarchy_column=hierarchy_column,
                        depth=int(url_depth),
                        max_links_per_parent=int(max_links_per_parent),
                        include_horizontal=include_horizontal,
                        link_weight=float(link_weight),
                        use_semantic_priority=use_semantic_priority,
                        exclude_as_source_mask=exclude_source_mask,
                        exclude_as_target_mask=exclude_target_mask,
                    )

                    # Filtrar enlaces existentes si están cargados
                    existing_links = st.session_state.get("linking_existing_links_set")
                    if existing_links and len(report_df) > 0:
                        report_df, excluded_count = filter_new_link_recommendations(
                            report_df, existing_links, "Origen URL", "Destino URL"
                        )
                        if excluded_count > 0:
                            st.info(f"ℹ️ Se excluyeron {excluded_count} enlaces que ya existen")

                    st.session_state["linking_structural_report"] = report_df
                    st.success(f"✅ {len(report_df)} enlaces estructurales generados.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Error al generar enlaces estructurales: {exc}")
                    import traceback
                    st.code(traceback.format_exc())

        if st.session_state.get("linking_structural_report") is not None:
            report_df = st.session_state["linking_structural_report"]
            st.dataframe(report_df, use_container_width=True)

            # Mostrar estadísticas por estrategia
            if "Estrategia" in report_df.columns:
                with st.expander("📊 Distribución por estrategia", expanded=False):
                    strategy_counts = report_df["Estrategia"].value_counts()
                    st.bar_chart(strategy_counts)

            download_dataframe_button(
                report_df,
                filename="linking_structural_recommendations.xlsx",
                label="📥 Descargar recomendaciones estructurales"
            )

    # ========================================================================
    # SECCIÓN FINAL: INTERPRETACIÓN CON GEMINI
    # ========================================================================
    st.markdown("---")
    st.markdown("### 🤖 Interpretación con Gemini AI")
    st.caption(
        "Envía los resultados de los diferentes modos a Gemini para obtener conclusiones estratégicas, "
        "detectar riesgos y recibir acciones recomendadas."
    )

    gemini_col, notes_col = st.columns([1, 1])
    with gemini_col:
        if st.button("🧠 Interpretar resultados con Gemini", key="linking_interpret_gemini"):
            gemini_key_value = get_gemini_api_key_from_context()
            gemini_model_value = get_gemini_model_from_context()

            if not gemini_key_value or not gemini_key_value.strip():
                st.error("Configura tu API key de Gemini en el panel lateral.")
            else:
                payload = build_linking_reports_payload(max_rows=40)
                if not payload:
                    st.warning("No hay resultados generados aún en ningún modo. Ejecuta al menos un algoritmo primero.")
                else:
                    extra_notes = st.session_state.get("linking_lab_gemini_notes", "")
                    with st.spinner("Generando interpretación con Gemini..."):
                        try:
                            summary = interpret_linking_reports_with_gemini(
                                api_key=gemini_key_value,
                                model_name=gemini_model_value,
                                payload=payload,
                                extra_notes=extra_notes,
                            )
                            st.session_state["linking_lab_gemini_summary"] = summary
                            st.session_state["gemini_api_key"] = gemini_key_value.strip()
                            st.session_state["gemini_model_name"] = gemini_model_value.strip() or get_gemini_model_from_context()
                            st.success("Interpretación generada correctamente.")
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"Error al interpretar con Gemini: {exc}")
    with notes_col:
        st.session_state["linking_lab_gemini_notes"] = st.text_area(
            "Notas o contexto para Gemini (opcional)",
            value=st.session_state.get("linking_lab_gemini_notes", ""),
            height=120,
        )

    if st.session_state.get("linking_lab_gemini_summary"):
        st.markdown("**Conclusiones generadas por Gemini**")
        st.markdown(st.session_state["linking_lab_gemini_summary"])

__all__ = [
    "render_linking_lab",
]
