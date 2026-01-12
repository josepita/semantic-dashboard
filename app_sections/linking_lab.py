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
)

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

    # Sección de configuración de enlaces existentes (inlinks)
    with st.expander("⚙️ Configurar enlaces existentes (opcional - mejora PageRank)", expanded=False):
        st.caption(
            "Sube un archivo CSV/Excel con tus enlaces internos existentes. Esto mejorará el cálculo de PageRank "
            "y evitará recomendar enlaces que ya tienes. Formato requerido: columnas 'source' (URL origen) y "
            "'target' (URL destino). Opcionalmente: 'weight' (peso del enlace, por defecto 1.0)."
        )
        inlinks_upload = st.file_uploader(
            "Archivo de enlaces existentes",
            type=["csv", "xlsx", "xls"],
            key="linking_lab_inlinks_uploader",
            label_visibility="collapsed",
        )
        if inlinks_upload:
            try:
                if inlinks_upload.name.lower().endswith(".csv"):
                    inlinks_df = pd.read_csv(inlinks_upload)
                else:
                    inlinks_df = pd.read_excel(inlinks_upload)

                # Validar columnas requeridas
                if "source" not in inlinks_df.columns or "target" not in inlinks_df.columns:
                    st.error("El archivo debe contener columnas 'source' y 'target'.")
                else:
                    # Procesar enlaces
                    inlinks_df["source"] = inlinks_df["source"].astype(str).str.strip()
                    inlinks_df["target"] = inlinks_df["target"].astype(str).str.strip()

                    # Añadir weight si no existe
                    if "weight" not in inlinks_df.columns:
                        inlinks_df["weight"] = 1.0
                    else:
                        inlinks_df["weight"] = pd.to_numeric(inlinks_df["weight"], errors="coerce").fillna(1.0)

                    # Filtrar enlaces válidos
                    inlinks_df = inlinks_df[
                        (inlinks_df["source"].str.len() > 0) &
                        (inlinks_df["target"].str.len() > 0) &
                        (inlinks_df["source"] != inlinks_df["target"])
                    ]

                    if inlinks_df.empty:
                        st.warning("No se encontraron enlaces válidos en el archivo.")
                    else:
                        st.session_state["linking_existing_edges"] = [
                            (row["source"], row["target"], row["weight"])
                            for _, row in inlinks_df.iterrows()
                        ]
                        st.success(f"✅ {len(inlinks_df)} enlaces existentes cargados correctamente.")
                        st.dataframe(inlinks_df.head(10), use_container_width=True)

            except Exception as exc:  # noqa: BLE001
                st.error(f"Error al leer el archivo de enlaces: {exc}")

        # Opción para limpiar enlaces existentes
        if st.session_state.get("linking_existing_edges"):
            num_edges = len(st.session_state["linking_existing_edges"])
            st.info(f"📊 {num_edges} enlaces existentes en memoria")
            if st.button("🗑️ Limpiar enlaces existentes", key="linking_clear_inlinks"):
                st.session_state["linking_existing_edges"] = None
                st.success("Enlaces existentes eliminados.")
                st.rerun()

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
                    with st.spinner("Calculando recomendaciones semánticas básicas..."):
                        try:
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
                            )
                            st.session_state["linking_basic_report"] = report_df
                            st.success(f"✅ {len(report_df)} recomendaciones generadas.")
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"Error al generar recomendaciones: {exc}")

            if st.session_state.get("linking_basic_report") is not None:
                report_df = st.session_state["linking_basic_report"]
                st.dataframe(report_df, use_container_width=True)
                download_dataframe_button(
                    report_df,
                    filename_prefix="linking_basic_recommendations",
                    button_label="📥 Descargar recomendaciones básicas"
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
                            )
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
                    filename_prefix="linking_advanced_recommendations",
                    button_label="📥 Descargar recomendaciones avanzadas"
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
                            )
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
                    filename_prefix="linking_hybrid_cls_recommendations",
                    button_label="📥 Descargar recomendaciones híbridas (CLS)"
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

        # Opción de usar jerarquía custom o extraer de URLs
        use_custom_hierarchy = st.checkbox(
            "Usar columna de jerarquía custom (categoría, silo, etc.)",
            key="linking_structural_use_custom_hierarchy"
        )

        hierarchy_column = None
        if use_custom_hierarchy:
            hierarchy_candidates = list(processed_df.columns)
            hierarchy_column = st.selectbox(
                "Columna de jerarquía custom",
                options=hierarchy_candidates,
                key="linking_structural_hierarchy_column"
            )

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
                    report_df = structural_taxonomy_linking(
                        df=processed_df,
                        url_column=url_column,
                        hierarchy_column=hierarchy_column if use_custom_hierarchy else None,
                        depth=int(url_depth),
                        max_links_per_parent=int(max_links_per_parent),
                        include_horizontal=include_horizontal,
                        link_weight=float(link_weight),
                        use_semantic_priority=use_semantic_priority,
                    )
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
                filename_prefix="linking_structural_recommendations",
                button_label="📥 Descargar recomendaciones estructurales"
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
