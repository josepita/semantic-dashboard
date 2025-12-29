from __future__ import annotations

import json
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple, Set

import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# Gemini helpers - inline implementation for standalone app
def get_gemini_api_key_from_context() -> str:
    import os
    candidates = [
        st.session_state.get("gemini_api_key"),
        os.environ.get("GEMINI_API_KEY"),
        os.environ.get("GOOGLE_API_KEY"),
    ]
    for candidate in candidates:
        if candidate:
            return candidate.strip()
    return ""

def get_gemini_model_from_context() -> str:
    return st.session_state.get("gemini_model", "gemini-2.0-flash-exp")

from modules.semantic_tools import download_dataframe_button

try:
    import google.generativeai as genai
except ModuleNotFoundError:
    genai = None

def render_linking_lab() -> None:
    """Pantalla dedicada al laboratorio de enlazado interno."""
    st.subheader("ðŸ”— Laboratorio de enlazado interno")
    st.caption(
        "Este laboratorio combina teor?a de PageRank, sem?ntica vectorial y arquitectura de la informaci?n para dise?ar "
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
                                dataset_ready = True
    processed_df = st.session_state.get("processed_df")
    url_column = st.session_state.get("url_column")
    if processed_df is None or url_column is None:
        st.warning("Carga un dataset en esta sección o en la Caja 1 para habilitar el laboratorio.")
        return

    # Cargar enlaces existentes (inlinks) opcional
    with st.expander("ðŸ“„ Cargar archivo de enlaces existentes (opcional)", expanded=False):
        st.caption(
            "Sube un archivo CSV o Excel con los enlaces internos actuales de tu sitio "
            "(por ejemplo, exportado desde Screaming Frog con la columna 'Link Position'). "
            "Esto mejorará el cálculo de PageRank y evitará recomendar enlaces que ya existen."
        )
        st.markdown(
            "**Ventajas de incluir ubicación del enlace:**\n"
            "- Filtrar enlaces de navegación (menú, footer) que tienen menos valor semántico\n"
            "- Ponderar diferente enlaces desde contenido editorial vs. estructurales\n"
            "- Control total sobre qué enlaces influyen en el PageRank"
        )
        
        inlinks_upload = st.file_uploader(
            "Archivo de inlinks",
            type=["csv", "xlsx", "xls"],
            key="inlinks_file_uploader",
            label_visibility="collapsed",
        )
        
        if inlinks_upload:
            try:
                if inlinks_upload.name.lower().endswith(".csv"):
                    inlinks_df = pd.read_csv(inlinks_upload)
                else:
                    inlinks_df = pd.read_excel(inlinks_upload)
                
                st.dataframe(inlinks_df.head(10), use_container_width=True)
                st.caption(f"ðŸ“Š Total de filas en el archivo: {len(inlinks_df):,}")
                
                # Selección manual de columnas
                st.markdown("#### ðŸ”§ Configuración de columnas")
                
                available_columns = list(inlinks_df.columns)
                
                col_select_1, col_select_2 = st.columns(2)
                
                with col_select_1:
                    # Sugerir columna Source
                    source_suggestions = [col for col in available_columns 
                                         if any(term in col.lower() for term in ("source", "origen", "from", "de", "address"))]
                    source_default_idx = 0
                    if source_suggestions:
                        source_default_idx = available_columns.index(source_suggestions[0])
                    
                    source_col = st.selectbox(
                        "ðŸ“ Columna Source (Origen)",
                        options=available_columns,
                        index=source_default_idx,
                        key="inlinks_source_column",
                        help="Columna que contiene la URL de origen del enlace"
                    )
                
                with col_select_2:
                    # Sugerir columna Target
                    target_suggestions = [col for col in available_columns 
                                         if any(term in col.lower() for term in ("destination", "target", "destino", "to", "a"))]
                    target_default_idx = min(1, len(available_columns) - 1)
                    if target_suggestions:
                        target_default_idx = available_columns.index(target_suggestions[0])
                    
                    target_col = st.selectbox(
                        "ðŸŽ¯ Columna Target (Destino)",
                        options=available_columns,
                        index=target_default_idx,
                        key="inlinks_target_column",
                        help="Columna que contiene la URL de destino del enlace"
                    )
                
                # Verificar que no sean la misma columna
                if source_col == target_col:
                    st.error("âš ï¸ Las columnas Source y Target no pueden ser la misma. Selecciona columnas diferentes.")
                else:
                    # Columna opcional de tipo/ubicación de enlace
                    st.markdown("#### ðŸ” Filtrado por ubicación (opcional)")
                    
                    # Detectar si hay columna de tipo de enlace
                    type_suggestions = [col for col in available_columns 
                                       if any(term in col.lower() for term in ("type", "tipo", "position", "ubicacion", "ubicación", "location", "link position"))]
                    
                    has_type_column = st.checkbox(
                        "Filtrar por ubicación/tipo de enlace",
                        value=bool(type_suggestions),
                        key="inlinks_use_type_filter",
                        help="Activa esto si tu archivo tiene una columna que indica dónde está el enlace (menú, contenido, footer, etc.)"
                    )
                    
                    link_type_col = None
                    included_types = None
                    type_weights = {}
                    
                    if has_type_column:
                        type_default_idx = 0
                        if type_suggestions:
                            type_default_idx = available_columns.index(type_suggestions[0])
                        
                        link_type_col = st.selectbox(
                            "Columna de ubicación/tipo",
                            options=available_columns,
                            index=type_default_idx,
                            key="inlinks_type_column",
                            help="Columna que indica la ubicación del enlace (ej: 'Link Position' en Screaming Frog)"
                        )
                        
                        # Mostrar valores únicos de tipo
                        unique_types = inlinks_df[link_type_col].dropna().unique().tolist()
                        st.caption(f"Valores únicos encontrados: {len(unique_types)}")
                        
                        if unique_types:
                            # Mostrar distribución
                            type_counts = inlinks_df[link_type_col].value_counts()
                            st.markdown("**Distribución de enlaces por tipo:**")
                            for type_val, count in type_counts.head(10).items():
                                st.caption(f"â€¢ {type_val}: {count:,} enlaces ({count/len(inlinks_df)*100:.1f}%)")
                            
                            # Selector de tipos incluidos
                            st.markdown("**Tipos de enlaces a incluir:**")
                            
                            # Sugerir valores por defecto (excluir navegación/footer)
                            nav_keywords = ["nav", "menu", "footer", "header", "sidebar", "breadcrumb"]
                            suggested_include = [t for t in unique_types 
                                               if not any(kw in str(t).lower() for kw in nav_keywords)]
                            
                            # Si no hay sugerencias, incluir todos
                            if not suggested_include:
                                suggested_include = unique_types
                            
                            included_types = st.multiselect(
                                "Incluir solo estos tipos",
                                options=unique_types,
                                default=suggested_include,
                                key="inlinks_included_types",
                                help="Solo estos tipos de enlaces se usarán para PageRank. Excluye navegación/footer si solo quieres enlaces editoriales."
                            )
                            
                            # Opción de ponderar diferente según tipo
                            use_weights = st.checkbox(
                                "âš–ï¸ Aplicar pesos diferentes según ubicación",
                                value=False,
                                key="inlinks_use_weights",
                                help="Permite dar más/menos peso a ciertos tipos de enlaces en el PageRank"
                            )
                            
                            if use_weights and included_types:
                                st.markdown("**Ajustar pesos (multiplicador del boost de PageRank):**")
                                st.caption("Base weight = 2.0. Los enlaces se multiplicarán por este factor.")
                                
                                weight_cols = st.columns(min(3, len(included_types)))
                                for idx, link_type in enumerate(included_types):
                                    with weight_cols[idx % 3]:
                                        # Sugerir pesos por defecto basados en tipo
                                        default_weight = 1.0
                                        type_lower = str(link_type).lower()
                                        if any(kw in type_lower for kw in ["content", "contenido", "body", "article"]):
                                            default_weight = 1.5  # Más peso a enlaces de contenido
                                        elif any(kw in type_lower for kw in ["sidebar", "related", "relacionado"]):
                                            default_weight = 0.8  # Menos peso a sidebar
                                        
                                        type_weights[link_type] = st.slider(
                                            f"{link_type}",
                                            min_value=0.1,
                                            max_value=2.0,
                                            value=default_weight,
                                            step=0.1,
                                            key=f"inlinks_weight_{idx}",
                                            help=f"Multiplicador para enlaces de tipo '{link_type}'"
                                        )
                    
                    # Botón para procesar y cargar
                    if st.button("âœ… Cargar enlaces en sesión", type="primary", key="inlinks_load_button"):
                        # Parsear enlaces existentes con filtrado y pesos
                        existing_edges: List[Tuple[str, str, float]] = []  # Ahora con peso
                        filtered_count = 0
                        
                        for _, row in inlinks_df.iterrows():
                            source_url = str(row[source_col]).strip()
                            target_url = str(row[target_col]).strip()
                            
                            if not source_url or not target_url:
                                continue
                            
                            # Filtrar por tipo si está activado
                            if has_type_column and link_type_col and included_types is not None:
                                link_type_value = row.get(link_type_col)
                                if link_type_value not in included_types:
                                    filtered_count += 1
                                    continue
                                
                                # Aplicar peso si está configurado
                                weight = type_weights.get(link_type_value, 1.0) if type_weights else 1.0
                            else:
                                weight = 1.0
                            
                            existing_edges.append((source_url, target_url, weight))
                        
                        st.session_state["existing_inlinks"] = existing_edges
                        
                        # Mensaje de éxito con estadísticas
                        st.success(
                            f"âœ… Cargados **{len(existing_edges):,}** enlaces existentes\n\n"
                            f"- Columnas: `{source_col}` â†’ `{target_col}`"
                        )
                        
                        if filtered_count > 0:
                            st.info(f"ðŸ” Filtrados {filtered_count:,} enlaces (tipos excluidos)")
                        
                        if type_weights:
                            st.info(
                                f"âš–ï¸ Pesos aplicados:\n" +
                                "\n".join([f"- {k}: {v}x" for k, v in list(type_weights.items())[:5]])
                            )
                        
                        st.markdown(
                            "**Estos enlaces se utilizarán para:**\n"
                            "- Mejorar el cálculo de PageRank (boost de autoridad)\n"
                            "- Evitar recomendar enlaces duplicados"
                        )
                    
            except Exception as exc:
                st.error(f"Error al procesar archivo de inlinks: {exc}")
        
        # Mostrar estado de inlinks cargados
        existing_inlinks = st.session_state.get("existing_inlinks")
        if existing_inlinks:
            # Distinguir si son tuplas de 2 (antiguo) o 3 elementos (nuevo con peso)
            has_weights = len(existing_inlinks[0]) == 3 if existing_inlinks else False
            
            if has_weights:
                st.caption(f"ðŸ“Š **Estado:** {len(existing_inlinks):,} enlaces cargados (con pesos) en sesión")
            else:
                st.caption(f"ðŸ“Š **Estado:** {len(existing_inlinks):,} enlaces cargados en sesión")
            
            if st.button("ðŸ—‘ï¸ Limpiar enlaces cargados", key="clear_inlinks_button"):
                st.session_state["existing_inlinks"] = None
                st.rerun()
        else:
            st.caption("ðŸ“Š **Estado:** No hay enlaces cargados (modo sin enlaces existentes)")

    entity_payload = st.session_state.get("entity_payload_by_url") or {}
    helper_entity_col = "EntityPayloadFromGraph"
    if entity_payload:
        try:
            if helper_entity_col not in processed_df.columns:
                processed_df[helper_entity_col] = (
                    processed_df[url_column].astype(str).str.strip().map(
                        lambda url: entity_payload.get(url) or entity_payload.get(url.rstrip("/")) or {}
                    )
                )
        except Exception:
            pass

    session_defaults = {
        "linking_basic_report": None,
        "linking_adv_report": None,
        "linking_adv_orphans": None,
        "linking_hybrid_report": None,
        "linking_hybrid_orphans": None,
        "linking_hybrid_pagerank": None,
        "linking_structural_report": None,
        "linking_lab_gemini_summary": "",
    }
    for key, default in session_defaults.items():
        st.session_state.setdefault(key, default)

    type_candidates = detect_page_type_columns(processed_df)
    fallback_type_cols = processed_df.select_dtypes(include=["object", "category"]).columns.tolist()
    type_options = type_candidates or fallback_type_cols


    explainer_col, status_col = st.columns([1.5, 1])
    with explainer_col:
        st.markdown("#### ?C?mo funciona este laboratorio?")
        st.markdown(
            "- **1. Prepara el dataset:** sube embeddings y define las columnas clave.\n"
            "- **2. Ejecuta los modos:** B?sico (similitud), Avanzado (silos), H?brido (CLS + PageRank) y Estructural.\n"
            "- **3. Eval?a resultados:** revisa hu?rfanas, descarga Excel y prioriza enlaces.\n"
            "- **4. Interpreta:** usa Gemini para convertir los hallazgos en acciones."
        )
        st.caption("El modo h?brido puede usar entidades guardadas desde el an?lisis de conocimiento para ponderar mejor las recomendaciones.")
    with status_col:
        st.markdown("#### Estado r?pido")
        if len(processed_df):
            st.success(f"Dataset listo con {len(processed_df):,} filas.")
        else:
            st.warning("Dataset vac?o, sube un archivo.")
        if entity_payload:
            st.info(f"Entidades guardadas para {len(entity_payload)} URLs.")
        else:
            st.caption("A?n no hay entidades consolidadas; genera el grafo sem?ntico para habilitarlas.")

    tabs = st.tabs(
        [
            "Básico (similitud)",
            "Avanzado por silos",
            "Híbrido CLS",
            "Estructural / Taxonomía",
        ]
    )

    with tabs[0]:
        st.markdown(
            "**Similitud coseno tradicional**\n\n"
            "Calcula la proximidad entre embeddings para detectar URLs con la misma intenci?n. "
            "?salo para proponer enlaces contextuales r?pidos que repartan autoridad sin canibalizar."
        )
        if not type_options:
            st.info("No se detectaron columnas categóricas para identificar el tipo de página.")
        else:
            default_type_col = st.session_state.get("page_type_column")
            default_index = type_options.index(default_type_col) if default_type_col in type_options else 0
            page_type_col = st.selectbox(
                "Columna con el tipo de página",
                options=type_options,
                index=default_index,
                key="linking_basic_type_column",
            )
            st.session_state["page_type_column"] = page_type_col

            type_series = processed_df[page_type_col].astype(str).str.strip()
            unique_types = list(dict.fromkeys([val for val in type_series if val and val.lower() != "nan"]))
            if not unique_types:
                st.warning("La columna seleccionada no contiene valores válidos.")
            else:
                default_source_value = guess_default_type(unique_types, ["blog", "post"]) or unique_types[0]
                remaining_types = [val for val in unique_types if val != default_source_value]
                default_priority_value = (
                    guess_default_type(unique_types, ["trat", "serv", "categ", "servicio"]) or default_source_value
                )
                source_defaults = [default_source_value] if default_source_value else unique_types[:1]
                primary_defaults = (
                    [default_priority_value]
                    if default_priority_value and default_priority_value not in source_defaults
                    else source_defaults
                )
                secondary_defaults = source_defaults

                col_basic_types = st.columns(2)
                with col_basic_types[0]:
                    source_types = st.multiselect(
                        "Tipos origen",
                        options=unique_types,
                        default=source_defaults,
                        key="linking_basic_source_types",
                    )
                with col_basic_types[1]:
                    primary_targets = st.multiselect(
                        "Destinos prioritarios",
                        options=unique_types,
                        default=primary_defaults,
                        key="linking_basic_primary_targets",
                    )
                secondary_targets = st.multiselect(
                    "Destinos secundarios (opcional)",
                    options=unique_types,
                    default=secondary_defaults,
                    key="linking_basic_secondary_targets",
                )

                col_basic_params = st.columns([2, 1])
                with col_basic_params[0]:
                    threshold_percent = st.slider(
                        "Umbral mínimo de similitud (%)",
                        min_value=40,
                        max_value=98,
                        value=80,
                        step=1,
                        key="linking_basic_threshold",
                    )
                with col_basic_params[1]:
                    max_links = int(
                        st.number_input(
                            "Enlaces por origen",
                            min_value=1,
                            max_value=10,
                            value=3,
                            step=1,
                            key="linking_basic_max_links",
                        )
                    )

                col_basic_limits = st.columns(2)
                with col_basic_limits[0]:
                    max_primary = int(
                        st.number_input(
                            "Destinos prioritarios",
                            min_value=0,
                            max_value=max_links,
                            value=min(2, max_links),
                            step=1,
                            key="linking_basic_max_primary",
                        )
                    )
                with col_basic_limits[1]:
                    max_secondary = int(
                        st.number_input(
                            "Destinos complementarios",
                            min_value=0,
                            max_value=max_links,
                            value=max(0, max_links - max_primary),
                            step=1,
                            key="linking_basic_max_secondary",
                        )
                    )

                source_urls = (
                    processed_df.loc[processed_df[page_type_col].isin(source_types), url_column]
                    .astype(str)
                    .str.strip()
                    .tolist()
                )
                unique_sources = list(dict.fromkeys(source_urls))
                source_limit_value: Optional[int] = None
                source_url_filter: List[str] = []
                if unique_sources:
                    limit_raw = st.number_input(
                        "Máximo de páginas origen (0 = todas)",
                        min_value=0,
                        max_value=len(unique_sources),
                        value=min(50, len(unique_sources)),
                        step=1,
                        key="linking_basic_source_limit",
                    )
                    source_limit_value = int(limit_raw) if limit_raw > 0 else None
                    st.caption(f"{len(unique_sources)} URLs disponibles como origen tras los filtros.")
                    preview_options = unique_sources[:500]
                    source_url_filter = st.multiselect(
                        "Limitar a URLs específicas (opcional)",
                        options=preview_options,
                        key="linking_basic_source_filter",
                    )
                    if len(unique_sources) > len(preview_options):
                        st.caption("Sólo se muestran las primeras 500 URLs para agilizar la búsqueda.")
                else:
                    st.info("No hay URLs que coincidan con los tipos de origen seleccionados.")

                if st.button("Generar recomendaciones básicas", type="primary", key="linking_basic_button"):
                    if not source_types:
                        st.error("Selecciona al menos un tipo de origen.")
                    else:
                        with st.spinner("Calculando similitudes entre páginas..."):
                            basic_report = semantic_link_recommendations(
                                processed_df,
                                url_column=url_column,
                                type_column=page_type_col,
                                source_types=source_types,
                                primary_target_types=primary_targets,
                                secondary_target_types=secondary_targets or None,
                                similarity_threshold=threshold_percent / 100.0,
                                max_links_per_source=max_links,
                                max_primary=max_primary,
                                max_secondary=max_secondary,
                                source_limit=source_limit_value,
                                selected_source_urls=source_url_filter or None,
                            )
                        if basic_report.empty:
                            st.info("No se encontraron coincidencias con los criterios establecidos.")
                            st.session_state["linking_basic_report"] = None
                        else:
                            st.success(f"Se generaron {len(basic_report)} recomendaciones.")
                            st.session_state["linking_basic_report"] = basic_report

        basic_report = st.session_state.get("linking_basic_report")
        if isinstance(basic_report, pd.DataFrame) and not basic_report.empty:
            st.dataframe(basic_report, use_container_width=True)
            download_dataframe_button(
                basic_report,
                "recomendaciones_enlazado_basico.xlsx",
                "Descargar recomendaciones",
            )

    with tabs[1]:
        st.markdown(
            "**Modo avanzado por silos**\n\n"
            "Combina taxonom?as y reglas de negocio para conectar URLs madre-hija dentro de un mismo silo tem?tico. "
            "Ideal para reforzar keyword clusters, distribuir PageRank vertical y detectar hu?rfanas estrat?gicas."
        )
        if not type_options:
            st.info("No se detectaron columnas categóricas para este análisis.")
        else:
            adv_default_col = st.session_state.get("page_type_column")
            adv_index = type_options.index(adv_default_col) if adv_default_col in type_options else 0
            adv_type_column = st.selectbox(
                "Columna de tipo",
                options=type_options,
                index=adv_index,
                key="linking_adv_type_column",
            )
            adv_values = processed_df[adv_type_column].astype(str).str.strip()
            adv_unique_types = list(dict.fromkeys([val for val in adv_values if val and val.lower() != "nan"]))
            if not adv_unique_types:
                st.warning("La columna seleccionada no tiene valores válidos.")
            else:
                adv_default_source = guess_default_type(adv_unique_types, ["blog", "post"]) or adv_unique_types[0]
                adv_default_primary = guess_default_type(adv_unique_types, ["trat", "serv", "cat"]) or adv_unique_types[0]
                col_types = st.columns(2)
                with col_types[0]:
                    adv_source_types = st.multiselect(
                        "Tipos origen",
                        options=adv_unique_types,
                        default=[adv_default_source] if adv_default_source else adv_unique_types[:1],
                        key="linking_adv_source_types",
                    )
                with col_types[1]:
                    adv_primary_targets = st.multiselect(
                        "Destinos prioritarios",
                        options=adv_unique_types,
                        default=[adv_default_primary] if adv_default_primary else adv_unique_types[:1],
                        key="linking_adv_primary_targets",
                    )
                adv_secondary_targets = st.multiselect(
                    "Destinos secundarios (opcional)",
                    options=adv_unique_types,
                    default=[adv_default_source] if adv_default_source else [],
                    key="linking_adv_secondary_targets",
                )
                col_params = st.columns(3)
                with col_params[0]:
                    adv_threshold_percent = st.slider(
                        "Umbral mínimo de similitud (%)",
                        min_value=50,
                        max_value=98,
                        value=78,
                        step=1,
                        key="linking_adv_threshold",
                    )
                with col_params[1]:
                    adv_max_links = int(
                        st.number_input(
                            "Enlaces por origen",
                            min_value=1,
                            max_value=10,
                            value=4,
                            step=1,
                            key="linking_adv_max_links",
                        )
                    )
                with col_params[2]:
                    adv_silo_boost_percent = st.slider(
                        "Boost por silo (%)",
                        min_value=0.0,
                        max_value=25.0,
                        value=8.0,
                        step=1.0,
                        key="linking_adv_silo_boost",
                    )
                col_extra = st.columns(3)
                with col_extra[0]:
                    adv_silo_depth = int(
                        st.slider(
                            "Profundidad del segmento",
                            min_value=1,
                            max_value=5,
                            value=2,
                            key="linking_adv_silo_depth",
                        )
                    )
                with col_extra[1]:
                    adv_max_primary = int(
                        st.number_input(
                            "Destinos prioritarios",
                            min_value=0,
                            max_value=adv_max_links,
                            value=min(2, adv_max_links),
                            step=1,
                            key="linking_adv_max_primary",
                        )
                    )
                with col_extra[2]:
                    adv_max_secondary = int(
                        st.number_input(
                            "Destinos secundarios",
                            min_value=0,
                            max_value=adv_max_links,
                            value=min(max(adv_max_links - adv_max_primary, 0), adv_max_links),
                            step=1,
                            key="linking_adv_max_secondary",
                        )
                    )

                adv_source_candidates = (
                    processed_df.loc[processed_df[adv_type_column].isin(adv_source_types), url_column]
                    .astype(str)
                    .str.strip()
                    .tolist()
                )
                adv_unique_sources = list(dict.fromkeys(adv_source_candidates))
                adv_limit_value: Optional[int] = None
                if adv_unique_sources:
                    adv_limit_raw = st.number_input(
                        "Máximo de páginas origen (0 = todas)",
                        min_value=0,
                        max_value=len(adv_unique_sources),
                        value=min(100, len(adv_unique_sources)),
                        step=1,
                        key="linking_adv_source_limit",
                    )
                    adv_limit_value = int(adv_limit_raw) if adv_limit_raw > 0 else None
                    st.caption(f"{len(adv_unique_sources)} URLs candidatas como origen.")
                else:
                    st.info("No hay URLs que coincidan con los tipos de origen seleccionados.")

                if st.button("Generar estrategia avanzada", type="primary", key="linking_adv_button"):
                    if not adv_source_types or not adv_primary_targets:
                        st.error("Selecciona al menos un tipo de origen y un destino prioritario.")
                    else:
                        with st.spinner("Ejecutando análisis avanzado de silos..."):
                            adv_report, adv_orphans = advanced_semantic_linking(
                                processed_df,
                                url_column=url_column,
                                type_column=adv_type_column,
                                source_types=adv_source_types,
                                primary_target_types=adv_primary_targets,
                                secondary_target_types=(adv_secondary_targets or None),
                                similarity_threshold=adv_threshold_percent / 100.0,
                                max_links_per_source=adv_max_links,
                                max_primary=adv_max_primary,
                                max_secondary=adv_max_secondary,
                                silo_depth=adv_silo_depth,
                                silo_boost=adv_silo_boost_percent / 100.0,
                                source_limit=adv_limit_value,
                            )
                        if adv_report.empty:
                            st.info("No se generaron recomendaciones con los criterios indicados.")
                            st.session_state["linking_adv_report"] = None
                            st.session_state["linking_adv_orphans"] = adv_orphans
                        else:
                            st.success(f"Se generaron {len(adv_report)} recomendaciones avanzadas.")
                            st.session_state["linking_adv_report"] = adv_report
                            st.session_state["linking_adv_orphans"] = adv_orphans

        adv_report = st.session_state.get("linking_adv_report")
        if isinstance(adv_report, pd.DataFrame) and not adv_report.empty:
            st.dataframe(adv_report, use_container_width=True)
            download_dataframe_button(
                adv_report,
                "reporte_enlazado_avanzado.xlsx",
                "Descargar estrategia avanzada",
            )
        adv_orphans = st.session_state.get("linking_adv_orphans") or []
        if adv_orphans:
            st.warning(
                f"Se detectaron {len(adv_orphans)} páginas prioritarias sin enlaces recomendados. Refuérzalas manualmente."
            )
            st.write(adv_orphans)

    with tabs[2]:
        st.markdown(
        st.markdown(
            "**CLS h?brido (Composite Linking Score)**\n\n"
            "Funde similitud vectorial, PageRank t?pico y solapamiento de entidades para priorizar enlaces seg?n autoridad y contexto. "
            "?salo cuando quieras balancear relevancia editorial con se?ales de entidad/autoridad."
        )
        )
        entity_columns = [
            col
            for col in processed_df.columns
            if processed_df[col].apply(lambda val: isinstance(val, (dict, str))).any()
        ]
        if not entity_columns:
            st.warning(
                "No se detectó ninguna columna con entidades serializadas. Importa los resultados del grafo o agrega una "
                "columna con un diccionario {QID: prominence}."
            )
        else:
            entity_column = st.selectbox(
                "Columna con entidades + prominence score",
                options=entity_columns,
                key="linking_hybrid_entity_column",
            )
            if not type_options:
                st.info("No se detectaron columnas de tipo para clasificar las páginas.")
            else:
                hyb_default_col = st.session_state.get("page_type_column")
                hyb_index = type_options.index(hyb_default_col) if hyb_default_col in type_options else 0
                hyb_type_column = st.selectbox(
                    "Columna de tipo",
                    options=type_options,
                    index=hyb_index,
                    key="linking_hybrid_type_column",
                )
                type_series = processed_df[hyb_type_column].astype(str).str.strip()
                unique_types = list(dict.fromkeys([val for val in type_series if val and val.lower() != "nan"]))
                if not unique_types:
                    st.warning("La columna seleccionada no contiene valores válidos.")
                else:
                    hybrid_source_types = st.multiselect(
                        "Tipos origen",
                        options=unique_types,
                        default=[unique_types[0]],
                        key="linking_hybrid_source_types",
                    )
                    hybrid_primary_targets = st.multiselect(
                        "Destinos prioritarios",
                        options=unique_types,
                        default=[unique_types[0]],
                        key="linking_hybrid_primary_targets",
                    )
                    col_hybrid_weights = st.columns(3)
                    with col_hybrid_weights[0]:
                        weight_sem = st.slider(
                            "Peso similitud",
                            min_value=0.1,
                            max_value=0.8,
                            value=0.4,
                            step=0.05,
                            key="linking_hybrid_weight_sem",
                        )
                    with col_hybrid_weights[1]:
                        weight_auth = st.slider(
                            "Peso autoridad",
                            min_value=0.1,
                            max_value=0.8,
                            value=0.35,
                            step=0.05,
                            key="linking_hybrid_weight_auth",
                        )
                    with col_hybrid_weights[2]:
                        weight_ent = st.slider(
                            "Peso entidades",
                            min_value=0.1,
                            max_value=0.8,
                            value=0.25,
                            step=0.05,
                            key="linking_hybrid_weight_ent",
                        )
                    col_hybrid_params = st.columns(3)
                    with col_hybrid_params[0]:
                        similarity_threshold = st.slider(
                            "Filtro mínimo CLS (0-1)",
                            min_value=0.3,
                            max_value=0.95,
                            value=0.55,
                            step=0.05,
                            key="linking_hybrid_threshold",
                        )
                    with col_hybrid_params[1]:
                        max_links = int(
                            st.number_input(
                                "Enlaces por origen",
                                min_value=1,
                                max_value=10,
                                value=4,
                                step=1,
                                key="linking_hybrid_max_links",
                            )
                        )
                    with col_hybrid_params[2]:
                        max_primary = int(
                            st.number_input(
                                "Destinos prioritarios",
                                min_value=0,
                                max_value=max_links,
                                value=min(2, max_links),
                                step=1,
                                key="linking_hybrid_max_primary",
                            )
                        )
                    col_hybrid_extra = st.columns(3)
                    with col_hybrid_extra[0]:
                        decay_factor = st.slider(
                            "Factor de decaimiento",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.15,
                            step=0.05,
                            help="Penaliza destinos con muchos enlaces en este lote.",
                            key="linking_hybrid_decay",
                        )
                    with col_hybrid_extra[1]:
                        source_limit = st.number_input(
                            "Máximo de páginas origen (0 = todas)",
                            min_value=0,
                            max_value=len(processed_df),
                            value=min(80, len(processed_df)),
                            step=1,
                            key="linking_hybrid_source_limit",
                        )
                    with col_hybrid_extra[2]:
                        top_k_edges = st.number_input(
                            "Aristas por nodo (PageRank)",
                            min_value=3,
                            max_value=15,
                            value=5,
                            step=1,
                            key="linking_hybrid_topk_edges",
                        )

                    if st.button("Generar recomendaciones híbridas (CLS)", type="primary", key="linking_hybrid_button"):
                        if not hybrid_source_types or not hybrid_primary_targets:
                            st.error("Selecciona al menos un tipo de origen y un destino prioritario.")
                        else:
                            # Obtener enlaces existentes si están cargados
                            existing_inlinks = st.session_state.get("existing_inlinks")
                            
                            with st.spinner("Calculando Composite Linking Score..."):
                                try:
                                    hybrid_report, hybrid_orphans, pagerank_scores = hybrid_semantic_linking(
                                        processed_df,
                                        url_column=url_column,
                                        type_column=hyb_type_column,
                                        entity_column=entity_column,
                                        source_types=hybrid_source_types,
                                        primary_target_types=hybrid_primary_targets,
                                        similarity_threshold=similarity_threshold,
                                        max_links_per_source=max_links,
                                        max_primary=max_primary,
                                        decay_factor=decay_factor,
                                        weights={
                                            "semantic": weight_sem,
                                            "authority": weight_auth,
                                            "entity_overlap": weight_ent,
                                        },
                                        source_limit=source_limit if source_limit > 0 else None,
                                        top_k_edges=int(top_k_edges),
                                        existing_edges=existing_inlinks,  # Nuevo parámetro
                                    )
                                except ValueError as exc:
                                    st.error(str(exc))
                                else:
                                    if hybrid_report.empty:
                                        st.info("No se generaron recomendaciones con los criterios indicados.")
                                        st.session_state["linking_hybrid_report"] = None
                                        st.session_state["linking_hybrid_orphans"] = hybrid_orphans
                                        st.session_state["linking_hybrid_pagerank"] = pagerank_scores
                                    else:
                                        st.success(f"Se generaron {len(hybrid_report)} recomendaciones híbridas.")
                                        st.session_state["linking_hybrid_report"] = hybrid_report
                                        st.session_state["linking_hybrid_orphans"] = hybrid_orphans
                                        st.session_state["linking_hybrid_pagerank"] = pagerank_scores

        hybrid_report = st.session_state.get("linking_hybrid_report")
        if isinstance(hybrid_report, pd.DataFrame) and not hybrid_report.empty:
            st.dataframe(hybrid_report, use_container_width=True)
            download_dataframe_button(
                hybrid_report,
                "recomendaciones_hibridas_cls.xlsx",
                "Descargar recomendaciones CLS",
            )
        hybrid_orphans = st.session_state.get("linking_hybrid_orphans") or []
        if hybrid_orphans:
            st.warning(f"Páginas prioritarias sin enlaces en este lote: {len(hybrid_orphans)}")
            st.write(hybrid_orphans)
        pagerank_scores = st.session_state.get("linking_hybrid_pagerank") or {}
        if pagerank_scores:
            pagerank_df = (
                pd.DataFrame({"URL": list(pagerank_scores.keys()), "PageRank": list(pagerank_scores.values())})
                .sort_values("PageRank", ascending=False)
                .reset_index(drop=True)
            )
            st.markdown("**PageRank tópico (referencia de autoridad):**")
            st.dataframe(pagerank_df.head(50), use_container_width=True)

    with tabs[3]:
        st.markdown(
        st.markdown(
            "**Enlazado estructural / taxon?mico**\n\n"
            "Deriva la jerarqu?a de URLs o columnas personalizadas para construir enlaces Padre-Hijo y entre hermanos. "
            "Es el sost?n SEO que asegura profundidad uniforme y evita calles sin salida para los robots."
        )
        )
        hierarchy_options = ["(Derivar de la URL)"] + [
            col for col in processed_df.columns if processed_df[col].dtype == object and col != url_column
        ]
        hierarchy_selection = st.selectbox(
            "Columna jerárquica",
            options=hierarchy_options,
            help="Selecciona una columna de taxonomía o deriva el camino desde la URL.",
        )
        hierarchy_column = None if hierarchy_selection == "(Derivar de la URL)" else hierarchy_selection
        depth = st.slider(
            "Profundidad derivada desde la URL",
            min_value=1,
            max_value=5,
            value=2,
            help="Se usa sólo cuando derivamos la jerarquía desde la URL.",
        )
        max_links_parent = st.number_input(
            "Máximo de enlaces por nodo",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
        )
        include_horizontal = st.checkbox("Incluir enlaces entre hermanos", value=True)
        link_weight = st.slider(
            "Peso del enlace estructural",
            min_value=0.2,
            max_value=1.0,
            value=0.6,
            step=0.05,
        )

        if st.button("Generar enlaces estructurales", type="primary", key="linking_structural_button"):
            with st.spinner("Calculando enlaces padre-hijo y hermanos..."):
                structural_df = structural_taxonomy_linking(
                    processed_df,
                    url_column=url_column,
                    hierarchy_column=hierarchy_column,
                    depth=int(depth),
                    max_links_per_parent=int(max_links_parent),
                    include_horizontal=include_horizontal,
                    link_weight=float(link_weight),
                )
            if structural_df.empty:
                st.info("No se generaron enlaces estructurales con los parámetros actuales.")
                st.session_state["linking_structural_report"] = None
            else:
                st.success(f"Se generaron {len(structural_df)} sugerencias estructurales.")
                st.session_state["linking_structural_report"] = structural_df

        structural_report = st.session_state.get("linking_structural_report")
        if isinstance(structural_report, pd.DataFrame) and not structural_report.empty:
            st.dataframe(structural_report, use_container_width=True)
            download_dataframe_button(
                structural_report,
                "recomendaciones_estructurales.xlsx",
                "Descargar enlaces estructurales",
            )



    st.divider()
    st.markdown("#### Interpretaci?n autom?tica del laboratorio (Gemini)")
    st.caption("Resume los hallazgos de los distintos modos y genera acciones sugeridas.")
    gemini_payload = build_linking_reports_payload()
    interpret_col, notes_col = st.columns([1.2, 1])
    with interpret_col:
        gemini_key_value = st.text_input(
            "Gemini API Key (usa la del panel lateral si ya la guardaste)",
            type="password",
            value=st.session_state.get("gemini_api_key", get_gemini_api_key_from_context()),
            key="linking_lab_gemini_key_input",
        )
        gemini_model_value = st.text_input(
            "Modelo Gemini",
            value=st.session_state.get("gemini_model_name", get_gemini_model_from_context()),
            key="linking_lab_gemini_model_input",
        )
        interpret_disabled = not gemini_payload
        if interpret_disabled:
            st.warning("Ejecuta al menos uno de los modos para habilitar la interpretaci?n.")
        if st.button(
            "Interpretar resultados con Gemini",
            type="primary",
            disabled=interpret_disabled,
            key="linking_lab_gemini_button",
        ):
            try:
                summary_text = interpret_linking_reports_with_gemini(
                    gemini_key_value,
                    gemini_model_value,
                    gemini_payload,
                    extra_notes=st.session_state.get("linking_lab_gemini_notes", ""),
                )
            except ValueError as exc:
                st.warning(str(exc))
            except Exception as exc:
                st.error(f"No se pudo generar la interpretaci?n: {exc}")
            else:
                st.session_state["linking_lab_gemini_summary"] = summary_text
                st.session_state["gemini_api_key"] = gemini_key_value.strip()
                st.session_state["gemini_model_name"] = gemini_model_value.strip() or get_gemini_model_from_context()
                st.success("Interpretaci?n generada correctamente.")
    with notes_col:
        st.session_state["linking_lab_gemini_notes"] = st.text_area(
            "Notas o contexto para Gemini (opcional)",
            value=st.session_state.get("linking_lab_gemini_notes", ""),
            height=120,
        )

    if st.session_state.get("linking_lab_gemini_summary"):
        st.markdown("**Conclusiones generadas por Gemini**")
        st.markdown(st.session_state["linking_lab_gemini_summary"])

def detect_embedding_columns(df: pd.DataFrame) -> List[str]:
    candidate_cols = []
    for col in df.columns:
        sample_values = df[col].dropna().astype(str).head(3)
        if sample_values.empty:
            continue
        first_val = sample_values.iloc[0]
        if "," in first_val and len(first_val.split(",")) > 50:
            candidate_cols.append(col)
    return candidate_cols

def convert_embedding_cell(value: str) -> Optional[np.ndarray]:
    try:
        clean = str(value).replace("[", "").replace("]", "")
        vector = np.array([float(x) for x in clean.split(",")])
        if vector.size == 0:
            return None
        if np.linalg.norm(vector) == 0:
            return None
        return vector
    except Exception:
        return None

def preprocess_embeddings(df: pd.DataFrame, embedding_col: str) -> Tuple[pd.DataFrame, List[str]]:
    messages: List[str] = []
    df_local = df.copy()
    df_local["EmbeddingsFloat"] = df_local[embedding_col].apply(convert_embedding_cell)
    before_drop = len(df_local)
    df_local = df_local[df_local["EmbeddingsFloat"].notna()].copy()
    dropped = before_drop - len(df_local)
    if dropped:
        messages.append(f"Se descartaron {dropped} filas por embeddings inválidos.")

    if df_local.empty:
        raise ValueError("No quedan filas con embeddings válidos.")

    lengths = df_local["EmbeddingsFloat"].apply(len)
    if lengths.nunique() > 1:
        mode_length = lengths.mode().iloc[0]
        df_local = df_local[lengths == mode_length].copy()
        messages.append(
            "Los embeddings tenían longitudes distintas. "
            f"Se conservaron {len(df_local)} filas con longitud {mode_length}."
        )

    df_local.reset_index(drop=True, inplace=True)
    return df_local, messages

def detect_url_columns(df: pd.DataFrame) -> List[str]:
    patterns = ("url", "address", "dirección", "link", "href")
    matches = [col for col in df.columns if any(pat in col.lower() for pat in patterns)]
    return matches or df.columns.tolist()

def detect_page_type_columns(df: pd.DataFrame, max_unique_values: int = 40) -> List[str]:
    """
    Intenta identificar columnas categóricas candidatas a representar el tipo de página.
    Se priorizan nombres con palabras clave como 'tipo', 'category', 'segmento', etc.
    """
    candidates: List[Tuple[int, str]] = []
    keywords = ("tipo", "type", "categoria", "category", "segment", "seccion", "page_type", "familia")

    for col in df.columns:
        if col == "EmbeddingsFloat":
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        unique_values = series.astype(str).str.strip().unique()
        unique_count = len(unique_values)
        if unique_count <= 1 or unique_count > max_unique_values:
            continue
        if unique_count == len(df):
            # Probablemente es un identificador único, no una categoría.
            continue
        score = 1
        lower_name = col.lower()
        if any(keyword in lower_name for keyword in keywords):
            score = 0
        candidates.append((score, col))

    candidates.sort(key=lambda item: (item[0], item[1]))
    return [col for _, col in candidates]

def extract_url_silo(url: str, depth: int = 2, default: str = "general") -> str:
    """
    Obtiene un segmento intermedio de la URL para usarlo como 'silo' o tema.
    depth=2 toma el segundo segmento: /tipo/tema/slug -> tema.
    """
    try:
        cleaned = str(url).split("?")[0]
        segments = [part for part in cleaned.split("/") if part]
        if not segments:
            return default
        index = min(max(depth - 1, 0), len(segments) - 1)
        candidate = segments[index].strip().lower()
        return candidate or default
    except Exception:
        return default

def suggest_anchor_from_url(url: str) -> str:
    """
    Genera un anchor legible a partir del último segmento de la URL.
    """
    try:
        clean_url = str(url).split("?")[0].split("#")[0]
        segments = [segment for segment in clean_url.split("/") if segment]
        if not segments:
            slug = clean_url
        else:
            slug = segments[-1]
            if not slug.strip() and len(segments) >= 2:
                slug = segments[-2]

        slug = slug.split(".")[0]
        slug = re.sub(r"[-_]+", " ", slug)
        slug = re.sub(r"\s+", " ", slug).strip()
        return slug.title() if slug else "Leer Más"
    except Exception:
        return "Leer Más"

def format_topic_label(raw_value: Optional[str]) -> Optional[str]:
    """
    Limpia el valor del silo para usarlo como tema en anchors.
    """
    if not raw_value:
        return None
    lowered = str(raw_value).strip().lower()
    if lowered in {"", "general", "none", "null"}:
        return None
    cleaned = re.sub(r"[-_]+", " ", lowered)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.title() if cleaned else None

def generate_contextual_anchor(page_type: str, silo_value: str, url: str) -> str:
    """
    Construye un anchor copy práctico usando tipo de página y silo.
    """
    topic_label = format_topic_label(silo_value)
    page_type_clean = (page_type or "").strip().lower()

    if topic_label:
        if page_type_clean in {"tratamiento", "servicio"}:
            return f"Ver {page_type_clean} de {topic_label}"
        if page_type_clean in {"categoria", "categoría"}:
            return f"Explorar categoría {topic_label}"
        if page_type_clean in {"blog", "post", "artículo"}:
            return f"Leer guía sobre {topic_label}"
        if page_type_clean in {"producto", "servicios"}:
            return f"Descubrir {page_type_clean} de {topic_label}"
        if page_type_clean:
            return f"Más sobre {topic_label} ({page_type_clean})"
        return f"Aprender más sobre {topic_label}"

    if page_type_clean:
        return f"Ver {page_type_clean}"

    return suggest_anchor_from_url(url)

def extract_url_hierarchy(url: str, depth: int = 2) -> str:
    cleaned = str(url).strip()
    if not cleaned:
        return "root"
    cleaned = re.sub(r"https?://[^/]+", "", cleaned)
    segments = [segment for segment in cleaned.split("/") if segment]
    if not segments:
        return "root"
    depth = max(1, int(depth))
    return "/".join(segments[:depth])

def calculate_weighted_entity_overlap(
    source_entities: Dict[str, float],
    target_entities: Dict[str, float],
) -> float:
    if not source_entities or not target_entities:
        return 0.0
    source_keys = set(source_entities.keys())
    target_keys = set(target_entities.keys())
    union_keys = source_keys.union(target_keys)
    if not union_keys:
        return 0.0
    intersection_keys = source_keys.intersection(target_keys)
    weighted_intersection = sum(
        min(source_entities.get(key, 0.0), target_entities.get(key, 0.0)) for key in intersection_keys
    )
    weighted_union = sum(max(source_entities.get(key, 0.0), target_entities.get(key, 0.0)) for key in union_keys)
    if weighted_union == 0.0:
        return 0.0
    return float(weighted_intersection / weighted_union)

def parse_entity_payload(value: object) -> Dict[str, float]:
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items() if isinstance(v, (int, float))}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return {str(k): float(v) for k, v in parsed.items() if isinstance(v, (int, float))}
        if isinstance(parsed, list):
            result: Dict[str, float] = {}
            for item in parsed:
                if isinstance(item, dict) and "id" in item and "score" in item:
                    try:
                        result[str(item["id"])] = float(item["score"])
                    except (TypeError, ValueError):
                        continue
            return result
    return {}

def build_similarity_edges(
    embeddings_norm: np.ndarray,
    urls: Sequence[str],
    min_threshold: float,
    top_k: int = 5,
) -> List[Tuple[str, str, float]]:
    edges: List[Tuple[str, str, float]] = []
    min_threshold = float(min_threshold)
    n = len(urls)
    for idx in range(n):
        sims = embeddings_norm @ embeddings_norm[idx]
        order = np.argsort(sims)[::-1]
        added = 0
        for candidate_idx in order:
            if candidate_idx == idx:
                continue
            score = float(sims[candidate_idx])
            if score < min_threshold:
                break
            edges.append((urls[idx], urls[candidate_idx], max(score, 1e-6)))
            added += 1
            if added >= top_k:
                break
    return edges

def calculate_topical_pagerank(
    df: pd.DataFrame,
    url_column: str,
    type_column: str,
    primary_target_types: Sequence[str],
    graph_edges: Sequence[Tuple[str, str, float]],
    alpha: float = 0.85,
    existing_edges: Optional[Sequence[Tuple[str, str]]] = None,
) -> Dict[str, float]:
    """
    Calcula PageRank temático combinando aristas de similitud semántica con enlaces explícitos del sitio.
    
    Args:
        df: DataFrame con las URLs y tipos de página
        url_column: Nombre de la columna con las URLs
        type_column: Nombre de la columna con los tipos de página
        primary_target_types: Tipos de página considerados prioritarios
        graph_edges: Aristas semánticas (source, target, weight basado en similitud)
        alpha: Factor de damping para PageRank (default 0.85)
        existing_edges: Tuplas (source, target) o (source, target, weight) de enlaces reales existentes
    
    Returns:
        Diccionario {url: pagerank_score}
    """
    urls = df[url_column].astype(str).str.strip().tolist()
    url_set = set(urls)
    graph = nx.DiGraph()
    graph.add_nodes_from(urls)
    
    # 1. Añadir aristas semánticas
    for source, target, weight in graph_edges:
        if source not in url_set or target not in url_set:
            continue
        graph.add_edge(source, target, weight=max(float(weight), 1e-6))

    # 2. Añadir enlaces explícitos existentes con boost de autoridad
    if existing_edges:
        for edge_tuple in existing_edges:
            # Soportar tanto tuplas de 2 (source, target) como de 3 (source, target, weight_multiplier)
            if len(edge_tuple) == 3:
                source, target, weight_multiplier = edge_tuple
                weight_multiplier = float(weight_multiplier)
            elif len(edge_tuple) == 2:
                source, target = edge_tuple
                weight_multiplier = 1.0
            else:
                continue
            
            # Normalizar URLs para matching
            source_clean = str(source).strip()
            target_clean = str(target).strip()
            
            if source_clean not in url_set or target_clean not in url_set:
                continue
                
            # Aplicar peso base (2.0) multiplicado por el factor configurado
            base_weight = 2.0
            final_weight = base_weight * weight_multiplier
                
            if graph.has_edge(source_clean, target_clean):
                # Si ya existe arista semántica, añadir boost adicional
                # Enlace real = evidencia fuerte de relevancia
                graph[source_clean][target_clean]['weight'] += final_weight
            else:
                # Enlace real sin similitud semántica alta
                # Aún así es valioso por la estructura del sitio
                graph.add_edge(source_clean, target_clean, weight=final_weight)

    # 3. Personalización: dar más peso a páginas objetivo prioritarias
    personalization: Dict[str, float] = {}
    primary_set = {str(t).strip() for t in primary_target_types}
    primary_weight = 0.5
    other_weight = 0.05
    for url_value, page_type in zip(urls, df[type_column].astype(str).str.strip()):
        personalization[url_value] = primary_weight if page_type in primary_set else other_weight
    total = sum(personalization.values())
    if not total:
        personalization = {url: 1.0 / len(urls) for url in urls} if urls else {}
    else:
        personalization = {url: weight / total for url, weight in personalization.items()}

    # 4. Calcular PageRank
    try:
        pr_scores = nx.pagerank(
            graph,
            alpha=alpha,
            personalization=personalization if personalization else None,
            weight="weight",
        )
    except nx.NetworkXException:
        pr_scores = {url: 1.0 / len(graph) for url in graph} if graph else {}
    return pr_scores

def guess_default_type(values: Sequence[str], keywords: Sequence[str]) -> Optional[str]:
    """
    Busca el primer valor que contenga alguno de los keywords indicados.
    """
    for keyword in keywords:
        for value in values:
            if keyword in value.lower():
                return value
    return None

def semantic_link_recommendations(
    df: pd.DataFrame,
    url_column: str,
    type_column: str,
    source_types: Sequence[str],
    primary_target_types: Sequence[str],
    secondary_target_types: Optional[Sequence[str]],
    similarity_threshold: float,
    max_links_per_source: int,
    max_primary: int,
    max_secondary: int,
    source_limit: Optional[int] = None,
    selected_source_urls: Optional[Sequence[str]] = None,
    embedding_col: str = "EmbeddingsFloat",
) -> pd.DataFrame:
    """
    Genera recomendaciones de enlazado interno respetando prioridades de tipos de destino.
    """
    required_columns = {url_column, type_column, embedding_col}
    if not required_columns.issubset(df.columns):
        missing = ", ".join(required_columns - set(df.columns))
        raise ValueError(f"Hacen falta las columnas: {missing}")
    if not source_types:
        raise ValueError("Selecciona al menos un tipo de página de origen.")

    df_local = df.copy()
    df_local[url_column] = df_local[url_column].astype(str).str.strip()
    df_local[type_column] = df_local[type_column].astype(str).str.strip()

    embeddings = np.vstack(df_local[embedding_col].values)
    embeddings_norm = normalize(embeddings)

    urls = df_local[url_column].tolist()
    page_types = df_local[type_column].tolist()

    source_set = {str(value).strip() for value in source_types}
    primary_set = {str(value).strip() for value in primary_target_types or []}
    secondary_set = (
        {str(value).strip() for value in secondary_target_types} if secondary_target_types is not None else set()
    )

    source_indices = [idx for idx, page_type in enumerate(page_types) if page_type in source_set]
    if selected_source_urls:
        selected_set = {str(url).strip() for url in selected_source_urls}
        source_indices = [idx for idx in source_indices if urls[idx] in selected_set]

    if source_limit is not None and source_limit > 0:
        source_indices = source_indices[: int(source_limit)]

    columns = [
        "Origen URL",
        "Origen Tipo",
        "Destino Sugerido URL",
        "Destino Tipo",
        "Score Similitud (%)",
        "Acción SEO",
    ]
    if not source_indices:
        return pd.DataFrame(columns=columns)

    recommendations: List[Dict[str, str]] = []
    total_rows = len(df_local)

    for src_idx in source_indices:
        similarities = embeddings_norm @ embeddings_norm[src_idx]
        candidate_indices = [
            idx
            for idx in range(total_rows)
            if idx != src_idx and similarities[idx] >= similarity_threshold and urls[idx] != urls[src_idx]
        ]
        if not candidate_indices:
            continue

        candidate_indices.sort(key=lambda idx: float(similarities[idx]), reverse=True)
        primary_candidates = [idx for idx in candidate_indices if page_types[idx] in primary_set] if primary_set else []
        primary_idx_set = set(primary_candidates)
        secondary_candidates = (
            [idx for idx in candidate_indices if idx not in primary_idx_set and page_types[idx] in secondary_set]
            if secondary_set
            else []
        )
        secondary_idx_set = set(secondary_candidates)
        fallback_candidates = [
            idx for idx in candidate_indices if idx not in primary_idx_set and idx not in secondary_idx_set
        ]

        selected_pairs: List[Tuple[int, str]] = []
        used_indices: set[int] = set()

        def extend(indices: Sequence[int], limit: Optional[int], label: str) -> None:
            if limit is not None and limit <= 0:
                return
            taken = 0
            for idx in indices:
                if len(selected_pairs) >= max_links_per_source:
                    break
                if idx in used_indices:
                    continue
                selected_pairs.append((idx, label))
                used_indices.add(idx)
                taken += 1
                if limit is not None and taken >= limit:
                    break

        if primary_candidates:
            extend(primary_candidates, int(max_primary), "Objetivo prioritario")
        if secondary_candidates:
            extend(secondary_candidates, int(max_secondary), "Cluster complementario")
        if len(selected_pairs) < max_links_per_source:
            extend(fallback_candidates, max_links_per_source - len(selected_pairs), "Exploración")

        if not selected_pairs:
            continue

        for candidate_idx, action_label in selected_pairs:
            score = float(similarities[candidate_idx]) * 100.0
            recommendations.append(
                {
                    "Origen URL": urls[src_idx],
                    "Origen Tipo": page_types[src_idx],
                    "Destino Sugerido URL": urls[candidate_idx],
                    "Destino Tipo": page_types[candidate_idx],
                    "Score Similitud (%)": round(score, 2),
                    "Acción SEO": action_label,
                }
            )

    if not recommendations:
        return pd.DataFrame(columns=columns)

    return (
        pd.DataFrame(recommendations)
        .sort_values(["Origen URL", "Score Similitud (%)"], ascending=[True, False])
        .reset_index(drop=True)
    )

def advanced_semantic_linking(
    df: pd.DataFrame,
    url_column: str,
    type_column: str,
    source_types: Sequence[str],
    primary_target_types: Sequence[str],
    secondary_target_types: Optional[Sequence[str]],
    similarity_threshold: float,
    max_links_per_source: int,
    max_primary: int,
    max_secondary: int,
    silo_depth: int,
    silo_boost: float,
    embedding_col: str = "EmbeddingsFloat",
    source_limit: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Variante avanzada que añade señal de arquitectura (silos) y reporte de páginas huérfanas.
    """
    required_columns = {url_column, type_column, embedding_col}
    if not required_columns.issubset(df.columns):
        missing = ", ".join(required_columns - set(df.columns))
        raise ValueError(f"Faltan columnas necesarias: {missing}")
    if not source_types:
        raise ValueError("Debes seleccionar tipos de origen.")
    if not primary_target_types:
        raise ValueError("Debes seleccionar al menos un tipo destino prioritario.")

    df_local = df.copy()
    df_local[url_column] = df_local[url_column].astype(str).str.strip()
    df_local[type_column] = df_local[type_column].astype(str).str.strip()
    df_local["Silo"] = df_local[url_column].apply(lambda url: extract_url_silo(url, depth=silo_depth))
    df_local["SuggestedAnchor"] = df_local.apply(
        lambda row: generate_contextual_anchor(row[type_column], row["Silo"], row[url_column]),
        axis=1,
    )

    embeddings = np.vstack(df_local[embedding_col].values)
    embeddings_norm = normalize(embeddings)

    urls = df_local[url_column].tolist()
    page_types = df_local[type_column].tolist()
    silos = df_local["Silo"].tolist()
    anchors = df_local["SuggestedAnchor"].tolist()
    total_rows = len(df_local)
    url_type_map: Dict[str, str] = {}
    for url_value, page_type in zip(urls, page_types):
        url_type_map.setdefault(url_value, page_type)

    source_set = {str(t).strip() for t in source_types}
    primary_set = {str(t).strip() for t in primary_target_types}
    secondary_set = (
        {str(t).strip() for t in secondary_target_types} if secondary_target_types is not None else set()
    )
    allowed_target_types = primary_set.union(secondary_set) if secondary_set else set(primary_set)

    source_indices = [idx for idx, page_type in enumerate(page_types) if page_type in source_set]
    if source_limit is not None and source_limit > 0:
        source_indices = source_indices[: int(source_limit)]

    max_primary = min(max_primary, max_links_per_source)
    max_secondary = min(max_secondary, max_links_per_source)

    recommendations: List[Dict[str, object]] = []
    target_counts: Dict[str, int] = {url: 0 for url in urls}

    for src_idx in source_indices:
        source_url = urls[src_idx]
        source_type = page_types[src_idx]
        source_silo = silos[src_idx]

        similarities = embeddings_norm @ embeddings_norm[src_idx]
        same_silo_mask = np.array([1.0 if silo == source_silo else 0.0 for silo in silos])
        boosted = similarities + same_silo_mask * float(silo_boost)

        candidate_indices = [
            idx
            for idx in range(total_rows)
            if idx != src_idx
            and boosted[idx] >= similarity_threshold
            and urls[idx] != source_url
            and page_types[idx] in allowed_target_types
        ]
        if not candidate_indices:
            continue

        candidate_indices.sort(key=lambda i: float(boosted[i]), reverse=True)

        primary_candidates = [idx for idx in candidate_indices if page_types[idx] in primary_set]
        primary_idx_set = set(primary_candidates)
        secondary_candidates = (
            [idx for idx in candidate_indices if idx not in primary_idx_set and page_types[idx] in secondary_set]
            if secondary_set
            else []
        )
        secondary_idx_set = set(secondary_candidates)
        fallback_candidates = [
            idx for idx in candidate_indices if idx not in primary_idx_set and idx not in secondary_idx_set
        ]

        selected_pairs: List[Tuple[int, str]] = []
        used_indices: set[int] = set()

        def extend(indices: Sequence[int], limit: Optional[int], label: str) -> None:
            if limit is not None and limit <= 0:
                return
            taken = 0
            for idx in indices:
                if len(selected_pairs) >= max_links_per_source:
                    break
                if idx in used_indices:
                    continue
                selected_pairs.append((idx, label))
                used_indices.add(idx)
                taken += 1
                if limit is not None and taken >= limit:
                    break

        extend(primary_candidates, int(max_primary), "Silo vertical / Money page")
        if len(selected_pairs) < max_links_per_source and secondary_candidates:
            extend(secondary_candidates, int(max_secondary), "Cluster relacionado")
        if len(selected_pairs) < max_links_per_source and fallback_candidates:
            extend(fallback_candidates, max_links_per_source - len(selected_pairs), "Exploración semántica")

        if not selected_pairs:
            continue

        for candidate_idx, action_label in selected_pairs:
            target_counts[urls[candidate_idx]] += 1
            final_score = float(boosted[candidate_idx]) * 100.0
            base_score = float(similarities[candidate_idx]) * 100.0
            recommendations.append(
                {
                    "Origen URL": source_url,
                    "Origen Tipo": source_type,
                    "Origen Silo": source_silo,
                    "Destino URL": urls[candidate_idx],
                    "Destino Tipo": page_types[candidate_idx],
                    "Destino Silo": silos[candidate_idx],
                    "Anchor Text Sugerido": anchors[candidate_idx],
                    "Score Base (%)": round(base_score, 2),
                    "Score Final (%)": round(final_score, 2),
                    "Boost Aplicado": round(float(final_score - base_score), 2),
                    "Estrategia": action_label
                    if source_silo != silos[candidate_idx]
                    else ("Silo reforzado" if "Money" in action_label else action_label),
                }
            )

    if recommendations:
        report_df = (
            pd.DataFrame(recommendations)
            .sort_values(["Origen URL", "Score Final (%)"], ascending=[True, False])
            .reset_index(drop=True)
        )
    else:
        report_df = pd.DataFrame(
            columns=[
                "Origen URL",
                "Origen Tipo",
                "Origen Silo",
                "Destino URL",
                "Destino Tipo",
                "Destino Silo",
                "Anchor Text Sugerido",
                "Score Base (%)",
                "Score Final (%)",
                "Boost Aplicado",
                "Estrategia",
            ]
        )

    orphan_urls = [
        url for url, count in target_counts.items() if count == 0 and url_type_map.get(url) in primary_set
    ]

    return report_df, orphan_urls

def structural_taxonomy_linking(
    df: pd.DataFrame,
    url_column: str,
    hierarchy_column: Optional[str],
    depth: int,
    max_links_per_parent: int,
    include_horizontal: bool,
    link_weight: float,
    use_semantic_priority: bool = False,
    embedding_col: str = "EmbeddingsFloat",
) -> pd.DataFrame:
    """
    Genera recomendaciones de enlaces basadas en la estructura jerárquica de URLs.
    
    Args:
        use_semantic_priority: Si True, ordena hermanos por similitud semántica antes de seleccionarlos
        embedding_col: Columna con embeddings (necesaria si use_semantic_priority=True)
    """
    df_local = df.copy()
    df_local[url_column] = df_local[url_column].astype(str).str.strip()
    if hierarchy_column and hierarchy_column in df_local.columns:
        df_local["HierarchyPath"] = (
            df_local[hierarchy_column].astype(str).str.strip().replace("", "root").fillna("root")
        )
    else:
        df_local["HierarchyPath"] = df_local[url_column].apply(lambda url: extract_url_hierarchy(url, depth=depth))
    df_local["HierarchyPath"] = df_local["HierarchyPath"].replace("", "root")
    df_local["ParentPath"] = df_local["HierarchyPath"].apply(
        lambda path: "/".join(path.split("/")[:-1]) if path and "/" in path else "root"
    )

    path_to_urls: Dict[str, List[str]] = (
        df_local.groupby("HierarchyPath")[url_column].apply(list).to_dict()
        if not df_local.empty
        else {}
    )
    parent_to_children: Dict[str, List[str]] = (
        df_local.groupby("ParentPath")[url_column].apply(list).to_dict()
        if not df_local.empty
        else {}
    )

    recommendations: List[Dict[str, object]] = []

    for _, row in df_local.iterrows():
        source_url = row[url_column]
        parent_path = row["ParentPath"]
        if parent_path != "root":
            parent_candidates = path_to_urls.get(parent_path, [])
            parent_url = parent_candidates[0] if parent_candidates else None
            if parent_url:
                recommendations.append(
                    {
                        "Origen URL": source_url,
                        "Destino URL": parent_url,
                        "Estrategia": "Estructural ascendente (breadcrumb)",
                        "Anchor Text Sugerido": f"Volver a {parent_path.split('/')[-1].replace('-', ' ').title()}",
                        "Link Weight": float(link_weight),
                    }
                )

        if include_horizontal:
            siblings = parent_to_children.get(parent_path, [])
            # Filtrar el propio source_url
            siblings_filtered = [sib for sib in siblings if sib != source_url]
            
            # Priorización semántica opcional
            if use_semantic_priority and embedding_col in df_local.columns and len(siblings_filtered) > 0:
                try:
                    # Obtener embedding de la página origen
                    source_emb_row = df_local.loc[df_local[url_column] == source_url, embedding_col]
                    if not source_emb_row.empty:
                        source_emb = source_emb_row.iloc[0]
                        
                        # Calcular similitud con cada hermano
                        sibling_similarities = []
                        for sib_url in siblings_filtered:
                            sib_emb_row = df_local.loc[df_local[url_column] == sib_url, embedding_col]
                            if not sib_emb_row.empty:
                                sib_emb = sib_emb_row.iloc[0]
                                # Calcular similitud coseno
                                sim = float(np.dot(source_emb, sib_emb) / 
                                          (np.linalg.norm(source_emb) * np.linalg.norm(sib_emb) + 1e-10))
                                sibling_similarities.append((sib_url, sim))
                        
                        # Ordenar por similitud descendente
                        sibling_similarities.sort(key=lambda x: x[1], reverse=True)
                        # Tomar top-k más similares
                        siblings_to_link = [url for url, _ in sibling_similarities[:max_links_per_parent]]
                    else:
                        # Si no hay embedding para source, usar orden original
                        siblings_to_link = siblings_filtered[:max_links_per_parent]
                except Exception:
                    # En caso de error, usar orden original sin priorización
                    siblings_to_link = siblings_filtered[:max_links_per_parent]
            else:
                # Sin priorización semántica, tomar los primeros N
                siblings_to_link = siblings_filtered[:max_links_per_parent]
            
            for sibling_url in siblings_to_link:
                recommendations.append(
                    {
                        "Origen URL": source_url,
                        "Destino URL": sibling_url,
                        "Estrategia": "Estructural horizontal (hermanos)",
                        "Anchor Text Sugerido": generate_contextual_anchor("", parent_path, sibling_url),
                        "Link Weight": float(link_weight) * 0.9,
                    }
                )

    for parent_path, children_urls in parent_to_children.items():
        parent_candidates = path_to_urls.get(parent_path, [])
        parent_url = parent_candidates[0] if parent_candidates else None
        if not parent_url:
            continue
        limited_children = children_urls[: max_links_per_parent]
        for child_url in limited_children:
            recommendations.append(
                {
                    "Origen URL": parent_url,
                    "Destino URL": child_url,
                    "Estrategia": "Estructural descendente (destacados)",
                    "Anchor Text Sugerido": generate_contextual_anchor("", child_url.split("/")[-2] if "/" in child_url else "", child_url),
                    "Link Weight": float(link_weight) * 0.85,
                }
            )

    if not recommendations:
        return pd.DataFrame(
            columns=[
                "Origen URL",
                "Destino URL",
                "Estrategia",
                "Anchor Text Sugerido",
                "Link Weight",
            ]
        )
    return pd.DataFrame(recommendations).drop_duplicates(
        subset=["Origen URL", "Destino URL", "Estrategia"]
    )

def hybrid_semantic_linking(
    df: pd.DataFrame,
    url_column: str,
    type_column: str,
    entity_column: str,
    source_types: Sequence[str],
    primary_target_types: Sequence[str],
    similarity_threshold: float,
    max_links_per_source: int,
    max_primary: int,
    decay_factor: float,
    weights: Optional[Dict[str, float]] = None,
    embedding_col: str = "EmbeddingsFloat",
    source_limit: Optional[int] = None,
    top_k_edges: int = 5,
    existing_edges: Optional[Sequence[Tuple[str, str]]] = None,
) -> Tuple[pd.DataFrame, List[str], Dict[str, float]]:
    """
    Genera recomendaciones de enlaces internos combinando similitud semántica, 
    autoridad (PageRank) y solapamiento de entidades.
    
    Args:
        existing_edges: Lista de tuplas (source_url, target_url) de enlaces que ya existen,
                       para evitar recomendarlos nuevamente y mejorar el cálculo de PageRank
    """
    required_columns = {url_column, type_column, embedding_col, entity_column}
    if not required_columns.issubset(df.columns):
        missing = ", ".join(required_columns - set(df.columns))
        raise ValueError(f"Faltan columnas necesarias: {missing}")

    weights = weights or {"semantic": 0.4, "authority": 0.35, "entity_overlap": 0.25}
    weight_sem = max(0.0, float(weights.get("semantic", 0.4)))
    weight_auth = max(0.0, float(weights.get("authority", 0.35)))
    weight_ent = max(0.0, float(weights.get("entity_overlap", 0.25)))
    total_weights = weight_sem + weight_auth + weight_ent
    if total_weights == 0:
        weight_sem = 0.4
        weight_auth = 0.35
        weight_ent = 0.25
        total_weights = 1.0
    weight_sem /= total_weights
    weight_auth /= total_weights
    weight_ent /= total_weights

    df_local = df.copy()
    df_local[url_column] = df_local[url_column].astype(str).str.strip()
    df_local[type_column] = df_local[type_column].astype(str).str.strip()

    entity_maps: List[Dict[str, float]] = df_local[entity_column].apply(parse_entity_payload).tolist()
    embeddings = np.vstack(df_local[embedding_col].values)
    embeddings_norm = normalize(embeddings)
    urls = df_local[url_column].tolist()
    page_types = df_local[type_column].tolist()
    total_rows = len(df_local)

    similarity_edges = build_similarity_edges(
        embeddings_norm=embeddings_norm,
        urls=urls,
        min_threshold=similarity_threshold,
        top_k=top_k_edges,
    )
    
    # Pasar enlaces existentes al cálculo de PageRank para mejor precisión
    pagerank_scores = calculate_topical_pagerank(
        df_local,
        url_column=url_column,
        type_column=type_column,
        primary_target_types=primary_target_types,
        graph_edges=similarity_edges,
        alpha=0.85,
        existing_edges=existing_edges,
    )
    max_pr = max(pagerank_scores.values()) if pagerank_scores else 1.0
    pr_norm = {url: score / max_pr for url, score in pagerank_scores.items()} if max_pr else pagerank_scores

    source_set = {str(t).strip() for t in source_types}
    primary_set = {str(t).strip() for t in primary_target_types}
    source_indices = [idx for idx, page_type in enumerate(page_types) if page_type in source_set]
    if source_limit is not None and source_limit > 0:
        source_indices = source_indices[: int(source_limit)]

    recommendations: List[Dict[str, object]] = []
    target_link_counts: Dict[str, int] = {url: 0 for url in urls}
    
    # Crear conjunto de enlaces existentes para filtrado rápido
    existing_links_set: Set[Tuple[str, str]] = set()
    if existing_edges:
        for edge_tuple in existing_edges:
            # Soportar tanto tuplas de 2 (source, target) como de 3 (source, target, weight)
            if len(edge_tuple) == 3:
                src, tgt, _ = edge_tuple  # Ignorar peso aquí, solo para filtrar duplicados
            elif len(edge_tuple) == 2:
                src, tgt = edge_tuple
            else:
                continue
            existing_links_set.add((str(src).strip(), str(tgt).strip()))

    for src_idx in source_indices:
        source_url = urls[src_idx]
        source_entities = entity_maps[src_idx]
        similarities = embeddings_norm @ embeddings_norm[src_idx]
        candidate_scores: List[Tuple[int, float, float, float]] = []
        for cand_idx in range(total_rows):
            if cand_idx == src_idx:
                continue
            target_url = urls[cand_idx]
            
            # Filtrar enlaces que ya existen
            if (source_url, target_url) in existing_links_set:
                continue
                
            semantic_score = float((similarities[cand_idx] + 1.0) / 2.0)
            if semantic_score < similarity_threshold:
                continue
            target_entities = entity_maps[cand_idx]
            authority_score = pr_norm.get(target_url, 0.0)
            entity_overlap = calculate_weighted_entity_overlap(source_entities, target_entities)
            composite_score = (
                weight_sem * semantic_score + weight_auth * authority_score + weight_ent * entity_overlap
            )
            candidate_scores.append((cand_idx, composite_score, semantic_score, entity_overlap))

        if not candidate_scores:
            continue

        candidate_scores.sort(key=lambda item: item[1], reverse=True)
        selected_pairs: List[Tuple[int, float, float, float, str, float]] = []
        used_indices: Set[int] = set()

        def apply_selection(
            candidates: List[Tuple[int, float, float, float]],
            limit: Optional[int],
            label: str,
        ) -> None:
            if limit is not None and limit <= 0:
                return
            taken = 0
            for cand_idx, cls_raw, semantic_value, entity_value in candidates:
                if len(selected_pairs) >= max_links_per_source:
                    break
                if cand_idx in used_indices:
                    continue
                target_url = urls[cand_idx]
                current_count = target_link_counts.get(target_url, 0)
                decay_penalty = math.log(1 + max(decay_factor, 0.0) * current_count) if decay_factor > 0 else 0.0
                adjusted_cls = max(0.0, cls_raw - decay_penalty)
                if adjusted_cls <= 0:
                    continue
                selected_pairs.append(
                    (cand_idx, adjusted_cls, cls_raw, semantic_value, entity_value, label)
                )
                used_indices.add(cand_idx)
                target_link_counts[target_url] = current_count + 1
                taken += 1
                if limit is not None and taken >= limit:
                    break

        primary_candidates = [item for item in candidate_scores if page_types[item[0]] in primary_set]
        secondary_candidates = [item for item in candidate_scores if page_types[item[0]] not in primary_set]
        apply_selection(primary_candidates, int(max_primary), "Objetivo prioritario (CLS)")
        remaining_limit = max_links_per_source - len(selected_pairs)
        apply_selection(secondary_candidates, remaining_limit, "Exploración semántica (CLS)")

        for cand_idx, adjusted_cls, cls_raw, semantic_val, entity_val, label in selected_pairs:
            target_url = urls[cand_idx]
            recommendations.append(
                {
                    "Origen URL": source_url,
                    "Origen Tipo": page_types[src_idx],
                    "Destino URL": target_url,
                    "Destino Tipo": page_types[cand_idx],
                    "Anchor Text Sugerido": generate_contextual_anchor(
                        page_types[cand_idx],
                        extract_url_silo(target_url),
                        target_url,
                    ),
                    "Score Semántico (%)": round(semantic_val * 100.0, 2),
                    "Score Entidades (%)": round(entity_val * 100.0, 2),
                    "Score Autoridad (PR)": round(pr_norm.get(target_url, 0.0), 4),
                    "CLS Ajustado (%)": round(adjusted_cls * 100.0, 2),
                    "CLS Base (%)": round(cls_raw * 100.0, 2),
                    "Estrategia": label,
                }
            )

    if recommendations:
        report_df = (
            pd.DataFrame(recommendations)
            .sort_values("CLS Ajustado (%)", ascending=False)
            .reset_index(drop=True)
        )
    else:
        report_df = pd.DataFrame(
            columns=[
                "Origen URL",
                "Origen Tipo",
                "Destino URL",
                "Destino Tipo",
                "Anchor Text Sugerido",
                "Score Semántico (%)",
                "Score Entidades (%)",
                "Score Autoridad (PR)",
                "CLS Ajustado (%)",
                "CLS Base (%)",
                "Estrategia",
            ]
        )

    orphan_urls: List[str] = []
    for url, count in target_link_counts.items():
        if count != 0:
            continue
        type_values = df_local.loc[df_local[url_column] == url, type_column]
        if type_values.empty:
            continue
        if type_values.iloc[0] in primary_set:
            orphan_urls.append(url)

    return report_df, orphan_urls, pagerank_scores

def build_entity_payload_from_doc_relations(doc_relations_df: Optional[pd.DataFrame]) -> Dict[str, Dict[str, float]]:
    """
    Convierte el resumen pagina-entidad en un diccionario {url: {entidad: prominence}} para reutilizarlo en el laboratorio.
    """
    payload: Dict[str, Dict[str, float]] = {}
    if doc_relations_df is None or doc_relations_df.empty:
        return payload

    for _, row in doc_relations_df.iterrows():
        url_value = str(row.get("URL", "")).strip()
        if not url_value:
            continue
        entity_id = str(row.get("QID") or "").strip()
        if not entity_id or entity_id in {"â€”", "-", "None", "nan"}:
            entity_id = str(row.get("Entidad") or "").strip()
        if not entity_id:
            continue
        prominence = row.get("Prominence documento")
        try:
            prominence_value = float(prominence)
        except (TypeError, ValueError):
            freq_value = row.get("Frecuencia documento", 0)
            try:
                prominence_value = float(freq_value)
            except (TypeError, ValueError):
                prominence_value = 0.0
        if prominence_value <= 0:
            continue
        url_bucket = payload.setdefault(url_value, {})
        url_bucket[entity_id] = url_bucket.get(entity_id, 0.0) + prominence_value
    return payload

def build_linking_reports_payload(max_rows: int = 40) -> Dict[str, object]:
    """
    Recopila los diferentes dataframes del laboratorio de enlazado para resumirlos con Gemini.
    Se limita el numero de filas para no saturar el prompt.
    """
    payload: Dict[str, object] = {}

    def _add(tag: str, df: Optional[pd.DataFrame]) -> None:
        if isinstance(df, pd.DataFrame) and not df.empty:
            payload[tag] = {
                "total_rows": int(len(df)),
                "columns": list(df.columns),
                "sample": df.head(max_rows).to_dict("records"),
            }

    _add("basic", st.session_state.get("linking_basic_report"))
    _add("advanced", st.session_state.get("linking_adv_report"))
    _add("hybrid_cls", st.session_state.get("linking_hybrid_report"))
    _add("structural", st.session_state.get("linking_structural_report"))

    adv_orphans = st.session_state.get("linking_adv_orphans") or []
    if adv_orphans:
        payload.setdefault("advanced", {}).update({"orphans": adv_orphans[:max_rows]})
    hybrid_orphans = st.session_state.get("linking_hybrid_orphans") or []
    if hybrid_orphans:
        payload.setdefault("hybrid_cls", {}).update({"orphans": hybrid_orphans[:max_rows]})
    pagerank_scores = st.session_state.get("linking_hybrid_pagerank") or {}
    if pagerank_scores:
        sorted_scores = sorted(pagerank_scores.items(), key=lambda item: item[1], reverse=True)[:max_rows]
        payload.setdefault("hybrid_cls", {}).update({"pagerank": sorted_scores})

    return payload

def interpret_linking_reports_with_gemini(
    api_key: str,
    model_name: str,
    payload: Dict[str, object],
    extra_notes: str = "",
) -> str:
    """
    Envía a Gemini un resumen de los modos del laboratorio para obtener conclusiones de alto nivel.
    """
    if genai is None:
        raise RuntimeError("Instala `google-generativeai` para usar la interpretacion con Gemini.")
    cleaned_key = (api_key or "").strip()
    if not cleaned_key:
        raise ValueError("Introduce una API key valida de Gemini (panel lateral).")
    cleaned_model = (model_name or "gemini-2.5-flash").strip()
    if not payload:
        raise ValueError("No hay resultados que interpretar todavia.")

    genai.configure(api_key=cleaned_key)
    model = genai.GenerativeModel(cleaned_model)
    payload_json = json.dumps(payload, ensure_ascii=False)
    prompt = f"""
Eres un consultor SEO especializado en enlazado interno. Resume los hallazgos del laboratorio
de enlazado y genera conclusiones accionables.

Resultados (JSON recortado):
{payload_json}

Notas adicionales del usuario: {extra_notes or "Sin observaciones"}.

Redacta en español:
- Insight general por cada modo evaluado (basico, avanzado, hibrido, estructural) si hay datos.
- Riesgos detectados (por ejemplo, falta de enlaces, tipos sin cobertura, competidores fuertes).
- Acciones recomendadas para el siguiente sprint.
"""
    response = model.generate_content(prompt)
    text = getattr(response, "text", "") or ""
    if not text and response.candidates:
        text = "".join(
            getattr(part, "text", "")
            for part in response.candidates[0].content.parts
        ).strip()
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*", "", text).strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    return text

__all__ = [
    "render_linking_lab",
    "detect_embedding_columns",
    "preprocess_embeddings",
    "detect_url_columns",
    "detect_page_type_columns",
    "build_entity_payload_from_doc_relations",
    "build_linking_reports_payload",
    "interpret_linking_reports_with_gemini",
]
