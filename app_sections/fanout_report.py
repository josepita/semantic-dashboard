"""
Fan-Out Report - UI principal para el hub de Embedding Insights.

Este módulo proporciona la función render_fanout_report() que se integra
en streamlit_app.py para ofrecer análisis de fan-out queries desde el hub principal.
"""
from __future__ import annotations

import ast
from io import BytesIO
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# Importar módulos de fanout_analyzer
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()
fanout_modules = project_root / "apps" / "fanout_analyzer" / "modules"
if str(fanout_modules) not in sys.path:
    sys.path.insert(0, str(fanout_modules))

from fanout_extraction import extract_fanout_queries, fanout_results_to_dataframe
from chatgpt_import import parse_chatgpt_fanout_csv
from domain_coverage import (
    analyze_domain_coverage,
    DEFAULT_THRESHOLDS,
    CLASSIFICATION_LABELS,
    CLASSIFICATION_ORDER,
)


def render_fanout_report() -> None:
    """Renderiza la interfaz completa de Fan-Out Query Analyzer."""
    st.header("🔍 Fan-Out Query Analyzer")
    st.markdown(
        "Extrae fan-out queries de Gemini y ChatGPT, y analiza la cobertura "
        "de tu dominio usando similitud semántica con embeddings."
    )

    tabs = st.tabs([
        "🏠 Inicio",
        "🌐 Extracción Gemini",
        "💬 Importar ChatGPT",
        "📊 Análisis Cobertura",
    ])

    with tabs[0]:
        _render_intro()

    with tabs[1]:
        _render_gemini_extraction()

    with tabs[2]:
        _render_chatgpt_import()

    with tabs[3]:
        _render_coverage_analysis()


def _render_intro() -> None:
    """Renderiza la introducción y explicación del flujo."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 ¿Qué son las Fan-Out Queries?")
        st.markdown("""
        Cuando las IAs responden preguntas complejas, generan internamente
        **múltiples consultas de búsqueda**. Estas revelan:

        - **Qué busca la IA** para responder
        - **Subtemas** relevantes
        - **Gaps** en tu contenido
        """)

    with col2:
        st.markdown("### 🚀 Flujo de Trabajo")
        st.markdown("""
        1. **Extraer queries** via Gemini o ChatGPT
        2. **Subir CSV** de tu sitio con embeddings
        3. **Analizar cobertura** semántica
        4. **Identificar gaps** y oportunidades
        """)

    st.markdown("---")
    st.markdown("### 📊 Clasificaciones")

    cols = st.columns(5)
    icons = ["🟢", "🔵", "🟡", "🟠", "🔴"]
    labels = list(CLASSIFICATION_LABELS.values())
    thresholds = ["≥80%", "65-80%", "45-65%", "25-45%", "<25%"]

    for col, icon, label, thresh in zip(cols, icons, labels, thresholds):
        with col:
            st.markdown(f"**{icon} {label}**")
            st.caption(f"Sim: {thresh}")


def _render_gemini_extraction() -> None:
    """Renderiza la sección de extracción via Gemini."""
    api_key = st.session_state.get("gemini_api_key", "")
    model_id = st.session_state.get("gemini_model_name", "gemini-2.5-flash")

    tab1, tab2 = st.tabs(["🚀 Extraer via API", "📥 Importar CSV Previo"])

    with tab1:
        if not api_key:
            st.warning("⚠️ Configura tu API key de Gemini en el sidebar")
            return

        st.info(f"Modelo: `{model_id}`")

        col1, col2 = st.columns([3, 1])

        with col1:
            prompts_text = st.text_area(
                "Prompts (uno por línea):",
                height=150,
                placeholder="¿Cuál es el mejor CRM para pequeñas empresas?\n...",
                key="hub_fanout_prompts",
            )

        with col2:
            delay = st.slider("Delay (s)", 0.1, 3.0, 0.5, 0.1, key="hub_fanout_delay")

        prompts = [p.strip() for p in prompts_text.split("\n") if p.strip()]
        st.caption(f"{len(prompts)} prompts a procesar")

        if st.button("🚀 Extraer Fan-Out", disabled=len(prompts) == 0, key="hub_fanout_extract"):
            progress = st.progress(0)
            status = st.empty()

            def cb(cur, tot, prompt):
                progress.progress((cur + 1) / tot)
                status.text(f"{cur + 1}/{tot}: {prompt[:40]}...")

            results = extract_fanout_queries(
                prompts=prompts,
                api_key=api_key,
                model_id=model_id,
                delay=delay,
                progress_callback=cb,
            )

            progress.empty()
            status.empty()

            df = fanout_results_to_dataframe(results)
            _merge_queries_to_session(df)

            successes = df[df["error"] == ""]
            st.success(f"✅ {len(successes)} queries extraídas")

    with tab2:
        st.markdown("**Importa un CSV con fan-out queries ya extraídas.**")
        st.caption("Columnas esperadas: `prompt`, `web_search_query`")

        uploaded = st.file_uploader("CSV/Excel con extracción previa", type=["csv", "xlsx", "xls"], key="hub_import_prev")

        if uploaded:
            try:
                filename = uploaded.name.lower()
                if filename.endswith(".csv"):
                    import_df = pd.read_csv(uploaded)
                else:
                    import_df = pd.read_excel(uploaded)

                st.success(f"✅ {len(import_df)} filas")
                st.dataframe(import_df.head(5), use_container_width=True)

                cols = import_df.columns.tolist()
                cols_lower = {c.lower().strip(): c for c in cols}

                prompt_col = cols_lower.get("prompt") or cols_lower.get("user query") or cols[0]
                query_col = cols_lower.get("web_search_query") or cols_lower.get("query") or (cols[1] if len(cols) > 1 else cols[0])

                c1, c2 = st.columns(2)
                with c1:
                    prompt_col = st.selectbox("Columna Prompt", cols, index=cols.index(prompt_col) if prompt_col in cols else 0, key="hub_imp_prompt")
                with c2:
                    query_col = st.selectbox("Columna Query", cols, index=cols.index(query_col) if query_col in cols else 0, key="hub_imp_query")

                if st.button("➕ Importar", key="hub_import_prev_btn"):
                    rows = []
                    for _, row in import_df.iterrows():
                        p = str(row.get(prompt_col, "")).strip()
                        q = str(row.get(query_col, "")).strip()
                        if p and q and p != "nan" and q != "nan":
                            rows.append({"prompt": p, "web_search_query": q, "source": "gemini_import", "error": ""})

                    if rows:
                        new_df = pd.DataFrame(rows)
                        new_df["query_index"] = new_df.groupby("prompt").cumcount()
                        _merge_queries_to_session(new_df)
                        st.success(f"✅ {len(rows)} queries importadas")
                        st.rerun()
                    else:
                        st.warning("No se encontraron datos válidos")

            except Exception as e:
                st.error(f"Error: {e}")

    _show_current_queries("gemini")


def _render_chatgpt_import() -> None:
    """Renderiza la sección de importación desde ChatGPT."""
    with st.expander("📖 Instrucciones del Bookmarklet", expanded=False):
        st.markdown("""
        1. Ejecuta el bookmarklet en una conversación de ChatGPT
        2. Marca "Queries" en el selector de columnas
        3. Click "Export Selected" → descarga `Queries_Report.csv`
        4. Sube el CSV aquí
        """)

    uploaded = st.file_uploader("CSV del bookmarklet", type=["csv"], key="hub_chatgpt_csv")

    if uploaded:
        try:
            df = parse_chatgpt_fanout_csv(uploaded)
            st.success(f"✅ {len(df)} queries de {df['prompt'].nunique()} prompts")
            st.dataframe(df.head(10), use_container_width=True)

            if st.button("➕ Añadir a sesión", key="hub_chatgpt_add"):
                _merge_queries_to_session(df)
                st.success("Queries añadidas")
                st.rerun()

        except Exception as e:
            st.error(f"Error: {e}")

    _show_current_queries("chatgpt")


def _render_coverage_analysis() -> None:
    """Renderiza la sección de análisis de cobertura."""
    queries_df = st.session_state.get("fanout_queries_df")

    if queries_df is None or queries_df.empty:
        st.warning("⚠️ Primero extrae o importa fan-out queries")
        return

    valid_queries = queries_df[queries_df["error"] == ""] if "error" in queries_df.columns else queries_df
    st.info(f"📝 {len(valid_queries)} queries para analizar")

    st.subheader("1. Archivo del Sitio")

    uploaded = st.file_uploader("CSV o Excel con URLs + Embeddings", type=["csv", "xlsx", "xls"], key="hub_site_file")

    if uploaded:
        try:
            filename = uploaded.name.lower()
            if filename.endswith(".csv"):
                site_df = pd.read_csv(uploaded)
            elif filename.endswith((".xlsx", ".xls")):
                site_df = pd.read_excel(uploaded)
            else:
                st.error("Formato no soportado")
                return

            st.success(f"✅ {len(site_df)} filas cargadas")

            with st.expander("📋 Preview", expanded=False):
                st.dataframe(site_df.head(5), use_container_width=True)

            cols = site_df.columns.tolist()
            url_cands = [c for c in cols if any(x in c.lower() for x in ["url", "address", "link", "page"])]
            emb_cands = [c for c in cols if any(x in c.lower() for x in ["embed", "vector", "emb"])]
            title_cands = [c for c in cols if any(x in c.lower() for x in ["title", "titulo", "h1"])]

            st.markdown("**Selecciona las columnas:**")
            c1, c2, c3 = st.columns(3)
            with c1:
                url_col = st.selectbox("Columna URL *", cols, index=cols.index(url_cands[0]) if url_cands else 0, key="hub_url_col")
            with c2:
                emb_col = st.selectbox("Columna Embeddings *", cols, index=cols.index(emb_cands[0]) if emb_cands else 0, key="hub_emb_col")
            with c3:
                title_opts = ["(No usar)"] + cols
                title_idx = title_opts.index(title_cands[0]) if title_cands and title_cands[0] in title_opts else 0
                title_col = st.selectbox("Columna Título (opcional)", title_opts, index=title_idx, key="hub_title_col")

            st.session_state["fanout_url_column"] = url_col
            st.session_state["fanout_title_column"] = title_col if title_col != "(No usar)" else None

            if site_df[emb_col].dtype == object:
                def parse_emb(x):
                    if isinstance(x, str):
                        try:
                            return np.array(ast.literal_eval(x))
                        except:
                            return None
                    return x
                site_df["EmbeddingsFloat"] = site_df[emb_col].apply(parse_emb)
                site_df = site_df.dropna(subset=["EmbeddingsFloat"])
            else:
                site_df["EmbeddingsFloat"] = site_df[emb_col]

            st.session_state["fanout_site_df"] = site_df
            st.caption(f"{len(site_df)} URLs con embeddings válidos")

        except Exception as e:
            st.error(f"Error: {e}")
            return

    site_df = st.session_state.get("fanout_site_df")
    url_col = st.session_state.get("fanout_url_column")

    if site_df is None:
        return

    st.markdown("---")
    st.subheader("2. Analizar")

    if st.button("🔍 Analizar Cobertura", key="hub_analyze"):
        with st.spinner("Analizando..."):
            try:
                detail_df, summary_df = analyze_domain_coverage(
                    queries_df=valid_queries,
                    site_df=site_df,
                    url_column=url_col,
                    embedding_col="EmbeddingsFloat",
                )
                st.session_state["fanout_coverage_detail"] = detail_df
                st.session_state["fanout_coverage_summary"] = summary_df
                st.success("✅ Análisis completado")
            except Exception as e:
                st.error(f"Error: {e}")
                return

    detail_df = st.session_state.get("fanout_coverage_detail")
    summary_df = st.session_state.get("fanout_coverage_summary")

    if detail_df is not None and summary_df is not None:
        _render_results(detail_df, summary_df)


def _render_results(detail_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    """Renderiza los resultados del análisis."""
    st.markdown("---")
    st.subheader("📈 Resultados")

    c1, c2 = st.columns([1, 2])

    with c1:
        icons = {"perfect_coverage": "🟢", "aligned": "🔵", "related_gap": "🟡", "clear_gap": "🟠", "no_coverage": "🔴"}
        for _, row in summary_df.iterrows():
            icon = icons.get(row["classification"], "⚪")
            st.metric(f"{icon} {row['label']}", f"{row['count']} ({row['percentage']}%)")

    with c2:
        import plotly.express as px
        fig = px.pie(
            summary_df, values="count", names="label",
            title="Distribución de Cobertura",
            color="classification",
            color_discrete_map={
                "perfect_coverage": "#22c55e", "aligned": "#3b82f6",
                "related_gap": "#eab308", "clear_gap": "#f97316", "no_coverage": "#ef4444",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Detalle")

    filter_cls = st.multiselect(
        "Filtrar:", list(CLASSIFICATION_LABELS.values()), default=list(CLASSIFICATION_LABELS.values()), key="hub_filter"
    )
    filtered = detail_df[detail_df["classification_label"].isin(filter_cls)]
    st.dataframe(filtered[["prompt", "web_search_query", "best_url", "similarity", "classification_label"]], use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("📥 CSV Detalle", detail_df.to_csv(index=False).encode(), "fanout_detail.csv", "text/csv")
    with c2:
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            summary_df.to_excel(w, sheet_name="Resumen", index=False)
            detail_df.to_excel(w, sheet_name="Detalle", index=False)
        st.download_button("📥 Excel Completo", buf.getvalue(), "fanout_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def _merge_queries_to_session(df: pd.DataFrame) -> None:
    """Añade queries al DataFrame de sesión."""
    existing = st.session_state.get("fanout_queries_df")
    if existing is not None and not existing.empty:
        df = pd.concat([existing, df], ignore_index=True)
    st.session_state["fanout_queries_df"] = df


def _show_current_queries(source: Optional[str] = None) -> None:
    """Muestra las queries actuales en sesión."""
    df = st.session_state.get("fanout_queries_df")
    if df is None or df.empty:
        return

    st.markdown("---")
    st.caption("**Queries en sesión:**")

    if source:
        filtered = df[df["source"] == source]
        st.caption(f"{len(filtered)} de {source}")
    else:
        st.caption(f"Total: {len(df)}")

    if st.button("🗑️ Limpiar queries", key=f"hub_clear_{source or 'all'}"):
        st.session_state["fanout_queries_df"] = None
        st.rerun()


__all__ = ["render_fanout_report"]
