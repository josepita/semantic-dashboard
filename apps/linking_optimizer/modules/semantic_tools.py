from __future__ import annotations

import io
import os
import re
import textwrap
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import trafilatura
from bs4 import BeautifulSoup
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from urllib.parse import parse_qs, unquote, urlparse

# Note: Authority Gap and Google KG features disabled in standalone app
# from modules.authority_advance import (
#     AuthorityGapResult,
#     run_authority_gap_from_embeddings,
#     run_authority_gap_simulation,
# )
# from modules.google_kg import ensure_google_kg_api_key, query_google_enterprise_kg

DEFAULT_SENTENCE_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
MAX_COMPETITOR_CONTENT_CHARS = 8000
MAX_URL_BODY_CHARS = 6000
MAX_URL_DESCRIPTION_CHARS = 600


def parse_line_input(raw_text: str, separators: Tuple[str, ...] = ("\n", ";")) -> List[str]:
    if not raw_text:
        return []
    normalized = raw_text
    for separator in separators:
        if separator != "\n":
            normalized = normalized.replace(separator, "\n")
    return [value.strip() for value in normalized.splitlines() if value.strip()]


def _strip_prefix(value: str, prefixes: Tuple[str, ...]) -> str:
    value_trimmed = value.strip()
    lower_value = value_trimmed.lower()
    for prefix in prefixes:
        if lower_value.startswith(prefix):
            return value_trimmed[len(prefix) :].strip()
    return value_trimmed


def parse_faq_blocks(raw_text: str) -> List[Tuple[str, str]]:
    if not raw_text:
        return []
    blocks = [block for block in re.split(r"\n\s*\n", raw_text.strip()) if block.strip()]
    entries: List[Tuple[str, str]] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        question = _strip_prefix(lines[0], ("pregunta:", "question:", "q:"))
        answer_lines = [_strip_prefix(line, ("respuesta:", "answer:", "a:")) for line in lines[1:]]
        answer = " ".join(answer_lines).strip()
        entries.append((question, answer))
    return entries


def parse_faq_from_excel(uploaded_file, question_col: str, answer_col: str) -> List[Tuple[str, str]]:
    """
    Lee un archivo Excel y extrae FAQs de columnas especificadas.

    Args:
        uploaded_file: Archivo Excel subido por el usuario
        question_col: Nombre de la columna que contiene las preguntas
        answer_col: Nombre de la columna que contiene las respuestas

    Returns:
        Lista de tuplas (pregunta, respuesta)
    """
    try:
        # Leer Excel
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        elif uploaded_file.name.endswith('.xls'):
            df = pd.read_excel(uploaded_file, engine='xlrd')
        else:
            return []

        # Validar columnas
        if question_col not in df.columns or answer_col not in df.columns:
            return []

        # Extraer FAQs (ignorar filas con valores nulos)
        entries: List[Tuple[str, str]] = []
        for _, row in df.iterrows():
            question = str(row[question_col]).strip()
            answer = str(row[answer_col]).strip()

            # Ignorar filas vacías o con 'nan'
            if question and answer and question != 'nan' and answer != 'nan':
                entries.append((question, answer))

        return entries

    except Exception as e:
        st.error(f"Error al leer el archivo Excel: {e}")
        return []


@st.cache_resource(show_spinner=False)
def get_sentence_transformer(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def compute_text_keyword_similarity(
    text: str,
    keywords: List[str],
    model: SentenceTransformer,
) -> pd.DataFrame:
    cleaned_text = text.strip()
    if not cleaned_text or not keywords:
        return pd.DataFrame(columns=["Keyword", "Similitud", "Similitud (%)"])
    text_embedding = model.encode([cleaned_text], convert_to_numpy=True)
    keyword_embeddings = model.encode(keywords, convert_to_numpy=True)
    similarities = cosine_similarity(text_embedding, keyword_embeddings)[0]
    results = pd.DataFrame(
        {
            "Keyword": keywords,
            "Similitud": similarities,
            "Similitud (%)": similarities * 100,
        }
    )
    return results.sort_values("Similitud", ascending=False).reset_index(drop=True)


def compute_faq_keyword_similarity(
    faq_entries: List[Tuple[str, str]],
    keywords: List[str],
    model: SentenceTransformer,
) -> pd.DataFrame:
    if not faq_entries or not keywords:
        return pd.DataFrame(columns=["Pregunta", "Respuesta", "Keyword", "Similitud", "Similitud (%)"])
    texts = []
    filtered_entries: List[Tuple[str, str]] = []
    for question, answer in faq_entries:
        question_clean = question.strip()
        answer_clean = answer.strip()
        combined = " ".join(part for part in (question_clean, answer_clean) if part).strip()
        if not combined:
            continue
        texts.append(combined)
        filtered_entries.append((question_clean, answer_clean))
    if not filtered_entries:
        return pd.DataFrame(columns=["Pregunta", "Respuesta", "Keyword", "Similitud", "Similitud (%)"])
    faq_embeddings = model.encode(texts, convert_to_numpy=True)
    keyword_embeddings = model.encode(keywords, convert_to_numpy=True)
    similarity_matrix = cosine_similarity(faq_embeddings, keyword_embeddings)
    rows: List[Dict[str, object]] = []
    for faq_idx, (question, answer) in enumerate(filtered_entries):
        for keyword_idx, keyword in enumerate(keywords):
            score = float(similarity_matrix[faq_idx, keyword_idx])
            rows.append(
                {
                    "Pregunta": question,
                    "Respuesta": answer,
                    "Keyword": keyword,
                    "Similitud": score,
                    "Similitud (%)": score * 100,
                }
            )
    df = pd.DataFrame(rows)
    return df.sort_values(["Pregunta", "Similitud"], ascending=[True, False]).reset_index(drop=True)


def top_n_by_group(df: pd.DataFrame, group_column: str, score_column: str, n: int) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        df.sort_values([group_column, score_column], ascending=[True, False])
        .groupby(group_column, as_index=False)
        .head(n)
        .reset_index(drop=True)
    )


def download_dataframe_button(df: pd.DataFrame, filename: str, label: str) -> None:
    if df.empty:
        st.info("No hay datos disponibles para descargar.")
        return
    buffer = io.BytesIO()
    if filename.endswith(".csv"):
        buffer.write(df.to_csv(index=False).encode("utf-8"))
    else:
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
    buffer.seek(0)
    st.download_button(label=label, data=buffer, file_name=filename, mime="application/octet-stream")


def fetch_url_content(url: str) -> str:
    try:
        downloaded = trafilatura.fetch_url(url)
    except Exception:
        return ""
    if not downloaded:
        return ""
    try:
        content = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
        )
    except Exception:
        return ""
    if not content:
        return ""
    return content.strip()[:MAX_COMPETITOR_CONTENT_CHARS]


def create_competitor_heatmap(df: pd.DataFrame) -> plt.Figure:
    pivot = df.pivot_table(values="Score", index="URL", columns="Query", aggfunc="first")
    num_queries = max(len(pivot.columns), 1)
    num_urls = max(len(pivot.index), 1)
    max_width = 34
    max_height = 28
    fig_width = float(np.clip(10 + num_queries * 1.2, 10, max_width))
    fig_height = float(np.clip(num_urls * 0.6, 4, max_height))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cell_count = num_queries * num_urls
    show_annotations = cell_count <= 225
    annot_font = max(7, 12 - cell_count / 60)
    annot_kws = {"fontsize": annot_font}

    sns.heatmap(
        pivot,
        annot=show_annotations,
        fmt=".2f",
        cmap="YlOrRd",
        cbar_kws={"label": "Similitud coseno", "pad": 0.02, "shrink": 0.9},
        ax=ax,
        annot_kws=annot_kws,
        linewidths=0.4,
        linecolor="white",
    )

    def _wrap_labels(labels: Sequence[str], width: int) -> List[str]:
        return [textwrap.fill(str(label), width=max(5, width)) for label in labels]

    x_wrap = 28 if num_queries <= 6 else 18 if num_queries <= 12 else 12
    y_wrap = 42 if num_urls <= 10 else 30 if num_urls <= 20 else 20
    x_rotation = 35 if num_queries <= 8 else 55
    x_font = max(8, 14 - num_queries * 0.3)
    y_font = max(8, 14 - num_urls * 0.2)

    ax.set_xticklabels(_wrap_labels(pivot.columns, x_wrap), rotation=x_rotation, ha="right", fontsize=x_font)
    ax.set_yticklabels(_wrap_labels(pivot.index, y_wrap), rotation=0, fontsize=y_font)
    ax.tick_params(axis="x", pad=12)
    ax.tick_params(axis="y", pad=6)
    ax.set_title("Heatmap de relevancia semantica", fontsize=14)
    ax.set_xlabel("Query")
    ax.set_ylabel("URL")
    left_margin = 0.2 if num_urls <= 12 else 0.26 if num_urls <= 30 else 0.34
    bottom_margin = 0.18 if num_queries <= 8 else 0.24 if num_queries <= 16 else 0.32
    right_margin = 0.92 if num_queries <= 12 else 0.88
    fig.subplots_adjust(left=left_margin, right=right_margin, bottom=bottom_margin, top=0.9)
    return fig


def normalize_url_to_text(url: str) -> str:
    parsed = urlparse(url.strip())
    segments: List[str] = []
    if parsed.netloc:
        host_tokens = parsed.netloc.replace(".", " ").replace("-", " ").split()
        segments.extend(host_tokens)
    path_parts = [part for part in parsed.path.split("/") if part]
    for part in path_parts:
        clean = unquote(part)
        clean = clean.replace("-", " ").replace("_", " ")
        segments.extend(clean.split())
    if parsed.query:
        query = parse_qs(parsed.query)
        for key, values in query.items():
            key_clean = key.replace("_", " ").replace("-", " ")
            segments.append(key_clean)
            for value in values:
                value_clean = unquote(value).replace("-", " ").replace("_", " ")
                segments.extend(value_clean.split())
    if parsed.fragment:
        fragment = unquote(parsed.fragment).replace("-", " ").replace("_", " ")
        segments.extend(fragment.split())
    text = " ".join(token for token in segments if token)
    return text.strip()


@st.cache_data(show_spinner=False)
def fetch_url_text_variants(url: str) -> Dict[str, str]:
    variants = {"Body": "", "Descripcion": "", "URL": ""}
    try:
        html = trafilatura.fetch_url(url)
    except Exception:
        return variants
    if not html:
        return variants

    body_text = ""
    try:
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
        )
        if extracted:
            body_text = extracted.strip()
    except Exception:
        body_text = ""

    if body_text:
        variants["Body"] = body_text[:MAX_URL_BODY_CHARS]

    description = ""
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        soup = None

    if soup:
        meta_desc = soup.find("meta", attrs={"name": re.compile("description", re.IGNORECASE)})
        if not meta_desc:
            meta_desc = soup.find("meta", attrs={"property": re.compile("description", re.IGNORECASE)})
        if meta_desc and meta_desc.get("content"):
            description = meta_desc.get("content", "").strip()
        if not description:
            first_paragraphs = [
                p.get_text(" ", strip=True)
                for p in soup.find_all("p")
                if p.get_text(strip=True)
            ]
            if first_paragraphs:
                description = " ".join(first_paragraphs[:2]).strip()

    if not description and body_text:
        description = body_text[:MAX_URL_DESCRIPTION_CHARS]

    variants["Descripcion"] = description[:MAX_URL_DESCRIPTION_CHARS] if description else ""
    variants["URL"] = normalize_url_to_text(url)

    return variants


def build_url_variant_entries(urls: Sequence[str]) -> Tuple[List[Dict[str, str]], Dict[str, Dict[str, str]], List[str]]:
    entries: List[Dict[str, str]] = []
    extracted: Dict[str, Dict[str, str]] = {}
    issues: List[str] = []
    for url in urls:
        variants = fetch_url_text_variants(url)
        extracted[url] = variants
        has_text = False
        for label, text in variants.items():
            clean_text = text.strip()
            if not clean_text:
                continue
            has_text = True
            entries.append({"URL": url, "Tipo": label, "Texto": clean_text})
        if not has_text:
            issues.append(f"No se pudo extraer contenido util de {url}.")
    return entries, extracted, issues


def compute_url_variant_keyword_similarity(
    entries: Sequence[Dict[str, str]],
    keywords: Sequence[str],
    model: SentenceTransformer,
) -> pd.DataFrame:
    if not entries or not keywords:
        return pd.DataFrame(columns=["URL", "Tipo", "Keyword", "Similitud", "Similitud (%)"])
    texts = [entry["Texto"] for entry in entries]
    text_embeddings = model.encode(texts, convert_to_numpy=True)
    keyword_embeddings = model.encode(list(keywords), convert_to_numpy=True)
    similarity_matrix = cosine_similarity(text_embeddings, keyword_embeddings)

    rows: List[Dict[str, object]] = []
    for entry_idx, entry in enumerate(entries):
        for keyword_idx, keyword in enumerate(keywords):
            score = float(similarity_matrix[entry_idx, keyword_idx])
            rows.append(
                {
                    "URL": entry["URL"],
                    "Tipo": entry["Tipo"],
                    "Keyword": keyword,
                    "Similitud": score,
                    "Similitud (%)": score * 100,
                }
            )

    df = pd.DataFrame(rows)
    df.sort_values(["URL", "Keyword", "Tipo", "Similitud"], ascending=[True, True, True, False], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def best_similarity_per_url_keyword(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    idx = df.groupby(["URL", "Keyword"])["Similitud"].idxmax()
    return (
        df.loc[idx]
        .sort_values(["URL", "Keyword"])
        .reset_index(drop=True)
    )


def keyword_relevance(
    df: pd.DataFrame,
    keywords: List[str],
    api_key: str,
    embedding_col: str,
    url_column: str,
    top_n: int,
    min_score: Optional[float],
) -> pd.DataFrame:
    client = OpenAI(api_key=api_key)
    embeddings = np.vstack(df[embedding_col].values)
    embeddings_norm = normalize(embeddings)

    results: List[Dict[str, object]] = []
    progress = st.progress(0.0)
    status = st.empty()
    for idx, keyword in enumerate(keywords, start=1):
        status.text(f"Procesando '{keyword}' ({idx}/{len(keywords)})...")
        try:
            response = client.embeddings.create(input=keyword, model="text-embedding-3-small")
            keyword_vector = np.array(response.data[0].embedding)
            keyword_vector /= np.linalg.norm(keyword_vector)

            scores = embeddings_norm @ keyword_vector * 100
            ranked_indices = np.argsort(scores)[::-1]

            keyword_rows: List[Dict[str, object]] = []
            for page_idx in ranked_indices:
                score = scores[page_idx]
                if len(keyword_rows) < top_n or (min_score is not None and score >= min_score):
                    keyword_rows.append(
                        {
                            "Keyword": keyword,
                            "URL": df.iloc[page_idx][url_column],
                            "RelevanceScore": score,
                        }
                    )
                else:
                    break
            results.extend(keyword_rows)
        except Exception as exc:
            results.append(
                {
                    "Keyword": keyword,
                    "URL": "ERROR",
                    "RelevanceScore": 0.0,
                    "Error": str(exc),
                }
            )
        progress.progress(idx / len(keywords))
    progress.empty()
    status.empty()
    return pd.DataFrame(results)


def _extract_competitor_contents(urls: Sequence[str]) -> Dict[str, str]:
    progress = st.progress(0.0)
    status = st.empty()
    extracted: Dict[str, str] = {}
    for idx, url in enumerate(urls, start=1):
        status.text(f"Extrayendo contenido de {url} ({idx}/{len(urls)})")
        content = fetch_url_content(url)
        if content:
            extracted[url] = content
        else:
            st.warning(f"No se pudo extraer contenido de {url}")
        progress.progress(idx / len(urls))
    progress.empty()
    status.empty()

    if not extracted:
        raise RuntimeError("No se extrajo contenido util de las URLs indicadas.")

    return extracted


def _render_kg_manual_tab() -> None:
    st.markdown("### Consultas manuales a Google Enterprise Knowledge Graph")
    st.caption("Envia consultas directas al KG sin necesidad de subir archivos o generar el grafo con spaCy.")

    manual_google_api_key = st.text_input(
        "API key de Google Enterprise KG (standalone)",
        value=st.session_state.get("google_kg_api_key", os.environ.get("GOOGLE_EKG_API_KEY", "")),
        type="password",
        key="google_kg_manual_api_key",
        help="Define la clave aqui o mediante la variable de entorno GOOGLE_EKG_API_KEY.",
    )
    if manual_google_api_key:
        st.session_state["google_kg_api_key"] = manual_google_api_key

    manual_mentions_raw = st.text_area(
        "Entidades o consultas (una por linea o separadas por ';')",
        value="Marca principal\nProducto estrella\nAutor reconocido",
        key="google_kg_manual_entities",
    )
    manual_mentions = parse_line_input(manual_mentions_raw, separators=("\n", ";", ","))

    manual_languages_text = st.text_input(
        "Idiomas preferidos (codigos ISO separados por comas)",
        value="es,en",
        key="google_kg_manual_languages",
    )

    manual_types_text = st.text_input(
        "Filtrar por tipos (opcional, separados por comas)",
        value="",
        key="google_kg_manual_types",
        help="Ejemplos: Person,Organization,Product. Deja vacio para permitir cualquier tipo.",
    )

    manual_limit = st.slider(
        "Resultados por consulta (Google KG)",
        min_value=1,
        max_value=10,
        value=3,
        key="google_kg_manual_limit",
    )

    manual_language_tokens = [token.strip() for token in manual_languages_text.split(",") if token.strip()]
    manual_type_tokens = [token.strip() for token in manual_types_text.split(",") if token.strip()]

    st.caption(f"Se enviaran {len(manual_mentions)} consultas (limite interno de 50 por llamada).")

    if st.button("Consultar Google KG (standalone)", key="google_kg_manual_button", disabled=not manual_mentions):
        try:
            manual_api_key = ensure_google_kg_api_key(manual_google_api_key)
            with st.spinner("Consultando Google Enterprise Knowledge Graph..."):
                manual_results_df = query_google_enterprise_kg(
                    mentions=manual_mentions,
                    api_key=manual_api_key,
                    limit=manual_limit,
                    languages=manual_language_tokens or None,
                    types=manual_type_tokens or None,
                )
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Error consultando Google KG: {exc}")
        else:
            st.session_state["google_kg_manual_results"] = manual_results_df
            if manual_results_df.empty:
                st.warning("Google KG no devolvio resultados para las consultas enviadas.")
            else:
                st.success(f"Se recuperaron {len(manual_results_df)} resultados desde Google KG.")

    manual_results_cached = st.session_state.get("google_kg_manual_results")
    if isinstance(manual_results_cached, pd.DataFrame) and not manual_results_cached.empty:
        st.dataframe(manual_results_cached, use_container_width=True)
        download_dataframe_button(
            manual_results_cached,
            "google_kg_manual_results.xlsx",
            "Descargar resultados manuales de Google KG",
        )
    elif isinstance(manual_results_cached, pd.DataFrame):
        st.info("Todavia no hay resultados manuales de Google KG para mostrar.")


def _render_text_vs_keywords_tab(default_model: str) -> None:
    st.markdown("Compara un texto con un listado de palabras clave usando embeddings multilingues.")
    main_text = st.text_area(
        "Texto principal",
        height=220,
        placeholder="Pega aqui el contenido que quieres analizar.",
        key="text_similarity_main",
    )
    keywords_raw = st.text_area(
        "Palabras clave (una por linea, acepta punto y coma o coma como separadores)",
        height=180,
        key="text_similarity_keywords",
    )
    model_name_text = st.text_input(
        "Modelo de Sentence Transformers",
        value=default_model,
        help="Ejemplo: paraphrase-multilingual-MiniLM-L12-v2",
        key="text_similarity_model",
    )

    if st.button("Calcular relevancia", key="text_similarity_button"):
        keywords = parse_line_input(keywords_raw, separators=("\n", ";", ","))
        if not main_text.strip():
            st.warning("Introduce el texto que deseas analizar.")
        elif not keywords:
            st.warning("Introduce al menos una palabra clave.")
        else:
            model_name_clean = model_name_text.strip()
            if not model_name_clean:
                st.warning("Indica un nombre de modelo valido.")
            else:
                try:
                    with st.spinner("Generando embeddings y calculando similitudes..."):
                        model = get_sentence_transformer(model_name_clean)
                        results_df = compute_text_keyword_similarity(main_text, keywords, model)
                except Exception as exc:
                    st.error(f"No se pudo cargar el modelo '{model_name_clean}': {exc}")
                else:
                    if results_df.empty:
                        st.info("No se generaron resultados. Revisa la informacion introducida.")
                    else:
                        st.success("Calculo completado.")
                        st.dataframe(results_df, use_container_width=True)
                        download_dataframe_button(
                            results_df,
                            "texto_vs_keywords.xlsx",
                            "Descargar resultados Excel",
                        )


def _render_faq_tab(default_model: str) -> None:
    st.markdown(
        "Analiza un bloque de preguntas frecuentes y obten la relevancia semantica de cada pregunta-respuesta frente a un listado de keywords."
    )

    # Selector de método de entrada
    input_method = st.radio(
        "Método de entrada de FAQs",
        options=["📝 Entrada manual", "📁 Cargar archivo Excel/CSV"],
        horizontal=True,
        key="faq_input_method"
    )

    faq_entries = []

    if input_method == "📝 Entrada manual":
        st.caption(
            "Formato recomendado: separa cada FAQ con una linea en blanco. Ejemplo:\n"
            "Pregunta: ¿Cual es el horario?\n"
            "Respuesta: Nuestro servicio esta disponible 24/7."
        )
        faq_text = st.text_area(
            "Preguntas frecuentes",
            height=260,
            placeholder="Introduce las preguntas frecuentes con su respuesta...",
            key="faq_text_area",
        )
        if faq_text.strip():
            faq_entries = parse_faq_blocks(faq_text)

    else:  # Cargar archivo
        st.caption("Sube un archivo Excel o CSV con columnas separadas para preguntas y respuestas")

        uploaded_faq_file = st.file_uploader(
            "Selecciona archivo Excel/CSV",
            type=['xlsx', 'xls', 'csv'],
            key="faq_file_uploader",
            help="El archivo debe tener al menos dos columnas: una para preguntas y otra para respuestas"
        )

        if uploaded_faq_file is not None:
            try:
                # Leer preview del archivo
                if uploaded_faq_file.name.endswith('.csv'):
                    df_preview = pd.read_csv(uploaded_faq_file)
                elif uploaded_faq_file.name.endswith('.xlsx'):
                    df_preview = pd.read_excel(uploaded_faq_file, engine='openpyxl')
                elif uploaded_faq_file.name.endswith('.xls'):
                    df_preview = pd.read_excel(uploaded_faq_file, engine='xlrd')
                else:
                    df_preview = None

                if df_preview is not None and not df_preview.empty:
                    st.success(f"✅ Archivo cargado: {len(df_preview)} filas, {len(df_preview.columns)} columnas")

                    # Mostrar preview
                    with st.expander("📋 Vista previa del archivo", expanded=False):
                        st.dataframe(df_preview.head(10), use_container_width=True)

                    # Selección de columnas
                    col1, col2 = st.columns(2)
                    with col1:
                        question_col = st.selectbox(
                            "Columna de PREGUNTAS",
                            options=df_preview.columns.tolist(),
                            key="faq_question_col",
                            help="Selecciona la columna que contiene las preguntas"
                        )
                    with col2:
                        answer_col = st.selectbox(
                            "Columna de RESPUESTAS",
                            options=df_preview.columns.tolist(),
                            index=min(1, len(df_preview.columns) - 1),  # Por defecto segunda columna
                            key="faq_answer_col",
                            help="Selecciona la columna que contiene las respuestas"
                        )

                    # Validar selección
                    if question_col == answer_col:
                        st.warning("⚠️ Las columnas de pregunta y respuesta deben ser diferentes")
                    else:
                        # Parsear FAQs
                        uploaded_faq_file.seek(0)  # Resetear el puntero del archivo
                        faq_entries = parse_faq_from_excel(uploaded_faq_file, question_col, answer_col)

                        if faq_entries:
                            st.info(f"📝 {len(faq_entries)} preguntas frecuentes detectadas")
                        else:
                            st.warning("No se encontraron FAQs válidas en las columnas seleccionadas")

            except Exception as e:
                st.error(f"❌ Error al procesar el archivo: {e}")

    # Keywords (común para ambos métodos)
    faq_keywords_raw = st.text_area(
        "Palabras clave (una por linea)",
        height=180,
        key="faq_keywords_area",
    )

    # Parámetros comunes
    col_params1, col_params2 = st.columns(2)
    with col_params1:
        top_n_faq = st.number_input(
            "Top N resultados por pregunta",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            key="faq_topn",
        )
    with col_params2:
        model_name_faq = st.text_input(
            "Modelo de Sentence Transformers",
            value=default_model,
            help="El mismo modelo se reutiliza para preguntas y keywords.",
            key="faq_model_name",
        )

    # Botón de cálculo
    if st.button("Calcular relevancia en FAQs", key="faq_button", type="primary"):
        keywords = parse_line_input(faq_keywords_raw, separators=("\n", ";", ","))

        if not faq_entries:
            st.warning("⚠️ Introduce al menos una pregunta frecuente con su respuesta.")
        elif not keywords:
            st.warning("⚠️ Introduce al menos una palabra clave.")
        else:
            model_name_clean = model_name_faq.strip()
            if not model_name_clean:
                st.warning("⚠️ Indica un nombre de modelo valido.")
            else:
                try:
                    with st.spinner("Calculando relevancia semantica en FAQs..."):
                        model = get_sentence_transformer(model_name_clean)
                        faq_results = compute_faq_keyword_similarity(faq_entries, keywords, model)
                except Exception as exc:
                    st.error(f"❌ No se pudo cargar el modelo '{model_name_clean}': {exc}")
                else:
                    if faq_results.empty:
                        st.info("No se generaron resultados. Revisa la informacion introducida.")
                    else:
                        st.success("✅ Calculo completado.")
                        parsed_df = pd.DataFrame(faq_entries, columns=["Pregunta", "Respuesta"])
                        st.markdown("**Preguntas frecuentes detectadas**")
                        st.dataframe(parsed_df, use_container_width=True)
                        top_results = top_n_by_group(faq_results, "Pregunta", "Similitud", int(top_n_faq))
                        st.markdown("**Relevancia semantica por pregunta**")
                        st.dataframe(top_results, use_container_width=True)
                        download_dataframe_button(
                            faq_results,
                            "faq_vs_keywords.xlsx",
                            "Descargar resultados completos",
                        )


def _render_competitors_tab(default_model: str) -> None:
    st.markdown(
        "Extrae el contenido de URLs de competidores, genera embeddings y calcula la relevancia respecto a tus queries."
    )
    competitor_urls_raw = st.text_area(
        "URLs de competidores (una por linea)",
        height=220,
        placeholder="https://example.com/pagina-1\nhttps://example.com/pagina-2",
        key="competitor_urls_area",
    )
    competitor_queries_raw = st.text_area(
        "Consultas o keywords (una por linea)",
        height=180,
        placeholder="keyword uno\nkeyword dos",
        key="competitor_queries_area",
    )
    top_n_competitors = st.number_input(
        "Top N resultados por query",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        key="competitor_topn",
    )
    model_name_competitors = st.text_input(
        "Modelo de Sentence Transformers",
        value=default_model,
        help="El modelo se usa tanto para las URLs como para las consultas.",
        key="competitor_model_name",
    )

    if st.button("Analizar competidores", key="competitor_button"):
        urls = parse_line_input(competitor_urls_raw, separators=("\n", ";"))
        queries = parse_line_input(competitor_queries_raw, separators=("\n", ";", ","))
        if not urls:
            st.warning("Introduce al menos una URL de competidor.")
            return
        if not queries:
            st.warning("Introduce al menos una consulta.")
            return

        model_name_clean = model_name_competitors.strip() or default_model
        try:
            model = get_sentence_transformer(model_name_clean)
        except Exception as exc:
            st.error(f"No se pudo cargar el modelo '{model_name_clean}': {exc}")
            return

        try:
            extracted = _extract_competitor_contents(urls)
        except RuntimeError as exc:
            st.error(str(exc))
            return

        ordered_urls = list(extracted.keys())
        with st.spinner("Calculando relevancia semantica..."):
            url_embeddings = model.encode(
                [extracted[url] for url in ordered_urls],
                convert_to_numpy=True,
            )
            query_embeddings = model.encode(queries, convert_to_numpy=True)
            similarity_matrix = cosine_similarity(query_embeddings, url_embeddings)

        rows: List[Dict[str, object]] = []
        for query_idx, query in enumerate(queries):
            for url_idx, url in enumerate(ordered_urls):
                score = float(similarity_matrix[query_idx, url_idx])
                rows.append(
                    {
                        "Query": query,
                        "URL": url,
                        "Score": score,
                        "Score (%)": score * 100,
                    }
                )

        results_df = pd.DataFrame(rows)

        if results_df.empty:
            st.info("No se generaron resultados. Revisa la informacion introducida.")
            return

        results_df.sort_values(["Query", "Score"], ascending=[True, False], inplace=True)
        results_df.reset_index(drop=True, inplace=True)
        top_results = top_n_by_group(results_df, "Query", "Score", int(top_n_competitors))
        st.success("Analisis completado.")
        st.markdown("**Top resultados por query**")
        st.dataframe(top_results, use_container_width=True)
        download_dataframe_button(
            results_df,
            "competidores_vs_queries.xlsx",
            "Descargar resultados completos",
        )

        st.session_state["competitor_tool_payload"] = {
            "urls": ordered_urls,
            "contents": dict(extracted),
            "model_name": model_name_clean,
            "queries": queries,
            "timestamp": time.time(),
        }

        if len(ordered_urls) > 1 and len(queries) > 1:
            heatmap_fig = create_competitor_heatmap(results_df)
            st.pyplot(heatmap_fig, clear_figure=True)

        with st.expander("Contenido extraido (primeros 400 caracteres por URL)"):
            for url in ordered_urls:
                preview = extracted[url][:400].replace("\n", " ")
                suffix = "..." if len(extracted[url]) > 400 else ""
                st.markdown(f"**{url}**")
                st.write(preview + suffix)

        st.caption(
            f"Se analizaron {len(ordered_urls)} URLs con un limite de {MAX_COMPETITOR_CONTENT_CHARS} caracteres por pagina."
        )


def _render_url_variants_tab(default_model: str) -> None:
    st.markdown(
        "Procesa un listado de URLs, genera embeddings especificos (cuerpo, descripcion larga y texto de la URL) y evalua la relevancia frente a tus palabras clave."
    )
    upload_cols = st.columns(2)
    with upload_cols[0]:
        urls_file = st.file_uploader(
            "Archivo con URLs (Excel o CSV)",
            type=["xlsx", "xls", "csv"],
            key="urls_variant_upload",
        )
    with upload_cols[1]:
        keywords_file_variant = st.file_uploader(
            "Archivo con keywords (opcional, Excel/CSV)",
            type=["xlsx", "xls", "csv"],
            key="urls_keywords_upload",
        )

    file_urls: List[str] = []
    if urls_file:
        try:
            if urls_file.name.lower().endswith(".csv"):
                urls_df = pd.read_csv(urls_file)
            else:
                urls_df = pd.read_excel(urls_file)
        except Exception as exc:
            st.error(f"No se pudo leer el archivo de URLs: {exc}")
            urls_df = None
        if isinstance(urls_df, pd.DataFrame):
            st.dataframe(urls_df.head())
            columns = urls_df.columns.tolist()
            if columns:
                selected_url_column = st.selectbox(
                    "Columna que contiene las URLs",
                    options=columns,
                    key="urls_column_selector",
                )
                file_urls = (
                    urls_df[selected_url_column]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .tolist()
                )
                st.caption(f"Se detectaron {len(file_urls)} URLs en el archivo.")
            else:
                st.warning("El archivo de URLs no contiene columnas detectables.")

    urls_raw_text = st.text_area(
        "URLs adicionales (una por linea)",
        height=180,
        key="urls_variant_textarea",
    )
    manual_urls = parse_line_input(urls_raw_text, separators=("\n", ";", ","))

    file_keywords: List[str] = []
    if keywords_file_variant:
        try:
            if keywords_file_variant.name.lower().endswith(".csv"):
                keywords_df_variant = pd.read_csv(keywords_file_variant)
            else:
                keywords_df_variant = pd.read_excel(keywords_file_variant)
        except Exception as exc:
            st.error(f"No se pudo leer el archivo de keywords: {exc}")
            keywords_df_variant = None
        if isinstance(keywords_df_variant, pd.DataFrame):
            st.dataframe(keywords_df_variant.head())
            keyword_columns = keywords_df_variant.columns.tolist()
            if keyword_columns:
                selected_keyword_column = st.selectbox(
                    "Columna de palabras clave (archivo)",
                    options=keyword_columns,
                    key="urls_keywords_column",
                )
                file_keywords = (
                    keywords_df_variant[selected_keyword_column]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .unique()
                    .tolist()
                )
            else:
                st.warning("El archivo de keywords no contiene columnas detectables.")

    urls_keywords_raw = st.text_area(
        "Palabras clave adicionales (una por linea)",
        height=180,
        key="urls_keywords_textarea",
    )
    manual_keywords = parse_line_input(urls_keywords_raw, separators=("\n", ";", ","))

    urls_model_name = st.text_input(
        "Modelo de Sentence Transformers",
        value=default_model,
        help="El modelo se usa para generar los embeddings de pagina y las palabras clave.",
        key="urls_model_name",
    )

    top_n_per_url = st.number_input(
        "Top N resultados por URL y keyword",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        key="urls_topn",
    )

    if st.button("Analizar URLs y keywords", key="urls_analyze_button"):
        combined_urls: List[str] = []
        seen_urls: set[str] = set()
        for source in (file_urls, manual_urls):
            for url_value in source:
                url_clean = url_value.strip()
                if not url_clean or url_clean in seen_urls:
                    continue
                combined_urls.append(url_clean)
                seen_urls.add(url_clean)

        combined_keywords: List[str] = []
        seen_keywords: set[str] = set()
        for source in (file_keywords, manual_keywords):
            for keyword in source:
                keyword_clean = keyword.strip()
                if not keyword_clean or keyword_clean in seen_keywords:
                    continue
                combined_keywords.append(keyword_clean)
                seen_keywords.add(keyword_clean)

        if not combined_urls:
            st.warning("Introduce al menos una URL mediante archivo o campo de texto.")
            return
        if not combined_keywords:
            st.warning("Introduce al menos una palabra clave (archivo o campo de texto).")
            return

        model_name_clean = urls_model_name.strip() or default_model
        try:
            model = get_sentence_transformer(model_name_clean)
        except Exception as exc:
            st.error(f"No se pudo cargar el modelo '{model_name_clean}': {exc}")
            return

        progress = st.progress(0.0)
        status = st.empty()
        entries: List[Dict[str, str]] = []
        extracted_variants: Dict[str, Dict[str, str]] = {}
        issues: List[str] = []

        total_urls = len(combined_urls)
        for idx, url_value in enumerate(combined_urls, start=1):
            status.text(f"Extrayendo variantes de {url_value} ({idx}/{total_urls})")
            try:
                new_entries, components_map, entry_issues = build_url_variant_entries([url_value])
            except Exception as exc:
                issues.append(f"Error procesando {url_value}: {exc}")
                extracted_variants[url_value] = {"Body": "", "Descripcion": "", "URL": ""}
            else:
                entries.extend(new_entries)
                extracted_variants.update(components_map)
                issues.extend(entry_issues)
            progress.progress(idx / total_urls if total_urls else 1.0)
        progress.empty()
        status.empty()

        if not entries:
            st.error("No se pudo extraer contenido util de las URLs indicadas.")
            for warning_msg in issues:
                st.warning(warning_msg)
            return

        for warning_msg in issues:
            st.warning(warning_msg)

        with st.spinner("Calculando relevancia semantica..."):
            results_df = compute_url_variant_keyword_similarity(entries, combined_keywords, model)

        if results_df.empty:
            st.info("No se generaron resultados. Revisa las URLs y palabras clave proporcionadas.")
            return

        st.success("Analisis completado.")
        st.dataframe(results_df, use_container_width=True)
        download_dataframe_button(
            results_df,
            "urls_variantes_vs_keywords.xlsx",
            "Descargar resultados completos",
        )

        best_df = best_similarity_per_url_keyword(results_df)
        if not best_df.empty:
            st.markdown("**Mejor coincidencia por URL y keyword**")
            st.dataframe(best_df, use_container_width=True)
            download_dataframe_button(
                best_df,
                "urls_variantes_mejor_match.xlsx",
                "Descargar mejor coincidencia",
            )

        limited_df = (
            results_df.sort_values(["URL", "Keyword", "Similitud"], ascending=[True, True, False])
            .groupby(["URL", "Keyword"], as_index=False)
            .head(int(top_n_per_url))
            .reset_index(drop=True)
        )
        st.markdown("**Top resultados por URL y keyword**")
        st.dataframe(limited_df, use_container_width=True)
        download_dataframe_button(
            limited_df,
            "urls_variantes_top.xlsx",
            "Descargar top resultados",
        )

        with st.expander("Detalle de textos extraidos por URL"):
            for url_value in combined_urls:
                components = extracted_variants.get(
                    url_value,
                    {"Body": "", "Descripcion": "", "URL": ""},
                )
                st.markdown(f"**{url_value}**")
                for label, text_value in components.items():
                    if not text_value:
                        st.markdown(f"- `{label}`: (sin contenido)")
                        continue
                    preview = text_value[:500].replace("\n", " ").strip()
                    ellipsis = "..." if len(text_value) > 500 else ""
                    st.markdown(f"- `{label}`: {preview}{ellipsis}")
                st.markdown("---")


def _embed_urls(
    urls: Sequence[str],
    label: str,
    model: SentenceTransformer,
    cached_map: Optional[Dict[str, str]] = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    cached_map = cached_map or {}
    clean_urls: List[str] = []
    seen: Set[str] = set()
    for url_value in urls:
        url_clean = url_value.strip()
        if not url_clean or url_clean in seen:
            continue
        clean_urls.append(url_clean)
        seen.add(url_clean)
    if not clean_urls:
        raise ValueError(f"Introduce al menos una URL para {label}.")

    texts: List[str] = []
    used_urls: List[str] = []
    issues: List[str] = []

    progress = st.progress(0.0)
    status = st.empty()
    total = len(clean_urls)
    for idx, url_value in enumerate(clean_urls, start=1):
        status.text(f"{label}: extrayendo {url_value} ({idx}/{total})")
        cached_text = cached_map.get(url_value) if cached_map else None
        content = cached_text or fetch_url_content(url_value)
        if not content:
            issues.append(f"No se obtuvo contenido de {url_value}")
        else:
            used_urls.append(url_value)
            texts.append(content)
        progress.progress(idx / total)
    progress.empty()
    status.empty()

    if not texts:
        raise ValueError(
            f"No se obtuvieron contenidos validos para las URLs de {label}. "
            "Comprueba que las paginas sean accesibles."
        )
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings, used_urls, issues


def _render_authority_gaps_tab(default_model: str) -> None:
    st.markdown("### Brechas de autoridad sin CSV")
    st.caption(
        "Simula brechas con datos ficticios o reutiliza las URLs de la pestaña de competidores para detectar huecos tematicos sin subir archivos."
    )

    st.session_state.setdefault("authority_gap_result", None)
    competitor_payload = st.session_state.get("competitor_tool_payload")
    if competitor_payload:
        st.info(
            f"Hay {len(competitor_payload.get('urls', []))} URLs guardadas desde 'Competidores vs queries'. Ejecuta de nuevo esa pestaña para refrescarlas."
        )
    else:
        st.caption("Aun no hay URLs almacenadas desde la pestaña de competidores en esta sesion.")

    mode = st.radio(
        "Modo de trabajo",
        options=["Simulacion guiada", "URLs manuales"],
        horizontal=True,
        key="authority_gap_mode",
    )

    if mode == "Simulacion guiada":
        col_counts = st.columns(3)
        with col_counts[0]:
            n_comp_docs = st.number_input(
                "Docs competencia",
                min_value=3,
                max_value=60,
                value=10,
                step=1,
                key="authority_comp_docs",
            )
        with col_counts[1]:
            n_site_docs = st.number_input(
                "Docs sitio propio",
                min_value=3,
                max_value=60,
                value=8,
                step=1,
                key="authority_site_docs",
            )
        with col_counts[2]:
            embedding_dim = st.number_input(
                "Dimension embeddings",
                min_value=32,
                max_value=512,
                value=200,
                step=16,
                key="authority_embedding_dim",
            )

        col_params = st.columns(3)
        with col_params[0]:
            n_clusters = st.slider(
                "Clusters competencia",
                min_value=2,
                max_value=10,
                value=4,
                step=1,
                key="authority_cluster_count",
            )
        with col_params[1]:
            gap_threshold_percent = st.slider(
                "Umbral de gap (%)",
                min_value=40,
                max_value=90,
                value=70,
                step=5,
                key="authority_gap_threshold",
            )
        with col_params[2]:
            seed_value = st.number_input(
                "Semilla",
                min_value=0,
                max_value=9999,
                value=42,
                step=1,
                key="authority_seed",
            )

        if st.button("Simular cobertura tematica", key="authority_run_button"):
            try:
                with st.spinner("Calculando cobertura simulada..."):
                    result = run_authority_gap_simulation(
                        n_competencia=int(n_comp_docs),
                        n_propias=int(n_site_docs),
                        dimensiones=int(embedding_dim),
                        n_clusters=int(n_clusters),
                        umbral_gap=float(gap_threshold_percent) / 100.0,
                        seed=int(seed_value),
                    )
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state["authority_gap_result"] = result
                st.success("Simulacion completada. Revisa los clusters y brechas detectadas.")
    else:
        default_competitors = "\n".join(competitor_payload.get("urls", [])) if competitor_payload else ""
        competitor_area = st.text_area(
            "URLs de competencia (una por linea)",
            value=default_competitors,
            height=200,
            key="authority_manual_comp_urls",
        )
        own_area = st.text_area(
            "URLs de tu sitio (una por linea)",
            height=200,
            key="authority_manual_site_urls",
        )

        manual_cols = st.columns(3)
        with manual_cols[0]:
            manual_clusters = st.slider(
                "Clusters competencia",
                min_value=2,
                max_value=10,
                value=4,
                step=1,
                key="authority_manual_clusters",
            )
        with manual_cols[1]:
            manual_gap_percent = st.slider(
                "Umbral de gap (%)",
                min_value=40,
                max_value=90,
                value=70,
                step=5,
                key="authority_manual_gap",
            )
        with manual_cols[2]:
            manual_model_name = st.text_input(
                "Modelo Sentence Transformers",
                value=competitor_payload.get("model_name", DEFAULT_SENTENCE_MODEL)
                if competitor_payload
                else DEFAULT_SENTENCE_MODEL,
                key="authority_manual_model",
            )

        if st.button("Analizar brechas con mis URLs", key="authority_run_manual"):
            competitor_urls = parse_line_input(competitor_area, separators=("\n", ";", ","))
            own_urls = parse_line_input(own_area, separators=("\n", ";", ","))
            if not competitor_urls:
                st.warning("Introduce al menos una URL de competencia.")
            elif not own_urls:
                st.warning("Introduce al menos una URL propia.")
            else:
                model_name_clean = manual_model_name.strip() or DEFAULT_SENTENCE_MODEL
                try:
                    model = get_sentence_transformer(model_name_clean)
                except Exception as exc:
                    st.error(f"No se pudo cargar el modelo solicitado: {exc}")
                else:
                    try:
                        cached_comp = competitor_payload.get("contents") if competitor_payload else {}
                        with st.spinner("Extrayendo contenido y generando embeddings..."):
                            comp_embeddings, comp_used, comp_issues = _embed_urls(
                                competitor_urls,
                                "competencia",
                                model,
                                cached_map=cached_comp,
                            )
                            site_embeddings, site_used, site_issues = _embed_urls(
                                own_urls,
                                "sitio propio",
                                model,
                            )
                            result = run_authority_gap_from_embeddings(
                                embeddings_prop=site_embeddings,
                                embeddings_comp=comp_embeddings,
                                docs_prop=site_used,
                                docs_comp=comp_used,
                                n_clusters=int(manual_clusters),
                                umbral_gap=float(manual_gap_percent) / 100.0,
                            )
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        for warning_msg in comp_issues + site_issues:
                            st.warning(warning_msg)
                        st.session_state["authority_gap_result"] = result
                        st.success("Analisis completado con tus URLs. Revisa los clusters detectados.")

    stored_result = st.session_state.get("authority_gap_result")
    if isinstance(stored_result, AuthorityGapResult):
        clusters_rows: List[Dict[str, object]] = []
        gap_ids = {gap.cluster_id for gap in stored_result.gaps}
        for cluster in stored_result.clusters:
            clusters_rows.append(
                {
                    "Cluster": cluster.cluster_id,
                    "Similitud (%)": round(cluster.similarity * 100, 2),
                    "Documentos competencia": ", ".join(cluster.documents) or "-",
                    "Es gap": "Si" if cluster.cluster_id in gap_ids else "No",
                }
            )

        clusters_df = pd.DataFrame(clusters_rows)
        meta_cols = st.columns(4)
        meta_cols[0].metric("Clusters evaluados", stored_result.metadata.get("n_clusters", len(clusters_df)))
        umbral_registrado = stored_result.metadata.get("umbral_gap")
        meta_cols[1].metric(
            "Umbral gap",
            f"{int(round(float(umbral_registrado) * 100))}%"
            if isinstance(umbral_registrado, (int, float))
            else "N/D",
        )
        meta_cols[2].metric(
            "Docs competencia", stored_result.metadata.get("n_competencia", len(stored_result.competitor_documents))
        )
        meta_cols[3].metric(
            "Docs propios", stored_result.metadata.get("n_propias", len(stored_result.site_documents))
        )

        st.markdown("**Cobertura de la competencia**")
        st.dataframe(clusters_df, use_container_width=True)
        download_dataframe_button(
            clusters_df,
            "brechas_autoridad.xlsx",
            "Descargar tabla de brechas",
        )

        gap_rows = [row for row in clusters_rows if row["Es gap"] == "Si"]
        if gap_rows:
            st.warning(f"Se detectaron {len(gap_rows)} gap(s) tematicos por debajo del umbral.")
            st.dataframe(pd.DataFrame(gap_rows), use_container_width=True)
        else:
            st.success("Con los parametros actuales no se registran gaps de autoridad.")
    else:
        if mode == "Simulacion guiada":
            st.info("Configura los parametros y pulsa el boton para simular posibles gaps sin cargar un CSV.")
        else:
            st.info("Introduce tus URLs y ejecuta el analisis para detectar brechas reales sin CSV.")


def render_semantic_toolkit_section() -> None:
    """
    Orquesta el conjunto de herramientas adicionales que no requieren cargar un dataset completo.
    """
    st.markdown("### Herramientas de analisis semantico adicionales")
    st.caption("Explora comparativas de texto, FAQs, competidores y URLs enriquecidas sin necesidad de subir un dataset completo.")
    tab_text, tab_faq, tab_competitors, tab_url_variants, tab_authority, tab_kg = st.tabs(
        [
            "Texto vs keywords",
            "FAQs vs keywords",
            "Competidores vs queries",
            "URLs enriquecidas",
            "Brechas de autoridad",
            "Consultas KG Manuales",
        ]
    )

    default_model = DEFAULT_SENTENCE_MODEL

    with tab_kg:
        _render_kg_manual_tab()

    with tab_text:
        _render_text_vs_keywords_tab(default_model)

    with tab_faq:
        _render_faq_tab(default_model)

    with tab_competitors:
        _render_competitors_tab(default_model)

    with tab_url_variants:
        _render_url_variants_tab(default_model)

    with tab_authority:
        _render_authority_gaps_tab(default_model)


__all__ = [
    "DEFAULT_SENTENCE_MODEL",
    "MAX_COMPETITOR_CONTENT_CHARS",
    "MAX_URL_BODY_CHARS",
    "MAX_URL_DESCRIPTION_CHARS",
    "best_similarity_per_url_keyword",
    "build_url_variant_entries",
    "compute_faq_keyword_similarity",
    "compute_text_keyword_similarity",
    "compute_url_variant_keyword_similarity",
    "download_dataframe_button",
    "fetch_url_content",
    "fetch_url_text_variants",
    "get_sentence_transformer",
    "keyword_relevance",
    "normalize_url_to_text",
    "parse_faq_blocks",
    "parse_line_input",
    "top_n_by_group",
    "render_semantic_toolkit_section",
]
