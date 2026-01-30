"""
Módulo Content Plan Generator.

Genera estructuras de encabezados (H1/H2/H3) con Gemini a partir de un plan de contenidos,
valida coherencia semántica, detecta canibalización, sugiere enlazado interno
y exporta resultados listos para CMS.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

try:
    import google.generativeai as genai
except ModuleNotFoundError:
    genai = None

try:
    from umap import UMAP

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import networkx as nx
from pyvis.network import Network

# ── Constants ──────────────────────────────────────────────
CHECKPOINT_INTERVAL = 5
MAX_RETRIES = 3
CANNIBALIZATION_THRESHOLD = 0.85
SHORT_KW_MAX_WORDS = 2
REQUIRED_COLUMNS = {"titulo", "kw", "kw_secundarias"}
SS = "cp_"  # session_state prefix


# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════


def _parse_gemini_json(raw_text: str) -> dict:
    """Strip markdown fences and parse JSON."""
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    return json.loads(text)


def _compute_embeddings(texts: List[str], model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> np.ndarray:
    """Compute sentence-transformer embeddings with session cache."""
    from semantic_tools import get_sentence_transformer

    cache_key = f"{SS}embeddings_cache"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = {}

    cache = st.session_state[cache_key]
    to_compute = []
    to_compute_idx = []

    for i, t in enumerate(texts):
        h = hashlib.md5(f"{t}:{model_name}".encode()).hexdigest()
        if h not in cache:
            to_compute.append(t)
            to_compute_idx.append((i, h))

    if to_compute:
        model = get_sentence_transformer(model_name)
        new_embs = model.encode(to_compute, show_progress_bar=False)
        for (_, h), emb in zip(to_compute_idx, new_embs):
            cache[h] = emb

    result = []
    for t in texts:
        h = hashlib.md5(f"{t}:{model_name}".encode()).hexdigest()
        result.append(cache[h])

    return np.array(result)


def _clean_illegal_excel_chars(s: str) -> str:
    """Remove illegal XML characters for Excel."""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", s)


def _compute_upload_hash(df: pd.DataFrame) -> str:
    """MD5 of titulo column for version control."""
    content = "|".join(sorted(df["titulo"].astype(str).tolist()))
    return hashlib.md5(content.encode()).hexdigest()


# ══════════════════════════════════════════════════════════
# PHASE 1: CORE GENERATION
# ══════════════════════════════════════════════════════════


def _upload_excel() -> Optional[pd.DataFrame]:
    """Upload Excel and let the user map columns to required fields."""
    uploaded = st.file_uploader(
        "Archivo del plan de contenidos",
        type=["csv", "xlsx", "xls"],
        key=f"{SS}uploader",
    )
    if not uploaded:
        return None

    try:
        name = uploaded.name.lower()
        if name.endswith(".csv"):
            raw_df = pd.read_csv(uploaded)
        else:
            raw_df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Error leyendo archivo: {e}")
        return None

    st.dataframe(raw_df.head(5), use_container_width=True, hide_index=True)

    columns = list(raw_df.columns)
    none_option = "(No usar)"

    # Auto-detect best default index for each field
    def _guess_col(hints: List[str]) -> int:
        for hint in hints:
            for i, c in enumerate(columns):
                if hint in c.strip().lower():
                    return i
        return 0

    st.markdown("##### Mapeo de columnas")
    c1, c2, c3 = st.columns(3)

    with c1:
        titulo_col = st.selectbox(
            "Columna de Título",
            options=columns,
            index=_guess_col(["titulo", "title", "nombre", "tema"]),
            key=f"{SS}col_titulo",
        )
    with c2:
        kw_col = st.selectbox(
            "Columna de KW principal",
            options=columns,
            index=_guess_col(["kw", "keyword", "palabra"]),
            key=f"{SS}col_kw",
        )
    with c3:
        kw_sec_col = st.selectbox(
            "Columna de KW secundarias",
            options=[none_option] + columns,
            index=_guess_col(["secundaria", "secondary", "kw_sec"]) + 1,  # +1 for none_option
            key=f"{SS}col_kw_sec",
        )

    if not st.button("✅ Confirmar columnas", key=f"{SS}confirm_cols", type="primary"):
        st.caption("Selecciona las columnas y pulsa confirmar para continuar.")
        return None

    # Build normalized DataFrame
    df = pd.DataFrame()
    df["titulo"] = raw_df[titulo_col].astype(str)
    df["kw"] = raw_df[kw_col].astype(str)
    df["kw_secundarias"] = (
        raw_df[kw_sec_col].fillna("").astype(str) if kw_sec_col != none_option else ""
    )

    # Preserve extra columns for reference
    for col in columns:
        if col not in [titulo_col, kw_col, kw_sec_col] and col not in df.columns:
            df[f"_extra_{col}"] = raw_df[col]

    df = df[df["titulo"].str.strip() != ""].reset_index(drop=True)
    df = df[df["kw"].str.strip() != ""].reset_index(drop=True)

    return df


def _estimate_tokens(df: pd.DataFrame) -> Dict[str, int]:
    """Estimate tokens and cost for the batch."""
    n_rows = len(df)
    avg_input_chars = df.apply(
        lambda r: len(str(r["titulo"])) + len(str(r["kw"])) + len(str(r["kw_secundarias"])),
        axis=1,
    ).mean()
    prompt_template_chars = 600  # approximate prompt template length

    est_input_tokens = int((avg_input_chars + prompt_template_chars) / 4 * n_rows)
    est_output_tokens = int(350 * n_rows)  # ~350 tokens per structured response
    est_total = est_input_tokens + est_output_tokens

    # Gemini Flash pricing (approximate)
    cost_per_million_input = 0.075
    cost_per_million_output = 0.30
    est_cost = (est_input_tokens / 1_000_000 * cost_per_million_input) + (
        est_output_tokens / 1_000_000 * cost_per_million_output
    )

    return {
        "n_rows": n_rows,
        "est_input_tokens": est_input_tokens,
        "est_output_tokens": est_output_tokens,
        "est_total_tokens": est_total,
        "est_cost_usd": round(est_cost, 4),
        "est_api_calls": n_rows,
    }


def _build_heading_prompt(titulo: str, kw_principal: str, kw_secundarias: str) -> str:
    """Build Gemini prompt for H1/H2/H3 structured generation."""
    return f"""Eres un experto en arquitectura de contenido SEO en español.

Artículo planificado:
- Título: {titulo}
- Keyword principal: {kw_principal}
- Keywords secundarias: {kw_secundarias}

Genera una estructura de encabezados optimizada para SEO.

Reglas:
1. Exactamente 1 H1 que incluya la keyword principal de forma natural.
2. Entre 3-6 H2 que cubran subtemas relevantes. Al menos 2 H2 deben contener keywords secundarias.
3. Cada H2 puede tener 1-3 H3 de profundización.
4. Los encabezados deben ser naturales, no keyword-stuffed.
5. Ordena lógicamente: introducción → desarrollo → conclusión/CTA.

Devuelve SOLO JSON válido, sin markdown ni comentarios:

{{
  "h1": "...",
  "sections": [
    {{
      "h2": "...",
      "h3s": ["...", "..."]
    }}
  ],
  "meta_title": "... (max 60 caracteres)",
  "meta_description": "... (max 155 caracteres)"
}}"""


def _generate_headings_for_row(
    model, titulo: str, kw_principal: str, kw_secundarias: str
) -> Tuple[Optional[dict], Optional[str]]:
    """
    Call Gemini for one row with retry logic.
    Returns (parsed_dict, error_string).
    """
    prompt = _build_heading_prompt(titulo, kw_principal, kw_secundarias)

    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(prompt)

            # Check safety filter
            if response.candidates and hasattr(response.candidates[0], "finish_reason"):
                reason = str(response.candidates[0].finish_reason)
                if "SAFETY" in reason.upper():
                    return None, "Bloqueado por filtro de seguridad de IA"

            raw = getattr(response, "text", "") or ""
            if not raw and response.candidates:
                raw = "".join(
                    part.text
                    for part in response.candidates[0].content.parts
                    if hasattr(part, "text")
                )

            if not raw.strip():
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1)
                    continue
                return None, "Respuesta vacía de Gemini"

            parsed = _parse_gemini_json(raw)

            # Validate structure
            if "h1" not in parsed or "sections" not in parsed:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1)
                    continue
                return None, "JSON válido pero estructura incompleta (falta h1 o sections)"

            return parsed, None

        except json.JSONDecodeError as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
                continue
            return None, f"JSON inválido tras {MAX_RETRIES} intentos: {e}"
        except Exception as e:
            err_str = str(e)
            if "SAFETY" in err_str.upper() or "blocked" in err_str.lower():
                return None, "Bloqueado por filtro de seguridad de IA"
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
                continue
            return None, f"Error API: {e}"

    return None, "Error desconocido tras reintentos"


def _batch_generate_headings(df: pd.DataFrame, model) -> Tuple[List[Optional[dict]], List[dict]]:
    """
    Process all rows with checkpoint saves.
    Returns (headings_list, errors_list).
    """
    n = len(df)
    headings = st.session_state.get(f"{SS}generated_headings", [None] * n)
    errors = st.session_state.get(f"{SS}generation_errors", [])
    start_idx = st.session_state.get(f"{SS}checkpoint_index", 0)

    if len(headings) < n:
        headings.extend([None] * (n - len(headings)))

    progress = st.progress(start_idx / n if n > 0 else 0, text=f"Procesando {start_idx}/{n}...")

    for i in range(start_idx, n):
        row = df.iloc[i]
        progress.progress((i + 1) / n, text=f"Procesando {i + 1}/{n}: {row['titulo'][:50]}...")

        result, error = _generate_headings_for_row(
            model,
            str(row["titulo"]),
            str(row["kw"]),
            str(row["kw_secundarias"]),
        )

        headings[i] = result
        if error:
            errors.append({"row": i, "titulo": row["titulo"], "error": error})

        # Checkpoint
        if (i + 1) % CHECKPOINT_INTERVAL == 0 or i == n - 1:
            st.session_state[f"{SS}generated_headings"] = headings
            st.session_state[f"{SS}generation_errors"] = errors
            st.session_state[f"{SS}checkpoint_index"] = i + 1

    progress.empty()
    return headings, errors


def _validate_semantic_coherence(
    df: pd.DataFrame, headings: List[Optional[dict]], model_name: str
) -> pd.DataFrame:
    """Compute cosine similarity between KW↔H2 and H1↔H3."""
    rows = []
    for i, h in enumerate(headings):
        if h is None:
            rows.append({
                "idx": i,
                "titulo": df.iloc[i]["titulo"],
                "kw_h2_sim_avg": 0.0,
                "h1_h3_sim_avg": 0.0,
                "kw_h2_sim_min": 0.0,
                "h1_h3_sim_min": 0.0,
                "status": "sin_generar",
            })
            continue

        kw = str(df.iloc[i]["kw"])
        h1 = h.get("h1", "")
        h2s = [s["h2"] for s in h.get("sections", []) if s.get("h2")]
        h3s = []
        for s in h.get("sections", []):
            h3s.extend(s.get("h3s", []))

        # KW vs H2
        if h2s:
            all_texts = [kw] + h2s
            embs = _compute_embeddings(all_texts, model_name)
            kw_emb = embs[0:1]
            h2_embs = embs[1:]
            sims = cosine_similarity(kw_emb, h2_embs)[0]
            kw_h2_avg = float(sims.mean())
            kw_h2_min = float(sims.min())
        else:
            kw_h2_avg = 0.0
            kw_h2_min = 0.0

        # H1 vs H3
        if h3s and h1:
            all_texts = [h1] + h3s
            embs = _compute_embeddings(all_texts, model_name)
            h1_emb = embs[0:1]
            h3_embs = embs[1:]
            sims = cosine_similarity(h1_emb, h3_embs)[0]
            h1_h3_avg = float(sims.mean())
            h1_h3_min = float(sims.min())
        else:
            h1_h3_avg = 1.0
            h1_h3_min = 1.0

        status = "ok"
        if kw_h2_avg < 0.4:
            status = "baja_coherencia_h2"
        if h1_h3_min < 0.3:
            status = "deriva_semantica_h3"

        rows.append({
            "idx": i,
            "titulo": df.iloc[i]["titulo"],
            "kw_h2_sim_avg": round(kw_h2_avg, 3),
            "h1_h3_sim_avg": round(h1_h3_avg, 3),
            "kw_h2_sim_min": round(kw_h2_min, 3),
            "h1_h3_sim_min": round(h1_h3_min, 3),
            "status": status,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════
# PHASE 2: ANALYSIS
# ══════════════════════════════════════════════════════════


def _detect_cannibalization(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Cross-compare all titulo+kw embeddings for cannibalization."""
    combined = (df["titulo"].astype(str) + " " + df["kw"].astype(str)).tolist()
    embs = _compute_embeddings(combined, model_name)
    sim_matrix = cosine_similarity(embs)

    pairs = []
    for i in range(len(sim_matrix)):
        for j in range(i + 1, len(sim_matrix)):
            sim = float(sim_matrix[i, j])
            if sim >= CANNIBALIZATION_THRESHOLD:
                pairs.append({
                    "Artículo A": df.iloc[i]["titulo"],
                    "KW A": df.iloc[i]["kw"],
                    "Artículo B": df.iloc[j]["titulo"],
                    "KW B": df.iloc[j]["kw"],
                    "Similitud": round(sim, 3),
                })

    return pd.DataFrame(pairs)


def _ngram_validation(headings: List[Optional[dict]], df: pd.DataFrame) -> pd.DataFrame:
    """Check if secondary KWs appear physically in generated H2/H3 text."""
    rows = []
    for i, h in enumerate(headings):
        if h is None:
            continue
        kw_sec_raw = str(df.iloc[i]["kw_secundarias"])
        if not kw_sec_raw.strip():
            continue

        # Split secondary KWs by comma or semicolon
        kw_list = [k.strip().lower() for k in re.split(r"[,;|]", kw_sec_raw) if k.strip()]

        # Collect all heading text
        all_text = h.get("h1", "").lower()
        for s in h.get("sections", []):
            all_text += " " + s.get("h2", "").lower()
            all_text += " " + " ".join(s.get("h3s", [])).lower()

        for kw in kw_list:
            found = kw in all_text
            rows.append({
                "idx": i,
                "titulo": df.iloc[i]["titulo"],
                "kw_secundaria": kw,
                "presente": found,
            })

    return pd.DataFrame(rows)


def _detect_semantic_drift(headings: List[Optional[dict]], model_name: str) -> pd.DataFrame:
    """Measure H1↔H3 distance per row."""
    rows = []
    for i, h in enumerate(headings):
        if h is None:
            continue
        h1 = h.get("h1", "")
        if not h1:
            continue
        for s in h.get("sections", []):
            h2 = s.get("h2", "")
            for h3 in s.get("h3s", []):
                embs = _compute_embeddings([h1, h3], model_name)
                sim = float(cosine_similarity(embs[0:1], embs[1:2])[0, 0])
                rows.append({
                    "idx": i,
                    "h1": h1,
                    "h2": h2,
                    "h3": h3,
                    "h1_h3_sim": round(sim, 3),
                    "drift": sim < 0.35,
                })
    return pd.DataFrame(rows)


def _enrich_short_keywords(df: pd.DataFrame) -> pd.DataFrame:
    """For short KWs, prepend titulo context for richer embeddings."""
    df = df.copy()
    df["kw_enriched"] = df.apply(
        lambda r: (
            f"{r['titulo']} {r['kw']}"
            if len(str(r["kw"]).split()) <= SHORT_KW_MAX_WORDS
            else str(r["kw"])
        ),
        axis=1,
    )
    return df


def _detect_outliers(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows in bottom 10th percentile of similarity scores."""
    if scores_df.empty:
        return pd.DataFrame()
    threshold = scores_df["kw_h2_sim_avg"].quantile(0.10)
    return scores_df[scores_df["kw_h2_sim_avg"] <= threshold].copy()


# ══════════════════════════════════════════════════════════
# PHASE 3: INTERLINKING
# ══════════════════════════════════════════════════════════


def _cluster_articles(df: pd.DataFrame, model_name: str) -> Tuple[pd.DataFrame, int]:
    """KMeans with auto-K selection via silhouette score."""
    combined = (df["titulo"].astype(str) + " " + df["kw"].astype(str)).tolist()
    embs = _compute_embeddings(combined, model_name)

    n = len(df)
    if n < 4:
        labels = list(range(n))
        df_out = df.copy()
        df_out["cluster"] = labels
        df_out["cluster_label"] = [f"Silo {l + 1}" for l in labels]
        return df_out, n

    max_k = min(10, n - 1)
    best_k = 2
    best_score = -1

    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embs)
        score = silhouette_score(embs, labels)
        if score > best_score:
            best_score = score
            best_k = k

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(embs)

    df_out = df.copy()
    df_out["cluster"] = labels
    df_out["cluster_label"] = [f"Silo {l + 1}" for l in labels]

    return df_out, best_k


def _suggest_internal_links(
    df: pd.DataFrame, model_name: str, max_links: int = 3
) -> pd.DataFrame:
    """Suggest top-N internal links prioritizing same-cluster articles."""
    combined = (df["titulo"].astype(str) + " " + df["kw"].astype(str)).tolist()
    embs = _compute_embeddings(combined, model_name)
    sim_matrix = cosine_similarity(embs)

    has_clusters = "cluster" in df.columns
    links = []

    for i in range(len(df)):
        candidates = []
        for j in range(len(df)):
            if i == j:
                continue
            sim = float(sim_matrix[i, j])
            same_cluster = bool(has_clusters and df.iloc[i]["cluster"] == df.iloc[j]["cluster"])
            # Boost same-cluster by 15%
            score = sim * 1.15 if same_cluster else sim
            candidates.append((j, sim, score, same_cluster))

        candidates.sort(key=lambda x: x[2], reverse=True)

        for j, sim, score, same_cluster in candidates[:max_links]:
            links.append({
                "source_idx": i,
                "source_titulo": df.iloc[i]["titulo"],
                "target_idx": j,
                "target_titulo": df.iloc[j]["titulo"],
                "target_kw": df.iloc[j]["kw"],
                "similitud": round(sim, 3),
                "score_final": round(score, 3),
                "mismo_silo": same_cluster,
            })

    return pd.DataFrame(links)


def _build_anchor_prompt(source_title: str, target_title: str, target_kw: str) -> str:
    """Build Gemini prompt for contextual anchor text."""
    return f"""Genera texto ancla para un enlace interno en español.

Artículo origen: {source_title}
Artículo destino: {target_title}
Keyword del destino: {target_kw}

Reglas:
1. El anchor debe ser natural, 2-6 palabras.
2. Debe incluir o sugerir la keyword del destino.
3. No uses "haz clic aquí" ni genéricos.
4. Proporciona 3 variantes con distinta formulación.

Devuelve SOLO JSON válido:
{{
  "anchors": ["variante1", "variante2", "variante3"],
  "context_sentence": "Frase de ejemplo donde se usaría el enlace"
}}"""


def _generate_anchor_texts(links_df: pd.DataFrame, model) -> pd.DataFrame:
    """Call Gemini to generate anchor texts for each unique link pair."""
    if links_df.empty:
        return links_df

    anchors_col = []
    context_col = []
    progress = st.progress(0, text="Generando anchor texts...")

    for i, row in links_df.iterrows():
        progress.progress((i + 1) / len(links_df), text=f"Anchor text {i + 1}/{len(links_df)}...")

        prompt = _build_anchor_prompt(row["source_titulo"], row["target_titulo"], row["target_kw"])
        try:
            response = model.generate_content(prompt)
            raw = getattr(response, "text", "") or ""
            parsed = _parse_gemini_json(raw)
            anchors_col.append(", ".join(parsed.get("anchors", [])))
            context_col.append(parsed.get("context_sentence", ""))
        except Exception:
            anchors_col.append("")
            context_col.append("")

        # Rate limiting
        time.sleep(0.3)

    progress.empty()

    result = links_df.copy()
    result["anchor_texts"] = anchors_col
    result["contexto"] = context_col
    return result


def _detect_reciprocal_links(links_df: pd.DataFrame) -> pd.DataFrame:
    """Find A→B and B→A pairs."""
    if links_df.empty:
        return pd.DataFrame()

    pairs_set = set()
    reciprocals = []

    for _, row in links_df.iterrows():
        pair = (row["source_idx"], row["target_idx"])
        reverse = (row["target_idx"], row["source_idx"])
        if reverse in pairs_set:
            reciprocals.append({
                "Artículo A": row["source_titulo"],
                "Artículo B": row["target_titulo"],
                "Similitud A→B": row["similitud"],
            })
        pairs_set.add(pair)

    return pd.DataFrame(reciprocals)


def _detect_orphan_articles(links_df: pd.DataFrame, total_rows: int) -> List[int]:
    """Articles not appearing as target in any link."""
    if links_df.empty:
        return list(range(total_rows))
    targeted = set(links_df["target_idx"].unique())
    return [i for i in range(total_rows) if i not in targeted]


# ══════════════════════════════════════════════════════════
# PHASE 4: VISUALIZATION & EXPORT
# ══════════════════════════════════════════════════════════


def _render_coverage_heatmap(df: pd.DataFrame, model_name: str) -> None:
    """UMAP/t-SNE 2D density plot."""
    combined = (df["titulo"].astype(str) + " " + df["kw"].astype(str)).tolist()
    embs = _compute_embeddings(combined, model_name)

    n = len(df)
    if n < 4:
        st.warning("Se necesitan al menos 4 artículos para la visualización.")
        return

    if UMAP_AVAILABLE and n >= 10:
        reducer = UMAP(n_components=2, n_neighbors=min(15, n - 1), min_dist=0.1, random_state=42)
        coords = reducer.fit_transform(embs)
        method = "UMAP"
    elif n >= 4:
        perp = min(30, max(2, n - 1))
        reducer = TSNE(n_components=2, perplexity=perp, random_state=42)
        coords = reducer.fit_transform(embs)
        method = "t-SNE"
    else:
        reducer = PCA(n_components=2, random_state=42)
        coords = reducer.fit_transform(embs)
        method = "PCA"

    plot_df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "Título": df["titulo"].tolist(),
        "KW": df["kw"].tolist(),
    })

    if "cluster_label" in df.columns:
        plot_df["Silo"] = df["cluster_label"].tolist()
        fig = px.scatter(
            plot_df, x="x", y="y", color="Silo", text="Título",
            title=f"Mapa de Cobertura Temática ({method})",
            hover_data={"KW": True, "x": False, "y": False},
        )
    else:
        fig = px.scatter(
            plot_df, x="x", y="y", text="Título",
            title=f"Mapa de Cobertura Temática ({method})",
            hover_data={"KW": True, "x": False, "y": False},
        )

    fig.update_traces(textposition="top center", marker=dict(size=12))
    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)


def _render_network_graph(links_df: pd.DataFrame, df: pd.DataFrame) -> None:
    """pyvis interactive network graph."""
    if links_df.empty:
        st.info("No hay enlaces sugeridos para visualizar.")
        return

    G = nx.DiGraph()

    # Add nodes
    for i, row in df.iterrows():
        label = str(row["titulo"])[:30]
        cluster = row.get("cluster_label", "Sin silo")
        G.add_node(i, label=label, title=f"{row['titulo']}\nKW: {row['kw']}\n{cluster}")

    # Add edges
    for _, link in links_df.iterrows():
        G.add_edge(
            link["source_idx"],
            link["target_idx"],
            weight=link["similitud"],
            title=f"Sim: {link['similitud']:.2f}",
        )

    net = Network(height="600px", width="100%", directed=True, notebook=False)
    net.from_nx(G)

    # Color by cluster
    colors = px.colors.qualitative.Set2
    if "cluster" in df.columns:
        cluster_ids = df["cluster"].unique()
        color_map = {c: colors[i % len(colors)] for i, c in enumerate(cluster_ids)}
        for node in net.nodes:
            idx = node["id"]
            if idx < len(df):
                cluster_id = df.iloc[idx].get("cluster", 0)
                node["color"] = color_map.get(cluster_id, "#97C2FC")

    net.set_options("""
    {
        "physics": {
            "enabled": true,
            "repulsion": {"centralGravity": 0.1, "springLength": 200, "nodeDistance": 250}
        }
    }
    """)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as f:
        net.save_graph(f.name)
        tmpfile = f.name

    with open(tmpfile, "r", encoding="utf-8") as f:
        html = f.read()
    st.components.v1.html(html, height=650)

    try:
        os.unlink(tmpfile)
    except Exception:
        pass


def _render_silo_health(df: pd.DataFrame, links_df: pd.DataFrame) -> None:
    """Bar chart of links per cluster, avg similarity."""
    if "cluster_label" not in df.columns or links_df.empty:
        st.info("Ejecuta primero el clustering y enlazado.")
        return

    cluster_stats = []
    for label in df["cluster_label"].unique():
        cluster_idxs = set(df[df["cluster_label"] == label].index)
        n_articles = len(cluster_idxs)

        # Links within this cluster
        internal_links = links_df[
            links_df["source_idx"].isin(cluster_idxs) & links_df["target_idx"].isin(cluster_idxs)
        ]
        avg_sim = float(internal_links["similitud"].mean()) if not internal_links.empty else 0

        # Orphans in cluster
        targeted_in_cluster = set(links_df[links_df["target_idx"].isin(cluster_idxs)]["target_idx"])
        orphans = len(cluster_idxs - targeted_in_cluster)

        cluster_stats.append({
            "Silo": label,
            "Artículos": n_articles,
            "Enlaces internos": len(internal_links),
            "Sim. promedio": round(avg_sim, 3),
            "Huérfanos": orphans,
        })

    stats_df = pd.DataFrame(cluster_stats)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

    fig = px.bar(
        stats_df, x="Silo", y="Enlaces internos", color="Sim. promedio",
        title="Salud de Silos: Enlaces internos por cluster",
        color_continuous_scale="RdYlGn",
    )
    st.plotly_chart(fig, use_container_width=True)


def _export_to_excel(
    df: pd.DataFrame,
    headings: List[Optional[dict]],
    scores: pd.DataFrame,
    links: pd.DataFrame,
    cannibalization: pd.DataFrame,
    ngram_df: pd.DataFrame,
) -> bytes:
    """Multi-sheet Excel export."""
    buffer = io.BytesIO()

    # Build main sheet
    main_rows = []
    for i, h in enumerate(headings):
        row_data = {
            "Título": df.iloc[i]["titulo"],
            "KW Principal": df.iloc[i]["kw"],
            "KW Secundarias": df.iloc[i]["kw_secundarias"],
        }
        if h:
            row_data["H1"] = h.get("h1", "")
            row_data["Meta Title"] = h.get("meta_title", "")
            row_data["Meta Description"] = h.get("meta_description", "")
            sections = h.get("sections", [])
            for j, s in enumerate(sections[:6]):
                row_data[f"H2_{j+1}"] = s.get("h2", "")
                h3s = s.get("h3s", [])
                for k, h3 in enumerate(h3s[:3]):
                    row_data[f"H2_{j+1}_H3_{k+1}"] = h3
        else:
            row_data["H1"] = "ERROR"

        if not scores.empty and i < len(scores):
            score_row = scores[scores["idx"] == i]
            if not score_row.empty:
                row_data["KW↔H2 Sim"] = score_row.iloc[0]["kw_h2_sim_avg"]
                row_data["H1↔H3 Sim"] = score_row.iloc[0]["h1_h3_sim_avg"]
                row_data["Estado"] = score_row.iloc[0]["status"]

        if "cluster_label" in df.columns:
            row_data["Silo"] = df.iloc[i]["cluster_label"]

        main_rows.append(row_data)

    main_df = pd.DataFrame(main_rows)

    # Clean illegal chars
    for col in main_df.select_dtypes(include=["object"]).columns:
        main_df[col] = main_df[col].apply(lambda x: _clean_illegal_excel_chars(str(x)) if pd.notna(x) else x)

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        main_df.to_excel(writer, sheet_name="Plan de Contenidos", index=False)
        if not scores.empty:
            scores.to_excel(writer, sheet_name="Validación Semántica", index=False)
        if not links.empty:
            links.to_excel(writer, sheet_name="Enlaces Sugeridos", index=False)
        if not cannibalization.empty:
            cannibalization.to_excel(writer, sheet_name="Canibalización", index=False)
        if not ngram_df.empty:
            ngram_df.to_excel(writer, sheet_name="N-gram Validation", index=False)

    buffer.seek(0)
    return buffer.getvalue()


def _export_cms_ready(df: pd.DataFrame, headings: List[Optional[dict]]) -> bytes:
    """JSON for WordPress import."""
    entries = []
    for i, h in enumerate(headings):
        if h is None:
            continue
        entry = {
            "titulo": df.iloc[i]["titulo"],
            "kw": df.iloc[i]["kw"],
            "meta_title": h.get("meta_title", ""),
            "meta_description": h.get("meta_description", ""),
            "h1": h.get("h1", ""),
            "estructura": h.get("sections", []),
        }
        entries.append(entry)

    return json.dumps(entries, ensure_ascii=False, indent=2).encode("utf-8")


def _detect_changes(new_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Compare re-upload: mark rows as new/changed/unchanged."""
    prev_hash = st.session_state.get(f"{SS}previous_upload_hash")
    new_hash = _compute_upload_hash(new_df)

    if prev_hash is None:
        st.session_state[f"{SS}previous_upload_hash"] = new_hash
        return None

    if prev_hash == new_hash:
        return None  # No changes

    old_df = st.session_state.get(f"{SS}uploaded_df")
    if old_df is None:
        st.session_state[f"{SS}previous_upload_hash"] = new_hash
        return None

    old_titles = set(old_df["titulo"].astype(str))
    changes = []
    for _, row in new_df.iterrows():
        titulo = str(row["titulo"])
        if titulo in old_titles:
            changes.append({"titulo": titulo, "estado": "existente"})
        else:
            changes.append({"titulo": titulo, "estado": "nuevo"})

    st.session_state[f"{SS}previous_upload_hash"] = new_hash
    return pd.DataFrame(changes)


# ══════════════════════════════════════════════════════════
# MAIN UI
# ══════════════════════════════════════════════════════════


def render_content_plan() -> None:
    """Main entry point for Content Plan Generator."""
    st.title("📋 Content Plan Generator")
    st.markdown(
        "Genera estructuras de encabezados SEO con IA, valida coherencia semántica, "
        "detecta canibalización y sugiere enlazado interno."
    )

    # Gemini check
    if genai is None:
        st.error("Se requiere `google-generativeai`. Instala con: `pip install google-generativeai`")
        return

    from gemini_utils import (
        configure_gemini,
        get_gemini_api_key,
        get_gemini_model,
        render_gemini_config_ui,
    )
    from semantic_tools import AVAILABLE_MODELS, MODEL_DESCRIPTIONS

    # Config sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ⚙️ Config Content Plan")
        api_key, model_name = render_gemini_config_ui(key_prefix="cp_")

        emb_model_key = st.selectbox(
            "Modelo embeddings",
            options=list(AVAILABLE_MODELS.keys()),
            index=0,
            format_func=lambda x: MODEL_DESCRIPTIONS[x],
            key=f"{SS}emb_model",
        )
        emb_model_name = AVAILABLE_MODELS[emb_model_key]

    # Tabs
    tab_gen, tab_analysis, tab_links, tab_export = st.tabs([
        "🔨 Generación",
        "🔍 Análisis",
        "🔗 Enlazado",
        "📥 Exportación",
    ])

    # ── Tab 1: Generation ──
    with tab_gen:
        st.header("1. Carga tu plan de contenidos")
        st.caption("El archivo debe tener columnas: **titulo**, **kw**, **kw_secundarias**")

        df = _upload_excel()
        if df is not None:
            st.session_state[f"{SS}uploaded_df"] = df

            # Version control
            changes = _detect_changes(df)
            if changes is not None:
                new_count = len(changes[changes["estado"] == "nuevo"])
                if new_count > 0:
                    st.info(f"📝 {new_count} artículos nuevos detectados respecto al upload anterior.")

            st.success(f"✅ {len(df)} artículos cargados")
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)

            # Token estimator
            est = _estimate_tokens(df)
            with st.expander("💰 Estimación de coste", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Artículos", est["n_rows"])
                c2.metric("Tokens aprox.", f"{est['est_total_tokens']:,}")
                c3.metric("Llamadas API", est["est_api_calls"])
                c4.metric("Coste aprox.", f"${est['est_cost_usd']:.4f}")

            # Enrich short KWs
            df = _enrich_short_keywords(df)
            short_count = (df["kw"].str.split().str.len() <= SHORT_KW_MAX_WORDS).sum()
            if short_count > 0:
                st.caption(f"🔤 {short_count} keywords cortas enriquecidas con contexto del título.")

            # Generate button
            gemini_key = get_gemini_api_key()
            gemini_model = get_gemini_model()

            if not gemini_key:
                st.warning("⚠️ Configura la API key de Gemini en la barra lateral.")
            else:
                checkpoint_idx = st.session_state.get(f"{SS}checkpoint_index", 0)
                if checkpoint_idx > 0 and checkpoint_idx < len(df):
                    st.info(f"📌 Checkpoint: {checkpoint_idx}/{len(df)} artículos ya procesados.")

                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    generate = st.button(
                        "🚀 Generar encabezados" if checkpoint_idx == 0 else f"▶️ Continuar desde {checkpoint_idx}",
                        type="primary",
                        key=f"{SS}generate_btn",
                    )
                with col_btn2:
                    if checkpoint_idx > 0:
                        if st.button("🔄 Reiniciar generación", key=f"{SS}reset_btn"):
                            for k in [f"{SS}generated_headings", f"{SS}generation_errors", f"{SS}checkpoint_index"]:
                                st.session_state.pop(k, None)
                            st.rerun()

                if generate:
                    if not configure_gemini(gemini_key):
                        st.error("Error configurando Gemini.")
                    else:
                        model = genai.GenerativeModel(gemini_model.strip() or "gemini-2.0-flash-exp")
                        with st.spinner("Generando encabezados..."):
                            headings, errors = _batch_generate_headings(df, model)

                        st.session_state[f"{SS}generated_headings"] = headings
                        generated_count = sum(1 for h in headings if h is not None)
                        st.success(f"✅ {generated_count}/{len(df)} artículos generados correctamente.")

                        if errors:
                            with st.expander(f"⚠️ {len(errors)} errores", expanded=False):
                                for err in errors:
                                    st.warning(f"Fila {err['row']}: {err['titulo']} → {err['error']}")

            # Show generated headings
            headings = st.session_state.get(f"{SS}generated_headings")
            if headings:
                st.markdown("---")
                st.subheader("📄 Encabezados generados")

                for i, h in enumerate(headings):
                    if h is None:
                        continue
                    with st.expander(f"**{df.iloc[i]['titulo']}**", expanded=False):
                        st.markdown(f"### {h.get('h1', '')}")
                        for s in h.get("sections", []):
                            st.markdown(f"#### {s.get('h2', '')}")
                            for h3 in s.get("h3s", []):
                                st.markdown(f"- {h3}")
                        st.caption(f"Meta Title: {h.get('meta_title', '')} | Meta Desc: {h.get('meta_description', '')}")

    # ── Tab 2: Analysis ──
    with tab_analysis:
        st.header("2. Análisis de calidad")

        df = st.session_state.get(f"{SS}uploaded_df")
        headings = st.session_state.get(f"{SS}generated_headings")

        if df is None or headings is None:
            st.info("⬅️ Primero genera los encabezados en la pestaña Generación.")
        else:
            if st.button("🔍 Ejecutar análisis completo", key=f"{SS}run_analysis", type="primary"):
                with st.spinner("Validando coherencia semántica..."):
                    scores = _validate_semantic_coherence(df, headings, emb_model_name)
                    st.session_state[f"{SS}validation_scores"] = scores

                with st.spinner("Detectando canibalización..."):
                    canib = _detect_cannibalization(df, emb_model_name)
                    st.session_state[f"{SS}cannibalization_pairs"] = canib

                with st.spinner("Validando N-grams..."):
                    ngram = _ngram_validation(headings, df)
                    st.session_state[f"{SS}ngram_validation"] = ngram

                with st.spinner("Detectando deriva semántica..."):
                    drift = _detect_semantic_drift(headings, emb_model_name)
                    st.session_state[f"{SS}semantic_drift"] = drift

                st.success("✅ Análisis completado.")

            # Display results
            scores = st.session_state.get(f"{SS}validation_scores")
            canib = st.session_state.get(f"{SS}cannibalization_pairs")
            ngram = st.session_state.get(f"{SS}ngram_validation")
            drift = st.session_state.get(f"{SS}semantic_drift")

            if scores is not None and not scores.empty:
                st.subheader("📊 Coherencia Semántica")

                # Color-coded summary
                ok_count = len(scores[scores["status"] == "ok"])
                warn_count = len(scores[scores["status"] != "ok"])
                c1, c2, c3 = st.columns(3)
                c1.metric("✅ Coherentes", ok_count)
                c2.metric("⚠️ Con alertas", warn_count)
                c3.metric("Sim KW↔H2 promedio", f"{scores['kw_h2_sim_avg'].mean():.3f}")

                st.dataframe(
                    scores.style.background_gradient(subset=["kw_h2_sim_avg", "h1_h3_sim_avg"], cmap="RdYlGn", vmin=0, vmax=1),
                    use_container_width=True,
                    hide_index=True,
                )

                # Outliers
                outliers = _detect_outliers(scores)
                if not outliers.empty:
                    with st.expander(f"🔍 {len(outliers)} outliers detectados (baja similitud)", expanded=False):
                        st.dataframe(outliers, use_container_width=True, hide_index=True)

            if canib is not None and not canib.empty:
                st.subheader("🚨 Canibalización detectada")
                st.warning(f"Se detectaron {len(canib)} pares de artículos potencialmente canibalizantes.")
                st.dataframe(
                    canib.style.background_gradient(subset=["Similitud"], cmap="Reds", vmin=0.8, vmax=1),
                    use_container_width=True,
                    hide_index=True,
                )
            elif canib is not None:
                st.success("✅ No se detectó canibalización.")

            if ngram is not None and not ngram.empty:
                st.subheader("📝 Validación de KW Secundarias (N-grams)")
                missing = ngram[~ngram["presente"]]
                present = ngram[ngram["presente"]]
                c1, c2 = st.columns(2)
                c1.metric("✅ Presentes", len(present))
                c2.metric("❌ Ausentes", len(missing))
                if not missing.empty:
                    with st.expander("KW secundarias no encontradas en encabezados"):
                        st.dataframe(missing, use_container_width=True, hide_index=True)

            if drift is not None and not drift.empty:
                st.subheader("🌊 Deriva Semántica H1↔H3")
                drifted = drift[drift["drift"]]
                if not drifted.empty:
                    st.warning(f"{len(drifted)} H3 con baja similitud al H1 (posible contenido de relleno).")
                    st.dataframe(
                        drifted.style.background_gradient(subset=["h1_h3_sim"], cmap="RdYlGn", vmin=0, vmax=1),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.success("✅ Todos los H3 mantienen coherencia con el H1.")

    # ── Tab 3: Interlinking ──
    with tab_links:
        st.header("3. Enlazado interno")

        df = st.session_state.get(f"{SS}uploaded_df")
        headings = st.session_state.get(f"{SS}generated_headings")

        if df is None or headings is None:
            st.info("⬅️ Primero genera los encabezados en la pestaña Generación.")
        else:
            max_links = st.slider("Enlaces máximos por artículo", 1, 10, 3, key=f"{SS}max_links")

            if st.button("🔗 Generar enlazado interno", key=f"{SS}run_linking", type="primary"):
                with st.spinner("Clusterizando artículos..."):
                    df_clustered, n_clusters = _cluster_articles(df, emb_model_name)
                    st.session_state[f"{SS}uploaded_df"] = df_clustered
                    st.session_state[f"{SS}clusters"] = df_clustered
                    df = df_clustered

                st.success(f"📊 {n_clusters} silos temáticos detectados.")

                with st.spinner("Sugiriendo enlaces internos..."):
                    links = _suggest_internal_links(df, emb_model_name, max_links)
                    st.session_state[f"{SS}link_suggestions"] = links

                st.success(f"🔗 {len(links)} enlaces sugeridos.")

                # Orphan detection
                orphans = _detect_orphan_articles(links, len(df))
                st.session_state[f"{SS}orphan_articles"] = orphans

            # Display results
            links = st.session_state.get(f"{SS}link_suggestions")
            orphans = st.session_state.get(f"{SS}orphan_articles")
            df = st.session_state.get(f"{SS}uploaded_df")

            if links is not None and not links.empty:
                st.subheader("📋 Enlaces sugeridos")
                st.dataframe(
                    links[["source_titulo", "target_titulo", "similitud", "mismo_silo", "anchor_texts"]].head(50)
                    if "anchor_texts" in links.columns
                    else links[["source_titulo", "target_titulo", "similitud", "mismo_silo"]].head(50),
                    use_container_width=True,
                    hide_index=True,
                )

                # Reciprocal detection
                reciprocals = _detect_reciprocal_links(links)
                if not reciprocals.empty:
                    with st.expander(f"⚠️ {len(reciprocals)} enlaces recíprocos detectados"):
                        st.caption("Enlaces bidireccionales (A↔B) pueden parecer poco naturales para Google.")
                        st.dataframe(reciprocals, use_container_width=True, hide_index=True)

                # Generate anchor texts
                if "anchor_texts" not in links.columns:
                    gemini_key = get_gemini_api_key()
                    if gemini_key and st.button("✍️ Generar anchor texts con Gemini", key=f"{SS}gen_anchors"):
                        if configure_gemini(gemini_key):
                            gemini_model = get_gemini_model()
                            model = genai.GenerativeModel(gemini_model.strip() or "gemini-2.0-flash-exp")
                            links_with_anchors = _generate_anchor_texts(links, model)
                            st.session_state[f"{SS}link_suggestions"] = links_with_anchors
                            st.rerun()

            if orphans is not None and len(orphans) > 0 and df is not None:
                with st.expander(f"🚨 {len(orphans)} artículos huérfanos (sin enlaces entrantes)"):
                    for idx in orphans:
                        if idx < len(df):
                            st.write(f"- {df.iloc[idx]['titulo']}")

    # ── Tab 4: Export ──
    with tab_export:
        st.header("4. Visualización y Exportación")

        df = st.session_state.get(f"{SS}uploaded_df")
        headings = st.session_state.get(f"{SS}generated_headings")

        if df is None or headings is None:
            st.info("⬅️ Primero genera los encabezados en la pestaña Generación.")
        else:
            # Visualizations
            st.subheader("🗺️ Mapa de Cobertura Temática")
            with st.spinner("Generando mapa..."):
                _render_coverage_heatmap(df, emb_model_name)

            links = st.session_state.get(f"{SS}link_suggestions")
            if links is not None and not links.empty:
                st.subheader("🕸️ Grafo de Enlazado Interno")
                with st.spinner("Generando grafo..."):
                    _render_network_graph(links, df)

                st.subheader("📊 Salud de Silos")
                _render_silo_health(df, links)

            # Export
            st.markdown("---")
            st.subheader("📥 Descargar resultados")

            scores = st.session_state.get(f"{SS}validation_scores", pd.DataFrame())
            links = st.session_state.get(f"{SS}link_suggestions", pd.DataFrame())
            canib = st.session_state.get(f"{SS}cannibalization_pairs", pd.DataFrame())
            ngram = st.session_state.get(f"{SS}ngram_validation", pd.DataFrame())

            col_exp1, col_exp2 = st.columns(2)

            with col_exp1:
                excel_bytes = _export_to_excel(df, headings, scores, links, canib, ngram)
                st.download_button(
                    "📥 Descargar Excel completo",
                    data=excel_bytes,
                    file_name="content_plan_export.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            with col_exp2:
                cms_bytes = _export_cms_ready(df, headings)
                st.download_button(
                    "📥 Descargar JSON para CMS",
                    data=cms_bytes,
                    file_name="content_plan_cms.json",
                    mime="application/json",
                )
