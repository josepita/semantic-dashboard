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
from datetime import datetime
from pathlib import Path
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
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None

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
_SAVE_FILENAME = "content_plan_session.json"


# ══════════════════════════════════════════════════════════
# SESSION PERSISTENCE
# ══════════════════════════════════════════════════════════


def _get_save_path() -> Optional[str]:
    """Return path to session file inside the current project directory, or None."""
    cfg = st.session_state.get("project_config")
    if not cfg:
        return None
    db = cfg.get("db_path", "")
    if not db:
        return None
    return str(Path(db).parent / _SAVE_FILENAME)


def _save_session_to_disk() -> None:
    """Persist current headings, errors, checkpoint and uploaded_df to a JSON file."""
    save_path = _get_save_path()
    if save_path is None:
        return

    headings = st.session_state.get(f"{SS}generated_headings")
    errors = st.session_state.get(f"{SS}generation_errors", [])
    checkpoint = st.session_state.get(f"{SS}checkpoint_index", 0)
    df = st.session_state.get(f"{SS}uploaded_df")

    if headings is None and df is None:
        return

    payload = {
        "headings": headings,
        "errors": errors,
        "checkpoint_index": checkpoint,
        "uploaded_df": df.to_dict(orient="records") if df is not None else None,
        "saved_at": datetime.now().isoformat(),
    }

    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # fail silently — disk save is best-effort


def _load_session_from_disk() -> bool:
    """Restore headings/errors/df from disk into session_state. Returns True if restored."""
    save_path = _get_save_path()
    if save_path is None or not Path(save_path).exists():
        return False

    # Don't overwrite if session already has headings
    if st.session_state.get(f"{SS}generated_headings"):
        return False

    try:
        with open(save_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return False

    headings = payload.get("headings")
    if not headings:
        return False

    st.session_state[f"{SS}generated_headings"] = headings
    st.session_state[f"{SS}generation_errors"] = payload.get("errors", [])
    st.session_state[f"{SS}checkpoint_index"] = payload.get("checkpoint_index", 0)

    df_records = payload.get("uploaded_df")
    if df_records:
        st.session_state[f"{SS}uploaded_df"] = pd.DataFrame(df_records)

    return True


def _delete_session_from_disk() -> None:
    """Remove the saved session file."""
    save_path = _get_save_path()
    if save_path and Path(save_path).exists():
        try:
            Path(save_path).unlink()
        except Exception:
            pass


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
# LLM ABSTRACTION (OpenAI / Gemini)
# ══════════════════════════════════════════════════════════

PROVIDER_GEMINI = "gemini"
PROVIDER_OPENAI = "openai"

OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
]


def _call_llm(prompt: str) -> str:
    """
    Call the configured LLM provider and return the raw text response.
    Reads provider/key/model from session_state.
    Raises Exception on failure.
    """
    provider = st.session_state.get(f"{SS}llm_provider", PROVIDER_GEMINI)

    if provider == PROVIDER_OPENAI:
        api_key = st.session_state.get("openai_api_key", "")
        model_name = st.session_state.get("openai_model", "gpt-4o-mini")
        if not api_key:
            raise ValueError("API key de OpenAI no configurada")
        if OpenAI is None:
            raise ImportError("openai no instalado")

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content or ""

    else:  # Gemini
        api_key = st.session_state.get("gemini_api_key", "")
        model_name = st.session_state.get("gemini_model_name", "gemini-2.5-flash")
        if not api_key:
            raise ValueError("API key de Gemini no configurada")
        if genai is None:
            raise ImportError("google-generativeai no instalado. Instala con: pip install google-generativeai")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name.strip() or "gemini-2.5-flash")
        response = model.generate_content(prompt)

        # Check safety filter
        if response.candidates and hasattr(response.candidates[0], "finish_reason"):
            reason = str(response.candidates[0].finish_reason)
            if "SAFETY" in reason.upper():
                raise ValueError("Bloqueado por filtro de seguridad de IA")

        raw = getattr(response, "text", "") or ""
        if not raw and response.candidates:
            raw = "".join(
                part.text
                for part in response.candidates[0].content.parts
                if hasattr(part, "text")
            )
        return raw


def _render_llm_config_sidebar() -> None:
    """Render LLM provider selector in sidebar (keys are set in API Settings page)."""
    st.markdown("#### 🤖 Proveedor de IA")

    # Detect which providers are configured
    has_openai = bool(st.session_state.get("openai_api_key"))
    has_gemini = bool(st.session_state.get("gemini_api_key"))

    if not has_openai and not has_gemini:
        st.warning("⚠️ Configura al menos una API en **🔑 Configuración API**.")
        return

    available = []
    if has_openai:
        available.append(PROVIDER_OPENAI)
    if has_gemini:
        available.append(PROVIDER_GEMINI)

    provider = st.radio(
        "Usar",
        options=available,
        format_func=lambda x: "Google Gemini" if x == PROVIDER_GEMINI else "OpenAI (GPT)",
        key=f"{SS}llm_provider",
        horizontal=True,
    )

    if provider == PROVIDER_OPENAI:
        model = st.session_state.get("openai_model", "gpt-4o-mini")
        st.caption(f"Modelo: **{model}**")
    else:
        model = st.session_state.get("gemini_model_name", "gemini-2.5-flash")
        st.caption(f"Modelo: **{model}**")

    st.success(f"✅ {provider.capitalize()} listo")


def _is_llm_configured() -> bool:
    """Check if the selected LLM provider is configured."""
    provider = st.session_state.get(f"{SS}llm_provider")
    if not provider:
        # No provider selected yet — check if any key exists
        return bool(st.session_state.get("openai_api_key") or st.session_state.get("gemini_api_key"))
    if provider == PROVIDER_OPENAI:
        return bool(st.session_state.get("openai_api_key"))
    else:
        return bool(st.session_state.get("gemini_api_key"))


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
        st.session_state.pop(f"{SS}confirmed_df", None)
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

    if st.button("✅ Confirmar columnas", key=f"{SS}confirm_cols", type="primary"):
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

        st.session_state[f"{SS}confirmed_df"] = df
        return df

    # Return previously confirmed df if it exists (persists across reruns)
    if f"{SS}confirmed_df" in st.session_state:
        return st.session_state[f"{SS}confirmed_df"]

    st.caption("Selecciona las columnas y pulsa confirmar para continuar.")
    return None


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


def _build_heading_prompt(titulo: str, kw_principal: str, kw_secundarias: str, user_nuances: str = "") -> str:
    """Build Gemini prompt for H1/H2/H3 structured generation."""
    nuances_block = ""
    if user_nuances and user_nuances.strip():
        nuances_block = f"""

Instrucciones adicionales del usuario (aplícalas con prioridad):
{user_nuances.strip()}
"""

    return f"""Eres un experto en arquitectura de contenido SEO en español.

Artículo planificado:
- Título: {titulo}
- Keyword principal: {kw_principal}
- Keywords secundarias: {kw_secundarias}
{nuances_block}
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
    titulo: str, kw_principal: str, kw_secundarias: str, user_nuances: str = "",
    custom_prompt: str = "",
) -> Tuple[Optional[dict], Optional[str]]:
    """
    Call LLM for one row with retry logic.
    Returns (parsed_dict, error_string).
    If custom_prompt is provided, it is used directly instead of building one.
    """
    prompt = custom_prompt if custom_prompt.strip() else _build_heading_prompt(
        titulo, kw_principal, kw_secundarias, user_nuances
    )

    for attempt in range(MAX_RETRIES):
        try:
            raw = _call_llm(prompt)

            if not raw.strip():
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1)
                    continue
                return None, "Respuesta vacía del modelo"

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


def _batch_generate_headings(
    df: pd.DataFrame, user_nuances: str = "", custom_prompt: str = "",
    only_rows: Optional[List[int]] = None,
) -> Tuple[List[Optional[dict]], List[dict]]:
    """
    Process all rows (or only specific rows) with checkpoint saves.
    Returns (headings_list, errors_list).
    """
    n = len(df)
    headings = st.session_state.get(f"{SS}generated_headings", [None] * n)
    errors = st.session_state.get(f"{SS}generation_errors", [])
    start_idx = st.session_state.get(f"{SS}checkpoint_index", 0)

    if len(headings) < n:
        headings.extend([None] * (n - len(headings)))

    # Determine which rows to process
    if only_rows is not None:
        rows_to_process = [i for i in only_rows if 0 <= i < n]
    else:
        rows_to_process = list(range(start_idx, n))

    total_to_process = len(rows_to_process)
    if total_to_process == 0:
        return headings, errors

    progress = st.progress(0, text=f"Procesando 0/{total_to_process}...")

    for step, i in enumerate(rows_to_process):
        row = df.iloc[i]
        progress.progress((step + 1) / total_to_process, text=f"Procesando {step + 1}/{total_to_process}: {row['titulo'][:50]}...")

        result, error = _generate_headings_for_row(
            str(row["titulo"]),
            str(row["kw"]),
            str(row["kw_secundarias"]),
            user_nuances,
            custom_prompt,
        )

        headings[i] = result
        if error:
            errors.append({"row": i, "titulo": row["titulo"], "error": error})
        else:
            # Remove previous error for this row if retry succeeded
            errors = [e for e in errors if e["row"] != i]

        # Checkpoint
        if (step + 1) % CHECKPOINT_INTERVAL == 0 or step == total_to_process - 1:
            st.session_state[f"{SS}generated_headings"] = headings
            st.session_state[f"{SS}generation_errors"] = errors
            if only_rows is None:
                st.session_state[f"{SS}checkpoint_index"] = i + 1
            _save_session_to_disk()

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


def _build_alternative_titles_prompt(titulo: str, kw_principal: str, kw_secundarias: str) -> str:
    """Build prompt for generating alternative titles."""
    return f"""Eres un experto en SEO y copywriting en español.

Artículo planificado:
- Título actual: {titulo}
- Keyword principal: {kw_principal}
- Keywords secundarias: {kw_secundarias}

Genera 5 títulos alternativos optimizados para SEO y CTR.

Reglas:
1. Cada título debe incluir la keyword principal de forma natural.
2. Varía el enfoque: informativo, lista, pregunta, cómo hacer, beneficios.
3. Máximo 60-65 caracteres por título.
4. Deben ser atractivos para el usuario y optimizados para buscadores.
5. No repitas el mismo patrón en todos los títulos.

Devuelve SOLO JSON válido:
{{
  "titulos_alternativos": ["título 1", "título 2", "título 3", "título 4", "título 5"]
}}"""


def _build_search_queries_prompt(titulo: str, kw_principal: str, kw_secundarias: str) -> str:
    """Build prompt for generating related search queries."""
    return f"""Eres un experto en SEO y análisis de intención de búsqueda en español.

Artículo planificado:
- Título: {titulo}
- Keyword principal: {kw_principal}
- Keywords secundarias: {kw_secundarias}

Genera las búsquedas relacionadas que los usuarios podrían hacer en Google sobre este tema.

Incluye:
1. 3 queries informacionales (qué es, cómo funciona, guías)
2. 3 queries transaccionales (comprar, precio, mejor)
3. 2 queries comparativas (vs, diferencias, alternativas)
4. 2 queries long-tail específicas

Devuelve SOLO JSON válido:
{{
  "queries_informacionales": ["query 1", "query 2", "query 3"],
  "queries_transaccionales": ["query 1", "query 2", "query 3"],
  "queries_comparativas": ["query 1", "query 2"],
  "queries_longtail": ["query 1", "query 2"]
}}"""


def _generate_alternative_titles(titulo: str, kw_principal: str, kw_secundarias: str) -> Tuple[List[str], Optional[str]]:
    """Generate alternative titles using LLM."""
    prompt = _build_alternative_titles_prompt(titulo, kw_principal, kw_secundarias)

    for attempt in range(MAX_RETRIES):
        try:
            raw = _call_llm(prompt)
            if not raw.strip():
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1)
                    continue
                return [], "Respuesta vacía del modelo"

            parsed = _parse_gemini_json(raw)
            titles = parsed.get("titulos_alternativos", [])
            return titles, None

        except json.JSONDecodeError as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
                continue
            return [], f"JSON inválido: {e}"
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
                continue
            return [], f"Error API: {e}"

    return [], "Error desconocido"


def _generate_search_queries(titulo: str, kw_principal: str, kw_secundarias: str) -> Tuple[dict, Optional[str]]:
    """Generate search queries using LLM."""
    prompt = _build_search_queries_prompt(titulo, kw_principal, kw_secundarias)

    for attempt in range(MAX_RETRIES):
        try:
            raw = _call_llm(prompt)
            if not raw.strip():
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1)
                    continue
                return {}, "Respuesta vacía del modelo"

            parsed = _parse_gemini_json(raw)
            return parsed, None

        except json.JSONDecodeError as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
                continue
            return {}, f"JSON inválido: {e}"
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
                continue
            return {}, f"Error API: {e}"

    return {}, "Error desconocido"


def _batch_generate_extras(
    df: pd.DataFrame,
    generate_titles: bool = True,
    generate_queries: bool = True
) -> Tuple[List[List[str]], List[dict]]:
    """Generate alternative titles and search queries for all rows."""
    n = len(df)
    alt_titles_list: List[List[str]] = [[] for _ in range(n)]
    queries_list: List[dict] = [{} for _ in range(n)]

    total_ops = n * (int(generate_titles) + int(generate_queries))
    if total_ops == 0:
        return alt_titles_list, queries_list

    progress = st.progress(0, text="Generando contenido adicional...")
    step = 0

    for i in range(n):
        row = df.iloc[i]
        titulo = str(row["titulo"])
        kw = str(row["kw"])
        kw_sec = str(row["kw_secundarias"])

        if generate_titles:
            step += 1
            progress.progress(step / total_ops, text=f"Títulos alternativos {i+1}/{n}: {titulo[:40]}...")
            titles, _ = _generate_alternative_titles(titulo, kw, kw_sec)
            alt_titles_list[i] = titles
            time.sleep(0.3)  # Rate limiting

        if generate_queries:
            step += 1
            progress.progress(step / total_ops, text=f"Querys de búsqueda {i+1}/{n}: {titulo[:40]}...")
            queries, _ = _generate_search_queries(titulo, kw, kw_sec)
            queries_list[i] = queries
            time.sleep(0.3)  # Rate limiting

    progress.empty()
    return alt_titles_list, queries_list


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


def _generate_anchor_texts(links_df: pd.DataFrame) -> pd.DataFrame:
    """Call LLM to generate anchor texts for each unique link pair."""
    if links_df.empty:
        return links_df

    anchors_col = []
    context_col = []
    progress = st.progress(0, text="Generando anchor texts...")

    for i, row in links_df.iterrows():
        progress.progress((i + 1) / len(links_df), text=f"Anchor text {i + 1}/{len(links_df)}...")

        prompt = _build_anchor_prompt(row["source_titulo"], row["target_titulo"], row["target_kw"])
        try:
            raw = _call_llm(prompt)
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
    alt_titles: Optional[List[List[str]]] = None,
    search_queries: Optional[List[dict]] = None,
    manual_queries: Optional[Dict[int, str]] = None,
) -> bytes:
    """Multi-sheet Excel export with optional extra columns."""
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

        # Add alternative titles
        if alt_titles and i < len(alt_titles) and alt_titles[i]:
            row_data["Títulos Alternativos"] = " | ".join(alt_titles[i])
            # Also add individual columns for each alternative
            for t_idx, alt_title in enumerate(alt_titles[i][:5]):
                row_data[f"Título Alt {t_idx+1}"] = alt_title

        # Add search queries (from IA or manual)
        queries_text = ""
        if search_queries and i < len(search_queries) and search_queries[i]:
            q = search_queries[i]
            all_queries = []
            all_queries.extend(q.get("queries_informacionales", []))
            all_queries.extend(q.get("queries_transaccionales", []))
            all_queries.extend(q.get("queries_comparativas", []))
            all_queries.extend(q.get("queries_longtail", []))
            queries_text = " | ".join(all_queries)
            # Separate columns by type
            row_data["Querys Informacionales"] = ", ".join(q.get("queries_informacionales", []))
            row_data["Querys Transaccionales"] = ", ".join(q.get("queries_transaccionales", []))
            row_data["Querys Comparativas"] = ", ".join(q.get("queries_comparativas", []))
            row_data["Querys Long-tail"] = ", ".join(q.get("queries_longtail", []))

        # Add manual queries if provided
        if manual_queries and i in manual_queries:
            row_data["Querys Manuales"] = manual_queries[i]

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

    # Restore previous session from disk if available
    restored = _load_session_from_disk()
    if restored:
        saved_path = _get_save_path()
        saved_at = ""
        if saved_path and Path(saved_path).exists():
            try:
                with open(saved_path, "r", encoding="utf-8") as _f:
                    saved_at = json.load(_f).get("saved_at", "")
            except Exception:
                pass
        headings_count = sum(1 for h in st.session_state.get(f"{SS}generated_headings", []) if h is not None)
        st.success(
            f"Se ha restaurado una sesion anterior con {headings_count} encabezados generados."
            + (f" (guardado: {saved_at[:19].replace('T', ' ')})" if saved_at else "")
        )

    from semantic_tools import AVAILABLE_MODELS, MODEL_DESCRIPTIONS

    # Config sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ⚙️ Config Content Plan")
        _render_llm_config_sidebar()

        emb_model_key = st.selectbox(
            "Modelo embeddings",
            options=list(AVAILABLE_MODELS.keys()),
            index=0,
            format_func=lambda x: MODEL_DESCRIPTIONS[x],
            key=f"{SS}emb_model",
        )
        emb_model_name = AVAILABLE_MODELS[emb_model_key]

    # Tabs
    tab_gen, tab_extras, tab_analysis, tab_links, tab_export = st.tabs([
        "🔨 Generación",
        "✨ Extras (Títulos/Querys)",
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

            # ── Prompt: show full editable prompt ──
            st.markdown("---")
            st.subheader("✏️ Prompt de generación")
            st.caption(
                "Este es el prompt completo que se enviará a la IA para cada artículo. "
                "Las variables `{titulo}`, `{kw_principal}` y `{kw_secundarias}` se sustituyen automáticamente por los datos de cada fila. "
                "Puedes modificarlo directamente para ajustar tono, estructura o reglas."
            )

            # Build default prompt with placeholders for display
            _default_prompt = _build_heading_prompt("{titulo}", "{kw_principal}", "{kw_secundarias}")

            # Initialize editable prompt in session if not present
            if f"{SS}custom_prompt_template" not in st.session_state:
                st.session_state[f"{SS}custom_prompt_template"] = _default_prompt

            custom_prompt_template = st.text_area(
                "Prompt (editable)",
                value=st.session_state.get(f"{SS}custom_prompt_template", _default_prompt),
                height=350,
                key=f"{SS}custom_prompt_template",
            )

            if st.button("↩️ Restaurar prompt por defecto", key=f"{SS}reset_prompt"):
                st.session_state[f"{SS}custom_prompt_template"] = _default_prompt
                st.rerun()

            # ── Preview: test with first row ──
            st.markdown("---")
            st.subheader("🔎 Vista previa (1 artículo de prueba)")
            st.caption("Genera la estructura para un artículo y revisa el resultado antes de procesar todo el archivo.")

            preview_row_idx = st.number_input(
                "Fila a previsualizar",
                min_value=0,
                max_value=len(df) - 1,
                value=0,
                key=f"{SS}preview_row_idx",
            )

            if not _is_llm_configured():
                st.warning("⚠️ Configura la API key del proveedor de IA en la barra lateral.")
            else:
                # Show current LLM config for transparency
                provider = st.session_state.get(f"{SS}llm_provider", PROVIDER_GEMINI)
                if provider == PROVIDER_OPENAI:
                    _active_model = st.session_state.get("openai_model", "gpt-4o-mini")
                else:
                    _active_model = st.session_state.get("gemini_model_name", "gemini-2.5-flash")
                st.caption(f"Proveedor: **{provider}** | Modelo: **{_active_model}**")

                if st.button("👁️ Generar vista previa", key=f"{SS}preview_btn"):
                    preview_row = df.iloc[preview_row_idx]

                    # Build the actual prompt for this row from the template
                    _preview_prompt = custom_prompt_template.replace(
                        "{titulo}", str(preview_row["titulo"])
                    ).replace(
                        "{kw_principal}", str(preview_row["kw"])
                    ).replace(
                        "{kw_secundarias}", str(preview_row["kw_secundarias"])
                    )

                    # Quick validation: test a minimal call to catch config errors early
                    with st.spinner(f"Verificando conexión con {_active_model}..."):
                        try:
                            _call_llm("Responde solo: OK")
                        except Exception as e:
                            err_msg = str(e)
                            st.error(f"Error de conexión con la API: {err_msg}")
                            if "not found" in err_msg.lower() or "not supported" in err_msg.lower():
                                st.warning(
                                    f"El modelo **{_active_model}** no está disponible. "
                                    "Ve a **Configuración API** y selecciona un modelo válido "
                                    "(ej: `gemini-2.5-flash`, `gemini-2.0-flash`)."
                                )
                            elif "api key" in err_msg.lower() or "401" in err_msg or "403" in err_msg:
                                st.warning("La API key parece inválida o expirada. Revísala en **Configuración API**.")
                            st.stop()

                    with st.spinner(f"Generando preview para: {preview_row['titulo'][:60]}..."):
                        preview_result, preview_error = _generate_headings_for_row(
                            str(preview_row["titulo"]),
                            str(preview_row["kw"]),
                            str(preview_row["kw_secundarias"]),
                            custom_prompt=_preview_prompt,
                        )

                    if preview_error:
                        st.error(f"Error en preview: {preview_error}")
                        if "not found" in preview_error.lower() or "not supported" in preview_error.lower():
                            st.warning(
                                f"El modelo **{_active_model}** no está disponible. "
                                "Cambia el modelo en **Configuración API**."
                            )
                    else:
                        st.session_state[f"{SS}preview_result"] = preview_result
                        st.session_state[f"{SS}preview_row"] = preview_row

                # Show preview result
                preview_result = st.session_state.get(f"{SS}preview_result")
                preview_row_data = st.session_state.get(f"{SS}preview_row")
                if preview_result and preview_row_data is not None:
                    with st.container(border=True):
                        st.markdown(f"**Artículo:** {preview_row_data['titulo']}")
                        st.markdown(f"**KW:** {preview_row_data['kw']}")
                        st.markdown("---")
                        st.markdown(f"### {preview_result.get('h1', '')}")
                        for s in preview_result.get("sections", []):
                            st.markdown(f"#### {s.get('h2', '')}")
                            for h3 in s.get("h3s", []):
                                st.markdown(f"- {h3}")
                        st.caption(
                            f"Meta Title: {preview_result.get('meta_title', '')} | "
                            f"Meta Desc: {preview_result.get('meta_description', '')}"
                        )
                    st.info("Si el resultado es correcto, pulsa **Generar encabezados** para procesar todo el archivo.")

                # ── Full generation ──
                st.markdown("---")
                st.subheader("🚀 Generación completa")

                checkpoint_idx = st.session_state.get(f"{SS}checkpoint_index", 0)
                if checkpoint_idx > 0 and checkpoint_idx < len(df):
                    st.info(f"📌 Checkpoint: {checkpoint_idx}/{len(df)} artículos ya procesados.")

                col_btn1, col_btn2, col_btn3 = st.columns(3)
                with col_btn1:
                    generate = st.button(
                        "🚀 Generar encabezados" if checkpoint_idx == 0 else f"▶️ Continuar desde {checkpoint_idx}",
                        type="primary",
                        key=f"{SS}generate_btn",
                    )
                with col_btn2:
                    # Retry failed rows button
                    prev_errors = st.session_state.get(f"{SS}generation_errors", [])
                    retry_failed = False
                    if prev_errors:
                        retry_failed = st.button(
                            f"🔁 Reintentar {len(prev_errors)} fallidos",
                            key=f"{SS}retry_failed_btn",
                        )
                with col_btn3:
                    if checkpoint_idx > 0:
                        if st.button("🔄 Reiniciar todo", key=f"{SS}reset_btn"):
                            for k in [f"{SS}generated_headings", f"{SS}generation_errors", f"{SS}checkpoint_index"]:
                                st.session_state.pop(k, None)
                            _delete_session_from_disk()
                            st.rerun()

                # Helper to build per-row prompt from template
                def _build_row_prompt(row_data: pd.Series) -> str:
                    return custom_prompt_template.replace(
                        "{titulo}", str(row_data["titulo"])
                    ).replace(
                        "{kw_principal}", str(row_data["kw"])
                    ).replace(
                        "{kw_secundarias}", str(row_data["kw_secundarias"])
                    )

                if generate:
                    with st.spinner("Generando encabezados..."):
                        headings, errors = _batch_generate_headings(
                            df, custom_prompt=custom_prompt_template.replace(
                                "{titulo}", "{titulo}"  # keep as-is; handled per-row below
                            ),
                        )
                        # The template has placeholders — need per-row substitution.
                        # Re-process: _batch uses _generate_headings_for_row which builds
                        # prompt via _build_heading_prompt when custom_prompt is empty.
                        # We need to pass the actual custom prompt per row.
                        # Override: reset and process manually with per-row prompts.
                    # Actually, let's do it correctly: process each row with its prompt
                    n = len(df)
                    headings = st.session_state.get(f"{SS}generated_headings", [None] * n)
                    errors_list: List[dict] = []
                    start = st.session_state.get(f"{SS}checkpoint_index", 0)
                    progress = st.progress(0, text=f"Procesando 0/{n}...")
                    for i in range(start, n):
                        row = df.iloc[i]
                        progress.progress((i + 1) / n, text=f"Procesando {i + 1}/{n}: {row['titulo'][:50]}...")
                        row_prompt = _build_row_prompt(row)
                        result, error = _generate_headings_for_row(
                            str(row["titulo"]), str(row["kw"]), str(row["kw_secundarias"]),
                            custom_prompt=row_prompt,
                        )
                        headings[i] = result
                        if error:
                            errors_list.append({"row": i, "titulo": row["titulo"], "error": error})
                        if (i + 1) % CHECKPOINT_INTERVAL == 0 or i == n - 1:
                            st.session_state[f"{SS}generated_headings"] = headings
                            st.session_state[f"{SS}generation_errors"] = errors_list
                            st.session_state[f"{SS}checkpoint_index"] = i + 1
                            _save_session_to_disk()
                    progress.empty()

                    generated_count = sum(1 for h in headings if h is not None)
                    st.success(f"✅ {generated_count}/{n} artículos generados correctamente.")

                    if errors_list:
                        with st.expander(f"⚠️ {len(errors_list)} errores", expanded=True):
                            for err in errors_list:
                                st.warning(f"Fila {err['row']}: {err['titulo']} → {err['error']}")

                if retry_failed and prev_errors:
                    failed_indices = [e["row"] for e in prev_errors]
                    st.info(f"Reintentando {len(failed_indices)} artículos fallidos...")
                    # Clear old errors for these rows
                    st.session_state[f"{SS}generation_errors"] = [
                        e for e in st.session_state.get(f"{SS}generation_errors", [])
                        if e["row"] not in failed_indices
                    ]
                    headings = st.session_state.get(f"{SS}generated_headings", [None] * len(df))
                    retry_errors: List[dict] = []
                    progress = st.progress(0, text="Reintentando...")
                    for step, i in enumerate(failed_indices):
                        row = df.iloc[i]
                        progress.progress((step + 1) / len(failed_indices), text=f"Reintentando {step + 1}/{len(failed_indices)}: {row['titulo'][:50]}...")
                        row_prompt = _build_row_prompt(row)
                        result, error = _generate_headings_for_row(
                            str(row["titulo"]), str(row["kw"]), str(row["kw_secundarias"]),
                            custom_prompt=row_prompt,
                        )
                        headings[i] = result
                        if error:
                            retry_errors.append({"row": i, "titulo": row["titulo"], "error": error})
                    progress.empty()

                    st.session_state[f"{SS}generated_headings"] = headings
                    # Merge remaining errors
                    existing_errors = st.session_state.get(f"{SS}generation_errors", [])
                    st.session_state[f"{SS}generation_errors"] = existing_errors + retry_errors
                    _save_session_to_disk()

                    recovered = len(failed_indices) - len(retry_errors)
                    st.success(f"✅ {recovered}/{len(failed_indices)} artículos recuperados.")
                    if retry_errors:
                        with st.expander(f"⚠️ {len(retry_errors)} siguen fallando", expanded=True):
                            for err in retry_errors:
                                st.warning(f"Fila {err['row']}: {err['titulo']} → {err['error']}")
                    st.rerun()

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

    # ── Tab 2: Extras (Títulos Alternativos / Querys) ──
    with tab_extras:
        st.header("2. Títulos Alternativos y Querys de Búsqueda")
        st.markdown(
            "Genera títulos alternativos con IA y añade querys de búsqueda relacionadas "
            "(puedes generarlas con IA o introducirlas manualmente desde herramientas como Fancy Out)."
        )

        df = st.session_state.get(f"{SS}uploaded_df")
        headings = st.session_state.get(f"{SS}generated_headings")

        if df is None:
            st.info("⬅️ Primero carga un archivo en la pestaña Generación.")
        else:
            # ── Títulos Alternativos ──
            st.subheader("📝 Títulos Alternativos")
            st.caption("Genera 5 variantes de título para cada artículo usando IA.")

            alt_titles = st.session_state.get(f"{SS}alternative_titles", [])
            has_alt_titles = bool(alt_titles and any(alt_titles))

            col1, col2 = st.columns([2, 1])
            with col1:
                if not _is_llm_configured():
                    st.warning("⚠️ Configura la API key del proveedor de IA en la barra lateral.")
                else:
                    if st.button("🎯 Generar títulos alternativos", key=f"{SS}gen_alt_titles", type="primary"):
                        with st.spinner("Generando títulos alternativos..."):
                            alt_titles, _ = _batch_generate_extras(df, generate_titles=True, generate_queries=False)
                            st.session_state[f"{SS}alternative_titles"] = alt_titles
                        st.success(f"✅ Títulos generados para {len([t for t in alt_titles if t])} artículos.")
                        st.rerun()

            with col2:
                if has_alt_titles:
                    st.metric("Artículos con títulos", len([t for t in alt_titles if t]))

            # Show generated alternative titles
            if has_alt_titles:
                with st.expander("📋 Ver títulos alternativos generados", expanded=True):
                    for i, titles in enumerate(alt_titles):
                        if titles and i < len(df):
                            st.markdown(f"**{df.iloc[i]['titulo']}**")
                            for j, t in enumerate(titles[:5], 1):
                                st.markdown(f"  {j}. {t}")
                            st.markdown("---")

            # ── Querys de Búsqueda ──
            st.markdown("---")
            st.subheader("🔍 Querys de Búsqueda")

            query_mode = st.radio(
                "Modo de entrada de querys:",
                options=["manual", "ia", "ambos"],
                format_func=lambda x: {
                    "manual": "📋 Manual (pegar desde Fancy Out u otras herramientas)",
                    "ia": "🤖 Generar con IA",
                    "ambos": "📋+🤖 Combinar manual + IA"
                }[x],
                key=f"{SS}query_mode",
                horizontal=True,
            )

            # Manual queries input
            if query_mode in ["manual", "ambos"]:
                st.markdown("##### Querys manuales")
                st.caption(
                    "Pega las querys extraídas de Fancy Out u otras herramientas. "
                    "Puedes usar el selector para elegir el artículo."
                )

                # Initialize manual queries dict
                if f"{SS}manual_queries" not in st.session_state:
                    st.session_state[f"{SS}manual_queries"] = {}

                manual_queries = st.session_state[f"{SS}manual_queries"]

                # Row selector
                selected_row = st.selectbox(
                    "Selecciona artículo:",
                    options=range(len(df)),
                    format_func=lambda x: f"{x+1}. {df.iloc[x]['titulo'][:60]}...",
                    key=f"{SS}manual_query_row",
                )

                # Text area for manual input
                current_manual = manual_queries.get(selected_row, "")
                new_manual = st.text_area(
                    f"Querys para: {df.iloc[selected_row]['titulo'][:50]}...",
                    value=current_manual,
                    height=150,
                    placeholder="Pega aquí las querys, una por línea o separadas por comas...",
                    key=f"{SS}manual_query_input_{selected_row}",
                )

                if new_manual != current_manual:
                    st.session_state[f"{SS}manual_queries"][selected_row] = new_manual

                # Show count of articles with manual queries
                filled_count = len([q for q in manual_queries.values() if q.strip()])
                if filled_count > 0:
                    st.success(f"✅ {filled_count}/{len(df)} artículos con querys manuales.")

            # IA-generated queries
            if query_mode in ["ia", "ambos"]:
                st.markdown("##### Querys generadas con IA")

                search_queries = st.session_state.get(f"{SS}search_queries", [])
                has_search_queries = bool(search_queries and any(search_queries))

                if not _is_llm_configured():
                    st.warning("⚠️ Configura la API key del proveedor de IA en la barra lateral.")
                else:
                    if st.button("🔍 Generar querys con IA", key=f"{SS}gen_search_queries", type="primary"):
                        with st.spinner("Generando querys de búsqueda..."):
                            _, search_queries = _batch_generate_extras(df, generate_titles=False, generate_queries=True)
                            st.session_state[f"{SS}search_queries"] = search_queries
                        st.success(f"✅ Querys generadas para {len([q for q in search_queries if q])} artículos.")
                        st.rerun()

                # Show generated queries
                if has_search_queries:
                    with st.expander("📋 Ver querys generadas", expanded=True):
                        for i, queries in enumerate(search_queries):
                            if queries and i < len(df):
                                st.markdown(f"**{df.iloc[i]['titulo']}**")
                                if queries.get("queries_informacionales"):
                                    st.markdown(f"  *Informacionales:* {', '.join(queries['queries_informacionales'])}")
                                if queries.get("queries_transaccionales"):
                                    st.markdown(f"  *Transaccionales:* {', '.join(queries['queries_transaccionales'])}")
                                if queries.get("queries_comparativas"):
                                    st.markdown(f"  *Comparativas:* {', '.join(queries['queries_comparativas'])}")
                                if queries.get("queries_longtail"):
                                    st.markdown(f"  *Long-tail:* {', '.join(queries['queries_longtail'])}")
                                st.markdown("---")

            # ── Generate both at once ──
            st.markdown("---")
            st.subheader("⚡ Generación rápida")
            st.caption("Genera títulos alternativos y querys de IA en una sola operación.")

            if _is_llm_configured():
                if st.button("🚀 Generar todo (Títulos + Querys IA)", key=f"{SS}gen_all_extras"):
                    with st.spinner("Generando títulos y querys..."):
                        alt_titles, search_queries = _batch_generate_extras(df, generate_titles=True, generate_queries=True)
                        st.session_state[f"{SS}alternative_titles"] = alt_titles
                        st.session_state[f"{SS}search_queries"] = search_queries
                    st.success("✅ Títulos y querys generados correctamente.")
                    st.rerun()

    # ── Tab 3: Analysis ──
    with tab_analysis:
        st.header("2. Análisis de calidad")

        df = st.session_state.get(f"{SS}uploaded_df")
        headings = st.session_state.get(f"{SS}generated_headings")

        if df is None or headings is None:
            st.info("⬅️ Primero genera los encabezados en la pestaña Generación.")
        else:
            # Embedding model selector for analysis
            st.markdown("##### Modelo de embeddings para el análisis")
            analysis_emb_key = st.selectbox(
                "Modelo embeddings",
                options=list(AVAILABLE_MODELS.keys()),
                index=0,
                format_func=lambda x: MODEL_DESCRIPTIONS[x],
                key=f"{SS}analysis_emb_model",
            )
            analysis_emb_model = AVAILABLE_MODELS[analysis_emb_key]

            if st.button("🔍 Ejecutar análisis completo", key=f"{SS}run_analysis", type="primary"):
                with st.spinner("Validando coherencia semántica..."):
                    scores = _validate_semantic_coherence(df, headings, analysis_emb_model)
                    st.session_state[f"{SS}validation_scores"] = scores

                with st.spinner("Detectando canibalización..."):
                    canib = _detect_cannibalization(df, analysis_emb_model)
                    st.session_state[f"{SS}cannibalization_pairs"] = canib

                with st.spinner("Validando N-grams..."):
                    ngram = _ngram_validation(headings, df)
                    st.session_state[f"{SS}ngram_validation"] = ngram

                with st.spinner("Detectando deriva semántica..."):
                    drift = _detect_semantic_drift(headings, analysis_emb_model)
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
                    if _is_llm_configured() and st.button("✍️ Generar anchor texts con IA", key=f"{SS}gen_anchors"):
                        links_with_anchors = _generate_anchor_texts(links)
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

            # Get extras (alternative titles and search queries)
            alt_titles = st.session_state.get(f"{SS}alternative_titles", [])
            search_queries = st.session_state.get(f"{SS}search_queries", [])
            manual_queries = st.session_state.get(f"{SS}manual_queries", {})

            # Show what will be exported
            extras_info = []
            if alt_titles and any(alt_titles):
                extras_info.append(f"✅ Títulos alternativos ({len([t for t in alt_titles if t])} artículos)")
            if search_queries and any(search_queries):
                extras_info.append(f"✅ Querys IA ({len([q for q in search_queries if q])} artículos)")
            if manual_queries and any(manual_queries.values()):
                extras_info.append(f"✅ Querys manuales ({len([q for q in manual_queries.values() if q.strip()])} artículos)")

            if extras_info:
                st.info("**Columnas extras incluidas en el Excel:**\n" + "\n".join(extras_info))
            else:
                st.caption("💡 Puedes añadir títulos alternativos y querys en la pestaña **✨ Extras**.")

            col_exp1, col_exp2 = st.columns(2)

            with col_exp1:
                excel_bytes = _export_to_excel(
                    df, headings, scores, links, canib, ngram,
                    alt_titles=alt_titles if alt_titles else None,
                    search_queries=search_queries if search_queries else None,
                    manual_queries=manual_queries if manual_queries else None,
                )
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
