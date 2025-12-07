from __future__ import annotations

import csv
import importlib
import io
import math
import os
import re
import textwrap
import time
import urllib.error
import urllib.request
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Set

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
import trafilatura
from bs4 import BeautifulSoup
from openai import OpenAI
from pyvis.network import Network
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from urllib.parse import parse_qs, unquote, urlparse, urlencode
import networkx as nx
from sentence_transformers import SentenceTransformer

from app_sections.authority_advance import (
    AuthorityGapResult,
    run_authority_gap_from_embeddings,
    run_authority_gap_simulation,
)

if TYPE_CHECKING:  # pragma: no cover - solo para anotaciones
    import spacy  # noqa: F401

try:
    import google.generativeai as genai
except ModuleNotFoundError:
    genai = None

COREFEREE_MODULE = None
COREFEREE_IMPORT_ERROR: Optional[Exception] = None
ENTITY_LINKER_MODULE = None
ENTITY_LINKER_IMPORT_ERROR: Optional[Exception] = None

SPACY_MODULE = None
SPACY_DOWNLOAD_FN = None
SPACY_IMPORT_ERROR: Optional[Exception] = None
SUPPORTED_COREF_LANGS = {"en", "de", "fr", "pl"}
POSITION_CHART_PRESETS: List[Tuple[str, str, str]] = [
    (
        "heatmap",
        "Heatmap por familia que compare la presencia relativa de cada dominio en las posiciones 1-10.",
        "???",
    ),
    (
        "competitors",
        "Grafico de barras con la frecuencia total de los principales competidores en el Top 10 para todas las keywords.",
        "??",
    ),
    (
        "radar",
        "Grafico radar que muestre la posicion media del dominio de la marca frente a competidores por familia.",
        "???",
    ),
    (
        "trendline",
        "Linea temporal con la evolucion de la posicion media por familia (si el CSV incluye fechas).",
        "??",
    ),
    (
        "stacked",
        "Barras apiladas por familia indicando que dominio ocupa cada posicion del 1 al 5.",
        "??",
    ),
]
DEFAULT_CHART_KEYS = ["heatmap", "competitors", "radar"]
ENTITY_PROFILE_PRESETS: Dict[str, List[str]] = {
    "Clinica / Salud": [
        "ORG",
        "PERSON",
        "PRODUCT",
        "GPE",
        "LOC",
        "FAC",
        "EVENT",
        "LAW",
        "NORP",
        "LANGUAGE",
        "WORK_OF_ART",
        "DISEASE",
        "SYMPTOM",
        "MEDICATION",
    ],
    "Editorial / Libros": [
        "ORG",
        "PERSON",
        "WORK_OF_ART",
        "PRODUCT",
        "EVENT",
        "GPE",
        "LOC",
        "LANGUAGE",
        "LAW",
        "NORP",
        "FAC",
    ],
    "Ecommerce / Retail": [
        "ORG",
        "PRODUCT",
        "PERSON",
        "GPE",
        "LOC",
        "FAC",
        "EVENT",
        "WORK_OF_ART",
        "LAW",
        "NORP",
    ],
}

def ensure_spacy_module():
    global SPACY_MODULE, SPACY_DOWNLOAD_FN, SPACY_IMPORT_ERROR
    if SPACY_MODULE is not None:
        return SPACY_MODULE
    try:
        spacy_module = importlib.import_module("spacy")
        spacy_cli = importlib.import_module("spacy.cli")
        download_fn = getattr(spacy_cli, "download")
    except Exception as exc:  # noqa: BLE001
        SPACY_IMPORT_ERROR = exc
        raise RuntimeError(
            "spaCy no está disponible en este entorno. Instálalo o corrige la versión (pip install spacy). "
            f"Detalle: {exc}"
        ) from exc
    SPACY_MODULE = spacy_module
    SPACY_DOWNLOAD_FN = download_fn
    return spacy_module


def is_spacy_available() -> bool:
    if SPACY_MODULE is not None:
        return True
    try:
        ensure_spacy_module()
    except RuntimeError:
        return False
    return True


def ensure_coreferee_module():
    global COREFEREE_MODULE, COREFEREE_IMPORT_ERROR
    if COREFEREE_MODULE is not None:
        return COREFEREE_MODULE
    if COREFEREE_IMPORT_ERROR is not None:
        raise RuntimeError(
            "No se pudo cargar coreferee previamente. Revisa la instalacion correcta del paquete."
        ) from COREFEREE_IMPORT_ERROR
    try:
        COREFEREE_MODULE = importlib.import_module("coreferee")
    except Exception as exc:  # noqa: BLE001
        COREFEREE_IMPORT_ERROR = exc
        raise RuntimeError("Para activar la coreferencia instala `coreferee` y sus modelos compatibles.") from exc
    return COREFEREE_MODULE


def ensure_entity_linker_module():
    global ENTITY_LINKER_MODULE, ENTITY_LINKER_IMPORT_ERROR
    if ENTITY_LINKER_MODULE is not None:
        return ENTITY_LINKER_MODULE
    if ENTITY_LINKER_IMPORT_ERROR is not None:
        raise RuntimeError(
            "No se pudo cargar spacy-entity-linker previamente. Revisa la instalacion y la descarga de su KB."
        ) from ENTITY_LINKER_IMPORT_ERROR
    try:
        ENTITY_LINKER_MODULE = importlib.import_module("spacy_entity_linker")
    except Exception as exc:  # noqa: BLE001
        ENTITY_LINKER_IMPORT_ERROR = exc
        raise RuntimeError(
            "Para activar el entity linking instala `spacy-entity-linker` y descarga la base de conocimiento."
        ) from exc
    return ENTITY_LINKER_MODULE


st.set_page_config(
    page_title="Embedding Insights Dashboard",
    layout="wide",
    page_icon="📈",
)

sns.set_theme(style="whitegrid")

DEFAULT_SENTENCE_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def get_gemini_api_key_from_context() -> str:
    """
    Devuelve la API key de Gemini guardada en la sesion o, en su defecto, en variables de entorno.
    Prioriza la clave almacenada en la barra lateral para no obligar al usuario a repetirla.
    """
    candidates = [
        st.session_state.get("gemini_api_key"),
        os.environ.get("GEMINI_API_KEY"),
        os.environ.get("GOOGLE_API_KEY"),
        os.environ.get("GOOGLE_GENAI_KEY"),
    ]
    for candidate in candidates:
        if candidate:
            return candidate.strip()
    return ""


def get_gemini_model_from_context(default: str = "gemini-2.5-flash") -> str:
    """
    Obtiene el nombre de modelo preferido para Gemini considerando sesion y entorno.
    """
    candidate = (
        st.session_state.get("gemini_model_name")
        or os.environ.get("GEMINI_MODEL")
        or default
    )
    return candidate.strip()


def render_api_settings_panel() -> None:
    """
    Muestra en la barra lateral un bloque fijo para introducir la API key de Gemini.
    Explica brevemente como configurarla y la conserva en st.session_state.
    """
    with st.sidebar:
        st.markdown("### ðŸ”‘ Configuracion de Gemini")
        st.caption(
            "Guarda tu API key una vez y la reutilizaremos en el laboratorio, el builder semantico "
            "y el informe de posiciones. Tambien puedes definir `GOOGLE_API_KEY` o `GEMINI_API_KEY` "
            "como variable de entorno o en `.streamlit/secrets.toml`."
        )
        if "sidebar_gemini_api_value" not in st.session_state:
            st.session_state["sidebar_gemini_api_value"] = get_gemini_api_key_from_context()
        if "sidebar_gemini_model_value" not in st.session_state:
            st.session_state["sidebar_gemini_model_value"] = get_gemini_model_from_context()

        sidebar_key = st.text_input(
            "Gemini API Key",
            type="password",
            key="sidebar_gemini_api_value",
            help="Introduce la clave de https://aistudio.google.com/app/apikey",
        )
        sidebar_model = st.text_input(
            "Modelo Gemini preferido",
            key="sidebar_gemini_model_value",
            help="Ejemplo: gemini-2.5-flash o gemini-1.5-pro",
        )
        if st.button("Guardar clave en esta sesion", key="sidebar_save_gemini"):
            cleaned_key = (sidebar_key or "").strip()
            cleaned_model = (sidebar_model or "").strip() or "gemini-2.5-flash"
            if cleaned_key:
                st.session_state["gemini_api_key"] = cleaned_key
                st.session_state["gemini_model_name"] = cleaned_model
                st.success("API key almacenada en la sesion actual.")
            else:
                st.warning("Introduce una API key valida antes de guardar.")


def apply_global_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-primary: #0f111a;
            --bg-secondary: #16192a;
            --bg-card: #1d2136;
            --border-color: rgba(255,255,255,0.08);
            --accent: #5c6bff;
            --accent-soft: rgba(92,107,255,0.15);
            --text-primary: #f5f7ff;
            --text-secondary: #a0a8c3;
        }
        body {
            background-color: var(--bg-primary);
            color: var(--text-primary);
        }
        .main {
            background-color: var(--bg-primary);
        }
        section[data-testid="stSidebar"] {
            background-color: #14172b;
        }
        section[data-testid="stSidebar"] * {
            color: var(--text-primary) !important;
        }
        .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5 {
            color: var(--text-primary);
        }
        .card-panel {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 18px;
            padding: 1.5rem 1.75rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 20px 45px rgba(5, 7, 12, 0.45);
        }
        .card-panel h3 {
            margin-top: 0.2rem;
            margin-bottom: 0.35rem;
            font-weight: 600;
        }
        div[data-baseweb="tab-list"] button {
            border-radius: 999px !important;
            border: 1px solid transparent !important;
            background: transparent !important;
            color: var(--text-secondary) !important;
            font-weight: 500;
            padding: 0.4rem 1.1rem !important;
            margin-right: 0.35rem;
        }
        div[data-baseweb="tab-list"] button[aria-selected="true"] {
            background: var(--accent-soft) !important;
            border-color: var(--accent) !important;
            color: #fff !important;
        }
        div[data-testid="stFileUploader"] {
            background: #1b1f30;
            border: 1px dashed rgba(255,255,255,0.2);
            border-radius: 18px;
            padding: 1.5rem;
        }
        div[data-testid="stFileUploader"] section {
            color: var(--text-secondary);
        }
        div[data-testid="stFileUploader"] svg {
            stroke: var(--accent);
        }
        .upload-extra {
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            margin-top: 1rem;
        }
        .upload-extra .extra-chip {
            flex: 1;
            min-width: 180px;
            border-radius: 14px;
            background: #222642;
            padding: 0.85rem 1rem;
            border: 1px solid rgba(255,255,255,0.08);
            color: var(--text-secondary);
            font-size: 0.9rem;
        }
        .primary-button button {
            background: var(--accent) !important;
            color: #fff !important;
            border: none !important;
            border-radius: 50px !important;
            padding: 0.6rem 1.6rem !important;
            font-weight: 600 !important;
            box-shadow: 0 12px 30px rgba(92,107,255,0.35);
        }
        .action-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
            margin-top: 1.5rem;
        }
        .action-card {
            background: #1b1f32;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 20px;
            padding: 1.75rem;
            box-shadow: 0 25px 45px rgba(8, 10, 20, 0.4);
            transition: transform 0.25s ease, border-color 0.25s ease, background 0.25s ease;
        }
        .action-card.primary {
            background: linear-gradient(180deg, rgba(92,107,255,0.18), rgba(92,107,255,0.08));
            border-color: rgba(92,107,255,0.4);
        }
        .action-card.secondary {
            background: linear-gradient(180deg, rgba(210,186,255,0.25), rgba(210,186,255,0.1));
            border-color: rgba(210,186,255,0.4);
        }
        .action-card:hover {
            transform: translateY(-6px);
            border-color: var(--accent);
        }
        .action-card h4 {
            margin-bottom: 0.5rem;
            font-size: 1.1rem;
        }
        .action-card p {
            color: var(--text-secondary);
            margin-bottom: 1.2rem;
        }
        .action-card .icon {
            font-size: 2.6rem;
            margin-bottom: 0.8rem;
        }
        .cta-button button {
            border-radius: 999px !important;
            width: 100%;
            padding: 0.55rem 1.2rem !important;
            font-weight: 600 !important;
            border: 1px solid rgba(255,255,255,0.2) !important;
            background: rgba(15,17,26,0.4) !important;
        }
        .cta-button button:hover {
            border-color: var(--accent) !important;
            color: #fff !important;
        }
        .back-link button {
            background: transparent !important;
            border: 1px solid rgba(255,255,255,0.18) !important;
            color: var(--text-secondary) !important;
            border-radius: 999px !important;
            padding: 0.35rem 0.9rem !important;
            font-size: 0.9rem !important;
        }
        .back-link button:hover {
            border-color: var(--accent) !important;
            color: #fff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def bordered_container():
    """
    Devuelve un contenedor con borde cuando la versión de Streamlit lo soporta;
    en versiones anteriores cae a un contenedor estándar.
    """
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()


def set_app_view(view: str) -> None:
    st.session_state["app_view"] = view
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


def render_back_to_landing() -> None:
    with st.container():
        st.markdown("")
        back_key = f"back_to_landing_{st.session_state.get('app_view', 'landing')}"
        with st.container():
            st.markdown('<div class="back-link">', unsafe_allow_html=True)
            if st.button("â† Volver a selección", key=back_key):
                set_app_view("landing")
            st.markdown("</div>", unsafe_allow_html=True)


def render_landing_view() -> None:
    st.subheader("Selección de funcionalidad")
    st.caption("Escoge si quieres cargar un dataset, usar herramientas rápidas, el builder semántico o el nuevo laboratorio de enlazado.")
    cards = [
        {
            "icon": "â˜ï¸",
            "title": "Trabajar mediante archivo CSV",
            "body": "Sube tu dataset para desbloquear los análisis principales (similitud, clustering, grafo).",
            "button": "Ir a carga CSV",
            "view": "csv",
            "style": "primary",
            "key": "cta_csv",
        },
        {
            "icon": "⚙️",
            "title": "Trabajar sin archivo CSV",
            "body": "Usa las herramientas adicionales (texto vs keywords, FAQs, competidores, URLs enriquecidas).",
            "button": "Ir a herramientas adicionales",
            "view": "tools",
            "style": "secondary",
            "key": "cta_tools",
        },
        {
            "icon": "🧠 ",
            "title": "Semantic Keyword Builder",
            "body": "Genera un universo EAV de keywords con intención, volumen cualitativo y clusters usando Gemini.",
            "button": "Ir a Semantic Keyword",
            "view": "keywords",
            "style": "secondary",
            "key": "cta_keywords",
        },
        {
            "icon": "🔗",
            "title": "Laboratorio de enlazado interno",
            "body": "Accede a los modos básico, avanzado, híbrido (CLS) y estructural para optimizar tu internal linking.",
            "button": "Ir al laboratorio de enlazado",
            "view": "linking",
            "style": "secondary",
            "key": "cta_linking",
        },
        {
            "icon": "POS",
            "title": "Informe de posiciones",
            "body": "Convierte un CSV de rankings en un informe HTML con insights competitivos y sugerencias graficas.",
            "button": "Ir a informe de posiciones",
            "view": "positions",
            "style": "secondary",
            "key": "cta_positions",
        },
    ]

    row_cols = None
    for idx, card in enumerate(cards):
        if idx % 2 == 0:
            row_cols = st.columns(2, gap="large")
        col = row_cols[idx % 2]
        with col:
            card_class = "primary" if card["style"] == "primary" else "secondary"
            st.markdown(
                f"""
                <div class="action-card {card_class}">
                    <div class="icon">{card['icon']}</div>
                    <h4>{card['title']}</h4>
                    <p>{card['body']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown('<div class="cta-button">', unsafe_allow_html=True)
            if st.button(card["button"], key=card["key"]):
                set_app_view(card["view"])
            st.markdown("</div>", unsafe_allow_html=True)


def build_semantic_keyword_prompt(
    country: str,
    language: str,
    website: str,
    niche: str,
    approx_keywords: int,
    seed_keywords: List[str],
) -> str:
    seeds_block = "\n".join(f"- {seed}" for seed in seed_keywords) if seed_keywords else "none"
    return f"""
You are an expert semantic SEO and keyword research assistant.

Your task:
Given the project context, generate a semantic keyword universe with entity-attribute-variable (EAV) annotations.

Project context:
- Country: {country}
- Language for keywords: {language}
- Website / brand: {website}
- Niche / service: {niche}

Optional seed keywords:
{seeds_block}

Methodology (do not describe in the output):
1) Understand the niche and search context.
2) Build a broad keyword universe that mixes head, mid-tail and long-tail queries across informational, commercial, transactional, navigational and mixed intents.
3) For each keyword provide EAV annotations (entity/topic label, attribute, variable), qualitative volume (high/medium/low/unknown), primary intent, topic and cluster ID.

Return ONLY valid JSON in this exact structure (no markdown, no commentary):

{{
  "project": {{
    "country": "{country}",
    "language": "{language}",
    "website": "{website}",
    "niche": "{niche}",
    "approx_keywords_requested": {approx_keywords}
  }},
  "keywords": [
    {{
      "keyword": "example keyword in {language}",
      "volume_level": "high | medium | low | unknown",
      "intent": "informational | commercial | transactional | navigational | mixed",
      "entity": "topic-level conceptual label in {language}",
      "attribute": "attribute in {language}",
      "variable": "variable value in {language}",
      "source": "ai_synthetic | seed | competitor | site_search | other",
      "topic": "human-friendly cluster name in {language}",
      "cluster_id": "C1"
    }}
  ]
}}

Constraints:
- Provide AT LEAST {approx_keywords} keywords.
- The keywords, entity, attribute, variable and topic fields MUST be in {language}.
- Intent must be one of [informational, commercial, transactional, navigational, mixed].
- Volume level must be one of [high, medium, low, unknown].
- Cluster IDs should be short codes like C1, C2 reused for the same topic.
"""


def generate_semantic_keyword_universe(
    api_key: str,
    model_name: str,
    country: str,
    language: str,
    website: str,
    niche: str,
    approx_keywords: int,
    seed_keywords: List[str],
) -> Tuple[Dict[str, object], str]:
    if genai is None:
        raise RuntimeError(
            "El módulo 'google-generativeai' no está instalado. Instálalo con `pip install google-generativeai`."
        )
    if not api_key:
        raise ValueError("Introduce tu API key de Google Generative AI.")

    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel(model_name.strip() or "gemini-2.5-flash")
    except Exception as exc:
        raise RuntimeError(f"No se pudo inicializar el modelo '{model_name}': {exc}") from exc

    prompt = build_semantic_keyword_prompt(
        country=country,
        language=language,
        website=website,
        niche=niche,
        approx_keywords=approx_keywords,
        seed_keywords=seed_keywords,
    )

    response = model.generate_content(prompt)
    json_text = getattr(response, "text", "") or ""
    json_text = json_text.strip()
    if not json_text and response.candidates:
        json_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, "text"))
        json_text = json_text.strip()

    if not json_text:
        raise RuntimeError("La respuesta del modelo llegó vacía.")

    if json_text.startswith("```json"):
        json_text = json_text[7:]
    if json_text.startswith("```"):
        json_text = json_text[3:]
    if json_text.endswith("```"):
        json_text = json_text[:-3]
    json_text = json_text.strip()

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"No se pudo parsear la respuesta como JSON: {exc}") from exc
    return data, json_text


def group_keywords_with_semantic_builder(
    api_key: str,
    model_name: str,
    keywords: Sequence[str],
    language: str,
    country: str,
    niche: str,
    brand_domain: str,
    competitors: Sequence[str],
) -> Tuple[Dict[str, str], Dict[str, object]]:
    """
    Reutiliza la metodología del Semantic Keyword Builder para agrupar keywords existentes en familias semánticas.
    Devuelve un mapping keyword -> familia y el JSON completo generado por Gemini.
    """
    if genai is None:
        raise RuntimeError("Instala `google-generativeai` para habilitar la agrupación automática.")
    cleaned_key = (api_key or "").strip()
    if not cleaned_key:
        raise ValueError("Introduce tu API key de Gemini para agrupar las keywords.")
    if not keywords:
        raise ValueError("No hay keywords disponibles para agrupar.")

    unique_keywords = list(dict.fromkeys([kw.strip() for kw in keywords if kw and kw.strip()]))
    if not unique_keywords:
        raise ValueError("No se detectaron keywords válidas.")
    # Limita el número para evitar prompts demasiado largos.
    max_keywords_supported = 250
    truncated = unique_keywords[:max_keywords_supported]
    trimmed_count = len(unique_keywords) - len(truncated)

    genai.configure(api_key=cleaned_key)
    model = genai.GenerativeModel((model_name or "gemini-2.5-flash").strip())
    competitors_text = ", ".join(filter(None, competitors)) or "No declarados"
    prompt = f"""
Actúa como el mismo analista que construye el Semantic Keyword Builder (enfoque EAV).
Dado un listado cerrado de keywords, clasifícalas sin inventar nuevas entradas.

Contexto:
- País / mercado: {country or "No especificado"}
- Idioma: {language or "es"}
- Nicho / sector: {niche or "general"}
- Dominio de la marca: {brand_domain or "No especificado"}
- Competidores: {competitors_text}

Instrucciones:
1. Agrupa las keywords por familias semánticas (entity-level) manteniendo coherencia temática.
2. Para cada keyword indica: family_name, topic_entity_label, intent, volume_level (high/medium/low/unknown).
3. Respeta la keyword original; no la traduzcas ni la modifiques.
4. Devuelve JSON válido con esta estructura:
{{
  "families": [
    {{
      "name": "Cluster principal",
      "rationale": "Explica brevemente el criterio",
      "keywords": ["keyword 1", "keyword 2"]
    }}
  ],
  "keywords": [
    {{
      "keyword": "keyword 1",
      "family": "Cluster principal",
      "topic_entity_label": "Entidad o tema",
      "intent": "informational|commercial|transactional|navigational|mixed",
      "volume_level": "high|medium|low|unknown"
    }}
  ]
}}
No incluyas texto adicional fuera del JSON.

Keywords a clasificar:
{os.linesep.join(f"- {kw}" for kw in truncated)}
"""
    response = model.generate_content(prompt)
    json_text = getattr(response, "text", "") or ""
    if not json_text and response.candidates:
        json_text = "".join(
            part.text for part in response.candidates[0].content.parts if hasattr(part, "text")
        )
    json_text = json_text.strip()
    if json_text.startswith("```json"):
        json_text = json_text[7:]
    if json_text.startswith("```"):
        json_text = json_text[3:]
    if json_text.endswith("```"):
        json_text = json_text[:-3]

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Gemini devolvió un JSON inválido al agrupar keywords: {exc}") from exc

    records = data.get("keywords") or []
    if not isinstance(records, list) or not records:
        raise ValueError("El modelo no devolvió asignaciones de keywords. Ajusta el prompt e inténtalo otra vez.")

    family_map: Dict[str, str] = {}
    for record in records:
        keyword = (record.get("keyword") or "").strip()
        family = (record.get("family") or record.get("topic_entity_label") or "Sin familia").strip()
        if not keyword:
            continue
        family_map[keyword] = family or "Sin familia"
        family_map[keyword.lower()] = family or "Sin familia"
    if trimmed_count > 0:
        data["notice"] = f"Se omitieron {trimmed_count} keywords por límite de token."
    return family_map, data
def render_semantic_keyword_builder() -> None:
    st.subheader("🧠  Semantic Keyword Builder")
    st.caption(
        "Genera un universo semántico de palabras clave con etiquetas EAV, intentos, volumen cualitativo y clusters usando Gemini."
    )

    if genai is None:
        st.error(
            "Para usar esta sección necesitas instalar `google-generativeai` en tu entorno (por ejemplo `pip install google-generativeai`)."
        )
        return

    if "skb_results" not in st.session_state:
        st.session_state["skb_results"] = None
    if "skb_api_key" not in st.session_state or not st.session_state["skb_api_key"]:
        st.session_state["skb_api_key"] = get_gemini_api_key_from_context()

    builder_card = bordered_container()
    with builder_card:
        with st.form("semantic_keyword_form"):
            st.markdown("#### Configuración del proyecto")
            api_key = st.text_input(
                "Google Generative AI API Key",
                type="password",
                value=st.session_state.get("skb_api_key", ""),
            )
            model_name = st.text_input("Modelo Gemini", value="gemini-2.5-flash")
            col_country, col_language = st.columns(2)
            with col_country:
                country = st.text_input("País", value="Spain")
            with col_language:
                language = st.text_input("Idioma de las keywords", value="Spanish")
            website = st.text_input("Website / marca", value="https://www.example.com")
            niche = st.text_input("Nicho / servicio", value="Servicio de consultoría SEO")
            approx_keywords = st.slider(
                "Número aproximado de keywords",
                min_value=20,
                max_value=200,
                value=60,
                step=10,
            )
            seed_keywords_text = st.text_area(
                "Seed keywords (opcional, una por línea)",
                value="seo tecnico\ncluster semantico\ninterlinking",
                height=120,
            )
            submitted = st.form_submit_button("Generar universo semántico")

        if submitted:
            seed_keywords = [
                line.strip()
                for line in seed_keywords_text.splitlines()
                if line.strip()
            ]
            try:
                data, raw = generate_semantic_keyword_universe(
                    api_key=api_key.strip(),
                    model_name=model_name.strip(),
                    country=country.strip(),
                    language=language.strip(),
                    website=website.strip(),
                    niche=niche.strip(),
                    approx_keywords=int(approx_keywords),
                    seed_keywords=seed_keywords,
                )
            except Exception as exc:
                st.error(str(exc))
            else:
                st.session_state["skb_api_key"] = api_key
                st.session_state["gemini_api_key"] = api_key.strip()
                cleaned_model = model_name.strip() or get_gemini_model_from_context()
                st.session_state["gemini_model_name"] = cleaned_model
                st.session_state["skb_results"] = (data, raw)
                st.success("Universo semántico generado correctamente.")

    stored = st.session_state.get("skb_results")
    if not stored:
        st.info("Configura el formulario y pulsa el botón para generar tu universo semántico.")
        return

    data, raw = stored
    keywords_list = data.get("keywords", [])
    project_info = data.get("project", {})

    if not keywords_list:
        st.warning("El modelo no devolvió keywords. Revisa la configuración e inténtalo de nuevo.")
        return

    df = pd.DataFrame(keywords_list)
    if "entity" in df.columns and "topic_entity_label" not in df.columns:
        df.rename(columns={"entity": "topic_entity_label"}, inplace=True)

    preferred_cols = [
        "keyword",
        "volume_level",
        "intent",
        "topic_entity_label",
        "attribute",
        "variable",
        "source",
        "topic",
        "cluster_id",
    ]
    ordered_cols = [col for col in preferred_cols if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in ordered_cols]
    df = df[ordered_cols + remaining_cols]

    st.markdown("#### Resultado del keyword builder")
    st.caption("Las entidades son etiquetas conceptuales (no entidades del Knowledge Graph).")
    st.dataframe(df, use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar CSV",
        data=csv_bytes,
        file_name="semantic_keyword_universe.csv",
        mime="text/csv",
    )

    st.markdown("#### Metadata del proyecto")
    st.json(project_info)

    with st.expander("Ver respuesta JSON completa"):
        st.code(raw, language="json")


def render_linking_lab() -> None:
    """Pantalla dedicada al laboratorio de enlazado interno."""
    st.subheader("🔗 Laboratorio de enlazado interno")
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
    with st.expander("📄 Cargar archivo de enlaces existentes (opcional)", expanded=False):
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
                st.caption(f"📊 Total de filas en el archivo: {len(inlinks_df):,}")
                
                # Selección manual de columnas
                st.markdown("#### 🔧 Configuración de columnas")
                
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
                        "📍 Columna Source (Origen)",
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
                        "🎯 Columna Target (Destino)",
                        options=available_columns,
                        index=target_default_idx,
                        key="inlinks_target_column",
                        help="Columna que contiene la URL de destino del enlace"
                    )
                
                # Verificar que no sean la misma columna
                if source_col == target_col:
                    st.error("⚠️ Las columnas Source y Target no pueden ser la misma. Selecciona columnas diferentes.")
                else:
                    # Columna opcional de tipo/ubicación de enlace
                    st.markdown("#### 🔍 Filtrado por ubicación (opcional)")
                    
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
                                st.caption(f"• {type_val}: {count:,} enlaces ({count/len(inlinks_df)*100:.1f}%)")
                            
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
                                "⚖️ Aplicar pesos diferentes según ubicación",
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
                    if st.button("✅ Cargar enlaces en sesión", type="primary", key="inlinks_load_button"):
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
                            f"✅ Cargados **{len(existing_edges):,}** enlaces existentes\n\n"
                            f"- Columnas: `{source_col}` → `{target_col}`"
                        )
                        
                        if filtered_count > 0:
                            st.info(f"🔍 Filtrados {filtered_count:,} enlaces (tipos excluidos)")
                        
                        if type_weights:
                            st.info(
                                f"⚖️ Pesos aplicados:\n" +
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
                st.caption(f"📊 **Estado:** {len(existing_inlinks):,} enlaces cargados (con pesos) en sesión")
            else:
                st.caption(f"📊 **Estado:** {len(existing_inlinks):,} enlaces cargados en sesión")
            
            if st.button("🗑️ Limpiar enlaces cargados", key="clear_inlinks_button"):
                st.session_state["existing_inlinks"] = None
                st.rerun()
        else:
            st.caption("📊 **Estado:** No hay enlaces cargados (modo sin enlaces existentes)")

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

def render_positions_report() -> None:
    st.subheader("📊 Informe de posiciones SEO")
    st.caption(
        "Procesa un CSV exportado de tu herramienta de rank tracking, genera tablas por dominio y crea un informe HTML con Gemini."
    )

    st.session_state.setdefault("positions_raw_df", None)
    st.session_state.setdefault("positions_report_html", None)
    st.session_state.setdefault("positions_competitors", [])
    st.session_state.setdefault("positions_semantic_groups", None)
    st.session_state.setdefault("positions_semantic_groups_raw", None)
    st.session_state.setdefault("positions_semantic_language", "es")
    st.session_state.setdefault("positions_semantic_country", "Spain")
    st.session_state.setdefault("positions_semantic_niche", "Proyecto SEO")
    if "positions_gemini_key" not in st.session_state or not st.session_state["positions_gemini_key"]:
        st.session_state["positions_gemini_key"] = get_gemini_api_key_from_context()
    if "positions_gemini_model" not in st.session_state or not st.session_state["positions_gemini_model"]:
        st.session_state["positions_gemini_model"] = get_gemini_model_from_context()

    col_main, col_side = st.columns([3, 1])
    with col_main:
        uploaded_csv = st.file_uploader("Archivo CSV con posiciones", type=["csv"], key="positions_csv_uploader")
    with col_side:
        max_keywords = st.slider("Keywords por familia", min_value=5, max_value=40, value=20, step=5)

    col_a, col_b = st.columns(2)
    with col_a:
        brand_domain = st.text_input(
            "Dominio principal",
            value=st.session_state.get("positions_brand", ""),
            help="Se usar? para resaltar la marca en las tablas.",
        )
    with col_b:
        report_title = st.text_input(
            "T?tulo del informe",
            value=st.session_state.get("positions_report_title", "Informe de posiciones org?nicas"),
        )

    competitor_domains_raw = st.text_input(
        "Dominios competidores (separa por coma)",
        value=", ".join(st.session_state.get("positions_competitors", [])),
        help="A?ade los dominios ra?z que quieres vigilar en el informe.",
    )
    competitor_domains = [
        normalize_domain(domain.strip())
        for domain in competitor_domains_raw.split(",")
        if domain.strip()
    ]
    st.session_state["positions_competitors"] = [domain for domain in competitor_domains if domain]

    families_instructions = st.text_area(
        "Definici?n de familias (formato: Familia: keyword1, keyword2, *fragmento*)",
        value=st.session_state.get(
            "positions_family_text",
            "Anillos: anillo, alianzas\nPendientes: pendiente, aro\nCollares: collar, colgante, gargantilla",
        ),
        height=140,
        help="Usa comas o punto y coma para separar keywords/patrones. El car?cter * permite coincidencias parciales.",
    )

    st.markdown("**Selecciona los gr?ficos que quieres incluir en el informe**")
    chart_columns = st.columns(3)
    default_chart_keys = st.session_state.get("positions_chart_selection") or DEFAULT_CHART_KEYS
    selected_chart_keys: List[str] = []
    cards_per_row = len(chart_columns)
    for idx, (chart_key, chart_label, chart_icon) in enumerate(POSITION_CHART_PRESETS):
        widget_key = f"positions_chart_{chart_key}"
        col = chart_columns[idx % cards_per_row]
        col.markdown(
            f"""
            <div style='border:1px solid rgba(255,255,255,0.2); border-radius:12px; padding:0.8rem; text-align:center;'>
                <div style='font-size:1.6rem'>{chart_icon}</div>
                <div style='font-size:0.9rem; opacity:0.8'>{chart_label}</div>
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
        st.info("Selecciona al menos un gr?fico o a?ade una nota personalizada para guiar al informe.")
    chart_notes = "\n".join(f"- {desc}" for desc in chart_descriptions).strip()

    use_semantic_builder = st.checkbox(
        "Usar agrupaci?n autom?tica (Semantic Keyword Builder)",
        value=st.session_state.get("positions_use_semantic_builder", False),
        help="Reutiliza la l?gica del builder para clasificar las keywords sin reglas manuales.",
    )
    st.session_state["positions_use_semantic_builder"] = use_semantic_builder
    semantic_grouping_map: Optional[Dict[str, str]] = None
    raw_positions_df = st.session_state.get("positions_raw_df")
    if use_semantic_builder:
        if raw_positions_df is None or raw_positions_df.empty:
            st.info("Carga un CSV antes de ejecutar la agrupaci?n autom?tica.")
        else:
            with st.expander(
                "Configurar agrupaci?n autom?tica",
                expanded=st.session_state.get("positions_semantic_groups") is None,
            ):
                cfg_cols = st.columns(3)
                with cfg_cols[0]:
                    semantic_language = st.text_input(
                        "Idioma de las keywords",
                        value=st.session_state.get("positions_semantic_language", "es"),
                        key="positions_semantic_language",
                    )
                with cfg_cols[1]:
                    semantic_country = st.text_input(
                        "Pa?s / mercado",
                        value=st.session_state.get("positions_semantic_country", "Spain"),
                        key="positions_semantic_country",
                    )
                with cfg_cols[2]:
                    semantic_niche = st.text_input(
                        "Nicho o proyecto",
                        value=st.session_state.get("positions_semantic_niche", "Proyecto SEO"),
                        key="positions_semantic_niche",
                    )
                semantic_api_key = st.text_input(
                    "Gemini API Key",
                    type="password",
                    value=st.session_state.get("positions_gemini_key", get_gemini_api_key_from_context()),
                    key="positions_semantic_api_key",
                )
                semantic_model_name = st.text_input(
                    "Modelo Gemini",
                    value=st.session_state.get("positions_gemini_model", get_gemini_model_from_context()),
                    key="positions_semantic_model_name",
                )
                if st.button("Agrupar keywords con Gemini", key="positions_semantic_group_button"):
                    try:
                        mapping, raw_groups = group_keywords_with_semantic_builder(
                            api_key=semantic_api_key.strip(),
                            model_name=semantic_model_name.strip() or get_gemini_model_from_context(),
                            keywords=raw_positions_df["Keyword"].dropna().astype(str).tolist(),
                            language=semantic_language.strip() or "es",
                            country=semantic_country.strip() or "Spain",
                            niche=semantic_niche.strip() or "Proyecto SEO",
                            brand_domain=brand_domain,
                            competitors=st.session_state.get("positions_competitors", []),
                        )
                    except Exception as exc:
                        st.error(str(exc))
                    else:
                        st.session_state["positions_semantic_groups"] = mapping
                        st.session_state["positions_semantic_groups_raw"] = raw_groups
                        st.session_state["positions_semantic_language"] = semantic_language
                        st.session_state["positions_semantic_country"] = semantic_country
                        st.session_state["positions_semantic_niche"] = semantic_niche
                        st.session_state["positions_gemini_key"] = semantic_api_key.strip()
                        st.session_state["positions_gemini_model"] = semantic_model_name.strip() or get_gemini_model_from_context()
                        total_families = len({fam for fam in mapping.values() if fam})
                        st.success(
                            f"Se agruparon {len(mapping)} keywords a través de {total_families} familias semánticas."
                        )
            semantic_grouping_map = st.session_state.get("positions_semantic_groups")
    else:
        semantic_grouping_map = None

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
            parsed_df = parse_position_tracking_csv(uploaded_csv)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.session_state["positions_raw_df"] = parsed_df
            st.session_state["positions_brand"] = brand_domain
            st.success(f"Se procesaron {len(parsed_df)} keywords.")

    raw_df = st.session_state.get("positions_raw_df")
    if raw_df is None:
        st.info("Carga un CSV de posiciones para comenzar.")
        return

    if use_semantic_builder and semantic_grouping_map:
        enriched_df = raw_df.copy()
        enriched_df["Familia"] = (
            enriched_df["Keyword"]
            .astype(str)
            .apply(
                lambda kw: semantic_grouping_map.get(kw)
                or semantic_grouping_map.get(kw.lower())
                or "Sin familia"
            )
        )
    else:
        if use_semantic_builder and not semantic_grouping_map:
            st.warning(
                "Activa la agrupación con Gemini para rellenar las familias o desmarca la opción para seguir con reglas manuales."
            )
        enriched_df = assign_keyword_families(raw_df, families_instructions)
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
        "Generar informe HTML con Gemini",
        type="primary",
        disabled=not gemini_available,
    ):
        with st.spinner("?? Generando informe con Gemini..."):
            try:
                html_report = generate_position_report_html(
                    api_key=gemini_api_key,
                    model_name=gemini_model,
                    report_title=report_title,
                    brand_domain=brand_domain or "Sin dominio especificado",
                    families_payload=family_payload,
                    overview=summary,
                    chart_notes=chart_notes_payload,
                    competitor_domains=competitor_domains,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"No se pudo generar el informe: {exc}")
            else:
                st.session_state["positions_report_html"] = html_report
                st.success("Informe generado correctamente.")

    if st.session_state.get("positions_report_html"):
        st.write("### Vista previa del informe")
        components.html(st.session_state["positions_report_html"], height=700, scrolling=True)
        st.download_button(
            label="Descargar informe HTML",
            data=st.session_state["positions_report_html"].encode("utf-8"),
            file_name="informe_posiciones.html",
            mime="text/html",
        )


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


def compute_similar_pages(
    df: pd.DataFrame,
    target_url: str,
    url_column: str,
    top_n: int,
    embedding_col: str = "EmbeddingsFloat",
) -> List[Tuple[str, float]]:
    if target_url not in df[url_column].values:
        return []

    embeddings = np.vstack(df[embedding_col].values)
    embeddings_norm = normalize(embeddings)
    target_idx = df[df[url_column] == target_url].index[0]
    target_embedding = embeddings_norm[target_idx]

    dot_products = embeddings_norm @ target_embedding
    similarities = (dot_products * 100).tolist()

    results = []
    for idx, url in enumerate(df[url_column]):
        if url == target_url:
            continue
        results.append((url, similarities[idx]))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]


def build_similarity_matrix(
    df: pd.DataFrame,
    url_column: str,
    similarity_threshold: Optional[float],
    max_results: Optional[int],
    embedding_col: str = "EmbeddingsFloat",
) -> pd.DataFrame:
    urls = df[url_column].tolist()
    embeddings = np.vstack(df[embedding_col].values)
    embeddings_norm = normalize(embeddings)
    sim_matrix = embeddings_norm @ embeddings_norm.T * 100

    n = len(urls)
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            score = sim_matrix[i, j]
            if similarity_threshold is not None and score < similarity_threshold:
                continue
            rows.append({"URL1": urls[i], "URL2": urls[j], "Similitud": score})
            if max_results is not None and len(rows) >= max_results:
                return pd.DataFrame(rows).sort_values("Similitud", ascending=False)
    return pd.DataFrame(rows).sort_values("Similitud", ascending=False)


def extract_words_from_url(url: str) -> List[str]:
    clean = url.split("://")[-1]
    clean = clean.replace("www.", "")
    segments = [segment for segment in clean.split("/") if segment and not segment.startswith("?")]
    words: List[str] = []
    for segment in segments:
        parts = segment.replace("-", " ").replace("_", " ").split()
        for part in parts:
            tokens = Counter()
            tokens.update(
                word.lower()
                for word in part.replace(".", " ").replace("=", " ").split()
                if word.isalpha() and len(word) > 2
            )
            words.extend(tokens.elements())
    return words


def auto_select_cluster_count(
    df: pd.DataFrame,
    min_clusters: int,
    max_clusters: int,
    embedding_col: str = "EmbeddingsFloat",
) -> Tuple[int, Dict[int, float]]:
    embeddings = np.vstack(df[embedding_col].values)
    embeddings_norm = normalize(embeddings)
    n_samples = embeddings_norm.shape[0]

    upper_bound = min(max_clusters, n_samples - 1)
    if upper_bound < 2 or upper_bound < min_clusters:
        raise ValueError("No hay suficientes páginas para evaluar el rango de clusters indicado.")

    best_k: Optional[int] = None
    best_score = -1.0
    scores: Dict[int, float] = {}

    for n_clusters in range(min_clusters, upper_bound + 1):
        if n_clusters < 2:
            continue
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(embeddings_norm)
        unique_labels = np.unique(labels)
        if unique_labels.size < 2:
            score = -1.0
        else:
            score = float(silhouette_score(embeddings_norm, labels, metric="cosine"))
        scores[n_clusters] = score
        if score > best_score:
            best_score = score
            best_k = n_clusters

    if best_k is None:
        raise ValueError("No se pudo determinar un número óptimo de clusters con el rango indicado.")

    return best_k, scores


def cluster_pages(
    df: pd.DataFrame,
    n_clusters: int,
    url_column: str,
    embedding_col: str = "EmbeddingsFloat",
) -> Tuple[pd.DataFrame, Dict[int, str], KMeans, np.ndarray]:
    embeddings = np.vstack(df[embedding_col].values)
    embeddings_norm = normalize(embeddings)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = model.fit_predict(embeddings_norm)

    df_clustered = df.copy()
    df_clustered["Cluster"] = cluster_labels

    centroids = model.cluster_centers_
    cluster_names: Dict[int, str] = {}
    general_themes = [
        "Información General",
        "Servicios",
        "Productos",
        "Recursos",
        "Documentación",
        "Ayuda",
        "Categoría Principal",
        "Sección Técnica",
        "Contenido Especializado",
        "Información Legal",
        "Ãrea de Usuario",
        "Novedades",
    ]
    stopwords = {
        "com",
        "org",
        "net",
        "html",
        "php",
        "index",
        "default",
        "es",
        "en",
        "ca",
        "mx",
        "cl",
        "ar",
        "co",
    }

    for cluster_id in range(n_clusters):
        indices = np.where(cluster_labels == cluster_id)[0]
        if len(indices) == 0:
            cluster_names[cluster_id] = f"Cluster vacío {cluster_id}"
            continue

        distances = cdist([centroids[cluster_id]], embeddings_norm[indices], "cosine")[0]
        top_indices_local = indices[np.argsort(distances)[: min(5, len(indices))]]

        words: Counter = Counter()
        for idx in top_indices_local:
            words.update(word for word in extract_words_from_url(df.iloc[idx][url_column]) if word not in stopwords)

        if words:
            top_words = [word.capitalize() for word, _ in words.most_common(3)]
            cluster_names[cluster_id] = " ".join(top_words)
        else:
            cluster_names[cluster_id] = f"{general_themes[cluster_id % len(general_themes)]} {cluster_id}"

    df_clustered["Nombre_Cluster"] = df_clustered["Cluster"].map(cluster_names)
    return df_clustered, cluster_names, model, embeddings_norm


def tsne_visualisation(df: pd.DataFrame, embeddings_norm: np.ndarray, cluster_names: Dict[int, str]) -> plt.Figure:
    n_samples = embeddings_norm.shape[0]
    if n_samples < 2:
        raise ValueError("Se requieren al menos dos páginas para visualizar t-SNE.")

    perplexity = min(30, max(5, n_samples // 2))
    if perplexity >= n_samples:
        perplexity = max(2, n_samples - 1)
    if perplexity < 2:
        raise ValueError("Añade más páginas para generar la visualización t-SNE (se requieren al menos 3).")

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings_norm)

    viz_df = pd.DataFrame(
        {
            "x": embeddings_2d[:, 0],
            "y": embeddings_2d[:, 1],
            "cluster": df["Cluster"],
            "ClusterName": df["Nombre_Cluster"],
            "URL": df[st.session_state["url_column"]],
        }
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    palette = sns.color_palette("husl", len(cluster_names))
    sns.scatterplot(
        data=viz_df,
        x="x",
        y="y",
        hue="cluster",
        palette=palette,
        s=120,
        alpha=0.8,
        ax=ax,
        legend=False,
    )

    for cluster_id, name in cluster_names.items():
        cluster_points = viz_df[viz_df["cluster"] == cluster_id]
        if cluster_points.empty:
            continue
        centroid_x = cluster_points["x"].mean()
        centroid_y = cluster_points["y"].mean()
        ax.text(
            centroid_x,
            centroid_y,
            name,
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.75),
        )

    ax.set_title("Clusters de páginas (t-SNE)", fontsize=16)
    ax.set_xlabel("Dimensión t-SNE 1")
    ax.set_ylabel("Dimensión t-SNE 2")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def generate_topic_cluster_graph(
    df: pd.DataFrame,
    cluster_names: Dict[int, str],
    model: KMeans,
    url_column: str,
    min_cluster_similarity: float = 70.0,
) -> str:
    graph = nx.Graph()
    cluster_counts = df["Cluster"].value_counts()

    for cluster_id, name in cluster_names.items():
        size = int(cluster_counts.get(cluster_id, 0))
        graph.add_node(
            f"cluster_{cluster_id}",
            label=name,
            title=f"{name}<br/>Cluster {cluster_id}<br/>{size} páginas",
            size=25 + size * 2,
            group=f"cluster_{cluster_id}",
        )

    for idx, row in df.iterrows():
        node_id = f"page_{idx}"
        label = row[url_column]
        short_label = label if len(label) <= 42 else f"{label[:39]}…"
        graph.add_node(
            node_id,
            label=short_label,
            title=label,
            size=10,
            group=f"cluster_{row['Cluster']}",
        )
        graph.add_edge(f"cluster_{row['Cluster']}", node_id, weight=1)

    centroids = model.cluster_centers_
    centroid_similarity = cosine_similarity(centroids) * 100
    n_clusters = len(centroids)
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            score = centroid_similarity[i, j]
            if score >= min_cluster_similarity:
                graph.add_edge(
                    f"cluster_{i}",
                    f"cluster_{j}",
                    weight=max(score / 10, 1),
                    title=f"Similitud clusters: {score:.1f}%",
                )

    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#333333")
    net.from_nx(graph)
    net.repulsion(node_distance=200, spring_length=200, damping=0.85)
    net.toggle_physics(True)
    return net.generate_html(notebook=False)


@st.cache_resource(show_spinner=False)
def load_spacy_model(model_name: str):
    spacy_module = ensure_spacy_module()
    try:
        return spacy_module.load(model_name)
    except OSError:
        download_fn = SPACY_DOWNLOAD_FN
        if download_fn is None:
            raise RuntimeError(
                f"No se pudo cargar el modelo spaCy '{model_name}' y no se puede descargar automáticamente."
            )
        try:
            download_fn(model_name)
            return spacy_module.load(model_name)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"No se pudo cargar el modelo spaCy '{model_name}'. "
                "Instálalo manualmente con 'python -m spacy download MODEL'."
            ) from exc


PROMINENCE_WEIGHTS = {
    "nsubj": 2.0,
    "nsubjpass": 1.8,
    "csubj": 1.7,
    "dobj": 1.5,
    "obj": 1.5,
    "pobj": 0.8,
    "attr": 1.1,
    "appos": 1.3,
    "ROOT": 1.4,
    "default": 0.5,
}


def get_prominence_weight(dep_label: str) -> float:
    if not dep_label:
        return PROMINENCE_WEIGHTS["default"]
    return PROMINENCE_WEIGHTS.get(dep_label.lower(), PROMINENCE_WEIGHTS["default"])


def add_optional_nlp_components(nlp, enable_coref: bool, enable_linking: bool) -> None:
    language_code = getattr(nlp, "lang", "").lower()
    if enable_coref and language_code not in SUPPORTED_COREF_LANGS:
        st.warning(
            f"Coreferee no soporta el idioma '{language_code or 'desconocido'}'. "
            "La coreferencia se desactiva para evitar errores."
        )
        enable_coref = False
    if enable_coref:
        ensure_coreferee_module()
        if "coreferee" not in nlp.pipe_names:
            nlp.add_pipe("coreferee")
    if enable_linking:
        try:
            ensure_entity_linker_module()
        except RuntimeError as exc:
            st.warning(f"No se pudo cargar spacy-entity-linker: {exc}")
            enable_linking = False
        else:
            if "entityLinker" not in nlp.pipe_names:
                nlp.add_pipe("entityLinker", last=True)


def build_coref_map(doc) -> Dict[Tuple[int, int], "spacy.tokens.Span"]:
    mapping: Dict[Tuple[int, int], "spacy.tokens.Span"] = {}
    if not hasattr(doc._, "coref_chains") or not doc._.coref_chains:
        return mapping
    for chain in doc._.coref_chains:
        main_span = chain.main
        for mention in chain:
            mapping[(mention.start, mention.end)] = main_span
    return mapping


def resolve_canonical_span(span: "spacy.tokens.Span", coref_map: Dict[Tuple[int, int], "spacy.tokens.Span"]):
    return coref_map.get((span.start, span.end), span)


def get_linker_metadata(ent: "spacy.tokens.Span", linker_pipe) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not linker_pipe or not hasattr(ent._, "kb_ents") or not ent._.kb_ents:
        return None, None, None
    kb_id, score = ent._.kb_ents[0]
    if not kb_id:
        return None, None, None
    kb_entry = getattr(linker_pipe, "kb", None)
    description = None
    kb_url = None
    if kb_entry and hasattr(kb_entry, "cui_to_entity"):
        entity = kb_entry.cui_to_entity.get(kb_id)
        if entity:
            description = getattr(entity, "description", None)
            kb_url = getattr(entity, "url", None)
    return kb_id, description, kb_url


def find_entity_for_token(token: "spacy.tokens.Token", token_entity_map: Dict[int, str]) -> Optional[str]:
    if token.i in token_entity_map:
        return token_entity_map[token.i]
    for descendant in token.subtree:
        if descendant.i in token_entity_map:
            return token_entity_map[descendant.i]
    return None


def extract_spo_relations(
    doc: "spacy.tokens.Doc",
    token_entity_map: Dict[int, str],
    source_url: str,
) -> List[Dict[str, str]]:
    relations: List[Dict[str, str]] = []
    for sent in doc.sents:
        root = sent.root
        if not root or root.pos_ not in {"VERB", "AUX"}:
            continue
        relation_label = root.lemma_.lower()
        subject_tokens = [token for token in sent if token.dep_.startswith("nsubj") and token.head == root]
        object_tokens = [
            token
            for token in sent
            if token.dep_ in {"dobj", "obj", "pobj", "attr", "ccomp", "xcomp", "dative"} and token.head == root
        ]
        subjects = [find_entity_for_token(token, token_entity_map) for token in subject_tokens]
        objects = [find_entity_for_token(token, token_entity_map) for token in object_tokens]
        subjects = [subj for subj in subjects if subj]
        objects = [obj for obj in objects if obj]
        if not subjects or not objects:
            continue
        for subj in subjects:
            for obj in objects:
                if subj == obj:
                    continue
                relations.append(
                    {
                        "subject": subj,
                        "predicate": relation_label,
                        "object": obj,
                        "source": source_url,
                    }
                )
    return relations


def generate_knowledge_graph_html_v2(
    df: pd.DataFrame,
    text_column: str,
    url_column: Optional[str],
    model_name: str,
    row_limit: int,
    max_entities: int,
    min_entity_frequency: int,
    include_pages: bool,
    max_pages: int,
    allowed_entity_labels: Optional[set[str]],
    enable_coref: bool = False,
    enable_linking: bool = False,
    n_process: int = 1,
    batch_size: int = 200,
) -> Tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nlp = load_spacy_model(model_name)
    add_optional_nlp_components(nlp, enable_coref=enable_coref, enable_linking=enable_linking)
    linker_pipe = nlp.get_pipe("entityLinker") if enable_linking and "entityLinker" in nlp.pipe_names else None

    sampled = df.head(row_limit).copy()
    sampled["__content__"] = sampled[text_column].fillna("").astype(str)
    documents: List[str] = []
    doc_urls: List[str] = []
    for _, row in sampled.iterrows():
        text = row["__content__"].strip()
        if not text or text.lower() == "nan":
            continue
        documents.append(text)
        if url_column and url_column in row:
            url_value = str(row[url_column]).strip()
        else:
            url_value = f"doc_{len(documents)}"
        doc_urls.append(url_value or f"doc_{len(documents)}")

    if not documents:
        raise ValueError("No se encontraron textos válidos en la columna seleccionada.")

    required_components = {"ner", "parser"}
    if enable_coref:
        required_components.add("coreferee")
    if enable_linking:
        required_components.add("entityLinker")
    components_to_disable = [component for component in nlp.pipe_names if component not in required_components]

    pipe_kwargs: Dict[str, object] = {"batch_size": max(1, int(batch_size))}
    if n_process > 1:
        pipe_kwargs["n_process"] = int(n_process)

    docs_iterator = nlp.pipe(
        documents,
        disable=components_to_disable,
        **pipe_kwargs,
    )

    entity_stats: Dict[str, Dict[str, object]] = {}
    doc_entity_stats: Dict[Tuple[str, str], Dict[str, float]] = {}
    spo_rows: List[Dict[str, object]] = []
    spo_edge_counter: Counter = Counter()

    for doc, source_url in zip(docs_iterator, doc_urls):
        coref_map = build_coref_map(doc) if enable_coref else {}
        token_entity_map: Dict[int, str] = {}

        for ent in doc.ents:
            ent_text = ent.text.strip()
            if not ent_text:
                continue
            if allowed_entity_labels and ent.label_ not in allowed_entity_labels:
                continue

            canonical_span = resolve_canonical_span(ent, coref_map) if coref_map else ent
            canonical_text = canonical_span.text.strip() or ent_text
            if not canonical_text:
                continue

            qid, kb_description, kb_url = get_linker_metadata(ent, linker_pipe)
            entity_id = qid or canonical_text.lower()

            stats = entity_stats.setdefault(
                entity_id,
                {
                    "canonical_name": canonical_text,
                    "label": ent.label_,
                    "qid": qid,
                    "kb_description": kb_description,
                    "kb_url": kb_url,
                    "frequency": 0,
                    "prominence": 0.0,
                    "pages": set(),
                },
            )
            stats["frequency"] += 1
            weight = get_prominence_weight(ent.root.dep_)
            stats["prominence"] += weight
            stats["pages"].add(source_url)
            if qid and not stats.get("qid"):
                stats["qid"] = qid
            if kb_description and not stats.get("kb_description"):
                stats["kb_description"] = kb_description
            if kb_url and not stats.get("kb_url"):
                stats["kb_url"] = kb_url

            doc_key = (source_url, entity_id)
            doc_entity_stats.setdefault(doc_key, {"frequency": 0, "prominence": 0.0})
            doc_entity_stats[doc_key]["frequency"] += 1
            doc_entity_stats[doc_key]["prominence"] += weight

            for token in ent:
                token_entity_map[token.i] = entity_id

        triplets = extract_spo_relations(doc, token_entity_map, source_url)
        for triplet in triplets:
            subj_id = triplet["subject"]
            obj_id = triplet["object"]
            if subj_id not in entity_stats or obj_id not in entity_stats:
                continue
            predicate = triplet["predicate"]
            spo_edge_counter[(subj_id, predicate, obj_id)] += 1
            spo_rows.append(
                {
                    "Sujeto": entity_stats[subj_id]["canonical_name"],
                    "Sujeto QID": entity_stats[subj_id]["qid"],
                    "Predicado": predicate,
                    "Objeto": entity_stats[obj_id]["canonical_name"],
                    "Objeto QID": entity_stats[obj_id]["qid"],
                    "Fuente": triplet["source"],
                }
            )

    if not entity_stats:
        raise ValueError("No se detectaron entidades en el texto proporcionado.")

    # Construcción del grafo con las SPO para análisis estructural
    graph_nodes = list(entity_stats.keys())
    graph_nx = nx.DiGraph()
    graph_nx.add_nodes_from(graph_nodes)
    for (subj_id, predicate, obj_id), weight in spo_edge_counter.items():
        if subj_id in entity_stats and obj_id in entity_stats:
            graph_nx.add_edge(subj_id, obj_id, weight=weight)

    if graph_nx.number_of_nodes() > 0:
        try:
            closeness_scores = nx.closeness_centrality(graph_nx, wf_improved=True)
        except Exception:  # noqa: BLE001
            closeness_scores = {node: 0.0 for node in graph_nx.nodes}
        try:
            betweenness_scores = nx.betweenness_centrality(graph_nx)
        except Exception:  # noqa: BLE001
            betweenness_scores = {node: 0.0 for node in graph_nx.nodes}
    else:
        closeness_scores = {}
        betweenness_scores = {}

    for entity_id, stats in entity_stats.items():
        closeness = closeness_scores.get(entity_id, 0.0)
        betweenness = betweenness_scores.get(entity_id, 0.0)
        stats["closeness"] = closeness
        stats["betweenness"] = betweenness
        stats["unified_authority_score"] = stats["prominence"] + (closeness * 100.0) + (betweenness * 100.0)

    filtered_entities = {
        entity_id: stats
        for entity_id, stats in entity_stats.items()
        if stats["frequency"] >= max(1, min_entity_frequency)
    }
    if not filtered_entities:
        raise ValueError(
            "No se encontraron entidades que cumplan la frecuencia mínima. Reduce el umbral o añade más contenido."
        )

    sorted_entities = sorted(
        filtered_entities.values(),
        key=lambda item: (item["unified_authority_score"], item["frequency"]),
        reverse=True,
    )
    if max_entities > 0:
        sorted_entities = sorted_entities[:max_entities]
    top_entity_ids = {stats["qid"] or stats["canonical_name"].lower() for stats in sorted_entities}

    net = Network(height="640px", width="100%", bgcolor="#0f111a", font_color="#f5f7ff")
    net.toggle_physics(True)
    net.repulsion(node_distance=250, spring_length=220, damping=0.8)

    for stats in sorted_entities:
        entity_id = stats["qid"] or stats["canonical_name"].lower()
        label = stats["canonical_name"]
        display_label = label if len(label) <= 38 else f"{label[:35]}…"
        tooltip_parts = [
            f"<strong>{label}</strong>",
            f"Tipo: {stats['label']}",
            f"Frecuencia consolidada: {stats['frequency']}",
            f"Prominence sintáctica: {stats['prominence']:.2f}",
            f"Closeness: {stats.get('closeness', 0.0):.4f}",
            f"Betweenness: {stats.get('betweenness', 0.0):.4f}",
            f"Autoridad tópica: {stats.get('unified_authority_score', 0.0):.2f}",
        ]
        if stats.get("qid"):
            tooltip_parts.append(f"Wikidata ID: {stats['qid']}")
        if stats.get("kb_description"):
            tooltip_parts.append(stats["kb_description"])
        net.add_node(
            entity_id,
            label=display_label,
            title="<br/>".join(tooltip_parts),
            size=18 + min(int(stats.get("unified_authority_score", 0.0) / 2), 36),
            value=stats.get("unified_authority_score", stats["prominence"]),
            color="#5c6bff",
        )

    for (subj_id, predicate, obj_id), weight in spo_edge_counter.items():
        if subj_id not in top_entity_ids or obj_id not in top_entity_ids:
            continue
        net.add_edge(
            subj_id,
            obj_id,
            label=predicate,
            title=f"{predicate} Â· peso {weight}",
            value=weight,
            color="#c2d1ff",
        )

    if include_pages and url_column:
        page_scores: Dict[str, float] = {}
        for (page_url, entity_id), metrics in doc_entity_stats.items():
            if entity_id not in top_entity_ids:
                continue
            page_scores[page_url] = page_scores.get(page_url, 0.0) + metrics["prominence"]
        ranked_pages = sorted(page_scores.items(), key=lambda item: item[1], reverse=True)[:max_pages]
        allowed_pages = {url for url, _ in ranked_pages}
        for url, score in ranked_pages:
            display_url = url if len(url) <= 42 else f"{url[:39]}…"
            net.add_node(
                f"page::{url}",
                label=display_url,
                title=f"{url}<br/>Prominence acumulado: {score:.2f}",
                shape="box",
                color="#5dade2",
                size=12,
            )
        for (page_url, entity_id), metrics in doc_entity_stats.items():
            if page_url not in allowed_pages or entity_id not in top_entity_ids:
                continue
            net.add_edge(
                f"page::{page_url}",
                entity_id,
                color="#95a5a6",
                title=f"{page_url} â†’ {entity_stats[entity_id]['canonical_name']} ({metrics['prominence']:.2f})",
            )

    entities_df = pd.DataFrame(
        [
            {
                "Entidad": stats["canonical_name"],
                "QID": stats.get("qid") or "—",
                "Tipo": stats["label"],
                "Frecuencia consolidada": int(stats["frequency"]),
                "Prominence sintáctica": round(float(stats["prominence"]), 3),
                "Closeness Centrality": round(float(stats.get("closeness", 0.0)), 5),
                "Betweenness Centrality": round(float(stats.get("betweenness", 0.0)), 5),
                "Autoridad tópica unificada": round(float(stats.get("unified_authority_score", 0.0)), 3),
                "Páginas únicas": len(stats["pages"]),
                "Descripción KB": stats.get("kb_description") or "",
                "URL KB": stats.get("kb_url") or "",
            }
            for stats in sorted_entities
        ]
    )

    doc_rows = [
        {
            "URL": page_url,
            "Entidad": entity_stats[entity_id]["canonical_name"],
            "QID": entity_stats[entity_id].get("qid") or "—",
            "Frecuencia documento": metrics["frequency"],
            "Prominence documento": round(metrics["prominence"], 3),
        }
        for (page_url, entity_id), metrics in doc_entity_stats.items()
        if entity_id in top_entity_ids and page_url
    ]
    doc_relations_df = pd.DataFrame(sorted(doc_rows, key=lambda row: row["Prominence documento"], reverse=True))

    spo_df = pd.DataFrame(spo_rows)

    return net.generate_html(notebook=False), entities_df, doc_relations_df, spo_df


def clean_and_extract_domain(url: str) -> str:
    if not isinstance(url, str):
        return ""
    candidate = url.strip().strip('"').strip("'")
    if not candidate or candidate.lower() in {"no encontrado", "na", "n/a"}:
        return ""
    if not candidate.startswith(("http://", "https://")):
        candidate = "http://" + candidate
    try:
        parsed = urlparse(candidate)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""


def normalize_domain(value: str) -> str:
    return clean_and_extract_domain(value) if value else ""


def parse_position_tracking_csv(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        raise ValueError("No se proporcionó archivo.")
    raw = uploaded_file.getvalue()
    if not raw:
        raise ValueError("El archivo está vacío.")
    text = raw.decode("utf-8", errors="ignore")
    reader = csv.reader(io.StringIO(text))
    processed_rows: List[Dict[str, str]] = []
    for row in reader:
        if not row:
            continue
        first_cell = row[0].strip()
        if not first_cell.startswith("Keyword:"):
            continue
        current_keyword = first_cell.replace("Keyword:", "", 1).strip().strip('"')
        # Skip header row if present
        header_row = next(reader, None)
        data_row = next(reader, None)
        if data_row is None:
            continue
        entry: Dict[str, str] = {"Keyword": current_keyword}
        for idx in range(1, 11):
            url_value = data_row[idx] if idx < len(data_row) else ""
            domain = clean_and_extract_domain(url_value)
            entry[f"Position {idx}"] = domain or "No encontrado"
        processed_rows.append(entry)
    if not processed_rows:
        raise ValueError("No se detectaron filas válidas en el CSV. Verifica el formato.")
    return pd.DataFrame(processed_rows)


def parse_family_instructions(raw_text: str) -> List[Tuple[str, List[str]]]:
    if not raw_text:
        return []
    instructions: List[Tuple[str, List[str]]] = []
    for line in raw_text.splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#") or ":" not in cleaned:
            continue
        name, patterns = cleaned.split(":", 1)
        tokens = [token.strip().lower() for token in re.split(r"[;,]", patterns) if token.strip()]
        if tokens:
            instructions.append((name.strip(), tokens))
    return instructions


def assign_keyword_families(df: pd.DataFrame, config_text: str) -> pd.DataFrame:
    instructions = parse_family_instructions(config_text)
    df_local = df.copy()
    if not instructions:
        df_local["Familia"] = "General"
        return df_local

    def resolver(keyword: str) -> str:
        key_lower = (keyword or "").lower()
        for family, patterns in instructions:
            for pattern in patterns:
                if "*" in pattern:
                    fragment = pattern.replace("*", "")
                    if fragment and fragment in key_lower:
                        return family
                elif key_lower == pattern:
                    return family
        return "Sin familia"

    df_local["Familia"] = df_local["Keyword"].apply(resolver)
    return df_local


def summarize_positions_overview(
    df: pd.DataFrame,
    brand_domain: str,
    competitor_domains: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    normalized_brand = normalize_domain(brand_domain)
    competitor_set = {
        normalize_domain(domain)
        for domain in (competitor_domains or [])
        if domain and normalize_domain(domain)
    }
    total_keywords = len(df)
    brand_positions: List[int] = []
    competitor_counter: Counter[str] = Counter()
    for _, row in df.iterrows():
        keyword_domains = {}
        for pos_idx in range(1, 11):
            domain = row.get(f"Position {pos_idx}", "")
            if not domain or domain == "No encontrado":
                continue
            competitor_counter[domain] += 1
            keyword_domains.setdefault(domain, pos_idx)
        if normalized_brand and normalized_brand in keyword_domains:
            brand_positions.append(keyword_domains[normalized_brand])

    overview = {
        "total_keywords": total_keywords,
        "brand_keywords_in_top10": len(brand_positions),
        "brand_average_position": round(sum(brand_positions) / len(brand_positions), 2) if brand_positions else None,
        "top_competitors_by_presence": competitor_counter.most_common(10),
        "tracked_competitors": sorted(competitor_set),
    }
    return overview


def build_family_payload(
    df: pd.DataFrame,
    brand_domain: str,
    max_keywords_per_family: int = 20,
    competitor_domains: Optional[Sequence[str]] = None,
) -> List[Dict[str, object]]:
    normalized_brand = normalize_domain(brand_domain)
    competitor_set: Set[str] = {
        normalize_domain(domain)
        for domain in (competitor_domains or [])
        if domain and normalize_domain(domain)
    }
    families_payload: List[Dict[str, object]] = []
    for family_name, fam_df in df.groupby("Familia"):
        records: List[Dict[str, object]] = []
        for _, row in fam_df.head(max_keywords_per_family).iterrows():
            positions: List[Dict[str, object]] = []
            for pos_idx in range(1, 11):
                domain = row.get(f"Position {pos_idx}", "")
                if not domain or domain == "No encontrado":
                    continue
                entry = {"domain": domain, "position": pos_idx}
                if domain == normalized_brand:
                    entry["is_brand"] = True
                if domain in competitor_set:
                    entry["is_competitor"] = True
                positions.append(entry)
            records.append({"keyword": row["Keyword"], "positions": positions})
        domain_universe = sorted({pos["domain"] for rec in records for pos in rec["positions"]})
        families_payload.append(
            {
                "family": family_name,
                "domains": domain_universe,
                "keywords": records,
                "tracked_competitors": sorted(competitor_set) if competitor_set else [],
            }
        )
    return families_payload


def generate_position_report_html(
    api_key: str,
    model_name: str,
    report_title: str,
    brand_domain: str,
    families_payload: List[Dict[str, object]],
    overview: Dict[str, object],
    chart_notes: str,
    competitor_domains: Optional[Sequence[str]] = None,
) -> str:
    if genai is None:
        raise RuntimeError("La biblioteca google-generativeai no está instalada.")
    cleaned_key = (api_key or "").strip()
    if not cleaned_key:
        raise ValueError("Introduce una API key de Gemini.")
    genai.configure(api_key=cleaned_key)
    model = genai.GenerativeModel((model_name or "gemini-2.5-flash").strip())

    payload_json = json.dumps(families_payload, ensure_ascii=False)
    if len(payload_json) > 15000:
        payload_json = payload_json[:15000] + "... (truncado)"
    overview_json = json.dumps(overview, ensure_ascii=False, indent=2)
    competitors_text = ", ".join(filter(None, competitor_domains or overview.get("tracked_competitors") or [])) or "No declarados"

    prompt = f"""
Eres un consultor SEO senior. Genera un informe HTML completo, elegante y bilingÃ¼e en español para analizar el posicionamiento
orgánico de la web '{brand_domain}'. Mantén un tono profesional y estratégico.

Requisitos clave:
- Devuelve HTML válido con <html>, <head>, <style> y <body>.
- Usa una paleta profesional (azules y grises) y define las clases CSS: .pos-1, .pos-2-3, .pos-4-7, .pos-8-10, .not-found y .brand-domain.
- Estructura recomendada:
  1. Portada con el título "{report_title}" y un resumen ejecutivo.
  2. Métricas globales usando los datos del JSON resumen.
  3. Un bloque por cada familia recibida, con análisis narrativo + tabla comparativa. Columnas: Keyword + una columna por dominio. Cada celda debe mostrar la posición numérica o "No encontrado".
     Resalta el dominio de la marca aplicando la clase .brand-domain.
  4. Sección "Sugerencias de visualización" describiendo los gráficos solicitados: {chart_notes}.
  5. Recomendaciones accionables y próximos pasos.
- Escribe todo en español, con subtítulos claros y viñetas donde aporte claridad.

Datos globales:
{overview_json}

Datos por familia (JSON):
{payload_json}
"""
    response = model.generate_content(prompt)
    html_text = getattr(response, "text", "") or ""
    html_text = html_text.strip()
    if html_text.startswith("```"):
        html_text = re.sub(r"^```[a-zA-Z]*", "", html_text).strip()
        if html_text.endswith("```"):
            html_text = html_text[:-3].strip()
    if "<html" not in html_text.lower():
        raise RuntimeError("La respuesta del modelo no contiene HTML.")
    return html_text


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
        if not entity_id or entity_id in {"—", "-", "None", "nan"}:
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


def load_keywords(file: io.BytesIO, column: str) -> List[str]:
    df_keywords = pd.read_excel(file)
    keywords = df_keywords[column].dropna().astype(str).str.strip().unique().tolist()
    return keywords


def ensure_openai_key(input_key: str) -> str:
    if input_key:
        return input_key.strip()
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    raise ValueError("No se proporcionó la clave de OpenAI. Ingrésala en el campo de texto o establece la variable OPENAI_API_KEY.")

def ensure_google_kg_api_key(input_key: Optional[str] = None) -> str:
    candidate = (input_key or "").strip()
    if candidate:
        return candidate
    env_key = os.environ.get("GOOGLE_EKG_API_KEY", "").strip()
    if env_key:
        return env_key
    raise ValueError(
        "No se encontrÂ¢ una API key de Google Enterprise KG. Introduce la clave en la interfaz o define GOOGLE_EKG_API_KEY."
    )


@st.cache_data(show_spinner=False, ttl=600)
def query_google_enterprise_kg(
    mentions: Sequence[str],
    api_key: str,
    limit: int = 3,
    languages: Optional[Sequence[str]] = None,
    types: Optional[Sequence[str]] = None,
    sleep_seconds: float = 0.2,
    max_requests: int = 50,
) -> pd.DataFrame:
    """
    Consulta la Enterprise Knowledge Graph Search API y devuelve un DataFrame con los resultados.
    """
    cleaned_key = api_key.strip()
    if not cleaned_key:
        raise ValueError("La API key de Google Enterprise KG estÂ  vacÂ¡a.")

    lang_tokens = [lang.strip() for lang in (languages or ["es", "en"]) if lang.strip()]
    lang_param = ",".join(lang_tokens) or "es,en"
    type_tokens = [tp.strip() for tp in (types or []) if tp.strip()]
    type_param = ",".join(type_tokens)

    unique_mentions: List[str] = []
    seen: Set[str] = set()
    for mention in mentions:
        text = (mention or "").strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        unique_mentions.append(text)
        if len(unique_mentions) >= max_requests:
            break

    columns = [
        "Consulta",
        "Entidad",
        "Tipos KG",
        "Descripcion corta",
        "Score",
        "ID KG",
        "Detalle",
        "URL detalle",
    ]
    if not unique_mentions:
        return pd.DataFrame(columns=columns)

    base_url = "https://kgsearch.googleapis.com/v1/entities:search"
    results: List[Dict[str, object]] = []

    for query in unique_mentions:
        params = {
            "query": query,
            "key": cleaned_key,
            "limit": max(1, min(int(limit), 20)),
            "languages": lang_param,
        }
        if type_param:
            params["types"] = type_param
        request_url = f"{base_url}?{urlencode(params)}"

        try:
            with urllib.request.urlopen(request_url, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in {429, 500, 503}:
                time.sleep(sleep_seconds * 2)
                continue
            raise RuntimeError(f"Google KG devolviÂ¢ un {exc.code} para '{query}'.") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"No fue posible contactar con Google KG: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("La respuesta de Google KG no es vÂ lida.") from exc

        for element in payload.get("itemListElement", []):
            result = element.get("result") or {}
            if not result.get("name"):
                continue
            details = result.get("detailedDescription") or {}
            result_types = result.get("@type")
            if isinstance(result_types, str):
                result_types = [result_types]
            results.append(
                {
                    "Consulta": query,
                    "Entidad": result.get("name", ""),
                    "Tipos KG": ", ".join(result_types or []),
                    "Descripcion corta": result.get("description", ""),
                    "Score": float(element.get("resultScore") or 0.0),
                    "ID KG": result.get("@id", ""),
                    "Detalle": details.get("articleBody", ""),
                    "URL detalle": details.get("url", ""),
                }
            )
        time.sleep(max(0.0, sleep_seconds))

    if not results:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(results, columns=columns)
        .sort_values(["Consulta", "Score"], ascending=[True, False])
        .reset_index(drop=True)
    )


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
            return value_trimmed[len(prefix):].strip()
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


MAX_COMPETITOR_CONTENT_CHARS = 8000
MAX_URL_BODY_CHARS = 6000
MAX_URL_DESCRIPTION_CHARS = 600


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
    ax.set_title("Heatmap de relevancia semántica", fontsize=14)
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
    variants = {"Body": "", "Descripción": "", "URL": ""}
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

    variants["Descripción"] = description[:MAX_URL_DESCRIPTION_CHARS] if description else ""
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
            issues.append(f"No se pudo extraer contenido útil de {url}.")
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


def render_semantic_toolkit_section() -> None:
    st.markdown("### Herramientas de análisis semántico adicionales")
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
        st.markdown("### Consultas manuales a Google Enterprise Knowledge Graph")
        st.caption(
            "Envía consultas directas al KG sin necesidad de subir archivos o generar el grafo con spaCy."
        )

        manual_google_api_key = st.text_input(
            "API key de Google Enterprise KG (standalone)",
            value=st.session_state.get("google_kg_api_key", os.environ.get("GOOGLE_EKG_API_KEY", "")),
            type="password",
            key="google_kg_manual_api_key",
            help="Define la clave aquí o mediante la variable de entorno GOOGLE_EKG_API_KEY.",
        )
        if manual_google_api_key:
            st.session_state["google_kg_api_key"] = manual_google_api_key

        manual_mentions_raw = st.text_area(
            "Entidades o consultas (una por línea o separadas por ';')",
            value="Marca principal\nProducto estrella\nAutor reconocido",
            key="google_kg_manual_entities",
        )
        manual_mentions = parse_line_input(manual_mentions_raw, separators=("\n", ";", ","))

        manual_languages_text = st.text_input(
            "Idiomas preferidos (códigos ISO separados por comas)",
            value="es,en",
            key="google_kg_manual_languages",
        )

        manual_types_text = st.text_input(
            "Filtrar por tipos (opcional, separados por comas)",
            value="",
            key="google_kg_manual_types",
            help="Ejemplos: Person,Organization,Product. Deja vacío para permitir cualquier tipo.",
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

        st.caption(f"Se enviarán {len(manual_mentions)} consultas (límite interno de 50 por llamada).")

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
                    st.warning("Google KG no devolvió resultados para las consultas enviadas.")
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
            st.info("Todavía no hay resultados manuales de Google KG para mostrar.")

    with tab_text:
        st.markdown(
            "Compara un texto con un listado de palabras clave usando embeddings multilingÃ¼es."
        )
        main_text = st.text_area(
            "Texto principal",
            height=220,
            placeholder="Pega aquí el contenido que quieres analizar.",
            key="text_similarity_main",
        )
        keywords_raw = st.text_area(
            "Palabras clave (una por línea, acepta punto y coma o coma como separadores)",
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
                    st.warning("Indica un nombre de modelo válido.")
                else:
                    try:
                        with st.spinner("Generando embeddings y calculando similitudes..."):
                            model = get_sentence_transformer(model_name_clean)
                            results_df = compute_text_keyword_similarity(main_text, keywords, model)
                    except Exception as exc:
                        st.error(f"No se pudo cargar el modelo '{model_name_clean}': {exc}")
                    else:
                        if results_df.empty:
                            st.info("No se generaron resultados. Revisa la información introducida.")
                        else:
                            st.success("Cálculo completado.")
                            st.dataframe(results_df, use_container_width=True)
                            download_dataframe_button(
                                results_df,
                                "texto_vs_keywords.xlsx",
                                "Descargar resultados Excel",
                            )

    with tab_faq:
        st.markdown(
            "Analiza un bloque de preguntas frecuentes y obtén la relevancia semántica de cada pregunta-respuesta frente a un listado de keywords."
        )
        st.caption(
            "Formato recomendado: separa cada FAQ con una línea en blanco. Ejemplo:\n"
            "Pregunta: ¿Cuál es el horario?\n"
            "Respuesta: Nuestro servicio está disponible 24/7."
        )
        faq_text = st.text_area(
            "Preguntas frecuentes",
            height=260,
            placeholder="Introduce las preguntas frecuentes con su respuesta...",
            key="faq_text_area",
        )
        faq_keywords_raw = st.text_area(
            "Palabras clave (una por línea)",
            height=180,
            key="faq_keywords_area",
        )
        top_n_faq = st.number_input(
            "Top N resultados por pregunta",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            key="faq_topn",
        )
        model_name_faq = st.text_input(
            "Modelo de Sentence Transformers",
            value=default_model,
            help="El mismo modelo se reutiliza para preguntas y keywords.",
            key="faq_model_name",
        )

        if st.button("Calcular relevancia en FAQs", key="faq_button"):
            faq_entries = parse_faq_blocks(faq_text)
            keywords = parse_line_input(faq_keywords_raw, separators=("\n", ";", ","))
            if not faq_entries:
                st.warning("Introduce al menos una pregunta frecuente con su respuesta.")
            elif not keywords:
                st.warning("Introduce al menos una palabra clave.")
            else:
                model_name_clean = model_name_faq.strip()
                if not model_name_clean:
                    st.warning("Indica un nombre de modelo válido.")
                else:
                    try:
                        with st.spinner("Calculando relevancia semántica en FAQs..."):
                            model = get_sentence_transformer(model_name_clean)
                            faq_results = compute_faq_keyword_similarity(faq_entries, keywords, model)
                    except Exception as exc:
                        st.error(f"No se pudo cargar el modelo '{model_name_clean}': {exc}")
                    else:
                        if faq_results.empty:
                            st.info("No se generaron resultados. Revisa la información introducida.")
                        else:
                            st.success("Cálculo completado.")
                            parsed_df = pd.DataFrame(faq_entries, columns=["Pregunta", "Respuesta"])
                            st.markdown("**Preguntas frecuentes detectadas**")
                            st.dataframe(parsed_df, use_container_width=True)
                            top_results = top_n_by_group(faq_results, "Pregunta", "Similitud", int(top_n_faq))
                            st.markdown("**Relevancia semántica por pregunta**")
                            st.dataframe(top_results, use_container_width=True)
                            download_dataframe_button(
                                faq_results,
                                "faq_vs_keywords.xlsx",
                                "Descargar resultados completos",
                            )

    with tab_competitors:
        st.markdown(
            "Extrae el contenido de URLs de competidores, genera embeddings y calcula la relevancia respecto a tus queries."
        )
        competitor_urls_raw = st.text_area(
            "URLs de competidores (una por línea)",
            height=220,
            placeholder="https://example.com/pagina-1\nhttps://example.com/pagina-2",
            key="competitor_urls_area",
        )
        competitor_queries_raw = st.text_area(
            "Consultas o keywords (una por línea)",
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
            elif not queries:
                st.warning("Introduce al menos una consulta.")
            else:
                model_name_clean = model_name_competitors.strip()
                if not model_name_clean:
                    st.warning("Indica un nombre de modelo válido.")
                else:
                    try:
                        with st.spinner("Cargando modelo..."):
                            model = get_sentence_transformer(model_name_clean)
                    except Exception as exc:
                        st.error(f"No se pudo cargar el modelo '{model_name_clean}': {exc}")
                    else:
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
                            st.error("No se extrajo contenido útil de las URLs indicadas.")
                        else:
                            with st.spinner("Calculando relevancia semántica..."):
                                ordered_urls = list(extracted.keys())
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
                                st.info("No se generaron resultados. Revisa la información introducida.")
                            else:
                                results_df.sort_values(["Query", "Score"], ascending=[True, False], inplace=True)
                                results_df.reset_index(drop=True, inplace=True)
                                top_results = top_n_by_group(results_df, "Query", "Score", int(top_n_competitors))
                                st.success("Análisis completado.")
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

                                with st.expander("Contenido extraído (primeros 400 caracteres por URL)"):
                                    for url in ordered_urls:
                                        preview = extracted[url][:400].replace("\n", " ")
                                        ellipsis = "…" if len(extracted[url]) > 400 else ""
                                        st.markdown(f"**{url}**")
                                        st.write(preview + ellipsis)

                                st.caption(
                                    f"Se analizaron {len(ordered_urls)} URLs con un límite de {MAX_COMPETITOR_CONTENT_CHARS} caracteres por página."
                                )

    with tab_url_variants:
        st.markdown(
            "Procesa un listado de URLs, genera embeddings específicos (cuerpo, descripción larga y texto de la URL) y evalúa la relevancia frente a tus palabras clave."
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
            "URLs adicionales (una por línea)",
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
            "Palabras clave adicionales (una por línea)",
            height=180,
            key="urls_keywords_textarea",
        )
        manual_keywords = parse_line_input(urls_keywords_raw, separators=("\n", ";", ","))

        urls_model_name = st.text_input(
            "Modelo de Sentence Transformers",
            value=default_model,
            help="El modelo se usa para generar los embeddings de página y las palabras clave.",
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
            elif not combined_keywords:
                st.warning("Introduce al menos una palabra clave (archivo o campo de texto).")
            else:
                model_name_clean = urls_model_name.strip()
                if not model_name_clean:
                    st.warning("Indica un nombre de modelo válido.")
                else:
                    try:
                        with st.spinner("Cargando modelo..."):
                            model = get_sentence_transformer(model_name_clean)
                    except Exception as exc:
                        st.error(f"No se pudo cargar el modelo '{model_name_clean}': {exc}")
                    else:
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
                                extracted_variants[url_value] = {"Body": "", "Descripción": "", "URL": ""}
                            else:
                                entries.extend(new_entries)
                                extracted_variants.update(components_map)
                                issues.extend(entry_issues)
                            progress.progress(idx / total_urls if total_urls else 1.0)
                        progress.empty()
                        status.empty()

                        if not entries:
                            st.error("No se pudo extraer contenido útil de las URLs indicadas.")
                            for warning_msg in issues:
                                st.warning(warning_msg)
                        else:
                            for warning_msg in issues:
                                st.warning(warning_msg)

                            with st.spinner("Calculando relevancia semántica..."):
                                results_df = compute_url_variant_keyword_similarity(entries, combined_keywords, model)

                            if results_df.empty:
                                st.info("No se generaron resultados. Revisa las URLs y palabras clave proporcionadas.")
                            else:
                                st.success("Análisis completado.")
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

                                with st.expander("Detalle de textos extraídos por URL"):
                                    for url_value in combined_urls:
                                        components = extracted_variants.get(
                                            url_value,
                                            {"Body": "", "Descripción": "", "URL": ""},
                                        )
                                        st.markdown(f"**{url_value}**")
                                        for label, text_value in components.items():
                                            if not text_value:
                                                st.markdown(f"- `{label}`: (sin contenido)")
                                                continue
                                            preview = text_value[:500].replace("\n", " ").strip()
                                            ellipsis = "…" if len(text_value) > 500 else ""
                                            st.markdown(f"- `{label}`: {preview}{ellipsis}")
                                        st.markdown("---")



    with tab_authority:
        st.markdown("### Brechas de autoridad sin CSV")
        st.caption(
            "Simula brechas con datos ficticios o reutiliza las URLs de la pestana de competidores para detectar huecos tematicos sin subir archivos."
        )

        st.session_state.setdefault("authority_gap_result", None)
        competitor_payload = st.session_state.get("competitor_tool_payload")
        if competitor_payload:
            st.info(
                f"Hay {len(competitor_payload.get('urls', []))} URLs guardadas desde 'Competidores vs queries'. Ejecuta de nuevo esa pestaña para refrescarlas."
            )
        else:
            st.caption("Aún no hay URLs almacenadas desde la pestaña de competidores en esta sesión.")

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
                    "Dimensión embeddings",
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
                    st.success("Simulación completada. Revisa los clusters y brechas detectadas.")
        else:
            default_competitors = "\n".join(competitor_payload.get("urls", [])) if competitor_payload else ""
            competitor_area = st.text_area(
                "URLs de competencia (una por línea)",
                value=default_competitors,
                height=200,
                key="authority_manual_comp_urls",
            )
            own_area = st.text_area(
                "URLs de tu sitio (una por línea)",
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

            def _embed_urls(
                urls: List[str],
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
                        f"No se obtuvieron contenidos válidos para las URLs de {label}. "
                        "Comprueba que las páginas sean accesibles."
                    )
                embeddings = model.encode(texts, convert_to_numpy=True)
                return embeddings, used_urls, issues

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
                            st.success("Análisis completado con tus URLs. Revisa los clusters detectados.")

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
                st.warning(f"Se detectaron {len(gap_rows)} gap(s) temáticos por debajo del umbral.")
                st.dataframe(pd.DataFrame(gap_rows), use_container_width=True)
            else:
                st.success("Con los parámetros actuales no se registran gaps de autoridad.")
        else:
            if mode == "Simulación guiada":
                st.info("Configura los parámetros y pulsa el botón para simular posibles gaps sin cargar un CSV.")
            else:
                st.info("Introduce tus URLs y ejecuta el análisis para detectar brechas reales sin CSV.")

def main():
    apply_global_styles()
    st.title("📈 Embedding Insights Dashboard")
    st.markdown(
        "Sube tus datos de embeddings para descubrir similitudes entre URLs, agruparlas en clusters "
        "y analizar la relevancia frente a palabras clave."
    )

    if "gemini_api_key" not in st.session_state:
        st.session_state["gemini_api_key"] = (
            os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
        )
    if "gemini_model_name" not in st.session_state:
        st.session_state["gemini_model_name"] = os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash"
    if "entity_payload_by_url" not in st.session_state:
        st.session_state["entity_payload_by_url"] = None
    if "processed_df" not in st.session_state:
        st.session_state["processed_df"] = None
    if "raw_df" not in st.session_state:
        st.session_state["raw_df"] = None
    if "url_column" not in st.session_state:
        st.session_state["url_column"] = None
    if "cluster_result" not in st.session_state:
        st.session_state["cluster_result"] = None
    if "tsne_figure" not in st.session_state:
        st.session_state["tsne_figure"] = None
    if "topic_graph_html" not in st.session_state:
        st.session_state["topic_graph_html"] = None
    if "knowledge_graph_html" not in st.session_state:
        st.session_state["knowledge_graph_html"] = None
    if "knowledge_entities" not in st.session_state:
        st.session_state["knowledge_entities"] = None
    if "knowledge_doc_relations" not in st.session_state:
        st.session_state["knowledge_doc_relations"] = None
    if "knowledge_spo_relations" not in st.session_state:
        st.session_state["knowledge_spo_relations"] = None
    if "auto_cluster_scores" not in st.session_state:
        st.session_state["auto_cluster_scores"] = None
    if "auto_cluster_best_k" not in st.session_state:
        st.session_state["auto_cluster_best_k"] = None
    if "page_type_column" not in st.session_state:
        st.session_state["page_type_column"] = None
    if "semantic_links_report" not in st.session_state:
        st.session_state["semantic_links_report"] = None
    if "semantic_links_adv_report" not in st.session_state:
        st.session_state["semantic_links_adv_report"] = None
    if "semantic_links_adv_orphans" not in st.session_state:
        st.session_state["semantic_links_adv_orphans"] = None

    if "app_view" not in st.session_state:
        st.session_state["app_view"] = "landing"

    render_api_settings_panel()
    render_sidebar_navigation()  # ← Nueva línea: menú de navegación permanente

    app_view = st.session_state["app_view"]

    if app_view == "csv":
        with st.sidebar:
            st.header("Pasos")
            st.markdown("1. Subir archivo con embeddings.")
            st.markdown("2. Ejecutar análisis de similitud.")
            st.markdown("3. Encontrar enlaces internos semánticos.")
            st.markdown("4. Aplicar enlazado avanzado por silos (opcional).")
            st.markdown("5. Ejecutar clustering (opcional).")
            st.markdown("6. Analizar relevancia por palabras clave.")
            st.markdown("7. Construir grafo de conocimiento (opcional).")

    if app_view == "landing":
        render_landing_view()
        return

    render_back_to_landing()

    if app_view == "csv":
        render_csv_workflow()
    elif app_view == "tools":
        render_semantic_toolkit_section()
    elif app_view == "keywords":
        render_semantic_keyword_builder()
    elif app_view == "linking":
        render_linking_lab()
    elif app_view == "positions":
        render_positions_report()


def render_csv_workflow():
    card = bordered_container()
    with card:
        st.markdown("### Carga de Datos")
        st.caption("Sube tu dataset con embeddings y define la columna vectorial antes de habilitar los análisis avanzados.")
        uploaded_file = st.file_uploader(
            "Archivo CSV o Excel con embeddings",
            type=["csv", "xlsx", "xls"],
            label_visibility="collapsed",
        )
        st.caption("Arrastra y suelta tu archivo en el área o presiona el botón para explorarlo (max. 200â€¯MB).")
        extra_cols = st.columns(3)
        extra_cols[0].markdown("✅ **Formato sugerido:** columnas con URL, tipo, embeddings.")
        extra_cols[1].markdown("📄 **Ejemplo de archivo:** `embeddings_site.xlsx`.")
        extra_cols[2].markdown("🧠­ **Tip:** asegúrate de que todos los embeddings tengan el mismo tamaño.")

    if uploaded_file:
        try:
            with st.spinner("Leyendo archivo..."):
                file_name = (uploaded_file.name or "").lower()
                if file_name.endswith(".csv"):
                    site_df = pd.read_csv(uploaded_file)
                else:
                    site_df = pd.read_excel(uploaded_file)
        except Exception as exc:
            st.error(f"No se pudo leer el archivo proporcionado: {exc}")
            return
        st.markdown("#### Vista previa de datos")
        st.dataframe(site_df.head(), use_container_width=True)

        candidate_embedding_cols = detect_embedding_columns(site_df)
        embedding_col = st.selectbox(
            "Selecciona la columna de embeddings",
            options=candidate_embedding_cols or site_df.columns.tolist(),
        )

        try:
            processed_df, info_messages = preprocess_embeddings(site_df, embedding_col)
            st.success(f"Embeddings procesados correctamente. {len(processed_df)} filas listas.")
            for msg in info_messages:
                st.info(msg)
        except ValueError as exc:
            st.error(str(exc))
            return

        url_cols = detect_url_columns(processed_df)
        url_column = st.selectbox("Selecciona la columna de URL", options=url_cols, index=0)
        processed_df[url_column] = processed_df[url_column].astype(str).str.strip()

        st.session_state["raw_df"] = site_df
        st.session_state["processed_df"] = processed_df
        st.session_state["url_column"] = url_column
        st.session_state["cluster_result"] = None
        st.session_state["tsne_figure"] = None
        st.session_state["topic_graph_html"] = None
        st.session_state["knowledge_graph_html"] = None
        st.session_state["knowledge_entities"] = None
        st.session_state["knowledge_doc_relations"] = None
        st.session_state["knowledge_spo_relations"] = None
        st.session_state["auto_cluster_scores"] = None
        st.session_state["auto_cluster_best_k"] = None
        st.session_state["page_type_column"] = None
        st.session_state["semantic_links_report"] = None
        st.session_state["semantic_links_adv_report"] = None
        st.session_state["semantic_links_adv_orphans"] = None

    processed_df = st.session_state.get("processed_df")
    url_column = st.session_state.get("url_column")

    if processed_df is None or url_column is None:
        st.warning("Sube y procesa un archivo para habilitar el resto de funcionalidades.")
        return

    st.divider()
    st.subheader("2️⃣ Análisis de similitud entre páginas")
    st.caption("Compara cada URL con el resto para encontrar sus páginas gemelas y generar matrices filtradas por similitud.")
    col_select, col_topn = st.columns([3, 1])
    with col_select:
        target_url = st.selectbox("Selecciona la URL objetivo", options=processed_df[url_column].tolist())
    with col_topn:
        top_n = st.number_input("Top N", min_value=3, max_value=50, value=10, step=1)

    if st.button("Calcular páginas más similares", type="primary"):
        with st.spinner("Calculando similitudes..."):
            results = compute_similar_pages(processed_df, target_url, url_column, top_n)
        if results:
            df_similar = pd.DataFrame(results, columns=["URL", "Similitud (%)"])
            st.dataframe(df_similar)
            download_dataframe_button(df_similar, "similares.csv", "Descargar resultados CSV")
        else:
            st.info("No se encontraron páginas similares o la URL no existe.")

    with st.expander("Opcional: generar matriz completa de similitud"):
        threshold = st.slider("Umbral mínimo de similitud (%)", min_value=0, max_value=100, value=70, step=1)
        max_pairs = st.number_input(
            "Máximo de pares a conservar (0 para ilimitado)",
            min_value=0,
            value=100000,
            step=1000,
        )
        if st.button("Construir matriz filtrada"):
            with st.spinner("Calculando matriz de similitud..."):
                matrix_df = build_similarity_matrix(
                    processed_df,
                    url_column,
                    similarity_threshold=threshold,
                    max_results=None if max_pairs == 0 else int(max_pairs),
                )
            if matrix_df.empty:
                st.info("No hay pares que cumplan el umbral establecido.")
            else:
                st.dataframe(matrix_df.head(100))
                st.caption(f"{len(matrix_df)} pares generados.")
                download_dataframe_button(matrix_df, "matriz_similitud.xlsx", "Descargar matriz Excel")

    st.info(
        "¿Buscas recomendaciones de enlazado interno? Abre el **Laboratorio de enlazado** desde la pantalla principal "
        "para acceder a los modos básico, avanzado, híbrido CLS y estructural."
    )

    st.divider()
    st.subheader("3️⃣ Clustering de páginas")
    st.caption("Agrupa las URLs en clusters temáticos, evalúa silhouette y genera visualizaciones t-SNE y grafos interactivos.")
    num_pages = len(processed_df)
    if num_pages < 3:
        st.info("Se necesitan al menos 3 páginas para realizar clustering.")
    else:
        max_clusters_allowed = min(25, num_pages)
        cluster_mode = st.radio(
            "Modo de selección de clusters",
            options=["Automática", "Manual"],
            index=0,
            horizontal=True,
            key="cluster_mode_selector",
        )

        if cluster_mode == "Manual":
            default_clusters = min(10, max(3, int(math.sqrt(num_pages))))
            default_clusters = min(default_clusters, max_clusters_allowed)
            n_clusters = st.slider(
                "Número de clusters",
                min_value=2,
                max_value=max_clusters_allowed,
                value=default_clusters,
                key="manual_cluster_slider",
            )
            if st.button("Ejecutar clustering", key="manual_cluster_button"):
                with st.spinner("Agrupando páginas..."):
                    df_clustered, cluster_names, model, embeddings_norm = cluster_pages(
                        processed_df, int(n_clusters), url_column
                    )
                st.session_state["cluster_result"] = (df_clustered, cluster_names, model, embeddings_norm)
                st.session_state["auto_cluster_scores"] = None
                st.session_state["auto_cluster_best_k"] = None
                st.success("Clustering completado.")
        else:
            auto_upper_bound = min(15, max_clusters_allowed, num_pages - 1)
            if auto_upper_bound <= 2:
                st.info("Con las páginas disponibles solo es posible evaluar 2 clusters.")
                auto_max_clusters = 2
            else:
                auto_max_clusters = st.slider(
                    "Máximo de clusters a evaluar",
                    min_value=2,
                    max_value=auto_upper_bound,
                    value=min(8, auto_upper_bound),
                    help="Se usa la métrica silhouette para elegir el mejor número entre 2 y el máximo indicado.",
                    key="auto_cluster_slider",
                )
            if st.button("Calcular y agrupar automáticamente", key="auto_cluster_button"):
                try:
                    best_k, score_map = auto_select_cluster_count(
                        processed_df,
                        min_clusters=2,
                        max_clusters=int(auto_max_clusters),
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    scores_df = (
                        pd.DataFrame(
                            {"Clusters": list(score_map.keys()), "Silhouette": list(score_map.values())}
                        )
                        .sort_values("Clusters")
                        .reset_index(drop=True)
                    )
                    scores_df["Silhouette"] = scores_df["Silhouette"].round(4)
                    scores_df["Mejor"] = np.where(scores_df["Clusters"] == best_k, "★", "")
                    st.session_state["auto_cluster_scores"] = scores_df
                    st.session_state["auto_cluster_best_k"] = best_k
                    best_score = score_map.get(best_k, float("nan"))
                    with st.spinner(f"Agrupando páginas con {best_k} clusters..."):
                        df_clustered, cluster_names, model, embeddings_norm = cluster_pages(
                            processed_df, best_k, url_column
                        )
                    st.session_state["cluster_result"] = (df_clustered, cluster_names, model, embeddings_norm)
                    st.success(
                        f"Clustering completado con {best_k} clusters."
                        + (f" (silhouette={best_score:.3f})" if not np.isnan(best_score) else "")
                    )

        if st.session_state["cluster_result"]:
            df_clustered, cluster_names, model, embeddings_norm = st.session_state["cluster_result"]
            st.dataframe(
                df_clustered[[url_column, "Cluster", "Nombre_Cluster"]].sort_values("Cluster"), use_container_width=True
            )

            auto_scores_df = st.session_state.get("auto_cluster_scores")
            if isinstance(auto_scores_df, pd.DataFrame) and not auto_scores_df.empty:
                st.markdown("**Evaluación silhouette por número de clusters**")
                st.dataframe(auto_scores_df, use_container_width=True)

            if st.button("Generar visualización t-SNE"):
                try:
                    fig = tsne_visualisation(df_clustered, embeddings_norm, cluster_names)
                    st.pyplot(fig, clear_figure=True)
                    st.session_state["tsne_figure"] = fig
                except ValueError as exc:
                    st.error(str(exc))

            if st.button("Generar grafo de topic clusters"):
                with st.spinner("Construyendo grafo interactivo..."):
                    graph_html = generate_topic_cluster_graph(
                        df_clustered,
                        cluster_names,
                        model,
                        url_column=url_column,
                    )
                st.session_state["topic_graph_html"] = graph_html

            download_dataframe_button(df_clustered, "paginas_clusterizadas.xlsx", "Descargar clustering Excel")

            if st.session_state["tsne_figure"]:
                buffer = io.BytesIO()
                st.session_state["tsne_figure"].savefig(buffer, format="png", dpi=300, bbox_inches="tight")
                buffer.seek(0)
                st.download_button(
                    "Descargar visualización PNG",
                    data=buffer,
                    file_name="clusters_tsne.png",
                    mime="image/png",
                )

            if st.session_state["topic_graph_html"]:
                components.html(st.session_state["topic_graph_html"], height=620, scrolling=True)
                st.download_button(
                    "Descargar grafo HTML",
                    data=st.session_state["topic_graph_html"].encode("utf-8"),
                    file_name="topic_clusters_graph.html",
                    mime="text/html",
                )

    st.divider()
    st.subheader("4️⃣ Relevancia de palabras clave (OpenAI)")
    st.caption("Cruza tus keywords con embeddings para determinar qué URLs son más relevantes por query (requiere `OPENAI_API_KEY`).")
    api_key_input = st.text_input("OpenAI API Key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
    keywords_file = st.file_uploader("Archivo Excel con palabras clave", type=["xlsx", "xls"], key="keywords_upload")

    if keywords_file:
        keywords_df = pd.read_excel(keywords_file)
        st.dataframe(keywords_df.head())
        keyword_columns = keywords_df.columns.tolist()
        keyword_col = st.selectbox("Columna de palabras clave", options=keyword_columns)
        keywords = keywords_df[keyword_col].dropna().astype(str).str.strip().unique().tolist()
        st.write(f"{len(keywords)} palabras clave detectadas.")

        top_n_keywords = st.number_input("Top N URLs por palabra clave", min_value=1, max_value=50, value=5)
        min_score = st.slider("Umbral mínimo de relevancia (%)", min_value=0, max_value=100, value=50, step=1)

        if st.button("Analizar relevancia"):
            try:
                api_key = ensure_openai_key(api_key_input)
            except ValueError as exc:
                st.error(str(exc))
            else:
                with st.spinner("Calculando relevancia..."):
                    relevance_df = keyword_relevance(
                        processed_df,
                        keywords,
                        api_key,
                        embedding_col="EmbeddingsFloat",
                        url_column=url_column,
                        top_n=top_n_keywords,
                        min_score=min_score,
                    )
                if "Error" in relevance_df.columns:
                    st.warning("Algunas palabras clave generaron errores. Revisa la tabla.")
                st.dataframe(relevance_df)
                download_dataframe_button(relevance_df, "relevancia_palabras_clave.xlsx", "Descargar relevancia Excel")


    st.divider()
    st.subheader("5️⃣ Grafo de conocimiento y entidades")
    st.caption(
        "Audita tus textos con canonicalización de entidades, QIDs de Wikidata y tripletes SPO para evaluar E-E-A-T y Topic Authority."
    )
    with st.expander("¿Qué incluye este análisis avanzado?"):
        st.markdown(
            "- **Canonicalización**: agrupa alias y pronombres mediante coreferencia opcional.\n"
            "- **Entity linking**: vincula entidades al Knowledge Graph (Wikidata) para enriquecer metadata.\n"
            "- **Prominence score**: pondera las menciones por su rol sintáctico (nsubj, dobj, etc.).\n"
            "- **Tripletes SPO**: detecta relaciones Sujeto-Predicado-Objeto para construir un grafo semántico interno."
        )
    raw_df = st.session_state.get("raw_df")
    if raw_df is None or raw_df.empty:
        st.info("Sube y prepara un archivo con embeddings para habilitar esta sección.")
    else:
        text_columns = raw_df.select_dtypes(include=["object"]).columns.tolist()
        if not text_columns:
            st.info("No se detectaron columnas de texto en el archivo original.")
        else:
            text_column = st.selectbox("Columna con contenido textual", options=text_columns)
            use_existing_url = st.checkbox(
                "Usar la misma columna de URL seleccionada anteriormente",
                value=st.session_state.get("url_column") in raw_df.columns,
            )
            if use_existing_url and st.session_state.get("url_column") in raw_df.columns:
                knowledge_url_col = st.session_state.get("url_column")
            else:
                candidate_url_cols = ["(ninguna)"] + detect_url_columns(raw_df)
                selected_url_option = st.selectbox("Columna de URL (opcional)", options=candidate_url_cols)
                knowledge_url_col = None if selected_url_option == "(ninguna)" else selected_url_option

            max_rows = min(len(raw_df), 200)
            row_limit = st.slider(
                "Número máximo de filas a procesar",
                min_value=1,
                max_value=max_rows,
                value=min(50, max_rows),
            )
            max_entities = st.slider(
                "Número máximo de entidades en el grafo",
                min_value=10,
                max_value=300,
                value=80,
                step=5,
            )
            include_pages = st.checkbox("Incluir nodos de página en el grafo", value=True)
            max_pages = 0
            if include_pages:
                max_pages = st.slider(
                    "Número máximo de páginas a conectar",
                    min_value=5,
                    max_value=100,
                    value=25,
                    step=5,
                )

            model_name = st.text_input(
                "Modelo de spaCy",
                value="es_core_news_sm",
                help="Ejemplos: 'es_core_news_sm' para español, 'en_core_web_sm' para inglés.",
            )
            st.caption(
                "El modelo debe estar instalado en el entorno. Si aparece un error, ejecuta "
                "'python -m spacy download nombre_del_modelo' antes de volver a intentarlo."
            )

            base_entity_labels = [
                "ORG",
                "PERSON",
                "PRODUCT",
                "GPE",
                "LOC",
                "FAC",
                "EVENT",
                "WORK_OF_ART",
                "LAW",
                "NORP",
                "LANGUAGE",
            ]
            extra_labels = sorted(
                {
                    label
                    for labels in ENTITY_PROFILE_PRESETS.values()
                    for label in labels
                    if label not in base_entity_labels
                }
            )
            entity_label_options = base_entity_labels + extra_labels
            profile_options = ["(Sin plantilla)"] + list(ENTITY_PROFILE_PRESETS.keys())
            stored_profile = st.session_state.get("knowledge_entity_profile_choice", "(Sin plantilla)")
            profile_index = profile_options.index(stored_profile) if stored_profile in profile_options else 0
            profile_choice = st.selectbox(
                "Plantilla de entidades seg?n el sector",
                options=profile_options,
                index=profile_index,
                help="Aplica una lista de entidades sugeridas seg?n el tipo de cliente (cl?nica, editorial, ecommerce, etc.).",
                key="knowledge_entity_profile_choice",
            )
            preset_labels = ENTITY_PROFILE_PRESETS.get(profile_choice, [])
            if profile_choice != st.session_state.get("knowledge_last_profile_choice"):
                if preset_labels:
                    st.session_state["knowledge_graph_entity_labels"] = preset_labels
                st.session_state["knowledge_last_profile_choice"] = profile_choice
            custom_extra = st.text_input(
                "Etiquetas adicionales (separadas por coma)",
                value=st.session_state.get("knowledge_entity_extra_labels", ""),
                help="A?ade etiquetas espec?ficas (por ejemplo DISEASE, MEDICATION, BOOK_SERIES).",
                key="knowledge_entity_extra_labels",
            )
            default_selection = st.session_state.get(
                "knowledge_graph_entity_labels",
                preset_labels or ["ORG", "PRODUCT", "PERSON", "GPE"],
            )
            selected_labels = st.multiselect(
                "Tipos de entidad a incluir",
                options=entity_label_options,
                default=default_selection,
                help="Filtra los tipos de entidades que se mostrar?n en el grafo. Si no seleccionas ninguna, se incluir?n todas.",
                key="knowledge_graph_entity_labels",
            )
            custom_label_set = {
                label.strip().upper()
                for label in (custom_extra or "").split(",")
                if label and label.strip()
            }
            combined_labels = set(selected_labels)
            if custom_label_set:
                combined_labels.update(custom_label_set)

            min_entity_freq = st.slider(
                "Frecuencia mínima de una entidad para incluirla",
                min_value=1,
                max_value=10,
                value=2,
                help="Una entidad debe aparecer al menos este número de veces para entrar en el grafo.",
            )
            col_pipeline = st.columns(2)
            with col_pipeline[0]:
                enable_coref = st.checkbox(
                    "Activar coreferencia (coreferee)",
                    value=False,
                    help="Agrupa pronombres y aliases bajo la misma entidad. Requiere instalar `coreferee`.",
                )
            with col_pipeline[1]:
                enable_linking = st.checkbox(
                    "Activar entity linking (Wikidata)",
                    value=False,
                    help="Necesita `spacy-entity-linker` y la descarga de su KB (~1.3GB).",
                )
            col_perf = st.columns(2)
            with col_perf[0]:
                n_process = st.number_input(
                    "Procesos paralelos (spaCy n_process)",
                    min_value=1,
                    max_value=8,
                    value=1,
                    step=1,
                )
            with col_perf[1]:
                batch_size = st.number_input(
                    "Batch size spaCy",
                    min_value=10,
                    max_value=2000,
                    value=200,
                    step=10,
                )
            
            # Control manual de entidades (whitelist y blacklist)
            st.markdown("#### 🎯 Control manual de entidades")
            st.caption(
                "Afina los resultados añadiendo entidades prioritarias o excluyendo ruido. "
                "Las entidades prioritarias recibirán un boost de prominence (3x)."
            )
            
            col_manual_1, col_manual_2 = st.columns(2)
            
            with col_manual_1:
                manual_entities_input = st.text_area(
                    "Entidades prioritarias (una por línea)",
                    value="",
                    height=120,
                    help="Lista de entidades que DEBEN aparecer con alta prominence. "
                         "Ejemplo: tratamientos médicos, productos específicos, etc.",
                    placeholder="fibromialgia\nartritis reumatoide\nozono terapia\nacupuntura",
                    key="knowledge_manual_entities"
                )
            
            with col_manual_2:
                default_blacklist = [
                    "página", "sitio web", "artículo", "post", "blog",
                    "usuario", "cliente", "persona",
                    "ver más", "leer más", "más información",
                    "consultas", "centros", "centros encontrados",
                    "pedir", "solicitar", "contactar"
                ]
                blacklist_input = st.text_area(
                    "Entidades a excluir (una por línea)",
                    value="\n".join(default_blacklist),
                    height=120,
                    help="Lista de entidades que serán ignoradas. "
                         "Útil para filtrar ruido como elementos de navegación.",
                    key="knowledge_blacklist_entities"
                )
            
            # Parsear inputs
            manual_entities_list = [
                line.strip() 
                for line in (manual_entities_input or "").split("\n") 
                if line.strip()
            ] if manual_entities_input else None
            
            blacklist_entities_list = [
                line.strip() 
                for line in (blacklist_input or "").split("\n") 
                if line.strip()
            ] if blacklist_input else None
            
            spacy_ready = is_spacy_available()
            if not spacy_ready:
                detail = (
                    f" Detalle: {SPACY_IMPORT_ERROR}" if SPACY_IMPORT_ERROR is not None else ""
                )
                st.warning(
                    "spaCy no está instalado o tiene conflictos binarios. "
                    "Instálalo/ajústalo (por ejemplo `pip install spacy==3.5.4`) antes de ejecutar el análisis."
                    + detail
                )

            if st.button("Generar grafo de conocimiento"):
                try:
                    from app_sections.knowledge_graph import generate_knowledge_graph_html_v2
                    graph_html, entities_df, doc_relations_df, spo_df = generate_knowledge_graph_html_v2(
                        raw_df,
                        text_column=text_column,
                        url_column=knowledge_url_col,
                        model_name=model_name.strip(),
                        row_limit=int(row_limit),
                        max_entities=int(max_entities),
                        min_entity_frequency=int(min_entity_freq),
                        include_pages=include_pages,
                        max_pages=int(max_pages),
                        allowed_entity_labels=combined_labels if combined_labels else None,
                        enable_coref=enable_coref,
                        enable_linking=enable_linking,
                        n_process=int(n_process),
                        batch_size=int(batch_size),
                        manual_entities=manual_entities_list,
                        blacklist_entities=blacklist_entities_list,
                    )
                except RuntimeError as exc:
                    st.error(str(exc))
                except ValueError as exc:
                    st.warning(str(exc))
                else:
                    st.success("Grafo de conocimiento generado correctamente.")
                    st.session_state["knowledge_graph_html"] = graph_html
                    st.session_state["knowledge_entities"] = entities_df
                    st.session_state["knowledge_doc_relations"] = doc_relations_df
                    st.session_state["knowledge_spo_relations"] = spo_df
                    try:
                        entity_payload = build_entity_payload_from_doc_relations(doc_relations_df)
                    except Exception:
                        entity_payload = {}
                    if entity_payload:
                        st.session_state["entity_payload_by_url"] = entity_payload
                        processed_df = st.session_state.get("processed_df")
                        url_column_session = st.session_state.get("url_column")
                        helper_col = "EntityPayloadFromGraph"
                        if (
                            isinstance(processed_df, pd.DataFrame)
                            and url_column_session in processed_df.columns
                        ):
                            processed_df[helper_col] = processed_df[url_column_session].astype(str).map(
                                lambda url: entity_payload.get(url.strip()) or {}
                            )
                            st.session_state["processed_df"] = processed_df
                        st.info(
                            "Los resultados de entidades se guardaron y ahora puedes usarlos en el laboratorio "
                            "de enlazado hibrido."
                        )

            if st.session_state.get("knowledge_graph_html"):
                components.html(st.session_state["knowledge_graph_html"], height=640, scrolling=True)
                st.download_button(
                    "Descargar grafo de conocimiento (HTML)",
                    data=st.session_state["knowledge_graph_html"].encode("utf-8"),
                    file_name="knowledge_graph.html",
                    mime="text/html",
                )

            entities_summary = st.session_state.get("knowledge_entities")
            if isinstance(entities_summary, pd.DataFrame) and not entities_summary.empty:
                st.markdown("**Entidades más relevantes**")
                st.dataframe(entities_summary, use_container_width=True)
                download_dataframe_button(
                    entities_summary,
                    "entidades_grafo.xlsx",
                    "Descargar listado de entidades",
                )
                st.markdown("**Enriquecer entidades con Google Enterprise Knowledge Graph**")
                google_kg_api_key_input = st.text_input(
                    "API key de Google Enterprise KG",
                    value=st.session_state.get("google_kg_api_key", ""),
                    type="password",
                    help="Necesitas habilitar la Enterprise Knowledge Graph Search API en Google Cloud.",
                )
                if google_kg_api_key_input:
                    st.session_state["google_kg_api_key"] = google_kg_api_key_input
                available_entities = (
                    entities_summary["Entidad"].dropna().astype(str).str.strip().unique().tolist()
                )
                default_selection = available_entities[: min(15, len(available_entities))]
                selected_entities = st.multiselect(
                    "Selecciona las entidades detectadas para consultarlas en el KG",
                    options=available_entities,
                    default=default_selection,
                    help="Limita la consulta al subconjunto más relevante de tu grafo.",
                )
                manual_mentions_raw = st.text_area(
                    "Entidades adicionales (una por línea o separadas por ';')",
                    value="",
                    help="Añade consultas personalizadas para cruzar información con Google KG.",
                )
                manual_mentions = parse_line_input(manual_mentions_raw, separators=("\n", ";", ","))
                google_kg_languages_value = st.text_input(
                    "Idiomas preferidos (códigos ISO separados por comas)",
                    value="es,en",
                )
                google_kg_types_value = st.text_input(
                    "Filtrar por tipos (opcional, separados por comas)",
                    value="",
                    help="Ejemplo: Person,Organization,Product. Deja vacío para permitir cualquier tipo.",
                )
                google_kg_limit = st.slider(
                    "Resultados por entidad (Google KG)",
                    min_value=1,
                    max_value=10,
                    value=3,
                )
                merged_mentions: List[str] = []
                seen_mentions: Set[str] = set()
                for entry in selected_entities + manual_mentions:
                    cleaned = (entry or "").strip()
                    if not cleaned:
                        continue
                    lowered = cleaned.lower()
                    if lowered in seen_mentions:
                        continue
                    seen_mentions.add(lowered)
                    merged_mentions.append(cleaned)
                st.caption(f"Se enviarán {len(merged_mentions)} entidades (límite interno de 50 por llamada).")
                language_tokens = [token.strip() for token in google_kg_languages_value.split(",") if token.strip()]
                type_tokens = [token.strip() for token in google_kg_types_value.split(",") if token.strip()]
                if st.button(
                    "Consultar Google Knowledge Graph",
                    disabled=not merged_mentions,
                ):
                    try:
                        google_kg_key = ensure_google_kg_api_key(google_kg_api_key_input)
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        with st.spinner("Consultando Enterprise Knowledge Graph..."):
                            google_kg_df = query_google_enterprise_kg(
                                merged_mentions,
                                api_key=google_kg_key,
                                limit=int(google_kg_limit),
                                languages=language_tokens or None,
                                types=type_tokens or None,
                            )
                        st.session_state["google_kg_results"] = google_kg_df
                        if google_kg_df.empty:
                            st.warning("Google KG no devolvió resultados para las consultas enviadas.")
                        else:
                            st.success(
                                f"Se recuperaron {len(google_kg_df)} filas enriquecidas desde Google KG."
                            )
                google_kg_results = st.session_state.get("google_kg_results")
                if isinstance(google_kg_results, pd.DataFrame) and not google_kg_results.empty:
                    st.dataframe(google_kg_results, use_container_width=True)
                    download_dataframe_button(
                        google_kg_results,
                        "google_enterprise_kg.xlsx",
                        "Descargar resultados de Google KG",
                    )
                elif isinstance(google_kg_results, pd.DataFrame):
                    st.info("No hay resultados previos de Google KG para mostrar.")

            doc_relations_summary = st.session_state.get("knowledge_doc_relations")
            if isinstance(doc_relations_summary, pd.DataFrame) and not doc_relations_summary.empty:
                st.markdown("**Cobertura por página y Prominence Score**")
                st.dataframe(doc_relations_summary, use_container_width=True)
                download_dataframe_button(
                    doc_relations_summary,
                    "relaciones_documentos.xlsx",
                    "Descargar cobertura por página",
                )

            spo_summary = st.session_state.get("knowledge_spo_relations")
            if isinstance(spo_summary, pd.DataFrame) and not spo_summary.empty:
                st.markdown("**Tripletes SPO detectados (Sujeto-Predicado-Objeto)**")
                st.dataframe(spo_summary, use_container_width=True)
                download_dataframe_button(
                    spo_summary,
                    "tripletes_spo.xlsx",
                    "Descargar relaciones SPO",
                )






if __name__ == "__main__":
    main()

