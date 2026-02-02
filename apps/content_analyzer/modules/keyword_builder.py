"""
Semantic Keyword Builder Module

Este módulo maneja la funcionalidad de generación de universos semánticos de keywords
usando Gemini AI con anotaciones EAV (Entity-Attribute-Variable).
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence, Tuple

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import google.generativeai as genai
except ModuleNotFoundError:
    genai = None

# Ensure project root is in sys.path for shared.* imports
_project_root = str(Path(__file__).resolve().parents[3])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from shared.ui_components import bordered_container  # noqa: E402
    from shared.gemini_utils import (  # noqa: E402
        get_gemini_api_key as get_gemini_api_key_from_context,
    )
    from shared.gemini_utils import (  # noqa: E402
        get_gemini_model as get_gemini_model_from_context,
    )
except ModuleNotFoundError:
    _shared_path = Path(__file__).resolve().parents[3] / "shared"
    if str(_shared_path) not in sys.path:
        sys.path.insert(0, str(_shared_path))
    from ui_components import bordered_container  # noqa: E402
    from gemini_utils import (  # noqa: E402
        get_gemini_api_key as get_gemini_api_key_from_context,
    )
    from gemini_utils import (  # noqa: E402
        get_gemini_model as get_gemini_model_from_context,
    )


def build_semantic_keyword_prompt(
    country: str,
    language: str,
    website: str,
    niche: str,
    approx_keywords: int,
    seed_keywords: List[str],
) -> str:
    seeds_block = "\\n".join(f"- {seed}" for seed in seed_keywords) if seed_keywords else "none"
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
    st.subheader("🧠 Semantic Keyword Builder")
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
                value="seo tecnico\\ncluster semantico\\ninterlinking",
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


__all__ = [
    "build_semantic_keyword_prompt",
    "generate_semantic_keyword_universe",
    "group_keywords_with_semantic_builder",
    "render_semantic_keyword_builder",
]
