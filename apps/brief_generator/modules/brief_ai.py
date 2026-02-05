"""
Brief AI Module
===============
Generación de propuestas SEO con Gemini/OpenAI.
- Títulos SEO optimizados
- Meta descriptions
- Estructura de encabezados (HNs)
"""

import json
import re
from typing import Dict, List, Optional, Tuple

import streamlit as st

# ══════════════════════════════════════════════════════════
# AI PROVIDERS
# ══════════════════════════════════════════════════════════

try:
    import google.generativeai as genai
except ModuleNotFoundError:
    genai = None

try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None

PROVIDER_GEMINI = "gemini"
PROVIDER_OPENAI = "openai"
SS = "bg_"  # Session state prefix


# ══════════════════════════════════════════════════════════
# LLM CONFIGURATION
# ══════════════════════════════════════════════════════════

def is_llm_configured() -> bool:
    """Verifica si hay algún proveedor de IA configurado."""
    return bool(
        st.session_state.get("openai_api_key") or
        st.session_state.get("gemini_api_key")
    )


def get_configured_provider() -> Optional[str]:
    """Obtiene el proveedor configurado."""
    provider = st.session_state.get(f"{SS}llm_provider")
    if provider:
        return provider
    # Auto-detect
    if st.session_state.get("gemini_api_key"):
        return PROVIDER_GEMINI
    if st.session_state.get("openai_api_key"):
        return PROVIDER_OPENAI
    return None


def call_llm(prompt: str) -> str:
    """
    Llama al LLM configurado (Gemini o OpenAI).

    Args:
        prompt: El prompt a enviar

    Returns:
        Respuesta del modelo

    Raises:
        ValueError: Si no hay API configurada
        Exception: En caso de error de la API
    """
    provider = get_configured_provider()

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

    else:  # Gemini (default)
        api_key = st.session_state.get("gemini_api_key", "")
        model_name = st.session_state.get("gemini_model_name", "gemini-2.5-flash")

        if not api_key:
            raise ValueError("API key de Gemini no configurada")
        if genai is None:
            raise ImportError("google-generativeai no instalado")

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


def render_llm_config_sidebar() -> None:
    """Renderiza el selector de proveedor de IA en el sidebar."""
    st.markdown("#### Proveedor de IA")

    has_openai = bool(st.session_state.get("openai_api_key"))
    has_gemini = bool(st.session_state.get("gemini_api_key"))

    if not has_openai and not has_gemini:
        st.warning("Configura una API key en la app principal (Configuracion API)")
        return

    available = []
    if has_gemini:
        available.append(PROVIDER_GEMINI)
    if has_openai:
        available.append(PROVIDER_OPENAI)

    provider = st.radio(
        "Usar",
        options=available,
        format_func=lambda x: "Google Gemini" if x == PROVIDER_GEMINI else "OpenAI GPT",
        key=f"{SS}llm_provider",
        horizontal=True,
    )

    if provider == PROVIDER_OPENAI:
        model = st.session_state.get("openai_model", "gpt-4o-mini")
        st.caption(f"Modelo: **{model}**")
    else:
        model = st.session_state.get("gemini_model_name", "gemini-2.5-flash")
        st.caption(f"Modelo: **{model}**")


# ══════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════

def _build_title_prompt(
    keyword: str,
    serp_titles: List[str],
    country: str = "ES"
) -> str:
    """Construye el prompt para generar títulos SEO."""

    competitors = "\n".join([f"- {t}" for t in serp_titles[:5]])

    return f"""Eres un experto en SEO. Genera 5 propuestas de títulos SEO para un artículo sobre "{keyword}" orientado a {country}.

COMPETENCIA ACTUAL (Top 5 Google):
{competitors}

REQUISITOS:
1. Máximo 60 caracteres por título
2. Incluir la keyword principal de forma natural
3. Ser únicos y diferenciados de la competencia
4. Usar ganchos emocionales o números cuando sea apropiado
5. Optimizados para CTR

Responde SOLO con un JSON array de 5 strings, sin explicaciones:
["título 1", "título 2", "título 3", "título 4", "título 5"]
"""


def _build_meta_prompt(
    keyword: str,
    selected_title: str,
    serp_snippets: List[str],
    country: str = "ES"
) -> str:
    """Construye el prompt para generar meta descriptions."""

    competitors = "\n".join([f"- {s[:150]}" for s in serp_snippets[:3]])

    return f"""Eres un experto en SEO. Genera 5 propuestas de meta description para un artículo con:
- Keyword: "{keyword}"
- Título: "{selected_title}"
- País: {country}

SNIPPETS DE COMPETENCIA:
{competitors}

REQUISITOS:
1. Máximo 155 caracteres por meta description
2. Incluir la keyword de forma natural
3. Incluir una llamada a la acción
4. Ser persuasivas y generar curiosidad
5. Diferenciarse de la competencia

Responde SOLO con un JSON array de 5 strings, sin explicaciones:
["meta 1", "meta 2", "meta 3", "meta 4", "meta 5"]
"""


def _build_hn_prompt(
    keyword: str,
    secondary_keywords: List[str],
    serp_titles: List[str],
    questions: List[str]
) -> str:
    """Construye el prompt para generar estructura de encabezados."""

    secondary_kws = ", ".join(secondary_keywords[:10]) if secondary_keywords else "N/A"
    competitors = "\n".join([f"- {t}" for t in serp_titles[:5]])
    paa = "\n".join([f"- {q}" for q in questions[:5]]) if questions else "N/A"

    return f"""Eres un arquitecto de contenido SEO. Genera una estructura de encabezados (H1, H2, H3) para un artículo sobre "{keyword}".

KEYWORDS SECUNDARIAS: {secondary_kws}

COMPETENCIA (títulos actuales):
{competitors}

PREGUNTAS FRECUENTES (PAA):
{paa}

REQUISITOS:
1. Un único H1 que contenga la keyword principal
2. Entre 4-7 secciones H2 que cubran el tema completo
3. Cada H2 puede tener 0-3 subsecciones H3
4. Incluir keywords secundarias de forma natural
5. Responder a las preguntas frecuentes en la estructura
6. Flujo lógico de información

Responde SOLO con JSON en este formato exacto:
{{
  "h1": "Título H1 aquí",
  "sections": [
    {{
      "h2": "Título H2",
      "h3": ["H3 opcional 1", "H3 opcional 2"]
    }}
  ]
}}
"""


# ══════════════════════════════════════════════════════════
# PARSING HELPERS
# ══════════════════════════════════════════════════════════

def _parse_json_response(text: str) -> any:
    """Parsea respuesta JSON del LLM, manejando formatos comunes."""
    # Limpiar markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Intentar extraer JSON del texto
        json_match = re.search(r'[\[{].*[\]}]', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

    return None


def _extract_list_from_text(text: str) -> List[str]:
    """Extrae una lista de items del texto si el JSON falla."""
    lines = text.strip().split('\n')
    items = []
    for line in lines:
        line = line.strip()
        # Quitar bullets, números, etc.
        line = re.sub(r'^[\d\.\-\*\•]+\s*', '', line)
        line = re.sub(r'^["\']+|["\']+$', '', line)
        if line and len(line) > 5:
            items.append(line)
    return items[:5]


# ══════════════════════════════════════════════════════════
# MAIN GENERATION FUNCTIONS
# ══════════════════════════════════════════════════════════

def generate_title_proposals(
    keyword: str,
    serp_data: Dict,
    country: str = "ES"
) -> Tuple[List[str], Optional[str]]:
    """
    Genera propuestas de títulos SEO.

    Args:
        keyword: Keyword principal
        serp_data: Datos de SERP (dict con organic_results)
        country: Código de país

    Returns:
        Tuple de (lista de títulos, error message o None)
    """
    if not is_llm_configured():
        return [], "No hay API de IA configurada"

    # Extraer títulos de SERP
    serp_titles = [
        r.get('title', '')
        for r in serp_data.get('organic_results', [])
        if r.get('title')
    ]

    if not serp_titles:
        serp_titles = ["Sin datos de competencia"]

    prompt = _build_title_prompt(keyword, serp_titles, country)

    try:
        response = call_llm(prompt)
        titles = _parse_json_response(response)

        if isinstance(titles, list):
            return titles[:5], None
        else:
            # Fallback: extraer del texto
            titles = _extract_list_from_text(response)
            return titles, None

    except Exception as e:
        return [], str(e)


def generate_meta_proposals(
    keyword: str,
    selected_title: str,
    serp_data: Dict,
    country: str = "ES"
) -> Tuple[List[str], Optional[str]]:
    """
    Genera propuestas de meta descriptions.

    Args:
        keyword: Keyword principal
        selected_title: Título seleccionado
        serp_data: Datos de SERP
        country: Código de país

    Returns:
        Tuple de (lista de metas, error message o None)
    """
    if not is_llm_configured():
        return [], "No hay API de IA configurada"

    # Extraer snippets de SERP
    serp_snippets = [
        r.get('snippet', '')
        for r in serp_data.get('organic_results', [])
        if r.get('snippet')
    ]

    if not serp_snippets:
        serp_snippets = ["Sin datos de competencia"]

    prompt = _build_meta_prompt(keyword, selected_title, serp_snippets, country)

    try:
        response = call_llm(prompt)
        metas = _parse_json_response(response)

        if isinstance(metas, list):
            return metas[:5], None
        else:
            metas = _extract_list_from_text(response)
            return metas, None

    except Exception as e:
        return [], str(e)


def generate_hn_structure(
    keyword: str,
    serp_data: Dict,
    secondary_keywords: Optional[List[str]] = None
) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Genera estructura de encabezados (H1/H2/H3).

    Args:
        keyword: Keyword principal
        serp_data: Datos de SERP
        secondary_keywords: Keywords secundarias opcionales

    Returns:
        Tuple de (estructura dict, error message o None)
    """
    if not is_llm_configured():
        return None, "No hay API de IA configurada"

    # Extraer datos de SERP
    serp_titles = [
        r.get('title', '')
        for r in serp_data.get('organic_results', [])
        if r.get('title')
    ]
    questions = serp_data.get('people_also_ask', [])

    if not secondary_keywords:
        secondary_keywords = serp_data.get('related_searches', [])

    prompt = _build_hn_prompt(keyword, secondary_keywords, serp_titles, questions)

    try:
        response = call_llm(prompt)
        structure = _parse_json_response(response)

        if isinstance(structure, dict) and 'h1' in structure:
            return structure, None
        else:
            return None, "No se pudo parsear la estructura de encabezados"

    except Exception as e:
        return None, str(e)


def generate_full_brief(
    keyword: str,
    serp_data: Dict,
    country: str = "ES",
    secondary_keywords: Optional[List[str]] = None
) -> Dict:
    """
    Genera un brief completo: títulos, metas y estructura HN.

    Args:
        keyword: Keyword principal
        serp_data: Datos de SERP
        country: Código de país
        secondary_keywords: Keywords secundarias

    Returns:
        Dict con todas las propuestas generadas
    """
    result = {
        'titles': [],
        'metas': [],
        'hn_structure': None,
        'errors': []
    }

    # Generar títulos
    titles, error = generate_title_proposals(keyword, serp_data, country)
    result['titles'] = titles
    if error:
        result['errors'].append(f"Títulos: {error}")

    # Generar metas (usando el primer título si hay)
    selected_title = titles[0] if titles else keyword
    metas, error = generate_meta_proposals(keyword, selected_title, serp_data, country)
    result['metas'] = metas
    if error:
        result['errors'].append(f"Metas: {error}")

    # Generar estructura HN
    hn_structure, error = generate_hn_structure(keyword, serp_data, secondary_keywords)
    result['hn_structure'] = hn_structure
    if error:
        result['errors'].append(f"Estructura: {error}")

    return result
