"""
Extracción de fan-out queries usando la API de Gemini con Google Search grounding.

Utiliza el SDK `google-genai` para enviar prompts a Gemini y extraer las
consultas de búsqueda web (webSearchQueries) que el modelo ejecuta internamente.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import pandas as pd


@dataclass
class FanOutResult:
    """Resultado de extracción de fan-out queries para un prompt."""
    prompt: str
    queries: List[str] = field(default_factory=list)
    error: str = ""


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def extract_fanout_queries(
    prompts: List[str],
    api_key: str,
    model_id: str = "gemini-2.5-flash",
    delay: float = 0.5,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> List[FanOutResult]:
    """Extrae web_search_queries de Gemini con Google Search grounding.

    Args:
        prompts: Lista de prompts a procesar.
        api_key: Clave API de Gemini.
        model_id: ID del modelo Gemini a usar.
        delay: Segundos de espera entre requests (rate limiting).
        progress_callback: Función (current, total, prompt) para reportar progreso.

    Returns:
        Lista de FanOutResult con las queries extraídas por prompt.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return [FanOutResult(
            prompt=p,
            error="Instala google-genai: pip install google-genai"
        ) for p in prompts]

    client = genai.Client(api_key=api_key)
    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(tools=[grounding_tool])

    results: List[FanOutResult] = []

    for idx, prompt in enumerate(prompts):
        prompt = prompt.strip()
        if not prompt:
            continue

        if progress_callback:
            progress_callback(idx, len(prompts), prompt)

        try:
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=config,
            )

            queries: List[str] = []
            for candidate in (response.candidates or []):
                gm = getattr(candidate, "grounding_metadata", None)
                if not gm:
                    continue
                # SDK Python expone snake_case: web_search_queries
                wsq = getattr(gm, "web_search_queries", None)
                if wsq:
                    queries.extend([q for q in wsq if q])
                # Fallback defensivo (camelCase)
                wsq2 = getattr(gm, "webSearchQueries", None)
                if wsq2:
                    queries.extend([q for q in wsq2 if q])

            results.append(FanOutResult(
                prompt=prompt,
                queries=_dedupe_preserve_order(queries),
            ))

        except Exception as e:
            results.append(FanOutResult(prompt=prompt, error=str(e)))

        if delay > 0 and idx < len(prompts) - 1:
            time.sleep(delay)

    return results


def fanout_results_to_dataframe(results: List[FanOutResult]) -> pd.DataFrame:
    """Convierte resultados de fan-out a DataFrame normalizado.

    Returns:
        DataFrame con columnas: prompt, query_index, web_search_query, source, error
    """
    rows = []
    for r in results:
        if r.error:
            rows.append({
                "prompt": r.prompt,
                "query_index": 0,
                "web_search_query": "",
                "source": "gemini",
                "error": r.error,
            })
        elif not r.queries:
            rows.append({
                "prompt": r.prompt,
                "query_index": 0,
                "web_search_query": "",
                "source": "gemini",
                "error": "No se obtuvieron queries",
            })
        else:
            for i, q in enumerate(r.queries):
                rows.append({
                    "prompt": r.prompt,
                    "query_index": i,
                    "web_search_query": q,
                    "source": "gemini",
                    "error": "",
                })
    return pd.DataFrame(rows)


__all__ = [
    "FanOutResult",
    "extract_fanout_queries",
    "fanout_results_to_dataframe",
]
