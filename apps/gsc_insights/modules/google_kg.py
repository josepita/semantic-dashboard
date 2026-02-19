from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Sequence, Set
from urllib.parse import urlencode

import pandas as pd

from shared.env_utils import get_env_value


def ensure_google_kg_api_key(input_key: Optional[str] = None) -> str:
    candidate = (input_key or "").strip()
    if candidate:
        return candidate
    env_key = get_env_value("GOOGLE_EKG_API_KEY")
    if env_key:
        return env_key
    raise ValueError(
        "No se encontró una API key de Google Enterprise KG. Introduce la clave en la interfaz o define GOOGLE_EKG_API_KEY."
    )


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
        raise ValueError("La API key de Google Enterprise KG está vacía.")

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
            raise RuntimeError(f"Google KG devolvió un {exc.code} para '{query}'.") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"No fue posible contactar con Google KG: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("La respuesta de Google KG no es válida.") from exc

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


__all__ = ["ensure_google_kg_api_key", "query_google_enterprise_kg"]
