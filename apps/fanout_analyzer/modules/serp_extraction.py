"""
Módulo de Extracción SERP.

Extrae títulos de los resultados de búsqueda de Google (SERP)
para cada query usando múltiples métodos:
1. Scraping directo con BeautifulSoup (gratis, limitado)
2. Google Custom Search API (100 queries/día gratis, muy fiable)
"""

from __future__ import annotations

import random
import re
import time
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import quote_plus, unquote

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ══════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════

# User agents actualizados (2024-2025)
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0",
]

# Headers base para las peticiones
BASE_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "es-ES,es;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Sec-CH-UA": '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
    "Sec-CH-UA-Mobile": "?0",
    "Sec-CH-UA-Platform": '"Windows"',
    "Cache-Control": "max-age=0",
}


def get_random_headers() -> Dict[str, str]:
    """Genera headers con User-Agent aleatorio."""
    headers = BASE_HEADERS.copy()
    headers["User-Agent"] = random.choice(USER_AGENTS)
    return headers


# ══════════════════════════════════════════════════════════
# MÉTODO 1: SCRAPING DIRECTO (MEJORADO)
# ══════════════════════════════════════════════════════════

def _extract_url_from_href(href: str) -> str:
    """Extrae la URL real de un href de Google."""
    if not href:
        return ""

    # URL directa
    if href.startswith("http") and "google.com" not in href:
        return href

    # Redirección de Google /url?q=
    if "/url?q=" in href:
        try:
            url = href.split("/url?q=")[1].split("&")[0]
            return unquote(url)
        except:
            pass

    return ""


def _parse_google_results_v1(soup: BeautifulSoup, num_results: int) -> List[Dict]:
    """Patrón 1: div.g con h3 (estructura clásica)."""
    results = []
    position = 1

    for result in soup.select("div.g"):
        if position > num_results:
            break

        # Buscar título en h3
        title_elem = result.select_one("h3")
        if not title_elem:
            continue

        title = title_elem.get_text(strip=True)
        if not title or len(title) < 3:
            continue

        # Buscar URL
        link_elem = result.select_one("a[href]")
        url = _extract_url_from_href(link_elem.get("href", "") if link_elem else "")

        # Buscar descripción (varios selectores posibles)
        description = ""
        for selector in ["div[data-sncf]", "div.VwiC3b", "span.aCOpRe", "div.IsZvec"]:
            desc_elem = result.select_one(selector)
            if desc_elem:
                description = desc_elem.get_text(strip=True)
                break

        results.append({
            "position": position,
            "title": title,
            "url": url,
            "description": description[:300] if description else "",
        })
        position += 1

    return results


def _parse_google_results_v2(soup: BeautifulSoup, num_results: int) -> List[Dict]:
    """Patrón 2: Buscar todos los h3 dentro de enlaces."""
    results = []
    position = 1
    seen_titles = set()

    for h3 in soup.find_all("h3"):
        if position > num_results:
            break

        title = h3.get_text(strip=True)
        if not title or len(title) < 3 or title in seen_titles:
            continue

        # Evitar elementos de navegación de Google
        if any(skip in title.lower() for skip in ["búsquedas relacionadas", "preguntas relacionadas", "más resultados"]):
            continue

        seen_titles.add(title)

        # Buscar enlace padre
        parent_a = h3.find_parent("a")
        url = ""
        if parent_a:
            url = _extract_url_from_href(parent_a.get("href", ""))

        # Buscar descripción en el contenedor padre
        description = ""
        parent_div = h3.find_parent("div", class_=True)
        if parent_div:
            # Buscar el siguiente div con texto
            for sibling in parent_div.find_all("div"):
                text = sibling.get_text(strip=True)
                if len(text) > 50 and text != title:
                    description = text
                    break

        results.append({
            "position": position,
            "title": title,
            "url": url,
            "description": description[:300] if description else "",
        })
        position += 1

    return results


def _parse_google_results_v3(soup: BeautifulSoup, num_results: int) -> List[Dict]:
    """Patrón 3: Buscar por estructura de datos cite + título."""
    results = []
    position = 1
    seen_urls = set()

    # Buscar elementos con cite (URL visible)
    for cite in soup.find_all("cite"):
        if position > num_results:
            break

        # Buscar el contenedor padre
        parent = cite.find_parent("div")
        if not parent:
            continue

        # Buscar h3 en el mismo contenedor o cercano
        container = parent.find_parent("div", class_=True)
        if not container:
            continue

        h3 = container.find("h3")
        if not h3:
            continue

        title = h3.get_text(strip=True)
        if not title or len(title) < 3:
            continue

        # Extraer URL del enlace
        link = container.find("a", href=True)
        url = _extract_url_from_href(link.get("href", "") if link else "")

        if url in seen_urls:
            continue
        seen_urls.add(url)

        results.append({
            "position": position,
            "title": title,
            "url": url,
            "description": "",
        })
        position += 1

    return results


def extract_serp_titles_scraping(
    query: str,
    num_results: int = 10,
    lang: str = "es",
    country: str = "es",
) -> Tuple[List[Dict[str, str]], Optional[str]]:
    """
    Extrae títulos de Google mediante scraping directo.
    Prueba múltiples patrones de parsing para mejor cobertura.
    """
    try:
        encoded_query = quote_plus(query)
        # Pedir más resultados de los necesarios por si filtramos algunos
        request_num = min(num_results + 10, 30)
        url = f"https://www.google.com/search?q={encoded_query}&hl={lang}&gl={country}&num={request_num}"

        headers = get_random_headers()
        response = requests.get(url, headers=headers, timeout=15)

        if response.status_code == 429:
            return [], "Rate limited (429) - Espera unos minutos"
        if response.status_code != 200:
            return [], f"HTTP {response.status_code}"

        soup = BeautifulSoup(response.text, "html.parser")

        # Intentar múltiples patrones de parsing
        results = _parse_google_results_v1(soup, num_results)

        if len(results) < num_results // 2:
            results_v2 = _parse_google_results_v2(soup, num_results)
            if len(results_v2) > len(results):
                results = results_v2

        if len(results) < num_results // 2:
            results_v3 = _parse_google_results_v3(soup, num_results)
            if len(results_v3) > len(results):
                results = results_v3

        # Limitar a num_results
        results = results[:num_results]

        if not results:
            return [], "No se encontraron resultados (posible CAPTCHA o cambio en estructura)"

        return results, None

    except requests.Timeout:
        return [], "Timeout en la petición"
    except requests.RequestException as e:
        return [], f"Error de conexión: {str(e)}"
    except Exception as e:
        return [], f"Error: {str(e)}"


# ══════════════════════════════════════════════════════════
# MÉTODO 2: GOOGLE CUSTOM SEARCH API (RECOMENDADO)
# ══════════════════════════════════════════════════════════

def extract_serp_titles_api(
    query: str,
    api_key: str,
    cx: str,  # Custom Search Engine ID
    num_results: int = 10,
    lang: str = "es",
    country: str = "es",
) -> Tuple[List[Dict[str, str]], Optional[str]]:
    """
    Extrae títulos usando Google Custom Search API.

    Más fiable que scraping. 100 queries/día gratis.

    Para obtener credenciales:
    1. Ve a https://console.cloud.google.com/apis/credentials
    2. Crea una API Key
    3. Ve a https://programmablesearchengine.google.com/
    4. Crea un buscador y obtén el CX (Search Engine ID)
    """
    try:
        results = []

        # La API devuelve máximo 10 resultados por petición
        # Para más, hay que hacer múltiples peticiones con start
        for start in range(1, num_results + 1, 10):
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": api_key,
                "cx": cx,
                "q": query,
                "num": min(10, num_results - len(results)),
                "start": start,
                "lr": f"lang_{lang}",
                "gl": country,
            }

            response = requests.get(url, params=params, timeout=15)

            if response.status_code == 429:
                return results if results else [], "Cuota de API agotada (100/día)"
            if response.status_code == 403:
                return [], "API Key inválida o sin permisos"
            if response.status_code != 200:
                return results if results else [], f"HTTP {response.status_code}"

            data = response.json()

            items = data.get("items", [])
            if not items:
                break

            for item in items:
                position = len(results) + 1
                results.append({
                    "position": position,
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "description": item.get("snippet", "")[:300],
                })

                if len(results) >= num_results:
                    break

            if len(results) >= num_results:
                break

        return results, None

    except requests.Timeout:
        return [], "Timeout en la petición"
    except requests.RequestException as e:
        return [], f"Error de conexión: {str(e)}"
    except Exception as e:
        return [], f"Error: {str(e)}"


# ══════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL DE EXTRACCIÓN
# ══════════════════════════════════════════════════════════

def extract_serp_titles(
    query: str,
    num_results: int = 10,
    lang: str = "es",
    country: str = "es",
    method: str = "scraping",  # "scraping" o "api"
    api_key: Optional[str] = None,
    cx: Optional[str] = None,
) -> Tuple[List[Dict[str, str]], Optional[str]]:
    """
    Extrae títulos de la SERP usando el método especificado.

    Args:
        query: Consulta de búsqueda
        num_results: Número máximo de resultados
        lang: Idioma
        country: País
        method: "scraping" (gratis) o "api" (más fiable, requiere credenciales)
        api_key: API Key de Google (solo para method="api")
        cx: Custom Search Engine ID (solo para method="api")
    """
    if method == "api" and api_key and cx:
        return extract_serp_titles_api(query, api_key, cx, num_results, lang, country)
    else:
        return extract_serp_titles_scraping(query, num_results, lang, country)


def batch_extract_serp(
    queries: List[str],
    num_results: int = 10,
    lang: str = "es",
    country: str = "es",
    delay_min: float = 2.0,
    delay_max: float = 5.0,
    method: str = "scraping",
    api_key: Optional[str] = None,
    cx: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> List[Dict]:
    """
    Extrae títulos SERP para múltiples queries.

    Para scraping: usa delays aleatorios para evitar bloqueos.
    Para API: no necesita delays (pero tiene límite de 100/día gratis).
    """
    all_results = []
    total = len(queries)
    consecutive_errors = 0
    max_consecutive_errors = 5

    for i, query in enumerate(queries):
        if progress_callback:
            progress_callback(i, total, query)

        serp_results, error = extract_serp_titles(
            query=query,
            num_results=num_results,
            lang=lang,
            country=country,
            method=method,
            api_key=api_key,
            cx=cx,
        )

        # Guardar resultados
        for result in serp_results:
            all_results.append({
                "query": query,
                "query_index": i,
                "position": result["position"],
                "title": result["title"],
                "url": result["url"],
                "description": result["description"],
                "error": "",
            })

        # Si hubo error, registrarlo
        if error:
            all_results.append({
                "query": query,
                "query_index": i,
                "position": 0,
                "title": "",
                "url": "",
                "description": "",
                "error": error,
            })
            consecutive_errors += 1

            # Si hay muchos errores consecutivos, puede ser bloqueo
            if consecutive_errors >= max_consecutive_errors and method == "scraping":
                all_results.append({
                    "query": "[DETENIDO]",
                    "query_index": i,
                    "position": 0,
                    "title": "",
                    "url": "",
                    "description": "",
                    "error": f"Demasiados errores consecutivos ({consecutive_errors}). Posible bloqueo de Google.",
                })
                break
        else:
            consecutive_errors = 0

        # Delay entre peticiones (solo para scraping)
        if i < total - 1 and method == "scraping":
            delay = random.uniform(delay_min, delay_max)
            time.sleep(delay)
        elif i < total - 1 and method == "api":
            # Pequeño delay para API también
            time.sleep(0.2)

    return all_results


def serp_results_to_dataframe(results: List[Dict]) -> pd.DataFrame:
    """Convierte los resultados de SERP a DataFrame."""
    if not results:
        return pd.DataFrame(columns=[
            "query", "query_index", "position", "title", "url", "description", "error"
        ])
    return pd.DataFrame(results)


def merge_serp_with_fanout(
    fanout_df: pd.DataFrame,
    serp_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combina los resultados de SERP con el DataFrame de fan-out queries.
    """
    if fanout_df.empty or serp_df.empty:
        return fanout_df

    # Agrupar títulos SERP por query
    serp_grouped = serp_df[serp_df["error"] == ""].groupby("query").agg({
        "title": lambda x: " | ".join(x.tolist()),
        "url": lambda x: " | ".join([u for u in x.tolist() if u]),
    }).reset_index()

    serp_grouped.columns = ["web_search_query", "serp_titles", "serp_urls"]

    # Merge con fan-out
    merged = fanout_df.merge(serp_grouped, on="web_search_query", how="left")
    merged["serp_titles"] = merged["serp_titles"].fillna("")
    merged["serp_urls"] = merged["serp_urls"].fillna("")

    return merged


def get_serp_summary(serp_df: pd.DataFrame) -> Dict:
    """Genera estadísticas resumen de la extracción SERP."""
    if serp_df.empty:
        return {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_results": 0,
            "avg_results_per_query": 0,
        }

    errors = serp_df[serp_df["error"] != ""]
    success = serp_df[serp_df["error"] == ""]

    unique_queries = serp_df["query"].nunique()
    failed_queries = errors["query"].nunique()
    successful_queries = unique_queries - failed_queries

    return {
        "total_queries": unique_queries,
        "successful_queries": successful_queries,
        "failed_queries": failed_queries,
        "total_results": len(success),
        "avg_results_per_query": round(len(success) / successful_queries, 1) if successful_queries > 0 else 0,
    }
