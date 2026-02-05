"""
SERP Scraper Module
===================
Scraping de resultados de Google usando BeautifulSoup.
"""

import random
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib.parse import quote_plus, urlparse

import requests
from bs4 import BeautifulSoup

# ══════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
]

# Dominios de Google por país
GOOGLE_DOMAINS = {
    "ES": "google.es",
    "MX": "google.com.mx",
    "AR": "google.com.ar",
    "CO": "google.com.co",
    "CL": "google.cl",
    "PE": "google.com.pe",
    "US": "google.com",
    "UK": "google.co.uk",
}

# Parámetros de idioma por país
LANG_PARAMS = {
    "ES": {"hl": "es", "gl": "es"},
    "MX": {"hl": "es", "gl": "mx"},
    "AR": {"hl": "es", "gl": "ar"},
    "CO": {"hl": "es", "gl": "co"},
    "CL": {"hl": "es", "gl": "cl"},
    "PE": {"hl": "es", "gl": "pe"},
    "US": {"hl": "en", "gl": "us"},
    "UK": {"hl": "en", "gl": "uk"},
}

COUNTRY_FLAGS = {
    "ES": "🇪🇸",
    "MX": "🇲🇽",
    "AR": "🇦🇷",
    "CO": "🇨🇴",
    "CL": "🇨🇱",
    "PE": "🇵🇪",
    "US": "🇺🇸",
    "UK": "🇬🇧",
}


# ══════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════

@dataclass
class SerpResult:
    """Resultado individual de SERP."""
    position: int
    title: str
    url: str
    domain: str
    snippet: str


@dataclass
class SerpData:
    """Datos completos de SERP."""
    keyword: str
    country: str
    organic_results: List[SerpResult]
    people_also_ask: List[str]
    related_searches: List[str]
    total_results: Optional[str] = None
    scraped_at: Optional[str] = None


# ══════════════════════════════════════════════════════════
# SCRAPER FUNCTIONS
# ══════════════════════════════════════════════════════════

def _get_random_headers() -> Dict[str, str]:
    """Genera headers aleatorios para la request."""
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }


def _build_google_url(keyword: str, country: str = "ES", num_results: int = 10) -> str:
    """Construye la URL de búsqueda de Google."""
    domain = GOOGLE_DOMAINS.get(country, "google.com")
    params = LANG_PARAMS.get(country, {"hl": "es", "gl": "es"})

    encoded_kw = quote_plus(keyword)
    url = f"https://www.{domain}/search?q={encoded_kw}&num={num_results}"

    for key, value in params.items():
        url += f"&{key}={value}"

    return url


def _extract_domain(url: str) -> str:
    """Extrae el dominio de una URL."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        # Quitar www.
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""


def _clean_text(text: str) -> str:
    """Limpia texto de caracteres extra."""
    if not text:
        return ""
    # Normalizar espacios
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _parse_organic_results(soup: BeautifulSoup) -> List[SerpResult]:
    """Parsea los resultados orgánicos del HTML."""
    results = []
    position = 0

    # Selectores para resultados orgánicos de Google
    # Google cambia frecuentemente su HTML, estos son los más comunes
    result_selectors = [
        'div.g',  # Selector clásico
        'div[data-hveid]',  # Selector por atributo
    ]

    result_divs = []
    for selector in result_selectors:
        result_divs = soup.select(selector)
        if result_divs:
            break

    for div in result_divs:
        try:
            # Buscar el enlace principal
            link = div.select_one('a[href^="http"]')
            if not link:
                continue

            url = link.get('href', '')

            # Filtrar URLs de Google (imágenes, videos, etc.)
            if 'google.com' in url or not url.startswith('http'):
                continue

            # Buscar el título
            title_elem = div.select_one('h3')
            if not title_elem:
                continue

            title = _clean_text(title_elem.get_text())
            if not title:
                continue

            # Buscar el snippet
            snippet = ""
            snippet_selectors = [
                'div[data-sncf]',
                'div.VwiC3b',
                'span.aCOpRe',
                'div.IsZvec',
            ]
            for sel in snippet_selectors:
                snippet_elem = div.select_one(sel)
                if snippet_elem:
                    snippet = _clean_text(snippet_elem.get_text())
                    break

            # Extraer dominio
            domain = _extract_domain(url)
            if not domain:
                continue

            position += 1
            results.append(SerpResult(
                position=position,
                title=title,
                url=url,
                domain=domain,
                snippet=snippet
            ))

            # Limitar a top 10
            if position >= 10:
                break

        except Exception:
            continue

    return results


def _parse_people_also_ask(soup: BeautifulSoup) -> List[str]:
    """Parsea la sección 'People Also Ask' (Preguntas relacionadas)."""
    questions = []

    # Selectores para PAA
    paa_selectors = [
        'div[data-sgrd] span',
        'div.related-question-pair span',
        'div[jsname="N760b"] span',
    ]

    for selector in paa_selectors:
        elements = soup.select(selector)
        for elem in elements:
            question = _clean_text(elem.get_text())
            if question and '?' in question and question not in questions:
                questions.append(question)

    return questions[:10]  # Limitar a 10


def _parse_related_searches(soup: BeautifulSoup) -> List[str]:
    """Parsea las búsquedas relacionadas."""
    searches = []

    # Selectores para búsquedas relacionadas
    related_selectors = [
        'div.s75CSd a',  # Selector común
        'a.k8XOCe',  # Otro selector
        'div[data-ved] a.related-searches',
    ]

    for selector in related_selectors:
        elements = soup.select(selector)
        for elem in elements:
            search = _clean_text(elem.get_text())
            if search and len(search) > 2 and search not in searches:
                searches.append(search)

    # También buscar en el footer
    footer_links = soup.select('div#botstuff a')
    for link in footer_links:
        text = _clean_text(link.get_text())
        if text and len(text) > 2 and text not in searches and 'google' not in text.lower():
            searches.append(text)

    return searches[:15]  # Limitar a 15


def _parse_total_results(soup: BeautifulSoup) -> Optional[str]:
    """Parsea el número total de resultados."""
    try:
        result_stats = soup.select_one('#result-stats')
        if result_stats:
            text = result_stats.get_text()
            # Extraer número
            match = re.search(r'[\d.,]+', text)
            if match:
                return match.group()
    except Exception:
        pass
    return None


def scrape_serp(
    keyword: str,
    country: str = "ES",
    num_results: int = 10,
    delay: float = 1.0
) -> Optional[SerpData]:
    """
    Scrapea los resultados de Google para una keyword.

    Args:
        keyword: Término de búsqueda
        country: Código de país (ES, MX, US, etc.)
        num_results: Número de resultados a obtener
        delay: Delay antes de la request (para rate limiting)

    Returns:
        SerpData con los resultados o None si hay error
    """
    # Rate limiting
    if delay > 0:
        time.sleep(delay)

    url = _build_google_url(keyword, country, num_results)
    headers = _get_random_headers()

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        # Verificar si nos bloquearon (CAPTCHA)
        if 'captcha' in response.text.lower() or response.status_code == 429:
            return None

        soup = BeautifulSoup(response.text, 'html.parser')

        # Parsear datos
        organic_results = _parse_organic_results(soup)
        people_also_ask = _parse_people_also_ask(soup)
        related_searches = _parse_related_searches(soup)
        total_results = _parse_total_results(soup)

        from datetime import datetime
        return SerpData(
            keyword=keyword,
            country=country,
            organic_results=organic_results,
            people_also_ask=people_also_ask,
            related_searches=related_searches,
            total_results=total_results,
            scraped_at=datetime.now().isoformat()
        )

    except requests.RequestException:
        return None
    except Exception:
        return None


def serp_to_dict(serp_data: SerpData) -> Dict:
    """Convierte SerpData a diccionario para almacenar en JSON."""
    return {
        "keyword": serp_data.keyword,
        "country": serp_data.country,
        "total_results": serp_data.total_results,
        "scraped_at": serp_data.scraped_at,
        "organic_results": [
            {
                "position": r.position,
                "title": r.title,
                "url": r.url,
                "domain": r.domain,
                "snippet": r.snippet
            }
            for r in serp_data.organic_results
        ],
        "people_also_ask": serp_data.people_also_ask,
        "related_searches": serp_data.related_searches
    }


def dict_to_serp(data: Dict) -> SerpData:
    """Convierte diccionario a SerpData."""
    organic_results = [
        SerpResult(**r) for r in data.get("organic_results", [])
    ]
    return SerpData(
        keyword=data.get("keyword", ""),
        country=data.get("country", "ES"),
        organic_results=organic_results,
        people_also_ask=data.get("people_also_ask", []),
        related_searches=data.get("related_searches", []),
        total_results=data.get("total_results"),
        scraped_at=data.get("scraped_at")
    )


# ══════════════════════════════════════════════════════════
# GOOGLE SUGGEST (Autocomplete)
# ══════════════════════════════════════════════════════════

def get_google_suggestions(keyword: str, country: str = "ES") -> List[str]:
    """
    Obtiene sugerencias de Google Autocomplete.

    Args:
        keyword: Término base
        country: Código de país

    Returns:
        Lista de sugerencias
    """
    params = LANG_PARAMS.get(country, {"hl": "es", "gl": "es"})
    url = f"https://suggestqueries.google.com/complete/search?client=firefox&q={quote_plus(keyword)}&hl={params['hl']}"

    try:
        response = requests.get(url, headers=_get_random_headers(), timeout=10)
        response.raise_for_status()

        data = response.json()
        if isinstance(data, list) and len(data) > 1:
            return data[1][:10]

    except Exception:
        pass

    return []


def get_keyword_variations(keyword: str, country: str = "ES") -> List[str]:
    """
    Genera variaciones de una keyword usando diferentes prefijos/sufijos.

    Args:
        keyword: Keyword base
        country: Código de país

    Returns:
        Lista de variaciones encontradas
    """
    variations = set()

    # Prefijos comunes
    prefixes = ["qué es", "cómo", "por qué", "cuánto", "dónde", "cuál", "mejor", ""]
    suffixes = ["", " precio", " opiniones", " 2024", " gratis", " online"]

    for prefix in prefixes:
        query = f"{prefix} {keyword}".strip()
        suggestions = get_google_suggestions(query, country)
        variations.update(suggestions)
        time.sleep(0.3)  # Rate limiting

    for suffix in suffixes:
        query = f"{keyword}{suffix}"
        suggestions = get_google_suggestions(query, country)
        variations.update(suggestions)
        time.sleep(0.3)

    # Quitar la keyword original
    variations.discard(keyword)

    return list(variations)[:30]
