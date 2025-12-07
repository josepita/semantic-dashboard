"""
Análisis semántico de URLs de competidores frente a un conjunto de consultas.

Comparte el mismo stack que el dashboard principal:
- sentence-transformers (SentenceTransformer)
- sklearn.metrics.pairwise.cosine_similarity
- Modelo paraphrase-multilingual-MiniLM-L12-v2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import pandas as pd
import trafilatura
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
MAX_CONTENT_CHARS = 8000


@dataclass
class CompetitorResult:
    query: str
    url: str
    score: float

    @property
    def percent(self) -> float:
        return self.score * 100


def fetch_url_content(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return ""
    extracted = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
    )
    if not extracted:
        return ""
    return extracted.strip()[:MAX_CONTENT_CHARS]


def normalize_strings(values: Iterable[str]) -> List[str]:
    return [item.strip() for item in values if item and item.strip()]


def analyze_competitors(urls: Sequence[str], queries: Sequence[str]) -> List[CompetitorResult]:
    cleaned_urls = normalize_strings(urls)
    cleaned_queries = normalize_strings(queries)

    if not cleaned_urls:
        raise ValueError("No se proporcionaron URLs válidas.")
    if not cleaned_queries:
        raise ValueError("No se proporcionaron queries válidas.")

    model = SentenceTransformer(MODEL_NAME)

    contents: Dict[str, str] = {}
    for url in cleaned_urls:
        content = fetch_url_content(url)
        if content:
            contents[url] = content

    if not contents:
        raise RuntimeError("No se pudo extraer contenido útil de las URLs.")

    ordered_urls = list(contents.keys())
    url_embeddings = model.encode([contents[url] for url in ordered_urls], convert_to_numpy=True)
    query_embeddings = model.encode(cleaned_queries, convert_to_numpy=True)
    similarity_matrix = cosine_similarity(query_embeddings, url_embeddings)

    results: List[CompetitorResult] = []
    for query_idx, query in enumerate(cleaned_queries):
        for url_idx, url in enumerate(ordered_urls):
            results.append(
                CompetitorResult(
                    query=query,
                    url=url,
                    score=float(similarity_matrix[query_idx, url_idx]),
                )
            )

    results.sort(key=lambda item: (item.query, -item.score))
    return results


def results_to_dataframe(results: Sequence[CompetitorResult]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame(columns=["Query", "URL", "Score", "Score (%)"])
    return pd.DataFrame(
        {
            "Query": [item.query for item in results],
            "URL": [item.url for item in results],
            "Score": [item.score for item in results],
            "Score (%)": [item.percent for item in results],
        }
    )


def top_n_by_query(results: Sequence[CompetitorResult], n: int) -> List[CompetitorResult]:
    top_results: List[CompetitorResult] = []
    current_query = None
    count = 0

    for item in results:
        if item.query != current_query:
            current_query = item.query
            count = 0
        if count < n:
            top_results.append(item)
            count += 1
    return top_results


if __name__ == "__main__":
    SAMPLE_URLS = [
        "https://parkos.es/parking-aeropuerto-madrid-barajas/",
        "https://www.reservarparking.com/",
        "https://www.myparking.es/parking_aeropuerto_madrid.php",
        "https://parclick.es/parking/parkings-en-el-aeropuerto-de-madrid-barajas",
    ]
    SAMPLE_QUERIES = [
        "parking aeropuerto madrid",
        "parking barajas",
        "reservar parking aeropuerto madrid",
        "parking aeropuerto madrid barato",
    ]

    print("=== Análisis semántico de competidores ===\n")
    try:
        competitor_results = analyze_competitors(SAMPLE_URLS, SAMPLE_QUERIES)
    except (ValueError, RuntimeError) as err:
        print(f"[Error] {err}")
    else:
        df = results_to_dataframe(competitor_results)
        print("Resultados completos (primeras filas):")
        print(df.head(20).to_string(index=False))

        print("\nTop 3 resultados por query:")
        for item in top_n_by_query(competitor_results, n=3):
            print(f"- {item.query} -> {item.url} | Score: {item.score:.4f} ({item.percent:.2f}%)")

        print("\nFin del análisis.")
