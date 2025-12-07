"""
Análisis semántico sencillo para un texto frente a un listado de keywords.

Utiliza el mismo modelo y librerías que el dashboard principal:
- sentence-transformers para generar embeddings
- sklearn.metrics.pairwise.cosine_similarity para calcular la similitud
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


@dataclass
class KeywordScore:
    keyword: str
    similarity: float

    @property
    def percent(self) -> float:
        return self.similarity * 100


def normalize_keywords(keywords: Sequence[str]) -> List[str]:
    return [kw.strip() for kw in keywords if kw and kw.strip()]


def compute_text_keyword_similarity(text: str, keywords: Sequence[str]) -> List[KeywordScore]:
    cleaned_text = text.strip()
    cleaned_keywords = normalize_keywords(keywords)

    if not cleaned_text:
        raise ValueError("El texto principal está vacío.")
    if not cleaned_keywords:
        raise ValueError("No se proporcionaron keywords válidas.")

    model = SentenceTransformer(MODEL_NAME)
    text_embedding = model.encode([cleaned_text], convert_to_numpy=True)
    keyword_embeddings = model.encode(cleaned_keywords, convert_to_numpy=True)
    similarities = cosine_similarity(text_embedding, keyword_embeddings)[0]

    scores = [
        KeywordScore(keyword=kw, similarity=float(score))
        for kw, score in zip(cleaned_keywords, similarities)
    ]
    scores.sort(key=lambda item: item.similarity, reverse=True)
    return scores


def format_scores(scores: Sequence[KeywordScore]) -> List[Tuple[str, str]]:
    return [(item.keyword, f"{item.similarity:.4f} ({item.percent:.2f}%)") for item in scores]


if __name__ == "__main__":
    SAMPLE_TEXT = """
    Somos especialistas en cursos online de analítica de datos. Enseñamos a crear dashboards interactivos,
    modelos de machine learning y automatizaciones con Python para equipos de marketing y ventas.
    """
    SAMPLE_KEYWORDS = [
        "curso de embeddings",
        "curso de ciencia de datos",
        "plataforma de dashboards",
        "automatizaciones marketing",
    ]

    print("=== Análisis semántico: texto vs keywords ===\n")
    try:
        results = compute_text_keyword_similarity(SAMPLE_TEXT, SAMPLE_KEYWORDS)
    except ValueError as err:
        print(f"[Error] {err}")
    else:
        print(f"Texto analizado (primeros 120 caracteres): {SAMPLE_TEXT.strip()[:120]}...")
        print("\nResultados ordenados por similitud:\n")
        for keyword, score_repr in format_scores(results):
            print(f"- {keyword}: {score_repr}")
        print("\nFin del análisis.")
