from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class ClusterSummary:
    cluster_id: int
    documents: List[str]
    similarity: float
    is_gap: bool = False


@dataclass
class AuthorityGapResult:
    clusters: List[ClusterSummary]
    gaps: List[ClusterSummary]
    gap_threshold: float
    competitor_documents: List[str]
    site_documents: List[str]
    metadata: Dict[str, float]


@dataclass
class SemanticPageRankResult:
    url: str
    pagerank_score: float
    semantic_boost: float
    incoming_links: int


def _ensure_2d(vector: np.ndarray) -> np.ndarray:
    return vector.reshape(1, -1) if vector.ndim == 1 else vector


def calcular_similitud_coseno(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    return float(cosine_similarity(_ensure_2d(vec_a), _ensure_2d(vec_b))[0][0])


def calcular_centroide_tematico(embeddings: np.ndarray) -> np.ndarray:
    if len(embeddings) == 0:
        raise ValueError("No hay embeddings para calcular el centroide.")
    return np.mean(embeddings, axis=0)


def agrupar_subtemas(
    embeddings: np.ndarray,
    documentos: Sequence[str],
    n_clusters: int,
) -> Tuple[Dict[int, List[str]], Dict[int, np.ndarray]]:
    if len(embeddings) == 0:
        raise ValueError("No se pueden agrupar embeddings vacios.")
    n_clusters = max(1, min(int(n_clusters), len(embeddings)))
    if len(embeddings) < int(n_clusters):
        logger.warning("Reduciendo n_clusters a %s por falta de datos", len(embeddings))
        n_clusters = len(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)
    temas = {
        i: [doc for j, doc in enumerate(documentos) if clusters[j] == i]
        for i in range(n_clusters)
    }
    centroides = {i: kmeans.cluster_centers_[i] for i in range(n_clusters)}
    return temas, centroides


def mapear_cobertura_tematica(
    embeddings_prop: np.ndarray,
    embeddings_comp: np.ndarray,
    documentos_comp: Sequence[str],
    n_temas: int,
    umbral_gap: float,
) -> AuthorityGapResult:
    if len(embeddings_prop) == 0 or len(embeddings_comp) == 0:
        raise ValueError("Se necesitan embeddings del sitio propio y competencia.")
    temas_comp, centroides_comp = agrupar_subtemas(embeddings_comp, documentos_comp, n_temas)
    centroide_prop = calcular_centroide_tematico(embeddings_prop)

    clusters: List[ClusterSummary] = []
    gaps: List[ClusterSummary] = []
    for cluster_id, centroide_cluster in centroides_comp.items():
        similarity = calcular_similitud_coseno(centroide_prop, centroide_cluster)
        documents = temas_comp.get(cluster_id, [])
        summary = ClusterSummary(
            cluster_id=cluster_id + 1,
            documents=documents,
            similarity=similarity,
            is_gap=similarity < umbral_gap,
        )
        clusters.append(summary)
        if summary.is_gap:
            gaps.append(summary)

    metadata = {
        "n_clusters": len(centroides_comp),
        "umbral_gap": umbral_gap,
    }
    return AuthorityGapResult(
        clusters=clusters,
        gaps=gaps,
        gap_threshold=umbral_gap,
        competitor_documents=list(documentos_comp),
        site_documents=[],
        metadata=metadata,
    )


def generar_embeddings_simulados(
    n_competencia: int,
    n_propias: int,
    dimensiones: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    if n_competencia <= 0 or n_propias <= 0 or dimensiones <= 0:
        raise ValueError("Los parametros de simulacion deben ser positivos.")
    rng = np.random.default_rng(seed)
    embeddings_comp = rng.random((n_competencia, dimensiones))
    embeddings_prop = rng.random((n_propias, dimensiones))
    if n_propias > 0 and n_competencia > 0:
        embeddings_prop[-1] = embeddings_comp[0] + 0.05 * rng.random(dimensiones)
    docs_comp = [f"Comp_Doc_{i + 1}" for i in range(n_competencia)]
    docs_prop = [f"Prop_Doc_{i + 1}" for i in range(n_propias)]
    return embeddings_comp, embeddings_prop, docs_comp, docs_prop


def run_authority_gap_simulation(
    n_competencia: int = 10,
    n_propias: int = 8,
    dimensiones: int = 200,
    n_clusters: int = 4,
    umbral_gap: float = 0.7,
    seed: int = 42,
) -> AuthorityGapResult:
    embeddings_comp, embeddings_prop, docs_comp, docs_prop = generar_embeddings_simulados(
        n_competencia=n_competencia,
        n_propias=n_propias,
        dimensiones=dimensiones,
        seed=seed,
    )
    analysis = mapear_cobertura_tematica(
        embeddings_prop=embeddings_prop,
        embeddings_comp=embeddings_comp,
        documentos_comp=docs_comp,
        n_temas=n_clusters,
        umbral_gap=umbral_gap,
    )
    analysis.site_documents = docs_prop
    analysis.metadata.update(
        {
            "n_competencia": n_competencia,
            "n_propias": n_propias,
            "dimensiones": dimensiones,
            "seed": seed,
        }
    )
    return analysis


def run_authority_gap_from_embeddings(
    embeddings_prop: np.ndarray,
    embeddings_comp: np.ndarray,
    docs_prop: Sequence[str],
    docs_comp: Sequence[str],
    n_clusters: int,
    umbral_gap: float,
) -> AuthorityGapResult:
    if embeddings_prop.ndim != 2 or embeddings_comp.ndim != 2:
        raise ValueError("Los embeddings deben tener forma (n, d).")
    if embeddings_prop.shape[1] != embeddings_comp.shape[1]:
        raise ValueError("Ambos conjuntos deben tener la misma dimension.")
    if len(docs_prop) != len(embeddings_prop) or len(docs_comp) != len(embeddings_comp):
        raise ValueError("El numero de documentos debe coincidir con los embeddings.")

    analysis = mapear_cobertura_tematica(
        embeddings_prop=embeddings_prop,
        embeddings_comp=embeddings_comp,
        documentos_comp=docs_comp,
        n_temas=n_clusters,
        umbral_gap=umbral_gap,
    )
    analysis.site_documents = list(docs_prop)
    analysis.competitor_documents = list(docs_comp)
    analysis.metadata.update(
        {
            "n_competencia": len(docs_comp),
            "n_propias": len(docs_prop),
            "dimensiones": int(embeddings_comp.shape[1]),
        }
    )
    return analysis


def cargar_datos_enlazado(ruta_excel: str) -> pd.DataFrame:
    required_columns = {"Source", "Target", "Anchor"}
    df = pd.read_excel(ruta_excel)
    if not required_columns.issubset(df.columns):
        raise ValueError(f"El Excel debe contener las columnas {sorted(required_columns)}")
    return df.dropna(subset=list(required_columns)).copy()


def calcular_pagerank_semantico(
    df_enlaces: pd.DataFrame,
    embeddings_anchor: np.ndarray,
    embeddings_target: np.ndarray,
    alpha: float = 0.85,
) -> List[SemanticPageRankResult]:
    if len(df_enlaces) == 0:
        raise ValueError("El DataFrame de enlaces esta vacio.")
    if len(df_enlaces) != len(embeddings_anchor) or len(df_enlaces) != len(embeddings_target):
        raise ValueError("La cantidad de embeddings debe coincidir con el numero de filas del Excel.")

    graph = nx.DiGraph()
    for idx, row in df_enlaces.iterrows():
        source = row["Source"]
        target = row["Target"]
        sim = calcular_similitud_coseno(embeddings_anchor[idx], embeddings_target[idx])
        weight = 1.0 + sim
        if graph.has_edge(source, target):
            existing = graph[source][target]["weight"]
            if weight > existing:
                graph[source][target]["weight"] = weight
                graph[source][target]["semantic_sim"] = sim
        else:
            graph.add_edge(source, target, weight=weight, semantic_sim=sim)

    pagerank_scores = nx.pagerank(graph, alpha=alpha, weight="weight")
    results: List[SemanticPageRankResult] = []
    for url, score in pagerank_scores.items():
        in_edges = list(graph.in_edges(url, data=True))
        avg_semantic_boost = (
            float(np.mean([data["semantic_sim"] for _, _, data in in_edges])) if in_edges else 0.0
        )
        results.append(
            SemanticPageRankResult(
                url=url,
                pagerank_score=score,
                semantic_boost=avg_semantic_boost,
                incoming_links=graph.in_degree(url),
            )
        )
    results.sort(key=lambda item: item.pagerank_score, reverse=True)
    return results


__all__ = [
    "AuthorityGapResult",
    "ClusterSummary",
    "SemanticPageRankResult",
    "cargar_datos_enlazado",
    "calcular_pagerank_semantico",
    "mapear_cobertura_tematica",
    "run_authority_gap_from_embeddings",
    "run_authority_gap_simulation",
]
