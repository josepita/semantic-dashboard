"""
Internal Linking Modules
=========================

Módulos especializados para algoritmos de enlazado interno.

Estructura:
- linking_pagerank: PageRank temático y grafos semánticos
- linking_algorithms: Algoritmos básicos y avanzados
- linking_structural: Enlazado estructural/taxonómico
- linking_hybrid: Algoritmo híbrido (semántica + autoridad + entidades)
- linking_utils: Utilidades de reporting y Gemini AI
"""

__version__ = '1.0.0'

from .linking_pagerank import (
    build_similarity_edges,
    calculate_topical_pagerank,
    detect_orphan_pages,
    calculate_graph_metrics,
)

from .linking_algorithms import (
    semantic_link_recommendations,
    advanced_semantic_linking,
)

from .linking_structural import structural_taxonomy_linking

from .linking_hybrid import hybrid_semantic_linking

from .linking_utils import (
    guess_default_type,
    build_entity_payload_from_doc_relations,
    build_linking_reports_payload,
    interpret_linking_reports_with_gemini,
)

__all__ = [
    # PageRank
    "build_similarity_edges",
    "calculate_topical_pagerank",
    "detect_orphan_pages",
    "calculate_graph_metrics",
    # Algoritmos
    "semantic_link_recommendations",
    "advanced_semantic_linking",
    "structural_taxonomy_linking",
    "hybrid_semantic_linking",
    # Utilidades
    "guess_default_type",
    "build_entity_payload_from_doc_relations",
    "build_linking_reports_payload",
    "interpret_linking_reports_with_gemini",
]
