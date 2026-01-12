"""
Positions Report Modules
========================

Módulos especializados para el análisis y generación de informes de posiciones SEO.

Estructura:
- positions_parsing: Parsing y normalización de archivos CSV/Excel
- positions_analysis: Análisis de datos y asignación de familias
- positions_payload: Construcción de payloads para informes
- positions_reports: Generación de reportes HTML (estáticos y con Gemini AI)
"""

__version__ = '1.0.0'

from .positions_parsing import (
    normalize_domain,
    parse_position_tracking_csv,
    parse_search_volume_file,
)

from .positions_analysis import (
    assign_keyword_families,
    summarize_positions_overview,
)

from .positions_payload import (
    build_family_payload,
    build_competitive_family_payload,
)

from .positions_reports import (
    generate_competitive_html_report,
    generate_position_report_html,
)

__all__ = [
    # Parsing
    "normalize_domain",
    "parse_position_tracking_csv",
    "parse_search_volume_file",
    # Analysis
    "assign_keyword_families",
    "summarize_positions_overview",
    # Payload builders
    "build_family_payload",
    "build_competitive_family_payload",
    # Report generation
    "generate_competitive_html_report",
    "generate_position_report_html",
]
