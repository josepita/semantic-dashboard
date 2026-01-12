"""
Módulo de generación de reportes HTML para informes de posiciones SEO.

Este módulo contiene funciones para generar informes HTML profesionales
tanto estáticos (competitivos) como dinámicos (con Gemini AI).
"""

from __future__ import annotations

from typing import Dict, List, Optional

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from .positions_parsing import normalize_domain


def generate_competitive_html_report(
    report_title: str,
    brand_domain: str,
    competitive_payload: List[Dict],
    overview: Dict,
    competitor_domains: Optional[List[str]] = None
) -> str:
    """
    Genera informe HTML competitivo con tablas keyword por keyword.

    Crea un informe HTML estático con tablas comparativas mostrando las posiciones
    de la marca vs competidores para cada keyword, con colores según ranking.

    Args:
        report_title: Título del informe
        brand_domain: Dominio de la marca
        competitive_payload: Payload con datos competitivos (de build_competitive_family_payload)
        overview: Resumen general
        competitor_domains: Lista de dominios competidores

    Returns:
        HTML completo del informe con CSS inline

    Examples:
        >>> payload = [{
        ...     "nombre": "Familia1",
        ...     "keywords_data": [{
        ...         "keyword": "ejemplo",
        ...         "volume": 1000,
        ...         "positions": {"example.com": 3, "competitor.com": 5}
        ...     }],
        ...     "domains": ["example.com", "competitor.com"]
        ... }]
        >>> html = generate_competitive_html_report("Informe SEO", "example.com", payload, {})
        >>> "<!DOCTYPE html>" in html
        True
    """
    brand_normalized = normalize_domain(brand_domain) if brand_domain else ""

    # Generar CSS
    css = """
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        line-height: 1.6;
        color: #333;
        max-width: 1200px;
        margin: 20px auto;
        padding: 0 20px;
        background-color: #f9f9f9;
    }
    h1, h2, h3 {
        color: #1a237e;
        border-bottom: 2px solid #c5cae9;
        padding-bottom: 10px;
    }
    h1 { font-size: 2.5em; text-align: center; margin-bottom: 30px;}
    h2 { font-size: 2em; margin-top: 40px; }
    table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    th, td {
        border: 1px solid #ddd;
        padding: 10px;
        text-align: center;
        font-size: 0.9em;
    }
    th {
        background-color: #3f51b5;
        color: white;
        font-weight: bold;
    }
    tbody tr:nth-child(even) {
        background-color: #f2f2f2;
    }
    tbody tr:hover {
        background-color: #e8eaf6;
    }
    .analysis-block {
        background-color: #fff;
        border-left: 5px solid #3f51b5;
        padding: 20px;
        margin-top: 15px;
        border-radius: 5px;
    }
    .analysis-block strong { color: #303f9f; }
    .not-found {
        color: #d32f2f;
        font-weight: bold;
        background-color: #ffcdd2;
    }
    .pos-1 { background-color: #4CAF50; color: white; font-weight: bold; }
    .pos-2-3 { background-color: #a5d6a7; color: #1b5e20; }
    .pos-4-7 { background-color: #fff59d; color: #5d4037; }
    .pos-8-10 { background-color: #ffcc80; color: #5d4037; }
    td:first-child, th:first-child { text-align: left; font-weight: bold; }
    """

    # Función auxiliar para clase CSS según posición
    def get_position_class(pos):
        if pos is None:
            return 'not-found'
        if pos == 1:
            return 'pos-1'
        elif 2 <= pos <= 3:
            return 'pos-2-3'
        elif 4 <= pos <= 7:
            return 'pos-4-7'
        elif 8 <= pos <= 10:
            return 'pos-8-10'
        return ''

    # Función auxiliar para texto de posición
    def get_position_text(pos):
        return str(pos) if pos is not None else 'No encontrado'

    # Generar HTML
    html_parts = [f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{report_title} - {brand_domain}</title>
<style>
{css}
</style>
</head>
<body>

<h1>{report_title}: {brand_domain}</h1>
"""]

    # Generar secciones por familia
    for idx, family in enumerate(competitive_payload, 1):
        family_name = family["nombre"]
        keywords_data = family["keywords_data"]
        domains = family["domains"]

        html_parts.append(f"""
<h2>{idx}. Familia de Keywords: {family_name}</h2>
<div class="analysis-block">
    <p><strong>Métricas:</strong> {len(keywords_data)} keywords analizadas""")

        if family.get("volumen_total"):
            html_parts.append(f""", volumen total: {family['volumen_total']:,}""")

        if family.get("posicion_media_marca"):
            html_parts.append(f""", posición media de {brand_domain}: {family['posicion_media_marca']}""")

        html_parts.append("""</p>
</div>
<table>
    <thead>
        <tr>
            <th>Keyword</th>""")

        # Agregar columnas para cada dominio
        for domain in domains:
            html_parts.append(f"<th>{domain}</th>")

        if any(k["volume"] > 0 for k in keywords_data):
            html_parts.append("<th>Volumen</th>")

        html_parts.append("""
        </tr>
    </thead>
    <tbody>""")

        # Agregar filas de keywords
        for kw_data in keywords_data:
            kw = kw_data["keyword"]
            positions = kw_data["positions"]
            volume = kw_data["volume"]

            html_parts.append(f"<tr><td>{kw}</td>")

            for domain in domains:
                pos = positions.get(domain)
                pos_class = get_position_class(pos)
                pos_text = get_position_text(pos)
                html_parts.append(f'<td class="{pos_class}">{pos_text}</td>')

            if any(k["volume"] > 0 for k in keywords_data):
                html_parts.append(f"<td>{volume:,}</td>")

            html_parts.append("</tr>")

        html_parts.append("""
    </tbody>
</table>
""")

    html_parts.append("""
</body>
</html>
""")

    return "".join(html_parts)


def generate_position_report_html(
    api_key: str,
    model_name: str,
    report_title: str,
    brand_domain: str,
    families_payload: List[Dict],
    overview: Dict,
    chart_notes: str,
    competitor_domains: Optional[List[str]] = None
) -> str:
    """
    Genera un informe HTML usando la API de Gemini.

    Crea un informe HTML profesional generado por Gemini AI con análisis,
    visualizaciones, insights y recomendaciones basadas en los datos.

    Args:
        api_key: API key de Gemini
        model_name: Nombre del modelo (ej: gemini-2.5-flash)
        report_title: Título del informe
        brand_domain: Dominio de la marca
        families_payload: Lista de familias con sus métricas
        overview: Resumen general de posiciones
        chart_notes: Notas sobre gráficos a incluir
        competitor_domains: Lista de dominios competidores

    Returns:
        HTML completo del informe generado por Gemini

    Raises:
        ValueError: Si genai no está disponible o hay error con la API

    Examples:
        >>> # Requiere API key válida
        >>> payload = [{"nombre": "Familia1", "total_keywords": 10, "posicion_media": 5.2}]
        >>> overview = {"total_keywords": 10, "brand_keywords_in_top10": 5}
        >>> html = generate_position_report_html(
        ...     "API_KEY", "gemini-2.5-flash", "Informe", "example.com",
        ...     payload, overview, "Incluir gráficos de tendencias"
        ... )  # doctest: +SKIP
    """
    if genai is None:
        raise ValueError(
            "La librería google-generativeai no está instalada. "
            "Instala con: pip install google-generativeai"
        )

    if not api_key:
        raise ValueError("Se requiere una API key de Gemini para generar el informe")

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        raise ValueError(f"Error al configurar Gemini: {e}")

    # Construir el prompt para Gemini
    competitors_text = ""
    if competitor_domains:
        competitors_text = f"\n\nCompetidores principales: {', '.join(competitor_domains)}"

    families_text = "\n".join([
        f"- {fam['nombre']}: {fam['total_keywords']} keywords, "
        f"posición media: {fam.get('posicion_media', 'N/A')}" +
        (f", volumen total: {fam.get('volumen_total', 0):,}" if 'volumen_total' in fam else "") +
        (f", volumen medio: {fam.get('volumen_medio', 0):.0f}" if 'volumen_medio' in fam else "")
        for fam in families_payload
    ])

    prompt = f"""
Genera un informe HTML profesional de posiciones SEO con los siguientes datos:

**Título del informe:** {report_title}
**Dominio de la marca:** {brand_domain}
**Total de keywords analizadas:** {overview.get('total_keywords', 0)}
**Keywords de la marca en Top 10:** {overview.get('brand_keywords_in_top10', 0)}
**Posición media de la marca:** {overview.get('brand_average_position', 'N/A')}
{competitors_text}

**Familias de keywords:**
{families_text}

**Gráficos a incluir:**
{chart_notes}

El informe debe:
1. Tener un diseño moderno y profesional con CSS inline
2. Incluir un resumen ejecutivo destacando insights clave
3. Presentar las familias de keywords con sus métricas (incluyendo volumen de búsqueda cuando esté disponible)
4. Incluir tablas con las keywords más importantes de cada familia, mostrando posición y volumen
5. Sugerir oportunidades de mejora basadas en los datos de posición y volumen
6. Incluir secciones para los gráficos solicitados (deja placeholders con títulos)
7. Priorizar oportunidades por volumen de búsqueda potencial
8. Usar colores corporativos suaves (azules, grises)
9. Ser completamente autocontenido (todo el CSS inline)

Genera SOLO el HTML completo, sin explicaciones adicionales.
"""

    try:
        response = model.generate_content(prompt)
        html_content = response.text

        # Limpiar el HTML si viene con markdown
        if html_content.startswith("```html"):
            html_content = html_content.split("```html")[1]
            html_content = html_content.split("```")[0]
        elif html_content.startswith("```"):
            html_content = html_content.split("```")[1]
            html_content = html_content.split("```")[0]

        return html_content.strip()

    except Exception as e:
        raise ValueError(f"Error al generar el informe con Gemini: {e}")


__all__ = [
    "generate_competitive_html_report",
    "generate_position_report_html",
]
