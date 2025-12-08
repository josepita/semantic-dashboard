"""
Módulo para el informe de posiciones SEO.

Este módulo gestiona la carga, procesamiento y generación de informes HTML
a partir de datos de rank tracking exportados desde herramientas SEO.
"""

import re
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from app_sections.keyword_builder import group_keywords_with_semantic_builder
from app_sections.semantic_tools import download_dataframe_button


# Constantes para configuración de gráficos
POSITION_CHART_PRESETS: List[Tuple[str, str, str]] = [
    (
        "heatmap",
        "Heatmap por familia que compare la presencia relativa de cada dominio en las posiciones 1-10.",
        "🗺️",
    ),
    (
        "competitors",
        "Grafico de barras con la frecuencia total de los principales competidores en el Top 10 para todas las keywords.",
        "📊",
    ),
    (
        "radar",
        "Grafico radar que muestre la posicion media del dominio de la marca frente a competidores por familia.",
        "🎯",
    ),
    (
        "trendline",
        "Linea temporal con la evolucion de la posicion media por familia (si el CSV incluye fechas).",
        "📈",
    ),
    (
        "stacked",
        "Barras apiladas por familia indicando que dominio ocupa cada posicion del 1 al 5.",
        "📉",
    ),
]
DEFAULT_CHART_KEYS = ["heatmap", "competitors", "radar"]


def normalize_domain(domain: str) -> str:
    """
    Normaliza un dominio extrayendo solo la parte raíz.

    Args:
        domain: Dominio a normalizar (puede incluir http://, www., etc.)

    Returns:
        Dominio normalizado sin protocolo ni subdominios comunes

    Examples:
        >>> normalize_domain("https://www.example.com/path")
        'example.com'
        >>> normalize_domain("subdomain.example.com")
        'example.com'
    """
    if not domain:
        return ""

    # Remover protocolo si existe
    if "://" in domain:
        domain = domain.split("://")[1]

    # Remover path si existe
    if "/" in domain:
        domain = domain.split("/")[0]

    # Remover www. y subdominios comunes
    domain = domain.lower().strip()
    if domain.startswith("www."):
        domain = domain[4:]

    return domain


def parse_position_tracking_csv(uploaded_file) -> pd.DataFrame:
    """
    Parsea un archivo CSV de rank tracking y lo normaliza.

    Se espera que el CSV contenga al menos las columnas:
    - Keyword o Query: La keyword rastreada
    - Position o Rank: La posición en SERP
    - URL: La URL que rankea

    Args:
        uploaded_file: Archivo CSV subido por Streamlit

    Returns:
        DataFrame normalizado con columnas estándar

    Raises:
        ValueError: Si el archivo no contiene las columnas requeridas
    """
    # Intentar diferentes configuraciones de lectura
    # Primero intentar detectar si hay filas de metadatos antes del header real
    uploaded_file.seek(0)
    skip_rows = 0

    # Leer las primeras líneas para detectar el header real
    for i in range(20):  # Revisar hasta 20 líneas
        uploaded_file.seek(0)
        try:
            temp_df = pd.read_csv(uploaded_file, nrows=1, skiprows=i, encoding="utf-8", on_bad_lines='skip')
            # Verificar si esta fila contiene columnas que parecen headers válidos
            cols_lower = [str(col).lower().strip() for col in temp_df.columns]
            if any(kw in " ".join(cols_lower) for kw in ["keyword", "query", "position", "rank", "url", "ranking"]):
                skip_rows = i
                break
        except:
            continue

    # Ahora leer el archivo completo saltando las filas de metadatos
    uploaded_file.seek(0)
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8", skiprows=skip_rows, on_bad_lines='skip')
    except UnicodeDecodeError:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding="latin-1", skiprows=skip_rows, on_bad_lines='skip')
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding="iso-8859-1", skiprows=skip_rows, on_bad_lines='skip')
    except Exception as e:
        # Intentar con delimitador automático
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding="utf-8", skiprows=skip_rows, sep=None, engine='python', on_bad_lines='skip')
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding="latin-1", skiprows=skip_rows, sep=';', on_bad_lines='skip')

    # Detectar formato del CSV
    # Formato 1: Keyword, Position, URL (formato simple)
    # Formato 2: Keyword, Position 1, Position 2, ..., Position 10 (formato SERP)

    position_columns = [col for col in df.columns if re.match(r'^Position\s+\d+$', col.strip(), re.IGNORECASE)]

    if position_columns:
        # Formato SERP: columnas Position 1, Position 2, etc.
        keyword_col = None
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in ["keyword", "query", "palabra clave", "consulta"]:
                keyword_col = col
                break

        if not keyword_col:
            raise ValueError(
                "El CSV debe contener una columna 'Keyword'. "
                f"Columnas encontradas: {', '.join(df.columns)}"
            )

        # Transformar formato SERP a formato simple
        rows = []
        for _, row in df.iterrows():
            keyword = str(row[keyword_col]).strip()
            if not keyword or keyword == "nan":
                continue

            for pos_col in position_columns:
                domain = str(row[pos_col]).strip()
                if domain and domain != "nan" and domain.lower() not in ["", "none", "null"]:
                    # Extraer número de posición del nombre de columna
                    match = re.search(r'\d+', pos_col)
                    if match:
                        position = int(match.group())
                        rows.append({
                            "Keyword": keyword,
                            "Position": position,
                            "Domain": normalize_domain(domain),
                            "URL": domain
                        })

        if not rows:
            raise ValueError("No se encontraron datos válidos en el CSV formato SERP")

        result_df = pd.DataFrame(rows)

    else:
        # Formato simple: Keyword, Position, URL
        keyword_col = None
        position_col = None
        url_col = None

        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in ["keyword", "query", "palabra clave", "consulta"]:
                keyword_col = col
            elif col_lower in ["position", "rank", "posicion", "ranking"]:
                position_col = col
            elif col_lower in ["url", "landing page", "página", "pagina"]:
                url_col = col

        if not keyword_col or not position_col:
            raise ValueError(
                "El CSV debe contener al menos columnas de 'Keyword' y 'Position'. "
                f"Columnas encontradas: {', '.join(df.columns)}"
            )

        # Normalizar nombres de columnas
        result_df = pd.DataFrame()
        result_df["Keyword"] = df[keyword_col].astype(str).str.strip()
        result_df["Position"] = pd.to_numeric(df[position_col], errors="coerce")

        if url_col:
            result_df["URL"] = df[url_col].astype(str).str.strip()
            # Extraer dominio de la URL
            result_df["Domain"] = result_df["URL"].apply(
                lambda x: normalize_domain(urlparse(str(x)).netloc) if pd.notna(x) else ""
            )

        # Eliminar filas sin keyword o posición
        result_df = result_df.dropna(subset=["Keyword", "Position"])
        result_df = result_df[result_df["Keyword"].str.len() > 0]

    return result_df


def assign_keyword_families(df: pd.DataFrame, families_text: str) -> pd.DataFrame:
    """
    Asigna familias a las keywords basándose en reglas textuales.

    El formato del texto de familias es:
    Familia1: keyword1, keyword2, *patron*
    Familia2: keyword3, keyword4

    El asterisco (*) indica coincidencia parcial.

    Args:
        df: DataFrame con columna "Keyword"
        families_text: Texto con definiciones de familias

    Returns:
        DataFrame con columna "Familia" añadida
    """
    df = df.copy()
    df["Familia"] = "Sin familia"

    # Parsear definiciones de familias
    families_rules: Dict[str, List[str]] = {}
    for line in families_text.strip().split("\n"):
        if ":" not in line:
            continue
        family_name, patterns_str = line.split(":", 1)
        family_name = family_name.strip()
        # Separar por comas o punto y coma
        patterns = re.split(r"[,;]", patterns_str)
        families_rules[family_name] = [p.strip() for p in patterns if p.strip()]

    # Asignar familias
    for idx, row in df.iterrows():
        keyword_lower = str(row["Keyword"]).lower()
        for family_name, patterns in families_rules.items():
            for pattern in patterns:
                pattern_lower = pattern.lower()
                # Si tiene asterisco, es coincidencia parcial
                if pattern.startswith("*") and pattern.endswith("*"):
                    pattern_clean = pattern_lower.strip("*")
                    if pattern_clean in keyword_lower:
                        df.at[idx, "Familia"] = family_name
                        break
                elif pattern.startswith("*"):
                    pattern_clean = pattern_lower.strip("*")
                    if keyword_lower.endswith(pattern_clean):
                        df.at[idx, "Familia"] = family_name
                        break
                elif pattern.endswith("*"):
                    pattern_clean = pattern_lower.strip("*")
                    if keyword_lower.startswith(pattern_clean):
                        df.at[idx, "Familia"] = family_name
                        break
                else:
                    # Coincidencia exacta de palabra
                    if pattern_lower == keyword_lower or pattern_lower in keyword_lower.split():
                        df.at[idx, "Familia"] = family_name
                        break
            else:
                continue
            break

    return df


def summarize_positions_overview(
    df: pd.DataFrame,
    brand_domain: str,
    competitor_domains: List[str]
) -> Dict:
    """
    Genera un resumen estadístico de las posiciones.

    Args:
        df: DataFrame con columnas Keyword, Position, Domain, Familia
        brand_domain: Dominio de la marca
        competitor_domains: Lista de dominios competidores

    Returns:
        Diccionario con métricas clave
    """
    brand_domain_normalized = normalize_domain(brand_domain) if brand_domain else ""

    summary = {
        "total_keywords": len(df),
        "brand_keywords_in_top10": 0,
        "brand_average_position": None,
        "top_competitors_by_presence": []
    }

    if "Domain" not in df.columns:
        return summary

    # Filtrar keywords de la marca
    brand_df = df[df["Domain"] == brand_domain_normalized] if brand_domain_normalized else pd.DataFrame()

    if not brand_df.empty:
        # Keywords en top 10
        summary["brand_keywords_in_top10"] = len(brand_df[brand_df["Position"] <= 10])

        # Posición media
        summary["brand_average_position"] = round(brand_df["Position"].mean(), 1)

    # Competidores más frecuentes en top 10
    top10_df = df[df["Position"] <= 10]
    if not top10_df.empty and "Domain" in top10_df.columns:
        competitor_counts = (
            top10_df["Domain"]
            .value_counts()
            .head(10)
            .reset_index()
        )
        competitor_counts.columns = ["Domain", "Count"]
        # Filtrar el dominio de la marca
        if brand_domain_normalized:
            competitor_counts = competitor_counts[
                competitor_counts["Domain"] != brand_domain_normalized
            ]
        summary["top_competitors_by_presence"] = competitor_counts.values.tolist()

    return summary


def build_family_payload(
    df: pd.DataFrame,
    brand_domain: str,
    max_keywords_per_family: int = 20,
    competitor_domains: Optional[List[str]] = None
) -> List[Dict]:
    """
    Construye el payload de familias para el informe.

    Args:
        df: DataFrame con datos procesados
        brand_domain: Dominio de la marca
        max_keywords_per_family: Máximo de keywords por familia
        competitor_domains: Lista de dominios competidores

    Returns:
        Lista de diccionarios con información por familia
    """
    if "Familia" not in df.columns:
        return []

    brand_domain_normalized = normalize_domain(brand_domain) if brand_domain else ""
    families_payload = []

    for family_name in df["Familia"].unique():
        if family_name == "Sin familia":
            continue

        family_df = df[df["Familia"] == family_name].copy()

        # Limitar keywords por familia
        family_df = family_df.head(max_keywords_per_family)

        # Calcular métricas
        family_info = {
            "nombre": family_name,
            "total_keywords": len(family_df),
            "keywords": family_df["Keyword"].tolist(),
        }

        if "Position" in family_df.columns:
            family_info["posicion_media"] = round(family_df["Position"].mean(), 1)

        if "Domain" in family_df.columns and brand_domain_normalized:
            brand_in_family = family_df[family_df["Domain"] == brand_domain_normalized]
            family_info["keywords_marca_top10"] = len(
                brand_in_family[brand_in_family["Position"] <= 10]
            )

        families_payload.append(family_info)

    return families_payload


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
        HTML del informe generado

    Raises:
        ValueError: Si genai no está disponible o hay error con la API
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
        f"posición media: {fam.get('posicion_media', 'N/A')}"
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
3. Presentar las familias de keywords con sus métricas
4. Sugerir oportunidades de mejora basadas en los datos
5. Incluir secciones para los gráficos solicitados (deja placeholders con títulos)
6. Usar colores corporativos suaves (azules, grises)
7. Ser completamente autocontenido (todo el CSS inline)

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


def render_positions_report() -> None:
    """
    Renderiza la sección de informe de posiciones SEO en Streamlit.

    Permite cargar un CSV de rank tracking, procesarlo, agrupar keywords
    en familias (manual o automáticamente con Gemini), y generar un informe
    HTML profesional con visualizaciones y análisis.
    """
    from app_sections.landing_page import get_gemini_api_key_from_context, get_gemini_model_from_context

    st.subheader("📊 Informe de posiciones SEO")
    st.caption(
        "Procesa un CSV exportado de tu herramienta de rank tracking, genera tablas por dominio y crea un informe HTML con Gemini."
    )

    st.session_state.setdefault("positions_raw_df", None)
    st.session_state.setdefault("positions_report_html", None)
    st.session_state.setdefault("positions_competitors", [])
    st.session_state.setdefault("positions_semantic_groups", None)
    st.session_state.setdefault("positions_semantic_groups_raw", None)
    st.session_state.setdefault("positions_semantic_language", "es")
    st.session_state.setdefault("positions_semantic_country", "Spain")
    st.session_state.setdefault("positions_semantic_niche", "Proyecto SEO")
    if "positions_gemini_key" not in st.session_state or not st.session_state["positions_gemini_key"]:
        st.session_state["positions_gemini_key"] = get_gemini_api_key_from_context()
    if "positions_gemini_model" not in st.session_state or not st.session_state["positions_gemini_model"]:
        st.session_state["positions_gemini_model"] = get_gemini_model_from_context()

    col_main, col_side = st.columns([3, 1])
    with col_main:
        uploaded_csv = st.file_uploader("Archivo CSV con posiciones", type=["csv"], key="positions_csv_uploader")
    with col_side:
        max_keywords = st.slider("Keywords por familia", min_value=5, max_value=40, value=20, step=5)

    col_a, col_b = st.columns(2)
    with col_a:
        brand_domain = st.text_input(
            "Dominio principal",
            value=st.session_state.get("positions_brand", ""),
            help="Se usará para resaltar la marca en las tablas.",
        )
    with col_b:
        report_title = st.text_input(
            "Título del informe",
            value=st.session_state.get("positions_report_title", "Informe de posiciones orgánicas"),
        )

    competitor_domains_raw = st.text_input(
        "Dominios competidores (separa por coma)",
        value=", ".join(st.session_state.get("positions_competitors", [])),
        help="Añade los dominios raíz que quieres vigilar en el informe.",
    )
    competitor_domains = [
        normalize_domain(domain.strip())
        for domain in competitor_domains_raw.split(",")
        if domain.strip()
    ]
    st.session_state["positions_competitors"] = [domain for domain in competitor_domains if domain]

    families_instructions = st.text_area(
        "Definición de familias (formato: Familia: keyword1, keyword2, *fragmento*)",
        value=st.session_state.get(
            "positions_family_text",
            "Anillos: anillo, alianzas\nPendientes: pendiente, aro\nCollares: collar, colgante, gargantilla",
        ),
        height=140,
        help="Usa comas o punto y coma para separar keywords/patrones. El carácter * permite coincidencias parciales.",
    )

    st.markdown("**Selecciona los gráficos que quieres incluir en el informe**")
    chart_columns = st.columns(3)
    default_chart_keys = st.session_state.get("positions_chart_selection") or DEFAULT_CHART_KEYS
    selected_chart_keys: List[str] = []
    cards_per_row = len(chart_columns)
    for idx, (chart_key, chart_label, chart_icon) in enumerate(POSITION_CHART_PRESETS):
        widget_key = f"positions_chart_{chart_key}"
        col = chart_columns[idx % cards_per_row]
        col.markdown(
            f"""
            <div style='border:1px solid rgba(255,255,255,0.2); border-radius:12px; padding:0.8rem; text-align:center;'>
                <div style='font-size:1.6rem'>{chart_icon}</div>
                <div style='font-size:0.9rem; opacity:0.8'>{chart_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        checked = col.checkbox(
            f"{chart_icon} {chart_key.title()}",
            key=widget_key,
            value=chart_key in default_chart_keys,
            help=chart_label,
        )
        if checked:
            selected_chart_keys.append(chart_key)
    custom_chart_note = st.text_area(
        "Notas adicionales para visualizaciones (opcional)",
        value=st.session_state.get("positions_chart_custom", ""),
        height=80,
        key="positions_chart_custom_input",
    )
    st.session_state["positions_chart_custom"] = custom_chart_note
    st.session_state["positions_chart_selection"] = selected_chart_keys
    chart_descriptions = [desc for key, desc, _ in POSITION_CHART_PRESETS if key in selected_chart_keys]
    if custom_chart_note.strip():
        chart_descriptions.append(custom_chart_note.strip())
    if not chart_descriptions:
        st.info("Selecciona al menos un gráfico o añade una nota personalizada para guiar al informe.")
    chart_notes = "\n".join(f"- {desc}" for desc in chart_descriptions).strip()

    use_semantic_builder = st.checkbox(
        "Usar agrupación automática (Semantic Keyword Builder)",
        value=st.session_state.get("positions_use_semantic_builder", False),
        help="Reutiliza la lógica del builder para clasificar las keywords sin reglas manuales.",
    )
    st.session_state["positions_use_semantic_builder"] = use_semantic_builder
    semantic_grouping_map: Optional[Dict[str, str]] = None
    raw_positions_df = st.session_state.get("positions_raw_df")
    if use_semantic_builder:
        if raw_positions_df is None or raw_positions_df.empty:
            st.info("Carga un CSV antes de ejecutar la agrupación automática.")
        else:
            with st.expander(
                "Configurar agrupación automática",
                expanded=st.session_state.get("positions_semantic_groups") is None,
            ):
                cfg_cols = st.columns(3)
                with cfg_cols[0]:
                    semantic_language = st.text_input(
                        "Idioma de las keywords",
                        value=st.session_state.get("positions_semantic_language", "es"),
                        key="positions_semantic_language",
                    )
                with cfg_cols[1]:
                    semantic_country = st.text_input(
                        "País / mercado",
                        value=st.session_state.get("positions_semantic_country", "Spain"),
                        key="positions_semantic_country",
                    )
                with cfg_cols[2]:
                    semantic_niche = st.text_input(
                        "Nicho o proyecto",
                        value=st.session_state.get("positions_semantic_niche", "Proyecto SEO"),
                        key="positions_semantic_niche",
                    )
                semantic_api_key = st.text_input(
                    "Gemini API Key",
                    type="password",
                    value=st.session_state.get("positions_gemini_key", get_gemini_api_key_from_context()),
                    key="positions_semantic_api_key",
                )
                semantic_model_name = st.text_input(
                    "Modelo Gemini",
                    value=st.session_state.get("positions_gemini_model", get_gemini_model_from_context()),
                    key="positions_semantic_model_name",
                )
                if st.button("Agrupar keywords con Gemini", key="positions_semantic_group_button"):
                    try:
                        mapping, raw_groups = group_keywords_with_semantic_builder(
                            api_key=semantic_api_key.strip(),
                            model_name=semantic_model_name.strip() or get_gemini_model_from_context(),
                            keywords=raw_positions_df["Keyword"].dropna().astype(str).tolist(),
                            language=semantic_language.strip() or "es",
                            country=semantic_country.strip() or "Spain",
                            niche=semantic_niche.strip() or "Proyecto SEO",
                            brand_domain=brand_domain,
                            competitors=st.session_state.get("positions_competitors", []),
                        )
                    except Exception as exc:
                        st.error(str(exc))
                    else:
                        st.session_state["positions_semantic_groups"] = mapping
                        st.session_state["positions_semantic_groups_raw"] = raw_groups
                        st.session_state["positions_semantic_language"] = semantic_language
                        st.session_state["positions_semantic_country"] = semantic_country
                        st.session_state["positions_semantic_niche"] = semantic_niche
                        st.session_state["positions_gemini_key"] = semantic_api_key.strip()
                        st.session_state["positions_gemini_model"] = semantic_model_name.strip() or get_gemini_model_from_context()
                        total_families = len({fam for fam in mapping.values() if fam})
                        st.success(
                            f"Se agruparon {len(mapping)} keywords a través de {total_families} familias semánticas."
                        )
            semantic_grouping_map = st.session_state.get("positions_semantic_groups")
    else:
        semantic_grouping_map = None

    col_api1, col_api2 = st.columns(2)
    with col_api1:
        gemini_api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=st.session_state.get("positions_gemini_key", ""),
        )
    with col_api2:
        gemini_model = st.text_input(
            "Modelo Gemini",
            value=st.session_state.get("positions_gemini_model", get_gemini_model_from_context()),
        )
    gemini_available = genai is not None
    if not gemini_available:
        st.warning(
            "Instala la librería `google-generativeai` en tu entorno para habilitar el informe "
            "(pip install google-generativeai)."
        )

    if uploaded_csv and st.button("Procesar CSV", type="primary"):
        try:
            parsed_df = parse_position_tracking_csv(uploaded_csv)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.session_state["positions_raw_df"] = parsed_df
            st.session_state["positions_brand"] = brand_domain
            st.success(f"Se procesaron {len(parsed_df)} keywords.")

    raw_df = st.session_state.get("positions_raw_df")
    if raw_df is None:
        st.info("Carga un CSV de posiciones para comenzar.")
        return

    if use_semantic_builder and semantic_grouping_map:
        enriched_df = raw_df.copy()
        enriched_df["Familia"] = (
            enriched_df["Keyword"]
            .astype(str)
            .apply(
                lambda kw: semantic_grouping_map.get(kw)
                or semantic_grouping_map.get(kw.lower())
                or "Sin familia"
            )
        )
    else:
        if use_semantic_builder and not semantic_grouping_map:
            st.warning(
                "Activa la agrupación con Gemini para rellenar las familias o desmarca la opción para seguir con reglas manuales."
            )
        enriched_df = assign_keyword_families(raw_df, families_instructions)
    st.session_state["positions_family_text"] = families_instructions
    st.session_state["positions_report_title"] = report_title

    st.write("### Vista previa de datos")
    st.dataframe(enriched_df.head(50), use_container_width=True)
    download_dataframe_button(enriched_df, "tabla_dominios.xlsx", "Descargar tabla procesada (Excel)")

    chart_notes_payload = chart_notes or "El analista definira los graficos adecuados."
    summary = summarize_positions_overview(enriched_df, brand_domain, competitor_domains)
    st.session_state["positions_summary"] = summary
    family_payload = build_family_payload(
        enriched_df,
        brand_domain,
        max_keywords_per_family=max_keywords,
        competitor_domains=competitor_domains,
    )
    st.session_state["positions_payload"] = family_payload

    col_metrics = st.columns(3)
    col_metrics[0].metric("Keywords analizadas", summary.get("total_keywords", 0))
    col_metrics[1].metric(
        "Keywords con la marca en Top10",
        summary.get("brand_keywords_in_top10", 0),
    )
    avg_pos = summary.get("brand_average_position")
    col_metrics[2].metric("Posicion media de la marca", avg_pos if avg_pos is not None else "No disponible")
    if competitor_domains:
        st.caption(f"Competidores definidos manualmente: {', '.join(competitor_domains)}")

    with st.expander("Competidores más frecuentes"):
        competitor_counts = summary.get("top_competitors_by_presence", [])
        if competitor_counts:
            competitor_df = pd.DataFrame(competitor_counts, columns=["Dominio", "Frecuencia"])
            st.dataframe(competitor_df, use_container_width=True)
        else:
            st.info("Sin datos de competidores en el Top 10.")

    st.session_state["positions_chart_notes"] = chart_notes_payload
    st.session_state["positions_gemini_key"] = gemini_api_key
    st.session_state["positions_gemini_model"] = gemini_model
    st.session_state["gemini_api_key"] = gemini_api_key.strip()
    st.session_state["gemini_model_name"] = gemini_model.strip() or get_gemini_model_from_context()

    if not family_payload:
        st.warning("No hay datos suficientes para generar el informe. Ajusta las familias o revisa el CSV.")
        return

    if st.button(
        "Generar informe HTML con Gemini",
        type="primary",
        disabled=not gemini_available,
    ):
        with st.spinner("⚙️ Generando informe con Gemini..."):
            try:
                html_report = generate_position_report_html(
                    api_key=gemini_api_key,
                    model_name=gemini_model,
                    report_title=report_title,
                    brand_domain=brand_domain or "Sin dominio especificado",
                    families_payload=family_payload,
                    overview=summary,
                    chart_notes=chart_notes_payload,
                    competitor_domains=competitor_domains,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"No se pudo generar el informe: {exc}")
            else:
                st.session_state["positions_report_html"] = html_report
                st.success("Informe generado correctamente.")

    if st.session_state.get("positions_report_html"):
        st.write("### Vista previa del informe")
        components.html(st.session_state["positions_report_html"], height=700, scrolling=True)
        st.download_button(
            label="Descargar informe HTML",
            data=st.session_state["positions_report_html"].encode("utf-8"),
            file_name="informe_posiciones.html",
            mime="text/html",
        )
