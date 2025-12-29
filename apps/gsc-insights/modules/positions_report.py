"""
Módulo para el informe de posiciones SEO.

Este módulo gestiona la carga, procesamiento y generación de informes HTML
a partir de datos de rank tracking exportados desde herramientas SEO.

Con soporte para persistencia en DuckDB por proyecto.
"""

import re
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
from datetime import datetime

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    import duckdb
except ImportError:
    duckdb = None

from modules.keyword_builder import group_keywords_with_semantic_builder
from modules.semantic_tools import download_dataframe_button


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
    Parsea un archivo CSV/Excel de rank tracking y lo normaliza.

    Se espera que el archivo contenga al menos las columnas:
    - Keyword o Query: La keyword rastreada
    - Position o Rank: La posición en SERP
    - URL: La URL que rankea
    O en formato SERP:
    - Keyword, Position 1, Position 2, ..., Position 10

    Args:
        uploaded_file: Archivo CSV o Excel subido por Streamlit

    Returns:
        DataFrame normalizado con columnas estándar

    Raises:
        ValueError: Si el archivo no contiene las columnas requeridas
    """
    uploaded_file.seek(0)
    file_extension = uploaded_file.name.split('.')[-1].lower()

    # Leer el archivo según su tipo
    if file_extension in ['xlsx', 'xls']:
        try:
            df = pd.read_excel(uploaded_file, engine='openpyxl' if file_extension == 'xlsx' else None)
        except Exception as e:
            raise ValueError(f"No se pudo leer el archivo Excel: {e}")
    else:
        # Es un CSV - intentar diferentes delimitadores y configuraciones
        uploaded_file.seek(0)
        df = None
        delimiters = [',', ';', '\t', '|']  # Delimitadores más comunes

        for delimiter in delimiters:
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                for skip_rows in range(0, 5):  # Probar saltando hasta 5 filas
                    try:
                        uploaded_file.seek(0)
                        temp_df = pd.read_csv(
                            uploaded_file,
                            delimiter=delimiter,
                            encoding=encoding,
                            skiprows=skip_rows,
                            on_bad_lines='skip'
                        )

                        # Verificar que tenga al menos 2 columnas y que una sea 'Keyword' o similar
                        if len(temp_df.columns) >= 2:
                            cols_lower = [str(col).lower().strip() for col in temp_df.columns]
                            if any(kw in " ".join(cols_lower) for kw in ["keyword", "query", "position"]):
                                df = temp_df
                                break
                    except:
                        continue

                if df is not None:
                    break
            if df is not None:
                break

        # Si no se pudo leer con ningún delimitador, intentar detección automática como último recurso
        if df is None:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='utf-8', on_bad_lines='skip')
            except:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='latin-1', on_bad_lines='skip')
                except Exception as e:
                    raise ValueError(
                        f"No se pudo leer el archivo CSV. "
                        f"Prueba convertirlo a Excel (.xlsx) o verifica el formato.\n"
                        f"Error: {e}"
                    )

    # Mostrar información de debug
    if df.empty:
        raise ValueError("El archivo está vacío o no se pudo leer correctamente.")

    # Debug: Si solo hay una columna, es probable que el delimitador sea incorrecto
    if len(df.columns) == 1:
        raise ValueError(
            f"⚠️ El archivo parece tener un problema de formato.\n\n"
            f"📋 Se encontró solo una columna: '{df.columns[0]}'\n\n"
            f"💡 Posibles soluciones:\n"
            f"   1. Convierte el archivo a Excel (.xlsx) y súbelo de nuevo\n"
            f"   2. Verifica que el CSV use comas (,) como separador\n"
            f"   3. Abre el archivo en Excel y guárdalo como 'CSV UTF-8 (delimitado por comas)'\n\n"
            f"📄 Primeras filas del archivo:\n{df.head(3).to_string()}"
        )

    # Detectar formato Serprobot multi-keyword
    # Este formato tiene múltiples secciones, una por keyword
    uploaded_file.seek(0)
    lines = []
    try:
        content = uploaded_file.read()
        if isinstance(content, bytes):
            # Probar diferentes encodings
            for enc in ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']:
                try:
                    text = content.decode(enc)
                    lines = text.split('\n')
                    break
                except:
                    continue
        else:
            lines = content.split('\n')
        uploaded_file.seek(0)
    except:
        uploaded_file.seek(0)

    # Verificar si es formato Serprobot multi-keyword
    is_serprobot_multi = False
    if lines:
        for line in lines[:20]:
            if 'keyword:' in line.lower() and line.count(',') < 3:
                is_serprobot_multi = True
                break

    if is_serprobot_multi:
        # Procesar formato Serprobot multi-keyword manualmente
        uploaded_file.seek(0)
        all_rows = []
        current_keyword = None
        header_found = False
        position_columns_list = []
        first_date_processed = False  # Flag para procesar solo la primera fecha (más reciente)

        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Detectar línea de keyword
            if 'keyword:' in line.lower() and line.count(',') < 3:
                # Extraer keyword
                parts = line.split(':', 1)
                if len(parts) > 1:
                    current_keyword = parts[1].strip().strip('"').strip()
                    header_found = False
                    first_date_processed = False  # Resetear para cada nueva keyword
                    continue

            # Detectar header
            if 'position 1' in line.lower() or ('date' in line.lower() and 'position' in line.lower()):
                # Parsear header
                import csv
                from io import StringIO
                reader = csv.reader(StringIO(line))
                cols = next(reader)
                position_columns_list = [col.strip().strip('"') for col in cols if 'position' in col.lower() and any(c.isdigit() for c in col)]
                header_found = True
                continue

            # Procesar datos - SOLO LA PRIMERA FECHA (más reciente)
            if header_found and current_keyword and position_columns_list and not first_date_processed:
                try:
                    import csv
                    from io import StringIO
                    reader = csv.reader(StringIO(line))
                    values = next(reader)

                    # La primera columna es la fecha, el resto son las posiciones
                    if len(values) > len(position_columns_list):
                        # Procesar esta fecha
                        for pos_idx, url in enumerate(values[1:len(position_columns_list)+1]):
                            url = url.strip().strip('"')
                            if url and url.lower() not in ['', 'nan', 'none', 'null', 'not found']:
                                # Extraer dominio
                                if '://' in url:
                                    from urllib.parse import urlparse
                                    try:
                                        parsed = urlparse(url)
                                        domain = normalize_domain(parsed.netloc)
                                    except:
                                        domain = normalize_domain(url)
                                else:
                                    domain = url.lower().strip()

                                all_rows.append({
                                    "Keyword": current_keyword,
                                    "Position": pos_idx + 1,
                                    "Domain": domain,
                                    "URL": url
                                })
                        # Marcar que ya procesamos la primera fecha para esta keyword
                        first_date_processed = True
                except:
                    continue

        if all_rows:
            result_df = pd.DataFrame(all_rows)
            return result_df
        else:
            raise ValueError("No se pudieron extraer datos del formato Serprobot multi-keyword")

    # Si no es formato multi-keyword, continuar con el procesamiento normal
    # Detectar y manejar metadata de Serprobot (primeras filas con info del proyecto)
    header_row_idx = None
    for idx in range(min(10, len(df))):
        row_values = df.iloc[idx].astype(str).tolist()
        row_text = ' '.join(row_values).lower()
        # Buscar indicadores de header
        if any(indicator in row_text for indicator in ['keyword', 'position 1', 'date', 'position']):
            # Si esta fila tiene información de header pero no es la primera, hay metadata
            if idx > 0:
                # Recargar el DataFrame saltando las filas de metadata
                uploaded_file.seek(0)
                if file_extension in ['xlsx', 'xls']:
                    df = pd.read_excel(uploaded_file, skiprows=idx, engine='openpyxl' if file_extension == 'xlsx' else None)
                else:
                    # Para CSV, usar el mismo delimitador que funcionó antes
                    uploaded_file.seek(0)
                    for delimiter in delimiters:
                        try:
                            temp_df = pd.read_csv(uploaded_file, delimiter=delimiter, encoding='utf-8', skiprows=idx, on_bad_lines='skip')
                            if len(temp_df.columns) >= 2:
                                df = temp_df
                                break
                        except:
                            continue
            break

    # Detectar formato del CSV
    # Formato 1: Keyword, Position, URL (formato simple)
    # Formato 2: Keyword, Position 1, Position 2, ..., Position 10 (formato SERP/Serprobot)

    position_columns = [col for col in df.columns if re.match(r'^Position\s+\d+$', col.strip(), re.IGNORECASE)]

    if position_columns:
        # Formato SERP: columnas Position 1, Position 2, etc.
        keyword_col = None
        date_col = None

        # Buscar columna de keyword y/o fecha
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in ["keyword", "query", "palabra clave", "consulta", "keywords"]:
                keyword_col = col
            elif 'date' in col_lower or 'fecha' in col_lower:
                date_col = col

        # Si no hay columna Keyword pero hay Date, es formato Serprobot serps_export
        # En este caso, la keyword está en el nombre del archivo o en metadata anterior
        if not keyword_col and date_col:
            # Intentar extraer keyword del nombre del archivo
            filename = uploaded_file.name
            # La keyword podría estar en las primeras filas como metadata
            # Por ahora, usar un placeholder y extraer de los datos
            keyword_col = None  # Lo manejaremos de forma especial
        elif not keyword_col:
            raise ValueError(
                f"❌ Formato SERP detectado, pero falta la columna 'Keyword'.\n"
                f"📋 Columnas encontradas: {', '.join(df.columns[:10])}\n"
                f"✅ Formato esperado: Keyword, Position 1, Position 2, ..., Position 10"
            )

        # Transformar formato SERP a formato simple
        rows = []

        # Si no hay columna Keyword, intentar extraer de las primeras filas del archivo original
        extracted_keyword = None
        if not keyword_col and date_col:
            # Leer las primeras líneas del archivo original para buscar la keyword
            uploaded_file.seek(0)
            for i in range(5):
                try:
                    line = uploaded_file.readline()
                    if isinstance(line, bytes):
                        line = line.decode('utf-8', errors='ignore')
                    # Buscar patrón "Keyword: xxx"
                    if 'keyword:' in line.lower():
                        parts = line.split(':', 1)
                        if len(parts) > 1:
                            extracted_keyword = parts[1].strip().strip('"').strip()
                            break
                except:
                    continue
            uploaded_file.seek(0)

        for _, row in df.iterrows():
            # Determinar la keyword para esta fila
            if keyword_col:
                keyword = str(row[keyword_col]).strip()
            elif extracted_keyword:
                keyword = extracted_keyword
            else:
                # Usar filename o placeholder
                keyword = uploaded_file.name.replace('.csv', '').replace('serprobot_serps_export_', '')

            if not keyword or keyword == "nan":
                continue

            for pos_col in position_columns:
                url_or_domain = str(row[pos_col]).strip()
                if url_or_domain and url_or_domain != "nan" and url_or_domain.lower() not in ["", "none", "null", "no encontrado", "not found"]:
                    # Extraer número de posición del nombre de columna
                    match = re.search(r'\d+', pos_col)
                    if match:
                        position = int(match.group())

                        # Extraer dominio de URL o usar directamente si ya es un dominio
                        if '://' in url_or_domain:
                            # Es una URL completa, extraer dominio
                            try:
                                from urllib.parse import urlparse
                                parsed = urlparse(url_or_domain)
                                domain = normalize_domain(parsed.netloc)
                            except:
                                domain = normalize_domain(url_or_domain)
                        elif url_or_domain.startswith('www.'):
                            domain = normalize_domain(url_or_domain)
                        else:
                            # Ya es un dominio limpio
                            domain = url_or_domain.lower().strip()

                        rows.append({
                            "Keyword": keyword,
                            "Position": position,
                            "Domain": domain,
                            "URL": url_or_domain
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
                f"❌ No se encontraron las columnas requeridas.\n"
                f"📋 Columnas encontradas: {', '.join(df.columns[:10])}\n"
                f"✅ Formato esperado:\n"
                f"   - Opción 1: Keyword, Position 1, Position 2, ..., Position 10\n"
                f"   - Opción 2: Keyword, Position, URL"
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

        # Agregar métricas de volumen de búsqueda si están disponibles
        if "SearchVolume" in family_df.columns:
            total_volume = family_df["SearchVolume"].sum()
            avg_volume = family_df["SearchVolume"].mean()
            family_info["volumen_total"] = int(total_volume)
            family_info["volumen_medio"] = round(avg_volume, 0)

            # Si hay datos de dominio, calcular volumen potencial de la marca
            if "Domain" in family_df.columns and brand_domain_normalized:
                brand_in_family = family_df[family_df["Domain"] == brand_domain_normalized]
                if not brand_in_family.empty:
                    family_info["volumen_marca"] = int(brand_in_family["SearchVolume"].sum())

        families_payload.append(family_info)

    return families_payload


def build_competitive_family_payload(
    df: pd.DataFrame,
    brand_domain: str,
    competitor_domains: Optional[List[str]] = None
) -> List[Dict]:
    """
    Construye payload competitivo con posiciones de todos los dominios por keyword.

    Args:
        df: DataFrame con columnas Keyword, Position, Domain, SearchVolume, Familia
        brand_domain: Dominio de la marca principal
        competitor_domains: Lista de dominios competidores

    Returns:
        Lista de diccionarios con estructura para tablas competitivas
    """
    if "Familia" not in df.columns:
        return []

    brand_domain_normalized = normalize_domain(brand_domain) if brand_domain else ""

    # Normalizar dominios competidores
    all_domains = [brand_domain_normalized]
    if competitor_domains:
        all_domains.extend([normalize_domain(d) for d in competitor_domains if d])

    families_payload = []

    for family_name in df["Familia"].unique():
        if family_name == "Sin familia":
            continue

        family_df = df[df["Familia"] == family_name].copy()

        # Obtener keywords únicas en esta familia
        unique_keywords = family_df["Keyword"].unique()

        keywords_data = []
        for kw in unique_keywords:
            kw_rows = family_df[family_df["Keyword"] == kw]

            # Obtener volumen (debería ser el mismo para todas las filas de la keyword)
            volume = 0
            if "SearchVolume" in kw_rows.columns:
                volume = kw_rows["SearchVolume"].iloc[0] if not kw_rows.empty else 0

            # Construir diccionario de posiciones por dominio
            positions = {}
            for domain in all_domains:
                domain_rows = kw_rows[kw_rows["Domain"] == domain]
                if not domain_rows.empty:
                    # Tomar la mejor posición si hay múltiples
                    positions[domain] = int(domain_rows["Position"].min())
                else:
                    positions[domain] = None  # No encontrado

            keywords_data.append({
                "keyword": kw,
                "volume": int(volume),
                "positions": positions
            })

        # Calcular métricas agregadas
        total_volume = sum(k["volume"] for k in keywords_data)
        avg_volume = total_volume / len(keywords_data) if keywords_data else 0

        # Posición media de la marca
        brand_positions = [k["positions"].get(brand_domain_normalized) for k in keywords_data
                          if k["positions"].get(brand_domain_normalized) is not None]
        avg_position = sum(brand_positions) / len(brand_positions) if brand_positions else None

        families_payload.append({
            "nombre": family_name,
            "total_keywords": len(keywords_data),
            "keywords_data": keywords_data,
            "domains": all_domains,
            "brand_domain": brand_domain_normalized,
            "volumen_total": total_volume,
            "volumen_medio": round(avg_volume, 0),
            "posicion_media_marca": round(avg_position, 1) if avg_position else None
        })

    return families_payload


def parse_search_volume_file(uploaded_file) -> pd.DataFrame:
    """
    Parsea un archivo Excel/CSV con volumen de búsqueda.

    Se espera que el archivo contenga al menos las columnas:
    - Keyword: La keyword
    - Volumen/Volume/Search Volume: El volumen de búsqueda

    Args:
        uploaded_file: Archivo Excel o CSV subido por Streamlit

    Returns:
        DataFrame normalizado con columnas Keyword y SearchVolume

    Raises:
        ValueError: Si el archivo no contiene las columnas requeridas
    """
    uploaded_file.seek(0)
    file_extension = uploaded_file.name.split('.')[-1].lower()

    try:
        if file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file, engine='openpyxl' if file_extension == 'xlsx' else None)
        else:
            # Es un CSV
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8")
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                try:
                    df = pd.read_csv(uploaded_file, encoding="latin-1")
                except Exception:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding="iso-8859-1")
    except Exception as e:
        raise ValueError(f"No se pudo leer el archivo de volumen: {e}")

    # Detectar columnas - primero intentar usar selección manual
    keyword_col = None
    volume_col = None

    # Verificar si hay selecciones manuales en session_state
    if "selected_keyword_col" in st.session_state and "selected_volume_col" in st.session_state:
        manual_kw_col = st.session_state["selected_keyword_col"]
        manual_vol_col = st.session_state["selected_volume_col"]

        # Verificar que las columnas seleccionadas existan en el DataFrame
        if manual_kw_col in df.columns and manual_vol_col in df.columns:
            keyword_col = manual_kw_col
            volume_col = manual_vol_col
        else:
            raise ValueError(
                f"Las columnas seleccionadas no existen en el archivo.\n"
                f"Columnas seleccionadas: '{manual_kw_col}', '{manual_vol_col}'\n"
                f"Columnas disponibles: {', '.join(df.columns)}"
            )
    else:
        # Detección automática si no hay selección manual
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in ["keyword", "query", "palabra clave", "consulta", "keywords"]:
                keyword_col = col
            elif col_lower in ["volumen", "volume", "search volume", "vol", "busquedas", "búsquedas", "searches"]:
                volume_col = col

        if not keyword_col or not volume_col:
            raise ValueError(
                f"El archivo debe contener columnas 'Keyword' y 'Volumen'. "
                f"Columnas encontradas: {', '.join(df.columns)}\n\n"
                f"💡 Sugerencia: Usa los selectores de arriba para indicar manualmente qué columna es la de keywords y cuál la de volumen."
            )

    # Normalizar nombres de columnas
    result_df = pd.DataFrame()
    result_df["Keyword"] = df[keyword_col].astype(str).str.strip()
    result_df["SearchVolume"] = pd.to_numeric(df[volume_col], errors="coerce").fillna(0).astype(int)

    # Eliminar filas sin keyword
    result_df = result_df[result_df["Keyword"].str.len() > 0]
    result_df = result_df[result_df["Keyword"] != "nan"]

    return result_df


def generate_competitive_html_report(
    report_title: str,
    brand_domain: str,
    competitive_payload: List[Dict],
    overview: Dict,
    competitor_domains: Optional[List[str]] = None
) -> str:
    """
    Genera informe HTML competitivo con tablas keyword por keyword.

    Args:
        report_title: Título del informe
        brand_domain: Dominio de la marca
        competitive_payload: Payload con datos competitivos (de build_competitive_family_payload)
        overview: Resumen general
        competitor_domains: Lista de dominios competidores

    Returns:
        HTML del informe
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


def save_gsc_data_to_db(df: pd.DataFrame, db_path: str) -> bool:
    """
    Guarda datos de GSC procesados en DuckDB.

    Args:
        df: DataFrame con columnas Keyword, Position, Domain, URL, SearchVolume (opcional), Familia (opcional)
        db_path: Ruta a la base de datos DuckDB

    Returns:
        True si se guardó correctamente, False si hubo error
    """
    if duckdb is None:
        st.warning("DuckDB no está instalado. Los datos no se guardarán.")
        return False

    try:
        conn = duckdb.connect(db_path)

        # Preparar DataFrame para insertar
        df_to_save = df.copy()

        # Asegurar que existan las columnas requeridas
        required_cols = ["Keyword", "Position"]
        for col in required_cols:
            if col not in df_to_save.columns:
                st.error(f"Columna requerida '{col}' no encontrada en los datos")
                conn.close()
                return False

        # Añadir columnas opcionales si no existen
        if "Domain" not in df_to_save.columns:
            df_to_save["Domain"] = ""
        if "URL" not in df_to_save.columns:
            df_to_save["URL"] = ""
        if "SearchVolume" not in df_to_save.columns:
            df_to_save["SearchVolume"] = 0

        # Limpiar tabla existente
        conn.execute("DELETE FROM gsc_positions")

        # Insertar datos
        insert_query = """
        INSERT INTO gsc_positions (keyword, url, position, impressions, clicks, ctr, date)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """

        current_date = datetime.now().date()

        for _, row in df_to_save.iterrows():
            conn.execute(
                insert_query,
                (
                    str(row["Keyword"]),
                    str(row.get("URL", "")),
                    float(row["Position"]),
                    int(row.get("SearchVolume", 0)),  # Usar SearchVolume como impressions
                    0,  # clicks (no disponible en CSV)
                    0.0,  # ctr (no disponible en CSV)
                    current_date
                )
            )

        # Guardar familias si existen
        if "Familia" in df_to_save.columns:
            families_data = {}
            for familia in df_to_save["Familia"].unique():
                if familia and familia != "Sin familia":
                    keywords = df_to_save[df_to_save["Familia"] == familia]["Keyword"].tolist()
                    families_data[familia] = keywords

            if families_data:
                # Limpiar y guardar familias
                conn.execute("DELETE FROM keyword_families")
                for familia_name, keywords_list in families_data.items():
                    import json
                    conn.execute(
                        """INSERT INTO keyword_families (family_name, keywords, description)
                           VALUES (?, ?, ?)""",
                        (familia_name, json.dumps(keywords_list), "")
                    )

        conn.close()
        return True

    except Exception as e:
        st.error(f"Error al guardar datos en DuckDB: {e}")
        import traceback
        st.code(traceback.format_exc())
        return False


def load_gsc_data_from_db(db_path: str) -> Optional[pd.DataFrame]:
    """
    Carga datos de GSC desde DuckDB.

    Args:
        db_path: Ruta a la base de datos DuckDB

    Returns:
        DataFrame con los datos o None si no hay datos o hubo error
    """
    if duckdb is None:
        return None

    try:
        conn = duckdb.connect(db_path, read_only=True)

        # Cargar datos
        query = """
        SELECT
            keyword as Keyword,
            url as URL,
            position as Position,
            impressions as SearchVolume,
            date
        FROM gsc_positions
        ORDER BY date DESC, keyword
        """

        df = conn.execute(query).fetch_df()

        if df.empty:
            conn.close()
            return None

        # Extraer dominios de URLs
        df["Domain"] = df["URL"].apply(
            lambda x: normalize_domain(urlparse(str(x)).netloc) if pd.notna(x) and x else ""
        )

        # Cargar familias si existen
        try:
            families_query = """
            SELECT family_name, keywords
            FROM keyword_families
            """
            families_df = conn.execute(families_query).fetch_df()

            if not families_df.empty:
                # Crear mapping de keyword -> familia
                import json
                familia_map = {}
                for _, row in families_df.iterrows():
                    familia_name = row["family_name"]
                    keywords_list = json.loads(row["keywords"])
                    for kw in keywords_list:
                        familia_map[kw] = familia_name

                # Aplicar familias al DataFrame
                df["Familia"] = df["Keyword"].apply(
                    lambda kw: familia_map.get(kw, "Sin familia")
                )
            else:
                df["Familia"] = "Sin familia"

        except Exception:
            df["Familia"] = "Sin familia"

        conn.close()
        return df

    except Exception as e:
        st.warning(f"No se pudieron cargar datos desde DuckDB: {e}")
        return None


def render_positions_report() -> None:
    """
    Renderiza la sección de informe de posiciones SEO en Streamlit.

    Permite cargar un CSV de rank tracking, procesarlo, agrupar keywords
    en familias (manual o automáticamente con Gemini), y generar un informe
    HTML profesional con visualizaciones y análisis.

    Con soporte para persistencia automática en DuckDB por proyecto.
    """
    from app_sections.landing_page import get_gemini_api_key_from_context, get_gemini_model_from_context

    st.subheader("📊 Informe de posiciones SEO")
    st.caption(
        "Procesa un CSV exportado de tu herramienta de rank tracking, añade volumen de búsqueda y genera informes HTML con Gemini."
    )

    st.session_state.setdefault("positions_raw_df", None)
    st.session_state.setdefault("positions_volume_df", None)
    st.session_state.setdefault("positions_report_html", None)
    st.session_state.setdefault("positions_competitors", [])
    st.session_state.setdefault("positions_semantic_groups", None)
    st.session_state.setdefault("positions_semantic_groups_raw", None)
    st.session_state.setdefault("positions_semantic_language", "es")
    st.session_state.setdefault("positions_semantic_country", "Spain")
    st.session_state.setdefault("positions_semantic_niche", "Proyecto SEO")
    st.session_state.setdefault("positions_auto_classify", True)
    if "positions_gemini_key" not in st.session_state or not st.session_state["positions_gemini_key"]:
        st.session_state["positions_gemini_key"] = get_gemini_api_key_from_context()
    if "positions_gemini_model" not in st.session_state or not st.session_state["positions_gemini_model"]:
        st.session_state["positions_gemini_model"] = get_gemini_model_from_context()

    # Verificar si hay proyecto seleccionado
    current_project = st.session_state.get("current_project")
    project_config = st.session_state.get("project_config")

    # Intentar cargar datos desde DuckDB si hay proyecto
    if current_project and project_config and duckdb is not None:
        db_path = project_config.get("db_path")
        if db_path:
            # Botón para cargar desde DuckDB
            if st.button("📊 Cargar datos guardados del proyecto", help="Carga los últimos datos guardados en la base de datos"):
                with st.spinner("Cargando datos desde DuckDB..."):
                    loaded_df = load_gsc_data_from_db(db_path)
                    if loaded_df is not None and not loaded_df.empty:
                        st.session_state["positions_raw_df"] = loaded_df
                        st.success(f"✅ Cargados {len(loaded_df)} registros desde la base de datos")
                    else:
                        st.info("No hay datos guardados en este proyecto. Sube un CSV para comenzar.")

    st.markdown("### 📥 Carga de archivos")

    col_upload1, col_upload2 = st.columns(2)
    with col_upload1:
        st.markdown("**CSV/Excel de posiciones** (requerido)")
        uploaded_csv = st.file_uploader(
            "Sube tu archivo",
            type=["csv", "xlsx", "xls"],
            key="positions_csv_uploader",
            help="Formato SERP con columnas: Keyword, Position 1, Position 2, ..., Position 10"
        )

    with col_upload2:
        st.markdown("**Excel con volumen de búsqueda** (opcional)")
        uploaded_volume = st.file_uploader(
            "Sube tu archivo",
            type=["xlsx", "xls", "csv"],
            key="positions_volume_uploader",
            help="Puede ser cualquier formato - seleccionarás las columnas después"
        )

    # Si se subió archivo de volumen, permitir selección manual de columnas
    if uploaded_volume:
        st.markdown("#### 🔧 Configurar archivo de volumen")
        try:
            uploaded_volume.seek(0)
            file_ext = uploaded_volume.name.split('.')[-1].lower()
            if file_ext in ['xlsx', 'xls']:
                preview_df = pd.read_excel(uploaded_volume, engine='openpyxl' if file_ext == 'xlsx' else None, nrows=5)
            else:
                preview_df = pd.read_csv(uploaded_volume, encoding='utf-8-sig', nrows=5)
            uploaded_volume.seek(0)

            col_sel1, col_sel2 = st.columns(2)
            with col_sel1:
                keyword_column = st.selectbox(
                    "Columna de Keywords",
                    options=list(preview_df.columns),
                    key="volume_keyword_col",
                    help="Selecciona la columna que contiene las keywords"
                )
            with col_sel2:
                volume_column = st.selectbox(
                    "Columna de Volumen",
                    options=list(preview_df.columns),
                    key="volume_volume_col",
                    help="Selecciona la columna que contiene el volumen de búsqueda"
                )

            # Guardar selecciones en session_state
            st.session_state["selected_keyword_col"] = keyword_column
            st.session_state["selected_volume_col"] = volume_column

            # Mostrar preview
            with st.expander("👁️ Vista previa del archivo de volumen"):
                st.dataframe(preview_df[[keyword_column, volume_column]].head(5))
        except Exception as e:
            st.warning(f"No se pudo leer el archivo para preview: {e}")
    else:
        # Limpiar selecciones manuales si no hay archivo de volumen
        if "selected_keyword_col" in st.session_state:
            del st.session_state["selected_keyword_col"]
        if "selected_volume_col" in st.session_state:
            del st.session_state["selected_volume_col"]

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
            <div style='border:2px solid rgba(255,255,255,0.5); border-radius:12px; padding:0.8rem; text-align:center; background-color: rgba(255,255,255,0.05);'>
                <div style='font-size:1.6rem'>{chart_icon}</div>
                <div style='font-size:0.9rem; color: #e0e0e0;'>{chart_label}</div>
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

    # Configuración de clasificación automática
    with st.expander("⚙️ Configuración de clasificación automática", expanded=False):
        auto_classify = st.checkbox(
            "Clasificar automáticamente al procesar CSV",
            value=st.session_state.get("positions_auto_classify", True),
            help="Usa Gemini para agrupar keywords automáticamente en familias semánticas al subir el CSV.",
        )
        st.session_state["positions_auto_classify"] = auto_classify

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

        if not auto_classify:
            st.info("💡 Usa las reglas manuales abajo para definir familias, o activa la clasificación automática.")

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

            # Procesar archivo de volumen si está presente
            if uploaded_volume:
                try:
                    volume_df = parse_search_volume_file(uploaded_volume)
                    st.session_state["positions_volume_df"] = volume_df
                    st.success(f"Se procesaron {len(parsed_df)} keywords y {len(volume_df)} registros de volumen.")
                except ValueError as vol_exc:
                    st.error(f"Error al procesar archivo de volumen: {vol_exc}")
                    st.session_state["positions_volume_df"] = None
            else:
                st.session_state["positions_volume_df"] = None
                st.success(f"Se procesaron {len(parsed_df)} keywords.")

            # Guardar automáticamente en DuckDB si hay proyecto seleccionado
            if current_project and project_config and duckdb is not None:
                db_path = project_config.get("db_path")
                if db_path:
                    with st.spinner("💾 Guardando datos en la base de datos del proyecto..."):
                        # Merge con volumen si existe
                        df_to_save = parsed_df.copy()
                        if uploaded_volume and st.session_state.get("positions_volume_df") is not None:
                            volume_df = st.session_state["positions_volume_df"]
                            df_to_save = df_to_save.merge(volume_df, on="Keyword", how="left")
                            df_to_save["SearchVolume"] = df_to_save["SearchVolume"].fillna(0).astype(int)

                        if save_gsc_data_to_db(df_to_save, db_path):
                            st.success("💾 Datos guardados en el proyecto")
            elif not current_project:
                st.info("💡 Crea un proyecto para guardar automáticamente los datos")

            # Clasificación automática si está habilitada y hay API key
            if st.session_state.get("positions_auto_classify", True) and gemini_api_key.strip():
                with st.spinner("🤖 Clasificando keywords automáticamente con Gemini..."):
                    try:
                        unique_keywords = parsed_df["Keyword"].dropna().astype(str).unique().tolist()
                        st.info(f"🔍 Enviando {len(unique_keywords)} keywords únicas a Gemini para clasificación...")

                        mapping, raw_groups = group_keywords_with_semantic_builder(
                            api_key=gemini_api_key.strip(),
                            model_name=gemini_model.strip() or get_gemini_model_from_context(),
                            keywords=unique_keywords,
                            language=st.session_state.get("positions_semantic_language", "es"),
                            country=st.session_state.get("positions_semantic_country", "Spain"),
                            niche=st.session_state.get("positions_semantic_niche", "Proyecto SEO"),
                            brand_domain=brand_domain,
                            competitors=st.session_state.get("positions_competitors", []),
                        )
                        st.session_state["positions_semantic_groups"] = mapping
                        st.session_state["positions_semantic_groups_raw"] = raw_groups
                        total_families = len({fam for fam in mapping.values() if fam})
                        keywords_classified = len([k for k in mapping.values() if k and k != "Sin familia"])
                        st.success(f"✅ Se clasificaron {keywords_classified} keywords en {total_families} familias semánticas.")

                        # Debug: mostrar una muestra del mapping
                        if mapping:
                            sample_items = list(mapping.items())[:3]
                            st.caption(f"📋 Ejemplo de clasificación: {dict(sample_items)}")
                    except Exception as exc:
                        st.error(f"❌ Error en clasificación automática: {exc}")
                        import traceback
                        st.code(traceback.format_exc())
                        st.session_state["positions_semantic_groups"] = None

    raw_df = st.session_state.get("positions_raw_df")
    if raw_df is None:
        st.info("Carga un CSV de posiciones para comenzar.")
        return

    # Merge con datos de volumen si están disponibles
    volume_df = st.session_state.get("positions_volume_df")
    if volume_df is not None and not volume_df.empty:
        # Hacer merge manteniendo todas las keywords de posiciones
        raw_df_with_volume = raw_df.merge(
            volume_df,
            on="Keyword",
            how="left"
        )
        # Rellenar valores faltantes con 0
        raw_df_with_volume["SearchVolume"] = raw_df_with_volume["SearchVolume"].fillna(0).astype(int)
    else:
        raw_df_with_volume = raw_df.copy()
        raw_df_with_volume["SearchVolume"] = 0

    # Leer las clasificaciones automáticas DESPUÉS de procesar el CSV
    semantic_grouping_map = st.session_state.get("positions_semantic_groups")
    use_semantic_builder = semantic_grouping_map is not None

    if use_semantic_builder and semantic_grouping_map:
        st.info(f"📊 Aplicando clasificación automática con {len(semantic_grouping_map)} keywords en el mapping")
        enriched_df = raw_df_with_volume.copy()
        enriched_df["Familia"] = (
            enriched_df["Keyword"]
            .astype(str)
            .apply(
                lambda kw: semantic_grouping_map.get(kw)
                or semantic_grouping_map.get(kw.lower())
                or "Sin familia"
            )
        )
        # Mostrar estadísticas de clasificación
        familias_asignadas = enriched_df[enriched_df["Familia"] != "Sin familia"]["Familia"].nunique()
        keywords_clasificadas = (enriched_df["Familia"] != "Sin familia").sum()
        st.caption(f"✅ {keywords_clasificadas} filas clasificadas en {familias_asignadas} familias diferentes")
    else:
        # Usar clasificación manual basada en reglas
        enriched_df = assign_keyword_families(raw_df_with_volume, families_instructions)
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

    # Métricas de volumen si están disponibles
    if "SearchVolume" in enriched_df.columns and enriched_df["SearchVolume"].sum() > 0:
        col_volume = st.columns(3)
        total_volume = int(enriched_df["SearchVolume"].sum())
        col_volume[0].metric("Volumen total de búsqueda", f"{total_volume:,}")

        # Volumen de la marca si hay datos de dominio
        brand_domain_normalized = normalize_domain(brand_domain) if brand_domain else ""
        if "Domain" in enriched_df.columns and brand_domain_normalized:
            brand_keywords = enriched_df[enriched_df["Domain"] == brand_domain_normalized]
            brand_volume = int(brand_keywords["SearchVolume"].sum())
            col_volume[1].metric("Volumen capturado por la marca", f"{brand_volume:,}")

            # Calcular potencial (keywords en top 10 pero no posición 1)
            brand_top10 = brand_keywords[brand_keywords["Position"] <= 10]
            potential_volume = int(brand_top10["SearchVolume"].sum())
            col_volume[2].metric("Volumen en Top 10", f"{potential_volume:,}")

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
        "Generar informe HTML competitivo",
        type="primary",
        disabled=not gemini_available,
    ):
        with st.spinner("⚙️ Generando informe competitivo..."):
            try:
                # Construir payload competitivo con posiciones de todos los dominios
                competitive_payload = build_competitive_family_payload(
                    df=enriched_df,
                    brand_domain=brand_domain or "Sin dominio especificado",
                    competitor_domains=competitor_domains
                )

                if not competitive_payload:
                    st.warning("No hay familias con datos para generar el informe")
                    return

                # Generar HTML competitivo
                html_report = generate_competitive_html_report(
                    report_title=report_title,
                    brand_domain=brand_domain or "Sin dominio especificado",
                    competitive_payload=competitive_payload,
                    overview=summary,
                    competitor_domains=competitor_domains,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"No se pudo generar el informe: {exc}")
                import traceback
                st.code(traceback.format_exc())
            else:
                st.session_state["positions_report_html"] = html_report
                st.success("Informe competitivo generado correctamente.")

    if st.session_state.get("positions_report_html"):
        st.write("### Vista previa del informe")
        components.html(st.session_state["positions_report_html"], height=700, scrolling=True)
        st.download_button(
            label="Descargar informe HTML",
            data=st.session_state["positions_report_html"].encode("utf-8"),
            file_name="informe_posiciones.html",
            mime="text/html",
        )
