"""
Módulo de parsing y normalización para informes de posiciones SEO.

Este módulo contiene funciones para parsear archivos CSV/Excel de rank tracking
y normalizar los datos a un formato estándar.

Formatos soportados:
- Formato simple: Keyword, Position, URL
- Formato SERP: Keyword, Position 1, Position 2, ..., Position 10
- Serprobot multi-keyword: Secciones con headers "Keyword: xxx"
- SE Ranking Brief export: Una fila por keyword con volumen, CPC y competidores
- SE Ranking Full export: Metadata del proyecto + historial de fechas por keyword
"""

from __future__ import annotations

import re
import csv
from dataclasses import dataclass, field
from io import StringIO
from typing import TYPE_CHECKING, List, Optional
from urllib.parse import urlparse

import pandas as pd

if TYPE_CHECKING:
    from typing import Any


@dataclass
class ParseResult:
    """Resultado del parsing de un CSV de posiciones."""
    df: pd.DataFrame
    volume_df: Optional[pd.DataFrame] = None
    detected_competitors: List[str] = field(default_factory=list)
    detected_brand_domain: str = ""
    export_format: str = "unknown"

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


def _detect_export_format(lines: list[str]) -> str:
    """Detecta el formato de exportación analizando las primeras líneas.

    Formatos:
    - brief: Header con "Keyword for:" o "current position" (una fila por keyword)
    - serp_multi: Secciones "Keyword: xxx" con columnas "Position 1, Position 2..."
      que contienen URLs. Puede tener metadata "Project Name" al inicio.
    - full: Secciones "Keyword: xxx" con columnas "Position, Found SERP,
      Search Volume..." (posición única + datos enriquecidos).
    """
    if not lines:
        return "unknown"

    # Brief export: tiene "keyword for:" en el header
    for line in lines[:5]:
        if "keyword for:" in line.lower() or "current position" in line.lower():
            return "brief"

    # Buscar pistas en las líneas de header dentro de secciones keyword
    # para distinguir SERP (Position 1, Position 2...) de Full (Position, Found SERP...)
    has_keyword_section = False
    has_position_numbered = False  # "Position 1", "Position 2"...
    has_found_serp = False  # "Found SERP" column
    has_best_position = False

    for line in lines[:50]:
        ll = line.lower().strip()
        # Detectar "Keyword: xxx" — puede tener comas de padding en SE Ranking
        # En full export, la keyword y "Best Position" van en la misma línea
        if ll.startswith("keyword:") and "keyword for:" not in ll:
            stripped = re.sub(r',+\s*$', '', line).strip()
            if stripped.count(",") < 5:
                has_keyword_section = True
        if "best position:" in ll:
            has_best_position = True
        if re.search(r'position\s+\d', ll):
            has_position_numbered = True
        if "found serp" in ll and "search volume" in ll:
            has_found_serp = True

    if has_keyword_section and has_position_numbered:
        return "serp_multi"
    if has_keyword_section and has_found_serp:
        return "full"
    if has_keyword_section and has_best_position:
        return "full"
    if has_keyword_section:
        return "serp_multi"

    return "unknown"


def _read_file_lines(uploaded_file: Any) -> list[str]:
    """Lee las líneas del archivo intentando diferentes encodings."""
    uploaded_file.seek(0)
    content = uploaded_file.read()
    uploaded_file.seek(0)
    if isinstance(content, bytes):
        for enc in ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']:
            try:
                return content.decode(enc).split('\n')
            except Exception:
                continue
    return content.split('\n') if isinstance(content, str) else []


def _parse_brief_export(uploaded_file: Any) -> ParseResult:
    """Parsea el formato brief export de SE Ranking."""
    uploaded_file.seek(0)
    file_ext = uploaded_file.name.split('.')[-1].lower()

    if file_ext in ['xlsx', 'xls']:
        df = pd.read_excel(uploaded_file, engine='openpyxl' if file_ext == 'xlsx' else None)
    else:
        df = None
        for delim in [',', ';', '\t']:
            for enc in ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']:
                try:
                    uploaded_file.seek(0)
                    temp = pd.read_csv(uploaded_file, delimiter=delim, encoding=enc, on_bad_lines='skip')
                    if len(temp.columns) >= 3:
                        df = temp
                        break
                except Exception:
                    continue
            if df is not None:
                break
        if df is None:
            raise ValueError("No se pudo leer el brief export")

    cols = list(df.columns)
    cols_lower = [str(c).lower().strip() for c in cols]

    # Detectar columna de keyword (ej: "Keyword for: (pedalmoto.com)")
    keyword_col = None
    brand_domain = ""
    for i, cl in enumerate(cols_lower):
        if "keyword for:" in cl or "keyword" in cl:
            keyword_col = cols[i]
            # Extraer dominio del nombre de columna si existe
            m = re.search(r'\(([^)]+)\)', str(cols[i]))
            if m:
                brand_domain = normalize_domain(m.group(1))
            break

    # Detectar columna de posición actual
    position_col = None
    for i, cl in enumerate(cols_lower):
        if "current position" in cl or cl == "position":
            position_col = cols[i]
            break

    # Detectar Found SERP
    url_col = None
    for i, cl in enumerate(cols_lower):
        if "found serp" in cl:
            url_col = cols[i]
            break

    # Detectar Best Position
    best_pos_col = None
    for i, cl in enumerate(cols_lower):
        if "best position" in cl:
            best_pos_col = cols[i]
            break

    # Detectar First Position
    first_pos_col = None
    for i, cl in enumerate(cols_lower):
        if "first position" in cl:
            first_pos_col = cols[i]
            break

    # Detectar volumen de búsqueda (local primero, luego global)
    volume_col = None
    for i, cl in enumerate(cols_lower):
        if "local search volume" in cl or "search volume local" in cl:
            volume_col = cols[i]
            break
    if not volume_col:
        for i, cl in enumerate(cols_lower):
            if "global search volume" in cl or "search volume global" in cl:
                volume_col = cols[i]
                break

    # Detectar CPC
    cpc_col = None
    for i, cl in enumerate(cols_lower):
        if "cpc" in cl and ("local" in cl or "(es)" in cl or cols_lower.index(cl) == i):
            cpc_col = cols[i]
            break
    if not cpc_col:
        for i, cl in enumerate(cols_lower):
            if "cpc" in cl:
                cpc_col = cols[i]
                break

    # Detectar competidores (columnas "Competitor: xxx.com")
    competitor_cols = {}
    for i, cl in enumerate(cols_lower):
        if cl.startswith("competitor:") or cl.startswith("competitor "):
            m = re.search(r'competitor[:\s]+(.+)', str(cols[i]), re.IGNORECASE)
            if m:
                comp_domain = normalize_domain(m.group(1).strip())
                if comp_domain:
                    competitor_cols[cols[i]] = comp_domain

    if not keyword_col:
        raise ValueError("No se encontró columna de keyword en el brief export")

    # Construir DataFrame de posiciones
    rows = []
    for _, row in df.iterrows():
        kw = str(row[keyword_col]).strip()
        if not kw or kw == "nan":
            continue

        # Posición actual del dominio principal
        pos = pd.to_numeric(row.get(position_col), errors="coerce") if position_col else None
        url_val = str(row.get(url_col, "")).strip() if url_col else ""
        if url_val in ("nan", "None", ""):
            url_val = ""

        # Extraer dominio de la URL encontrada
        domain = ""
        if url_val and "://" in url_val:
            try:
                domain = normalize_domain(urlparse(url_val).netloc)
            except Exception:
                domain = normalize_domain(url_val)
        elif brand_domain and pd.notna(pos) and pos > 0:
            domain = brand_domain

        if pd.notna(pos) and pos > 0:
            rows.append({
                "Keyword": kw,
                "Position": int(pos),
                "Domain": domain or brand_domain,
                "URL": url_val,
            })

        # Añadir posiciones de competidores
        for col_name, comp_domain in competitor_cols.items():
            comp_pos = pd.to_numeric(row.get(col_name), errors="coerce")
            if pd.notna(comp_pos) and comp_pos > 0:
                rows.append({
                    "Keyword": kw,
                    "Position": int(comp_pos),
                    "Domain": comp_domain,
                    "URL": "",
                })

    if not rows:
        raise ValueError("No se encontraron datos válidos en el brief export")

    result_df = pd.DataFrame(rows)

    # Construir DataFrame de volumen
    volume_df = None
    if volume_col:
        vol_rows = []
        for _, row in df.iterrows():
            kw = str(row[keyword_col]).strip()
            if not kw or kw == "nan":
                continue
            vol = pd.to_numeric(row.get(volume_col), errors="coerce")
            vol_rows.append({
                "Keyword": kw.lower(),
                "SearchVolume": int(vol) if pd.notna(vol) else 0,
            })
        if vol_rows:
            volume_df = pd.DataFrame(vol_rows)

    detected_competitors = list(competitor_cols.values())

    return ParseResult(
        df=result_df,
        volume_df=volume_df,
        detected_competitors=detected_competitors,
        detected_brand_domain=brand_domain,
        export_format="brief",
    )


def _parse_full_export(lines: list[str]) -> ParseResult:
    """Parsea el formato full export de SE Ranking (metadata + secciones por keyword)."""
    # Extraer metadata del proyecto de las primeras líneas
    brand_domain = ""
    for line in lines[:5]:
        if "project domain" in line.lower() or "project domain:" in line.lower():
            reader = csv.reader(StringIO(line))
            vals = next(reader)
            for v in vals:
                v = v.strip()
                if v.lower().startswith("project domain"):
                    parts = v.split(":", 1) if ":" in v else v.split(" ", 2)
                    if len(parts) > 1:
                        brand_domain = normalize_domain(parts[-1].strip())
                elif "." in v and not v.lower().startswith("project"):
                    # Podría ser el dominio directamente
                    candidate = normalize_domain(v)
                    if candidate and not brand_domain:
                        brand_domain = candidate

    # Parsear la metadata de la primera línea para buscar dominio
    if not brand_domain and lines:
        reader = csv.reader(StringIO(lines[0]))
        try:
            vals = next(reader)
            for v in vals:
                v = v.strip()
                if "." in v and len(v) > 3 and not any(k in v.lower() for k in ["project", "region", "name"]):
                    brand_domain = normalize_domain(v)
                    if brand_domain:
                        break
        except Exception:
            pass

    all_rows = []
    volume_data = {}
    competitor_domains_set = set()
    current_keyword = None
    header_cols = []
    first_data_processed = False

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # Detectar línea "Keyword: xxx" o "Keyword: xxx,Best Position: N,,,,"
        if line_stripped.lower().startswith("keyword:") and "keyword for:" not in line_stripped.lower():
            # Parsear con csv reader para manejar comas
            try:
                reader = csv.reader(StringIO(line_stripped))
                parts = next(reader)
                # El primer campo es "Keyword: xxx"
                kw_part = parts[0].strip()
                kw_split = kw_part.split(":", 1)
                if len(kw_split) > 1:
                    current_keyword = kw_split[1].strip().strip('"').strip()
                    if current_keyword:
                        header_cols = []
                        first_data_processed = False
                        continue
            except Exception:
                pass

        # Detectar línea de header (Date (UTC), Keyword, Position, Found SERP...)
        if current_keyword and "date" in line_stripped.lower()[:20] and ("keyword" in line_stripped.lower() or "position" in line_stripped.lower()):
            reader = csv.reader(StringIO(line_stripped))
            try:
                header_cols = [c.strip().strip('"') for c in next(reader)]
            except Exception:
                header_cols = []
            continue

        # Procesar primera fila de datos (fecha más reciente)
        if current_keyword and header_cols and not first_data_processed:
            try:
                reader = csv.reader(StringIO(line_stripped))
                values = next(reader)
                values = [v.strip().strip('"') for v in values]

                if len(values) < 3:
                    continue

                # Mapear valores a columnas
                col_map = {}
                for idx, col_name in enumerate(header_cols):
                    if idx < len(values):
                        col_map[col_name.lower()] = values[idx]

                # Extraer posición
                pos_val = None
                for key in col_map:
                    if key == "position":
                        pos_val = pd.to_numeric(col_map[key], errors="coerce")
                        break

                # Extraer Found SERP (URL)
                url_val = ""
                for key in col_map:
                    if "found serp" in key:
                        url_val = col_map[key]
                        break

                # Extraer Search Volume
                sv_val = 0
                for key in col_map:
                    if "search volume local" in key or "local search volume" in key:
                        sv_val = pd.to_numeric(col_map[key], errors="coerce")
                        if pd.isna(sv_val):
                            sv_val = 0
                        break
                if sv_val == 0:
                    for key in col_map:
                        if "search volume" in key:
                            sv_val = pd.to_numeric(col_map[key], errors="coerce")
                            if pd.isna(sv_val):
                                sv_val = 0
                            break

                volume_data[current_keyword.lower()] = int(sv_val)

                # Extraer dominio de la URL
                domain = ""
                if url_val and "://" in url_val:
                    try:
                        domain = normalize_domain(urlparse(url_val).netloc)
                    except Exception:
                        domain = normalize_domain(url_val)
                elif brand_domain:
                    domain = brand_domain

                if pd.notna(pos_val) and pos_val > 0:
                    all_rows.append({
                        "Keyword": current_keyword,
                        "Position": int(pos_val),
                        "Domain": domain or brand_domain,
                        "URL": url_val,
                    })

                # Extraer posiciones de competidores
                for key in col_map:
                    if key.startswith("competitor:") or key.startswith("competitor "):
                        m = re.search(r'competitor[:\s]+(.+)', key, re.IGNORECASE)
                        if m:
                            comp_domain = normalize_domain(m.group(1).strip())
                            if comp_domain:
                                competitor_domains_set.add(comp_domain)
                                comp_pos = pd.to_numeric(col_map[key], errors="coerce")
                                if pd.notna(comp_pos) and comp_pos > 0:
                                    all_rows.append({
                                        "Keyword": current_keyword,
                                        "Position": int(comp_pos),
                                        "Domain": comp_domain,
                                        "URL": "",
                                    })

                first_data_processed = True
            except Exception:
                continue

    if not all_rows:
        raise ValueError("No se pudieron extraer datos del full export")

    result_df = pd.DataFrame(all_rows)

    # Construir volume_df
    volume_df = None
    if volume_data:
        volume_df = pd.DataFrame([
            {"Keyword": kw, "SearchVolume": vol}
            for kw, vol in volume_data.items()
        ])

    return ParseResult(
        df=result_df,
        volume_df=volume_df,
        detected_competitors=sorted(competitor_domains_set),
        detected_brand_domain=brand_domain,
        export_format="full",
    )


def _parse_serp_multi_export(lines: list[str]) -> ParseResult:
    """Parsea formato SERP multi-keyword (Position 1..10 con URLs por keyword)."""
    all_rows: list[dict] = []
    current_keyword = None
    header_found = False
    position_columns_list: list[str] = []
    first_date_processed = False

    # Extraer brand domain de metadata "Project Domain: xxx" si existe
    serp_brand_domain = ""
    for line in lines[:5]:
        if "project domain" in line.lower():
            try:
                reader = csv.reader(StringIO(line))
                vals = next(reader)
                for v in vals:
                    v_stripped = v.strip()
                    if v_stripped.lower().startswith("project domain"):
                        parts = v_stripped.split(":", 1)
                        if len(parts) > 1:
                            serp_brand_domain = normalize_domain(parts[1].strip())
            except Exception:
                pass

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Saltar líneas de metadata del proyecto
        if "project name:" in line.lower() or "project region:" in line.lower() or "project domain:" in line.lower():
            continue

        # Detectar línea de keyword (puede tener comas de padding: "Keyword: xxx,,,,,,")
        if 'keyword:' in line.lower():
            stripped_line = re.sub(r',+\s*$', '', line).strip()
            if stripped_line.count(',') < 3:
                parts = stripped_line.split(':', 1)
                if len(parts) > 1:
                    kw_val = parts[1].strip().strip('"').strip()
                    kw_val = re.sub(r',+$', '', kw_val).strip()
                    if kw_val:
                        current_keyword = kw_val
                        header_found = False
                        first_date_processed = False
                        continue

        # Detectar header
        if 'position 1' in line.lower() or ('date' in line.lower() and 'position' in line.lower()):
            reader = csv.reader(StringIO(line))
            cols = next(reader)
            position_columns_list = [
                col.strip().strip('"') for col in cols
                if 'position' in col.lower() and any(c.isdigit() for c in col)
            ]
            header_found = True
            continue

        # Procesar datos - SOLO LA PRIMERA FECHA (más reciente)
        if header_found and current_keyword and position_columns_list and not first_date_processed:
            try:
                reader = csv.reader(StringIO(line))
                values = next(reader)

                if len(values) > len(position_columns_list):
                    for pos_idx, url in enumerate(values[1:len(position_columns_list) + 1]):
                        url = url.strip().strip('"')
                        if url and url.lower() not in ['', 'nan', 'none', 'null', 'not found']:
                            if '://' in url:
                                try:
                                    parsed = urlparse(url)
                                    domain = normalize_domain(parsed.netloc)
                                except Exception:
                                    domain = normalize_domain(url)
                            else:
                                domain = url.lower().strip()

                            all_rows.append({
                                "Keyword": current_keyword,
                                "Position": pos_idx + 1,
                                "Domain": domain,
                                "URL": url,
                            })
                    first_date_processed = True
            except Exception:
                continue

    if not all_rows:
        raise ValueError("No se pudieron extraer datos del formato SERP multi-keyword")

    return ParseResult(
        df=pd.DataFrame(all_rows),
        detected_brand_domain=serp_brand_domain,
        export_format="serp",
    )


def parse_position_tracking_csv(uploaded_file: Any, column_mapping: dict | None = None) -> ParseResult | pd.DataFrame:
    """
    Parsea un archivo CSV/Excel de rank tracking y lo normaliza.

    Soporta múltiples formatos:
    1. Formato simple: Keyword, Position, URL
    2. Formato SERP: Keyword, Position 1, Position 2, ..., Position 10
    3. Formato Serprobot multi-keyword: Múltiples secciones con headers "Keyword: xxx"

    Args:
        uploaded_file: Archivo CSV o Excel subido por Streamlit
        column_mapping: Mapeo manual de columnas proporcionado por el usuario.
            Claves opcionales: ``keyword``, ``position``, ``url``, ``domain``.

    Returns:
        DataFrame normalizado con columnas: Keyword, Position, Domain, URL

    Raises:
        ValueError: Si el archivo no contiene las columnas requeridas o no se puede parsear
    """
    uploaded_file.seek(0)
    file_extension = uploaded_file.name.split('.')[-1].lower()

    # --- Detección temprana de formatos SE Ranking (antes de leer con pandas) ---
    # Para CSV, detectar formato leyendo las líneas directamente
    if file_extension not in ['xlsx', 'xls']:
        lines = _read_file_lines(uploaded_file)
        export_format = _detect_export_format(lines)

        if export_format == "brief":
            return _parse_brief_export(uploaded_file)
        if export_format == "full":
            return _parse_full_export(lines)
        if export_format == "serp_multi":
            # Procesar formato SERP multi-keyword
            return _parse_serp_multi_export(lines)

    # Leer el archivo según su tipo (solo para formatos estándar)
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

    # Si no es formato SE Ranking, continuar con el procesamiento normal
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

        # Usar column_mapping si se proporcionó
        if column_mapping and "keyword" in column_mapping:
            keyword_col = column_mapping["keyword"]

        # Buscar columna de keyword y/o fecha por auto-detección
        if not keyword_col:
            for col in df.columns:
                col_lower = col.lower().strip()
                if col_lower in ["keyword", "query", "palabra clave", "consulta", "keywords"]:
                    keyword_col = col
                elif 'date' in col_lower or 'fecha' in col_lower:
                    date_col = col

        # Si no hay columna Keyword pero hay Date, es formato Serprobot serps_export
        if not keyword_col and date_col:
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
        keyword_col = column_mapping.get("keyword") if column_mapping else None
        position_col = column_mapping.get("position") if column_mapping else None
        url_col = column_mapping.get("url") if column_mapping else None
        domain_col = column_mapping.get("domain") if column_mapping else None

        # Auto-detección como fallback
        for col in df.columns:
            col_lower = col.lower().strip()
            if not keyword_col and col_lower in ["keyword", "query", "palabra clave", "consulta"]:
                keyword_col = col
            elif not position_col and col_lower in ["position", "rank", "posicion", "ranking"]:
                position_col = col
            elif not url_col and col_lower in ["url", "landing page", "página", "pagina"]:
                url_col = col
            elif not domain_col and col_lower in ["domain", "dominio", "site"]:
                domain_col = col

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
        if domain_col and domain_col in df.columns:
            result_df["Domain"] = df[domain_col].astype(str).str.strip().apply(normalize_domain)

        # Eliminar filas sin keyword o posición
        result_df = result_df.dropna(subset=["Keyword", "Position"])
        result_df = result_df[result_df["Keyword"].str.len() > 0]

    return result_df


def parse_search_volume_file(uploaded_file: Any) -> pd.DataFrame:
    """
    Parsea archivo de volumen de búsqueda y normaliza columnas.

    Formatos soportados:
    - keyword, search_volume (o variantes)
    - CSV/Excel con columnas de keyword y volumen

    Args:
        uploaded_file: Archivo CSV o Excel subido por Streamlit

    Returns:
        DataFrame con columnas: Keyword, SearchVolume

    Raises:
        ValueError: Si no se puede leer el archivo o faltan columnas
    """
    uploaded_file.seek(0)
    file_extension = uploaded_file.name.split('.')[-1].lower()

    # Leer el archivo
    if file_extension in ['xlsx', 'xls']:
        try:
            df = pd.read_excel(uploaded_file, engine='openpyxl' if file_extension == 'xlsx' else None)
        except Exception as e:
            raise ValueError(f"No se pudo leer el archivo Excel: {e}")
    else:
        # CSV
        delimiters = [',', ';', '\t', '|']
        df = None
        for delimiter in delimiters:
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    uploaded_file.seek(0)
                    temp_df = pd.read_csv(uploaded_file, delimiter=delimiter, encoding=encoding, on_bad_lines='skip')
                    if len(temp_df.columns) >= 2:
                        df = temp_df
                        break
                except:
                    continue
            if df is not None:
                break

        if df is None:
            raise ValueError("No se pudo leer el archivo de volumen de búsqueda")

    if df.empty:
        raise ValueError("El archivo de volumen está vacío")

    # Detectar columnas
    keyword_col = None
    volume_col = None

    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ["keyword", "query", "palabra clave", "consulta", "keywords"]:
            keyword_col = col
        elif any(term in col_lower for term in ["volume", "volumen", "search", "búsqueda", "busqueda", "searches"]):
            volume_col = col

    if not keyword_col or not volume_col:
        raise ValueError(
            f"❌ No se encontraron las columnas requeridas en el archivo de volumen.\n"
            f"📋 Columnas encontradas: {', '.join(df.columns[:10])}\n"
            f"✅ Formato esperado: Keyword/Query + SearchVolume/Volume"
        )

    # Normalizar
    result_df = pd.DataFrame()
    result_df["Keyword"] = df[keyword_col].astype(str).str.strip().str.lower()
    result_df["SearchVolume"] = pd.to_numeric(df[volume_col], errors="coerce").fillna(0).astype(int)

    # Eliminar filas sin keyword
    result_df = result_df[result_df["Keyword"].str.len() > 0]

    return result_df


__all__ = [
    "ParseResult",
    "normalize_domain",
    "parse_position_tracking_csv",
    "parse_search_volume_file",
]
