"""
Módulo de parsing y normalización para informes de posiciones SEO.

Este módulo contiene funciones para parsear archivos CSV/Excel de rank tracking
y normalizar los datos a un formato estándar.
"""

from __future__ import annotations

import re
import csv
from io import StringIO
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import pandas as pd

if TYPE_CHECKING:
    from typing import Any

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


def parse_position_tracking_csv(uploaded_file: Any) -> pd.DataFrame:
    """
    Parsea un archivo CSV/Excel de rank tracking y lo normaliza.

    Soporta múltiples formatos:
    1. Formato simple: Keyword, Position, URL
    2. Formato SERP: Keyword, Position 1, Position 2, ..., Position 10
    3. Formato Serprobot multi-keyword: Múltiples secciones con headers "Keyword: xxx"

    Args:
        uploaded_file: Archivo CSV o Excel subido por Streamlit

    Returns:
        DataFrame normalizado con columnas: Keyword, Position, Domain, URL

    Raises:
        ValueError: Si el archivo no contiene las columnas requeridas o no se puede parsear
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
                reader = csv.reader(StringIO(line))
                cols = next(reader)
                position_columns_list = [col.strip().strip('"') for col in cols if 'position' in col.lower() and any(c.isdigit() for c in col)]
                header_found = True
                continue

            # Procesar datos - SOLO LA PRIMERA FECHA (más reciente)
            if header_found and current_keyword and position_columns_list and not first_date_processed:
                try:
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
    "normalize_domain",
    "parse_position_tracking_csv",
    "parse_search_volume_file",
]
