"""
Parsing de CSVs exportados por el bookmarklet ChatGPT Conversations Analyzer.

El bookmarklet (de Jean-Christophe Chouinard) genera varios CSVs al pulsar
"Export Selected". El más relevante es Queries_Report.csv con formato:
    User Query, Search Queries
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def parse_chatgpt_fanout_csv(uploaded_file: Any) -> pd.DataFrame:
    """Parsea el CSV de queries exportado por el bookmarklet de ChatGPT.

    Soporta dos formatos:
    1. Queries_Report.csv: columnas "User Query", "Search Queries"
    2. Formato genérico: 2 columnas (prompt, query)

    Returns:
        DataFrame con columnas [prompt, query_index, web_search_query, source]
    """
    uploaded_file.seek(0)

    df = None
    for enc in ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']:
        for delim in [',', ';', '\t']:
            try:
                uploaded_file.seek(0)
                temp = pd.read_csv(
                    uploaded_file, delimiter=delim, encoding=enc, on_bad_lines='skip'
                )
                if len(temp.columns) >= 2:
                    df = temp
                    break
            except Exception:
                continue
        if df is not None:
            break

    if df is None:
        raise ValueError("No se pudo leer el CSV de ChatGPT. Verifica el formato.")

    cols_lower = {str(c).lower().strip(): c for c in df.columns}

    # Detectar columna de prompt (User Query)
    prompt_col = None
    for candidate in ["user query", "prompt", "user_query", "query"]:
        if candidate in cols_lower:
            prompt_col = cols_lower[candidate]
            break
    if not prompt_col:
        prompt_col = df.columns[0]

    # Detectar columna de search query
    query_col = None
    for candidate in ["search queries", "search_queries", "web_search_query", "queries"]:
        if candidate in cols_lower:
            query_col = cols_lower[candidate]
            break
    if not query_col:
        query_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    # Construir DataFrame normalizado
    rows = []
    for _, row in df.iterrows():
        prompt = str(row.get(prompt_col, "")).strip()
        query = str(row.get(query_col, "")).strip()
        if not prompt or prompt == "nan":
            continue
        if not query or query == "nan":
            continue
        rows.append({
            "prompt": prompt,
            "web_search_query": query,
            "source": "chatgpt",
        })

    if not rows:
        raise ValueError("No se encontraron datos válidos en el CSV de ChatGPT.")

    result_df = pd.DataFrame(rows)

    # Añadir query_index por prompt
    result_df["query_index"] = result_df.groupby("prompt").cumcount()

    return result_df[["prompt", "query_index", "web_search_query", "source"]]


__all__ = ["parse_chatgpt_fanout_csv"]
