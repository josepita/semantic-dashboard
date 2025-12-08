from __future__ import annotations

import re
import time
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

try:  # pragma: no cover - dependencia opcional
    import polars as pl
except Exception:  # pragma: no cover
    pl = None

sns.set_theme(style="whitegrid")

LOG_PATTERN = re.compile(
    r'(?P<ip>\d{1,3}(?:\.\d{1,3}){3}) - - '
    r'\[(?P<timestamp>[^\]]+)\] '
    r'"(?P<method>\w+) (?P<url>[^\s]+) [^"]+" '
    r'(?P<status>\d{3}) '
    r'(?P<size>\d+) '
    r'"(?P<referrer>[^"]*)" '
    r'"(?P<user_agent>[^"]*)"'
)

GOOGLEBOT_UAS = [
    r"Googlebot/\d\.\d",
    r"Googlebot-Image/\d\.\d",
    r"Googlebot-News",
    r"Googlebot-Video/\d\.\d",
    r"AdsBot-Google",
    r"AdsBot-Google-Mobile",
    r"APIs-Google",
    r"Mediapartners-Google",
    r"Storebot-Google",
    r"mobile Safari",
]
GOOGLEBOT_PATTERN = re.compile("|".join(GOOGLEBOT_UAS), re.IGNORECASE)

CMS_CLASSIFICATIONS = {
    "PrestaShop": {
        "Blog": r"(.*/blog/).*",
        "Ficha de producto": r"(.*\.html).*",
        "Home": r"^https?://[^/]+/?$",
        "Marca": r"(.*/brand/).*",
        "Categorias": r".*",
    },
    "WooCommerce": {
        "Blog": r"(.*/blog/).*",
        "Ficha de producto": r"(.*/producto/).*",
        "Home": r"^https?://[^/]+/?$",
        "Categoria de producto": r"(.*/categoria-producto/).*",
        "Otro": r".*",
    },
    "WebLib": {
        "Libro": r"(.*/libro/).*",
        "Producto": r"(.*/producto/).*",
        "Categorias Producto": r"(.*/productos-de/).*",
        "Categorias libros": r"(.*/libros-de/).*",
        "Landings": r"(.*/landings/).*",
        "Noticias": r"(.*/noticias/).*",
        "Especiales": r"(.*/especiales/).*",
        "Home": r"^https?://[^/]+/?$",
        "Otras": r".*",
    },
}

CMS_OPTIONS = list(CMS_CLASSIFICATIONS.keys()) + ["Otro"]


def classify_url_type(url: str) -> str:
    url = url or ""
    if re.search(r"\.(jpg|jpeg|png|gif|svg|webp)$", url, re.IGNORECASE):
        return "Imagen"
    if re.search(r"\.(css)$", url, re.IGNORECASE):
        return "CSS"
    if re.search(r"\.(js)$", url, re.IGNORECASE):
        return "JavaScript"
    if re.search(r"\.(xml|txt)$", url, re.IGNORECASE):
        return "Meta/Sitemap"
    if re.search(r"\.[a-z]{2,5}$", url, re.IGNORECASE):
        return "Otros archivos"
    return "HTML/Documento"


def classify_googlebot(user_agent: str) -> str:
    user_agent = user_agent or ""
    if re.search(r"compatible; Googlebot-Mobile", user_agent, re.IGNORECASE):
        return "Googlebot-Mobile"
    if re.search(r"compatible; Googlebot", user_agent, re.IGNORECASE):
        return "Googlebot-Desktop"
    if re.search(r"AdsBot-Google", user_agent, re.IGNORECASE):
        return "AdsBot"
    if re.search(r"Mediapartners-Google", user_agent, re.IGNORECASE):
        return "Mediapartners"
    return "Otro Googlebot"


def classify_url_page_type(url: str, cms_type: str) -> str:
    if cms_type not in CMS_CLASSIFICATIONS:
        return "Clasificacion pendiente (Otro CMS)"

    patterns = CMS_CLASSIFICATIONS[cms_type]
    if "Home" in patterns and re.match(patterns["Home"], url or ""):
        return "Home"
    for page_type, pattern in patterns.items():
        if page_type == "Home":
            continue
        if re.search(pattern, url or ""):
            return page_type
    return "Categorias" if cms_type == "PrestaShop" else "Otro"


def parse_log_line(line: str) -> Optional[dict]:
    match = LOG_PATTERN.match(line.strip())
    if not match:
        return None
    data = match.groupdict()
    for field in ("status", "size"):
        try:
            data[field] = int(data[field])
        except (ValueError, TypeError):
            data[field] = None
    return data


def load_uploaded_logs(uploaded_files: Sequence) -> Tuple[List[dict], float]:
    all_logs: List[dict] = []
    total_size_mb = 0.0
    for uploaded_file in uploaded_files:
        content = uploaded_file.read()
        total_size_mb += len(content) / (1024 * 1024)
        text = content.decode("utf-8", errors="ignore")
        parsed_lines = [parse_log_line(line) for line in text.splitlines()]
        all_logs.extend([entry for entry in parsed_lines if entry])
        uploaded_file.seek(0)
    return all_logs, total_size_mb


def process_logs_pandas(all_logs_data: Sequence[dict], cms_type: str) -> pd.DataFrame:
    if not all_logs_data:
        return pd.DataFrame()

    df_logs = pd.DataFrame(all_logs_data)
    if df_logs.empty:
        return df_logs

    df_logs["is_googlebot"] = df_logs["user_agent"].apply(
        lambda ua: bool(GOOGLEBOT_PATTERN.search(ua or ""))
    )
    df_logs = df_logs[df_logs["is_googlebot"]].copy()
    if df_logs.empty:
        return df_logs.drop(columns=["is_googlebot"])

    df_logs["datetime"] = pd.to_datetime(
        df_logs["timestamp"].str.split().str[0],
        format="%d/%b/%Y:%H:%M:%S",
        errors="coerce",
    )
    df_logs = df_logs.dropna(subset=["datetime"])
    if df_logs.empty:
        return df_logs

    df_logs["dia"] = df_logs["datetime"].dt.floor("D")
    df_logs["tipo_archivo"] = df_logs["url"].apply(classify_url_type)
    df_logs["tipo_bot"] = df_logs["user_agent"].apply(classify_googlebot)
    df_logs["tipo_pagina"] = df_logs["url"].apply(
        lambda url: classify_url_page_type(url, cms_type)
    )

    return df_logs.drop(columns=["is_googlebot"])


def process_logs_polars(all_logs_data: Sequence[dict], cms_type: str) -> pd.DataFrame:
    if pl is None:
        raise ValueError("Polars no esta disponible en este entorno.")
    if not all_logs_data:
        return pd.DataFrame()

    df_pl = pl.DataFrame(all_logs_data)
    if df_pl.height == 0:
        return pd.DataFrame()

    pattern = GOOGLEBOT_PATTERN.pattern
    df_processed = (
        df_pl.with_columns(
            [
                pl.col("status").cast(pl.Int32, strict=False),
                pl.col("size").cast(pl.Int64, strict=False),
                pl.col("timestamp")
                .str.split(" ")
                .list.get(0)
                .str.strptime(
                    pl.Datetime,
                    format="%d/%b/%Y:%H:%M:%S",
                    strict=False,
                )
                .alias("datetime"),
            ]
        )
        .drop_nulls(subset=["datetime"])
        .filter(pl.col("user_agent").str.contains(pattern, literal=False))
        .with_columns(
            [
                pl.col("datetime").dt.date().alias("dia"),
                pl.col("url").apply(classify_url_type).alias("tipo_archivo"),
                pl.col("user_agent")
                .apply(classify_googlebot)
                .alias("tipo_bot"),
                pl.col("url")
                .apply(lambda u: classify_url_page_type(u, cms_type))
                .alias("tipo_pagina"),
            ]
        )
    )

    return df_processed.to_pandas()


def process_logs(all_logs_data: Sequence[dict], cms_type: str, engine: str) -> pd.DataFrame:
    if engine == "Polars":
        return process_logs_polars(all_logs_data, cms_type)
    return process_logs_pandas(all_logs_data, cms_type)


def build_daily_aggregation(df_logs: pd.DataFrame) -> pd.DataFrame:
    if df_logs.empty or "dia" not in df_logs.columns:
        return pd.DataFrame(columns=["dia", "eventos", "urls_unicas"])
    return (
        df_logs.groupby("dia")
        .agg(eventos=("ip", "count"), urls_unicas=("url", "nunique"))
        .reset_index()
        .sort_values("dia")
    )


def build_type_aggregation(df_logs: pd.DataFrame) -> pd.DataFrame:
    if df_logs.empty or "tipo_archivo" not in df_logs.columns:
        return pd.DataFrame(columns=["tipo_archivo", "conteo"])
    return (
        df_logs.groupby("tipo_archivo")["ip"]
        .count()
        .reset_index(name="conteo")
        .sort_values("conteo", ascending=False)
    )


def render_bar_chart(df_tipo_archivo: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(
        data=df_tipo_archivo,
        x="tipo_archivo",
        y="conteo",
        palette="viridis",
        ax=ax,
    )
    ax.set_title("Tipologia de URLs rastreadas por Googlebot")
    ax.set_xlabel("Tipo de archivo")
    ax.set_ylabel("Eventos")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_line_chart(
    df_daily: pd.DataFrame, y_column: str, title: str, color: str, y_label: str
):
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(
        data=df_daily,
        x="dia",
        y=y_column,
        marker="o",
        ax=ax,
        color=color,
    )
    ax.set_title(title)
    ax.set_xlabel("Dia")
    ax.set_ylabel(y_label)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_logs_dashboard():
    st.header("Analisis de logs de Googlebot")
    st.caption(
        "Sube uno o varios archivos de log (Apache/Nginx) para filtrar Googlebot real, "
        "clasificar URLs y visualizar tendencias."
    )

    available_engines = ["Pandas"]
    if pl is not None:
        available_engines.append("Polars")

    uploaded_files = st.file_uploader(
        "Selecciona uno o varios archivos de log",
        type=["log", "txt", "csv"],
        accept_multiple_files=True,
    )
    engine = st.radio(
        "Motor de procesamiento",
        options=available_engines,
        horizontal=True,
    )
    cms_type = st.selectbox(
        "CMS / estructura del sitio",
        options=CMS_OPTIONS,
        index=0,
    )

    if st.button("Iniciar analisis", type="primary"):
        if not uploaded_files:
            st.warning("Sube al menos un archivo de log para continuar.")
            return

        with st.spinner("Procesando logs..."):
            start_time = time.time()
            all_logs_data, total_size_mb = load_uploaded_logs(uploaded_files)
            if not all_logs_data:
                st.error("No se encontro ninguna linea valida en los archivos subidos.")
                return

            try:
                df_logs = process_logs(all_logs_data, cms_type, engine)
            except ValueError as exc:
                st.error(str(exc))
                return

            processing_seconds = time.time() - start_time
            if df_logs.empty:
                st.warning("Despues de filtrar Googlebot, el DataFrame esta vacio.")
                return

        st.session_state["logs_processed_df"] = df_logs
        st.session_state["logs_total_size_mb"] = total_size_mb
        st.session_state["logs_processing_seconds"] = processing_seconds
        st.success(
            f"Procesadas {len(df_logs):,} lineas en {processing_seconds:.2f} s "
            f"({total_size_mb:.2f} MB) usando {engine}."
        )

    df_logs = st.session_state.get("logs_processed_df")
    if not isinstance(df_logs, pd.DataFrame) or df_logs.empty:
        st.info("Procesa tus logs para desbloquear las visualizaciones.")
        return

    total_size_mb = st.session_state.get("logs_total_size_mb", 0.0)
    processing_seconds = st.session_state.get("logs_processing_seconds", 0.0)

    col_metrics = st.columns(4)
    col_metrics[0].metric("Eventos", f"{len(df_logs):,}")
    col_metrics[1].metric("URLs unicas", f"{df_logs['url'].nunique():,}")
    col_metrics[2].metric("Dias analizados", f"{df_logs['dia'].nunique():,}")
    col_metrics[3].metric("Tiempo (s)", f"{processing_seconds:.2f}")
    st.caption(f"Tamanio total cargado: {total_size_mb:.2f} MB.")

    df_daily = build_daily_aggregation(df_logs)
    df_tipo_archivo = build_type_aggregation(df_logs)

    if not df_tipo_archivo.empty:
        render_bar_chart(df_tipo_archivo)
    if not df_daily.empty:
        render_line_chart(
            df_daily,
            "eventos",
            "Eventos diarios de Googlebot",
            color="#5c6bff",
            y_label="Eventos",
        )
        render_line_chart(
            df_daily,
            "urls_unicas",
            "URLs unicas rastreadas por dia",
            color="#ffa600",
            y_label="URLs unicas",
        )

    col_tables = st.columns(2)
    with col_tables[0]:
        st.subheader("Tipo de pagina")
        st.dataframe(
            df_logs["tipo_pagina"].value_counts().reset_index(
                names=["tipo_pagina", "conteo"]
            ),
            use_container_width=True,
        )
    with col_tables[1]:
        st.subheader("Tipo de bot")
        st.dataframe(
            df_logs["tipo_bot"].value_counts().reset_index(
                names=["tipo_bot", "conteo"]
            ),
            use_container_width=True,
        )

    st.subheader("Vista previa del DataFrame procesado")
    st.dataframe(df_logs.head(200), use_container_width=True)

    csv_export = df_logs.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar datos procesados (CSV)",
        data=csv_export,
        file_name="googlebot_logs.csv",
        mime="text/csv",
    )


__all__ = [
    "render_logs_dashboard",
]
