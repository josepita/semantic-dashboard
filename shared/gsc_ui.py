"""
Google Search Console UI Components
====================================

Componentes de Streamlit para conectar y usar datos de GSC.

Autor: Embedding Insights
Versión: 1.0.0
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from .gsc_client import (
    check_google_api_available,
    create_oauth_flow,
    get_authorization_url,
    exchange_code_for_credentials,
    credentials_from_dict,
    credentials_to_dict,
    GSCClient,
    normalize_site_url,
    merge_gsc_with_embeddings,
    identify_opportunity_pages,
    GOOGLE_API_AVAILABLE,
)


# ============================================================================
# CONSTANTES
# ============================================================================

GSC_CREDENTIALS_KEY = "gsc_credentials"
GSC_CLIENT_KEY = "gsc_client"
GSC_SITES_KEY = "gsc_sites"
GSC_DATA_KEY = "gsc_performance_data"
GSC_SELECTED_SITE_KEY = "gsc_selected_site"


# ============================================================================
# FUNCIONES DE ALMACENAMIENTO DE CREDENCIALES
# ============================================================================

def get_gsc_credentials_path() -> str:
    """Obtiene la ruta para guardar credenciales de GSC."""
    return os.path.join(os.path.dirname(__file__), '..', '.gsc_credentials.json')


def save_gsc_credentials(credentials_dict: Dict[str, Any]) -> bool:
    """Guarda las credenciales de GSC en disco."""
    try:
        path = get_gsc_credentials_path()
        with open(path, 'w') as f:
            json.dump(credentials_dict, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error guardando credenciales: {e}")
        return False


def load_gsc_credentials() -> Optional[Dict[str, Any]]:
    """Carga las credenciales de GSC desde disco."""
    try:
        path = get_gsc_credentials_path()
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return None


def delete_gsc_credentials() -> bool:
    """Elimina las credenciales de GSC guardadas."""
    try:
        path = get_gsc_credentials_path()
        if os.path.exists(path):
            os.remove(path)
        # Limpiar session state
        for key in [GSC_CREDENTIALS_KEY, GSC_CLIENT_KEY, GSC_SITES_KEY, GSC_DATA_KEY,
                    'gsc_oauth_flow', '_gsc_flow_creds_key']:
            if key in st.session_state:
                del st.session_state[key]
        return True
    except Exception as e:
        st.error(f"Error eliminando credenciales: {e}")
        return False


# ============================================================================
# COMPONENTES UI
# ============================================================================

def render_gsc_connection_panel() -> Optional[GSCClient]:
    """
    Renderiza el panel de conexión a Google Search Console.

    Returns:
        GSCClient si está conectado, None si no
    """
    st.markdown("### 📊 Conectar Google Search Console")

    # Verificar disponibilidad
    available, message = check_google_api_available()
    if not available:
        st.warning(message)
        st.code("pip install google-api-python-client google-auth-oauthlib")
        return None

    # Verificar si ya hay credenciales guardadas
    saved_creds = load_gsc_credentials()

    if saved_creds and GSC_CLIENT_KEY not in st.session_state:
        # Intentar restaurar cliente desde credenciales guardadas
        try:
            credentials = credentials_from_dict(saved_creds)
            if credentials:
                client = GSCClient(credentials)
                st.session_state[GSC_CLIENT_KEY] = client
                st.session_state[GSC_CREDENTIALS_KEY] = saved_creds
        except Exception as e:
            st.warning(f"Credenciales expiradas, reconecta: {e}")
            delete_gsc_credentials()

    # Si ya está conectado
    if GSC_CLIENT_KEY in st.session_state:
        st.success("✅ Conectado a Google Search Console")

        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔌 Desconectar", key="gsc_disconnect"):
                delete_gsc_credentials()
                st.rerun()

        return st.session_state[GSC_CLIENT_KEY]

    # Mostrar formulario de conexión
    with st.expander("🔑 Configurar credenciales de Google API", expanded=True):
        st.markdown("""
        Para conectar Google Search Console necesitas credenciales OAuth de Google Cloud:

        1. Ve a [Google Cloud Console](https://console.cloud.google.com/)
        2. Crea un proyecto o selecciona uno existente
        3. Activa la **Search Console API**
        4. En "Credenciales", crea un **ID de cliente OAuth 2.0** (tipo: Aplicación de escritorio)
        5. Copia el Client ID y Client Secret aquí
        """)

        client_id = st.text_input(
            "Client ID",
            type="password",
            key="gsc_client_id",
            placeholder="xxxxx.apps.googleusercontent.com",
        )

        client_secret = st.text_input(
            "Client Secret",
            type="password",
            key="gsc_client_secret",
            placeholder="GOCSPX-xxxxx",
        )

        if client_id and client_secret:
            # Crear flujo OAuth (recrear si las credenciales cambiaron)
            flow_credentials_key = f"{client_id}:{client_secret}"
            if (
                'gsc_oauth_flow' not in st.session_state
                or st.session_state.get('_gsc_flow_creds_key') != flow_credentials_key
            ):
                flow = create_oauth_flow(client_id, client_secret)
                st.session_state['gsc_oauth_flow'] = flow
                st.session_state['_gsc_flow_creds_key'] = flow_credentials_key

            flow = st.session_state['gsc_oauth_flow']
            auth_url = get_authorization_url(flow)

            st.markdown("---")
            st.markdown("**Paso 1:** Haz clic en el enlace para autorizar:")
            st.markdown(f"[🔗 Autorizar acceso a Search Console]({auth_url})")

            st.markdown(
                "**Paso 2:** Después de autorizar, el navegador redirigirá a una página "
                "que no cargará (esto es normal). **Copia el código** del parámetro `code=` "
                "de la URL en la barra de direcciones."
            )
            st.caption(
                "Ejemplo: `http://localhost/?code=4/0AfJohXl...&scope=...` → "
                "copia solo `4/0AfJohXl...` (hasta antes de `&scope`)"
            )
            auth_code = st.text_input(
                "Código de autorización",
                key="gsc_auth_code",
                placeholder="4/0AfJohXl...",
            )

            if auth_code and st.button("✅ Conectar", key="gsc_connect_btn"):
                with st.spinner("Conectando..."):
                    credentials = exchange_code_for_credentials(flow, auth_code)

                    if credentials:
                        try:
                            client = GSCClient(credentials)

                            # Guardar en session state
                            creds_dict = credentials_to_dict(credentials)
                            st.session_state[GSC_CLIENT_KEY] = client
                            st.session_state[GSC_CREDENTIALS_KEY] = creds_dict

                            # Guardar en disco
                            save_gsc_credentials(creds_dict)

                            # Limpiar flow temporal
                            del st.session_state['gsc_oauth_flow']

                            st.success("✅ Conectado correctamente!")
                            st.rerun()

                        except Exception as e:
                            st.error(f"Error creando cliente: {e}")
                    else:
                        st.error("No se pudo obtener las credenciales. Verifica el código.")

    return None


def render_gsc_site_selector(client: GSCClient) -> Optional[str]:
    """
    Renderiza el selector de sitio de GSC.

    Args:
        client: Cliente de GSC conectado

    Returns:
        URL del sitio seleccionado o None
    """
    # Cargar lista de sitios si no existe
    if GSC_SITES_KEY not in st.session_state:
        with st.spinner("Cargando sitios..."):
            sites = client.list_sites()
            st.session_state[GSC_SITES_KEY] = sites

    sites = st.session_state[GSC_SITES_KEY]

    if not sites:
        st.warning("No se encontraron sitios. Verifica que tu cuenta tenga acceso a alguna propiedad en Search Console.")
        return None

    # Crear opciones para el selector
    site_options = {site['site_url']: f"{site['site_url']} ({site['permission_level']})" for site in sites}

    selected_site = st.selectbox(
        "Selecciona el sitio",
        options=list(site_options.keys()),
        format_func=lambda x: site_options[x],
        key="gsc_site_selector",
    )

    st.session_state[GSC_SELECTED_SITE_KEY] = selected_site

    return selected_site


def render_gsc_data_loader(client: GSCClient, site_url: str) -> Optional[pd.DataFrame]:
    """
    Renderiza el cargador de datos de GSC.

    Args:
        client: Cliente de GSC
        site_url: URL del sitio

    Returns:
        DataFrame con datos de rendimiento o None
    """
    st.markdown("#### 📈 Cargar datos de rendimiento")

    col1, col2 = st.columns(2)

    with col1:
        days_back = st.slider(
            "Días de datos",
            min_value=7,
            max_value=90,
            value=28,
            key="gsc_days_back",
            help="Cuántos días de histórico cargar",
        )

    with col2:
        data_type = st.selectbox(
            "Tipo de datos",
            options=["Por URL", "Por Keyword", "URL + Keyword"],
            key="gsc_data_type",
        )

    if st.button("📥 Cargar datos de GSC", key="gsc_load_data"):
        end_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back + 3)).strftime('%Y-%m-%d')

        with st.spinner(f"Cargando datos de {site_url}..."):
            try:
                if data_type == "Por URL":
                    df = client.get_url_performance(
                        site_url=site_url,
                        start_date=start_date,
                        end_date=end_date,
                        row_limit=10000,
                    )
                elif data_type == "Por Keyword":
                    df = client.get_keyword_performance(
                        site_url=site_url,
                        start_date=start_date,
                        end_date=end_date,
                        row_limit=10000,
                    )
                else:  # URL + Keyword
                    df = client.get_url_keyword_performance(
                        site_url=site_url,
                        start_date=start_date,
                        end_date=end_date,
                        row_limit=25000,
                    )

                if df.empty:
                    st.warning("No se encontraron datos para el período seleccionado.")
                    return None

                st.session_state[GSC_DATA_KEY] = df
                st.success(f"✅ Cargados {len(df):,} registros")

            except Exception as e:
                st.error(f"Error cargando datos: {e}")
                return None

    # Mostrar datos si existen
    if GSC_DATA_KEY in st.session_state:
        df = st.session_state[GSC_DATA_KEY]

        # Métricas resumen
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Clics", f"{df['clicks'].sum():,}")
        with col2:
            st.metric("Total Impresiones", f"{df['impressions'].sum():,}")
        with col3:
            avg_ctr = df['ctr'].mean() if len(df) > 0 else 0
            st.metric("CTR Medio", f"{avg_ctr:.2f}%")
        with col4:
            avg_pos = df['position'].mean() if len(df) > 0 else 0
            st.metric("Posición Media", f"{avg_pos:.1f}")

        # Preview de datos
        with st.expander("Ver datos cargados", expanded=False):
            st.dataframe(
                df.sort_values('impressions', ascending=False).head(100),
                use_container_width=True,
            )

        return df

    return None


def render_gsc_linking_integration(
    gsc_df: pd.DataFrame,
    embeddings_df: pd.DataFrame,
    url_column: str,
) -> Optional[pd.DataFrame]:
    """
    Renderiza la integración de datos GSC con el dataset de embeddings.

    Args:
        gsc_df: DataFrame con datos de GSC
        embeddings_df: DataFrame con embeddings
        url_column: Columna de URL en embeddings_df

    Returns:
        DataFrame enriquecido con datos de GSC
    """
    st.markdown("#### 🔗 Integrar datos de GSC con embeddings")

    # Detectar columna de URL en GSC
    gsc_url_col = 'page' if 'page' in gsc_df.columns else gsc_df.columns[0]

    st.info(f"Se fusionarán {len(gsc_df):,} registros de GSC con {len(embeddings_df):,} URLs del dataset.")

    if st.button("🔄 Fusionar datos", key="gsc_merge_data"):
        with st.spinner("Fusionando datos..."):
            merged_df = merge_gsc_with_embeddings(
                embeddings_df=embeddings_df,
                gsc_df=gsc_df,
                url_column=url_column,
                gsc_url_column=gsc_url_col,
            )

            # Contar matches
            matched = (merged_df['impressions'] > 0).sum()
            total = len(merged_df)

            st.success(f"✅ Fusión completada: {matched:,}/{total:,} URLs con datos de GSC ({matched/total*100:.1f}%)")

            # Guardar en session state
            st.session_state['gsc_merged_df'] = merged_df

            return merged_df

    if 'gsc_merged_df' in st.session_state:
        return st.session_state['gsc_merged_df']

    return None


def render_opportunity_pages(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Renderiza análisis de páginas con oportunidad.

    Args:
        df: DataFrame con datos de GSC

    Returns:
        DataFrame con páginas de oportunidad
    """
    st.markdown("#### 🎯 Páginas con oportunidad de mejora")
    st.caption("Páginas con alta visibilidad pero bajo CTR - ideales para recibir más enlaces internos")

    col1, col2, col3 = st.columns(3)

    with col1:
        min_impressions = st.number_input(
            "Mínimo impresiones",
            min_value=10,
            max_value=10000,
            value=100,
            key="gsc_min_impressions",
        )

    with col2:
        max_position = st.number_input(
            "Posición máxima",
            min_value=1,
            max_value=100,
            value=20,
            key="gsc_max_position",
        )

    with col3:
        max_ctr = st.number_input(
            "CTR máximo (%)",
            min_value=0.1,
            max_value=20.0,
            value=3.0,
            key="gsc_max_ctr",
        )

    if st.button("🔍 Identificar oportunidades", key="gsc_find_opportunities"):
        opportunities = identify_opportunity_pages(
            df=df,
            min_impressions=int(min_impressions),
            max_position=float(max_position),
            max_ctr=float(max_ctr),
        )

        if opportunities.empty:
            st.info("No se encontraron páginas que cumplan los criterios.")
            return None

        st.success(f"✅ Encontradas {len(opportunities):,} páginas con oportunidad")

        st.session_state['gsc_opportunities'] = opportunities

        # Mostrar tabla
        display_cols = ['page', 'impressions', 'clicks', 'ctr', 'position', 'opportunity_score']
        display_cols = [c for c in display_cols if c in opportunities.columns]

        st.dataframe(
            opportunities[display_cols].head(50),
            use_container_width=True,
        )

        return opportunities

    if 'gsc_opportunities' in st.session_state:
        return st.session_state['gsc_opportunities']

    return None


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'render_gsc_connection_panel',
    'render_gsc_site_selector',
    'render_gsc_data_loader',
    'render_gsc_linking_integration',
    'render_opportunity_pages',
    'GSC_DATA_KEY',
    'GSC_SELECTED_SITE_KEY',
]
