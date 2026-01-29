"""
Google Search Console API Client
================================

Cliente para conectar con la API de Google Search Console y obtener
datos de rendimiento (posiciones, impresiones, clics, CTR).

Autor: Embedding Insights
Versión: 1.0.0
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Google API imports
try:
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import Flow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False
    Credentials = None
    Request = None
    Flow = None
    build = None
    HttpError = Exception


# ============================================================================
# CONSTANTES
# ============================================================================

GSC_SCOPES = [
    'https://www.googleapis.com/auth/webmasters.readonly',
]

# Límites de la API
MAX_ROWS_PER_REQUEST = 25000
DEFAULT_DAYS_BACK = 28


# ============================================================================
# FUNCIONES DE AUTENTICACIÓN
# ============================================================================

def check_google_api_available() -> Tuple[bool, str]:
    """
    Verifica si las librerías de Google API están disponibles.

    Returns:
        Tupla (disponible, mensaje)
    """
    if not GOOGLE_API_AVAILABLE:
        return False, (
            "Las librerías de Google API no están instaladas. "
            "Ejecuta: pip install google-api-python-client google-auth-oauthlib"
        )
    return True, "Google API disponible"


def create_oauth_flow(
    client_id: str,
    client_secret: str,
    redirect_uri: str = "http://localhost",
) -> Optional[Any]:
    """
    Crea un flujo OAuth para autenticación con Google.

    Args:
        client_id: ID de cliente de Google Cloud
        client_secret: Secreto de cliente
        redirect_uri: URI de redirección (por defecto: modo desktop)

    Returns:
        Objeto Flow de OAuth o None si no está disponible
    """
    if not GOOGLE_API_AVAILABLE:
        return None

    client_config = {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [redirect_uri],
        }
    }

    flow = Flow.from_client_config(
        client_config,
        scopes=GSC_SCOPES,
        redirect_uri=redirect_uri,
    )

    return flow


def get_authorization_url(flow: Any) -> str:
    """
    Obtiene la URL de autorización para el flujo OAuth.

    Args:
        flow: Objeto Flow de OAuth

    Returns:
        URL para autorización
    """
    auth_url, _ = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
        prompt='consent',
    )
    return auth_url


def exchange_code_for_credentials(flow: Any, code: str) -> Optional[Any]:
    """
    Intercambia el código de autorización por credenciales.

    Args:
        flow: Objeto Flow de OAuth
        code: Código de autorización recibido

    Returns:
        Credenciales de OAuth o None si falla
    """
    try:
        flow.fetch_token(code=code)
        return flow.credentials
    except Exception as e:
        print(f"Error intercambiando código: {e}")
        return None


def credentials_from_dict(creds_dict: Dict[str, Any]) -> Optional[Any]:
    """
    Crea credenciales desde un diccionario guardado.

    Args:
        creds_dict: Diccionario con datos de credenciales

    Returns:
        Objeto Credentials o None
    """
    if not GOOGLE_API_AVAILABLE:
        return None

    try:
        creds = Credentials(
            token=creds_dict.get('token'),
            refresh_token=creds_dict.get('refresh_token'),
            token_uri=creds_dict.get('token_uri', 'https://oauth2.googleapis.com/token'),
            client_id=creds_dict.get('client_id'),
            client_secret=creds_dict.get('client_secret'),
            scopes=creds_dict.get('scopes', GSC_SCOPES),
        )

        # Refresh si está expirado
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())

        return creds
    except Exception as e:
        print(f"Error creando credenciales: {e}")
        return None


def credentials_to_dict(credentials: Any) -> Dict[str, Any]:
    """
    Convierte credenciales a diccionario para guardar.

    Args:
        credentials: Objeto Credentials

    Returns:
        Diccionario con datos de credenciales
    """
    return {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': list(credentials.scopes) if credentials.scopes else GSC_SCOPES,
        'expiry': credentials.expiry.isoformat() if credentials.expiry else None,
    }


# ============================================================================
# CLIENTE DE SEARCH CONSOLE
# ============================================================================

class GSCClient:
    """
    Cliente para interactuar con la API de Google Search Console.
    """

    def __init__(self, credentials: Any):
        """
        Args:
            credentials: Credenciales de OAuth válidas
        """
        if not GOOGLE_API_AVAILABLE:
            raise ImportError(
                "google-api-python-client no está instalado. "
                "Ejecuta: pip install google-api-python-client google-auth-oauthlib"
            )

        self.credentials = credentials
        self.service = build('searchconsole', 'v1', credentials=credentials)

    def list_sites(self) -> List[Dict[str, str]]:
        """
        Lista todos los sitios verificados en Search Console.

        Returns:
            Lista de diccionarios con info de cada sitio
        """
        try:
            response = self.service.sites().list().execute()
            sites = response.get('siteEntry', [])
            return [
                {
                    'site_url': site['siteUrl'],
                    'permission_level': site.get('permissionLevel', 'unknown'),
                }
                for site in sites
            ]
        except HttpError as e:
            print(f"Error listando sitios: {e}")
            return []

    def query_search_analytics(
        self,
        site_url: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        dimensions: Optional[List[str]] = None,
        row_limit: int = MAX_ROWS_PER_REQUEST,
        start_row: int = 0,
        search_type: str = 'web',
        aggregation_type: str = 'auto',
        dimension_filter_groups: Optional[List[Dict]] = None,
    ) -> pd.DataFrame:
        """
        Consulta datos de Search Analytics.

        Args:
            site_url: URL del sitio (ej: 'https://example.com/' o 'sc-domain:example.com')
            start_date: Fecha inicio (YYYY-MM-DD). Por defecto: hace 28 días
            end_date: Fecha fin (YYYY-MM-DD). Por defecto: hace 3 días
            dimensions: Lista de dimensiones ['query', 'page', 'country', 'device', 'date']
            row_limit: Máximo de filas a retornar (máx 25000)
            start_row: Fila inicial para paginación
            search_type: Tipo de búsqueda ('web', 'image', 'video', 'news')
            aggregation_type: Tipo de agregación ('auto', 'byPage', 'byProperty')
            dimension_filter_groups: Filtros opcionales

        Returns:
            DataFrame con los datos de Search Analytics
        """
        # Fechas por defecto
        if not end_date:
            end_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=DEFAULT_DAYS_BACK + 3)).strftime('%Y-%m-%d')

        # Dimensiones por defecto
        if dimensions is None:
            dimensions = ['query', 'page']

        # Construir request body
        request_body = {
            'startDate': start_date,
            'endDate': end_date,
            'dimensions': dimensions,
            'rowLimit': min(row_limit, MAX_ROWS_PER_REQUEST),
            'startRow': start_row,
            'searchType': search_type,
            'aggregationType': aggregation_type,
        }

        if dimension_filter_groups:
            request_body['dimensionFilterGroups'] = dimension_filter_groups

        try:
            response = self.service.searchanalytics().query(
                siteUrl=site_url,
                body=request_body,
            ).execute()

            rows = response.get('rows', [])

            if not rows:
                return pd.DataFrame()

            # Convertir a DataFrame
            data = []
            for row in rows:
                row_data = {}
                keys = row.get('keys', [])
                for i, dim in enumerate(dimensions):
                    row_data[dim] = keys[i] if i < len(keys) else None
                row_data['clicks'] = row.get('clicks', 0)
                row_data['impressions'] = row.get('impressions', 0)
                row_data['ctr'] = row.get('ctr', 0) * 100  # Convertir a porcentaje
                row_data['position'] = row.get('position', 0)
                data.append(row_data)

            return pd.DataFrame(data)

        except HttpError as e:
            print(f"Error en query: {e}")
            return pd.DataFrame()

    def get_url_performance(
        self,
        site_url: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        row_limit: int = 5000,
    ) -> pd.DataFrame:
        """
        Obtiene datos de rendimiento agregados por URL.

        Args:
            site_url: URL del sitio
            start_date: Fecha inicio
            end_date: Fecha fin
            row_limit: Máximo de URLs

        Returns:
            DataFrame con columnas: page, clicks, impressions, ctr, position
        """
        return self.query_search_analytics(
            site_url=site_url,
            start_date=start_date,
            end_date=end_date,
            dimensions=['page'],
            row_limit=row_limit,
            aggregation_type='byPage',
        )

    def get_keyword_performance(
        self,
        site_url: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        row_limit: int = 5000,
    ) -> pd.DataFrame:
        """
        Obtiene datos de rendimiento por keyword.

        Args:
            site_url: URL del sitio
            start_date: Fecha inicio
            end_date: Fecha fin
            row_limit: Máximo de keywords

        Returns:
            DataFrame con columnas: query, clicks, impressions, ctr, position
        """
        return self.query_search_analytics(
            site_url=site_url,
            start_date=start_date,
            end_date=end_date,
            dimensions=['query'],
            row_limit=row_limit,
        )

    def get_url_keyword_performance(
        self,
        site_url: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        row_limit: int = 10000,
    ) -> pd.DataFrame:
        """
        Obtiene datos de rendimiento por URL y keyword combinados.

        Args:
            site_url: URL del sitio
            start_date: Fecha inicio
            end_date: Fecha fin
            row_limit: Máximo de filas

        Returns:
            DataFrame con columnas: page, query, clicks, impressions, ctr, position
        """
        return self.query_search_analytics(
            site_url=site_url,
            start_date=start_date,
            end_date=end_date,
            dimensions=['page', 'query'],
            row_limit=row_limit,
        )

    def get_all_url_performance(
        self,
        site_url: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Obtiene TODOS los datos de rendimiento por URL usando paginación.

        Args:
            site_url: URL del sitio
            start_date: Fecha inicio
            end_date: Fecha fin

        Returns:
            DataFrame completo con todas las URLs
        """
        all_data = []
        start_row = 0

        while True:
            df = self.query_search_analytics(
                site_url=site_url,
                start_date=start_date,
                end_date=end_date,
                dimensions=['page'],
                row_limit=MAX_ROWS_PER_REQUEST,
                start_row=start_row,
                aggregation_type='byPage',
            )

            if df.empty:
                break

            all_data.append(df)

            if len(df) < MAX_ROWS_PER_REQUEST:
                break

            start_row += MAX_ROWS_PER_REQUEST

        if not all_data:
            return pd.DataFrame()

        return pd.concat(all_data, ignore_index=True)


# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def normalize_site_url(url: str) -> str:
    """
    Normaliza la URL del sitio para formato de GSC.

    Args:
        url: URL en cualquier formato

    Returns:
        URL normalizada (con trailing slash o formato sc-domain:)
    """
    url = url.strip()

    # Si ya es dominio de propiedad
    if url.startswith('sc-domain:'):
        return url

    # Añadir trailing slash si no tiene
    if not url.endswith('/'):
        url += '/'

    return url


def merge_gsc_with_embeddings(
    embeddings_df: pd.DataFrame,
    gsc_df: pd.DataFrame,
    url_column: str = 'url',
    gsc_url_column: str = 'page',
) -> pd.DataFrame:
    """
    Fusiona datos de GSC con DataFrame de embeddings.

    Args:
        embeddings_df: DataFrame con embeddings y URLs
        gsc_df: DataFrame con datos de GSC
        url_column: Columna de URL en embeddings_df
        gsc_url_column: Columna de URL en gsc_df

    Returns:
        DataFrame fusionado con datos de GSC
    """
    # Normalizar URLs para merge
    embeddings_df = embeddings_df.copy()
    gsc_df = gsc_df.copy()

    # Crear columnas normalizadas
    embeddings_df['_url_norm'] = embeddings_df[url_column].astype(str).str.rstrip('/')
    gsc_df['_url_norm'] = gsc_df[gsc_url_column].astype(str).str.rstrip('/')

    # Merge
    merged = embeddings_df.merge(
        gsc_df[['_url_norm', 'clicks', 'impressions', 'ctr', 'position']],
        on='_url_norm',
        how='left',
    )

    # Limpiar
    merged = merged.drop(columns=['_url_norm'])

    # Rellenar NaN con 0
    for col in ['clicks', 'impressions', 'ctr', 'position']:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    return merged


def identify_opportunity_pages(
    df: pd.DataFrame,
    min_impressions: int = 100,
    max_position: float = 20,
    max_ctr: float = 3.0,
) -> pd.DataFrame:
    """
    Identifica páginas con oportunidad de mejora (alto potencial, bajo rendimiento).

    Criterios:
    - Impresiones altas (visibilidad)
    - Posición media-baja (potencial de subir)
    - CTR bajo (potencial de mejorar clics)

    Args:
        df: DataFrame con datos de GSC
        min_impressions: Mínimo de impresiones para considerar
        max_position: Posición máxima (1-20 típicamente)
        max_ctr: CTR máximo para considerar oportunidad

    Returns:
        DataFrame filtrado con páginas de oportunidad
    """
    mask = (
        (df['impressions'] >= min_impressions) &
        (df['position'] <= max_position) &
        (df['position'] > 1) &  # Ya no está en top 1
        (df['ctr'] <= max_ctr)
    )

    opportunities = df[mask].copy()

    # Score de oportunidad (más impresiones + peor posición = más oportunidad)
    if not opportunities.empty:
        opportunities['opportunity_score'] = (
            (opportunities['impressions'] / opportunities['impressions'].max()) * 0.5 +
            (1 - opportunities['ctr'] / 100) * 0.3 +
            (opportunities['position'] / max_position) * 0.2
        ) * 100

        opportunities = opportunities.sort_values('opportunity_score', ascending=False)

    return opportunities


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Verificación
    'check_google_api_available',
    'GOOGLE_API_AVAILABLE',
    # Auth
    'create_oauth_flow',
    'get_authorization_url',
    'exchange_code_for_credentials',
    'credentials_from_dict',
    'credentials_to_dict',
    'GSC_SCOPES',
    # Cliente
    'GSCClient',
    # Utilidades
    'normalize_site_url',
    'merge_gsc_with_embeddings',
    'identify_opportunity_pages',
]
