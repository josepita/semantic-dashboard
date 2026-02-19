"""
OAuth & Credentials Manager
============================

Gestión centralizada de credenciales OAuth y API keys por proyecto.

Características:
- Almacenamiento seguro de credenciales OAuth (GSC, Analytics)
- Encriptación de API keys con Fernet
- Auto-carga de credenciales al cambiar proyecto
- Verificación de estado de autenticación
- Fallback a variables de entorno

Autor: Embedding Insights
Versión: 1.0.0
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging
try:
    from shared.env_utils import get_env_value
except ModuleNotFoundError:
    from env_utils import get_env_value

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cryptography para encriptación de API keys
try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    logger.warning("cryptography no está instalado. API keys se guardarán sin encriptar.")
    CRYPTO_AVAILABLE = False

# Google OAuth
try:
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    GOOGLE_AUTH_AVAILABLE = True
except ImportError:
    logger.warning("google-auth no está instalado. OAuth de Google no disponible.")
    GOOGLE_AUTH_AVAILABLE = False
    Credentials = None
    Request = None


class OAuthManager:
    """
    Gestor de credenciales OAuth y API keys por proyecto.

    Almacena:
    - Tokens OAuth de Google (GSC, Analytics)
    - API keys encriptadas (OpenAI, Gemini, etc.)
    - Estado de autenticación

    Estructura:
    workspace/projects/[name]/oauth/
    ├── gsc_token.json           # Credenciales GSC
    ├── analytics_token.json     # Credenciales Analytics
    ├── api_keys.encrypted.json  # API keys encriptadas
    └── .encryption_key          # Clave de encriptación (gitignored)
    """

    def __init__(self, project_path: str):
        """
        Inicializa el OAuthManager para un proyecto.

        Args:
            project_path: Ruta al directorio del proyecto
        """
        self.project_path = Path(project_path)
        self.oauth_dir = self.project_path / "oauth"

        # Crear directorio oauth si no existe
        self.oauth_dir.mkdir(parents=True, exist_ok=True)

        # Rutas de archivos
        self.gsc_token_path = self.oauth_dir / "gsc_token.json"
        self.analytics_token_path = self.oauth_dir / "analytics_token.json"
        self.api_keys_path = self.oauth_dir / "api_keys.encrypted.json"
        self.encryption_key_path = self.oauth_dir / ".encryption_key"

        # Inicializar encriptación si está disponible
        self.fernet = None
        if CRYPTO_AVAILABLE:
            self.fernet = self._get_or_create_fernet()

        logger.info(f"OAuthManager inicializado para proyecto: {project_path}")

    # ========================================================================
    # Google OAuth - GSC (Search Console)
    # ========================================================================

    def save_gsc_credentials(self, credentials: Any) -> bool:
        """
        Guarda credenciales de Google Search Console.

        Args:
            credentials: Objeto Credentials de google.oauth2

        Returns:
            True si se guardó correctamente
        """
        if not GOOGLE_AUTH_AVAILABLE:
            logger.error("google-auth no está instalado")
            return False

        try:
            # Convertir credenciales a dict
            creds_dict = {
                'token': credentials.token,
                'refresh_token': credentials.refresh_token,
                'token_uri': credentials.token_uri,
                'client_id': credentials.client_id,
                'client_secret': credentials.client_secret,
                'scopes': credentials.scopes,
                'expiry': credentials.expiry.isoformat() if credentials.expiry else None
            }

            # Guardar como JSON
            with open(self.gsc_token_path, 'w') as f:
                json.dump(creds_dict, f, indent=2)

            logger.info("Credenciales GSC guardadas correctamente")
            return True

        except Exception as e:
            logger.error(f"Error al guardar credenciales GSC: {e}")
            return False

    def load_gsc_credentials(self) -> Optional[Any]:
        """
        Carga credenciales de Google Search Console.

        Returns:
            Objeto Credentials o None si no existen
        """
        if not GOOGLE_AUTH_AVAILABLE:
            logger.error("google-auth no está instalado")
            return None

        if not self.gsc_token_path.exists():
            logger.info("No hay credenciales GSC guardadas")
            return None

        try:
            # Leer JSON
            with open(self.gsc_token_path, 'r') as f:
                creds_dict = json.load(f)

            # Reconstruir objeto Credentials
            credentials = Credentials(
                token=creds_dict.get('token'),
                refresh_token=creds_dict.get('refresh_token'),
                token_uri=creds_dict.get('token_uri'),
                client_id=creds_dict.get('client_id'),
                client_secret=creds_dict.get('client_secret'),
                scopes=creds_dict.get('scopes')
            )

            # Verificar si necesita refresh
            if credentials.expired and credentials.refresh_token:
                logger.info("Token expirado, intentando refresh...")
                credentials.refresh(Request())
                # Guardar credenciales actualizadas
                self.save_gsc_credentials(credentials)

            logger.info("Credenciales GSC cargadas correctamente")
            return credentials

        except Exception as e:
            logger.error(f"Error al cargar credenciales GSC: {e}")
            return None

    def delete_gsc_credentials(self) -> bool:
        """Elimina credenciales de GSC."""
        try:
            if self.gsc_token_path.exists():
                self.gsc_token_path.unlink()
                logger.info("Credenciales GSC eliminadas")
                return True
            return False
        except Exception as e:
            logger.error(f"Error al eliminar credenciales GSC: {e}")
            return False

    # ========================================================================
    # Google OAuth - Analytics
    # ========================================================================

    def save_analytics_credentials(self, credentials: Any) -> bool:
        """
        Guarda credenciales de Google Analytics.

        Args:
            credentials: Objeto Credentials de google.oauth2

        Returns:
            True si se guardó correctamente
        """
        if not GOOGLE_AUTH_AVAILABLE:
            logger.error("google-auth no está instalado")
            return False

        try:
            # Convertir credenciales a dict
            creds_dict = {
                'token': credentials.token,
                'refresh_token': credentials.refresh_token,
                'token_uri': credentials.token_uri,
                'client_id': credentials.client_id,
                'client_secret': credentials.client_secret,
                'scopes': credentials.scopes,
                'expiry': credentials.expiry.isoformat() if credentials.expiry else None
            }

            # Guardar como JSON
            with open(self.analytics_token_path, 'w') as f:
                json.dump(creds_dict, f, indent=2)

            logger.info("Credenciales Analytics guardadas correctamente")
            return True

        except Exception as e:
            logger.error(f"Error al guardar credenciales Analytics: {e}")
            return False

    def load_analytics_credentials(self) -> Optional[Any]:
        """
        Carga credenciales de Google Analytics.

        Returns:
            Objeto Credentials o None si no existen
        """
        if not GOOGLE_AUTH_AVAILABLE:
            logger.error("google-auth no está instalado")
            return None

        if not self.analytics_token_path.exists():
            logger.info("No hay credenciales Analytics guardadas")
            return None

        try:
            # Leer JSON
            with open(self.analytics_token_path, 'r') as f:
                creds_dict = json.load(f)

            # Reconstruir objeto Credentials
            credentials = Credentials(
                token=creds_dict.get('token'),
                refresh_token=creds_dict.get('refresh_token'),
                token_uri=creds_dict.get('token_uri'),
                client_id=creds_dict.get('client_id'),
                client_secret=creds_dict.get('client_secret'),
                scopes=creds_dict.get('scopes')
            )

            # Verificar si necesita refresh
            if credentials.expired and credentials.refresh_token:
                logger.info("Token expirado, intentando refresh...")
                credentials.refresh(Request())
                # Guardar credenciales actualizadas
                self.save_analytics_credentials(credentials)

            logger.info("Credenciales Analytics cargadas correctamente")
            return credentials

        except Exception as e:
            logger.error(f"Error al cargar credenciales Analytics: {e}")
            return None

    def delete_analytics_credentials(self) -> bool:
        """Elimina credenciales de Analytics."""
        try:
            if self.analytics_token_path.exists():
                self.analytics_token_path.unlink()
                logger.info("Credenciales Analytics eliminadas")
                return True
            return False
        except Exception as e:
            logger.error(f"Error al eliminar credenciales Analytics: {e}")
            return False

    # ========================================================================
    # API Keys (Encriptadas)
    # ========================================================================

    def save_api_key(self, service: str, api_key: str) -> bool:
        """
        Guarda una API key encriptada.

        Args:
            service: Nombre del servicio (openai, gemini, serprobot, etc.)
            api_key: API key en texto plano

        Returns:
            True si se guardó correctamente
        """
        try:
            # Cargar API keys existentes
            api_keys = self._load_api_keys_file()

            # Encriptar si está disponible
            if self.fernet and CRYPTO_AVAILABLE:
                encrypted = self.fernet.encrypt(api_key.encode()).decode()
                api_keys[service] = {
                    'encrypted': True,
                    'value': encrypted,
                    'updated_at': datetime.now().isoformat()
                }
            else:
                # Guardar sin encriptar (warning ya emitido)
                api_keys[service] = {
                    'encrypted': False,
                    'value': api_key,
                    'updated_at': datetime.now().isoformat()
                }

            # Guardar archivo
            with open(self.api_keys_path, 'w') as f:
                json.dump(api_keys, f, indent=2)

            logger.info(f"API key de {service} guardada correctamente")
            return True

        except Exception as e:
            logger.error(f"Error al guardar API key de {service}: {e}")
            return False

    def load_api_key(self, service: str, fallback_env: Optional[str] = None) -> Optional[str]:
        """
        Carga una API key desencriptada.

        Args:
            service: Nombre del servicio
            fallback_env: Variable de entorno de fallback

        Returns:
            API key en texto plano o None
        """
        try:
            # Intentar cargar del proyecto
            api_keys = self._load_api_keys_file()

            if service in api_keys:
                key_data = api_keys[service]

                # Desencriptar si está encriptada
                if key_data.get('encrypted', False):
                    if self.fernet and CRYPTO_AVAILABLE:
                        decrypted = self.fernet.decrypt(key_data['value'].encode()).decode()
                        return decrypted
                    else:
                        logger.error(f"API key de {service} está encriptada pero cryptography no disponible")
                        return None
                else:
                    # Devolver sin encriptar
                    return key_data['value']

            # Fallback a variable de entorno
            if fallback_env:
                env_value = get_env_value(fallback_env)
                if env_value:
                    logger.info(f"API key de {service} cargada desde {fallback_env}")
                    return env_value

            logger.info(f"No hay API key para {service}")
            return None

        except Exception as e:
            logger.error(f"Error al cargar API key de {service}: {e}")
            return None

    def delete_api_key(self, service: str) -> bool:
        """Elimina una API key."""
        try:
            api_keys = self._load_api_keys_file()
            if service in api_keys:
                del api_keys[service]
                with open(self.api_keys_path, 'w') as f:
                    json.dump(api_keys, f, indent=2)
                logger.info(f"API key de {service} eliminada")
                return True
            return False
        except Exception as e:
            logger.error(f"Error al eliminar API key de {service}: {e}")
            return False

    def list_api_keys(self) -> List[str]:
        """
        Lista los servicios con API keys guardadas.

        Returns:
            Lista de nombres de servicios
        """
        try:
            api_keys = self._load_api_keys_file()
            return list(api_keys.keys())
        except Exception as e:
            logger.error(f"Error al listar API keys: {e}")
            return []

    # ========================================================================
    # Estado de Autenticación
    # ========================================================================

    def is_authenticated(self, service: str) -> bool:
        """
        Verifica si un servicio está autenticado.

        Args:
            service: Nombre del servicio (gsc, analytics, openai, gemini, etc.)

        Returns:
            True si hay credenciales válidas
        """
        if service == 'gsc':
            return self.gsc_token_path.exists()
        elif service == 'analytics':
            return self.analytics_token_path.exists()
        else:
            # API key service
            api_keys = self._load_api_keys_file()
            return service in api_keys

    def get_auth_status(self) -> Dict[str, bool]:
        """
        Obtiene el estado de autenticación de todos los servicios.

        Returns:
            Dict con estado de cada servicio
        """
        return {
            'gsc': self.is_authenticated('gsc'),
            'analytics': self.is_authenticated('analytics'),
            'api_keys': self.list_api_keys()
        }

    # ========================================================================
    # Helpers Internos
    # ========================================================================

    def _get_or_create_fernet(self) -> Optional[Any]:
        """
        Obtiene o crea una instancia de Fernet para encriptación.

        Returns:
            Fernet instance o None
        """
        if not CRYPTO_AVAILABLE:
            return None

        try:
            if self.encryption_key_path.exists():
                # Cargar clave existente
                with open(self.encryption_key_path, 'rb') as f:
                    key = f.read()
            else:
                # Generar nueva clave
                key = Fernet.generate_key()
                with open(self.encryption_key_path, 'wb') as f:
                    f.write(key)
                logger.info("Nueva clave de encriptación generada")

            return Fernet(key)

        except Exception as e:
            logger.error(f"Error al inicializar Fernet: {e}")
            return None

    def _load_api_keys_file(self) -> Dict[str, Any]:
        """
        Carga el archivo de API keys.

        Returns:
            Dict con API keys o dict vacío
        """
        if not self.api_keys_path.exists():
            return {}

        try:
            with open(self.api_keys_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error al leer API keys: {e}")
            return {}


# ============================================================================
# Helper Functions
# ============================================================================

def get_oauth_manager(project_config: Optional[Dict[str, Any]] = None) -> Optional[OAuthManager]:
    """
    Obtiene una instancia de OAuthManager para el proyecto actual.

    Args:
        project_config: Configuración del proyecto desde st.session_state

    Returns:
        OAuthManager instance o None
    """
    if not project_config:
        logger.warning("No hay project_config disponible")
        return None

    project_path = project_config.get('path')
    if not project_path:
        logger.error("project_config no tiene 'path'")
        return None

    return OAuthManager(project_path)


def ensure_oauth_gitignore(workspace_path: str = "workspace") -> bool:
    """
    Asegura que oauth/ esté en .gitignore.

    Args:
        workspace_path: Ruta al workspace

    Returns:
        True si se actualizó .gitignore
    """
    try:
        gitignore_path = Path(workspace_path).parent / ".gitignore"

        # Leer .gitignore existente
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                content = f.read()
        else:
            content = ""

        # Patrones a añadir
        patterns = [
            "# OAuth credentials (per-project)",
            "workspace/projects/*/oauth/",
            "workspace/projects/*/.encryption_key"
        ]

        # Verificar si ya están
        needs_update = False
        for pattern in patterns:
            if pattern not in content:
                needs_update = True
                break

        if needs_update:
            # Añadir al final
            with open(gitignore_path, 'a') as f:
                f.write("\n\n")
                for pattern in patterns:
                    if pattern not in content:
                        f.write(f"{pattern}\n")

            logger.info(".gitignore actualizado con patrones de oauth/")
            return True

        return False

    except Exception as e:
        logger.error(f"Error al actualizar .gitignore: {e}")
        return False


# ============================================================================
# Streamlit Integration Helpers
# ============================================================================

def render_oauth_status_indicator(oauth_manager: OAuthManager, service: str) -> None:
    """
    Renderiza un indicador visual de estado de autenticación.

    Args:
        oauth_manager: Instancia de OAuthManager
        service: Nombre del servicio
    """
    try:
        import streamlit as st

        is_auth = oauth_manager.is_authenticated(service)

        if is_auth:
            st.success(f"✅ {service.upper()} autenticado")
        else:
            st.warning(f"⚠️ {service.upper()} no autenticado")

    except ImportError:
        logger.warning("Streamlit no disponible para render_oauth_status_indicator")


def render_api_key_config_ui(oauth_manager: OAuthManager) -> None:
    """
    Renderiza UI para configurar API keys.

    Args:
        oauth_manager: Instancia de OAuthManager
    """
    try:
        import streamlit as st

        st.subheader("🔑 Configuración de API Keys")

        # Servicios soportados
        services = {
            'openai': {'label': 'OpenAI', 'env': 'OPENAI_API_KEY'},
            'gemini': {'label': 'Google Gemini', 'env': 'GEMINI_API_KEY'},
            'serprobot': {'label': 'Serprobot', 'env': 'SERPROBOT_API_KEY'},
            'anthropic': {'label': 'Anthropic (Claude)', 'env': 'ANTHROPIC_API_KEY'}
        }

        # Mostrar API keys existentes
        existing_keys = oauth_manager.list_api_keys()
        if existing_keys:
            st.info(f"API Keys configuradas: {', '.join(existing_keys)}")

        # Formulario para añadir/actualizar
        with st.form("api_key_form"):
            service = st.selectbox(
                "Servicio:",
                options=list(services.keys()),
                format_func=lambda x: services[x]['label']
            )

            current_api_key = oauth_manager.load_api_key(service, services[service]["env"]) or ""

            api_key = st.text_input(
                "API Key:",
                type="password",
                value=current_api_key,
                placeholder="sk-..."
            )

            col1, col2 = st.columns(2)

            with col1:
                submit = st.form_submit_button("💾 Guardar")

            with col2:
                delete = st.form_submit_button("🗑️ Eliminar")

            if submit and api_key:
                if oauth_manager.save_api_key(service, api_key):
                    st.success(f"✅ API key de {services[service]['label']} guardada")
                else:
                    st.error("Error al guardar API key")

            if delete:
                if oauth_manager.delete_api_key(service):
                    st.success(f"✅ API key de {services[service]['label']} eliminada")
                else:
                    st.warning("No hay API key para eliminar")

    except ImportError:
        logger.warning("Streamlit no disponible para render_api_key_config_ui")


if __name__ == "__main__":
    # Test básico
    print("OAuthManager - Test")
    print("=" * 50)

    # Crear manager de prueba
    test_project = "workspace/projects/test-oauth"
    Path(test_project).mkdir(parents=True, exist_ok=True)

    manager = OAuthManager(test_project)

    # Test API keys
    print("\nTest: Guardar API key")
    manager.save_api_key('openai', 'YOUR_API_KEY_HERE')

    print("\nTest: Cargar API key")
    key = manager.load_api_key('openai')
    print(f"API key cargada: {key}")

    print("\nTest: Listar API keys")
    keys = manager.list_api_keys()
    print(f"API keys disponibles: {keys}")

    print("\nTest: Estado de autenticación")
    status = manager.get_auth_status()
    print(f"Estado: {status}")

    print("\n✅ Tests completados")
