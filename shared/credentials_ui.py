"""
Credentials UI Components
==========================

Componentes de interfaz de usuario para gestión de credenciales en Streamlit.

Funcionalidades:
- UI para configurar API keys
- Indicadores de estado de autenticación
- Formularios de gestión de credenciales
- Integración con OAuthManager

Autor: Embedding Insights
Versión: 1.0.0
"""

import streamlit as st
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


def render_credentials_config_page(oauth_manager=None):
    """
    Renderiza página completa de configuración de credenciales.

    Args:
        oauth_manager: Instancia de OAuthManager (opcional)
    """
    st.header("🔐 Configuración de Credenciales")

    if not oauth_manager:
        st.warning("⚠️ No hay proyecto activo. Selecciona un proyecto en el sidebar.")
        return

    st.markdown("""
    Configura las credenciales y API keys que usarás en este proyecto.
    Todas las credenciales se guardan **encriptadas** en el proyecto.
    """)

    # Tabs para diferentes tipos de credenciales
    tab_api, tab_oauth, tab_info = st.tabs([
        "🔑 API Keys",
        "🔐 OAuth (Google)",
        "ℹ️ Información"
    ])

    with tab_api:
        render_api_keys_config(oauth_manager)

    with tab_oauth:
        render_oauth_config(oauth_manager)

    with tab_info:
        render_credentials_info()


def render_api_keys_config(oauth_manager):
    """
    Renderiza UI para configurar API keys.

    Args:
        oauth_manager: Instancia de OAuthManager
    """
    st.subheader("🔑 API Keys")

    # Servicios soportados
    services = {
        'openai': {
            'label': 'OpenAI',
            'env': 'OPENAI_API_KEY',
            'help': 'API key para GPT-4, GPT-3.5, embeddings, etc.',
            'placeholder': 'sk-...',
            'get_url': 'https://platform.openai.com/api-keys'
        },
        'gemini': {
            'label': 'Google Gemini',
            'env': 'GEMINI_API_KEY',
            'help': 'API key para Gemini 1.5, Gemini 2.0',
            'placeholder': 'AI...',
            'get_url': 'https://aistudio.google.com/app/apikey'
        },
        'anthropic': {
            'label': 'Anthropic (Claude)',
            'env': 'ANTHROPIC_API_KEY',
            'help': 'API key para Claude 3',
            'placeholder': 'sk-ant-...',
            'get_url': 'https://console.anthropic.com/settings/keys'
        },
        'serprobot': {
            'label': 'Serprobot',
            'env': 'SERPROBOT_API_KEY',
            'help': 'API key para rank tracking',
            'placeholder': 'serprobot_...',
            'get_url': 'https://serprobot.com/api'
        }
    }

    # Mostrar API keys existentes
    existing_keys = oauth_manager.list_api_keys()

    if existing_keys:
        st.success(f"✅ API Keys configuradas: **{', '.join(existing_keys)}**")
    else:
        st.info("ℹ️ No hay API keys configuradas en este proyecto")

    st.markdown("---")

    # Formulario para añadir/actualizar
    with st.form("api_key_form"):
        st.markdown("### Añadir o Actualizar API Key")

        service = st.selectbox(
            "Servicio:",
            options=list(services.keys()),
            format_func=lambda x: services[x]['label']
        )

        service_info = services[service]

        # Mostrar info del servicio
        col1, col2 = st.columns([3, 1])

        with col1:
            api_key = st.text_input(
                f"API Key de {service_info['label']}:",
                type="password",
                placeholder=service_info['placeholder'],
                help=service_info['help']
            )

        with col2:
            st.markdown("&nbsp;")  # Espacio
            st.markdown(f"[Obtener key]({service_info['get_url']})")

        # Botones
        col_save, col_delete, col_test = st.columns(3)

        with col_save:
            submit = st.form_submit_button("💾 Guardar", use_container_width=True)

        with col_delete:
            delete = st.form_submit_button("🗑️ Eliminar", use_container_width=True)

        with col_test:
            test = st.form_submit_button("🧪 Probar", use_container_width=True, disabled=True)

        # Acciones
        if submit and api_key:
            if oauth_manager.save_api_key(service, api_key):
                st.success(f"✅ API key de {service_info['label']} guardada correctamente")
                st.rerun()
            else:
                st.error("❌ Error al guardar API key")

        if delete:
            if oauth_manager.delete_api_key(service):
                st.success(f"✅ API key de {service_info['label']} eliminada")
                st.rerun()
            else:
                st.warning("⚠️ No hay API key para eliminar")

        if test:
            st.info("🧪 Función de test próximamente...")

    # Información adicional
    st.markdown("---")
    with st.expander("💡 Consejos de Seguridad", expanded=False):
        st.markdown("""
        **Seguridad de API Keys:**

        ✅ **Sí hacer:**
        - Usa API keys específicas por proyecto
        - Rota las keys periódicamente
        - Revoca keys que no uses
        - Usa límites de uso en la consola del proveedor

        ❌ **No hacer:**
        - Compartir keys por email o chat
        - Commitear keys en git
        - Usar la misma key en múltiples proyectos
        - Dejar keys activas sin uso

        **Encriptación:**
        - Las API keys se guardan encriptadas con Fernet (AES-128)
        - La clave de encriptación está en `.encryption_key` (gitignored)
        - Sin la clave de encriptación, las API keys son inútiles

        **Fallback:**
        - Si no hay API key en el proyecto, se busca en variables de entorno
        - Útil para desarrollo local sin commitear credenciales
        """)


def render_oauth_config(oauth_manager):
    """
    Renderiza UI para configurar OAuth de Google.

    Args:
        oauth_manager: Instancia de OAuthManager
    """
    st.subheader("🔐 OAuth de Google")

    st.info("""
    **OAuth de Google** permite autenticarte con servicios de Google sin exponer credenciales.

    **Servicios soportados:**
    - Google Search Console (GSC)
    - Google Analytics
    """)

    # Estado actual
    auth_status = oauth_manager.get_auth_status()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Google Search Console")
        if auth_status['gsc']:
            st.success("✅ Autenticado")

            if st.button("🗑️ Desconectar GSC", key="disconnect_gsc"):
                if oauth_manager.delete_gsc_credentials():
                    st.success("✅ Credenciales GSC eliminadas")
                    st.rerun()
        else:
            st.warning("⚠️ No autenticado")
            st.info("La autenticación OAuth de GSC se configurará próximamente")

    with col2:
        st.markdown("### Google Analytics")
        if auth_status.get('analytics', False):
            st.success("✅ Autenticado")

            if st.button("🗑️ Desconectar Analytics", key="disconnect_analytics"):
                if oauth_manager.delete_analytics_credentials():
                    st.success("✅ Credenciales Analytics eliminadas")
                    st.rerun()
        else:
            st.warning("⚠️ No autenticado")
            st.info("La autenticación OAuth de Analytics se configurará próximamente")

    st.markdown("---")

    # Información sobre OAuth
    with st.expander("📚 Cómo funciona OAuth", expanded=False):
        st.markdown("""
        **OAuth 2.0** es un protocolo de autorización que permite a las aplicaciones
        acceder a recursos protegidos sin exponer tu contraseña.

        **Flujo de autenticación:**
        1. Haces clic en "Conectar con Google"
        2. Se abre una ventana de Google para autorizar
        3. Aceptas los permisos solicitados
        4. Google devuelve un token de acceso
        5. El token se guarda en tu proyecto

        **Tokens guardados:**
        - `gsc_token.json` - Token de Search Console
        - `analytics_token.json` - Token de Analytics

        **Seguridad:**
        - Los tokens expiran automáticamente
        - Se renuevan automáticamente cuando es posible
        - Puedes revocar el acceso en cualquier momento
        - Están en `oauth/` que está gitignored

        **Próximos pasos:**
        - Implementación del flujo OAuth completo
        - Renovación automática de tokens
        - Detección de expiración
        """)


def render_credentials_info():
    """Renderiza información general sobre credenciales."""
    st.subheader("ℹ️ Información")

    st.markdown("""
    ## Sistema de Credenciales Multi-Proyecto

    Este sistema permite tener **credenciales diferentes por proyecto**, lo que es ideal
    para agencias que gestionan múltiples clientes.

    ### 📁 Estructura de Archivos

    Cada proyecto tiene su carpeta `oauth/`:

    ```
    workspace/projects/mi-cliente/
    ├── config.json
    ├── database.duckdb
    └── oauth/
        ├── .encryption_key          # Clave de encriptación
        ├── api_keys.encrypted.json  # API keys encriptadas
        ├── gsc_token.json           # Token de GSC (futuro)
        └── analytics_token.json     # Token Analytics (futuro)
    ```

    ### 🔒 Seguridad

    **Encriptación:**
    - API keys: Encriptadas con Fernet (AES-128)
    - OAuth tokens: JSON plano (solo accesibles localmente)
    - Clave de encriptación: Única por proyecto

    **Git:**
    - Todo en `oauth/` está gitignored
    - Nunca se subirán credenciales al repositorio
    - Seguro para colaboración en equipo

    ### 🔄 Auto-Carga de Credenciales

    Al cambiar de proyecto, las credenciales se cargan automáticamente:

    1. Se carga el proyecto seleccionado
    2. Se inicializa el OAuthManager
    3. Se cargan API keys al session_state
    4. Se cargan tokens OAuth si existen
    5. Las apps usan las credenciales del proyecto activo

    ### 🌐 Fallback a Variables de Entorno

    Si no hay API key en el proyecto, se busca en variables de entorno:

    - `OPENAI_API_KEY`
    - `GEMINI_API_KEY`
    - `ANTHROPIC_API_KEY`
    - `SERPROBOT_API_KEY`

    Útil para desarrollo local sin configurar credenciales en cada proyecto.

    ### 📦 Exportación de Proyectos

    **Al exportar un proyecto:**
    - ✅ Se exporta: database.duckdb, embeddings, config
    - ❌ No se exporta: oauth/ (por seguridad)

    **Al importar un proyecto:**
    - Deberás reconfigurar las credenciales
    - Rápido con esta UI

    ### 🚀 Próximas Mejoras

    - [ ] Flujo OAuth completo para GSC y Analytics
    - [ ] Test de API keys (validar que funcionen)
    - [ ] Compartir credenciales entre proyectos (opcional)
    - [ ] Importar credenciales desde .env
    - [ ] Logs de uso de API keys
    """)

    st.markdown("---")
    st.caption("Sistema de Credenciales - Fase 3 implementado en Embedding Insights Suite")


def render_credentials_sidebar_button():
    """
    Renderiza botón en sidebar para ir a configuración de credenciales.

    Returns:
        True si se debe mostrar la página de configuración
    """
    st.sidebar.markdown("---")

    if st.sidebar.button("⚙️ Configurar Credenciales", use_container_width=True):
        return True

    return False


def get_active_api_keys_summary(oauth_manager) -> str:
    """
    Obtiene un resumen de las API keys activas.

    Args:
        oauth_manager: Instancia de OAuthManager

    Returns:
        String con resumen
    """
    if not oauth_manager:
        return "No hay proyecto activo"

    api_keys = oauth_manager.list_api_keys()

    if not api_keys:
        return "Sin API keys"

    return f"{len(api_keys)} API keys: {', '.join(api_keys)}"


# ============================================================================
# Quick Access Components
# ============================================================================

def render_quick_api_key_input(oauth_manager, service: str, label: str = None):
    """
    Renderiza input rápido para una API key específica.

    Args:
        oauth_manager: Instancia de OAuthManager
        service: Nombre del servicio (openai, gemini, etc.)
        label: Label personalizado (opcional)

    Returns:
        API key ingresada o None
    """
    if not label:
        label = f"{service.capitalize()} API Key"

    # Intentar cargar del proyecto
    existing_key = oauth_manager.load_api_key(service) if oauth_manager else None

    if existing_key:
        st.success(f"✅ {label} configurada en el proyecto")
        return existing_key

    # Si no hay, pedir al usuario
    st.warning(f"⚠️ {label} no configurada")

    api_key = st.text_input(
        f"Ingresa tu {label}:",
        type="password",
        key=f"quick_input_{service}"
    )

    if api_key and oauth_manager:
        if st.button(f"💾 Guardar {service}", key=f"save_{service}"):
            if oauth_manager.save_api_key(service, api_key):
                st.success(f"✅ {label} guardada")
                st.rerun()
            else:
                st.error("Error al guardar")

    return api_key


if __name__ == "__main__":
    # Test básico (requiere Streamlit)
    print("Credentials UI Components")
    print("=" * 50)
    print("Este módulo contiene componentes de UI para Streamlit")
    print("Ejecuta desde una app Streamlit para ver los componentes")
