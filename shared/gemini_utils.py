"""
Utilidades para trabajar con Gemini AI.

Este módulo centraliza todas las funciones relacionadas con la API de Gemini
que se usan en múltiples módulos de las aplicaciones.
"""

import os
import streamlit as st
from typing import Optional


def get_gemini_api_key() -> str:
    """
    Obtiene la API key de Gemini solo desde session_state.

    La API key NO se persiste y debe introducirse en cada sesión.

    Busca en orden:
    1. st.session_state["gemini_api_key"]
    2. st.session_state["positions_gemini_key"]

    Returns:
        API key de Gemini o string vacío si no se encuentra
    """
    candidates = [
        st.session_state.get("gemini_api_key"),
        st.session_state.get("positions_gemini_key"),
    ]

    for candidate in candidates:
        if candidate:
            return candidate.strip()

    return ""


def get_gemini_model(default: str = "gemini-3-flash-preview") -> str:
    """
    Obtiene el modelo de Gemini configurado.
    
    Busca en orden:
    1. st.session_state["gemini_model_name"]
    2. st.session_state["gemini_model"]
    3. st.session_state["positions_gemini_model"]
    4. Variable de entorno GEMINI_MODEL
    5. Valor por defecto
    
    Args:
        default: Modelo por defecto si no se encuentra ninguno configurado
        
    Returns:
        Nombre del modelo de Gemini
    """
    candidate = (
        st.session_state.get("gemini_model_name")
        or st.session_state.get("gemini_model")
        or st.session_state.get("positions_gemini_model")
        or os.environ.get("GEMINI_MODEL")
        or default
    )
    
    return candidate.strip()


def configure_gemini(api_key: Optional[str] = None) -> bool:
    """
    Configura la API de Gemini con la key proporcionada o detectada.
    
    Args:
        api_key: API key opcional. Si no se proporciona, se busca automáticamente
        
    Returns:
        True si se configuró correctamente, False en caso contrario
    """
    try:
        import google.generativeai as genai
    except ImportError:
        st.error("google-generativeai no está instalado. Instálalo con: pip install google-generativeai")
        return False
    
    key = api_key or get_gemini_api_key()
    
    if not key:
        st.warning("No se encontró una API key de Gemini. Configúrala en las variables de entorno o en la interfaz.")
        return False
    
    try:
        genai.configure(api_key=key)
        return True
    except Exception as e:
        st.error(f"Error al configurar Gemini: {e}")
        return False


def get_gemini_client():
    """
    Obtiene el cliente de Gemini configurado.
    
    Returns:
        Módulo google.generativeai configurado o None si no está disponible
    """
    try:
        import google.generativeai as genai
        
        # Configurar si hay API key disponible
        api_key = get_gemini_api_key()
        if api_key:
            genai.configure(api_key=api_key)
        
        return genai
    except ImportError:
        return None


def is_gemini_available() -> bool:
    """
    Verifica si Gemini está disponible y configurado.
    
    Returns:
        True si Gemini está disponible, False en caso contrario
    """
    try:
        import google.generativeai as genai
        api_key = get_gemini_api_key()
        return bool(api_key)
    except ImportError:
        return False


def render_gemini_config_ui(key_prefix: str = "") -> tuple[str, str]:
    """
    Renderiza UI para configurar Gemini (API key y modelo).
    
    Args:
        key_prefix: Prefijo para las keys de Streamlit (para evitar conflictos)
        
    Returns:
        Tupla (api_key, model_name)
    """
    st.markdown("### 🤖 Configuración de Gemini AI")
    
    col1, col2 = st.columns(2)
    
    with col1:
        current_key = get_gemini_api_key()
        api_key = st.text_input(
            "API Key de Gemini",
            value=current_key,
            type="password",
            key=f"{key_prefix}gemini_api_key_input",
            help="Obtén tu API key en: https://aistudio.google.com/app/apikey"
        )
    
    with col2:
        current_model = get_gemini_model()
        model_name = st.selectbox(
            "Modelo",
            options=[
                "gemini-3-pro-preview",
                "gemini-3-flash-preview",
                "gemini-2.5-pro",
                "gemini-2.5-flash",
                "gemini-2.0-flash",
            ],
            index=1,  # gemini-3-flash-preview por defecto
            key=f"{key_prefix}gemini_model_select",
            help="Modelo de Gemini a utilizar"
        )
    
    # Guardar en session_state
    if api_key:
        st.session_state["gemini_api_key"] = api_key
    if model_name:
        st.session_state["gemini_model_name"] = model_name
    
    # Mostrar estado
    if is_gemini_available():
        st.success("✅ Gemini configurado correctamente")
    else:
        st.warning("⚠️ Gemini no está configurado. Ingresa tu API key.")
    
    return api_key, model_name


# Alias para compatibilidad con código existente
get_gemini_api_key_from_context = get_gemini_api_key
get_gemini_model_from_context = get_gemini_model
