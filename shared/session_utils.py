"""
Utilidades para gestión de session_state de Streamlit.

Este módulo proporciona funciones helper para trabajar de forma segura
con st.session_state, evitando KeyErrors y mejorando la consistencia.
"""

import streamlit as st
from typing import Any, Optional, Dict, List

try:
    from shared.env_utils import get_env_value
except ModuleNotFoundError:
    from env_utils import get_env_value


def get_session_value(key: str, default: Any = None) -> Any:
    """
    Obtiene un valor de session_state de forma segura.
    
    Args:
        key: Clave a buscar en session_state
        default: Valor por defecto si la clave no existe
        
    Returns:
        Valor almacenado o default
    """
    return st.session_state.get(key, default)


def set_session_value(key: str, value: Any) -> None:
    """
    Establece un valor en session_state.
    
    Args:
        key: Clave a establecer
        value: Valor a almacenar
    """
    st.session_state[key] = value


def has_session_key(key: str) -> bool:
    """
    Verifica si una clave existe en session_state.
    
    Args:
        key: Clave a verificar
        
    Returns:
        True si existe, False en caso contrario
    """
    return key in st.session_state


def delete_session_key(key: str) -> None:
    """
    Elimina una clave de session_state si existe.
    
    Args:
        key: Clave a eliminar
    """
    if key in st.session_state:
        del st.session_state[key]


def initialize_session_defaults(defaults: Dict[str, Any]) -> None:
    """
    Inicializa múltiples valores en session_state si no existen.
    
    Args:
        defaults: Diccionario con {key: default_value}
    """
    for key, default in defaults.items():
        st.session_state.setdefault(key, default)


def clear_session_keys(keys: List[str]) -> None:
    """
    Limpia múltiples claves de session_state.
    
    Args:
        keys: Lista de claves a eliminar
    """
    for key in keys:
        delete_session_key(key)


def get_session_dataframe(key: str):
    """
    Obtiene un DataFrame de session_state de forma segura.
    
    Args:
        key: Clave del DataFrame
        
    Returns:
        DataFrame o None si no existe o no es un DataFrame
    """
    import pandas as pd
    
    value = get_session_value(key)
    if isinstance(value, pd.DataFrame):
        return value
    return None


def set_session_dataframe(key: str, df) -> None:
    """
    Almacena un DataFrame en session_state.
    
    Args:
        key: Clave para almacenar
        df: DataFrame a almacenar
    """
    import pandas as pd
    
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"El valor debe ser un DataFrame, no {type(df)}")
    
    set_session_value(key, df)


def get_project_config() -> Optional[Dict]:
    """
    Obtiene la configuración del proyecto actual.
    
    Returns:
        Diccionario con configuración del proyecto o None
    """
    return get_session_value("project_config")


def get_current_project_name() -> Optional[str]:
    """
    Obtiene el nombre del proyecto actual.
    
    Returns:
        Nombre del proyecto o None
    """
    return get_session_value("current_project")


def is_project_loaded() -> bool:
    """
    Verifica si hay un proyecto cargado.
    
    Returns:
        True si hay un proyecto cargado
    """
    return has_session_key("current_project") and get_current_project_name() is not None


def get_oauth_manager():
    """
    Obtiene el OAuth manager del proyecto actual.
    
    Returns:
        OAuth manager o None
    """
    return get_session_value("oauth_manager")


def get_api_key(service: str) -> Optional[str]:
    """
    Obtiene una API key de session_state.
    
    Args:
        service: Nombre del servicio (ej: "openai", "gemini", "anthropic")
        
    Returns:
        API key o None
    """
    key_names = {
        "openai": ["openai_api_key", "OPENAI_API_KEY"],
        "gemini": ["gemini_api_key", "GEMINI_API_KEY", "google_api_key"],
        "anthropic": ["anthropic_api_key", "ANTHROPIC_API_KEY"],
    }
    
    possible_keys = key_names.get(service.lower(), [f"{service}_api_key"])
    
    for key in possible_keys:
        value = get_session_value(key)
        if value:
            return value

    env_by_service = {
        "openai": ("OPENAI_API_KEY",),
        "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "anthropic": ("ANTHROPIC_API_KEY",),
        "serprobot": ("SERPROBOT_API_KEY",),
    }
    env_value = get_env_value(*env_by_service.get(service.lower(), ()))
    if env_value:
        return env_value
    
    return None


def set_api_key(service: str, api_key: str) -> None:
    """
    Establece una API key en session_state.
    
    Args:
        service: Nombre del servicio
        api_key: API key a almacenar
    """
    set_session_value(f"{service}_api_key", api_key)


class SessionStateManager:
    """
    Gestor de session_state con contexto.
    
    Ejemplo:
        with SessionStateManager() as ssm:
            ssm.set("key", "value")
            value = ssm.get("key")
    """
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def get(self, key: str, default: Any = None) -> Any:
        """Obtiene un valor."""
        return get_session_value(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Establece un valor."""
        set_session_value(key, value)
    
    def has(self, key: str) -> bool:
        """Verifica si existe una clave."""
        return has_session_key(key)
    
    def delete(self, key: str) -> None:
        """Elimina una clave."""
        delete_session_key(key)
    
    def initialize(self, defaults: Dict[str, Any]) -> None:
        """Inicializa valores por defecto."""
        initialize_session_defaults(defaults)
    
    def clear(self, keys: List[str]) -> None:
        """Limpia múltiples claves."""
        clear_session_keys(keys)


# Instancia global para uso conveniente
session = SessionStateManager()
