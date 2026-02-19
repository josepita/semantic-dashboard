"""
Utilidades para cargar variables de entorno y API keys desde `.env`.

Este módulo evita depender de librerías externas para leer `.env` y
centraliza los defaults de API keys usados por las apps de Streamlit.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Mapping, MutableMapping, Optional, Sequence

_ENV_LOCK = threading.Lock()
_ENV_ALREADY_LOADED = False
_ENV_LOADED_PATH: Optional[Path] = None

SESSION_API_ENV_VARS = {
    "openai_api_key": ("OPENAI_API_KEY",),
    "gemini_api_key": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "anthropic_api_key": ("ANTHROPIC_API_KEY",),
    "serprobot_api_key": ("SERPROBOT_API_KEY",),
    "google_kg_api_key": ("GOOGLE_EKG_API_KEY",),
}

SESSION_MODEL_ENV_VARS = {
    "gemini_model_name": ("GEMINI_MODEL",),
    "openai_model": ("OPENAI_MODEL",),
}


def _find_env_path() -> Optional[Path]:
    project_root = Path(__file__).resolve().parents[1]
    cwd = Path.cwd().resolve()
    candidates = [
        project_root / ".env",
        cwd / ".env",
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _parse_env_value(raw_value: str) -> str:
    value = (raw_value or "").strip()
    if not value:
        return ""

    if value[0] in {'"', "'"}:
        quote = value[0]
        if len(value) >= 2 and value[-1] == quote:
            value = value[1:-1]
        else:
            value = value[1:]
        if quote == '"':
            value = (
                value.replace(r"\n", "\n")
                .replace(r"\r", "\r")
                .replace(r"\t", "\t")
                .replace(r"\\", "\\")
                .replace(r"\"", '"')
            )
        return value

    comment_index = value.find(" #")
    if comment_index >= 0:
        value = value[:comment_index]
    return value.strip()


def load_project_env(override: bool = False) -> Optional[Path]:
    """
    Carga variables desde `.env` a `os.environ` una sola vez por proceso.

    Args:
        override: Si es True, sobrescribe valores ya existentes en `os.environ`.

    Returns:
        Ruta del `.env` cargado o None si no se encontró.
    """
    global _ENV_ALREADY_LOADED, _ENV_LOADED_PATH

    with _ENV_LOCK:
        if _ENV_ALREADY_LOADED and not override:
            return _ENV_LOADED_PATH

        env_path = _find_env_path()
        _ENV_ALREADY_LOADED = True
        _ENV_LOADED_PATH = env_path

        if env_path is None:
            return None

        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue

            key, raw_value = line.split("=", 1)
            key = key.strip()
            if not key:
                continue

            parsed = _parse_env_value(raw_value)
            if override or not os.environ.get(key):
                os.environ[key] = parsed

        return env_path


def get_env_value(*names: str, default: str = "") -> str:
    """
    Devuelve el primer valor no vacío de una lista de variables de entorno.
    """
    load_project_env()
    for name in names:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return default


def get_session_or_env(
    session_state: Mapping[str, object],
    session_key: str,
    env_names: Sequence[str],
    default: str = "",
) -> str:
    """
    Resuelve valor priorizando session_state y usando `.env` como fallback.
    """
    current = str(session_state.get(session_key, "") or "").strip()
    if current:
        return current
    return get_env_value(*env_names, default=default)


def bootstrap_api_session_state(session_state: Optional[MutableMapping[str, object]] = None) -> None:
    """
    Inicializa session_state con API keys/modelos desde `.env` si faltan.
    """
    if session_state is None:
        try:
            import streamlit as st
        except Exception:
            return
        session_state = st.session_state

    load_project_env()

    for session_key, env_names in SESSION_API_ENV_VARS.items():
        if str(session_state.get(session_key, "") or "").strip():
            continue
        env_value = get_env_value(*env_names)
        if env_value:
            session_state[session_key] = env_value

    gemini_model = get_env_value(*SESSION_MODEL_ENV_VARS["gemini_model_name"], default="gemini-2.5-flash")
    if not str(session_state.get("gemini_model_name", "") or "").strip():
        session_state["gemini_model_name"] = gemini_model

    openai_model = get_env_value(*SESSION_MODEL_ENV_VARS["openai_model"], default="gpt-4o-mini")
    if not str(session_state.get("openai_model", "") or "").strip():
        session_state["openai_model"] = openai_model

    if not str(session_state.get("positions_gemini_key", "") or "").strip():
        session_state["positions_gemini_key"] = get_session_or_env(
            session_state,
            "gemini_api_key",
            SESSION_API_ENV_VARS["gemini_api_key"],
            default="",
        )

    if not str(session_state.get("positions_gemini_model", "") or "").strip():
        session_state["positions_gemini_model"] = session_state.get("gemini_model_name", gemini_model)
