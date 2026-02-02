"""Componentes de UI compartidos para todas las sub-apps."""
from __future__ import annotations

import streamlit as st


def bordered_container():
    """Container con borde, compatible con versiones antiguas de Streamlit."""
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()
