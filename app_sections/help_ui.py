"""
Componentes de UI para mostrar ayuda contextual en Streamlit.
"""

from __future__ import annotations

import streamlit as st

from app_sections.help_content import get_help_content


def show_help_section(
    section_key: str,
    expanded: bool = False,
    icon: str = "ℹ️"
) -> None:
    """
    Muestra un panel de ayuda contextual para una sección específica.

    Args:
        section_key: Clave de la sección (ej: "clustering", "knowledge_graph")
        expanded: Si True, el expander se muestra abierto por defecto
        icon: Emoji o icono para el título de ayuda
    """
    help_content = get_help_content(section_key)

    if not help_content:
        st.warning(f"No hay ayuda disponible para la sección: {section_key}")
        return

    with st.expander(f"{icon} Ayuda: {help_content['title']}", expanded=expanded):
        # Fundamento teórico SEO
        st.markdown("### 📚 Fundamento Teórico SEO")
        st.markdown(help_content["seo_theory"])

        st.markdown("---")

        # Cómo interpretar resultados
        st.markdown("### 🔍 Cómo Interpretar Resultados")
        st.markdown(help_content["interpretation"])

        st.markdown("---")

        # Mejores prácticas de uso
        st.markdown("### ✅ Mejores Prácticas de Uso")
        st.markdown(help_content["best_practices"])


def show_quick_help(section_key: str, help_type: str = "all") -> None:
    """
    Muestra ayuda rápida en formato compacto (sin expander).

    Args:
        section_key: Clave de la sección
        help_type: Tipo de ayuda a mostrar: "theory", "interpretation", "practices", "all"
    """
    help_content = get_help_content(section_key)

    if not help_content:
        return

    if help_type in ("theory", "all"):
        st.info(f"**📚 Fundamento SEO:** {help_content['seo_theory']}")

    if help_type in ("interpretation", "all"):
        st.info(f"**🔍 Interpretación:** {help_content['interpretation']}")

    if help_type in ("practices", "all"):
        st.success(f"**✅ Mejores Prácticas:** {help_content['best_practices']}")


def show_compact_help(section_key: str) -> None:
    """
    Muestra ayuda en formato muy compacto con tabs.

    Args:
        section_key: Clave de la sección
    """
    help_content = get_help_content(section_key)

    if not help_content:
        return

    tab1, tab2, tab3 = st.tabs(["📚 Fundamento SEO", "🔍 Interpretación", "✅ Mejores Prácticas"])

    with tab1:
        st.markdown(help_content["seo_theory"])

    with tab2:
        st.markdown(help_content["interpretation"])

    with tab3:
        st.markdown(help_content["best_practices"])


def show_help_sidebar(section_key: str) -> None:
    """
    Muestra ayuda en el sidebar (útil para páginas con mucho contenido).

    Args:
        section_key: Clave de la sección
    """
    help_content = get_help_content(section_key)

    if not help_content:
        return

    with st.sidebar:
        st.markdown(f"## ℹ️ Ayuda: {help_content['title']}")

        with st.expander("📚 Fundamento SEO", expanded=False):
            st.markdown(help_content["seo_theory"])

        with st.expander("🔍 Interpretación", expanded=False):
            st.markdown(help_content["interpretation"])

        with st.expander("✅ Mejores Prácticas", expanded=False):
            st.markdown(help_content["best_practices"])


def show_inline_tip(section_key: str, tip_type: str = "practices") -> None:
    """
    Muestra un consejo rápido inline (útil para colocar junto a controles).

    Args:
        section_key: Clave de la sección
        tip_type: Tipo de tip: "theory", "interpretation", "practices"
    """
    help_content = get_help_content(section_key)

    if not help_content:
        return

    tip_map = {
        "theory": ("📚", help_content["seo_theory"]),
        "interpretation": ("🔍", help_content["interpretation"]),
        "practices": ("✅", help_content["best_practices"]),
    }

    if tip_type in tip_map:
        icon, text = tip_map[tip_type]
        # Tomar solo la primera frase para mantenerlo compacto
        first_sentence = text.split(". ")[0] + "."
        st.caption(f"{icon} {first_sentence}")


__all__ = [
    "show_help_section",
    "show_quick_help",
    "show_compact_help",
    "show_help_sidebar",
    "show_inline_tip",
]
