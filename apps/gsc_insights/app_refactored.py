"""
GSC Insights & Reporting - Refactored
======================================

Versión refactorizada usando módulos compartidos para eliminar duplicación.

Autor: Embedding Insights
Versión: 1.0.1
"""

import streamlit as st
import sys
from pathlib import Path

# Añadir paths al sistema
current_dir = Path(__file__).parent.resolve()
shared_path = (current_dir.parent.parent / "shared").resolve()
modules_path = (current_dir / "modules").resolve()

if str(shared_path) not in sys.path:
    sys.path.insert(0, str(shared_path))
if str(modules_path) not in sys.path:
    sys.path.insert(0, str(modules_path))

# Importar módulos compartidos
from app_common import (
    setup_page_config,
    apply_global_styles,
    render_app_sidebar,
)

# Importar módulos específicos de la app
from modules.positions_report import render_positions_report

# Configurar página
setup_page_config(
    title="GSC Insights & Reporting",
    icon="📊",
    layout="wide"
)


def render_home():
    """Renderiza la página de inicio."""
    st.header("👋 Bienvenido a GSC Insights & Reporting")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 ¿Qué puedes hacer?")
        st.markdown("""
        **Análisis de Posiciones:**
        - ✅ Importar datos de rank tracking
        - ✅ Análisis competitivo automático
        - ✅ Agrupación de keywords por familias
        - ✅ Generación de informes HTML

        **Insights Avanzados:**
        - ✅ Heatmaps de presencia competitiva
        - ✅ Gráficos radar por familia
        - ✅ Análisis de volumen de búsqueda
        - ✅ Recomendaciones con Gemini AI

        **Exportación:**
        - ✅ Informes HTML interactivos
        - ✅ Datos en Excel
        - ✅ Gráficos descargables
        - ✅ Análisis por competidor
        """)

    with col2:
        st.markdown("### 🚀 Quick Start")
        st.markdown("""
        **1. Preparar Datos**
        - Exporta rank tracking desde tu herramienta SEO
        - Formato CSV con columnas: keyword, position, URL
        - Opcional: volumen de búsqueda, fecha

        **2. Cargar y Configurar**
        - Ve a "📈 Informe de Posiciones"
        - Sube CSV de rank tracking
        - Define tu dominio principal
        - Opcional: Agrega familias de keywords

        **3. Generar Informe**
        - Selecciona tipos de gráficos
        - Configura Gemini API (opcional)
        - Genera informe HTML
        - Descarga y comparte

        **4. Análisis Avanzado**
        - Revisa competidores principales
        - Identifica oportunidades por familia
        - Exporta datos para optimización
        - Implementa mejoras
        """)

    # Estadísticas
    st.markdown("---")
    st.markdown("### 📊 Tecnología")

    col_tech1, col_tech2, col_tech3, col_tech4 = st.columns(4)

    with col_tech1:
        st.metric("IA", "Gemini 2.0", help="Análisis automático")
    with col_tech2:
        st.metric("Formatos", "CSV/Excel", help="Import/Export")
    with col_tech3:
        st.metric("Gráficos", "Interactivos", help="HTML embebido")
    with col_tech4:
        st.metric("Familias", "Auto", help="Agrupación automática")

    # Características destacadas
    st.markdown("---")
    st.markdown("### ✨ Características Destacadas")

    col_feat1, col_feat2, col_feat3 = st.columns(3)

    with col_feat1:
        st.markdown("#### 📈 Análisis de Posiciones")
        st.markdown("""
        - Carga CSV de rank tracking
        - Detección automática de dominios
        - Normalización de URLs
        - Vista por keyword y por URL
        """)

    with col_feat2:
        st.markdown("#### 🎯 Familias de Keywords")
        st.markdown("""
        - **Agrupación automática** ⭐
        - Definición manual o CSV
        - Análisis por familia
        - Volumen agregado
        """)

    with col_feat3:
        st.markdown("#### 📊 Informes HTML")
        st.markdown("""
        - Gráficos interactivos
        - Heatmaps competitivos
        - Análisis con Gemini AI
        - Descarga y compartir
        """)

    # Tips de uso
    st.markdown("---")
    with st.expander("💡 Tips de Uso", expanded=False):
        st.markdown("""
        **Preparación de Datos:**
        - Exporta al menos 100 keywords para mejores insights
        - Incluye volumen de búsqueda si es posible
        - Mantén formato consistente en URLs

        **Familias de Keywords:**
        - Define 5-10 familias principales
        - Usa clustering automático con embeddings
        - Revisa y ajusta manualmente

        **Análisis Competitivo:**
        - Identifica competidores recurrentes
        - Analiza por familia, no general
        - Busca gaps de contenido

        **Gemini AI:**
        - Obtén API key gratis en Google AI Studio
        - Usa para insights automáticos
        - Revisa recomendaciones antes de implementar

        **Exportación:**
        - Genera HTML para presentaciones
        - Exporta Excel para análisis detallado
        - Comparte con equipo técnico
        """)

    # Casos de uso
    st.markdown("---")
    st.markdown("### 🎯 Casos de Uso")

    tab1, tab2, tab3 = st.tabs(["SEO Manager", "Content Strategist", "Agency"])

    with tab1:
        st.markdown("""
        **Para SEO Managers:**

        1. **Monitoreo Mensual**
           - Sube datos de rank tracker
           - Genera informe HTML
           - Presenta a stakeholders

        2. **Análisis Competitivo**
           - Identifica competidores por familia
           - Analiza gaps de contenido
           - Planifica estrategia

        3. **Priorización**
           - Revisa quick wins (posiciones 4-10)
           - Identifica keywords de alto volumen
           - Crea plan de acción
        """)

    with tab2:
        st.markdown("""
        **Para Content Strategists:**

        1. **Planificación de Contenido**
           - Analiza familias de keywords
           - Identifica temas sin cubrir
           - Crea calendario editorial

        2. **Optimización Existente**
           - Detecta contenido infraoptimizado
           - Revisa competencia por tema
           - Actualiza contenido

        3. **Clustering de Topics**
           - Agrupa keywords semánticamente
           - Define pillar pages
           - Planifica supporting content
        """)

    with tab3:
        st.markdown("""
        **Para Agencias:**

        1. **Reportes Cliente**
           - Genera informes HTML branded
           - Exporta datos para análisis
           - Presenta evolución mensual

        2. **Análisis Multi-Cliente**
           - Compara rendimiento
           - Identifica best practices
           - Escala estrategias exitosas

        3. **Automatización**
           - Integra con rank trackers
           - Programa generación mensual
           - Entrega automática
        """)


def main():
    """Main application entry point."""
    apply_global_styles()

    # Título y descripción
    st.title("📊 GSC Insights & Reporting")
    st.markdown(
        "Genera informes avanzados de posiciones SEO con análisis competitivo "
        "y agrupación inteligente de keywords."
    )

    # Sidebar con navegación
    tools = [
        "🏠 Inicio",
        "📈 Informe de Posiciones",
    ]
    
    tool = render_app_sidebar("GSC Insights & Reporting", tools)

    # Renderizar herramienta seleccionada
    if tool == "🏠 Inicio":
        render_home()
    elif tool == "📈 Informe de Posiciones":
        render_positions_report()


if __name__ == "__main__":
    main()
