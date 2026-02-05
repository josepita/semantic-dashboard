"""
License UI - Streamlit components for license management.

Provides:
- License activation screen
- License status display in sidebar
- Feature gating with upgrade prompts
"""
from __future__ import annotations

from typing import Callable, Optional

import streamlit as st

try:
    from .license_manager import (
        get_license_manager,
        LicenseInfo,
        is_feature_enabled,
        is_licensed,
        is_trial,
        TRIAL_FEATURES,
        LICENSE_DEV_MODE,
    )
except ImportError:
    from shared.license_manager import (
        get_license_manager,
        LicenseInfo,
        is_feature_enabled,
        is_licensed,
        is_trial,
        TRIAL_FEATURES,
        LICENSE_DEV_MODE,
    )


# ============== Session State Keys ==============

LICENSE_KEY_STATE = "license_key_input"
LICENSE_EMAIL_STATE = "license_email_input"
LICENSE_CHECKED_STATE = "license_checked"


# ============== License Activation Screen ==============

def render_license_activation() -> bool:
    """
    Render the license activation screen.

    Returns:
        True if license is valid and user can proceed, False otherwise.
    """
    # Dev mode: siempre permitir
    if LICENSE_DEV_MODE:
        return True

    # Note: page_config should be set by the calling app before this function
    # to avoid conflicts. We only set it if not already configured.
    if not st.session_state.get("_page_config_set"):
        try:
            st.set_page_config(
                page_title="Activar Licencia - Embedding Dashboard",
                page_icon="🔐",
                layout="centered",
            )
        except Exception:
            pass  # Page config already set by caller

    st.title("🔐 Activar Licencia")
    st.markdown("---")

    manager = get_license_manager()

    # Check for cached license on first load
    if not st.session_state.get(LICENSE_CHECKED_STATE):
        st.session_state[LICENSE_CHECKED_STATE] = True
        cached = manager.check_cached_license()
        if cached and cached.valid:
            return True

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("### Introduce tu licencia")
        st.markdown(
            "Para usar Embedding Dashboard necesitas una licencia válida. "
            "Si no tienes una, puedes continuar en **modo trial** con funcionalidad limitada."
        )

        license_key = st.text_input(
            "License Key",
            value=st.session_state.get(LICENSE_KEY_STATE, ""),
            placeholder="EMB-XXXX-XXXX-XXXX-XXXX",
            key="license_key_field",
        )

        email = st.text_input(
            "Email",
            value=st.session_state.get(LICENSE_EMAIL_STATE, ""),
            placeholder="tu@email.com",
            key="license_email_field",
        )

        col_a, col_b = st.columns(2)

        with col_a:
            if st.button("🔓 Activar Licencia", type="primary", use_container_width=True):
                if not license_key or not email:
                    st.error("Por favor introduce la licencia y el email")
                else:
                    with st.spinner("Verificando licencia..."):
                        success, message = manager.activate_license(license_key, email)

                    if success:
                        st.success(f"✅ {message}")
                        st.session_state[LICENSE_KEY_STATE] = license_key
                        st.session_state[LICENSE_EMAIL_STATE] = email
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")

        with col_b:
            if st.button("🆓 Modo Trial", use_container_width=True):
                st.session_state["trial_mode"] = True
                return True

        st.markdown("---")

        # Show trial features
        with st.expander("ℹ️ Funcionalidades del Modo Trial", expanded=False):
            st.markdown("""
            En modo trial tienes acceso a:
            - ✅ Hub principal
            - ✅ Herramientas semánticas básicas
            - ✅ Procesamiento CSV (máx. 100 filas)

            **Funcionalidades PRO:**
            - ❌ Keyword Builder
            - ❌ Linking Lab
            - ❌ Positions Report
            - ❌ Relaciones Semánticas
            - ❌ Fan-Out Analyzer
            - ❌ API REST
            - ❌ Exportación Excel
            """)

    return manager.is_licensed or st.session_state.get("trial_mode", False)


def check_license_or_block() -> bool:
    """
    Check for valid license or show activation screen.

    Call this at the start of your app's main() function.

    Returns:
        True if user can proceed (licensed or trial), False if blocked.
    """
    # Dev mode: siempre permitir acceso
    if LICENSE_DEV_MODE:
        return True

    manager = get_license_manager()

    # First check cached license
    if not st.session_state.get(LICENSE_CHECKED_STATE):
        cached = manager.check_cached_license()
        st.session_state[LICENSE_CHECKED_STATE] = True
        if cached and cached.valid:
            return True

    # If already licensed, proceed
    if manager.is_licensed:
        return True

    # If in trial mode, proceed
    if st.session_state.get("trial_mode", False):
        return True

    # Show activation screen (this will handle the page_config)
    return render_license_activation()


def check_license_silent() -> bool:
    """
    Silently check for valid license without showing UI.

    Useful for checking license status without blocking.

    Returns:
        True if licensed, False otherwise.
    """
    manager = get_license_manager()

    if not st.session_state.get(LICENSE_CHECKED_STATE):
        manager.check_cached_license()
        st.session_state[LICENSE_CHECKED_STATE] = True

    return manager.is_licensed


# ============== Sidebar License Status ==============

def render_license_status_sidebar() -> None:
    """
    Render license status in the sidebar.

    Shows:
    - Current license status (licensed/trial)
    - Plan and features if licensed
    - Upgrade prompt if trial
    """
    # Dev mode: no mostrar nada de licencias
    if LICENSE_DEV_MODE:
        return

    manager = get_license_manager()
    license_info = manager.current_license

    with st.sidebar:
        st.markdown("---")

        if license_info and license_info.valid:
            # Licensed
            st.markdown("### 🔐 Licencia")

            # Plan badge
            plan_colors = {"trial": "gray", "basic": "blue", "pro": "green"}
            plan_color = plan_colors.get(license_info.plan, "gray")
            st.markdown(f"**Plan:** :{plan_color}[{license_info.plan.upper()}]")

            # Expiration
            if license_info.days_remaining > 0:
                if license_info.days_remaining <= 30:
                    st.warning(f"⚠️ Expira en {license_info.days_remaining} días")
                else:
                    st.caption(f"Expira: {license_info.expires[:10]}")

            # Cache indicator
            if license_info.from_cache:
                st.caption("📴 Modo offline (cache)")

            # Deactivate option
            with st.expander("⚙️ Opciones", expanded=False):
                if st.button("🔓 Desactivar licencia", key="sidebar_deactivate"):
                    success, msg = manager.deactivate_license(license_info.license_key)
                    if success:
                        st.success(msg)
                        st.session_state[LICENSE_CHECKED_STATE] = False
                        st.rerun()
                    else:
                        st.error(msg)

        else:
            # Trial mode
            st.markdown("### 🆓 Modo Trial")
            st.caption("Funcionalidad limitada")

            if st.button("🔐 Activar Licencia", key="sidebar_activate"):
                st.session_state["trial_mode"] = False
                st.session_state[LICENSE_CHECKED_STATE] = False
                st.rerun()


# ============== Feature Gating ==============

def render_feature_or_upgrade(
    feature: str,
    render_func: Callable[[], None],
    feature_name: str = "",
) -> None:
    """
    Render a feature if enabled, or show upgrade prompt.

    Args:
        feature: Feature key to check
        render_func: Function to call if feature is enabled
        feature_name: Human-readable feature name for upgrade prompt
    """
    if is_feature_enabled(feature):
        render_func()
    else:
        _render_upgrade_prompt(feature_name or feature)


def require_feature(feature: str, feature_name: str = "") -> bool:
    """
    Check if a feature is available, show upgrade prompt if not.

    Args:
        feature: Feature key to check
        feature_name: Human-readable feature name

    Returns:
        True if feature is enabled, False otherwise
    """
    # Dev mode: todas las features habilitadas
    if LICENSE_DEV_MODE:
        return True

    if is_feature_enabled(feature):
        return True

    _render_upgrade_prompt(feature_name or feature)
    return False


def _render_upgrade_prompt(feature_name: str) -> None:
    """Render the upgrade prompt for a locked feature."""
    st.warning(f"🔒 **{feature_name}** requiere licencia")
    st.markdown("""
    Esta funcionalidad no está disponible en modo trial.

    **Obtén tu licencia en:** [tu-sitio.com](https://tu-sitio.com)
    """)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔐 Activar Licencia", key=f"upgrade_{feature_name}"):
            st.session_state["trial_mode"] = False
            st.session_state[LICENSE_CHECKED_STATE] = False
            st.rerun()


# ============== Feature Limits ==============

def check_csv_row_limit(row_count: int) -> tuple[bool, int]:
    """
    Check if CSV row count is within license limits.

    Args:
        row_count: Number of rows in the CSV

    Returns:
        Tuple of (allowed, max_rows)
        - allowed: True if within limit
        - max_rows: Maximum allowed rows (0 = unlimited)
    """
    if is_licensed():
        return True, 0  # Unlimited

    # Trial limit
    max_rows = 100
    return row_count <= max_rows, max_rows


def get_csv_limit_message(row_count: int) -> Optional[str]:
    """
    Get a message if CSV exceeds trial limit.

    Returns:
        Warning message if over limit, None otherwise.
    """
    allowed, max_rows = check_csv_row_limit(row_count)

    if allowed:
        return None

    return (
        f"⚠️ El archivo tiene {row_count} filas. "
        f"En modo trial el límite es {max_rows} filas. "
        "Activa tu licencia para procesar archivos más grandes."
    )


# ============== Initialization ==============

def init_license_check() -> None:
    """
    Initialize license check on app startup.

    Call this early in your app to set up the license manager.
    """
    # Dev mode: no necesita verificación
    if LICENSE_DEV_MODE:
        st.session_state[LICENSE_CHECKED_STATE] = True
        return

    if not st.session_state.get(LICENSE_CHECKED_STATE):
        manager = get_license_manager()
        manager.check_cached_license()
        st.session_state[LICENSE_CHECKED_STATE] = True
