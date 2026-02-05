"""Constantes y configuración centralizada del proyecto."""
from __future__ import annotations

import os
from typing import Dict, List

# ── License Server ──
# Modo desarrollo: True = todas las features habilitadas sin servidor
# TEMPORAL: Licencias desactivadas temporalmente
LICENSE_DEV_MODE = True  # TODO: Cambiar a False para activar verificación de licencias

# URL del servidor de licencias PHP en SiteGround
LICENSE_SERVER_URL = os.getenv(
    "LICENSE_SERVER_URL",
    "https://licencia.alejandroc105.sg-host.com/license_server_php/api.php"
)

# ── Modelos Gemini disponibles ──
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

GEMINI_MODELS: List[str] = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

# ── Presets de entidades por vertical ──
ENTITY_PROFILE_PRESETS: Dict[str, List[str]] = {
    "Clinica / Salud": [
        "ORG", "PERSON", "PRODUCT", "GPE", "LOC", "FAC",
        "EVENT", "LAW", "NORP", "LANGUAGE", "WORK_OF_ART",
        "DISEASE", "SYMPTOM", "MEDICATION",
    ],
    "Editorial / Libros": [
        "ORG", "PERSON", "WORK_OF_ART", "PRODUCT", "EVENT",
        "GPE", "LOC", "LANGUAGE", "LAW", "NORP", "FAC",
    ],
    "Ecommerce / Retail": [
        "ORG", "PRODUCT", "PERSON", "GPE", "LOC", "FAC",
        "EVENT", "WORK_OF_ART", "LAW", "NORP",
    ],
}
