from __future__ import annotations

import importlib
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import spacy  # noqa: F401

SPACY_MODULE = None
SPACY_DOWNLOAD_FN = None
SPACY_IMPORT_ERROR: Optional[Exception] = None
COREFEREE_MODULE = None
COREFEREE_IMPORT_ERROR: Optional[Exception] = None
ENTITY_LINKER_MODULE = None
ENTITY_LINKER_IMPORT_ERROR: Optional[Exception] = None
SUPPORTED_COREF_LANGS = {"en", "de", "fr", "pl"}


def ensure_spacy_module():
    """
    Carga perezosa del módulo principal de spaCy y expone la función de descarga.
    """
    global SPACY_MODULE, SPACY_DOWNLOAD_FN, SPACY_IMPORT_ERROR
    if SPACY_MODULE is not None:
        return SPACY_MODULE
    try:
        spacy_module = importlib.import_module("spacy")
        spacy_cli = importlib.import_module("spacy.cli")
        download_fn = getattr(spacy_cli, "download")
    except Exception as exc:  # noqa: BLE001
        SPACY_IMPORT_ERROR = exc
        raise RuntimeError(
            "spaCy no está disponible en este entorno. Instálalo o corrige la versión "
            "(pip install spacy). Detalle: {exc}".format(exc=exc)
        ) from exc
    SPACY_MODULE = spacy_module
    SPACY_DOWNLOAD_FN = download_fn
    return spacy_module


def is_spacy_available() -> bool:
    """
    Indica si spaCy se pudo importar correctamente.
    """
    if SPACY_MODULE is not None:
        return True
    try:
        ensure_spacy_module()
    except RuntimeError:
        return False
    return True


def ensure_coreferee_module():
    """
    Importa el módulo coreferee solo cuando se necesita el análisis de coreferencia.
    """
    global COREFEREE_MODULE, COREFEREE_IMPORT_ERROR
    if COREFEREE_MODULE is not None:
        return COREFEREE_MODULE
    if COREFEREE_IMPORT_ERROR is not None:
        raise RuntimeError(
            "No se pudo cargar coreferee previamente. Revisa la instalación correcta del paquete."
        ) from COREFEREE_IMPORT_ERROR
    try:
        COREFEREE_MODULE = importlib.import_module("coreferee")
    except Exception as exc:  # noqa: BLE001
        COREFEREE_IMPORT_ERROR = exc
        raise RuntimeError("Para activar la coreferencia instala `coreferee` y sus modelos compatibles.") from exc
    return COREFEREE_MODULE


def ensure_entity_linker_module():
    """
    Importa spacy-entity-linker y gestiona los errores derivados de su instalación.
    """
    global ENTITY_LINKER_MODULE, ENTITY_LINKER_IMPORT_ERROR
    if ENTITY_LINKER_MODULE is not None:
        return ENTITY_LINKER_MODULE
    if ENTITY_LINKER_IMPORT_ERROR is not None:
        raise RuntimeError(
            "No se pudo cargar spacy-entity-linker previamente. Revisa la instalación y la descarga de su KB."
        ) from ENTITY_LINKER_IMPORT_ERROR
    try:
        ENTITY_LINKER_MODULE = importlib.import_module("spacy_entity_linker")
    except Exception as exc:  # noqa: BLE001
        ENTITY_LINKER_IMPORT_ERROR = exc
        raise RuntimeError(
            "Para activar el entity linking instala `spacy-entity-linker` y descarga la base de conocimiento."
        ) from exc
    return ENTITY_LINKER_MODULE
