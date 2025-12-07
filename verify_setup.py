#!/usr/bin/env python3
"""Verificacion completa del entorno sin caracteres fuera de ASCII."""

import sys

SUPPORTED_COREF_LANGS = {"en", "de", "es", "fr", "pl"}
SPACY_MODEL = "es_core_news_sm"
TARGET_LANG = "es"


def _print_header() -> None:
    """Imprime la cabecera del script de verificación."""
    header = "VERIFICACION DEL ENTORNO PARA STREAMLIT + SPACY"
    print("=" * 60)
    print(header)
    print("=" * 60)

def _verify_packages() -> bool:
    # 1. Version de Python
    print(f"\n- Python: {sys.version.split()[0]}")

    # 2. Paquetes principales
    packages = [
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("contourpy", "Contourpy"),
        ("matplotlib", "Matplotlib"),
        ("sklearn", "Scikit-learn"),
        ("sentence_transformers", "Sentence Transformers"),
        ("spacy", "spaCy"),
        ("thinc", "Thinc"),
        ("coreferee", "Coreferee"),
    ]

    print("\n- PAQUETES INSTALADOS:")
    all_ok = True
    for module_name, display_name in packages:
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"  OK  {display_name:25s} {version}")
        except ImportError:
            print(f"  ERR {display_name:25s} NO INSTALADO")
            all_ok = False
    return all_ok


def _test_spacy(nlp) -> bool:
    """Ejecuta una prueba funcional de spaCy para NER."""
    print("\n- TEST FUNCIONAL DE SPACY:")
    try:
        doc = nlp("Maria trabaja en Google en Barcelona.")
        print("  OK  Modelo cargado correctamente")
        print(f"  OK  Procesados {len(doc)} tokens")
        if doc.ents:
            ents = [(ent.text, ent.label_) for ent in doc.ents]
            print(f"  OK  Entidades detectadas: {ents}")
            return True
        else:
            print("  WARN No se detectaron entidades (se esperaba PERSON/ORG/GPE)")
            return True  # No es un error fatal
    except Exception as exc:
        print(f"  ERR Error durante el test de spaCy: {exc}")
        return False


def _test_coreferee(nlp) -> bool:
    """Ejecuta una prueba funcional de Coreferee."""
    print("\n- TEST DE COREFEREE:")
    if TARGET_LANG not in SUPPORTED_COREF_LANGS:
        print(f"  INFO Coreferee no soporta el idioma '{TARGET_LANG}'. Test omitido.")
        return True

    try:
        if "coreferee" not in nlp.pipe_names:
            nlp.add_pipe("coreferee")

        doc = nlp("Juan vive en Madrid. El trabaja alli desde hace anos.")
        if hasattr(doc._, "coref_chains"):
            chains = doc._.coref_chains
            print("  OK  Coreferee activo")
            if chains:
                print(f"  OK  Detectadas {len(chains)} cadenas de correferencia")
            else:
                print("  WARN Sin cadenas detectadas (puede ser normal)")
        else:
            print("  ERR Atributo coref_chains no disponible")
            return False
    except Exception as exc:
        print(f"  ERR Error en Coreferee: {exc}")
        return False

    return True


def _test_entity_linker(nlp) -> None:
    """Ejecuta una prueba funcional de Entity Linking."""
    print("\n- TEST DE ENTITY LINKING (opcional):")
    try:
        import spacy_entity_linker

        if "entityLinker" not in nlp.pipe_names:
            nlp.add_pipe("entityLinker", last=True)

        doc = nlp("Google fue fundada en California.")
        if any(ent._.kb_ents for ent in doc.ents):
            print("  OK  Entity Linking funciona")
            for ent in doc.ents:
                if ent._.kb_ents:
                    kb_id, score = ent._.kb_ents[0]
                    print(f"       '{ent.text}' -> {kb_id} (score {score:.3f})")
        else:
            print("  WARN Entity Linking instalado pero sin enlaces encontrados")
    except ImportError:
        print("  INFO Entity Linking no instalado (es opcional)")
    except Exception as exc:
        print(f"  WARN Error en Entity Linking: {exc}")


def verify_environment() -> None:
    """Ejecuta una verificación completa del entorno."""
    _print_header()

    all_ok = _verify_packages()

    if all_ok:
        try:
            import spacy
            nlp = spacy.load(SPACY_MODEL)
            print(f"\n  OK  Modelo '{SPACY_MODEL}' cargado correctamente")

            all_ok &= _test_spacy(nlp)
            all_ok &= _test_coreferee(nlp)
            _test_entity_linker(nlp)

        except (ImportError, OSError) as exc:
            print(f"\n  ERR Error al cargar spaCy o el modelo '{SPACY_MODEL}': {exc}")
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("ENTORNO LISTO PARA USAR")
        print("Puedes ejecutar: streamlit run streamlit_app.py")
    else:
        print("HAY PROBLEMAS EN EL ENTORNO")
        print("Revisa los mensajes anteriores antes de ejecutar la app")
    print("=" * 60)


if __name__ == "__main__":
    verify_environment()
