from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from pyvis.network import Network

from modules.entity_filters import (
    is_valid_entity,
    normalize_entity_text,
    clean_entities_advanced,
    lemmatize_text,
)
from modules.semantic_depth import analyze_document_sds
from modules.semantic_tools import get_sentence_transformer
from modules.spacy_support import (
    SUPPORTED_COREF_LANGS,
    ensure_coreferee_module,
    ensure_entity_linker_module,
    ensure_spacy_module,
    SPACY_DOWNLOAD_FN,
)

if TYPE_CHECKING:  # pragma: no cover
    import spacy

PROMINENCE_WEIGHTS = {
    "nsubj": 2.0,
    "nsubjpass": 1.8,
    "csubj": 1.7,
    "dobj": 1.5,
    "obj": 1.5,
    "pobj": 0.8,
    "attr": 1.1,
    "appos": 1.3,
    "ROOT": 1.4,
    "default": 0.5,
}


@st.cache_resource(show_spinner=False)
def load_spacy_model(model_name: str):
    spacy_module = ensure_spacy_module()
    try:
        return spacy_module.load(model_name)
    except OSError:
        download_fn = SPACY_DOWNLOAD_FN
        if download_fn is None:
            raise RuntimeError(
                f"No se pudo cargar el modelo spaCy '{model_name}' y no se puede descargar automáticamente."
            )
        try:
            download_fn(model_name)
            return spacy_module.load(model_name)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"No se pudo cargar el modelo spaCy '{model_name}'. "
                "Instálalo manualmente con 'python -m spacy download MODEL'."
            ) from exc


def get_prominence_weight(dep_label: str) -> float:
    if not dep_label:
        return PROMINENCE_WEIGHTS["default"]
    return PROMINENCE_WEIGHTS.get(dep_label.lower(), PROMINENCE_WEIGHTS["default"])


def add_optional_nlp_components(nlp, enable_coref: bool, enable_linking: bool) -> None:
    language_code = getattr(nlp, "lang", "").lower()
    if enable_coref and language_code not in SUPPORTED_COREF_LANGS:
        st.warning(
            f"Coreferee no soporta el idioma '{language_code or 'desconocido'}'. "
            "La coreferencia se desactiva para evitar errores."
        )
        enable_coref = False
    if enable_coref:
        ensure_coreferee_module()
        if "coreferee" not in nlp.pipe_names:
            nlp.add_pipe("coreferee")
    if enable_linking:
        try:
            ensure_entity_linker_module()
        except RuntimeError as exc:
            st.warning(f"No se pudo cargar spacy-entity-linker: {exc}")
            enable_linking = False
        else:
            if "entityLinker" not in nlp.pipe_names:
                nlp.add_pipe("entityLinker", last=True)


def build_coref_map(doc) -> Dict[Tuple[int, int], "spacy.tokens.Span"]:
    mapping: Dict[Tuple[int, int], "spacy.tokens.Span"] = {}
    if not hasattr(doc._, "coref_chains") or not doc._.coref_chains:
        return mapping
    for chain in doc._.coref_chains:
        main_span = chain.main
        for mention in chain:
            mapping[(mention.start, mention.end)] = main_span
    return mapping


def resolve_canonical_span(span: "spacy.tokens.Span", coref_map: Dict[Tuple[int, int], "spacy.tokens.Span"]):
    return coref_map.get((span.start, span.end), span)


def get_linker_metadata(ent: "spacy.tokens.Span", linker_pipe) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not linker_pipe or not hasattr(ent._, "kb_ents") or not ent._.kb_ents:
        return None, None, None
    kb_id, _score = ent._.kb_ents[0]
    if not kb_id:
        return None, None, None
    kb_entry = getattr(linker_pipe, "kb", None)
    description = None
    kb_url = None
    if kb_entry and hasattr(kb_entry, "cui_to_entity"):
        entity = kb_entry.cui_to_entity.get(kb_id)
        if entity:
            description = getattr(entity, "description", None)
            kb_url = getattr(entity, "url", None)
    return kb_id, description, kb_url


def find_entity_for_token(token: "spacy.tokens.Token", token_entity_map: Dict[int, str]) -> Optional[str]:
    if token.i in token_entity_map:
        return token_entity_map[token.i]
    for descendant in token.subtree:
        if descendant.i in token_entity_map:
            return token_entity_map[descendant.i]
    return None


def extract_spo_relations(
    doc: "spacy.tokens.Doc",
    token_entity_map: Dict[int, str],
    source_url: str,
) -> List[Dict[str, str]]:
    relations: List[Dict[str, str]] = []
    for sent in doc.sents:
        root = sent.root
        if not root or root.pos_ not in {"VERB", "AUX"}:
            continue
        relation_label = root.lemma_.lower()
        subject_tokens = [token for token in sent if token.dep_.startswith("nsubj") and token.head == root]
        object_tokens = [
            token
            for token in sent
            if token.dep_ in {"dobj", "obj", "pobj", "attr", "ccomp", "xcomp", "dative"} and token.head == root
        ]
        subjects = [find_entity_for_token(token, token_entity_map) for token in subject_tokens]
        objects = [find_entity_for_token(token, token_entity_map) for token in object_tokens]
        subjects = [subj for subj in subjects if subj]
        objects = [obj for obj in objects if obj]
        if not subjects or not objects:
            continue
        for subj in subjects:
            for obj in objects:
                if subj == obj:
                    continue
                relations.append(
                    {
                        "subject": subj,
                        "predicate": relation_label,
                        "object": obj,
                        "source": source_url,
                    }
                )
    return relations


def generate_knowledge_graph_html_v2(
    df: pd.DataFrame,
    text_column: str,
    url_column: Optional[str],
    model_name: str,
    row_limit: int,
    max_entities: int,
    min_entity_frequency: int,
    include_pages: bool,
    max_pages: int,
    allowed_entity_labels: Optional[Set[str]],
    enable_coref: bool = False,
    enable_linking: bool = False,
    n_process: int = 1,
    batch_size: int = 200,
    manual_entities: Optional[Sequence[str]] = None,
    blacklist_entities: Optional[Sequence[str]] = None,
) -> Tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Genera un grafo de conocimiento interactivo a partir de texto con spaCy.

    Args:
        manual_entities: Lista de entidades que deben recibir boost de prominence (whitelist)
        blacklist_entities: Lista de entidades a excluir del análisis (blacklist)

    Returns:
        Tupla con:
        - HTML del grafo de conocimiento
        - DataFrame de entidades
        - DataFrame de relaciones documento-entidad
        - DataFrame de relaciones SPO (Subject-Predicate-Object)
        - DataFrame de Semantic Depth Score por documento
    """
    nlp = load_spacy_model(model_name)
    add_optional_nlp_components(nlp, enable_coref=enable_coref, enable_linking=enable_linking)
    linker_pipe = nlp.get_pipe("entityLinker") if enable_linking and "entityLinker" in nlp.pipe_names else None
    
    # Normalizar whitelist y blacklist
    manual_entities_lower = {e.strip().lower() for e in (manual_entities or [])} if manual_entities else set()
    blacklist_lower = {e.strip().lower() for e in (blacklist_entities or [])} if blacklist_entities else set()

    sampled = df.head(row_limit).copy()
    sampled["__content__"] = sampled[text_column].fillna("").astype(str)
    documents: List[str] = []
    doc_urls: List[str] = []
    for _, row in sampled.iterrows():
        text = row["__content__"].strip()
        if not text or text.lower() == "nan":
            continue
        documents.append(text)
        if url_column and url_column in row:
            url_value = str(row[url_column]).strip()
        else:
            url_value = f"doc_{len(documents)}"
        doc_urls.append(url_value or f"doc_{len(documents)}")

    if not documents:
        raise ValueError("No se encontraron textos válidos en la columna seleccionada.")

    required_components = {"ner", "parser"}
    if enable_coref:
        required_components.add("coreferee")
    if enable_linking:
        required_components.add("entityLinker")
    components_to_disable = [component for component in nlp.pipe_names if component not in required_components]

    pipe_kwargs: Dict[str, object] = {"batch_size": max(1, int(batch_size))}
    if n_process > 1:
        pipe_kwargs["n_process"] = int(n_process)

    docs_iterator = nlp.pipe(
        documents,
        disable=components_to_disable,
        **pipe_kwargs,
    )

    entity_stats: Dict[str, Dict[str, object]] = {}
    doc_entity_stats: Dict[Tuple[str, str], Dict[str, float]] = {}
    spo_rows: List[Dict[str, object]] = []
    spo_edge_counter: Counter = Counter()

    # Preparar modelo de embeddings para cálculo de cohesión vectorial (SDS)
    try:
        embedding_model = get_sentence_transformer()
    except Exception:
        embedding_model = None

    # Almacenar datos por documento para calcular SDS
    doc_sds_data: Dict[str, Dict[str, object]] = {}

    for doc, source_url in zip(docs_iterator, doc_urls):
        coref_map = build_coref_map(doc) if enable_coref else {}
        token_entity_map: Dict[int, str] = {}

        # Almacenar entidades del documento para SDS
        doc_entities_list = []

        for ent in doc.ents:
            ent_text = ent.text.strip()
            if not ent_text:
                continue
            if allowed_entity_labels and ent.label_ not in allowed_entity_labels:
                continue

            canonical_span = resolve_canonical_span(ent, coref_map) if coref_map else ent
            canonical_text = canonical_span.text.strip() or ent_text
            if not canonical_text:
                continue

            # Normalizar texto de entidad (eliminar comillas, puntuación final, etc.)
            canonical_text = normalize_entity_text(canonical_text)
            if not canonical_text:
                continue

            # Aplicar filtros de calidad para eliminar ruido
            if not is_valid_entity(
                text=canonical_text,
                entity_type=ent.label_,
                min_length=2,
                allow_common_names=False
            ):
                continue  # Skip entidad ruidosa

            # Aplicar blacklist
            entity_lower = canonical_text.lower()
            if blacklist_lower and any(black in entity_lower for black in blacklist_lower):
                continue  # Skip entidad en blacklist

            qid, kb_description, kb_url = get_linker_metadata(ent, linker_pipe)
            entity_id = qid or canonical_text.lower()

            stats = entity_stats.setdefault(
                entity_id,
                {
                    "canonical_name": canonical_text,
                    "label": ent.label_,
                    "qid": qid,
                    "kb_description": kb_description,
                    "kb_url": kb_url,
                    "frequency": 0,
                    "prominence": 0.0,
                    "pages": set(),
                    "is_manual": False,  # Nueva flag para marcar entidades manuales
                },
            )
            stats["frequency"] += 1
            weight = get_prominence_weight(ent.root.dep_)
            
            # Boost para entidades manuales
            is_manual_match = manual_entities_lower and any(manual in entity_lower for manual in manual_entities_lower)
            if is_manual_match:
                stats["is_manual"] = True
                weight *= 3.0  # Boost significativo (3x)
            
            stats["prominence"] += weight
            stats["pages"].add(source_url)
            if qid and not stats.get("qid"):
                stats["qid"] = qid
            if kb_description and not stats.get("kb_description"):
                stats["kb_description"] = kb_description
            if kb_url and not stats.get("kb_url"):
                stats["kb_url"] = kb_url

            doc_key = (source_url, entity_id)
            doc_entity_stats.setdefault(doc_key, {"frequency": 0, "prominence": 0.0})
            doc_entity_stats[doc_key]["frequency"] += 1
            doc_entity_stats[doc_key]["prominence"] += weight

            for token in ent:
                token_entity_map[token.i] = entity_id

            # Almacenar información de entidad para cálculo de SDS
            doc_entities_list.append({
                "text": canonical_text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            })

        # Calcular SDS para este documento
        doc_text = doc.text
        doc_sds_result = None

        if doc_entities_list and embedding_model is not None:
            try:
                # Generar embeddings de frases para cohesión vectorial
                sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
                if sentences:
                    embeddings = embedding_model.encode(sentences, convert_to_numpy=True)

                    # Calcular SDS
                    doc_sds_result = analyze_document_sds(
                        text=doc_text,
                        entities=doc_entities_list,
                        embeddings=embeddings,
                        w_er=0.5,
                        w_cv=0.5
                    )

                    # Almacenar resultado
                    doc_sds_data[source_url] = doc_sds_result
            except Exception:
                # Si falla el cálculo de SDS, continuar sin él
                pass

        triplets = extract_spo_relations(doc, token_entity_map, source_url)
        for triplet in triplets:
            subj_id = triplet["subject"]
            obj_id = triplet["object"]
            if subj_id not in entity_stats or obj_id not in entity_stats:
                continue
            predicate = triplet["predicate"]
            spo_edge_counter[(subj_id, predicate, obj_id)] += 1
            spo_rows.append(
                {
                    "Sujeto": entity_stats[subj_id]["canonical_name"],
                    "Sujeto QID": entity_stats[subj_id]["qid"],
                    "Predicado": predicate,
                    "Objeto": entity_stats[obj_id]["canonical_name"],
                    "Objeto QID": entity_stats[obj_id]["qid"],
                    "Fuente": triplet["source"],
                }
            )

    if not entity_stats:
        raise ValueError("No se detectaron entidades en el texto proporcionado.")

    graph_nodes = list(entity_stats.keys())
    graph_nx = nx.DiGraph()
    graph_nx.add_nodes_from(graph_nodes)
    for (subj_id, predicate, obj_id), weight in spo_edge_counter.items():
        if subj_id in entity_stats and obj_id in entity_stats:
            graph_nx.add_edge(subj_id, obj_id, weight=weight)

    if graph_nx.number_of_nodes() > 0:
        try:
            closeness_scores = nx.closeness_centrality(graph_nx, wf_improved=True)
        except Exception:  # noqa: BLE001
            closeness_scores = {node: 0.0 for node in graph_nx.nodes}
        try:
            betweenness_scores = nx.betweenness_centrality(graph_nx)
        except Exception:  # noqa: BLE001
            betweenness_scores = {node: 0.0 for node in graph_nx.nodes}
    else:
        closeness_scores = {}
        betweenness_scores = {}

    for entity_id, stats in entity_stats.items():
        closeness = closeness_scores.get(entity_id, 0.0)
        betweenness = betweenness_scores.get(entity_id, 0.0)
        stats["closeness"] = closeness
        stats["betweenness"] = betweenness
        stats["unified_authority_score"] = stats["prominence"] + (closeness * 100.0) + (betweenness * 100.0)

    filtered_entities = {
        entity_id: stats
        for entity_id, stats in entity_stats.items()
        if stats["frequency"] >= max(1, min_entity_frequency)
    }
    if not filtered_entities:
        raise ValueError(
            "No se encontraron entidades que cumplan la frecuencia mínima. Reduce el umbral o añade más contenido."
        )

    sorted_entities = sorted(
        filtered_entities.values(),
        key=lambda item: (item["unified_authority_score"], item["frequency"]),
        reverse=True,
    )
    if max_entities > 0:
        sorted_entities = sorted_entities[:max_entities]
    top_entity_ids = {stats["qid"] or stats["canonical_name"].lower() for stats in sorted_entities}

    net = Network(height="640px", width="100%", bgcolor="#0f111a", font_color="#f5f7ff")
    net.toggle_physics(True)
    net.repulsion(node_distance=250, spring_length=220, damping=0.8)

    for stats in sorted_entities:
        entity_id = stats["qid"] or stats["canonical_name"].lower()
        label = stats["canonical_name"]
        display_label = label if len(label) <= 38 else f"{label[:35]}…"
        tooltip_parts = [
            f"<strong>{label}</strong>",
            f"Tipo: {stats['label']}",
            f"Frecuencia consolidada: {stats['frequency']}",
            f"Prominence sintáctica: {stats['prominence']:.2f}",
            f"Closeness: {stats.get('closeness', 0.0):.4f}",
            f"Betweenness: {stats.get('betweenness', 0.0):.4f}",
            f"Autoridad tópica: {stats.get('unified_authority_score', 0.0):.2f}",
        ]
        if stats.get("qid"):
            tooltip_parts.append(f"Wikidata ID: {stats['qid']}")
        if stats.get("kb_description"):
            tooltip_parts.append(stats["kb_description"])
        net.add_node(
            entity_id,
            label=display_label,
            title="<br/>".join(tooltip_parts),
            size=18 + min(int(stats.get("unified_authority_score", 0.0) / 2), 36),
            value=stats.get("unified_authority_score", stats["prominence"]),
            color="#5c6bff",
        )

    for (subj_id, predicate, obj_id), weight in spo_edge_counter.items():
        if subj_id not in top_entity_ids or obj_id not in top_entity_ids:
            continue
        net.add_edge(
            subj_id,
            obj_id,
            label=predicate,
            title=f"{predicate} · peso {weight}",
            value=weight,
            color="#c2d1ff",
        )

    if include_pages and url_column:
        page_scores: Dict[str, float] = {}
        for (page_url, entity_id), metrics in doc_entity_stats.items():
            if entity_id not in top_entity_ids:
                continue
            page_scores[page_url] = page_scores.get(page_url, 0.0) + metrics["prominence"]
        ranked_pages = sorted(page_scores.items(), key=lambda item: item[1], reverse=True)[:max_pages]
        allowed_pages = {url for url, _ in ranked_pages}
        for url, score in ranked_pages:
            display_url = url if len(url) <= 42 else f"{url[:39]}…"
            net.add_node(
                f"page::{url}",
                label=display_url,
                title=f"{url}<br/>Prominence acumulado: {score:.2f}",
                shape="box",
                color="#5dade2",
                size=12,
            )
        for (page_url, entity_id), metrics in doc_entity_stats.items():
            if page_url not in allowed_pages or entity_id not in top_entity_ids:
                continue
            net.add_edge(
                f"page::{page_url}",
                entity_id,
                color="#95a5a6",
                title=f"{page_url} → {entity_stats[entity_id]['canonical_name']} ({metrics['prominence']:.2f})",
            )

    entities_df = pd.DataFrame(
        [
            {
                "Entidad": stats["canonical_name"],
                "QID": stats.get("qid") or "—",
                "Tipo": stats["label"],
                "Frecuencia consolidada": int(stats["frequency"]),
                "Prominence sintáctica": round(float(stats["prominence"]), 3),
                "Closeness Centrality": round(float(stats.get("closeness", 0.0)), 5),
                "Betweenness Centrality": round(float(stats.get("betweenness", 0.0)), 5),
                "Autoridad tópica unificada": round(float(stats.get("unified_authority_score", 0.0)), 3),
                "Páginas únicas": len(stats["pages"]),
                "Descripción KB": stats.get("kb_description") or "",
                "URL KB": stats.get("kb_url") or "",
            }
            for stats in sorted_entities
        ]
    )

    doc_rows = [
        {
            "URL": page_url,
            "Entidad": entity_stats[entity_id]["canonical_name"],
            "QID": entity_stats[entity_id].get("qid") or "—",
            "Frecuencia documento": metrics["frequency"],
            "Prominence documento": round(metrics["prominence"], 3),
        }
        for (page_url, entity_id), metrics in doc_entity_stats.items()
        if entity_id in top_entity_ids and page_url
    ]
    doc_relations_df = pd.DataFrame(sorted(doc_rows, key=lambda row: row["Prominence documento"], reverse=True))

    spo_df = pd.DataFrame(spo_rows)

    # Crear DataFrame con resultados de SDS por documento
    sds_rows = []
    for url, sds_result in doc_sds_data.items():
        sds_rows.append({
            "URL": url,
            "SDS (Semantic Depth Score)": sds_result.get("sds", 0.0),
            "Clasificación": sds_result.get("classification", "N/A"),
            "Score ER (Entity Relevance)": sds_result.get("score_er", 0.0),
            "Score CV (Vector Cohesion)": sds_result.get("score_cv", 0.0),
            "Densidad de Entidades": sds_result.get("entity_density", 0.0),
            "Número de Entidades": sds_result.get("entity_count", 0),
            "Cohesión Vectorial": sds_result.get("cohesion_raw", 0.0),
        })
    sds_df = pd.DataFrame(sds_rows).sort_values("SDS (Semantic Depth Score)", ascending=False) if sds_rows else pd.DataFrame()

    return net.generate_html(notebook=False), entities_df, doc_relations_df, spo_df, sds_df


def build_entity_payload_from_doc_relations(doc_relations_df: Optional[pd.DataFrame]) -> Dict[str, Dict[str, float]]:
    """
    Convierte el resumen pagina-entidad en un diccionario {url: {entidad: prominence}} para reutilizarlo en el laboratorio.
    """
    payload: Dict[str, Dict[str, float]] = {}
    if doc_relations_df is None or doc_relations_df.empty:
        return payload

    for _, row in doc_relations_df.iterrows():
        url_value = str(row.get("URL", "")).strip()
        if not url_value:
            continue
        entity_id = str(row.get("QID") or "").strip()
        if not entity_id or entity_id in {"-", "-", "None", "nan"}:
            entity_id = str(row.get("Entidad") or "").strip()
        if not entity_id:
            continue
        prominence = row.get("Prominence documento")
        try:
            prominence_value = float(prominence)
        except (TypeError, ValueError):
            freq_value = row.get("Frecuencia documento", 0)
            try:
                prominence_value = float(freq_value)
            except (TypeError, ValueError):
                prominence_value = 0.0
        if prominence_value <= 0:
            continue
        url_bucket = payload.setdefault(url_value, {})
        url_bucket[entity_id] = url_bucket.get(entity_id, 0.0) + prominence_value
    return payload


__all__ = [
    "add_optional_nlp_components",
    "build_coref_map",
    "build_entity_payload_from_doc_relations",
    "extract_spo_relations",
    "generate_knowledge_graph_html_v2",
    "get_prominence_weight",
    "load_spacy_model",
]
