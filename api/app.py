"""
Embedding Insights — API REST (FastAPI)
========================================

Ejecutar con:
    uvicorn api.app:app --host 0.0.0.0 --port 8001 --reload

Requiere:  pip install fastapi uvicorn python-multipart
"""

from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

# Ensure shared/ is importable
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_project_root / "shared") not in sys.path:
    sys.path.insert(0, str(_project_root / "shared"))

from shared.data_orchestrator import DataOrchestrator
from shared.project_manager import ProjectManager

app = FastAPI(
    title="Embedding Insights API",
    version="1.0.0",
    description="API REST para consultar y disparar analisis sobre proyectos SEO.",
)

_pm = ProjectManager()


def _get_orchestrator(project_name: str) -> DataOrchestrator:
    """Load a project and return its DataOrchestrator."""
    try:
        config = _pm.load_project(project_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Proyecto '{project_name}' no encontrado")
    return DataOrchestrator(config["db_path"])


# ═══════════════════════════════════════════
# READ ENDPOINTS
# ═══════════════════════════════════════════


@app.get("/api/projects", tags=["Proyectos"])
def list_projects() -> List[Dict[str, Any]]:
    """Listar todos los proyectos disponibles."""
    return _pm.list_projects()


@app.get("/api/projects/{name}/stats", tags=["Proyectos"])
def project_stats(name: str) -> Dict[str, Any]:
    """Obtener estadisticas del proyecto (conteos por tabla)."""
    orch = _get_orchestrator(name)
    stats = orch.get_stats()
    return {"project": name, **stats}


@app.get("/api/projects/{name}/urls", tags=["URLs"])
def get_urls(
    name: str,
    embedding_status: Optional[str] = Query(None, description="pending | completed"),
    limit: Optional[int] = Query(None, ge=1),
) -> Dict[str, Any]:
    """Obtener URLs del proyecto."""
    orch = _get_orchestrator(name)
    df = orch.get_urls(limit=limit, embedding_status=embedding_status)
    return {"count": len(df), "urls": df.to_dict(orient="records")}


@app.get("/api/projects/{name}/embeddings", tags=["Embeddings"])
def get_embeddings(
    name: str,
    model: str = Query(..., description="Nombre del modelo de embeddings"),
) -> Dict[str, Any]:
    """Obtener URLs con embeddings para un modelo dado."""
    orch = _get_orchestrator(name)
    cached = orch.get_cached_urls(model)
    return {"model": model, "count": len(cached), "urls": sorted(cached)}


@app.get("/api/projects/{name}/clusters", tags=["Clusters"])
def get_clusters(
    name: str,
    model: str = Query(..., description="Nombre del modelo"),
) -> Dict[str, Any]:
    """Obtener resultados de clustering."""
    orch = _get_orchestrator(name)
    df = orch.get_clusters(model)
    return {"model": model, "count": len(df), "clusters": df.to_dict(orient="records")}


@app.get("/api/projects/{name}/relations", tags=["Relaciones"])
def get_relations(
    name: str,
    min_score: float = Query(0.0, ge=0.0, le=1.0),
    source_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Obtener relaciones semanticas entre URLs."""
    orch = _get_orchestrator(name)
    df = orch.get_semantic_relations(source_url=source_url, min_score=min_score)
    return {"count": len(df), "relations": df.to_dict(orient="records")}


@app.get("/api/projects/{name}/entities", tags=["Entidades"])
def get_entities(
    name: str,
    entity_type: Optional[str] = None,
    min_frequency: int = Query(1, ge=1),
) -> Dict[str, Any]:
    """Obtener entidades extraidas."""
    orch = _get_orchestrator(name)
    df = orch.get_entities(entity_type=entity_type, min_frequency=min_frequency)
    return {"count": len(df), "entities": df.to_dict(orient="records")}


# ═══════════════════════════════════════════
# ANALYSIS ENDPOINTS (POST)
# ═══════════════════════════════════════════


@app.post("/api/projects/{name}/process-csv", tags=["Analisis"])
async def process_csv(
    name: str,
    file: UploadFile = File(...),
    url_column: str = Query("url", description="Nombre de la columna URL"),
    embedding_column: str = Query("embedding", description="Nombre de la columna de embeddings"),
) -> Dict[str, Any]:
    """Subir CSV con embeddings y persistirlos en la DB del proyecto."""
    orch = _get_orchestrator(name)

    content = await file.read()
    try:
        if file.filename and file.filename.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_excel(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error leyendo archivo: {exc}")

    if url_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Columna '{url_column}' no encontrada")
    if embedding_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Columna '{embedding_column}' no encontrada")

    saved = 0
    errors = 0
    for _, row in df.iterrows():
        url_val = str(row[url_column]).strip()
        raw = row[embedding_column]
        if not url_val or pd.isna(raw):
            errors += 1
            continue
        try:
            if isinstance(raw, str):
                raw = raw.strip("[] ")
                vec = np.fromstring(raw, sep=",", dtype=np.float32)
            else:
                vec = np.array(raw, dtype=np.float32)
            if vec.size == 0:
                errors += 1
                continue
            orch.save_embeddings(url_val, vec, "csv_upload")
            saved += 1
        except Exception:
            errors += 1

    return {"saved": saved, "errors": errors, "total": len(df)}


@app.post("/api/projects/{name}/similarity", tags=["Analisis"])
def compute_similarity(
    name: str,
    model: str = Query("csv_upload"),
    threshold: float = Query(80.0, ge=0, le=100, description="Umbral minimo de similitud (0-100)"),
    max_results: Optional[int] = Query(None, ge=1),
) -> Dict[str, Any]:
    """Calcular matriz de similitud entre todas las URLs con embeddings."""
    from sklearn.preprocessing import normalize as sk_normalize

    orch = _get_orchestrator(name)
    urls, vectors = orch.get_embedding_vectors(model)

    if len(urls) < 2:
        raise HTTPException(status_code=400, detail="Se necesitan al menos 2 URLs con embeddings")

    emb_norm = sk_normalize(vectors)
    sim_matrix = emb_norm @ emb_norm.T * 100

    i_idx, j_idx = np.triu_indices(len(urls), k=1)
    scores = sim_matrix[i_idx, j_idx]

    mask = scores >= threshold
    i_idx, j_idx, scores = i_idx[mask], j_idx[mask], scores[mask]

    order = np.argsort(scores)[::-1]
    if max_results:
        order = order[:max_results]
    i_idx, j_idx, scores = i_idx[order], j_idx[order], scores[order]

    url_arr = np.array(urls)
    pairs = [
        {"url1": url_arr[i], "url2": url_arr[j], "similarity": round(float(s), 2)}
        for i, j, s in zip(i_idx, j_idx, scores)
    ]
    return {"count": len(pairs), "threshold": threshold, "pairs": pairs}


@app.post("/api/projects/{name}/thin-content", tags=["Analisis"])
def detect_thin_content(
    name: str,
    model: str = Query("csv_upload"),
    duplicate_threshold: float = Query(90.0, ge=0, le=100),
    near_duplicate_threshold: float = Query(80.0, ge=0, le=100),
) -> Dict[str, Any]:
    """Detectar contenido duplicado y near-duplicate via embeddings."""
    from sklearn.preprocessing import normalize as sk_normalize

    orch = _get_orchestrator(name)
    urls, vectors = orch.get_embedding_vectors(model)

    if len(urls) < 2:
        raise HTTPException(status_code=400, detail="Se necesitan al menos 2 URLs con embeddings")

    emb_norm = sk_normalize(vectors)
    sim_matrix = emb_norm @ emb_norm.T * 100

    i_idx, j_idx = np.triu_indices(len(urls), k=1)
    scores = sim_matrix[i_idx, j_idx]

    url_arr = np.array(urls)
    duplicates = []
    near_duplicates = []

    mask_dup = scores >= duplicate_threshold
    for i, j, s in zip(i_idx[mask_dup], j_idx[mask_dup], scores[mask_dup]):
        duplicates.append({"url1": url_arr[i], "url2": url_arr[j], "similarity": round(float(s), 2)})

    mask_near = (scores >= near_duplicate_threshold) & (scores < duplicate_threshold)
    for i, j, s in zip(i_idx[mask_near], j_idx[mask_near], scores[mask_near]):
        near_duplicates.append({"url1": url_arr[i], "url2": url_arr[j], "similarity": round(float(s), 2)})

    return {
        "duplicates": {"count": len(duplicates), "pairs": duplicates},
        "near_duplicates": {"count": len(near_duplicates), "pairs": near_duplicates},
    }


@app.post("/api/projects/{name}/cluster", tags=["Analisis"])
def run_clustering(
    name: str,
    model: str = Query("csv_upload"),
    n_clusters: int = Query(5, ge=2, le=50),
) -> Dict[str, Any]:
    """Ejecutar K-Means clustering sobre embeddings existentes y guardar resultados."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize as sk_normalize

    orch = _get_orchestrator(name)
    urls, vectors = orch.get_embedding_vectors(model)

    if len(urls) < n_clusters:
        raise HTTPException(
            status_code=400,
            detail=f"Solo hay {len(urls)} URLs, se necesitan al menos {n_clusters}",
        )

    emb_norm = sk_normalize(vectors)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(emb_norm)

    # Build result DataFrame and save
    result_df = pd.DataFrame({
        "url": urls,
        "cluster_id": labels.tolist(),
        "cluster_label": [f"Cluster {l}" for l in labels],
        "distance_to_centroid": [
            float(np.linalg.norm(emb_norm[i] - km.cluster_centers_[labels[i]]))
            for i in range(len(urls))
        ],
    })
    orch.save_clusters(result_df, model=model, replace=True)

    # Summary
    clusters_summary = []
    for cid in range(n_clusters):
        cluster_urls = [u for u, l in zip(urls, labels) if l == cid]
        clusters_summary.append({"cluster_id": cid, "size": len(cluster_urls), "urls": cluster_urls})

    return {"n_clusters": n_clusters, "total_urls": len(urls), "clusters": clusters_summary}
