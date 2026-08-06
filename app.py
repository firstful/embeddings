"""embeddings — semantic search service for the Sutliff homelab.

Serves OpenAI-compatible /v1/embeddings plus /search and /ingest over a
SQLite vector store. Index sources:
  - homelab-docs (cloned from Gitea at startup / on /refresh)
  - Hermes session history (pushed by the host exporter via /ingest/sessions)

Model: sentence-transformers/all-MiniLM-L6-v2 via fastembed (ONNX, CPU).
Requires AVX (rook.io/avx=true node) — SIGILLs exit 132 on non-AVX CPUs.

LAN-only (embeddings.homelab.internal); never tunneled.
"""

import json
import math
import os
import re
import sqlite3
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import asyncio
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

APP_NAME = "embeddings"
PORT = int(os.environ.get("PORT", "8080"))
MODEL_NAME = os.environ.get("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
MODEL_CACHE = os.environ.get("MODEL_CACHE", "/models")
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
DB_PATH = DATA_DIR / "vectors.db"
DOCS_URL = os.environ.get("DOCS_URL", "http://192.168.1.2:3000/reeds/homelab-docs.git")
DOCS_TOKEN = os.environ.get("DOCS_TOKEN", "")
TOP_K_DEFAULT = int(os.environ.get("TOP_K_DEFAULT", "5"))
MIN_SCORE = float(os.environ.get("MIN_SCORE", "0.15"))

DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title=APP_NAME, version="1.0.0")

_model = None


def get_model():
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        # threads: cap to 2 — pod CPU limit is 2; onnxruntime's default pool
        # spawns per-core threads (12+ on the Ryzen) and blew past the 2Gi
        # memory limit during big ingests (OOMKilled 2026-08-06).
        _model = TextEmbedding(
            model_name=MODEL_NAME,
            cache_dir=MODEL_CACHE,
            threads=int(os.environ.get("ORT_THREADS", "2")),
        )
    return _model


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            title TEXT,
            text TEXT NOT NULL,
            meta TEXT,
            vec BLOB NOT NULL,
            updated_at REAL NOT NULL
        )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)")
    conn.commit()
    return conn


def _embed(texts: list[str]) -> list[np.ndarray]:
    model = get_model()
    out = []
    for v in model.embed(texts):
        out.append(np.asarray(v, dtype=np.float32))
    return out


async def _embed_async(texts: list[str]) -> list[np.ndarray]:
    """Embed off the event loop — onnxruntime is CPU-bound and would stall
    /health during large ingests (kubelet liveness killed the pod)."""
    return await asyncio.to_thread(_embed, texts)


def _vec_to_blob(v: np.ndarray) -> bytes:
    return v.astype(np.float32).tobytes()


def _blob_to_vec(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@app.get("/health")
async def health():
    return {"status": "ok", "app": APP_NAME, "model": MODEL_NAME}


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    """OpenAI-compatible embeddings endpoint: {input: str|list[str]} -> {data:[{embedding,index}]}."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": {"message": "invalid JSON"}})
    inp = body.get("input")
    if isinstance(inp, str):
        texts = [inp]
        single = True
    elif isinstance(inp, list):
        texts = [str(t) for t in inp]
        single = False
    else:
        return JSONResponse(status_code=400, content={"error": {"message": "input must be string or list"}})
    vecs = await _embed_async([t for t in texts if t.strip()])
    # fastembed skips empty strings; re-align with an empty-vector placeholder
    full = []
    j = 0
    for t in texts:
        if t.strip():
            full.append(vecs[j].tolist())
            j += 1
        else:
            full.append([0.0] * 384)
    data = [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(full)]
    return {"object": "list", "data": data, "model": MODEL_NAME, "usage": {"prompt_tokens": 0, "total_tokens": 0}}


@app.post("/ingest")
async def ingest(request: Request):
    """Upsert chunks. Body: {source, chunks:[{id?,title?,text,meta?}]}. id defaults to hash(text)."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": {"message": "invalid JSON"}})
    source = str(body.get("source", "")).strip()
    chunks = body.get("chunks") or []
    if not source or not chunks:
        return JSONResponse(status_code=400, content={"error": {"message": "source and chunks required"}})
    texts = [str(c.get("text", "")).strip() for c in chunks]
    if not any(texts):
        return JSONResponse(status_code=400, content={"error": {"message": "no non-empty chunk text"}})
    vecs = await _embed_async([t for t in texts if t])

    conn = _db()
    now = time.time()
    n = 0
    j = 0
    for c, t in zip(chunks, texts):
        if not t:
            continue
        cid = str(c.get("id") or uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{t}"))
        meta = c.get("meta") or {}
        conn.execute(
            "INSERT INTO chunks (id, source, title, text, meta, vec, updated_at) VALUES (?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET title=excluded.title, text=excluded.text, meta=excluded.meta, "
            "vec=excluded.vec, updated_at=excluded.updated_at",
            (cid, source, str(c.get("title") or ""), t, json.dumps(meta), _vec_to_blob(vecs[j]), now),
        )
        n += 1
        j += 1
    conn.commit()
    conn.close()
    return {"status": "ok", "source": source, "upserted": n}


@app.post("/ingest/sessions")
async def ingest_sessions(request: Request):
    """Sessions arrive pre-chunked from the host exporter: {chunks:[...]} with source forced to 'session'."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": {"message": "invalid JSON"}})
    chunks = body.get("chunks") or []
    payload = {"source": "session", "chunks": chunks}
    # reuse ingest logic
    texts = [str(c.get("text", "")).strip() for c in chunks]
    if not texts or not any(texts):
        return JSONResponse(status_code=400, content={"error": {"message": "no non-empty chunk text"}})
    vecs = await _embed_async([t for t in texts if t])
    conn = _db()
    now = time.time()
    n = 0
    j = 0
    for c, t in zip(chunks, texts):
        if not t:
            continue
        cid = str(c.get("id") or uuid.uuid5(uuid.NAMESPACE_URL, f"session:{t}"))
        meta = c.get("meta") or {}
        conn.execute(
            "INSERT INTO chunks (id, source, title, text, meta, vec, updated_at) VALUES (?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET title=excluded.title, text=excluded.text, meta=excluded.meta, "
            "vec=excluded.vec, updated_at=excluded.updated_at",
            (cid, "session", str(c.get("title") or ""), t, json.dumps(meta), _vec_to_blob(vecs[j]), now),
        )
        n += 1
        j += 1
    conn.commit()
    conn.close()
    return {"status": "ok", "source": "session", "upserted": n}


@app.post("/search")
async def search(request: Request):
    """Semantic search. Body: {query, top_k?, sources?[docs|session], min_score?} -> {results:[{id,source,title,text,meta,score}]}."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": {"message": "invalid JSON"}})
    query = str(body.get("query", "")).strip()
    if not query:
        return JSONResponse(status_code=400, content={"error": {"message": "query required"}})
    top_k = int(body.get("top_k", TOP_K_DEFAULT))
    sources = body.get("sources")
    min_score = float(body.get("min_score", MIN_SCORE))

    qv = (await _embed_async([query]))[0]
    conn = _db()
    if sources:
        marks = ",".join("?" * len(sources))
        rows = conn.execute(
            f"SELECT id, source, title, text, meta, vec FROM chunks WHERE source IN ({marks})", list(sources)
        ).fetchall()
    else:
        rows = conn.execute("SELECT id, source, title, text, meta, vec FROM chunks").fetchall()
    conn.close()

    scored = []
    for cid, source, title, text, meta, vblob in rows:
        v = _blob_to_vec(vblob)
        s = _cosine(qv, v)
        if s >= min_score:
            scored.append({"id": cid, "source": source, "title": title, "text": text, "meta": meta, "score": round(s, 4)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return {"query": query, "count": len(scored), "results": scored[:top_k]}


@app.get("/stats")
async def stats():
    conn = _db()
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    by_source = conn.execute("SELECT source, COUNT(*) FROM chunks GROUP BY source").fetchall()
    conn.close()
    return {"total": total, "by_source": {s: c for s, c in by_source}}


# ---------------- indexing ----------------

def _split_markdown(text: str, min_chars: int = 400, max_chars: int = 2000) -> list[str]:
    """Split markdown into chunks by headings (## / ###) then paragraphs."""
    sections = re.split(r"(?m)^(#{1,4}\s)", text)
    chunks = []
    buf = ""
    for part in sections:
        if re.match(r"^#{1,4}\s$", part):
            # heading marker: flush buf then start new
            if buf.strip():
                chunks.append(buf.strip())
            buf = part
        else:
            buf += part
            if len(buf) >= min_chars:
                # try to cut at paragraph boundary near max
                if len(buf) >= max_chars:
                    cut = buf.rfind("\n\n", 0, max_chars)
                    if cut > min_chars:
                        chunks.append(buf[:cut].strip())
                        buf = buf[cut:]
    if buf.strip():
        chunks.append(buf.strip())
    return [c for c in chunks if len(c) >= 120]


def _docs_chunks() -> list[dict]:
    """Clone homelab-docs fresh and chunk every .md file."""
    with tempfile.TemporaryDirectory() as td:
        url = DOCS_URL
        if DOCS_TOKEN:
            url = url.replace("http://", f"http://reeds:{DOCS_TOKEN}@")
        subprocess.run(["git", "clone", "-q", "--depth", "1", url, str(Path(td) / "docs")], check=True, timeout=120)
        repo = Path(td) / "docs"
        chunks = []
        for md in sorted(repo.rglob("*.md")):
            rel = str(md.relative_to(repo))
            try:
                text = md.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for i, piece in enumerate(_split_markdown(text)):
                chunks.append({
                    "id": f"docs:{rel}:{i}",
                    "title": rel,
                    "text": piece,
                    "meta": {"path": rel, "chunk": i},
                })
        return chunks


async def _refresh_docs():
    chunks = _docs_chunks()
    if not chunks:
        return {"status": "error", "detail": "no docs chunks"}
    conn = _db()
    conn.execute("DELETE FROM chunks WHERE source='docs'")
    conn.commit()
    conn.close()
    # embed + insert
    texts = [c["text"] for c in chunks]
    vecs = await _embed_async(texts)
    conn = _db()
    now = time.time()
    for c, v in zip(chunks, vecs):
        conn.execute(
            "INSERT INTO chunks (id, source, title, text, meta, vec, updated_at) VALUES (?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET title=excluded.title, text=excluded.text, meta=excluded.meta, "
            "vec=excluded.vec, updated_at=excluded.updated_at",
            (c["id"], "docs", c["title"], c["text"], json.dumps(c["meta"]), _vec_to_blob(v), now),
        )
    conn.commit()
    conn.close()
    return {"status": "ok", "source": "docs", "indexed": len(chunks)}


@app.post("/refresh")
async def refresh(request: Request):
    """Re-clone homelab-docs and rebuild the docs index. Sessions are pushed by the host exporter."""
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    if body.get("docs"):
        return await _refresh_docs()
    return {"status": "ok", "hint": "use {\"docs\": true} to rebuild docs; sessions are pushed via /ingest/sessions"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
