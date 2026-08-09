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
SOURCES_DIR = Path(os.environ.get("SOURCES_DIR", "/sources"))
SKILLS_DIR = Path(os.environ.get("SKILLS_DIR", "/skills"))
SESSION_DB = SOURCES_DIR / "hermes-state.db"
SESSION_CONSUMER = os.environ.get("SESSION_CONSUMER", "hermes-rook")
TOP_K_DEFAULT = int(os.environ.get("TOP_K_DEFAULT", "5"))
MIN_SCORE = float(os.environ.get("MIN_SCORE", "0.15"))

DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title=APP_NAME, version="1.2.4")

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
            consumer TEXT,
            title TEXT,
            text TEXT NOT NULL,
            meta TEXT,
            vec BLOB NOT NULL,
            updated_at REAL NOT NULL
        )"""
    )
    # migration: add consumer column if the table predates it
    cols = [r[1] for r in conn.execute("PRAGMA table_info(chunks)").fetchall()]
    if "consumer" not in cols:
        conn.execute("ALTER TABLE chunks ADD COLUMN consumer TEXT")
        conn.commit()
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
    """Upsert chunks. Body: {source, consumer?, chunks:[{id?,title?,text,meta?}]}. id defaults to hash(text)."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": {"message": "invalid JSON"}})
    source = str(body.get("source", "")).strip()
    consumer = str(body.get("consumer") or "").strip() or None  # None = shared/global
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
        cid = str(c.get("id") or uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{consumer or ''}:{t}"))
        meta = c.get("meta") or {}
        conn.execute(
            "INSERT INTO chunks (id, source, consumer, title, text, meta, vec, updated_at) VALUES (?,?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET consumer=excluded.consumer, title=excluded.title, text=excluded.text, meta=excluded.meta, "
            "vec=excluded.vec, updated_at=excluded.updated_at",
            (cid, source, consumer, str(c.get("title") or ""), t, json.dumps(meta), _vec_to_blob(vecs[j]), now),
        )
        n += 1
        j += 1
    conn.commit()
    conn.close()
    return {"status": "ok", "source": source, "consumer": consumer, "upserted": n}


@app.post("/ingest/sessions")
async def ingest_sessions(request: Request):
    """Sessions arrive pre-chunked from the host exporter: {consumer?, chunks:[...]}. Source forced to 'session'."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": {"message": "invalid JSON"}})
    chunks = body.get("chunks") or []
    consumer = str(body.get("consumer") or "").strip() or None  # None = legacy/shared (should not happen post-v1.1)
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
        cid = str(c.get("id") or uuid.uuid5(uuid.NAMESPACE_URL, f"session:{consumer or ''}:{t}"))
        meta = c.get("meta") or {}
        conn.execute(
            "INSERT INTO chunks (id, source, consumer, title, text, meta, vec, updated_at) VALUES (?,?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET consumer=excluded.consumer, title=excluded.title, text=excluded.text, meta=excluded.meta, "
            "vec=excluded.vec, updated_at=excluded.updated_at",
            (cid, "session", consumer, str(c.get("title") or ""), t, json.dumps(meta), _vec_to_blob(vecs[j]), now),
        )
        n += 1
        j += 1
    conn.commit()
    conn.close()
    return {"status": "ok", "source": "session", "consumer": consumer, "upserted": n}


@app.post("/search")
async def search(request: Request):
    """Semantic search. Body: {query, top_k?, sources?[docs|session], min_score?, consumers?: list[str]}.

    Consumer scoping (v1.1): docs chunks are shared (consumer NULL); session chunks
    belong to one consumer. Pass `consumers` to restrict session hits to those
    consumers — docs always included. Omitting `consumers` is legacy behavior and
    matches EVERYTHING (backward compat for pre-v1.1 callers)."""
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
    consumers = body.get("consumers")
    if consumers is not None and not isinstance(consumers, list):
        consumers = [str(consumers)]
    if consumers is not None:
        consumers = [str(c).strip() for c in consumers if str(c).strip()]

    qv = (await _embed_async([query]))[0]
    conn = _db()

    where = []
    params: list = []
    if sources:
        marks = ",".join("?" * len(sources))
        where.append(f"source IN ({marks})")
        params.extend(sources)
    if consumers is not None:
        if consumers:
            marks = ",".join("?" * len(consumers))
            # docs (shared, consumer NULL) always visible; sessions scoped to listed consumers
            where.append(f"(consumer IS NULL OR consumer IN ({marks}))")
            params.extend(consumers)
        else:
            # empty consumer list -> shared chunks only (docs)
            where.append("consumer IS NULL")
    sql = "SELECT id, source, consumer, title, text, meta, vec FROM chunks"
    if where:
        sql += " WHERE " + " AND ".join(where)
    rows = conn.execute(sql, params).fetchall()
    conn.close()

    scored = []
    for cid, source, consumer, title, text, meta, vblob in rows:
        v = _blob_to_vec(vblob)
        s = _cosine(qv, v)
        if s >= min_score:
            scored.append({"id": cid, "source": source, "consumer": consumer, "title": title, "text": text, "meta": meta, "score": round(s, 4)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return {"query": query, "count": len(scored), "results": scored[:top_k]}


@app.get("/stats")
async def stats():
    conn = _db()
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    by_source = conn.execute("SELECT source, COUNT(*) FROM chunks GROUP BY source").fetchall()
    by_consumer = conn.execute("SELECT COALESCE(consumer,'shared'), COUNT(*) FROM chunks GROUP BY consumer").fetchall()
    conn.close()
    return {"total": total, "by_source": {s: c for s, c in by_source}, "by_consumer": {s: c for s, c in by_consumer}}


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
            "INSERT INTO chunks (id, source, consumer, title, text, meta, vec, updated_at) VALUES (?,?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET consumer=excluded.consumer, title=excluded.title, text=excluded.text, meta=excluded.meta, "
            "vec=excluded.vec, updated_at=excluded.updated_at",
            (c["id"], "docs", None, c["title"], c["text"], json.dumps(c["meta"]), _vec_to_blob(v), now),
        )
    conn.commit()
    conn.close()
    return {"status": "ok", "source": "docs", "indexed": len(chunks)}


def _split_markdown_para(text: str, min_chars: int = 150, max_chars: int = 1800) -> list[str]:
    """Split markdown by headings then paragraph boundaries (skills + session text)."""
    sections = re.split(r"(?m)^(#{1,4}\s)", text)
    chunks = []
    buf = ""
    for part in sections:
        if re.match(r"^#{1,4}\s$", part):
            if buf.strip():
                chunks.append(buf.strip())
            buf = part
        else:
            buf += part
            if len(buf) >= min_chars:
                if len(buf) >= max_chars:
                    cut = buf.rfind("\n\n", 0, max_chars)
                    if cut > min_chars:
                        chunks.append(buf[:cut].strip())
                        buf = buf[cut:]
    if buf.strip():
        chunks.append(buf.strip())
    return [c for c in chunks if len(c) >= 80]


def _session_chunks(db_path: Path, consumer: str, chunk_chars: int = 1500) -> list[dict]:
    """Read the state.db snapshot and chunk conversations per session (same logic as
    the old host exporter: role: content lines, ~1500-char windows).

    immutable=1: the snapshot is a static .backup copy served over a READ-ONLY
    NFS mount. A WAL-mode DB (state.db is WAL) cannot be opened read-only — SQLite
    must write the -shm lock file — so treat the file as immutable (no locking).
    Without this, _refresh_sources dies with "unable to open database file"
    (hit 2026-08-09, v1.2.3).
    """
    import sqlite3 as _sq

    conn = _sq.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    conn.row_factory = _sq.Row
    sessions = {}
    try:
        for r in conn.execute("SELECT id, title, started_at, source FROM sessions"):
            sessions[r["id"]] = {"title": r["title"] or r["id"], "started_at": r["started_at"], "source": r["source"] or ""}
    except _sq.Error:
        pass
    rows = conn.execute(
        "SELECT id, session_id, role, content, timestamp FROM messages ORDER BY id"
    ).fetchall()
    conn.close()

    by_session: dict[str, list[dict]] = {}
    for r in rows:
        by_session.setdefault(r["session_id"], []).append(
            {"id": r["id"], "role": r["role"], "content": r["content"], "timestamp": r["timestamp"]}
        )

    out: list[dict] = []
    for sid, msgs in by_session.items():
        meta = sessions.get(sid, {})
        buf: list[str] = []
        buf_len = 0
        idx = 0
        for m in msgs:
            content = m.get("content") or ""
            line = f"{m.get('role', 'user')}: {content}".strip()
            if not line:
                continue
            line += "\n"
            if buf and buf_len + len(line) > chunk_chars:
                text = "".join(buf).strip()
                if len(text) >= 60:
                    out.append({
                        "id": f"session:{sid}:{idx}",
                        "title": meta.get("title") or sid,
                        "text": text,
                        "meta": {"session_id": sid, "session_title": meta.get("title") or "", "source_type": meta.get("source") or "", "started_at": meta.get("started_at")},
                    })
                    idx += 1
                buf = []
                buf_len = 0
            buf.append(line)
            buf_len += len(line)
        if buf:
            text = "".join(buf).strip()
            if len(text) >= 60:
                out.append({
                    "id": f"session:{sid}:{idx}",
                    "title": meta.get("title") or sid,
                    "text": text,
                    "meta": {"session_id": sid, "session_title": meta.get("title") or "", "source_type": meta.get("source") or "", "started_at": meta.get("started_at")},
                })
    # tag consumer on the way out
    for c in out:
        c["consumer"] = consumer
    return out


def _bundled_skill_names() -> set[str]:
    """Skill names that ship with Hermes, from .bundled_manifest (name:hash per line).

    Hermes keeps a copy of every bundled skill in ~/.hermes/skills (mixed with
    user skills); the manifest is the authoritative shipped-with-Hermes list.
    """
    out: set[str] = set()
    try:
        for line in (SKILLS_DIR / ".bundled_manifest").read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if ":" in line:
                out.add(line.split(":", 1)[0])
    except OSError:
        pass
    return out


_BUNDLED_SKILLS = _bundled_skill_names()


def _skill_name(rel: Path) -> str | None:
    """Skill dir name for a repo-relative path: nearest ancestor holding SKILL.md."""
    p = rel.parent
    while str(p) not in (".", ""):
        if (SKILLS_DIR / p / "SKILL.md").is_file():
            return p.name
        p = p.parent
    return None


def _skills_chunks() -> list[dict]:
    """Walk the mounted skills dir, chunk .md files.

    Reed 2026-08-09: ONLY custom skills go in the RAG corpus — skip anything in
    .bundled_manifest (ships with Hermes) and all Hermes internal dot-dirs
    (.archive, .curator_backups, .curator_state, .usage.json, ...).
    """
    out: list[dict] = []
    if not SKILLS_DIR.is_dir():
        return out
    for md in sorted(SKILLS_DIR.rglob("*.md")):
        relp = md.relative_to(SKILLS_DIR)
        if any(p.startswith(".") for p in relp.parts):
            continue
        name = _skill_name(relp)
        if name is None or name in _BUNDLED_SKILLS:
            # None = not part of any skill (category DESCRIPTION.md, stray files)
            continue
        rel = str(relp)
        try:
            text = md.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if len(text.strip()) < 80:
            continue
        for i, piece in enumerate(_split_markdown_para(text)):
            out.append({
                "id": f"skill:{rel}:{i}",
                "title": rel,
                "text": piece,
                "meta": {"skill_path": rel, "chunk": i},
            })
    return out


_REBUILD_BATCH = 250


async def _rebuild_source(source: str, chunks: list[dict], *, preserve_live: bool = False) -> int:
    """Delete a source's backfill chunks, then re-embed + insert in batches.

    Batched (and yielding to the event loop between batches) so a huge rebuild
    never starves /health: a single unbatched 13k-chunk embed blocked the loop
    past the 5s liveness deadline and the kubelet killed the pod mid-refresh
    (hit 2026-08-09, v1.2.2 — skills were left deleted). preserve_live keeps
    live:* session chunks (only backfill 'session:%' ids are deleted).
    """
    if not chunks:
        return 0
    conn = _db()
    if preserve_live:
        conn.execute("DELETE FROM chunks WHERE source=? AND id LIKE ?", (source, f"{source}:%"))
    else:
        conn.execute("DELETE FROM chunks WHERE source=?", (source,))
    conn.commit()
    conn.close()

    conn = _db()
    now = time.time()
    n = 0
    for start in range(0, len(chunks), _REBUILD_BATCH):
        batch = chunks[start:start + _REBUILD_BATCH]
        vecs = await _embed_async([c["text"] for c in batch])
        for c, v in zip(batch, vecs):
            cid = c["id"]
            conn.execute(
                "INSERT INTO chunks (id, source, consumer, title, text, meta, vec, updated_at) VALUES (?,?,?,?,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET consumer=excluded.consumer, title=excluded.title, text=excluded.text, meta=excluded.meta, "
                "vec=excluded.vec, updated_at=excluded.updated_at",
                (cid, source, c.get("consumer"), c.get("title") or "", c["text"], json.dumps(c.get("meta") or {}), _vec_to_blob(v), now),
            )
        conn.commit()
        n += len(batch)
        await asyncio.sleep(0.05)  # let /health + probes breathe between batches
    conn.close()
    return n


async def _refresh_sources():
    """Rebuild skills + sessions from the mounted /sources dir (NFS)."""
    results = {}

    # skills — shared knowledge (consumer NULL)
    skill_chunks = _skills_chunks()
    if skill_chunks:
        results["skill"] = await _rebuild_source("skill", skill_chunks)
    else:
        results["skill"] = "SKIP (no /sources/skills mount)"

    # sessions — snapshot DB from NFS, tagged with SESSION_CONSUMER.
    # Only the backfill chunks (id 'session:<sid>:<i>') are rebuilt here; live
    # chunks (id 'live:<consumer>:<ts>', pushed by llm-gateway in real time)
    # are preserved — they're the freshest session memory.
    if SESSION_DB.is_file():
        sess_chunks = _session_chunks(SESSION_DB, SESSION_CONSUMER)
        if sess_chunks:
            results["session"] = await _rebuild_source("session", sess_chunks, preserve_live=True)
    else:
        results["session"] = "SKIP (no /sources/hermes-state.db mount)"

    return {"status": "ok", **results}


@app.post("/refresh")
async def refresh(request: Request):
    """Re-clone homelab-docs and/or reindex from the mounted /sources dir.

    Body flags: {"docs": true} rebuilds docs (git clone);
    {"sources": true} rebuilds skills + sessions from the NFS mount (skills
    symlink + hermes-state.db snapshot) — replaces the old host exporters.
    """
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    if body.get("docs"):
        return await _refresh_docs()
    if body.get("sources"):
        return await _refresh_sources()
    return {"status": "ok", "hint": 'use {"docs": true} or {"sources": true}'}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
