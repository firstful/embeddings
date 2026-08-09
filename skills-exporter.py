#!/usr/bin/env python3
"""embeddings-skills-exporter — push Hermes skill markdown into the embeddings service.

Walks ~/.hermes/skills (excluding .archive), chunks every .md file (SKILL.md +
references/ + supporting docs) into ~1200-char windows, and POSTs to the
embeddings service /ingest with source=skill, consumer=None (shared knowledge).

Skills are the SINGLE SOURCE OF TRUTH for operational procedures — this ingests
them directly so the RAG index never drifts from the skill content (the earlier
port-to-docs approach created a parallel copy; this supersedes it for coverage).

Runs on sutliffserver via cron (daily). Full re-push each run is idempotent
(chunk-id upserts: skill:<relpath>:<i>).
"""

import json
import os
import re
import sys
import time
import urllib.request
from http.client import HTTPConnection

SKILLS_ROOT = "/root/.hermes/skills"
EMBEDDINGS_URL = os.environ.get("EMBEDDINGS_URL", "http://embeddings.homelab.internal")
CHUNK_CHARS = 1200
BATCH = 50  # chunks per POST


# NAT64/DNS64 gotcha: force IPv4 like curl -4 (host resolver returns IPv6 first).
class _IPv4HTTPSConnection(HTTPConnection):
    def connect(self):
        self.sock = __import__("socket").create_connection((self.host, self.port), self.timeout, source_address=("0.0.0.0", 0))


class _IPv4HTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return self.do_open(_IPv4HTTPSConnection, req)


_OPENER = urllib.request.build_opener(_IPv4HTTPHandler)


def _split_markdown(text: str, min_chars: int = 150, max_chars: int = 1800) -> list[str]:
    """Split markdown by headings then paragraph boundaries. Mirrors the service's chunker."""
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


def _bundled_skill_names() -> set[str]:
    """Skill names that ship with Hermes, from .bundled_manifest (name:hash per line)."""
    out: set[str] = set()
    try:
        with open(os.path.join(SKILLS_ROOT, ".bundled_manifest"), encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if ":" in line:
                    out.add(line.split(":", 1)[0])
    except OSError:
        pass
    return out


_BUNDLED_SKILLS = _bundled_skill_names()


def _skill_name(relpath: str) -> str | None:
    """Skill dir name for a repo-relative path: nearest ancestor holding SKILL.md."""
    p = os.path.dirname(relpath)
    while p and p != ".":
        if os.path.isfile(os.path.join(SKILLS_ROOT, p, "SKILL.md")):
            return os.path.basename(p)
        p = os.path.dirname(p)
    return None


def _iter_md():
    """Yield (relpath, text) for every .md under SKILLS_ROOT.

    Reed 2026-08-09: custom skills ONLY — skip bundled skills (.bundled_manifest)
    and Hermes internal dot-dirs (.archive, .curator_backups, ...).
    NOTE: superseded as the active ingest path by the pod's /refresh
    ({"sources": true}, 2h cadence) — kept for manual/legacy use.
    """
    for root, dirs, files in os.walk(SKILLS_ROOT):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in sorted(files):
            if not f.endswith(".md"):
                continue
            p = os.path.join(root, f)
            rel = os.path.relpath(p, SKILLS_ROOT)
            name = _skill_name(rel)
            if name is None or name in _BUNDLED_SKILLS:
                # None = not part of any skill (category DESCRIPTION.md, stray files)
                continue
            try:
                text = open(p, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            if len(text.strip()) < 80:
                continue
            yield rel, text


def _post(chunks: list[dict]):
    body = json.dumps({"source": "skill", "chunks": chunks}).encode()
    req = urllib.request.Request(
        f"{EMBEDDINGS_URL}/ingest",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with _OPENER.open(req, timeout=180) as resp:
        data = json.loads(resp.read())
    if data.get("status") != "ok":
        raise RuntimeError(f"ingest failed: {data}")


def main():
    pending = []
    total_chunks = 0
    total_files = 0
    for rel, text in _iter_md():
        total_files += 1
        for i, piece in enumerate(_split_markdown(text)):
            pending.append({
                "id": f"skill:{rel}:{i}",
                "title": rel,
                "text": piece,
                "meta": {"skill_path": rel, "chunk": i},
            })
            if len(pending) >= BATCH:
                _post(pending)
                total_chunks += len(pending)
                pending = []
    if pending:
        _post(pending)
        total_chunks += len(pending)
    print(f"pushed {total_files} skill files -> {total_chunks} chunks (source=skill)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
