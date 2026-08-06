#!/usr/bin/env python3
"""embeddings-session-exporter — push Hermes session history into the embeddings service.

Reads ~/.hermes/state.db (sessions + messages), chunks each session's conversation
into ~1500-char windows, and POSTs to the embeddings service /ingest/sessions.
Incremental: tracks the last-synced message id in a state file.

Runs on sutliffserver via cron (host has root access to state.db). The embeddings
pod itself never touches the Hermes DB — this exporter is the only bridge.
"""

import json
import os
import socket
import sqlite3
import sys
import time
import urllib.request

STATE_DB = "/root/.hermes/state.db"
LAST_ID_FILE = "/var/lib/embeddings-sync/last_msg_id"
EMBEDDINGS_URL = os.environ.get("EMBEDDINGS_URL", "http://embeddings.homelab.internal")
CHUNK_CHARS = 1500
BATCH = 50  # chunks per POST (smaller = lower pod memory per request)


# NAT64/DNS64 gotcha: this host's resolver returns 64:ff9b:: IPv6 addrs FIRST
# and Python urllib hangs on IPv6. Force IPv4 the same way curl -4 does.
class _IPv4HTTPSConnection(urllib.request.HTTPConnection):
    def connect(self):
        self.sock = socket.create_connection((self.host, self.port), self.timeout, source_address=("0.0.0.0", 0))


class _IPv4HTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return self.do_open(_IPv4HTTPSConnection, req)


_OPENER = urllib.request.build_opener(_IPv4HTTPHandler)

def chunk_conversation(messages: list[dict]) -> list[dict]:
    """Group messages into rolling text windows; each becomes one chunk."""
    chunks = []
    buf = []
    buf_len = 0
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content") or ""
        text = f"{role}: {content}".strip()
        if not text:
            continue
        line = text + "\n"
        if buf and buf_len + len(line) > CHUNK_CHARS:
            chunks.append("".join(buf).strip())
            buf = []
            buf_len = 0
        buf.append(line)
        buf_len += len(line)
    if buf:
        chunks.append("".join(buf).strip())
    return [c for c in chunks if len(c) >= 60]


def main():
    if not os.path.exists(STATE_DB):
        print(f"error: {STATE_DB} not found", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(LAST_ID_FILE), exist_ok=True)
    last_id = 0
    if os.path.exists(LAST_ID_FILE):
        try:
            last_id = int(open(LAST_ID_FILE).read().strip())
        except ValueError:
            last_id = 0

    conn = sqlite3.connect(f"file:{STATE_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    # session meta (title, started_at) for context
    sessions = {}
    try:
        for r in conn.execute("SELECT id, title, started_at, source FROM sessions"):
            sessions[r["id"]] = {"title": r["title"] or r["id"], "started_at": r["started_at"], "source": r["source"] or ""}
    except sqlite3.Error as e:
        print(f"warning: sessions read failed: {e}", file=sys.stderr)

    rows = conn.execute(
        "SELECT id, session_id, role, content, timestamp FROM messages WHERE id > ? ORDER BY id",
        (last_id,),
    ).fetchall()
    conn.close()

    if not rows:
        print("no new messages")
        return 0

    # group by session, ordered
    by_session: dict[str, list[dict]] = {}
    for r in rows:
        by_session.setdefault(r["session_id"], []).append(
            {"id": r["id"], "role": r["role"], "content": r["content"], "timestamp": r["timestamp"]}
        )

    total_chunks = 0
    max_id = last_id
    pending = []
    for sid, msgs in by_session.items():
        meta = sessions.get(sid, {})
        chunks = chunk_conversation(msgs)
        for i, text in enumerate(chunks):
            pending.append({
                "id": f"session:{sid}:{i}",
                "title": meta.get("title") or sid,
                "text": text,
                "meta": {"session_id": sid, "session_title": meta.get("title") or "", "source_type": meta.get("source") or "", "started_at": meta.get("started_at")},
            })
        max_id = max(max_id, max(m["id"] for m in msgs))

        if len(pending) >= BATCH:
            _post(pending, EMBEDDINGS_URL)
            total_chunks += len(pending)
            pending = []

    if pending:
        _post(pending, EMBEDDINGS_URL)
        total_chunks += len(pending)

    # persist watermark AFTER successful posts
    with open(LAST_ID_FILE, "w") as f:
        f.write(str(max_id))
    print(f"pushed {len(rows)} messages -> {total_chunks} chunks (last_msg_id={max_id})")
    return 0


def _post(chunks: list[dict], base_url: str):
    body = json.dumps({"chunks": chunks}).encode()
    req = urllib.request.Request(
        f"{base_url}/ingest/sessions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with _OPENER.open(req, timeout=180) as resp:
        data = json.loads(resp.read())
    if data.get("status") != "ok":
        raise RuntimeError(f"ingest failed: {data}")


if __name__ == "__main__":
    sys.exit(main())
