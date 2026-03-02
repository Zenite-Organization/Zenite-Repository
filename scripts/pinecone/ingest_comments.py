import os
import json
import time
from typing import Dict, Any, List, Tuple, Optional

import mysql.connector
from mysql.connector.cursor import MySQLCursorDict

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from tqdm import tqdm

# =========================
# Config / Env
# =========================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "tawos-rag")

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASS = os.getenv("MYSQL_PASS", "")
MYSQL_DB = os.getenv("MYSQL_DB", "tawos")

EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

MYSQL_FETCH_SIZE = int(os.getenv("MYSQL_FETCH_SIZE", "2000"))
UPSERT_BATCH = int(os.getenv("UPSERT_BATCH", "200"))
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "200"))

CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", ".checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "20000"))  # evita estourar 8192 tokens

# =========================
# Clients
# =========================
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY não encontrada no .env")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY não encontrada no .env")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# =========================
# SQL
# =========================
GET_PROJECT_KEY_SQL = """
SELECT Project_Key
FROM Project
WHERE ID = %s;
"""

COMMENTS_BY_PROJECT_SQL = """
WITH issue_ids AS (
	SELECT
	  id as issue_id
	FROM issue i
	WHERE
	  i.resolution_date IS NOT NULL
	  AND i.total_effort_minutes >= 15
	  AND i.status IN ('Closed','Done','Resolved','Complete')
	  AND i.resolution IN ('Fixed','Done','Complete','Completed','Works as Designed')
	  AND project_id = %s
)

SELECT
  c.id,
  c.comment_text,
  c.creation_date,
  c.author_id,
  c.issue_id,
  p.ID AS project_id,
  p.Project_Key AS project_key,
  i.Issue_Key AS issue_key
FROM comment c
JOIN Issue i ON i.ID = c.issue_id
JOIN Project p ON p.ID = i.Project_ID
WHERE
  c.issue_id IN (SELECT * FROM issue_ids)
ORDER BY c.id;
"""

# =========================
# Helpers
# =========================
def mysql_connect():
    return mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASS,
        database=MYSQL_DB,
    )

def retry(func, max_tries=5, base_sleep=1.0, what="op"):
    for attempt in range(1, max_tries + 1):
        try:
            return func()
        except Exception as e:
            if attempt == max_tries:
                raise
            sleep_s = base_sleep * (2 ** (attempt - 1))
            print(f"[WARN] Falha em {what} (tentativa {attempt}/{max_tries}): {e}")
            print(f"[WARN] Aguardando {sleep_s:.1f}s e tentando novamente...")
            time.sleep(sleep_s)

def norm(s: Optional[str]) -> str:
    if not s:
        return ""
    return s.replace("\r\n", "\n").strip()

def safe_iso(dt) -> Optional[str]:
    if dt is None:
        return None
    try:
        return dt.isoformat()
    except Exception:
        return str(dt)

def clamp_text(s: str, max_chars: int = MAX_TEXT_CHARS) -> str:
    if not s:
        return ""
    s = s.strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n\n[TRUNCATED]"

def drop_nulls(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}

def get_project_key(project_id: int) -> str:
    conn = mysql_connect()
    cur = conn.cursor()
    cur.execute(GET_PROJECT_KEY_SQL, (project_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row or not row[0]:
        raise RuntimeError(f"Project_ID={project_id} não encontrado na tabela Project")
    return str(row[0])

def namespace_for_comments(project_key: str) -> str:
    return f"{project_key.strip().lower()}_comments"

def checkpoint_path(project_id: int) -> str:
    # checkpoint por ID (pra não depender de key)
    return os.path.join(CHECKPOINT_DIR, f"project_{project_id}_comments.json")

def load_checkpoint(project_id: int) -> Dict[str, Any]:
    path = checkpoint_path(project_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"last_comment_id": 0, "ingested": 0}

def save_checkpoint(project_id: int, last_comment_id: int, ingested: int) -> None:
    path = checkpoint_path(project_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"last_comment_id": last_comment_id, "ingested": ingested}, f, ensure_ascii=False, indent=2)

def build_comment_text(r: Dict[str, Any]) -> str:
    parts = [
        f"[Project] {r.get('project_key')} (ID={r.get('project_id')})",
        f"[Issue] {r.get('issue_key')} (issue_id={r.get('issue_id')})",
        f"[CommentID] {r.get('id')}",
        f"[AuthorID] {r.get('author_id')}",
        f"[CreatedAt] {safe_iso(r.get('creation_date'))}",
        "",
        norm(r.get("comment_text")),
    ]
    full_text = "\n".join([p for p in parts if p is not None]).strip()
    return clamp_text(full_text)

def build_comment_metadata(r: Dict[str, Any]) -> Dict[str, Any]:
    return drop_nulls({
        "doc_type": "comment",
        "source": "TAWOS",

        "project_id": int(r["project_id"]) if r.get("project_id") is not None else None,
        "project_key": r.get("project_key"),

        "issue_id": int(r["issue_id"]) if r.get("issue_id") is not None else None,
        "issue_key": r.get("issue_key"),

        "comment_id": int(r["id"]) if r.get("id") is not None else None,
        "author_id": int(r["author_id"]) if r.get("author_id") is not None else None,
        "creation_date": safe_iso(r.get("creation_date")),
    })

def embed_texts(texts: List[str]) -> List[List[float]]:
    def _call():
        resp = openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
        return [d.embedding for d in resp.data]
    return retry(_call, what="OpenAI embeddings")

def pinecone_upsert(payload: List[Dict[str, Any]], namespace: str) -> None:
    def _call():
        index.upsert(vectors=payload, namespace=namespace)
    retry(_call, what="Pinecone upsert")

# =========================
# Ingestion
# =========================
def ingest_project_comments(project_id: int) -> None:
    project_key = get_project_key(project_id)
    ns = namespace_for_comments(project_key)

    ckpt = load_checkpoint(project_id)
    last_comment_id = int(ckpt.get("last_comment_id", 0))
    ingested = int(ckpt.get("ingested", 0))

    print(f"\n=== Ingest Comments | ProjectID={project_id} ({project_key}) | Namespace={ns} | Resume from comment_id>{last_comment_id} ===")

    conn = mysql_connect()
    cur: MySQLCursorDict = conn.cursor(dictionary=True)
    cur.execute(COMMENTS_BY_PROJECT_SQL, (project_id,))

    batch_docs: List[Tuple[str, str, Dict[str, Any], int]] = []  # (doc_id, text, meta, comment_id)
    pbar = tqdm(desc=f"ProjectID={project_id} comments", unit="comment")

    while True:
        rows = cur.fetchmany(MYSQL_FETCH_SIZE)
        if not rows:
            break

        for r in rows:
            cid = int(r["id"])
            if cid <= last_comment_id:
                continue

            text = build_comment_text(r)
            if not text:
                last_comment_id = cid
                save_checkpoint(project_id, last_comment_id, ingested)
                continue

            meta = build_comment_metadata(r)
            doc_id = f"comment:{cid}"

            batch_docs.append((doc_id, text, meta, cid))

            if len(batch_docs) >= UPSERT_BATCH:
                flush_batch(batch_docs, ns)
                last_comment_id = max(x[3] for x in batch_docs)
                ingested += len(batch_docs)
                save_checkpoint(project_id, last_comment_id, ingested)
                pbar.update(len(batch_docs))
                batch_docs = []

    if batch_docs:
        flush_batch(batch_docs, ns)
        last_comment_id = max(x[3] for x in batch_docs)
        ingested += len(batch_docs)
        save_checkpoint(project_id, last_comment_id, ingested)
        pbar.update(len(batch_docs))

    pbar.close()
    cur.close()
    conn.close()

    print(f"✅ Done: ProjectID={project_id} ({project_key}) | comments_ingested={ingested} | last_comment_id={last_comment_id} | namespace={ns}")

def flush_batch(batch_docs: List[Tuple[str, str, Dict[str, Any], int]], namespace: str) -> None:
    payload_all: List[Dict[str, Any]] = []
    i = 0
    while i < len(batch_docs):
        chunk = batch_docs[i:i+EMBED_BATCH]
        ids = [x[0] for x in chunk]
        texts = [x[1] for x in chunk]
        metas = [x[2] for x in chunk]

        vectors = embed_texts(texts)
        for _id, vec, meta in zip(ids, vectors, metas):
            payload_all.append({"id": _id, "values": vec, "metadata": meta})

        i += EMBED_BATCH

    pinecone_upsert(payload_all, namespace)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest TAWOS Comments into Pinecone (by project).")
    parser.add_argument("--project-id", type=int, required=True, help="Project ID (ex.: 34)")
    args = parser.parse_args()
    ingest_project_comments(args.project_id)
