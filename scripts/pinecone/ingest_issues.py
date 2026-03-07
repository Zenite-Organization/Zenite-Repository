import os
import json
import time
import re

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
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# MySQL (ajuste no .env se quiser)
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_PORT = int(os.getenv("MYSQL_PORT"))
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASS = os.getenv("MYSQL_PASS")
MYSQL_DB = os.getenv("MYSQL_DB")

# Embedding
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# Batches
MYSQL_FETCH_SIZE = int(os.getenv("MYSQL_FETCH_SIZE", "2000"))   # quantas rows puxar por vez
UPSERT_BATCH = int(os.getenv("UPSERT_BATCH", "200"))           # quantos docs por upsert
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "200"))             # quantos textos por chamada de embedding (<= UPSERT_BATCH)

# Checkpoint
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", ".checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "20000"))  # ~bem abaixo de 8192 tokens na prática


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
LIST_PROJECTS_SQL = """
SELECT ID, Project_Key
FROM Project
ORDER BY ID;
"""

# Issue fields principais
ISSUES_BY_PROJECT_SQL = """
SELECT
  i.ID AS issue_id,
  i.issue_key,
  p.ID as project_id,
  p.project_key,
  i.title,
  i.description_text,
  i.Type AS issue_type,
  i.priority,
  cast((i.resolution_time_minutes/60) as UNSIGNED) as resolution_time_hours,
  cast((i.total_effort_minutes/60) as UNSIGNED) as total_effort_hours,
  i.creator_id, 
  i.reporter_id, 
  i.assignee_id
FROM issue i
JOIN Project p ON p.ID = i.Project_ID
WHERE
  i.resolution_date IS NOT NULL
  AND i.status IN ('Closed','Done','Resolved','Complete')
  AND i.resolution IN ('Fixed','Done','Complete','Completed','Works as Designed')
  AND cast((x.total_effort_minutes/60) as SIGNED)  BETWEEN 1 AND 300
  AND length(description_text) >= 100 
  AND i.project_id = %s
ORDER BY i.ID
LIMIT 5;
"""

# =========================
# Helpers
# =========================
def norm(s: Optional[str]) -> str:
    if not s:
        return ""
    return s.replace("\r\n", "\n").strip()

def clamp_text(s: str, max_chars: int = MAX_TEXT_CHARS) -> str:
    """Evita textos gigantes que estouram limite de tokens do embedding."""
    if not s:
        return ""
    s = s.strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n\n[TRUNCATED]"


def safe_iso(dt) -> Optional[str]:
    if dt is None:
        return None
    try:
        return dt.isoformat()
    except Exception:
        return str(dt)
    
def drop_nulls(d: Dict[str, Any]) -> Dict[str, Any]:
    """Remove chaves com valor None (Pinecone não aceita null em metadata)."""
    return {k: v for k, v in d.items() if v is not None}


def strip_wrapping_quotes(s: str) -> str:
    """
    Limpa:
    - Aspas envolventes simples, duplas ou triplas
    - Sequências escapadas como \" ou \"\"\" 
    - Aspas duplicadas no meio da string
    """

    if not s:
        return ""

    s = s.strip()

    # 1️⃣ Remove escapes tipo \" -> "
    s = s.replace('\\"', '"')
    s = s.replace("\\'", "'")

    # 2️⃣ Remove múltiplas aspas duplicadas (""" -> ")
    s = re.sub(r'"{2,}', '"', s)
    s = re.sub(r"'{2,}", "'", s)

    # 3️⃣ Remove aspas envolventes repetidamente
    while True:
        if (s.startswith('"""') and s.endswith('"""')) or \
           (s.startswith("'''") and s.endswith("'''")):
            s = s[3:-3].strip()
            continue

        if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
            s = s[1:-1].strip()
            continue

        break

    return s

def project_namespace(project_key: str) -> str:
    # namespace padronizado (minúsculo)
    return f"{project_key.strip().lower()}_issues"

def checkpoint_path(project_key: str) -> str:
    return os.path.join(CHECKPOINT_DIR, f"{project_key.lower()}_issues.json")

def load_checkpoint(project_key: str) -> Dict[str, Any]:
    path = checkpoint_path(project_key)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"last_issue_id": 0, "ingested": 0}

def save_checkpoint(project_key: str, last_issue_id: int, ingested: int) -> None:
    path = checkpoint_path(project_key)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"last_issue_id": last_issue_id, "ingested": ingested}, f, ensure_ascii=False, indent=2)

def build_issue_text(r: Dict[str, Any]) -> str:
    # Texto canônico (o que o LLM vai ler)
    parts = [
        f"[Title] {strip_wrapping_quotes(norm(r.get('title')))}",
        f"[Description] {strip_wrapping_quotes(norm(r.get("description_text")))}",
    ]
    full_text = "\n".join([p for p in parts if p is not None]).strip()
    return clamp_text(full_text)


def build_issue_metadata(r: Dict[str, Any]) -> Dict[str, Any]:
    return drop_nulls({
        "doc_type": "issue",
        "source": "TAWOS",

        # Projeto / Issue
        "issue_id": int(r["issue_id"]),
        "issue_title": strip_wrapping_quotes(norm(r.get('title'))),

        # Classificações
        "issue_type": r.get("issue_type"),
        "priority": r.get("priority"),

        # Métricas
        "total_effort_hours": float(r["total_effort_hours"]) if r.get("total_effort_hours") is not None else None,
        "resolution_time_hours": float(r["resolution_time_hours"]) if r.get("resolution_time_hours") is not None else None,

        # Pessoas / vínculos
        "creator_id": int(r["creator_id"]) if r.get("creator_id") is not None else None,
        "reporter_id": int(r["reporter_id"]) if r.get("reporter_id") is not None else None,
        "assignee_id": int(r["assignee_id"]) if r.get("assignee_id") is not None else None,

        # Descricao
        "description": strip_wrapping_quotes(norm(r.get('description_text')))[:1000],
    })


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

def embed_texts(texts: List[str]) -> List[List[float]]:
    # Chamada batch ao endpoint de embeddings
    def _call():
        resp = openai_client.embeddings.create(
            model=EMBED_MODEL,
            input=texts
        )
        # garante a mesma ordem
        return [d.embedding for d in resp.data]

    return retry(_call, what="OpenAI embeddings")

def pinecone_upsert(payload: List[Dict[str, Any]], namespace: str) -> None:
    def _call():
        index.upsert(vectors=payload, namespace=namespace)
    retry(_call, what="Pinecone upsert")

# =========================
# DB utils
# =========================
def mysql_connect():
    return mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASS,
        database=MYSQL_DB,
    )

def list_projects() -> List[str]:
    conn = mysql_connect()
    cur = conn.cursor()
    cur.execute(LIST_PROJECTS_SQL)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [r[0] for r in rows if r and r[0]]

# =========================
# Ingestion
# =========================
def ingest_project_issues(project_id: int) -> None:
    ns = None  # vamos definir depois que lermos a 1ª linha (para pegar project_key do banco)
    ckpt_key = f"project_{project_id}"  # usado só para o checkpoint (pra não depender do project_key)
    ckpt = load_checkpoint(ckpt_key)
    last_id = int(ckpt.get("last_issue_id", 0))
    ingested = int(ckpt.get("ingested", 0))

    print(f"\n=== Ingest Issues | ProjectID={project_id} | Namespace=(auto) | Resume from ID>{last_id} ===")

    conn = mysql_connect()
    cur: MySQLCursorDict = conn.cursor(dictionary=True)

    # Puxa tudo do projeto, mas vamos pular os IDs já processados via checkpoint
    cur.execute(ISSUES_BY_PROJECT_SQL, (project_id,))

    buffer: List[Dict[str, Any]] = []
    batch_docs: List[Tuple[str, str, Dict[str, Any], int]] = []  # (id, text, meta, issue_id)

    pbar = tqdm(desc=f"ProjectID={project_id} issues", unit="issue")


    while True:
        rows = cur.fetchmany(MYSQL_FETCH_SIZE)
        if not rows:
            break

        for r in rows:
            issue_id = int(r["issue_id"])
            if issue_id <= last_id:
                continue

            if ns is None:
                ns = project_namespace(r["project_key"])


            text = build_issue_text(r)
            if not text:
                # mesmo assim atualiza checkpoint pro ID, pra não travar em registro ruim
                last_id = issue_id
                save_checkpoint(ckpt_key, last_id, ingested)
                continue

            meta = build_issue_metadata(r)
            doc_id = f"issue:{issue_id}"

            batch_docs.append((doc_id, text, meta, issue_id))

            if len(batch_docs) >= UPSERT_BATCH:
                if ns is None:
                    raise RuntimeError(f"Namespace não definido (ProjectID={project_id}). Nenhum registro válido foi lido.")
                flush_batch(batch_docs, ns)
                # checkpoint no maior ID desse batch
                last_id = max(x[3] for x in batch_docs)
                ingested += len(batch_docs)
                save_checkpoint(ckpt_key, last_id, ingested)
                pbar.update(len(batch_docs))
                batch_docs = []

    # flush final
    if batch_docs:
        if ns is None:
            raise RuntimeError(f"Namespace não definido (ProjectID={project_id}). Nenhum registro válido foi lido.")
        flush_batch(batch_docs, ns)
        last_id = max(x[3] for x in batch_docs)
        ingested += len(batch_docs)
        save_checkpoint(ckpt_key, last_id, ingested)
        pbar.update(len(batch_docs))

    pbar.close()
    cur.close()
    conn.close()

    print(f"✅ Done: ProjectID={project_id} | ingested={ingested} | last_issue_id={last_id} | namespace={ns}")


def flush_batch(batch_docs: List[Tuple[str, str, Dict[str, Any], int]], namespace: str) -> None:
    # Embeddings podem ser feitos em sub-batches se UPSERT_BATCH > EMBED_BATCH
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

# =========================
# CLI entry
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest TAWOS Issues into Pinecone (by project -> namespace).")
    parser.add_argument("--project-id", type=int, help="Project_ID do TAWOS (ex.: 12). Se omitido e --all não usado, lista projetos.")
    parser.add_argument("--all", action="store_true", help="Ingerir todos os projetos (um por vez).")
    args = parser.parse_args()

    if args.all:
        projects = list_projects()
        print(f"Projetos encontrados: {len(projects)}")
        for p in projects:
            ingest_project_issues(p)
    elif args.project_id is not None:
        ingest_project_issues(args.project_id)
    else:
        projects = list_projects()
        print("Use assim:")
        print("  python ingest_issues.py --project HADOOP")
        print("  python ingest_issues.py --all")
        print("\nProjetos disponíveis (Project_Key):")
        print(", ".join(projects))
