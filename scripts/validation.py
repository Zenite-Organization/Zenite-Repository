import os, sys, uuid, asyncio
from sqlalchemy import create_engine, text

# Rode a partir da raiz do repo:
# python scripts/run_holdout_validation_mysql.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ai.dtos.issues_estimation_dto import IssueEstimationDTO
from application.services.estimation_service import EstimationService


def get_engine():
    db_url = os.getenv("DATABASE_URL")  # mysql+pymysql://user:pass@host:3306/db
    if not db_url:
        raise RuntimeError("Defina DATABASE_URL (mysql+pymysql://...)")
    return create_engine(db_url, pool_pre_ping=True)


def build_dto_from_row(r: dict) -> IssueEstimationDTO:
    """
    Usa APENAS os campos do seu SELECT:
    id, project_id, title, description_text, type, story_point, assignee_id

    Como o DTO do projeto é mais geral, a gente preenche o mínimo e "embute"
    type/story_point/assignee_id no texto (sem vazamento de campos reais).
    """
    issue_id = int(r["id"])
    project_id = str(r.get("project_id"))

    title = (r.get("title") or "").strip()
    description_text = (r.get("description_text") or "").strip()
    issue_type = (r.get("type") or "").strip()
    #story_point = r.get("story_point")
    assignee_id = r.get("assignee_id")

    # Monta um "contexto" só com o que você forneceu
    extra_context_lines = [
        f"ProjectId: {project_id}",
        f"Type: {issue_type}" if issue_type else "Type: (null)",
        #f"StoryPoint: {story_point}" if story_point is not None else "StoryPoint: (null)",
        f"AssigneeId: {assignee_id}" if assignee_id is not None else "AssigneeId: (null)",
    ]
    extra_context = "\n".join(extra_context_lines)

    # Conteúdo final SEM campos leakage
    final_description = (
        f"{extra_context}\n\n"
        f"Description:\n{description_text if description_text else '(empty)'}"
    )

    # Preenche somente o necessário (o resto fica neutro)
    return IssueEstimationDTO(
        issue_number=issue_id,              # não existe issue_number no seu SELECT -> usamos id
        repository=f"project:{project_id}", # não existe repository -> colocamos um identificador neutro
        issue_type=issue_type,
        title=title,
        description=final_description,
        labels=[],                          # não fornecido
        assignees=[],                       # não fornecido (assignee_id já vai no texto)
        state="open",                     # pelo seu filtro, são issues finalizadas
        is_open=False,
        comments_count=0,
        age_in_days=0,
        author_login="unknown",
        author_role="NONE",
        repo_language=None,
        repo_size=None,
        is_estimation_issue=False,
        has_assignee=(assignee_id is not None),
        has_description=bool(description_text),
    )



def fetch_issues_for_validation(engine, project_id: int, limit: int) -> list[dict]:
    # Sua query, mantendo os filtros e o ROW_NUMBER
    sql = text("""
        SELECT
          id,
          project_id,
          title,
          description_text,
          type,
          assignee_id
        FROM (
            SELECT
              x.id,
              x.project_id,
              x.title,
              x.description_text,
              x.type,
              x.assignee_id,
              ROW_NUMBER() OVER (
                PARTITION BY project_id
                ORDER BY id ASC
              ) AS rn
            FROM issue x
            WHERE
                x.resolution_date IS NOT NULL
                AND x.status IN ('Closed','Done','Resolved','Complete')
                AND x.resolution IN ('Fixed','Done','Complete','Completed','Works as Designed')
                AND x.total_effort_minutes >= 15
                AND x.project_id = :project_id
        ) i
        ORDER BY rn
        LIMIT :limit
    """)

    with engine.connect() as conn:
        rows = conn.execute(sql, {"project_id": project_id, "limit": limit}).mappings().all()
        return [dict(r) for r in rows]


async def main():
    engine = get_engine()

    project_id = int(os.getenv("PROJECT_ID"))
    limit = int(os.getenv("BATCH_LIMIT"))
    run_id = os.getenv("VALIDATION_RUN_ID", str(uuid.uuid4()))
    model_version = os.getenv("MODEL_VERSION")

    issues = fetch_issues_for_validation(engine, project_id=project_id, limit=limit)
    if not issues:
        print("Nenhuma issue encontrada com os filtros informados.")
        return

    svc = EstimationService()

    upsert_sql = text("""
        INSERT INTO issue_estimation_validation
          (validation_run_id, project_key, issue_id, issue_number, repository,
           predicted_hours, confidence, justification, model_version, predicted_at)
        VALUES
          (:validation_run_id, :project_key, :issue_id, :issue_number, :repository,
           :predicted_hours, :confidence, :justification, :model_version, NOW())
        ON DUPLICATE KEY UPDATE
          predicted_hours = VALUES(predicted_hours),
          confidence        = VALUES(confidence),
          justification     = VALUES(justification),
          model_version     = VALUES(model_version),
          predicted_at      = NOW()
    """)

    print(f"run_id={run_id} | project_id={project_id} | issues={len(issues)}")

    with engine.begin() as conn:
        for r in issues:
            dto = build_dto_from_row(r)
            state = await svc.run(dto)
            final_estimation = state.get("final_estimation", {}) or {}

            predicted_hours = final_estimation.get("estimate_hours")

            conn.execute(upsert_sql, {
                "validation_run_id": run_id,
                "project_key": str(r["project_id"]),        # sua tabela usa project_key, aqui vai o project_id
                "issue_id": int(r["id"]),
                "issue_number": int(r["id"]),              # opcional; como não existe, repetimos id
                "repository": f"project:{r['project_id']}",

                "predicted_hours": predicted_hours,
                "confidence": final_estimation.get("confidence"),
                "justification": final_estimation.get("justification", ""),
                "model_version": model_version,
            })

            print(f"[OK] issue_id={r['id']} predicted_hours={predicted_hours}")

    print("Fim.")


if __name__ == "__main__":
    asyncio.run(main())