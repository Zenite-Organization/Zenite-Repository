from typing import Dict, Any, List
from ai.core.llm_client import LLMClient
from ai.core.json_utils import parse_llm_json_response
from ai.core.prompt_utils import build_system_prompt
import json


SYSTEM_ROLE = (
    "Você é um Product Owner e Scrum Master sênior, "
    "especialista em priorização de backlog e planejamento de sprint."
)

INSTRUCTION = """
Você recebeu uma lista de tarefas atribuídas a UM único usuário.

Seu objetivo é PRIORIZAR essas tarefas para planejamento de sprint.

Considere obrigatoriamente:
- valor de negócio
- urgência
- impacto no fluxo principal
- risco técnico
- dependências implícitas entre tarefas

IMPORTANTE:
- Compare SOMENTE as tarefas desta lista entre si
- Não compare com tarefas de outros usuários
- A prioridade é RELATIVA dentro do conjunto

Retorne APENAS um JSON válido no formato:

{
  "user": "<login_usuario>",
  "tasks": [
    {
      "task_id": "123",
      "priority": "alta | media | baixa",
      "reason": "justificativa curta"
    }
  ]
}

Regras:
- Tarefas bloqueadoras tendem a prioridade "alta"
- Tarefas de manutenção isoladas tendem a "baixa"
- Use "media" quando não houver forte evidência
- Não inclua texto fora do JSON
"""

def run_task_prioritization_for_user(
    user_login: str,
    tasks: List[Dict[str, Any]],
    llm: LLMClient,
    sprint_context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Prioriza tarefas atribuídas a um único usuário.
    """

    system_prompt = build_system_prompt(SYSTEM_ROLE, INSTRUCTION)

    # Normaliza as tarefas para o prompt
    tasks_payload = []
    for t in tasks:
        tasks_payload.append({
            "issue_number": t.get("issue_number"),
            "title": t.get("title"),
            "description": t.get("description"),
            "labels": t.get("labels", []),
            "repository": t.get("repository"),
        })

    sprint_info = ""
    if sprint_context:
        sprint_title = sprint_context.get("title") or sprint_context.get("sprint_title") or sprint_context.get("name")
        sprint_desc = sprint_context.get("description")
        sprint_labels = sprint_context.get("labels")
        sprint_info = "\n=== CONTEXTO DO SPRINT ===\n"
        if sprint_title:
            sprint_info += f"Título: {sprint_title}\n"
        if sprint_desc:
            sprint_info += f"Descrição: {sprint_desc}\n"
        if sprint_labels:
            sprint_info += f"Labels: {', '.join(sprint_labels)}\n"

    prompt = (
        system_prompt
        + sprint_info
        + f"\n=== USUÁRIO ===\n{user_login}\n"
        + "\n=== TAREFAS DO USUÁRIO ===\n"
        + json.dumps(tasks_payload, indent=2, ensure_ascii=False)
    )


    response = llm.send_prompt(
        prompt,
        temperature=0.0,
        max_tokens=600
    )

    try:
        parsed = parse_llm_json_response(response)

        # Garantia mínima de contrato
        if "tasks" not in parsed:
            raise ValueError("Campo 'tasks' ausente no JSON")

        return parsed

    except Exception as e:
        return {
            "user": user_login,
            "tasks": [],
            "error": str(e),
            "raw_response": response
        }
