# ZeniteAI

ZeniteAI e um sistema modular em Python com FastAPI + LangGraph para automacao de triagem, estimativa e planejamento de tarefas de engenharia.

## Arquitetura

O projeto segue uma abordagem hexagonal pragmatica:

- `web`: camada HTTP (entrada/saida, validacao, headers, status code)
- `application`: casos de uso e orquestracao de fluxo
- `domain`: regras puras e tipos de decisao
- `clients` / `ai.core`: adaptadores de infraestrutura (GitHub, LLM, GraphQL, etc)
- `ai.workflows`: grafos LangGraph usados pelos casos de uso

## Estrutura atual

```bash
src/
├── main.py
├── domain/
│   ├── webhook_models.py
│   └── webhook_rules.py
├── application/
│   ├── services/
│   │   ├── estimation_service.py
│   │   └── sprint_planning_service.py
│   └── use_cases/
│       ├── handle_github_webhook.py
│       ├── run_issue_estimation.py
│       └── run_sprint_planning.py
├── web/
│   ├── schemas/
│   │   └── github_payloads.py
│   └── routes/
│       └── github_webhook.py
├── clients/
│   └── github/
│       ├── github_auth.py
│       ├── github_graphql.py
│       ├── github_provider.py
│       └── graphql/
│           ├── comments.py
│           └── projects.py
└── ai/
    ├── agents/
    ├── core/
    ├── dtos/
    └── workflows/
```

## Webhook GitHub

Endpoint:

- `POST /webhook/github`

Contrato de resposta:

```json
{
  "status": "ignored | processed | error",
  "event": "issues",
  "action": "opened | edited | labeled",
  "flow": "estimation | planning | none",
  "details": {}
}
```

Fluxos:

- `Planning` label -> executa planejamento de sprint
- `Estimate` label -> executa estimativa de esforco
- apos processar a estimativa, a label `Estimate` e removida automaticamente da issue
- ao fechar uma issue (`action=closed`), o Zenite gera embedding (titulo+descricao) e faz upsert no Pinecone (RAG)
- sem labels de controle -> ignora webhook

## Setup rapido

1. Python 3.11+
2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

3. Configurar `.env`:

- `GITHUB_WEBHOOK_SECRET`
- `GEMINI_API_KEY`
- `APP_ID`
- `APP_PRIVATE_KEY` ou `APP_PRIVATE_KEY_path`
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`
- `OPENAI_API_KEY_RAG`

4. Rodar API:

```bash
python src/main.py
```
