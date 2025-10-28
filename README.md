src/
├── ai/
│   ├── agents/                # Agentes LangGraph (triage, estimation, etc)
│   ├── memory/                # Memória, contexto e estados
│   └── workflows/             # Definições de fluxo (grafo)
│
├── core/
│   ├── domain/                # Entidades e regras puras
│   │   ├── models/
│   │   └── interfaces/
│   └── services/              # Casos de uso (ex: calcular estimativa)
│
├── infrastructure/
│   ├── api_clients/           # Implementações GraphQL/REST
│   ├── repositories/          # Persistência (se houver)
│   └── webhook/               # Handlers de webhooks
│
├── entrypoints/
│   ├── api/                   # FastAPI app
│   ├── routes/                # Endpoints (webhooks, agents)
│   └── events/                # Recebe eventos assíncronos (ex: GitHub webhook)
│
├── config/                    # Configurações, env, logging
├── main.py                    # Ponto de entrada FastAPI
└── requirements.txt