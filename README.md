# 🧠 ZeniteAI

O **ZeniteAI** é um sistema modular em **Python** que utiliza **LangGraph**, **FastAPI** e uma **arquitetura hexagonal** para criar agentes inteligentes capazes de **triagem, análise e estimativa automática de tarefas**.

O objetivo é oferecer uma plataforma **extensível e desacoplada**, permitindo integrar agentes de IA em pipelines de desenvolvimento, priorização de demandas e automação de processos.

---

## ⚙️ Tecnologias Principais

| Tecnologia | Função |
|-------------|--------|
| 🐍 **Python 3.11+** | Linguagem principal |
| ⚡ **FastAPI** | Criação de APIs REST e webhooks de alta performance |
| 🧩 **LangGraph** | Framework para criação e orquestração de agentes de IA |
| 🧠 **pydantic** | Validação e tipagem de dados |
| 🌐 **httpx / requests** | Integração com APIs externas |
| 🪶 **uvicorn** | Servidor ASGI para FastAPI |
| 🧾 **dotenv** | Gerenciamento de variáveis de ambiente |
| 🪵 **logging** | Registro estruturado e monitoramento |

---

## 🚀 Funcionalidades

✨ **Triagem Automática de Tarefas**  
> Analisa a descrição e contexto de uma tarefa para classificá-la automaticamente (prioridade, categoria, tipo).

⚖️ **Estimativa Inteligente de Esforço**  
> Gera estimativas de tempo ou pontos de história com justificativas curtas e coerentes.

🧠 **Memória Contextual**  
> Mantém histórico e contexto entre interações, melhorando a consistência das respostas.

🔄 **Workflows Modulares com LangGraph**  
> Permite montar pipelines personalizados de agentes (ex: *triagem → estimativa → validação*).

🌍 **Integrações Externas via Webhook**  
> Recebe e processa eventos assíncronos (GitHub Issues, Jira Tickets, Slack, etc).

⚙️ **API REST**  
> Endpoints simples e performáticos via FastAPI.

---

## 🧱 Estrutura do Projeto

```bash
src/
├── ai/
│   ├── agents/                # 🤖 Agentes LangGraph (triage, estimation, etc)
│   ├── memory/                # 🧠 Memória, contexto e estados
│   └── workflows/             # 🔁 Definições de fluxo (grafo)
│
├── core/
│   ├── domain/                # 🧩 Entidades e regras puras
│   │   ├── models/            # Modelos de domínio (Task, Estimate, etc)
│   │   └── interfaces/        # Contratos e abstrações (repos, services)
│   └── services/              # ⚙️ Casos de uso (ex: calcular estimativa)
│
├── infrastructure/
│   ├── api_clients/           # 🌐 Integrações externas (GraphQL, REST)
│   ├── repositories/          # 🗄️ Persistência e cache
│   └── webhook/               # 📨 Handlers de webhooks
│
├── entrypoints/
│   ├── api/                   # 🚪 Aplicação FastAPI
│   ├── routes/                # 🛣️ Endpoints (webhooks, agents)
│   └── events/                # ⚡ Consumidores de eventos assíncronos
│
├── config/                    # ⚙️ Configurações globais (env, logging)
├── main.py                    # 🏁 Ponto de entrada FastAPI
└── requirements.txt           # 📦 Dependências do projeto
