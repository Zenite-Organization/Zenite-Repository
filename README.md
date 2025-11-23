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
│   ├── agents/                # Agentes LangGraph (triage, estimation, etc)
│   ├── memory/                # Memória, contexto e estados
│   └── workflows/             # Definições de fluxo (grafo, run_estimation_flow)
│
├── clients/
│   └── github/
│       ├── github_auth.py               # JWT + Installation token
│       ├── github_graphql.py            # requisições GraphQL
│       └── github_provider.py           # provider principal
│
├── web/
│   ├── schemas/
│   │   └── github_payload.py                   
│   └── routes/
│       └── github_webhook.py         # rota específica do GitHub
│
├── config/                    # Configurações globais (env, logging)
├── main.py
├── requirements.txt           # Dependências do projeto
