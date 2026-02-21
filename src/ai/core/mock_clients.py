import re
from typing import List, Dict, Any


class MockVectorStoreClient:
    """Mock simples de um vector store: retorna um conjunto fixo de issues similares."""
    def __init__(self):
        # Tecnologias (como exibido no GitHub - languages breakdown)
        self.technologies = {
            "Python": 60,
            "JavaScript": 20,
            "Dockerfile": 5,
            "HTML": 5,
            "CSS": 5,
            "Shell": 5,
        }

        # Lista de issues mock com campos similares aos disponibilizados pelo GitHub
        self.issues = [
            {"id": "1", "number": 101, "title": "Integrar API de pagamentos", "body": "Criar integração com provedor X usando OAuth2 e webhooks.", "description": "Criar integração com provedor X", "estimated_hours": 10, "state": "closed", "labels": ["payments", "backend"], "created_at": "2025-08-01T10:12:00Z", "updated_at": "2025-08-10T12:00:00Z", "url": "https://github.com/org/repo/issues/101", "user": {"login": "alice"}, "assignees": [{"login": "bob"}], "comments": 3, "repository": "org/repo"},
            {"id": "2", "number": 102, "title": "Corrigir bug de autenticação", "body": "Token de sessão expira indevidamente causando logout do usuário.", "description": "Token expira indevidamente", "estimated_hours": 4, "state": "closed", "labels": ["bug", "auth"], "created_at": "2025-07-15T09:00:00Z", "updated_at": "2025-07-16T11:00:00Z", "url": "https://github.com/org/repo/issues/102", "user": {"login": "carol"}, "assignees": [], "comments": 2, "repository": "org/repo"},
            {"id": "3", "number": 103, "title": "Refatorar módulo de relatórios", "body": "Separar camadas, otimizar consultas e reduzir tempo de geração de relatórios.", "description": "Separar camadas e otimizar consultas", "estimated_hours": 20, "state": "closed", "labels": ["refactor", "performance"], "created_at": "2025-06-10T14:20:00Z", "updated_at": "2025-06-30T16:00:00Z", "url": "https://github.com/org/repo/issues/103", "user": {"login": "dan"}, "assignees": [{"login": "erin"}], "comments": 5, "repository": "org/repo"},
            {"id": "4", "number": 104, "title": "Adicionar testes unitários", "body": "Cobertura para serviços críticos do domínio de cobrança.", "description": "Cobertura para serviços críticos", "estimated_hours": 8, "state": "closed", "labels": ["tests", "quality"], "created_at": "2025-05-01T08:00:00Z", "updated_at": "2025-05-20T10:00:00Z", "url": "https://github.com/org/repo/issues/104", "user": {"login": "frank"}, "assignees": [], "comments": 1, "repository": "org/repo"},
            {"id": "5", "number": 105, "title": "Ajuste UI responsiva", "body": "Corrigir layout em mobile e ajustar breakpoints.", "description": "Corrigir layout em mobile", "estimated_hours": 6, "state": "closed", "labels": ["frontend", "ui"], "created_at": "2025-04-12T12:30:00Z", "updated_at": "2025-04-15T09:00:00Z", "url": "https://github.com/org/repo/issues/105", "user": {"login": "gina"}, "assignees": [{"login": "hank"}], "comments": 4, "repository": "org/repo"},
            {"id": "6", "number": 106, "title": "Adicionar suporte a múltiplas moedas", "body": "Permitir transações em USD, EUR e BRL com conversão e rounding.", "description": "Suporte a múltiplas moedas", "estimated_hours": 12, "state": "closed", "labels": ["payments", "feature"], "created_at": "2025-03-01T10:00:00Z", "updated_at": "2025-03-25T13:00:00Z", "url": "https://github.com/org/repo/issues/106", "user": {"login": "ian"}, "assignees": [{"login": "jane"}], "comments": 6, "repository": "org/repo"},
            {"id": "7", "number": 107, "title": "Melhorar logs e tracing", "body": "Adicionar structured logging e tracing distribuído para diagnósticos.", "description": "Logs e tracing", "estimated_hours": 5, "state": "closed", "labels": ["observability"], "created_at": "2025-02-18T11:40:00Z", "updated_at": "2025-02-28T12:00:00Z", "url": "https://github.com/org/repo/issues/107", "user": {"login": "kate"}, "assignees": [], "comments": 0, "repository": "org/repo"},
            {"id": "8", "number": 108, "title": "Criar endpoint de relatórios CSV", "body": "Exportar relatórios em CSV paginados para análises externas.", "description": "Endpoint CSV para relatórios", "estimated_hours": 7, "state": "closed", "labels": ["feature", "reports"], "created_at": "2025-01-10T09:10:00Z", "updated_at": "2025-01-20T10:00:00Z", "url": "https://github.com/org/repo/issues/108", "user": {"login": "liam"}, "assignees": [{"login": "mia"}], "comments": 2, "repository": "org/repo"},
            {"id": "9", "number": 109, "title": "Atualizar dependências de segurança", "body": "Atualizar bibliotecas críticas para corrigir vulnerabilidades.", "description": "Atualizar dependencies", "estimated_hours": 3, "state": "closed", "labels": ["maintenance", "security"], "created_at": "2024-12-01T07:00:00Z", "updated_at": "2024-12-05T08:00:00Z", "url": "https://github.com/org/repo/issues/109", "user": {"login": "noah"}, "assignees": [], "comments": 1, "repository": "org/repo"},
            {"id": "10", "number": 110, "title": "Documentar fluxo de deploy", "body": "Criar documentação e playbook para deploys em produção.", "description": "Documentação de deploy", "estimated_hours": 2, "state": "closed", "labels": ["docs", "devops"], "created_at": "2024-11-01T08:30:00Z", "updated_at": "2024-11-03T09:00:00Z", "url": "https://github.com/org/repo/issues/110", "user": {"login": "olivia"}, "assignees": [], "comments": 0, "repository": "org/repo"},
        ]

    def upsert(self, docs: List[Dict[str, Any]]) -> None:
        # adiciona docs mock (não usado no demo)
        for d in docs:
            self.issues.append(d)

    def get_repository_technologies(self) -> Dict[str, float]:
        """Retorna um dict simulando o breakdown de linguagens do GitHub."""
        return dict(self.technologies)

    def list_namespaces(self) -> List[str]:
        return [
            "org",
            "org_comments",
            "org_changelog",
            "mule",
            "mule_comments",
            "mule_changelog",
        ]

    def semantic_search(
        self,
        text: str,
        namespaces: List[str] | None = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        # Simples heurística: retorna os top_k com base em presença de palavras-chave
        text_l = text.lower()
        scored = []
        for it in self.issues:
            score = 0
            searchable = " ".join([
                str(it.get("title", "")),
                str(it.get("description", "")),
                str(it.get("body", "")),
                " ".join(it.get("labels", [])) if it.get("labels") else "",
                it.get("user", {}).get("login", "") if it.get("user") else "",
                it.get("repository", ""),
            ]).lower()
            # tokens matching technologies should boost score
            tech_boost = 0
            for tok in text_l.split():
                if tok in searchable:
                    score += 1
                # boost when matching a technology name
                for tech in self.technologies.keys():
                    if tok == tech.lower():
                        tech_boost += 2
            score += tech_boost
            scored.append((score, it))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [it for s, it in scored if s > 0]
        if not results:
            # fallback: return top_k most recent
            results = self.issues[:top_k]
        else:
            results = results[:top_k]

        if not namespaces:
            return results

        allowed = {str(ns).strip().lower() for ns in namespaces}
        filtered = []
        for it in results:
            repo = str(it.get("repository") or "").lower()
            org = repo.split("/", 1)[0] if "/" in repo else repo
            ns = org
            if ns in allowed:
                enriched = dict(it)
                enriched["namespace"] = ns
                filtered.append(enriched)
        return filtered


class MockLLMClient:
    """Mock LLM que responde com JSON baseado no prompt recebido.

    Ele tenta detectar se o prompt é para heurística, analógico ou correção
    e constrói um JSON plausível.
    """
    def __init__(self):
        pass

    def send_prompt(self, prompt: str, **kwargs) -> str:
        p = prompt.lower()
        # Heurístico: procura por palavras-chave do heuristic agent
        if "analise a descricao" in p or "analista" in p or "fatores" in p:
            # simple heuristics: estimate = words_count/10 * 2
            words = len(re.findall(r"\w+", prompt))
            est = max(1.0, round(words / 10.0 * 2.0, 1))
            conf = 0.6 + min(0.3, (est / 20.0))
            resp = {"estimate_hours": est, "confidence": round(conf, 2), "justification": "Heurística simulada baseada no tamanho e fatores."}
            import json
            return json.dumps(resp)

                # Analogical: tenta extrair estimated_hours da lista de issues no prompt
        if "est:" in p or "estimated_hours" in p or "issues similares" in p:
            nums = [float(x) for x in re.findall(r"est[:=]?\s*(\d+(?:\.\d+)?)", prompt)]
            if not nums:
                nums = [float(x) for x in re.findall(r"estimated_hours\"?:?\s*(\d+(?:\.\d+)?)", prompt)]
            if nums:
                avg = sum(nums) / len(nums)
                conf = 0.6 if len(nums) < 3 else 0.8
                import json
                return json.dumps({"estimate_hours": round(avg, 1), "confidence": conf, "justification": "Estimativa baseada em issues historicas similares."})
            # fallback
            import json
            return json.dumps({"estimate_hours": 8.0, "confidence": 0.4, "justification": "Fallback analogico: sem similares extraidos."})
        # Correction: tenta extrair factors real/estimated
        if "fator" in p or "correc" in p or "estimativa base" in p:
            pairs = re.findall(r"est[:=]?\s*(\d+(?:\.\d+)?)h|estimated_hours[:=]?\s*(\d+(?:\.\d+)?)", prompt)
            # find both est and real mentions
            ests = [float(x) for x in re.findall(r"est[:=]?\s*(\d+(?:\.\d+)?)", prompt)]
            reals = [float(x) for x in re.findall(r"real[:=]?\s*(\d+(?:\.\d+)?)", prompt)]
            factor = 1.0
            if ests and reals:
                # align first pairs
                factors = [r / e if e > 0 else 1.0 for e, r in zip(ests, reals)]
                factor = sum(factors) / len(factors)
            base_match = re.search(r"estimativa base[:=]?\s*(\d+(?:\.\d+)?)", prompt)
            if base_match:
                base = float(base_match.group(1))
                adjusted = round(base * factor, 1)
            else:
                adjusted = round((sum(reals) / len(reals)) if reals else 8.0, 1)
            import json
            return json.dumps({"estimate_hours": adjusted, "confidence": 0.75, "justification": f"Ajuste por fator medio {factor:.2f}x"})

        # Default fallback
        # Combine/refine handling
        if "combinar" in p or "combine" in p or "sintet" in p or "synthesize" in p:
            # try to extract estimate_hours and confidences
            ests = [float(x) for x in re.findall(r"estimate_hours\"?:?\s*([0-9]+(?:\.[0-9]+)?)", prompt)]
            confs = [float(x) for x in re.findall(r"confidence\"?:?\s*([0-9]+(?:\.[0-9]+)?)", prompt)]
            if ests:
                # weight by confidences if present
                if confs and len(confs) == len(ests):
                    weights = [max(0.01, c) for c in confs]
                    total_w = sum(weights)
                    weighted = sum(e * w for e, w in zip(ests, weights)) / total_w
                else:
                    weighted = sum(ests) / len(ests)
                if confs:
                    import math
                    prod_val = math.prod([(1.0 - min(0.99, max(0.0, float(c)))) for c in confs])
                    combined_conf = 1.0 - prod_val
                else:
                    combined_conf = 0.6
                combined_conf = max(0.0, min(0.99, combined_conf))
                import json
                return json.dumps({"estimate_hours": round(weighted, 2), "confidence": round(combined_conf, 2), "justification": "Refinamento sintetizado pelo mock LLM."})

        import json
        return json.dumps({"estimate_hours": 8.0, "confidence": 0.5, "justification": "Resposta padrao do mock LLM."})

