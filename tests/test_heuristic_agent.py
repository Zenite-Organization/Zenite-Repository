import unittest

from ai.agents.heuristic_agent import run_heuristic


class _CaptureLLM:
    def __init__(self):
        self.last_kwargs = {}

    def send_prompt(self, _prompt: str, **kwargs) -> str:
        self.last_kwargs = kwargs
        return '{"size":"M","estimate_hours":8,"confidence":0.6,"justification":"ok"}'


class TestHeuristicAgent(unittest.TestCase):
    def test_run_heuristic_forwards_temperature(self):
        llm = _CaptureLLM()
        out = run_heuristic(
            issue_context={"title": "Task", "description": "Desc"},
            llm=llm,
            temperature=0.42,
        )
        self.assertEqual(out["estimate_hours"], 8)
        self.assertEqual(llm.last_kwargs["temperature"], 0.42)


if __name__ == "__main__":
    unittest.main()
