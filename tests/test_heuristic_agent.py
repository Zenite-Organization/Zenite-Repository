import unittest

from ai.agents.heuristic_agent import run_heuristic


class _CaptureLLM:
    def __init__(self):
        self.last_kwargs = {}
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def send_prompt(self, _prompt: str, **kwargs) -> str:
        self.last_kwargs = kwargs
        return '{"size_bucket":"S","bucket_rank":2,"confidence":0.6,"justification":"ok"}'

    def get_last_token_usage(self):
        return self.last_usage


class TestHeuristicAgent(unittest.TestCase):
    def test_run_heuristic_forwards_temperature(self):
        llm = _CaptureLLM()
        out = run_heuristic(
            issue_context={"title": "Task", "description": "Desc"},
            llm=llm,
            temperature=0.42,
        )
        self.assertEqual(out["size_bucket"], "S")
        self.assertEqual(out["bucket_rank"], 2)
        self.assertEqual(llm.last_kwargs["temperature"], 0.42)


if __name__ == "__main__":
    unittest.main()
