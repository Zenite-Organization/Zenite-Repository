import unittest

from ai.core.json_utils import parse_llm_json_response


class TestJsonUtils(unittest.TestCase):
    def test_parse_raw_json(self):
        parsed = parse_llm_json_response('{"a": 1, "b": "ok"}')
        self.assertEqual(parsed["a"], 1)
        self.assertEqual(parsed["b"], "ok")

    def test_parse_fenced_json(self):
        parsed = parse_llm_json_response("```json\n{\"x\": 2}\n```")
        self.assertEqual(parsed["x"], 2)

    def test_parse_json_inside_text(self):
        parsed = parse_llm_json_response('resultado:\n{"status":"ok"}\nobrigado')
        self.assertEqual(parsed["status"], "ok")

    def test_raises_when_json_not_found(self):
        with self.assertRaises(ValueError):
            parse_llm_json_response("no json here")


if __name__ == "__main__":
    unittest.main()
