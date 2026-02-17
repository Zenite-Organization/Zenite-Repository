import json
import re
from typing import Any


def parse_llm_json_response(response: str) -> Any:
    content = response.strip()

    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", content, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        content = fenced.group(1).strip()

    try:
        return json.loads(content)
    except Exception:
        pass

    start_obj = content.find("{")
    end_obj = content.rfind("}")
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        candidate = content[start_obj : end_obj + 1]
        return json.loads(candidate)

    start_arr = content.find("[")
    end_arr = content.rfind("]")
    if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
        candidate = content[start_arr : end_arr + 1]
        return json.loads(candidate)

    raise ValueError("No valid JSON content found in LLM response")
