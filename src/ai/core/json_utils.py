import json
import re
from typing import Any


def _extract_text_from_llm_response(response: Any) -> str:
    if response is None:
        return ""

    if isinstance(response, str):
        return response

    # Some LLM SDKs return a list of content blocks, e.g.:
    # [{"type": "text", "text": "{...json...}", "extras": {...}}, ...]
    if isinstance(response, list):
        parts: list[str] = []
        for item in response:
            if isinstance(item, dict) and "text" in item and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "\n".join(parts)
        return str(response)

    if isinstance(response, dict) and "text" in response and isinstance(response.get("text"), str):
        return response["text"]

    return str(response)


def parse_llm_json_response(response: Any) -> Any:
    content = _extract_text_from_llm_response(response).strip()
    if not content:
        raise ValueError("Empty LLM response")

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
