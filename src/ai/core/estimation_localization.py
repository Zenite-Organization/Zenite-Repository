from typing import Any


_GENERIC_SPLIT_REASON_PT = (
    "Há indícios de que o trabalho está grande demais para um único item e pode precisar "
    "ser dividido em partes menores."
)


def normalize_split_reason(reason: Any, agile_hours_limit: float | None = None) -> str:
    text = str(reason or "").strip()
    if not text:
        return ""

    translations = {
        "Final range indicates a large backlog item and likely refinement or split.": (
            "A faixa final indica um item grande de backlog, com sinais de que pode "
            "precisar de refinamento ou divisão."
        ),
        "Issue groups multiple deliverables.": "A issue agrupa múltiplas entregas.",
        "Issue groups multiple deliverables and likely should be split.": (
            "A issue agrupa múltiplas entregas e provavelmente deve ser dividida."
        ),
    }
    if agile_hours_limit is not None:
        translations[
            f"Consolidated estimate exceeds the healthy agile limit of {agile_hours_limit:.0f}h."
        ] = (
            f"A estimativa consolidada ultrapassa o limite saudável de "
            f"{agile_hours_limit:.0f}h para um item ágil."
        )

    translated = translations.get(text)
    if translated:
        return translated

    lowered = f" {text.casefold()} "
    english_hints = (
        " issue ",
        " should ",
        " likely ",
        " split ",
        " refinement ",
        " deliverable ",
        " deliverables ",
        " smaller ",
        " manageable ",
        " stories ",
        " backlog item ",
        " final range ",
        " healthy agile limit ",
        " exceeds ",
    )
    if any(hint in lowered for hint in english_hints):
        return _GENERIC_SPLIT_REASON_PT
    return text
