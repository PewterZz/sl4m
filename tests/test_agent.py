from __future__ import annotations

from sl4m.agent import route_agent_prompt


def test_route_agent_prompt_keeps_short_generation_on_baseline() -> None:
    route = route_agent_prompt("Write a quicksort implementation.")

    assert route.mode == "baseline"
    assert route.reason == "short/open-generation"
    assert route.prompt_chars > 0


def test_route_agent_prompt_sends_edit_code_to_pld() -> None:
    code = "\n".join(
        [
            "def add(a, b):",
            "    return a + b",
            "",
            "def sub(a, b):",
            "    return a - b",
            "",
        ]
        * 30
    )
    prompt = (
        f"{code}\n\nRefactor the code above for clarity. Preserve behavior, "
        "add type hints, keep tests, and return only the updated code."
    )

    route = route_agent_prompt(prompt)

    assert route.mode == "pld"
    assert route.confidence >= 0.55
    assert route.code_marker_count >= 4
    assert route.repeated_line_count > 0


def test_route_agent_prompt_long_non_code_stays_baseline() -> None:
    prompt = "Explain the history of numerical computing. " * 80

    route = route_agent_prompt(prompt)

    assert route.mode == "baseline"
    assert route.prompt_chars > 700
