from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


AgentMode = Literal["baseline", "pld"]


EDIT_KEYWORDS = (
    "refactor",
    "rewrite",
    "modify",
    "update",
    "fix",
    "preserve behavior",
    "return only",
    "updated code",
    "existing code",
    "code above",
    "patch",
    "diff",
    "type hints",
    "tests",
)

CODE_MARKERS = (
    "def ",
    "class ",
    "import ",
    "from ",
    "pytest",
    "assert ",
    "```",
    "{",
    "}",
)


@dataclass(frozen=True)
class AgentRoute:
    mode: AgentMode
    confidence: float
    reason: str
    prompt_chars: int
    line_count: int
    code_marker_count: int
    repeated_line_count: int


def _line_repetition_score(lines: list[str]) -> int:
    normalized = [
        re.sub(r"\s+", " ", line.strip())
        for line in lines
        if len(line.strip()) >= 8
    ]
    seen: set[str] = set()
    repeated = 0
    for line in normalized:
        if line in seen:
            repeated += 1
        seen.add(line)
    return repeated


def route_agent_prompt(
    prompt: str,
    *,
    min_edit_chars: int = 700,
    min_code_markers: int = 4,
) -> AgentRoute:
    """Conservatively choose baseline or PLD for agentic prompts.

    PLD helps when generated text is likely to mirror local prompt structure.
    That usually means code-edit/refactor requests with a substantial pasted
    context. Short/open-ended prompts stay on baseline because PLD overhead can
    dominate.
    """
    lowered = prompt.lower()
    lines = prompt.splitlines()
    keyword_hits = sum(1 for kw in EDIT_KEYWORDS if kw in lowered)
    code_marker_count = sum(prompt.count(marker) for marker in CODE_MARKERS)
    repeated_line_count = _line_repetition_score(lines)

    score = 0.0
    reasons: list[str] = []
    if len(prompt) >= min_edit_chars:
        score += 0.30
        reasons.append("long-context")
    if keyword_hits:
        score += min(0.30, 0.10 * keyword_hits)
        reasons.append(f"edit-keywords={keyword_hits}")
    if code_marker_count >= min_code_markers:
        score += 0.25
        reasons.append(f"code-markers={code_marker_count}")
    if repeated_line_count:
        score += min(0.15, 0.05 * repeated_line_count)
        reasons.append(f"repeated-lines={repeated_line_count}")

    confidence = min(score, 1.0)
    if confidence >= 0.55:
        return AgentRoute(
            mode="pld",
            confidence=confidence,
            reason=", ".join(reasons) or "edit-shaped",
            prompt_chars=len(prompt),
            line_count=len(lines),
            code_marker_count=code_marker_count,
            repeated_line_count=repeated_line_count,
        )
    return AgentRoute(
        mode="baseline",
        confidence=1.0 - confidence,
        reason=", ".join(reasons) if reasons else "short/open-generation",
        prompt_chars=len(prompt),
        line_count=len(lines),
        code_marker_count=code_marker_count,
        repeated_line_count=repeated_line_count,
    )
