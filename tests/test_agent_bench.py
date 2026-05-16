from __future__ import annotations

import json
from pathlib import Path

import pytest

from sl4m.agent_bench import (
    AgentBenchResult,
    AgentBenchTask,
    AgentBenchValidation,
    ValidationResult,
    load_agent_task,
    load_agent_tasks,
    validate_modes,
    validate_output,
    write_agent_bench_jsonl,
)


def test_load_agent_task_from_toml(tmp_path: Path) -> None:
    fixture = tmp_path / "code.py"
    fixture.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    task_file = tmp_path / "task.toml"
    task_file.write_text(
        """
[task]
name = "sample"
context_file = "code.py"
instruction = "Add type hints and return only code."
validations = ["python-ast", { kind = "contains", value = "def add" }]
""".strip(),
        encoding="utf-8",
    )

    task = load_agent_task(task_file)

    assert task.name == "sample"
    assert "def add" in task.context
    assert task.prompt.endswith("Add type hints and return only code.")
    assert [v.kind for v in task.validations] == ["python-ast", "contains"]


def test_load_agent_tasks_requires_toml(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no .toml task files"):
        load_agent_tasks(tmp_path)


def test_validate_modes() -> None:
    assert validate_modes(["baseline", "pld", "adaptive-pld", "agent-auto"]) == [
        "baseline",
        "pld",
        "adaptive-pld",
        "agent-auto",
    ]
    with pytest.raises(ValueError, match="unsupported agent-bench mode"):
        validate_modes(["baseline", "bad"])


def test_validate_output_python_ast_and_contains() -> None:
    task = AgentBenchTask(
        name="t",
        instruction="return code",
        validations=(),
    )
    ok_task = AgentBenchTask(
        name="t",
        instruction="return code",
        validations=(
            AgentBenchValidation("python-ast"),
            AgentBenchValidation("contains", "def f"),
        ),
    )

    results = validate_output(ok_task, "```python\ndef f() -> int:\n    return 1\n```")

    assert all(r.ok for r in results)
    assert validate_output(task, "anything") == ()


def test_validate_output_reports_syntax_error() -> None:
    validation = AgentBenchValidation("python-ast")
    task = AgentBenchTask(name="t", instruction="return code", validations=(validation,))

    results = validate_output(task, "def broken(:\n")

    assert results[0].ok is False
    assert "invalid syntax" in results[0].detail


def test_write_agent_bench_jsonl(tmp_path: Path) -> None:
    out = tmp_path / "bench.jsonl"
    result = AgentBenchResult(
        task="t",
        mode="baseline",
        selected_mode="baseline",
        seconds=1.0,
        tokens=2,
        tok_s=2.0,
        first_token_s=0.2,
        peak_mem_gb=None,
        route=None,
        validations=(ValidationResult("python-ast", True),),
    )

    write_agent_bench_jsonl(out, [result])

    row = json.loads(out.read_text(encoding="utf-8"))
    assert row["task"] == "t"
    assert row["validation_ok"] is True
