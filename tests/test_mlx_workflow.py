from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from sl4m.mlx_workflow import (
    EvalRecipe,
    ServeRecipe,
    TrainRecipe,
    WorkflowError,
    detect_dataset_format,
    validate_dataset_dir,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in records),
        encoding="utf-8",
    )


def test_detect_dataset_formats() -> None:
    assert detect_dataset_format({"messages": [{"role": "user", "content": "hi"}]}) == "chat"
    assert detect_dataset_format({"messages": [{"role": "user", "content": "hi"}], "tools": []}) == "tools"
    assert detect_dataset_format({"prompt": "Q", "completion": "A"}) == "completions"
    assert detect_dataset_format({"text": "hello"}) == "text"
    with pytest.raises(WorkflowError):
        detect_dataset_format({"input": "Q", "output": "A"})


def test_validate_dataset_dir_reports_files(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "train.jsonl",
        [{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]}],
    )
    _write_jsonl(tmp_path / "valid.jsonl", [{"prompt": "Q", "completion": "A"}])

    report = validate_dataset_dir(tmp_path)

    assert report.files == ("train.jsonl", "valid.jsonl")
    assert report.formats["train.jsonl"] == "chat"
    assert report.formats["valid.jsonl"] == "completions"
    assert report.examples["train.jsonl"] == 1


def test_validate_dataset_dir_requires_train(tmp_path: Path) -> None:
    with pytest.raises(WorkflowError, match="missing required files"):
        validate_dataset_dir(tmp_path)


def test_train_recipe_command_uses_mac_memory_defaults(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "train.jsonl", [{"prompt": "Q", "completion": "A"}])
    recipe = TrainRecipe(
        model="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        data=tmp_path,
        adapter_path=tmp_path / "adapters",
        iters=25,
        batch_size=1,
        grad_accumulation_steps=4,
    )

    cmd = recipe.to_command()

    assert cmd[:3] == [sys.executable, "-m", "mlx_lm.lora"]
    assert "--train" in cmd
    assert cmd[cmd.index("--model") + 1] == recipe.model
    assert cmd[cmd.index("--data") + 1] == str(tmp_path)
    assert cmd[cmd.index("--adapter-path") + 1] == str(tmp_path / "adapters")
    assert cmd[cmd.index("--batch-size") + 1] == "1"
    assert cmd[cmd.index("--grad-accumulation-steps") + 1] == "4"
    assert "--grad-checkpoint" in cmd
    assert "--mask-prompt" in cmd


def test_train_recipe_from_toml(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "train.jsonl", [{"text": "hello"}])
    recipe_path = tmp_path / "recipe.toml"
    recipe_path.write_text(
        f"""
[train]
model = "mlx-community/test"
data = "{tmp_path}"
adapter_path = "{tmp_path / 'out'}"
iters = 12
fine_tune_type = "dora"
mask_prompt = false
""".strip(),
        encoding="utf-8",
    )

    recipe = TrainRecipe.from_toml(recipe_path)

    assert recipe.model == "mlx-community/test"
    assert recipe.data == tmp_path
    assert recipe.adapter_path == tmp_path / "out"
    assert recipe.iters == 12
    assert recipe.fine_tune_type == "dora"
    assert recipe.mask_prompt is False


def test_eval_recipe_requires_test_jsonl(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "train.jsonl", [{"text": "hello"}])
    recipe = EvalRecipe(model="mlx-community/test", data=tmp_path)

    with pytest.raises(WorkflowError, match="test.jsonl"):
        recipe.to_command()


def test_serve_recipe_command_includes_adapters_and_draft(tmp_path: Path) -> None:
    cmd = ServeRecipe(
        model="mlx-community/test",
        adapter_path=tmp_path / "adapters",
        draft_model="mlx-community/draft",
        port=8090,
    ).to_command()

    assert cmd[:3] == [sys.executable, "-m", "mlx_lm.server"]
    assert cmd[cmd.index("--port") + 1] == "8090"
    assert cmd[cmd.index("--adapter-path") + 1] == str(tmp_path / "adapters")
    assert cmd[cmd.index("--draft-model") + 1] == "mlx-community/draft"
