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
    parse_mlx_lora_output,
    summarize_model_config,
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
    assert report.train_examples == 1
    assert report.rough_train_tokens > 0


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

    assert cmd[:4] == [sys.executable, "-m", "mlx_lm", "lora"]
    assert "--train" in cmd
    assert cmd[cmd.index("--model") + 1] == recipe.model
    assert cmd[cmd.index("--data") + 1] == str(tmp_path)
    assert cmd[cmd.index("--adapter-path") + 1] == str(tmp_path / "adapters")
    assert cmd[cmd.index("--batch-size") + 1] == "1"
    assert cmd[cmd.index("--grad-accumulation-steps") + 1] == "4"
    assert "--grad-checkpoint" in cmd
    assert "--mask-prompt" in cmd


def test_train_recipe_command_includes_hard_test_knobs(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "train.jsonl", [{"text": "hello"}])

    cmd = TrainRecipe(
        model="mlx-community/test",
        data=tmp_path,
        optimizer="adamw",
        val_batches=1,
        max_seq_length=256,
        clear_cache_threshold=1_000_000,
    ).to_command()

    assert cmd[cmd.index("--optimizer") + 1] == "adamw"
    assert cmd[cmd.index("--val-batches") + 1] == "1"
    assert cmd[cmd.index("--max-seq-length") + 1] == "256"
    assert cmd[cmd.index("--clear-cache-threshold") + 1] == "1000000"


def test_train_recipe_writes_lora_parameter_config(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "train.jsonl", [{"text": "hello"}])
    cfg_path = tmp_path / "adapter" / "slam_lora_config.json"

    recipe = TrainRecipe(
        model="mlx-community/test",
        data=tmp_path,
        adapter_path=tmp_path / "adapter",
        lora_rank=16,
        lora_scale=32.0,
        lora_dropout=0.05,
    ).write_config(cfg_path)

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cmd = recipe.to_command()

    assert cfg["lora_parameters"] == {"rank": 16, "scale": 32.0, "dropout": 0.05}
    assert cmd == [sys.executable, "-m", "mlx_lm", "lora", "--config", str(cfg_path)]


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


def test_train_recipe_plan_reads_local_model_config(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _write_jsonl(data / "train.jsonl", [{"text": "hello world"} for _ in range(9)])
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "hidden_size": 4096,
                "intermediate_size": 12288,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "quantization": {"bits": 4},
            }
        ),
        encoding="utf-8",
    )
    recipe = TrainRecipe(
        model=str(model),
        data=data,
        iters=20,
        batch_size=2,
        grad_accumulation_steps=2,
        num_layers=12,
    )

    plan = recipe.plan()

    assert plan.model_config.model_type == "qwen3"
    assert plan.model_config.quantization_bits == 4
    assert plan.effective_batch_size == 4
    assert plan.optimizer_steps_per_epoch == 3
    assert plan.estimated_epochs == pytest.approx(20 / 3)
    assert plan.trainable_layer_count == 12


def test_train_recipe_plan_reads_nested_text_config(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _write_jsonl(data / "train.jsonl", [{"text": "hello"}])
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "quantization": {"bits": 4},
                "text_config": {
                    "hidden_size": 4096,
                    "intermediate_size": 12288,
                    "num_hidden_layers": 32,
                    "num_attention_heads": 16,
                },
            }
        ),
        encoding="utf-8",
    )

    plan = TrainRecipe(model=str(model), data=data, num_layers=-1).plan()

    assert plan.model_config.hidden_size == 4096
    assert plan.model_config.intermediate_size == 12288
    assert plan.model_config.num_hidden_layers == 32
    assert plan.trainable_layer_count == 32


def test_summarize_model_config_missing_config_is_unknown(tmp_path: Path) -> None:
    summary = summarize_model_config(tmp_path)

    assert summary.path is None
    assert summary.hidden_size is None


def test_eval_recipe_requires_test_jsonl(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "train.jsonl", [{"text": "hello"}])
    recipe = EvalRecipe(model="mlx-community/test", data=tmp_path)

    with pytest.raises(WorkflowError, match="test.jsonl"):
        recipe.to_command()


def test_eval_recipe_command_supports_small_test_batch(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "test.jsonl", [{"text": "hello"}])

    cmd = EvalRecipe(
        model="mlx-community/test",
        data=tmp_path,
        batch_size=1,
        test_batches=1,
    ).to_command()

    assert cmd[:4] == [sys.executable, "-m", "mlx_lm", "lora"]
    assert cmd[cmd.index("--batch-size") + 1] == "1"
    assert cmd[cmd.index("--test-batches") + 1] == "1"
    assert "--test" in cmd


def test_parse_mlx_lora_output() -> None:
    metrics = parse_mlx_lora_output(
        """
Trainable parameters: 0.007% (0.639M/8953.802M)
Iter 2: Val loss 1.198, Val took 1.062s
Iter 2: Train loss 1.052, Learning Rate 1.000e-05, It/sec 2.052, Tokens/sec 13.955, Trained Tokens 68, Peak mem 6.148 GB
Test loss 2.045, Test ppl 7.731.
""".strip()
    )

    assert metrics["trainable_pct"] == 0.007
    assert metrics["trainable_params_m"] == 0.639
    assert metrics["val_loss"] == 1.198
    assert metrics["train_loss"] == 1.052
    assert metrics["tokens_per_sec"] == 13.955
    assert metrics["trained_tokens"] == 68
    assert metrics["peak_mem_gb"] == 6.148
    assert metrics["test_loss"] == 2.045
    assert metrics["test_ppl"] == 7.731


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
