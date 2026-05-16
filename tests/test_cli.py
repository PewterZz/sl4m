from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from sl4m.cli import app


runner = CliRunner()


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def test_cli_help_exposes_slam_surface() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "mac-profile" in result.output
    assert "mac-linear-bench" in result.output
    assert "mlx-train" in result.output


def test_cli_mlx_check_data_reports_dataset(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "train.jsonl", [{"prompt": "Q", "completion": "A"}])

    result = runner.invoke(app, ["mlx-check-data", str(tmp_path)])

    assert result.exit_code == 0
    assert "train.jsonl" in result.output
    assert "completions" in result.output


def test_cli_mlx_check_data_requires_expected_split(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "train.jsonl", [{"text": "hello"}])

    result = runner.invoke(app, ["mlx-check-data", str(tmp_path), "--require-test"])

    assert result.exit_code != 0
    assert "test.jsonl" in result.output


def test_cli_mac_linear_bench_rejects_bad_shape() -> None:
    result = runner.invoke(app, ["mac-linear-bench", "--shapes", "1x2", "--repeats", "1"])

    assert result.exit_code != 0
    assert "expected MxKxN" in result.output


def test_cli_mac_compare_renders_results(monkeypatch) -> None:
    from sl4m.mac_efficiency import BenchmarkResult

    monkeypatch.setattr(
        "sl4m.cli.compare_mlx_lm_vs_slam",
        lambda **kwargs: [
            BenchmarkResult(
                "mlx-lm-direct",
                kwargs["model"],
                4,
                2,
                1.0,
                2.0,
                metadata={"token_identity": True, "repeat": 0},
            ),
            BenchmarkResult(
                "slam-baseline",
                kwargs["model"],
                4,
                2,
                0.5,
                4.0,
                metadata={"token_identity": True, "repeat": 0},
            ),
        ],
    )

    result = runner.invoke(app, ["mac-compare", "model-path", "--max-tokens", "2"])

    assert result.exit_code == 0
    assert "mlx-lm-direct" in result.output
    assert "slam-baseline" in result.output
    assert "slam/direct=2.00" in result.output


def test_cli_agent_run_dry_run_routes_edit_prompt_to_pld() -> None:
    prompt = (
        "\n".join(["def f(x):", "    return x + 1"] * 80)
        + "\n\nRefactor the code above. Preserve behavior and return only updated code."
    )

    result = runner.invoke(
        app,
        ["agent-run", "model-path", "--prompt", prompt, "--dry-run"],
    )

    assert result.exit_code == 0
    assert "mode=adaptive-pld" in result.output
    assert "auto=pld" in result.output


def test_cli_agent_run_rejects_bad_mode() -> None:
    result = runner.invoke(
        app,
        ["agent-run", "model-path", "--prompt", "hello", "--mode", "missing", "--dry-run"],
    )

    assert result.exit_code != 0
    assert "mode must be auto, baseline, pld, or adaptive-pld" in result.output


def test_cli_agent_bench_dry_run(tmp_path: Path) -> None:
    task = tmp_path / "task.toml"
    task.write_text(
        """
[task]
name = "sample"
context = "def f(x):\\n    return x + 1\\n"
instruction = "Refactor the code above. Preserve behavior and return only code."
validations = ["python-ast"]
""".strip(),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["agent-bench", str(tmp_path), "--dry-run"])

    assert result.exit_code == 0
    assert "sample" in result.output
    assert "python-ast" in result.output


def test_cli_agent_bench_requires_model_without_dry_run(tmp_path: Path) -> None:
    task = tmp_path / "task.toml"
    task.write_text(
        """
[task]
name = "sample"
instruction = "Write code."
""".strip(),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["agent-bench", str(tmp_path)])

    assert result.exit_code != 0
    assert "--model is required" in result.output


def test_cli_mlx_train_dry_run_from_recipe(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _write_jsonl(data / "train.jsonl", [{"text": "hello"}])
    recipe = tmp_path / "recipe.toml"
    recipe.write_text(
        f"""
[train]
model = "mlx-community/test"
data = "{data}"
adapter_path = "{tmp_path / 'adapters'}"
iters = 2
""".strip(),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["mlx-train", "--recipe", str(recipe), "--dry-run"])

    assert result.exit_code == 0
    assert "mlx_lm.lora" in result.output
    assert "--adapter-path" in result.output


def test_cli_mlx_train_plan(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _write_jsonl(data / "train.jsonl", [{"text": "hello"} for _ in range(5)])
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "tiny", "num_hidden_layers": 4, "hidden_size": 128}),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "mlx-train-plan",
            "--model",
            str(model),
            "--data",
            str(data),
            "--batch-size",
            "2",
            "--grad-accumulation-steps",
            "2",
            "--iters",
            "8",
        ],
    )

    assert result.exit_code == 0
    assert "MLX training plan" in result.output
    assert "effective_batch" in result.output
    assert "estimated_epochs" in result.output
