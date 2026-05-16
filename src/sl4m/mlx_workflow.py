from __future__ import annotations

import json
import math
import platform
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - py3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]


SUPPORTED_FINE_TUNE_TYPES = {"lora", "dora", "full"}
SUPPORTED_DATASET_FORMATS = {"chat", "tools", "completions", "text"}


class WorkflowError(ValueError):
    """Raised when a Mac/MLX workflow recipe is invalid."""


@dataclass(frozen=True)
class DatasetReport:
    path: Path
    files: tuple[str, ...]
    formats: dict[str, str]
    examples: dict[str, int]
    chars: dict[str, int]

    @property
    def train_examples(self) -> int:
        return self.examples.get("train.jsonl", 0)

    @property
    def train_chars(self) -> int:
        return self.chars.get("train.jsonl", 0)

    @property
    def rough_train_tokens(self) -> int:
        return max(1, self.train_chars // 4) if self.train_chars else 0


@dataclass(frozen=True)
class ModelConfigSummary:
    path: Optional[Path]
    model_type: Optional[str]
    hidden_size: Optional[int]
    intermediate_size: Optional[int]
    num_hidden_layers: Optional[int]
    num_attention_heads: Optional[int]
    quantization_bits: Optional[int]


@dataclass(frozen=True)
class TrainPlan:
    recipe: "TrainRecipe"
    dataset: DatasetReport
    model_config: ModelConfigSummary
    effective_batch_size: int
    optimizer_steps_per_epoch: int
    estimated_epochs: float
    trainable_layer_count: Optional[int]

    @property
    def rough_tokens_per_step(self) -> int:
        if self.optimizer_steps_per_epoch <= 0:
            return 0
        return max(1, self.dataset.rough_train_tokens // self.optimizer_steps_per_epoch)


@dataclass(frozen=True)
class AdapterBenchResult:
    name: str
    fine_tune_type: str
    num_layers: int
    rank: Optional[int]
    scale: Optional[float]
    dropout: Optional[float]
    adapter_path: Path
    train_seconds: float
    train_returncode: int
    eval_seconds: Optional[float]
    eval_returncode: Optional[int]
    train_loss: Optional[float]
    val_loss: Optional[float]
    test_loss: Optional[float]
    test_ppl: Optional[float]
    it_per_sec: Optional[float]
    tokens_per_sec: Optional[float]
    trained_tokens: Optional[int]
    peak_mem_gb: Optional[float]
    trainable_params_m: Optional[float]
    trainable_pct: Optional[float]
    adapter_size_mb: Optional[float]
    stdout_tail: str
    stderr_tail: str

    def to_record(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["adapter_path"] = str(self.adapter_path)
        return payload


def _coerce_path(value: str | Path) -> Path:
    return Path(value).expanduser()


def _read_toml(path: str | Path) -> dict[str, Any]:
    with _coerce_path(path).open("rb") as f:
        return tomllib.load(f)


def _jsonl_summary(path: Path) -> tuple[dict[str, Any], int, int]:
    count = 0
    chars = 0
    first: Optional[dict[str, Any]] = None
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            count += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise WorkflowError(f"{path}:{lineno}: invalid JSONL record: {e}") from e
            if not isinstance(obj, dict):
                raise WorkflowError(f"{path}:{lineno}: each JSONL record must be an object")
            if first is None:
                first = obj
            chars += len(line)
    if first is None:
        raise WorkflowError(f"{path}: file is empty")
    return first, count, chars


def detect_dataset_format(record: dict[str, Any]) -> str:
    if "messages" in record and "tools" in record:
        return "tools"
    if "messages" in record:
        if not isinstance(record["messages"], list) or not record["messages"]:
            raise WorkflowError("chat dataset records need a non-empty messages list")
        return "chat"
    if "prompt" in record and "completion" in record:
        return "completions"
    if "text" in record:
        return "text"
    raise WorkflowError(
        "unsupported dataset record; expected one of chat/tools/completions/text formats"
    )


def validate_dataset_dir(data_dir: str | Path, require: Iterable[str] = ("train.jsonl",)) -> DatasetReport:
    root = _coerce_path(data_dir)
    if not root.exists():
        raise WorkflowError(f"dataset directory does not exist: {root}")
    if not root.is_dir():
        raise WorkflowError(f"dataset path must be a directory: {root}")

    missing = [name for name in require if not (root / name).exists()]
    if missing:
        raise WorkflowError(f"dataset directory {root} missing required files: {', '.join(missing)}")

    files = tuple(name for name in ("train.jsonl", "valid.jsonl", "test.jsonl") if (root / name).exists())
    formats: dict[str, str] = {}
    examples: dict[str, int] = {}
    chars: dict[str, int] = {}
    for name in files:
        first, count, char_count = _jsonl_summary(root / name)
        formats[name] = detect_dataset_format(first)
        examples[name] = count
        chars[name] = char_count
    return DatasetReport(path=root, files=files, formats=formats, examples=examples, chars=chars)


def summarize_model_config(model: str | Path) -> ModelConfigSummary:
    root = _coerce_path(model)
    cfg_path = root / "config.json"
    if not cfg_path.exists():
        return ModelConfigSummary(
            path=None,
            model_type=None,
            hidden_size=None,
            intermediate_size=None,
            num_hidden_layers=None,
            num_attention_heads=None,
            quantization_bits=None,
        )
    try:
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise WorkflowError(f"{cfg_path}: invalid model config JSON: {e}") from e
    text = raw.get("text_config") if isinstance(raw.get("text_config"), dict) else {}
    quant = raw.get("quantization") or raw.get("quantization_config") or {}
    return ModelConfigSummary(
        path=cfg_path,
        model_type=raw.get("model_type") or text.get("model_type") or raw.get("architectures", [None])[0],
        hidden_size=_maybe_int(raw.get("hidden_size") or text.get("hidden_size")),
        intermediate_size=_maybe_int(raw.get("intermediate_size") or text.get("intermediate_size")),
        num_hidden_layers=_maybe_int(raw.get("num_hidden_layers") or text.get("num_hidden_layers")),
        num_attention_heads=_maybe_int(raw.get("num_attention_heads") or text.get("num_attention_heads")),
        quantization_bits=_maybe_int(quant.get("bits") or quant.get("bits_per_weight")),
    )


def _maybe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def ensure_macos() -> None:
    if platform.system() != "Darwin":
        raise WorkflowError("MLX training/serving commands are Mac-first and require macOS")


def _append_opt(args: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        args.extend([flag, str(value)])


@dataclass(frozen=True)
class TrainRecipe:
    model: str
    data: Path
    adapter_path: Path = Path("adapters")
    fine_tune_type: str = "lora"
    iters: int = 600
    batch_size: int = 1
    learning_rate: float = 1e-5
    num_layers: int = 16
    grad_accumulation_steps: int = 1
    optimizer: Optional[str] = None
    val_batches: Optional[int] = None
    max_seq_length: Optional[int] = None
    clear_cache_threshold: Optional[int] = None
    lora_rank: Optional[int] = None
    lora_scale: Optional[float] = None
    lora_dropout: Optional[float] = None
    config_path: Optional[Path] = None
    grad_checkpoint: bool = True
    mask_prompt: bool = True
    seed: Optional[int] = None
    report_to: Optional[str] = None
    project_name: Optional[str] = None
    resume_adapter_file: Optional[Path] = None

    @classmethod
    def from_toml(cls, path: str | Path) -> "TrainRecipe":
        doc = _read_toml(path)
        raw = doc.get("train", doc)
        if "model" not in raw or "data" not in raw:
            raise WorkflowError("train recipe needs `model` and `data`")
        return cls(
            model=str(raw["model"]),
            data=_coerce_path(raw["data"]),
            adapter_path=_coerce_path(raw.get("adapter_path", "adapters")),
            fine_tune_type=str(raw.get("fine_tune_type", "lora")),
            iters=int(raw.get("iters", 600)),
            batch_size=int(raw.get("batch_size", 1)),
            learning_rate=float(raw.get("learning_rate", 1e-5)),
            num_layers=int(raw.get("num_layers", 16)),
            grad_accumulation_steps=int(raw.get("grad_accumulation_steps", 1)),
            optimizer=str(raw["optimizer"]) if "optimizer" in raw else None,
            val_batches=int(raw["val_batches"]) if "val_batches" in raw else None,
            max_seq_length=int(raw["max_seq_length"]) if "max_seq_length" in raw else None,
            clear_cache_threshold=int(raw["clear_cache_threshold"])
            if "clear_cache_threshold" in raw
            else None,
            lora_rank=int(raw["lora_rank"]) if "lora_rank" in raw else None,
            lora_scale=float(raw["lora_scale"]) if "lora_scale" in raw else None,
            lora_dropout=float(raw["lora_dropout"]) if "lora_dropout" in raw else None,
            config_path=_coerce_path(raw["config_path"]) if "config_path" in raw else None,
            grad_checkpoint=bool(raw.get("grad_checkpoint", True)),
            mask_prompt=bool(raw.get("mask_prompt", True)),
            seed=int(raw["seed"]) if "seed" in raw else None,
            report_to=str(raw["report_to"]) if "report_to" in raw else None,
            project_name=str(raw["project_name"]) if "project_name" in raw else None,
            resume_adapter_file=_coerce_path(raw["resume_adapter_file"])
            if "resume_adapter_file" in raw
            else None,
        )

    def validate(self) -> DatasetReport:
        if self.fine_tune_type not in SUPPORTED_FINE_TUNE_TYPES:
            raise WorkflowError(
                f"fine_tune_type must be one of {sorted(SUPPORTED_FINE_TUNE_TYPES)}"
            )
        if self.iters <= 0:
            raise WorkflowError("iters must be > 0")
        if self.batch_size <= 0:
            raise WorkflowError("batch_size must be > 0")
        if self.learning_rate <= 0:
            raise WorkflowError("learning_rate must be > 0")
        if self.num_layers == 0 or self.num_layers < -1:
            raise WorkflowError("num_layers must be -1 or a positive integer")
        if self.grad_accumulation_steps <= 0:
            raise WorkflowError("grad_accumulation_steps must be > 0")
        return validate_dataset_dir(self.data, require=("train.jsonl",))

    def to_command(self) -> list[str]:
        self.validate()
        if self.config_path is not None:
            return [sys.executable, "-m", "mlx_lm", "lora", "--config", str(self.config_path)]
        args = [
            sys.executable,
            "-m",
            "mlx_lm",
            "lora",
            "--model",
            self.model,
            "--train",
            "--data",
            str(self.data),
            "--adapter-path",
            str(self.adapter_path),
            "--fine-tune-type",
            self.fine_tune_type,
            "--iters",
            str(self.iters),
            "--batch-size",
            str(self.batch_size),
            "--learning-rate",
            str(self.learning_rate),
            "--num-layers",
            str(self.num_layers),
            "--grad-accumulation-steps",
            str(self.grad_accumulation_steps),
        ]
        if self.grad_checkpoint:
            args.append("--grad-checkpoint")
        if self.mask_prompt:
            args.append("--mask-prompt")
        _append_opt(args, "--optimizer", self.optimizer)
        _append_opt(args, "--val-batches", self.val_batches)
        _append_opt(args, "--max-seq-length", self.max_seq_length)
        _append_opt(args, "--clear-cache-threshold", self.clear_cache_threshold)
        _append_opt(args, "--seed", self.seed)
        _append_opt(args, "--report-to", self.report_to)
        _append_opt(args, "--project-name", self.project_name)
        _append_opt(args, "--resume-adapter-file", self.resume_adapter_file)
        return args

    def to_config(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "model": self.model,
            "train": True,
            "data": str(self.data),
            "adapter_path": str(self.adapter_path),
            "fine_tune_type": self.fine_tune_type,
            "iters": self.iters,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_layers": self.num_layers,
            "grad_accumulation_steps": self.grad_accumulation_steps,
            "grad_checkpoint": self.grad_checkpoint,
            "mask_prompt": self.mask_prompt,
        }
        optional = {
            "optimizer": self.optimizer,
            "val_batches": self.val_batches,
            "max_seq_length": self.max_seq_length,
            "clear_cache_threshold": self.clear_cache_threshold,
            "seed": self.seed,
            "report_to": self.report_to,
            "project_name": self.project_name,
            "resume_adapter_file": str(self.resume_adapter_file)
            if self.resume_adapter_file is not None
            else None,
        }
        cfg.update({k: v for k, v in optional.items() if v is not None})
        if any(v is not None for v in (self.lora_rank, self.lora_scale, self.lora_dropout)):
            cfg["lora_parameters"] = {
                "rank": self.lora_rank if self.lora_rank is not None else 8,
                "scale": self.lora_scale if self.lora_scale is not None else 20.0,
                "dropout": self.lora_dropout if self.lora_dropout is not None else 0.0,
            }
        return cfg

    def write_config(self, path: Path) -> "TrainRecipe":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_config(), indent=2), encoding="utf-8")
        return TrainRecipe(
            model=self.model,
            data=self.data,
            adapter_path=self.adapter_path,
            fine_tune_type=self.fine_tune_type,
            iters=self.iters,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            num_layers=self.num_layers,
            grad_accumulation_steps=self.grad_accumulation_steps,
            optimizer=self.optimizer,
            val_batches=self.val_batches,
            max_seq_length=self.max_seq_length,
            clear_cache_threshold=self.clear_cache_threshold,
            lora_rank=self.lora_rank,
            lora_scale=self.lora_scale,
            lora_dropout=self.lora_dropout,
            grad_checkpoint=self.grad_checkpoint,
            mask_prompt=self.mask_prompt,
            seed=self.seed,
            report_to=self.report_to,
            project_name=self.project_name,
            resume_adapter_file=self.resume_adapter_file,
            config_path=path,
        )

    def plan(self) -> TrainPlan:
        report = self.validate()
        effective_batch = self.batch_size * self.grad_accumulation_steps
        steps_per_epoch = max(1, math.ceil(report.train_examples / effective_batch))
        model_config = summarize_model_config(self.model)
        trainable_layers = (
            model_config.num_hidden_layers
            if self.num_layers == -1
            else min(self.num_layers, model_config.num_hidden_layers)
            if model_config.num_hidden_layers is not None
            else self.num_layers
        )
        return TrainPlan(
            recipe=self,
            dataset=report,
            model_config=model_config,
            effective_batch_size=effective_batch,
            optimizer_steps_per_epoch=steps_per_epoch,
            estimated_epochs=self.iters / steps_per_epoch,
            trainable_layer_count=trainable_layers,
        )


@dataclass(frozen=True)
class EvalRecipe:
    model: str
    data: Path
    adapter_path: Path = Path("adapters")
    batch_size: int = 4
    test_batches: int = -1

    def validate(self) -> DatasetReport:
        if self.batch_size <= 0:
            raise WorkflowError("batch_size must be > 0")
        return validate_dataset_dir(self.data, require=("test.jsonl",))

    def to_command(self) -> list[str]:
        self.validate()
        return [
            sys.executable,
            "-m",
            "mlx_lm",
            "lora",
            "--model",
            self.model,
            "--adapter-path",
            str(self.adapter_path),
            "--data",
            str(self.data),
            "--batch-size",
            str(self.batch_size),
            "--test-batches",
            str(self.test_batches),
            "--test",
        ]


@dataclass(frozen=True)
class FuseRecipe:
    model: str
    adapter_path: Path = Path("adapters")
    save_path: Path = Path("fused_model")
    upload_repo: Optional[str] = None

    def to_command(self) -> list[str]:
        args = [
            sys.executable,
            "-m",
            "mlx_lm.fuse",
            "--model",
            self.model,
            "--adapter-path",
            str(self.adapter_path),
            "--save-path",
            str(self.save_path),
        ]
        _append_opt(args, "--upload-repo", self.upload_repo)
        return args


@dataclass(frozen=True)
class ServeRecipe:
    model: str
    adapter_path: Optional[Path] = None
    draft_model: Optional[str] = None
    host: str = "127.0.0.1"
    port: int = 8080

    def to_command(self) -> list[str]:
        args = [
            sys.executable,
            "-m",
            "mlx_lm.server",
            "--model",
            self.model,
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        _append_opt(args, "--adapter-path", self.adapter_path)
        _append_opt(args, "--draft-model", self.draft_model)
        return args


def parse_mlx_lora_output(text: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    trainable = re.search(
        r"Trainable parameters:\s+([0-9.]+)%\s+\(([0-9.]+)M/([0-9.]+)M\)",
        text,
    )
    if trainable:
        metrics["trainable_pct"] = float(trainable.group(1))
        metrics["trainable_params_m"] = float(trainable.group(2))
        metrics["total_params_m"] = float(trainable.group(3))

    val_losses = [float(x) for x in re.findall(r"Val loss\s+([0-9.]+)", text)]
    if val_losses:
        metrics["val_loss"] = val_losses[-1]

    final = re.search(
        r"Train loss\s+([0-9.]+).*?It/sec\s+([0-9.]+),\s+Tokens/sec\s+([0-9.]+),\s+"
        r"Trained Tokens\s+([0-9]+),\s+Peak mem\s+([0-9.]+)\s+GB",
        text,
        re.S,
    )
    if final:
        metrics["train_loss"] = float(final.group(1))
        metrics["it_per_sec"] = float(final.group(2))
        metrics["tokens_per_sec"] = float(final.group(3))
        metrics["trained_tokens"] = int(final.group(4))
        metrics["peak_mem_gb"] = float(final.group(5))

    test = re.search(r"Test loss\s+([0-9]+(?:\.[0-9]+)?),\s+Test ppl\s+([0-9]+(?:\.[0-9]+)?)", text)
    if test:
        metrics["test_loss"] = float(test.group(1))
        metrics["test_ppl"] = float(test.group(2))
    return metrics


def _adapter_size_mb(adapter_path: Path) -> Optional[float]:
    target = adapter_path / "adapters.safetensors"
    if not target.exists():
        return None
    return target.stat().st_size / 1024**2


def run_adapter_bench(
    *,
    model: str,
    data: Path,
    output_dir: Path,
    fine_tune_types: Iterable[str],
    num_layers: Iterable[int],
    ranks: Iterable[int] = (8,),
    scales: Iterable[float] = (20.0,),
    dropouts: Iterable[float] = (0.0,),
    iters: int,
    batch_size: int = 1,
    grad_accumulation_steps: int = 1,
    learning_rate: float = 1e-5,
    optimizer: Optional[str] = None,
    val_batches: int = 1,
    test_batches: int = 1,
    max_seq_length: Optional[int] = None,
    seed: Optional[int] = 0,
) -> list[AdapterBenchResult]:
    results: list[AdapterBenchResult] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for fine_tune_type in fine_tune_types:
        for layers in num_layers:
            for rank in ranks:
                for scale in scales:
                    for dropout in dropouts:
                        name = (
                            f"{fine_tune_type}-layers{layers}-"
                            f"r{rank}-a{scale:g}-d{dropout:g}"
                        )
                        adapter_path = output_dir / name
                        recipe = TrainRecipe(
                            model=model,
                            data=data,
                            adapter_path=adapter_path,
                            fine_tune_type=fine_tune_type,
                            iters=iters,
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                            num_layers=layers,
                            grad_accumulation_steps=grad_accumulation_steps,
                            optimizer=optimizer,
                            val_batches=val_batches,
                            max_seq_length=max_seq_length,
                            seed=seed,
                            lora_rank=rank,
                            lora_scale=scale,
                            lora_dropout=dropout,
                        ).write_config(adapter_path / "slam_lora_config.json")
                        t0 = time.monotonic()
                        train = subprocess.run(
                            recipe.to_command(), capture_output=True, text=True, check=False
                        )
                        train_seconds = time.monotonic() - t0
                        train_text = f"{train.stdout}\n{train.stderr}"
                        metrics = parse_mlx_lora_output(train_text)

                        eval_seconds: Optional[float] = None
                        eval_returncode: Optional[int] = None
                        eval_stdout = ""
                        eval_stderr = ""
                        if train.returncode == 0 and (data / "test.jsonl").exists():
                            eval_recipe = EvalRecipe(
                                model=model,
                                data=data,
                                adapter_path=adapter_path,
                                batch_size=batch_size,
                                test_batches=test_batches,
                            )
                            t_eval = time.monotonic()
                            eval_proc = subprocess.run(
                                eval_recipe.to_command(),
                                capture_output=True,
                                text=True,
                                check=False,
                            )
                            eval_seconds = time.monotonic() - t_eval
                            eval_returncode = eval_proc.returncode
                            eval_stdout = eval_proc.stdout
                            eval_stderr = eval_proc.stderr
                            metrics.update(parse_mlx_lora_output(f"{eval_stdout}\n{eval_stderr}"))

                        results.append(
                            AdapterBenchResult(
                                name=name,
                                fine_tune_type=fine_tune_type,
                                num_layers=layers,
                                rank=rank,
                                scale=scale,
                                dropout=dropout,
                                adapter_path=adapter_path,
                                train_seconds=train_seconds,
                                train_returncode=train.returncode,
                                eval_seconds=eval_seconds,
                                eval_returncode=eval_returncode,
                                train_loss=metrics.get("train_loss"),
                                val_loss=metrics.get("val_loss"),
                                test_loss=metrics.get("test_loss"),
                                test_ppl=metrics.get("test_ppl"),
                                it_per_sec=metrics.get("it_per_sec"),
                                tokens_per_sec=metrics.get("tokens_per_sec"),
                                trained_tokens=metrics.get("trained_tokens"),
                                peak_mem_gb=metrics.get("peak_mem_gb"),
                                trainable_params_m=metrics.get("trainable_params_m"),
                                trainable_pct=metrics.get("trainable_pct"),
                                adapter_size_mb=_adapter_size_mb(adapter_path),
                                stdout_tail=(train.stdout + eval_stdout)[-4000:],
                                stderr_tail=(train.stderr + eval_stderr)[-4000:],
                            )
                        )
    return results


def run_command(args: list[str], *, dry_run: bool = False) -> int:
    if dry_run:
        print(" ".join(args))
        return 0
    try:
        return subprocess.call(args)
    except ModuleNotFoundError as e:
        raise RuntimeError("MLX-LM is not installed. Install with `pip install -e '.[mac]'`.") from e
