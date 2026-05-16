from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass
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


def _coerce_path(value: str | Path) -> Path:
    return Path(value).expanduser()


def _read_toml(path: str | Path) -> dict[str, Any]:
    with _coerce_path(path).open("rb") as f:
        return tomllib.load(f)


def _first_jsonl_record(path: Path) -> tuple[dict[str, Any], int]:
    count = 0
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
    if first is None:
        raise WorkflowError(f"{path}: file is empty")
    return first, count


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
    for name in files:
        first, count = _first_jsonl_record(root / name)
        formats[name] = detect_dataset_format(first)
        examples[name] = count
    return DatasetReport(path=root, files=files, formats=formats, examples=examples)


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
        return validate_dataset_dir(self.data, require=("train.jsonl",))

    def to_command(self) -> list[str]:
        self.validate()
        args = [
            sys.executable,
            "-m",
            "mlx_lm.lora",
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
        _append_opt(args, "--seed", self.seed)
        _append_opt(args, "--report-to", self.report_to)
        _append_opt(args, "--project-name", self.project_name)
        _append_opt(args, "--resume-adapter-file", self.resume_adapter_file)
        return args


@dataclass(frozen=True)
class EvalRecipe:
    model: str
    data: Path
    adapter_path: Path = Path("adapters")

    def validate(self) -> DatasetReport:
        return validate_dataset_dir(self.data, require=("test.jsonl",))

    def to_command(self) -> list[str]:
        self.validate()
        return [
            sys.executable,
            "-m",
            "mlx_lm.lora",
            "--model",
            self.model,
            "--adapter-path",
            str(self.adapter_path),
            "--data",
            str(self.data),
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


def run_command(args: list[str], *, dry_run: bool = False) -> int:
    if dry_run:
        print(" ".join(args))
        return 0
    try:
        return subprocess.call(args)
    except ModuleNotFoundError as e:
        raise RuntimeError("MLX-LM is not installed. Install with `pip install -e '.[mac]'`.") from e
