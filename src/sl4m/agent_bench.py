from __future__ import annotations

import ast
import json
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - py3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]

from .adaptive_pld import resolve_agent_mode
from .agent import AgentRoute, route_agent_prompt
from .backend import GenerationSettings, MLXBackend
from .budget import StaticBudget, auto_detect_limits
from .runtime import Session
from .telemetry import NullRecorder


SUPPORTED_MODES = {"baseline", "pld", "adaptive-pld", "agent-auto"}
SUPPORTED_OUTPUTS = {"code", "patch", "text"}
SUPPORTED_VALIDATORS = {"python-ast", "py-compile", "contains"}


@dataclass(frozen=True)
class AgentBenchValidation:
    kind: str
    value: Optional[str] = None


@dataclass(frozen=True)
class AgentBenchTask:
    name: str
    instruction: str
    context: str = ""
    output: str = "code"
    max_tokens: int = 256
    temperature: float = 0.0
    validations: tuple[AgentBenchValidation, ...] = ()
    path: Optional[Path] = None

    @property
    def prompt(self) -> str:
        if self.context:
            return f"{self.context.rstrip()}\n\n{self.instruction.strip()}"
        return self.instruction.strip()


@dataclass(frozen=True)
class ValidationResult:
    kind: str
    ok: bool
    detail: str = ""


@dataclass(frozen=True)
class AgentBenchResult:
    task: str
    mode: str
    selected_mode: str
    seconds: float
    tokens: int
    tok_s: float
    first_token_s: Optional[float]
    peak_mem_gb: Optional[float]
    route: Optional[dict[str, Any]]
    validations: tuple[ValidationResult, ...]
    output_prefix: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def validation_ok(self) -> bool:
        return all(v.ok for v in self.validations)

    def to_record(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["validation_ok"] = self.validation_ok
        return payload


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _read_context(root: Path, raw: dict[str, Any]) -> str:
    if "context" in raw:
        return str(raw["context"])
    if "context_file" in raw:
        p = (root / str(raw["context_file"])).resolve()
        if not p.exists():
            raise ValueError(f"context_file does not exist: {p}")
        return p.read_text(encoding="utf-8")
    return ""


def load_agent_task(path: str | Path) -> AgentBenchTask:
    p = Path(path)
    raw = _load_toml(p)
    task = raw.get("task", raw)
    validations = []
    for item in task.get("validations", []):
        if isinstance(item, str):
            validations.append(AgentBenchValidation(kind=item))
        elif isinstance(item, dict):
            validations.append(
                AgentBenchValidation(
                    kind=str(item.get("kind", "")),
                    value=str(item["value"]) if "value" in item else None,
                )
            )
        else:
            raise ValueError(f"{p}: validation entries must be strings or tables")

    loaded = AgentBenchTask(
        name=str(task.get("name", p.stem)),
        instruction=str(task["instruction"]),
        context=_read_context(p.parent, task),
        output=str(task.get("output", "code")),
        max_tokens=int(task.get("max_tokens", 256)),
        temperature=float(task.get("temperature", 0.0)),
        validations=tuple(validations),
        path=p,
    )
    validate_agent_task(loaded)
    return loaded


def load_agent_tasks(path: str | Path) -> list[AgentBenchTask]:
    p = Path(path)
    if p.is_file():
        return [load_agent_task(p)]
    if not p.is_dir():
        raise ValueError(f"agent-bench path must be a TOML file or directory: {p}")
    tasks = [load_agent_task(child) for child in sorted(p.glob("*.toml"))]
    if not tasks:
        raise ValueError(f"no .toml task files found in {p}")
    return tasks


def validate_agent_task(task: AgentBenchTask) -> None:
    if not task.name:
        raise ValueError("task name cannot be empty")
    if task.output not in SUPPORTED_OUTPUTS:
        raise ValueError(f"{task.name}: output must be one of {sorted(SUPPORTED_OUTPUTS)}")
    if task.max_tokens <= 0:
        raise ValueError(f"{task.name}: max_tokens must be > 0")
    for validation in task.validations:
        if validation.kind not in SUPPORTED_VALIDATORS:
            raise ValueError(
                f"{task.name}: unsupported validation {validation.kind!r}; "
                f"expected one of {sorted(SUPPORTED_VALIDATORS)}"
            )
        if validation.kind == "contains" and not validation.value:
            raise ValueError(f"{task.name}: contains validation needs value")


def validate_modes(modes: Iterable[str]) -> list[str]:
    out = [mode.strip() for mode in modes if mode.strip()]
    bad = [mode for mode in out if mode not in SUPPORTED_MODES]
    if bad:
        raise ValueError(f"unsupported agent-bench mode: {bad[0]}")
    if not out:
        raise ValueError("at least one mode is required")
    return out


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def validate_output(task: AgentBenchTask, output: str) -> tuple[ValidationResult, ...]:
    text = _strip_code_fence(output) if task.output == "code" else output.strip()
    results: list[ValidationResult] = []
    for validation in task.validations:
        if validation.kind == "python-ast":
            try:
                ast.parse(text)
            except SyntaxError as e:
                results.append(ValidationResult(validation.kind, False, str(e)))
            else:
                results.append(ValidationResult(validation.kind, True))
        elif validation.kind == "py-compile":
            with tempfile.TemporaryDirectory() as td:
                target = Path(td) / "candidate.py"
                target.write_text(text, encoding="utf-8")
                proc = subprocess.run(
                    ["python", "-m", "py_compile", str(target)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                results.append(
                    ValidationResult(
                        validation.kind,
                        proc.returncode == 0,
                        (proc.stderr or proc.stdout).strip(),
                    )
                )
        elif validation.kind == "contains":
            needle = validation.value or ""
            results.append(
                ValidationResult(
                    validation.kind,
                    needle in output,
                    "" if needle in output else f"missing {needle!r}",
                )
            )
    return tuple(results)


def _route_dict(route: AgentRoute) -> dict[str, Any]:
    return {
        "mode": route.mode,
        "confidence": route.confidence,
        "reason": route.reason,
        "prompt_chars": route.prompt_chars,
        "line_count": route.line_count,
        "code_marker_count": route.code_marker_count,
        "repeated_line_count": route.repeated_line_count,
    }


def _try_peak_memory_gb() -> Optional[float]:
    try:
        import mlx.core as mx

        if hasattr(mx, "get_peak_memory"):
            return float(mx.get_peak_memory()) / 1024**3
        if hasattr(mx, "metal") and hasattr(mx.metal, "get_peak_memory"):
            return float(mx.metal.get_peak_memory()) / 1024**3
    except Exception:
        pass
    return None


def _reset_peak_memory() -> None:
    try:
        import mlx.core as mx

        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "reset_peak_memory"):
            mx.metal.reset_peak_memory()
    except Exception:
        pass


def run_agent_bench(
    *,
    tasks: list[AgentBenchTask],
    model: str,
    modes: list[str],
    headroom_gb: float = 4.0,
    num_draft: int = 4,
    max_ngram: int = 3,
    min_ngram: int = 2,
) -> list[AgentBenchResult]:
    selected_modes = validate_modes(modes)
    results: list[AgentBenchResult] = []
    for task in tasks:
        for mode in selected_modes:
            route = route_agent_prompt(task.prompt)
            selected_mode = resolve_agent_mode(mode, route.mode)

            limits = auto_detect_limits(headroom_bytes=int(headroom_gb * 1024**3))
            budget = StaticBudget(limits)
            backend = MLXBackend()
            backend.load(model, None, budget)
            session = Session(backend=backend, budget=budget, recorder=NullRecorder())
            settings = GenerationSettings(
                max_tokens=task.max_tokens,
                temperature=task.temperature,
            )
            text_parts: list[str] = []
            token_count = 0
            first_token_s: Optional[float] = None
            _reset_peak_memory()
            t0 = time.monotonic()
            try:
                if selected_mode == "pld":
                    stream = session.generate_pld_speculative(
                        task.prompt,
                        settings,
                        num_draft=num_draft,
                        max_ngram_size=max_ngram,
                        min_ngram_size=min_ngram,
                    )
                elif selected_mode == "adaptive-pld":
                    stream = session.generate_adaptive_pld(
                        task.prompt,
                        settings,
                        num_draft=num_draft,
                        max_ngram_size=max_ngram,
                        min_ngram_size=min_ngram,
                    )
                else:
                    stream = session.generate(task.prompt, settings)
                for tok in stream:
                    text_parts.append(tok.text)
                    if tok.token_id == -1:
                        continue
                    if first_token_s is None:
                        first_token_s = time.monotonic() - t0
                    token_count += 1
            finally:
                seconds = time.monotonic() - t0
                peak = _try_peak_memory_gb()
                session.close()

            output = "".join(text_parts)
            results.append(
                AgentBenchResult(
                    task=task.name,
                    mode=mode,
                    selected_mode=selected_mode,
                    seconds=seconds,
                    tokens=token_count,
                    tok_s=token_count / seconds if seconds > 0 else 0.0,
                    first_token_s=first_token_s,
                    peak_mem_gb=peak,
                    route=_route_dict(route),
                    validations=validate_output(task, output),
                    output_prefix=output[:240],
                    metadata={
                        "task_path": str(task.path) if task.path else None,
                        "output": task.output,
                    },
                )
            )
    return results


def write_agent_bench_jsonl(path: str | Path, results: Iterable[AgentBenchResult]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result.to_record()) + "\n")
