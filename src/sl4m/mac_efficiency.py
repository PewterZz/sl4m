from __future__ import annotations

import json
import platform
import re
import subprocess
import time
import gc
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import psutil

from .backend import GenerationSettings, MLXBackend
from .budget import StaticBudget, auto_detect_limits
from .runtime import Session
from .telemetry import NullRecorder


@dataclass(frozen=True)
class MacProfile:
    system: str
    release: str
    machine: str
    cpu: str
    ram_gb: float
    python: str
    apple_chip_generation: int = 0
    mlx_version: Optional[str] = None
    mlx_device: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    model: str
    prompt_chars: int
    tokens: int
    seconds: float
    tok_s: float
    first_token_s: Optional[float] = None
    peak_mem_gb: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "prompt_chars": self.prompt_chars,
            "tokens": self.tokens,
            "seconds": self.seconds,
            "tok_s": self.tok_s,
            "first_token_s": self.first_token_s,
            "peak_mem_gb": self.peak_mem_gb,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class AccelerationOpportunity:
    name: str
    status: str
    applies_to: str
    reason: str
    guardrail: str


@dataclass(frozen=True)
class MacAccelerationPlan:
    chip_generation: int
    opportunities: tuple[AccelerationOpportunity, ...]


def detect_apple_chip_generation() -> int:
    if platform.system() != "Darwin":
        return 0
    candidates: list[str] = []
    try:
        brand = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        ).stdout.strip()
        if brand:
            candidates.append(brand)
    except Exception:
        pass
    try:
        import mlx.core as mx

        info = mx.device_info()
        device_name = str(info.get("device_name", ""))
        if device_name:
            candidates.append(device_name)
    except Exception:
        pass
    for value in candidates:
        match = re.search(r"Apple M(\d+)", value)
        if match:
            return int(match.group(1))
    return 0


def build_mac_acceleration_plan(chip_generation: int) -> MacAccelerationPlan:
    opportunities: list[AccelerationOpportunity] = [
        AccelerationOpportunity(
            name="prefill/decode split",
            status="production",
            applies_to="inference, serving, eval",
            reason="Long-context prefill and single-token decode stress different kernels and should be benchmarked separately.",
            guardrail="Report TTFT/prefill and decode tok/s separately; do not optimize one with only aggregate tok/s.",
        ),
        AccelerationOpportunity(
            name="custom MLX Metal kernels",
            status="production candidate",
            applies_to="inference hot paths; training only after backward path exists",
            reason="MLX custom kernels preserve lazy graph scheduling when exposed as MLX operations.",
            guardrail="Every kernel needs reference correctness, JSONL benchmark metadata, and shape/dtype gates.",
        ),
    ]

    if chip_generation >= 5:
        opportunities.append(
            AccelerationOpportunity(
                name="W8A8 / W4A8 prefill via INT8 TensorOps",
                status="research candidate",
                applies_to="inference prefill and VLM prefill; not LoRA training yet",
                reason="Apple M5+ exposes INT8 TensorOps through Metal 4 cooperative tensor APIs; Cider reports prefill gains when activation quantization is fused with matmul/dequant.",
                guardrail="Enable only for calibrated, quantization-friendly models; keep decode fallback and run perplexity/correctness checks.",
            )
        )
    elif chip_generation > 0:
        opportunities.append(
            AccelerationOpportunity(
                name="W8A8 / W4A8 prefill via INT8 TensorOps",
                status="unavailable",
                applies_to="M5+ only",
                reason=f"Detected Apple M{chip_generation}; Cider gates these kernels to M5+.",
                guardrail="Install and run pure MLX fallback paths cleanly on M1-M4.",
            )
        )
    else:
        opportunities.append(
            AccelerationOpportunity(
                name="W8A8 / W4A8 prefill via INT8 TensorOps",
                status="unknown",
                applies_to="M5+ only",
                reason="Apple chip generation could not be detected.",
                guardrail="Require explicit hardware detection before enabling INT8 TensorOps kernels.",
            )
        )

    opportunities.append(
        AccelerationOpportunity(
            name="ANE+GPU tensor split",
            status="experimental only",
            applies_to="prefill-only research",
            reason="Cider observed idle ANE capacity, but their ANE split lacks end-to-end gains without MLX lazy-eval-compatible integration and uses private APIs.",
            guardrail="Do not put this in default inference/training paths; keep behind explicit experimental flags.",
        )
    )

    return MacAccelerationPlan(
        chip_generation=chip_generation,
        opportunities=tuple(opportunities),
    )


def collect_mac_profile() -> MacProfile:
    mlx_version: Optional[str] = None
    mlx_device: Optional[dict[str, Any]] = None
    try:
        import mlx.core as mx

        mlx_version = getattr(mx, "__version__", None)
        mlx_device = dict(mx.device_info())
    except Exception:
        pass

    return MacProfile(
        system=platform.system(),
        release=platform.release(),
        machine=platform.machine(),
        cpu=platform.processor(),
        ram_gb=round(psutil.virtual_memory().total / 1024**3, 2),
        python=platform.python_version(),
        apple_chip_generation=detect_apple_chip_generation(),
        mlx_version=mlx_version,
        mlx_device=mlx_device,
    )


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            payload = asdict(row) if hasattr(row, "__dataclass_fields__") else row
            f.write(json.dumps(payload) + "\n")


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


def _clear_mlx_cache() -> None:
    try:
        import mlx.core as mx

        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
    except Exception:
        pass
    gc.collect()


def _measure_generation(
    name: str,
    model: str,
    prompt: str,
    fn: Callable[[Session, GenerationSettings], Iterable[Any]],
    *,
    max_tokens: int,
    temperature: float,
    headroom_gb: float,
) -> BenchmarkResult:
    limits = auto_detect_limits(headroom_bytes=int(headroom_gb * 1024**3))
    budget = StaticBudget(limits)
    backend = MLXBackend()
    backend.load(model, None, budget)
    session = Session(backend=backend, budget=budget, recorder=NullRecorder())
    settings = GenerationSettings(max_tokens=max_tokens, temperature=temperature)

    first_token_s: Optional[float] = None
    token_count = 0
    _reset_peak_memory()
    t0 = time.monotonic()
    try:
        for tok in fn(session, settings):
            if tok.token_id == -1:
                continue
            if first_token_s is None:
                first_token_s = time.monotonic() - t0
            token_count += 1
    finally:
        elapsed = time.monotonic() - t0
        peak = _try_peak_memory_gb()
        session.close()

    return BenchmarkResult(
        name=name,
        model=model,
        prompt_chars=len(prompt),
        tokens=token_count,
        seconds=elapsed,
        tok_s=(token_count / elapsed) if elapsed > 0 else 0.0,
        first_token_s=first_token_s,
        peak_mem_gb=peak,
    )


def _measure_mlx_lm_direct(
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float = 0.95,
) -> tuple[BenchmarkResult, list[int], str]:
    try:
        import mlx.core as mx
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler
    except ImportError as e:  # pragma: no cover - optional dependency
        raise RuntimeError("mlx-lm is not installed. Install with `pip install -e '.[mac]'`.") from e

    model_obj, tokenizer = load(model)
    sampler = make_sampler(temp=temperature, top_p=top_p)
    tokens: list[int] = []
    text: list[str] = []
    first_token_s: Optional[float] = None
    _reset_peak_memory()
    t0 = time.monotonic()
    try:
        for resp in stream_generate(
            model_obj,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        ):
            if first_token_s is None:
                first_token_s = time.monotonic() - t0
            tokens.append(int(resp.token))
            text.append(resp.text)
    finally:
        elapsed = time.monotonic() - t0
        peak = _try_peak_memory_gb()
        del model_obj
        del tokenizer
        _clear_mlx_cache()

    result = BenchmarkResult(
        name="mlx-lm-direct",
        model=model,
        prompt_chars=len(prompt),
        tokens=len(tokens),
        seconds=elapsed,
        tok_s=(len(tokens) / elapsed) if elapsed > 0 else 0.0,
        first_token_s=first_token_s,
        peak_mem_gb=peak,
        metadata={"text_prefix": "".join(text)[:200]},
    )
    return result, tokens, "".join(text)


def _measure_slam_baseline(
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float = 0.95,
    headroom_gb: float = 4.0,
) -> tuple[BenchmarkResult, list[int], str]:
    limits = auto_detect_limits(headroom_bytes=int(headroom_gb * 1024**3))
    budget = StaticBudget(limits)
    backend = MLXBackend()
    backend.load(model, None, budget)
    session = Session(backend=backend, budget=budget, recorder=NullRecorder())
    settings = GenerationSettings(max_tokens=max_tokens, temperature=temperature, top_p=top_p)
    tokens: list[int] = []
    text: list[str] = []
    first_token_s: Optional[float] = None
    _reset_peak_memory()
    t0 = time.monotonic()
    try:
        for tok in session.generate(prompt, settings):
            if tok.token_id == -1:
                text.append(tok.text)
                continue
            if first_token_s is None:
                first_token_s = time.monotonic() - t0
            tokens.append(tok.token_id)
            text.append(tok.text)
    finally:
        elapsed = time.monotonic() - t0
        peak = _try_peak_memory_gb()
        session.close()
        _clear_mlx_cache()

    result = BenchmarkResult(
        name="slam-baseline",
        model=model,
        prompt_chars=len(prompt),
        tokens=len(tokens),
        seconds=elapsed,
        tok_s=(len(tokens) / elapsed) if elapsed > 0 else 0.0,
        first_token_s=first_token_s,
        peak_mem_gb=peak,
        metadata={"text_prefix": "".join(text)[:200]},
    )
    return result, tokens, "".join(text)


def compare_mlx_lm_vs_slam(
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    repeats: int = 1,
    headroom_gb: float = 4.0,
    top_p: float = 0.95,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    for repeat in range(repeats):
        direct, direct_tokens, _direct_text = _measure_mlx_lm_direct(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        slam_result, slam_tokens, _slam_text = _measure_slam_baseline(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            headroom_gb=headroom_gb,
        )
        n = min(len(direct_tokens), len(slam_tokens))
        diverge = next((i for i in range(n) if direct_tokens[i] != slam_tokens[i]), None)
        same = diverge is None and len(direct_tokens) == len(slam_tokens)
        common = {
            "repeat": repeat,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "token_identity": same,
            "token_prefix_match": n if diverge is None else diverge,
            "direct_tokens": len(direct_tokens),
            "slam_tokens": len(slam_tokens),
        }
        results.append(
            BenchmarkResult(
                **{
                    **direct.to_record(),
                    "metadata": {**direct.metadata, **common, "comparison": "direct"},
                }
            )
        )
        results.append(
            BenchmarkResult(
                **{
                    **slam_result.to_record(),
                    "metadata": {**slam_result.metadata, **common, "comparison": "slam"},
                }
            )
        )
    return results


def run_mlx_generation_bench(
    *,
    model: str,
    prompt: str,
    modes: list[str],
    max_tokens: int,
    temperature: float,
    repeats: int,
    headroom_gb: float,
    num_draft: int = 4,
    max_ngram: int = 3,
    min_ngram: int = 2,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    for rep in range(repeats):
        for mode in modes:
            if mode == "baseline":
                fn = lambda session, settings: session.generate(prompt, settings)
            elif mode == "pld":
                fn = lambda session, settings: session.generate_pld_speculative(
                    prompt,
                    settings,
                    num_draft=num_draft,
                    max_ngram_size=max_ngram,
                    min_ngram_size=min_ngram,
                )
            else:
                raise ValueError(f"unsupported benchmark mode: {mode}")
            result = _measure_generation(
                mode,
                model,
                prompt,
                fn,
                max_tokens=max_tokens,
                temperature=temperature,
                headroom_gb=headroom_gb,
            )
            results.append(
                BenchmarkResult(
                    **{**asdict(result), "metadata": {**result.metadata, "repeat": rep}}
                )
            )
    return results


def run_command_benchmark(
    name: str,
    command: list[str],
    *,
    timeout_s: float = 600.0,
) -> BenchmarkResult:
    """Benchmark an external baseline command by wall time.

    This is deliberately generic so llama.cpp/Ollama/MLX-LM command baselines
    can be logged beside slam runs even before their output schemas match.
    """
    t0 = time.monotonic()
    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    elapsed = time.monotonic() - t0
    return BenchmarkResult(
        name=name,
        model="external",
        prompt_chars=0,
        tokens=0,
        seconds=elapsed,
        tok_s=0.0,
        metadata={
            "command": command,
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        },
    )


def summarize_by_best(results: Iterable[BenchmarkResult]) -> list[BenchmarkResult]:
    best: dict[str, BenchmarkResult] = {}
    for result in results:
        current = best.get(result.name)
        if current is None or result.tok_s > current.tok_s:
            best[result.name] = result
    return [best[k] for k in sorted(best)]


def run_fused_silu_mul_bench(
    *,
    size: int = 16_777_216,
    dtype: str = "float16",
    repeats: int = 50,
) -> list[BenchmarkResult]:
    try:
        import mlx.core as mx
    except ImportError as e:  # pragma: no cover - optional dependency
        raise RuntimeError("mlx is not installed. Install with `pip install -e '.[mac]'`.") from e

    from .metal_kernels import fused_silu_mul, reference_silu_mul

    mx_dtype = getattr(mx, dtype)
    a = mx.random.normal((size,)).astype(mx_dtype)
    b = mx.random.normal((size,)).astype(mx_dtype)

    def measure(name: str, fn: Callable[[], Any]) -> BenchmarkResult:
        # Compile/warm before timing.
        mx.eval(fn())
        _reset_peak_memory()
        t0 = time.monotonic()
        for _ in range(repeats):
            y = fn()
            mx.eval(y)
        elapsed = time.monotonic() - t0
        elems = size * repeats
        return BenchmarkResult(
            name=name,
            model="kernel:fused_silu_mul",
            prompt_chars=0,
            tokens=elems,
            seconds=elapsed,
            tok_s=elems / elapsed if elapsed > 0 else 0.0,
            peak_mem_gb=_try_peak_memory_gb(),
            metadata={"size": size, "dtype": dtype, "repeats": repeats, "unit": "elem/s"},
        )

    custom = fused_silu_mul(a, b)
    ref = reference_silu_mul(a, b)
    if not bool(mx.allclose(custom, ref, rtol=1e-3, atol=1e-3)):
        raise RuntimeError("custom fused_silu_mul failed correctness check vs MLX reference")

    return [
        measure("mlx-reference-silu-mul", lambda: reference_silu_mul(a, b)),
        measure("slam-metal-fused-silu-mul", lambda: fused_silu_mul(a, b)),
    ]


def run_rms_norm_bench(
    *,
    rows: int = 256,
    hidden: int = 4096,
    dtype: str = "float16",
    eps: float = 1e-6,
    repeats: int = 50,
) -> list[BenchmarkResult]:
    try:
        import mlx.core as mx
    except ImportError as e:  # pragma: no cover - optional dependency
        raise RuntimeError("mlx is not installed. Install with `pip install -e '.[mac]'`.") from e

    from .metal_kernels import (
        reference_residual_rms_norm,
        reference_rms_norm,
        residual_rms_norm,
        rms_norm,
    )

    mx_dtype = getattr(mx, dtype)
    x = mx.random.normal((rows, hidden)).astype(mx_dtype)
    residual = mx.random.normal((rows, hidden)).astype(mx_dtype)
    weight = mx.random.normal((hidden,)).astype(mx_dtype)

    custom = rms_norm(x, weight, eps=eps)
    ref = reference_rms_norm(x, weight, eps=eps)
    if not bool(mx.allclose(custom, ref, rtol=1e-3, atol=1e-3).item()):
        max_err = float(mx.max(mx.abs(custom - ref)).item())
        raise RuntimeError(f"custom rms_norm failed correctness check; max_err={max_err:.4g}")

    def measure(name: str, fn: Callable[[], Any]) -> BenchmarkResult:
        mx.eval(fn())
        _reset_peak_memory()
        t0 = time.monotonic()
        for _ in range(repeats):
            y = fn()
            mx.eval(y)
        elapsed = time.monotonic() - t0
        elems = rows * hidden * repeats
        return BenchmarkResult(
            name=name,
            model="kernel:rms_norm",
            prompt_chars=0,
            tokens=elems,
            seconds=elapsed,
            tok_s=elems / elapsed if elapsed > 0 else 0.0,
            peak_mem_gb=_try_peak_memory_gb(),
            metadata={
                "rows": rows,
                "hidden": hidden,
                "dtype": dtype,
                "eps": eps,
                "repeats": repeats,
                "unit": "elem/s",
            },
        )

    residual_custom = residual_rms_norm(x, residual, weight, eps=eps)
    residual_ref = reference_residual_rms_norm(x, residual, weight, eps=eps)
    if not bool(mx.allclose(residual_custom, residual_ref, rtol=1e-3, atol=1e-3).item()):
        max_err = float(mx.max(mx.abs(residual_custom - residual_ref)).item())
        raise RuntimeError(
            f"custom residual_rms_norm failed correctness check; max_err={max_err:.4g}"
        )

    return [
        measure("mlx-reference-rms-norm", lambda: reference_rms_norm(x, weight, eps=eps)),
        measure("slam-metal-rms-norm", lambda: rms_norm(x, weight, eps=eps)),
        measure(
            "mlx-reference-residual-rms-norm",
            lambda: reference_residual_rms_norm(x, residual, weight, eps=eps),
        ),
        measure(
            "slam-metal-residual-rms-norm",
            lambda: residual_rms_norm(x, residual, weight, eps=eps),
        ),
    ]
