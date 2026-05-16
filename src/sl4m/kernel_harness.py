from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from .mac_efficiency import detect_apple_chip_generation


@dataclass(frozen=True)
class KernelCapability:
    name: str
    available: bool
    reason: str


@dataclass(frozen=True)
class LinearShape:
    name: str
    m: int
    k: int
    n: int
    kind: str


@dataclass(frozen=True)
class CorrectnessMetrics:
    max_abs_error: float
    mean_abs_error: float
    cosine_similarity: float
    allclose: bool


@dataclass(frozen=True)
class LinearBenchResult:
    backend: str
    shape: LinearShape
    dtype: str
    repeats: int
    seconds: float
    matmul_s: float
    peak_mem_gb: Optional[float]
    correctness: Optional[CorrectnessMetrics] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tflops(self) -> float:
        ops = float(self.metadata.get("ops") or 0.0)
        return ops / self.matmul_s / 1e12 if self.matmul_s > 0 else 0.0

    def to_record(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "shape": {
                "name": self.shape.name,
                "m": self.shape.m,
                "k": self.shape.k,
                "n": self.shape.n,
                "kind": self.shape.kind,
            },
            "dtype": self.dtype,
            "repeats": self.repeats,
            "seconds": self.seconds,
            "matmul_s": self.matmul_s,
            "tflops": self.tflops,
            "peak_mem_gb": self.peak_mem_gb,
            "correctness": None
            if self.correctness is None
            else {
                "max_abs_error": self.correctness.max_abs_error,
                "mean_abs_error": self.correctness.mean_abs_error,
                "cosine_similarity": self.correctness.cosine_similarity,
                "allclose": self.correctness.allclose,
            },
            "metadata": self.metadata,
        }


def kernel_capabilities() -> tuple[KernelCapability, ...]:
    caps: list[KernelCapability] = []
    try:
        import mlx.core as mx

        has_metal_kernel = hasattr(mx, "fast") and hasattr(mx.fast, "metal_kernel")
        caps.append(
            KernelCapability(
                name="mlx-metal-jit",
                available=has_metal_kernel,
                reason="MLX mx.fast.metal_kernel is available"
                if has_metal_kernel
                else "MLX custom Metal JIT API not found",
            )
        )
    except Exception as e:
        caps.append(KernelCapability("mlx-metal-jit", False, f"MLX import failed: {e}"))

    chip = detect_apple_chip_generation()
    caps.append(
        KernelCapability(
            name="m5-int8-tensorops",
            available=chip >= 5,
            reason=f"Detected Apple M{chip}" if chip else "Apple chip generation unknown",
        )
    )
    caps.append(
        KernelCapability(
            name="experimental-ane",
            available=False,
            reason="Private ANE APIs are intentionally excluded from default slam paths",
        )
    )
    return tuple(caps)


def default_linear_shapes(profile: str = "tiny") -> tuple[LinearShape, ...]:
    if profile == "tiny":
        return (
            LinearShape("decode-tiny", 1, 512, 512, "decode"),
            LinearShape("prefill-tiny", 64, 512, 512, "prefill"),
        )
    if profile == "qwen2_7b":
        hidden = 3584
        intermediate = 18944
        return (
            LinearShape("decode-q-proj", 1, hidden, hidden, "decode"),
            LinearShape("prefill-q-proj-256", 256, hidden, hidden, "prefill"),
            LinearShape("decode-gate-up", 1, hidden, intermediate, "decode"),
            LinearShape("prefill-gate-up-256", 256, hidden, intermediate, "prefill"),
            LinearShape("decode-down", 1, intermediate, hidden, "decode"),
            LinearShape("prefill-down-256", 256, intermediate, hidden, "prefill"),
        )
    if profile == "llama3_8b":
        hidden = 4096
        intermediate = 14336
        return (
            LinearShape("decode-q-proj", 1, hidden, hidden, "decode"),
            LinearShape("prefill-q-proj-256", 256, hidden, hidden, "prefill"),
            LinearShape("decode-gate-up", 1, hidden, intermediate, "decode"),
            LinearShape("prefill-gate-up-256", 256, hidden, intermediate, "prefill"),
            LinearShape("decode-down", 1, intermediate, hidden, "decode"),
            LinearShape("prefill-down-256", 256, intermediate, hidden, "prefill"),
        )
    raise ValueError(f"unknown linear shape profile: {profile}")


def parse_linear_shapes(spec: str) -> tuple[LinearShape, ...]:
    shapes: list[LinearShape] = []
    for idx, item in enumerate(part.strip() for part in spec.split(",") if part.strip()):
        try:
            m_s, k_s, n_s = item.lower().split("x")
            m, k, n = int(m_s), int(k_s), int(n_s)
        except ValueError as e:
            raise ValueError(f"invalid shape {item!r}; expected MxKxN") from e
        kind = "decode" if m == 1 else "prefill"
        shapes.append(LinearShape(f"custom-{idx}-{m}x{k}x{n}", m, k, n, kind))
    return tuple(shapes)


def _peak_memory_gb() -> Optional[float]:
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


def _correctness(actual: Any, expected: Any, *, rtol: float, atol: float) -> CorrectnessMetrics:
    import mlx.core as mx

    diff = mx.abs(actual - expected)
    flat_actual = actual.reshape(-1).astype(mx.float32)
    flat_expected = expected.reshape(-1).astype(mx.float32)
    denom = mx.linalg.norm(flat_actual) * mx.linalg.norm(flat_expected)
    cosine = mx.sum(flat_actual * flat_expected) / denom
    mx.eval(diff, cosine)
    return CorrectnessMetrics(
        max_abs_error=float(mx.max(diff).item()),
        mean_abs_error=float(mx.mean(diff).item()),
        cosine_similarity=float(cosine.item()),
        allclose=bool(mx.allclose(actual, expected, rtol=rtol, atol=atol).item()),
    )


def _make_inputs(shape: LinearShape, dtype: str, seed: int) -> tuple[Any, Any]:
    import mlx.core as mx

    mx.random.seed(seed)
    mx_dtype = getattr(mx, dtype)
    x = mx.random.normal((shape.m, shape.k)).astype(mx_dtype)
    w = mx.random.normal((shape.n, shape.k)).astype(mx_dtype)
    mx.eval(x, w)
    return x, w


def _time_backend(fn: Callable[[], Any], repeats: int) -> tuple[float, float, Optional[float]]:
    import mlx.core as mx

    mx.eval(fn())
    _reset_peak_memory()
    t0 = time.monotonic()
    for _ in range(repeats):
        y = fn()
        mx.eval(y)
    seconds = time.monotonic() - t0
    return seconds, seconds / repeats if repeats else 0.0, _peak_memory_gb()


def _mlx_matmul(x: Any, w: Any) -> Any:
    return x @ w.T


def _mlx_quantized_roundtrip(x: Any, w: Any, group_size: int, bits: int) -> Any:
    import mlx.core as mx

    quantized = mx.quantize(w, group_size=group_size, bits=bits)
    if len(quantized) == 2:
        q_weight, scales = quantized
        biases = None
    elif len(quantized) == 3:
        q_weight, scales, biases = quantized
    else:
        raise RuntimeError(f"unexpected mx.quantize return arity: {len(quantized)}")
    q_w = mx.dequantize(q_weight, scales, biases, group_size, bits)
    return x @ q_w.T


def run_linear_kernel_bench(
    *,
    shapes: Iterable[LinearShape],
    backends: list[str],
    dtype: str = "float16",
    repeats: int = 20,
    seed: int = 0,
    quant_group_size: int = 64,
    quant_bits: int = 4,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> list[LinearBenchResult]:
    supported_backends = {"mlx-matmul", "mlx-quantized-roundtrip"}
    unsupported = [backend for backend in backends if backend not in supported_backends]
    if unsupported:
        raise ValueError(f"unsupported linear backend: {unsupported[0]}")

    results: list[LinearBenchResult] = []
    for shape in shapes:
        x, w = _make_inputs(shape, dtype, seed)
        ref = _mlx_matmul(x, w)
        for backend in backends:
            if backend == "mlx-matmul":
                fn = lambda x=x, w=w: _mlx_matmul(x, w)
                correctness = None
            elif backend == "mlx-quantized-roundtrip":
                fn = lambda x=x, w=w: _mlx_quantized_roundtrip(
                    x, w, group_size=quant_group_size, bits=quant_bits
                )
                correctness = _correctness(fn(), ref, rtol=rtol, atol=atol)
            else:
                raise ValueError(f"unsupported linear backend: {backend}")

            seconds, matmul_s, peak = _time_backend(fn, repeats)
            results.append(
                LinearBenchResult(
                    backend=backend,
                    shape=shape,
                    dtype=dtype,
                    repeats=repeats,
                    seconds=seconds,
                    matmul_s=matmul_s,
                    peak_mem_gb=peak,
                    correctness=correctness,
                    metadata={
                        "ops": 2 * shape.m * shape.k * shape.n,
                        "quant_group_size": quant_group_size
                        if backend == "mlx-quantized-roundtrip"
                        else None,
                        "quant_bits": quant_bits
                        if backend == "mlx-quantized-roundtrip"
                        else None,
                    },
                )
            )
    return results
