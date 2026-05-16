from __future__ import annotations

from functools import lru_cache
from typing import Any


FUSED_SILU_MUL_SOURCE = r"""
uint elem = thread_position_in_grid.x;
if (elem < n) {
    T av = a[elem];
    T bv = b[elem];
    T one = T(1);
    out[elem] = (av / (one + metal::exp(-av))) * bv;
}
"""


RMS_NORM_SOURCE = r"""
uint col = thread_position_in_grid.x;
uint row = thread_position_in_grid.y;
uint tid = thread_index_in_threadgroup;
uint k = n_cols;
float eps_v = eps;

threadgroup float scratch[256];
float sum_sq = 0.0f;
uint base = row * k;

for (uint c = tid; c < k; c += 256) {
    float xv = float(x[base + c]);
    sum_sq += xv * xv;
}
scratch[tid] = sum_sq;
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
        scratch[tid] += scratch[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

float inv_rms = metal::rsqrt((scratch[0] / float(k)) + eps_v);
for (uint c = tid; c < k; c += 256) {
    float xv = float(x[base + c]);
    float wv = float(weight[c]);
    out[base + c] = T(xv * inv_rms * wv);
}
"""


@lru_cache(maxsize=1)
def _fused_silu_mul_kernel() -> Any:
    try:
        import mlx.core as mx
    except ImportError as e:  # pragma: no cover - optional dependency
        raise RuntimeError("mlx is not installed. Install with `pip install -e '.[mac]'`.") from e

    return mx.fast.metal_kernel(
        name="slam_fused_silu_mul",
        input_names=["a", "b", "n"],
        output_names=["out"],
        source=FUSED_SILU_MUL_SOURCE,
        ensure_row_contiguous=True,
    )


def fused_silu_mul(a: Any, b: Any) -> Any:
    """Compute silu(a) * b with one custom MLX Metal kernel.

    This targets the common gated-MLP hot path used by Llama/Qwen-style blocks.
    It is intentionally narrow: prove correctness and speed on a simple fused
    elementwise op before moving to attention/KV-cache kernels.
    """
    try:
        import mlx.core as mx
    except ImportError as e:  # pragma: no cover - optional dependency
        raise RuntimeError("mlx is not installed. Install with `pip install -e '.[mac]'`.") from e

    if a.shape != b.shape:
        raise ValueError(f"a and b must have the same shape, got {a.shape} and {b.shape}")

    kernel = _fused_silu_mul_kernel()
    outputs = kernel(
        inputs=[a, b, mx.array(a.size, mx.uint32)],
        template=[("T", a.dtype)],
        grid=(a.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[a.shape],
        output_dtypes=[a.dtype],
    )
    return outputs[0]


def reference_silu_mul(a: Any, b: Any) -> Any:
    try:
        import mlx.core as mx
    except ImportError as e:  # pragma: no cover - optional dependency
        raise RuntimeError("mlx is not installed. Install with `pip install -e '.[mac]'`.") from e
    return (a / (1 + mx.exp(-a))) * b


@lru_cache(maxsize=1)
def _rms_norm_kernel() -> Any:
    try:
        import mlx.core as mx
    except ImportError as e:  # pragma: no cover - optional dependency
        raise RuntimeError("mlx is not installed. Install with `pip install -e '.[mac]'`.") from e

    return mx.fast.metal_kernel(
        name="slam_rms_norm",
        input_names=["x", "weight", "n_cols", "eps"],
        output_names=["out"],
        source=RMS_NORM_SOURCE,
        ensure_row_contiguous=True,
    )


def rms_norm(x: Any, weight: Any, eps: float = 1e-6) -> Any:
    try:
        import mlx.core as mx
    except ImportError as e:  # pragma: no cover - optional dependency
        raise RuntimeError("mlx is not installed. Install with `pip install -e '.[mac]'`.") from e

    if x.ndim != 2:
        raise ValueError(f"x must be 2D [rows, hidden], got shape {x.shape}")
    if weight.shape != (x.shape[-1],):
        raise ValueError(f"weight must have shape {(x.shape[-1],)}, got {weight.shape}")

    rows, cols = x.shape
    kernel = _rms_norm_kernel()
    outputs = kernel(
        inputs=[
            x,
            weight,
            mx.array(cols, mx.uint32),
            mx.array(eps, mx.float32),
        ],
        template=[("T", x.dtype)],
        grid=(256, rows, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )
    return outputs[0]


def reference_rms_norm(x: Any, weight: Any, eps: float = 1e-6) -> Any:
    try:
        import mlx.core as mx
    except ImportError as e:  # pragma: no cover - optional dependency
        raise RuntimeError("mlx is not installed. Install with `pip install -e '.[mac]'`.") from e
    x32 = x.astype(mx.float32)
    variance = mx.mean(x32 * x32, axis=-1, keepdims=True)
    return (x32 * mx.rsqrt(variance + eps) * weight.astype(mx.float32)).astype(x.dtype)
