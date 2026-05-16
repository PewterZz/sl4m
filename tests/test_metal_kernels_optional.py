from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from sl4m.metal_kernels import (
    reference_residual_rms_norm,
    reference_rms_norm,
    residual_rms_norm,
    rms_norm,
)


def test_rms_norm_matches_reference_small() -> None:
    x = mx.random.normal((2, 512)).astype(mx.float16)
    weight = mx.random.normal((512,)).astype(mx.float16)

    actual = rms_norm(x, weight)
    expected = reference_rms_norm(x, weight)
    mx.eval(actual, expected)

    assert bool(mx.allclose(actual, expected, rtol=1e-3, atol=1e-3).item())


def test_rms_norm_rejects_bad_weight_shape() -> None:
    x = mx.random.normal((2, 512)).astype(mx.float16)
    weight = mx.ones((256,), dtype=mx.float16)

    with pytest.raises(ValueError, match="weight must have shape"):
        rms_norm(x, weight)


def test_residual_rms_norm_matches_reference_small() -> None:
    x = mx.random.normal((2, 512)).astype(mx.float16)
    residual = mx.random.normal((2, 512)).astype(mx.float16)
    weight = mx.random.normal((512,)).astype(mx.float16)

    actual = residual_rms_norm(x, residual, weight)
    expected = reference_residual_rms_norm(x, residual, weight)
    mx.eval(actual, expected)

    assert bool(mx.allclose(actual, expected, rtol=1e-3, atol=1e-3).item())


def test_residual_rms_norm_rejects_bad_residual_shape() -> None:
    x = mx.random.normal((2, 512)).astype(mx.float16)
    residual = mx.random.normal((1, 512)).astype(mx.float16)
    weight = mx.ones((512,), dtype=mx.float16)

    with pytest.raises(ValueError, match="residual must have shape"):
        residual_rms_norm(x, residual, weight)
