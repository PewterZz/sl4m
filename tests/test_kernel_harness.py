from __future__ import annotations

import pytest

from sl4m.kernel_harness import (
    CorrectnessMetrics,
    LinearBenchResult,
    LinearShape,
    default_linear_shapes,
    kernel_capabilities,
    parse_linear_shapes,
    run_linear_kernel_bench,
)


def test_default_linear_shapes_tiny_separates_decode_and_prefill() -> None:
    shapes = default_linear_shapes("tiny")
    assert [s.kind for s in shapes] == ["decode", "prefill"]
    assert shapes[0].m == 1
    assert shapes[1].m > 1


def test_default_linear_shapes_unknown_profile_fails() -> None:
    with pytest.raises(ValueError, match="unknown linear shape profile"):
        default_linear_shapes("missing")


def test_parse_linear_shapes() -> None:
    shapes = parse_linear_shapes("1x4096x4096,256x4096x14336")
    assert shapes == (
        LinearShape("custom-0-1x4096x4096", 1, 4096, 4096, "decode"),
        LinearShape("custom-1-256x4096x14336", 256, 4096, 14336, "prefill"),
    )


def test_parse_linear_shapes_rejects_bad_spec() -> None:
    with pytest.raises(ValueError, match="expected MxKxN"):
        parse_linear_shapes("1x2")


def test_kernel_capabilities_have_expected_names() -> None:
    names = {cap.name for cap in kernel_capabilities()}
    assert "mlx-metal-jit" in names
    assert "m5-int8-tensorops" in names
    assert "experimental-ane" in names


def test_linear_bench_result_record_and_tflops() -> None:
    result = LinearBenchResult(
        backend="candidate",
        shape=LinearShape("s", 2, 3, 4, "prefill"),
        dtype="float16",
        repeats=10,
        seconds=1.0,
        matmul_s=0.1,
        peak_mem_gb=0.5,
        correctness=CorrectnessMetrics(
            max_abs_error=0.1,
            mean_abs_error=0.01,
            cosine_similarity=0.99,
            allclose=False,
        ),
        metadata={"ops": 48},
    )

    record = result.to_record()

    assert result.tflops == 48 / 0.1 / 1e12
    assert record["shape"]["m"] == 2
    assert record["correctness"]["allclose"] is False


def test_run_linear_kernel_bench_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unsupported linear backend"):
        run_linear_kernel_bench(
            shapes=[LinearShape("s", 1, 32, 32, "decode")],
            backends=["missing"],
            repeats=1,
        )
