from __future__ import annotations

import json
from pathlib import Path

from sl4m.mac_efficiency import (
    BenchmarkResult,
    build_mac_acceleration_plan,
    compare_mlx_lm_vs_slam,
    collect_mac_profile,
    detect_apple_chip_generation,
    summarize_by_best,
    write_jsonl,
)
from sl4m.metal_kernels import FUSED_SILU_MUL_SOURCE
from sl4m.metal_kernels import RMS_NORM_SOURCE


def test_collect_mac_profile_has_stable_fields() -> None:
    profile = collect_mac_profile()
    assert profile.system
    assert profile.python
    assert profile.ram_gb > 0
    assert profile.apple_chip_generation >= 0


def test_detect_apple_chip_generation_is_non_negative() -> None:
    assert detect_apple_chip_generation() >= 0


def test_acceleration_plan_gates_int8_tensorops_by_chip() -> None:
    m4 = build_mac_acceleration_plan(4)
    m5 = build_mac_acceleration_plan(5)

    m4_int8 = next(o for o in m4.opportunities if o.name.startswith("W8A8"))
    m5_int8 = next(o for o in m5.opportunities if o.name.startswith("W8A8"))
    ane = next(o for o in m5.opportunities if o.name.startswith("ANE"))

    assert m4_int8.status == "unavailable"
    assert m5_int8.status == "research candidate"
    assert ane.status == "experimental only"


def test_write_jsonl_serializes_dataclasses(tmp_path: Path) -> None:
    out = tmp_path / "bench.jsonl"
    row = BenchmarkResult(
        name="baseline",
        model="m",
        prompt_chars=3,
        tokens=10,
        seconds=2.0,
        tok_s=5.0,
    )

    write_jsonl(out, [row])

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["name"] == "baseline"
    assert data["tok_s"] == 5.0


def test_benchmark_result_to_record_is_stable() -> None:
    row = BenchmarkResult(
        name="mode",
        model="model",
        prompt_chars=12,
        tokens=6,
        seconds=3.0,
        tok_s=2.0,
        first_token_s=0.5,
        peak_mem_gb=1.25,
        metadata={"repeat": 1},
    )

    assert row.to_record() == {
        "name": "mode",
        "model": "model",
        "prompt_chars": 12,
        "tokens": 6,
        "seconds": 3.0,
        "tok_s": 2.0,
        "first_token_s": 0.5,
        "peak_mem_gb": 1.25,
        "metadata": {"repeat": 1},
    }


def test_summarize_by_best_keeps_fastest_per_name() -> None:
    rows = [
        BenchmarkResult("a", "m", 0, 1, 1.0, 1.0),
        BenchmarkResult("a", "m", 0, 1, 1.0, 3.0),
        BenchmarkResult("b", "m", 0, 1, 1.0, 2.0),
    ]

    best = summarize_by_best(rows)

    assert [(r.name, r.tok_s) for r in best] == [("a", 3.0), ("b", 2.0)]


def test_compare_mlx_lm_vs_slam_records_identity(monkeypatch) -> None:
    def direct(**kwargs):
        return (
            BenchmarkResult("mlx-lm-direct", kwargs["model"], 6, 3, 1.5, 2.0),
            [1, 2, 3],
            "abc",
        )

    def slam(**kwargs):
        return (
            BenchmarkResult("slam-baseline", kwargs["model"], 6, 3, 1.0, 3.0),
            [1, 2, 3],
            "abc",
        )

    monkeypatch.setattr("sl4m.mac_efficiency._measure_mlx_lm_direct", direct)
    monkeypatch.setattr("sl4m.mac_efficiency._measure_slam_baseline", slam)

    results = compare_mlx_lm_vs_slam(
        model="m",
        prompt="prompt",
        max_tokens=3,
        temperature=0.0,
    )

    assert [r.name for r in results] == ["mlx-lm-direct", "slam-baseline"]
    assert all(r.metadata["token_identity"] is True for r in results)
    assert results[1].tok_s / results[0].tok_s == 1.5


def test_compare_mlx_lm_vs_slam_records_divergence(monkeypatch) -> None:
    def direct(**kwargs):
        return (BenchmarkResult("mlx-lm-direct", kwargs["model"], 6, 3, 1.0, 3.0), [1, 2, 3], "")

    def slam(**kwargs):
        return (BenchmarkResult("slam-baseline", kwargs["model"], 6, 3, 1.0, 3.0), [1, 9, 3], "")

    monkeypatch.setattr("sl4m.mac_efficiency._measure_mlx_lm_direct", direct)
    monkeypatch.setattr("sl4m.mac_efficiency._measure_slam_baseline", slam)

    results = compare_mlx_lm_vs_slam(
        model="m",
        prompt="prompt",
        max_tokens=3,
        temperature=0.0,
    )

    assert results[0].metadata["token_identity"] is False
    assert results[0].metadata["token_prefix_match"] == 1


def test_fused_silu_mul_kernel_source_contains_bounds_and_fusion() -> None:
    assert "elem < n" in FUSED_SILU_MUL_SOURCE
    assert "metal::exp" in FUSED_SILU_MUL_SOURCE
    assert "* bv" in FUSED_SILU_MUL_SOURCE


def test_rms_norm_kernel_source_contains_reduction_and_weight() -> None:
    assert "threadgroup float scratch" in RMS_NORM_SOURCE
    assert "threadgroup_barrier" in RMS_NORM_SOURCE
    assert "metal::rsqrt" in RMS_NORM_SOURCE
    assert "* wv" in RMS_NORM_SOURCE
