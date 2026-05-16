from __future__ import annotations

import os

import pytest

from sl4m.backend import GenerationSettings, MLXBackend
from sl4m.budget import StaticBudget, auto_detect_limits
from sl4m.mac_efficiency import run_mlx_generation_bench
from sl4m.runtime import Session
from sl4m.telemetry import NullRecorder


pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")


MODEL = os.environ.get("SL4M_REAL_MODEL")
RUN_REASON = "set SL4M_REAL_MODEL to an MLX model id/path to run real-model smoke tests"


@pytest.mark.skipif(not MODEL, reason=RUN_REASON)
def test_real_model_baseline_generates_tokens() -> None:
    backend = MLXBackend()
    budget = StaticBudget(auto_detect_limits(headroom_bytes=4 * 1024**3))
    backend.load(MODEL, None, budget)
    session = Session(backend=backend, budget=budget, recorder=NullRecorder())

    try:
        tokens = list(
            session.generate(
                "Write one short sentence about Apple Silicon.",
                GenerationSettings(max_tokens=8, temperature=0.0),
            )
        )
    finally:
        session.close()

    assert any(tok.token_id != -1 for tok in tokens)
    assert "".join(tok.text for tok in tokens).strip()


@pytest.mark.skipif(not MODEL, reason=RUN_REASON)
def test_real_model_mac_bench_smoke() -> None:
    results = run_mlx_generation_bench(
        model=MODEL,
        prompt="Return the word ready.",
        modes=["baseline"],
        max_tokens=4,
        temperature=0.0,
        repeats=1,
        headroom_gb=4.0,
    )

    assert len(results) == 1
    assert results[0].tokens > 0
    assert results[0].tok_s > 0
