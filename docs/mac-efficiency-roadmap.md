# Mac Efficiency Roadmap

sl4m should become a Mac-first efficiency layer for local training,
fine-tuning, and serving. The rule is simple: every optimization needs a
baseline, a profile, a correctness check, and a repeatable benchmark.

## Baselines

Track sl4m against:

- MLX-LM direct generation and LoRA/fuse/server flows
- llama.cpp Metal for GGUF serving
- Ollama / LM Studio only when the comparison is user-visible serving latency
- sl4m baseline generation before any PLD/spec/custom-kernel path

Record:

- model id and quantization
- macOS version, chip, RAM, MLX version, Python version
- prompt length, output tokens, temperature, context size
- TTFT, decode tok/s, peak Metal memory when available
- correctness result for deterministic runs

## Kernel Targets

Start with kernels that are narrow enough to validate and common enough to
matter:

1. Fused SiLU multiply for gated MLPs: `silu(a) * b`
2. RMSNorm variants with residual fusion
3. M5-only W8A8 prefill linear: activation quantize + INT8 GEMM + fused dequant
4. KV-cache packing/unpacking and low-bit decode kernels
5. Attention verify kernels for speculative decoding
6. MoE routing/expert hot-set kernels once routing telemetry proves skew

## Current Custom Kernel

`sl4m.metal_kernels.fused_silu_mul` uses MLX's JIT custom Metal kernel API
and compares against `mx.silu(a) * b`.

Run:

```bash
slam mac-kernel-bench --size 16777216 --dtype float16 --repeats 50
```

The benchmark performs a correctness check before timing. Results can be
stored as JSONL:

```bash
slam mac-kernel-bench --jsonl runs/fused_silu_mul.jsonl
```

`sl4m.metal_kernels.rms_norm` is the first row-wise reduction kernel. It
uses one threadgroup per row and compares against a pure MLX RMSNorm expression:

```bash
slam mac-norm-bench --rows 256 --hidden 4096 --repeats 50 \
  --jsonl runs/rms_norm.jsonl
```

Small shapes are useful for correctness and compiler smoke tests only. Treat
performance claims as meaningful only after running model-representative hidden
sizes and prefill/decode row counts.

## Generation Benchmarks

Run comparable sl4m generation modes:

```bash
slam mac-profile --jsonl runs/mac-profile.jsonl
slam mac-bench mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
  --modes baseline,pld \
  --max-tokens 192 \
  --temperature 0.0 \
  --repeats 3 \
  --jsonl runs/qwen25-coder-mac-bench.jsonl
```

## Real-Model Smoke Tests

Normal tests do not download model weights. To exercise the MLX backend against
a real local or Hugging Face model, opt in explicitly:

```bash
SL4M_REAL_MODEL=mlx-community/Qwen2.5-0.5B-Instruct-4bit pytest -q tests/test_real_models_optional.py
```

Use the smallest model that reproduces the issue before moving to 7B+ models.
For performance runs, prefer:

```bash
slam mac-profile --jsonl runs/mac-profile.jsonl
slam mac-compare mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --max-tokens 32 \
  --temperature 0.0 \
  --repeats 3 \
  --jsonl runs/qwen25-05b-direct-vs-slam.jsonl
slam mac-bench mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --modes baseline,pld \
  --max-tokens 64 \
  --temperature 0.0 \
  --repeats 3 \
  --jsonl runs/qwen25-05b-smoke.jsonl
```

For local larger models, start short and deterministic:

```bash
slam mac-compare ~/Tools/models/Huihui-OmniCoder-9B-abliterated-4bit \
  --prompt "Write a Python function that returns the Fibonacci number for n." \
  --max-tokens 16 \
  --temperature 0.0 \
  --jsonl runs/omnicoder-9b-direct-vs-slam-smoke.jsonl
```

## Agentic Routing

Agentic coding workloads split into two broad classes:

- open generation: use baseline MLX generation
- edit/refactor/fix tasks over pasted code: use PLD

The `agent-run` command applies this conservative route:

```bash
slam agent-run ~/Tools/models/Huihui-OmniCoder-9B-abliterated-4bit \
  --prompt "$(cat tests/test_kernel_harness.py)

Refactor the code above for clarity. Preserve behavior and return only updated code." \
  --max-tokens 256 \
  --log runs/agent-route.jsonl
```

Use `--dry-run` to inspect the decision without loading the model. The route is
recorded in telemetry as `agent_route`.

## Linear Kernel Harness

Before adding a custom linear kernel, run the shape harness:

```bash
slam mac-linear-bench --profile tiny --repeats 20
slam mac-linear-bench --profile qwen2_7b --repeats 10 \
  --jsonl runs/qwen2_7b-linear.jsonl
```

Use explicit shapes when testing a candidate:

```bash
slam mac-linear-bench \
  --shapes 1x4096x4096,256x4096x14336 \
  --backends mlx-matmul,mlx-quantized-roundtrip \
  --jsonl runs/custom-linear.jsonl
```

The current backends are:

- `mlx-matmul`: reference dense matmul, `x @ w.T`
- `mlx-quantized-roundtrip`: MLX quantize/dequantize baseline for quality and
  overhead visibility; random uncalibrated weights are expected to show error

Future candidate kernels should be added as new backends and compared against
the same `LinearShape` records.

## Acceptance Bar

A kernel or model-level optimization is not considered landed until:

- deterministic correctness passes against the reference path
- benchmark metadata is written to JSONL
- a baseline comparison exists in the same run environment
- the README or docs name the workload where it helps and where it regresses

## Cider Takeaways

Cider's useful lesson is hardware-specific gating. INT8 TensorOps are an M5+
opportunity, not a universal Apple Silicon feature. ANE+GPU splitting is a
research path only until it preserves MLX lazy evaluation and avoids private API
risk. See [cider-lessons.md](cider-lessons.md).
