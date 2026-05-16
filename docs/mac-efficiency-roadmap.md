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

The same benchmark now includes `residual_rms_norm`, which fuses residual add
and RMSNorm into one Metal pass. This is a better transformer-block candidate
than standalone RMSNorm because agent decode repeatedly pays memory bandwidth
for residual reads/writes around normalization.

Small shapes are useful for correctness and compiler smoke tests only. Treat
performance claims as meaningful only after running model-representative hidden
sizes and prefill/decode row counts.

## Fine-Tuning Plan

Training should be measured before launched. The first planning layer validates
dataset splits, counts rough token volume, reads local `config.json` when present,
and estimates the effective optimizer schedule:

```bash
slam mlx-train-plan \
  --model ~/Tools/models/Huihui-OmniCoder-9B-abliterated-4bit \
  --data data/my-sft \
  --batch-size 1 \
  --grad-accumulation-steps 4 \
  --num-layers 16 \
  --iters 600
```

Use the output to decide whether a Mac run is worth starting, then launch the
same settings through `slam mlx-train`. Next training work should log tokens/s,
loss, memory peaks, adapter size, and eval perplexity into the same JSONL style
used by inference benchmarks.

Adapter sweeps are now captured by `mlx-adapter-bench`:

```bash
slam mlx-adapter-bench \
  --model ~/Tools/models/Huihui-OmniCoder-9B-abliterated-4bit \
  --data benchmarks/finetune_smoke \
  --fine-tune-types lora,dora \
  --layers 1,2,4 \
  --ranks 4,8,16 \
  --scales 16,20 \
  --learning-rates 1e-5,2e-5 \
  --iters 2 \
  --batch-size 1 \
  --val-batches 1 \
  --test-batches 1 \
  --jsonl runs/adapter-bench-omnicoder-9b-smoke.jsonl
```

Smoke matrix on OmniCoder 9B 4-bit:

| adapter | layers | test loss | tok/s | peak GB | adapter MB |
|---|---:|---:|---:|---:|---:|
| LoRA | 1 | 2.045 | 24.76 | 6.148 | 2.44 |
| LoRA | 2 | 2.025 | 13.52 | 6.549 | 5.07 |
| LoRA | 4 | **1.968** | 18.87 | 6.879 | 10.33 |
| DoRA | 1 | 2.044 | 17.14 | 7.624 | 2.60 |
| DoRA | 2 | 2.028 | 14.73 | 8.989 | 5.41 |
| DoRA | 4 | 1.972 | 13.23 | 11.602 | 11.01 |

This dataset is intentionally too small for quality claims. The actionable
systems result is that DoRA works on Mac/MLX but has a steep unified-memory
cost; LoRA-4 is the current default for larger local sweeps.

Rank/scale sweep, LoRA layer 1:

| rank | scale | test loss | tok/s | peak GB | adapter MB |
|---:|---:|---:|---:|---:|---:|
| 4 | 16 | 2.062 | 21.48 | 6.146 | 1.22 |
| 4 | 20 | 2.061 | 25.34 | 6.146 | 1.22 |
| 8 | 16 | 2.049 | 24.79 | 6.148 | 2.44 |
| 8 | 20 | 2.045 | 25.12 | 6.148 | 2.44 |
| 16 | 16 | 2.043 | 24.52 | 6.152 | 4.88 |
| 16 | 20 | **2.028** | 25.23 | 6.152 | 4.88 |

The short-run result says rank is worth sweeping before layer count on Mac:
rank-16/scale-20 matched LoRA-2 quality with lower adapter size and similar
memory. This is not a final recipe; it is a better next search prior.
The harness now sweeps learning rates in the same run, which is important for
adapter quality because rank/scale choices can look better or worse under a
single under-tuned LR.

## 2026 KV Direction

TurboQuant is relevant for long-context inference, not adapter training. Its
claim is training-free KV-cache compression through random rotation plus
coordinate quantization, with reported quality neutrality around 3.5 bits per
channel and marginal degradation around 2.5 bits per channel. For sl4m, the
right next step is a Mac KV harness:

1. measure MLX FP16/BF16 KV memory and correctness over long-context agent tasks
2. compare MLX built-in `--kv-bits 4/8` against baseline output identity and perplexity
3. prototype TurboQuant-style rotated KV packing/unpacking in MLX arrays
4. only then move the hot path into custom Metal kernels

First smoke run on OmniCoder 9B with a 3.2k-character prompt and 64 generated
tokens:

| mode | tok/s | TTFT | peak GB |
|---|---:|---:|---:|
| FP KV | 5.09 | 8.657s | 6.414 |
| 8-bit KV | 5.06 | 8.669s | 6.414 |
| 4-bit KV | 5.07 | 8.658s | 6.414 |

At this context length, MLX KV quantization does not move memory or speed. We
need longer-context agent prompts before TurboQuant-style work can show a real
memory advantage.

Before promoting any inference fast path, run token-level repeat checks:

```bash
slam determinism-bench ~/Tools/models/Huihui-OmniCoder-9B-abliterated-4bit \
  --modes baseline,pld,adaptive-pld,kv-8bit,kv-4bit \
  --repeats 3 \
  --temperature 0.0 \
  --jsonl runs/determinism-omnicoder-9b-smoke.jsonl
```

The JSONL records include token/text hashes, `matches_first`, and the first
divergent token index for each mode. That gives the agentic path a correctness
gate before speed numbers are trusted.

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
- edit/refactor/fix tasks over pasted code: use adaptive PLD

Adaptive PLD starts with prompt lookup decoding and watches early per-round
telemetry. If the first window has too few prompt matches or too few accepted
draft tokens, it switches to baseline continuation for the remaining token
budget. This keeps the edit-workload PLD win while limiting the short/open task
slowdown we measured under low hit rates.

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

Benchmark full agent tasks with validation:

```bash
slam agent-bench benchmarks/agent_tasks --dry-run
slam agent-bench benchmarks/agent_tasks \
  --model ~/Tools/models/Huihui-OmniCoder-9B-abliterated-4bit \
  --modes baseline,pld,adaptive-pld,agent-auto \
  --jsonl runs/agent-bench.jsonl
```

Task files are TOML and can define context, instructions, expected output type,
token settings, and validators such as `python-ast`, `py-compile`, and
`contains`.

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
