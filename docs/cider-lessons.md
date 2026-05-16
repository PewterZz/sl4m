# Cider Lessons for slam

Cider is useful prior art for sl4m because it targets a specific Apple
Silicon bottleneck instead of treating "Mac acceleration" as one problem.

Sources reviewed:

- <https://github.com/Mininglamp-AI/cider>
- <https://raw.githubusercontent.com/Mininglamp-AI/cider/main/cider/ops.py>
- <https://raw.githubusercontent.com/Mininglamp-AI/cider/main/cider/nn.py>
- <https://raw.githubusercontent.com/Mininglamp-AI/cider/main/experimental/README.md>

## What They Found

1. MLX's built-in quantized linear path is weight-only for common modes; Cider
   adds online activation quantization so W8A8 and W4A8 can use INT8 compute.
2. On M5+, Metal 4 cooperative tensor APIs expose INT8 TensorOps. Cider gates
   those kernels to M5+ and gracefully falls back on M4 and below.
3. Prefill and decode need different strategies:
   - Prefill, `M > 1`: quantize activations, INT8 GEMM, fused dequant store.
   - Decode, `M == 1`: direct INT8 matrix-vector path or fallback.
4. ANE is mostly idle during GPU inference, but Cider's ANE+GPU split remains
   experimental because private ANE APIs and non-lazy integration erase most
   end-to-end benefit.
5. Accuracy is not automatic. W8A8 activation quantization needs calibrated or
   quantization-friendly models: SmoothQuant, GPTQ, QuaRot, QAT, or equivalent.

## Implications for slam

Production-safe now:

- Detect Apple chip generation in `mac-profile`.
- Treat M5+ INT8 TensorOps as a separate capability, not a generic Metal path.
- Benchmark prefill/TTFT separately from decode throughput.
- Keep all acceleration optional with clean fallback on M1-M4.
- Require JSONL benchmark records and deterministic correctness checks.

Research track:

- Add an MLX custom primitive extension layer for linear replacements, separate
  from Python `mx.fast.metal_kernel` microkernels.
- Prototype W8A8 prefill for `nn.Linear` first, then `nn.QuantizedLinear`.
- Add calibration tooling before enabling activation quantized inference by
  default.
- Investigate training only after inference kernels have backward or explicit
  LoRA-only integration. W8A8 inference kernels are not automatically useful for
  fine-tuning.

Do not do by default:

- Do not use private ANE APIs in normal serving/training flows.
- Do not enable activation quantization without a model-quality gate.
- Do not report aggregate tokens/sec without splitting prefill and decode.

## Next Kernel Milestones

1. Keep `fused_silu_mul` as the custom-kernel smoke test.
2. Add shape-aware linear benchmark harness:
   - `M in {1, 16, 64, 256, 1024}`
   - `K/N` from Qwen/Llama hidden sizes
   - report memory, TTFT-equivalent prefill latency, and decode latency
3. Add M5-only extension scaffold with graceful no-op on older Macs.
4. Add quantization quality gate:
   - small perplexity set
   - token identity for deterministic prompts
   - max relative error for isolated linear layers
