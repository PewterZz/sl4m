from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .adaptive_pld import resolve_agent_mode
from .agent import route_agent_prompt
from .agent_bench import (
    load_agent_tasks,
    run_agent_bench,
    validate_modes,
    write_agent_bench_jsonl,
)
from .backend import GenerationSettings, LlamaCppBackend, MLXBackend
from .budget import StaticBudget, Tier, auto_detect_limits
from .kernel_harness import (
    default_linear_shapes,
    kernel_capabilities,
    parse_linear_shapes,
    run_linear_kernel_bench,
)
from .mac_efficiency import (
    build_mac_acceleration_plan,
    collect_mac_profile,
    compare_mlx_lm_vs_slam,
    run_fused_silu_mul_bench,
    run_mlx_generation_bench,
    run_rms_norm_bench,
    summarize_by_best,
    write_jsonl,
)
from .model import ARCH_REGISTRY
from .mlx_workflow import (
    EvalRecipe,
    FuseRecipe,
    ServeRecipe,
    TrainPlan,
    TrainRecipe,
    WorkflowError,
    ensure_macos,
    run_adapter_bench,
    run_command,
    validate_dataset_dir,
)
from .runtime import Session
from .telemetry import JsonlRecorder, NullRecorder

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.command()
def probe(headroom_gb: float = typer.Option(4.0, help="RAM bytes to leave for your other work")):
    """Show detected hardware tiers and what the budget would be."""
    headroom_bytes = int(headroom_gb * 1024**3)
    limits = auto_detect_limits(headroom_bytes=headroom_bytes)
    table = Table(title="Hardware tiers")
    table.add_column("Tier"); table.add_column("Available", justify="right")
    for tier, nbytes in limits.items():
        table.add_row(tier.name, f"{nbytes / 1024**3:.1f} GiB")
    console.print(table)
    console.print(f"[dim]headroom reserved: {headroom_gb} GiB RAM[/dim]")


@app.command()
def run(
    model: str = typer.Argument(..., help="HF repo id, local path, or GGUF file"),
    backend: str = typer.Option("auto", help="mlx | llama_cpp | auto"),
    arch: Optional[str] = typer.Option(None, help=f"Known arch: {sorted(ARCH_REGISTRY)}"),
    prompt: str = typer.Option("Hello, how are you?"),
    max_tokens: int = typer.Option(128),
    temperature: float = typer.Option(0.7),
    kv_bits: Optional[int] = typer.Option(
        None, help="Quantize KV cache to N bits (MLX: 4 or 8). None = off."
    ),
    kv_start: int = typer.Option(0, help="Step to begin quantizing KV (keep first N FP16)."),
    headroom_gb: float = typer.Option(4.0),
    log: Optional[str] = typer.Option(None, help="Path to jsonl telemetry log"),
):
    """Run a model end-to-end through the slam runtime."""
    spec = ARCH_REGISTRY.get(arch) if arch else None

    if backend == "auto":
        backend = "mlx" if sys.platform == "darwin" else "llama_cpp"
    be = MLXBackend() if backend == "mlx" else LlamaCppBackend()

    limits = auto_detect_limits(headroom_bytes=int(headroom_gb * 1024**3))
    budget = StaticBudget(limits)
    recorder = JsonlRecorder(log) if log else NullRecorder()

    console.print(f"[cyan]loading[/cyan] {model} via {backend}")
    be.load(model, spec, budget)

    session = Session(backend=be, budget=budget, spec=spec, recorder=recorder)
    settings = GenerationSettings(
        max_tokens=max_tokens,
        temperature=temperature,
        kv_bits=kv_bits,
        quantized_kv_start=kv_start,
    )

    try:
        for tok in session.generate(prompt, settings):
            console.print(tok.text, end="")
        console.print()
    finally:
        session.close()


@app.command("agent-run")
def agent_run(
    model: str = typer.Argument(..., help="HF repo id or local MLX model path"),
    prompt: str = typer.Option(..., help="Agent task prompt"),
    mode: str = typer.Option("auto", help="auto | baseline | pld | adaptive-pld"),
    max_tokens: int = typer.Option(256),
    temperature: float = typer.Option(0.0),
    num_draft: int = typer.Option(4),
    max_ngram: int = typer.Option(3),
    min_ngram: int = typer.Option(2),
    headroom_gb: float = typer.Option(4.0),
    log: Optional[str] = typer.Option(None, help="Path to jsonl telemetry log"),
    dry_run: bool = typer.Option(False, help="Print routing decision without loading the model"),
):
    """Run an agentic prompt with conservative auto-routing for Mac edit workflows."""
    route = route_agent_prompt(prompt)
    selected_mode = resolve_agent_mode(mode, route.mode)
    if selected_mode not in {"baseline", "pld", "adaptive-pld"}:
        raise typer.BadParameter("mode must be auto, baseline, pld, or adaptive-pld")

    console.print(
        f"[cyan]agent route[/cyan] mode={selected_mode} "
        f"auto={route.mode} confidence={route.confidence:.2f} reason={route.reason}"
    )
    if dry_run:
        return

    limits = auto_detect_limits(headroom_bytes=int(headroom_gb * 1024**3))
    budget = StaticBudget(limits)
    recorder = JsonlRecorder(log) if log else NullRecorder()
    backend = MLXBackend()
    backend.load(model, None, budget)
    session = Session(backend=backend, budget=budget, recorder=recorder)
    settings = GenerationSettings(max_tokens=max_tokens, temperature=temperature)
    recorder.record(
        "agent_route",
        mode=selected_mode,
        auto_mode=route.mode,
        confidence=route.confidence,
        reason=route.reason,
        prompt_chars=route.prompt_chars,
        line_count=route.line_count,
        code_marker_count=route.code_marker_count,
        repeated_line_count=route.repeated_line_count,
    )
    try:
        if selected_mode == "pld":
            stream = session.generate_pld_speculative(
                prompt,
                settings,
                num_draft=num_draft,
                max_ngram_size=max_ngram,
                min_ngram_size=min_ngram,
            )
        elif selected_mode == "adaptive-pld":
            stream = session.generate_adaptive_pld(
                prompt,
                settings,
                num_draft=num_draft,
                max_ngram_size=max_ngram,
                min_ngram_size=min_ngram,
            )
        else:
            stream = session.generate(prompt, settings)
        for tok in stream:
            console.print(tok.text, end="")
        console.print()
    finally:
        session.close()


@app.command("agent-bench")
def agent_bench(
    tasks_path: str = typer.Argument(..., help="TOML task file or directory of task files"),
    model: Optional[str] = typer.Option(None, help="HF repo id or local MLX model path"),
    modes: str = typer.Option("baseline,pld,adaptive-pld,agent-auto"),
    headroom_gb: float = typer.Option(4.0),
    num_draft: int = typer.Option(4),
    max_ngram: int = typer.Option(3),
    min_ngram: int = typer.Option(2),
    jsonl: Optional[str] = typer.Option(None),
    dry_run: bool = typer.Option(False, help="Validate tasks and print routing without loading a model"),
):
    """Benchmark agentic edit tasks across baseline, PLD, adaptive PLD, and auto routing."""
    try:
        tasks = load_agent_tasks(tasks_path)
        selected_modes = validate_modes([m.strip() for m in modes.split(",")])
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e

    if dry_run:
        table = Table(title="Agent benchmark tasks")
        table.add_column("task")
        table.add_column("prompt chars", justify="right")
        table.add_column("auto")
        table.add_column("confidence", justify="right")
        table.add_column("validations")
        for task in tasks:
            route = route_agent_prompt(task.prompt)
            table.add_row(
                task.name,
                str(len(task.prompt)),
                route.mode,
                f"{route.confidence:.2f}",
                ",".join(v.kind for v in task.validations) or "none",
            )
        console.print(table)
        return

    if not model:
        raise typer.BadParameter("--model is required unless --dry-run is set")

    results = run_agent_bench(
        tasks=tasks,
        model=model,
        modes=selected_modes,
        headroom_gb=headroom_gb,
        num_draft=num_draft,
        max_ngram=max_ngram,
        min_ngram=min_ngram,
    )
    table = Table(title="Agent benchmark")
    for col in ("task", "mode", "selected", "tokens", "tok/s", "valid", "peak GB"):
        table.add_column(col, justify="right" if col in {"tokens", "tok/s", "peak GB"} else "left")
    for result in results:
        table.add_row(
            result.task,
            result.mode,
            result.selected_mode,
            str(result.tokens),
            f"{result.tok_s:.2f}",
            "yes" if result.validation_ok else "no",
            f"{result.peak_mem_gb:.2f}" if result.peak_mem_gb is not None else "n/a",
        )
    console.print(table)
    if jsonl:
        write_agent_bench_jsonl(jsonl, results)


@app.command("mlx-check-data")
def mlx_check_data(
    data: str = typer.Argument(..., help="Directory with train/valid/test JSONL files"),
    require_test: bool = typer.Option(False, help="Require test.jsonl instead of train.jsonl"),
):
    """Validate a dataset directory before MLX fine-tuning/eval."""
    try:
        report = validate_dataset_dir(data, require=("test.jsonl",) if require_test else ("train.jsonl",))
    except WorkflowError as e:
        raise typer.BadParameter(str(e)) from e

    table = Table(title=f"Dataset: {report.path}")
    table.add_column("File")
    table.add_column("Format")
    table.add_column("Examples", justify="right")
    for name in report.files:
        table.add_row(name, report.formats[name], str(report.examples[name]))
    console.print(table)


@app.command("mlx-train")
def mlx_train(
    model: Optional[str] = typer.Option(None, help="HF repo id or local MLX model path"),
    data: Optional[str] = typer.Option(None, help="Dataset directory with train.jsonl"),
    adapter_path: str = typer.Option("adapters", help="Where adapter weights are written"),
    fine_tune_type: str = typer.Option("lora", help="lora | dora | full"),
    iters: int = typer.Option(600),
    batch_size: int = typer.Option(1),
    learning_rate: float = typer.Option(1e-5),
    num_layers: int = typer.Option(16, help="-1 means all layers"),
    grad_accumulation_steps: int = typer.Option(1),
    grad_checkpoint: bool = typer.Option(True, help="Trade compute for lower memory"),
    mask_prompt: bool = typer.Option(True, help="Train loss on completions only for chat/completion data"),
    recipe: Optional[str] = typer.Option(None, help="TOML recipe with a [train] table"),
    dry_run: bool = typer.Option(False, help="Print the MLX-LM LoRA command without running it"),
):
    """Mac-first LoRA/QLoRA/DoRA fine-tuning via MLX-LM."""
    try:
        if not dry_run:
            ensure_macos()
        if recipe:
            cfg = TrainRecipe.from_toml(recipe)
        else:
            if not model or not data:
                raise WorkflowError("provide --model and --data, or --recipe")
            cfg = TrainRecipe(
                model=model,
                data=Path(data),
                adapter_path=Path(adapter_path),
                fine_tune_type=fine_tune_type,
                iters=iters,
                batch_size=batch_size,
                learning_rate=learning_rate,
                num_layers=num_layers,
                grad_accumulation_steps=grad_accumulation_steps,
                grad_checkpoint=grad_checkpoint,
                mask_prompt=mask_prompt,
            )
        report = cfg.validate()
    except WorkflowError as e:
        raise typer.BadParameter(str(e)) from e

    console.print(
        f"[cyan]mlx train[/cyan] model={cfg.model} data={report.path} "
        f"adapter={cfg.adapter_path} type={cfg.fine_tune_type}"
    )
    raise typer.Exit(run_command(cfg.to_command(), dry_run=dry_run))


def _render_train_plan(plan: TrainPlan) -> None:
    table = Table(title="MLX training plan")
    table.add_column("Field")
    table.add_column("Value")
    cfg = plan.model_config
    table.add_row("model", plan.recipe.model)
    table.add_row("model_type", cfg.model_type or "unknown")
    table.add_row("hidden", str(cfg.hidden_size or "unknown"))
    table.add_row("layers", str(cfg.num_hidden_layers or "unknown"))
    table.add_row("heads", str(cfg.num_attention_heads or "unknown"))
    table.add_row("quant_bits", str(cfg.quantization_bits or "unknown"))
    table.add_row("data", str(plan.dataset.path))
    table.add_row("train_examples", str(plan.dataset.train_examples))
    table.add_row("rough_train_tokens", str(plan.dataset.rough_train_tokens))
    table.add_row("effective_batch", str(plan.effective_batch_size))
    table.add_row("steps_per_epoch", str(plan.optimizer_steps_per_epoch))
    table.add_row("iters", str(plan.recipe.iters))
    table.add_row("estimated_epochs", f"{plan.estimated_epochs:.2f}")
    table.add_row("trainable_layers", str(plan.trainable_layer_count or "unknown"))
    table.add_row("rough_tokens_per_step", str(plan.rough_tokens_per_step))
    console.print(table)


@app.command("mlx-train-plan")
def mlx_train_plan(
    model: Optional[str] = typer.Option(None, help="HF repo id or local MLX model path"),
    data: Optional[str] = typer.Option(None, help="Dataset directory with train.jsonl"),
    adapter_path: str = typer.Option("adapters", help="Where adapter weights are written"),
    fine_tune_type: str = typer.Option("lora", help="lora | dora | full"),
    iters: int = typer.Option(600),
    batch_size: int = typer.Option(1),
    learning_rate: float = typer.Option(1e-5),
    num_layers: int = typer.Option(16, help="-1 means all layers"),
    grad_accumulation_steps: int = typer.Option(1),
    grad_checkpoint: bool = typer.Option(True),
    mask_prompt: bool = typer.Option(True),
    recipe: Optional[str] = typer.Option(None, help="TOML recipe with a [train] table"),
):
    """Estimate Mac MLX fine-tuning shape before launching a long run."""
    try:
        if recipe:
            cfg = TrainRecipe.from_toml(recipe)
        else:
            if not model or not data:
                raise WorkflowError("provide --model and --data, or --recipe")
            cfg = TrainRecipe(
                model=model,
                data=Path(data),
                adapter_path=Path(adapter_path),
                fine_tune_type=fine_tune_type,
                iters=iters,
                batch_size=batch_size,
                learning_rate=learning_rate,
                num_layers=num_layers,
                grad_accumulation_steps=grad_accumulation_steps,
                grad_checkpoint=grad_checkpoint,
                mask_prompt=mask_prompt,
            )
        _render_train_plan(cfg.plan())
    except WorkflowError as e:
        raise typer.BadParameter(str(e)) from e


@app.command("mlx-eval")
def mlx_eval(
    model: str = typer.Option(..., help="HF repo id or local MLX model path"),
    data: str = typer.Option(..., help="Dataset directory with test.jsonl"),
    adapter_path: str = typer.Option("adapters"),
    batch_size: int = typer.Option(4),
    test_batches: int = typer.Option(-1),
    dry_run: bool = typer.Option(False),
):
    """Evaluate adapter perplexity through MLX-LM."""
    try:
        if not dry_run:
            ensure_macos()
        cfg = EvalRecipe(
            model=model,
            data=Path(data),
            adapter_path=Path(adapter_path),
            batch_size=batch_size,
            test_batches=test_batches,
        )
        cfg.validate()
    except WorkflowError as e:
        raise typer.BadParameter(str(e)) from e
    raise typer.Exit(run_command(cfg.to_command(), dry_run=dry_run))


@app.command("mlx-adapter-bench")
def mlx_adapter_bench(
    model: str = typer.Option(..., help="HF repo id or local MLX model path"),
    data: str = typer.Option(..., help="Dataset directory with train/valid/test JSONL"),
    output_dir: str = typer.Option("runs/adapter-bench"),
    fine_tune_types: str = typer.Option("lora,dora", help="Comma-separated: lora,dora,full"),
    layers: str = typer.Option("1,2", help="Comma-separated layer counts, e.g. 1,2,4"),
    iters: int = typer.Option(2),
    batch_size: int = typer.Option(1),
    grad_accumulation_steps: int = typer.Option(1),
    learning_rate: float = typer.Option(1e-5),
    optimizer: Optional[str] = typer.Option(None),
    val_batches: int = typer.Option(1),
    test_batches: int = typer.Option(1),
    max_seq_length: Optional[int] = typer.Option(None),
    seed: Optional[int] = typer.Option(0),
    jsonl: Optional[str] = typer.Option(None, help="Write benchmark records as JSONL"),
):
    """Run a small LoRA/DoRA adapter matrix and log train/eval metrics."""
    try:
        ensure_macos()
        selected_types = [x.strip() for x in fine_tune_types.split(",") if x.strip()]
        selected_layers = [int(x.strip()) for x in layers.split(",") if x.strip()]
        results = run_adapter_bench(
            model=model,
            data=Path(data),
            output_dir=Path(output_dir),
            fine_tune_types=selected_types,
            num_layers=selected_layers,
            iters=iters,
            batch_size=batch_size,
            grad_accumulation_steps=grad_accumulation_steps,
            learning_rate=learning_rate,
            optimizer=optimizer,
            val_batches=val_batches,
            test_batches=test_batches,
            max_seq_length=max_seq_length,
            seed=seed,
        )
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    except WorkflowError as e:
        raise typer.BadParameter(str(e)) from e

    table = Table(title="MLX adapter benchmark")
    for col in (
        "name",
        "rc",
        "eval",
        "train loss",
        "val loss",
        "test loss",
        "tok/s",
        "peak GB",
        "adapter MB",
    ):
        table.add_column(col, justify="right" if col != "name" else "left")
    for result in results:
        table.add_row(
            result.name,
            str(result.train_returncode),
            "n/a" if result.eval_returncode is None else str(result.eval_returncode),
            "n/a" if result.train_loss is None else f"{result.train_loss:.3f}",
            "n/a" if result.val_loss is None else f"{result.val_loss:.3f}",
            "n/a" if result.test_loss is None else f"{result.test_loss:.3f}",
            "n/a" if result.tokens_per_sec is None else f"{result.tokens_per_sec:.2f}",
            "n/a" if result.peak_mem_gb is None else f"{result.peak_mem_gb:.2f}",
            "n/a" if result.adapter_size_mb is None else f"{result.adapter_size_mb:.2f}",
        )
    console.print(table)
    if jsonl:
        write_jsonl(jsonl, [r.to_record() for r in results])


@app.command("mlx-fuse")
def mlx_fuse(
    model: str = typer.Option(..., help="Base model used for training"),
    adapter_path: str = typer.Option("adapters"),
    save_path: str = typer.Option("fused_model"),
    upload_repo: Optional[str] = typer.Option(None),
    dry_run: bool = typer.Option(False),
):
    """Fuse adapters into a standalone MLX model."""
    try:
        if not dry_run:
            ensure_macos()
        cfg = FuseRecipe(
            model=model,
            adapter_path=Path(adapter_path),
            save_path=Path(save_path),
            upload_repo=upload_repo,
        )
    except WorkflowError as e:
        raise typer.BadParameter(str(e)) from e
    raise typer.Exit(run_command(cfg.to_command(), dry_run=dry_run))


@app.command("mlx-serve")
def mlx_serve(
    model: str = typer.Option(..., help="HF repo id or local MLX model path"),
    adapter_path: Optional[str] = typer.Option(None, help="Adapter path to load for requests"),
    draft_model: Optional[str] = typer.Option(None, help="Optional draft model for mlx_lm.server speculation"),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8080),
    dry_run: bool = typer.Option(False),
):
    """Start MLX-LM's OpenAI-compatible local server."""
    try:
        if not dry_run:
            ensure_macos()
        cfg = ServeRecipe(
            model=model,
            adapter_path=Path(adapter_path) if adapter_path else None,
            draft_model=draft_model,
            host=host,
            port=port,
        )
    except WorkflowError as e:
        raise typer.BadParameter(str(e)) from e
    console.print(f"[cyan]serving[/cyan] http://{host}:{port}/v1/chat/completions")
    raise typer.Exit(run_command(cfg.to_command(), dry_run=dry_run))


@app.command("mac-profile")
def mac_profile(
    jsonl: Optional[str] = typer.Option(None, help="Write profile as one JSONL record"),
):
    """Capture Mac hardware, Python, and MLX runtime metadata."""
    profile = collect_mac_profile()
    table = Table(title="Mac profile")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("system", profile.system)
    table.add_row("release", profile.release)
    table.add_row("machine", profile.machine)
    table.add_row("cpu", profile.cpu)
    table.add_row("ram_gb", f"{profile.ram_gb:.2f}")
    table.add_row("python", profile.python)
    table.add_row(
        "apple_chip_generation",
        f"M{profile.apple_chip_generation}" if profile.apple_chip_generation else "unknown",
    )
    table.add_row("mlx_version", profile.mlx_version or "not installed")
    table.add_row("mlx_device", str(profile.mlx_device or {}))
    console.print(table)

    plan = build_mac_acceleration_plan(profile.apple_chip_generation)
    plan_table = Table(title="Acceleration plan")
    plan_table.add_column("Path")
    plan_table.add_column("Status")
    plan_table.add_column("Applies to")
    plan_table.add_column("Guardrail")
    for opp in plan.opportunities:
        plan_table.add_row(opp.name, opp.status, opp.applies_to, opp.guardrail)
    console.print(plan_table)
    if jsonl:
        write_jsonl(jsonl, [profile, plan])


@app.command("mac-bench")
def mac_bench(
    model: str = typer.Argument(..., help="HF repo id or local MLX model path"),
    prompt: str = typer.Option(
        "Write a Python function that validates and normalizes a nested JSON config.",
    ),
    modes: str = typer.Option("baseline,pld", help="Comma-separated: baseline,pld"),
    max_tokens: int = typer.Option(128),
    temperature: float = typer.Option(0.0),
    repeats: int = typer.Option(1),
    num_draft: int = typer.Option(4),
    max_ngram: int = typer.Option(3),
    min_ngram: int = typer.Option(2),
    headroom_gb: float = typer.Option(4.0),
    jsonl: Optional[str] = typer.Option(None, help="Write raw benchmark results"),
):
    """Benchmark slam Mac generation modes with comparable settings."""
    selected = [m.strip() for m in modes.split(",") if m.strip()]
    results = run_mlx_generation_bench(
        model=model,
        prompt=prompt,
        modes=selected,
        max_tokens=max_tokens,
        temperature=temperature,
        repeats=repeats,
        headroom_gb=headroom_gb,
        num_draft=num_draft,
        max_ngram=max_ngram,
        min_ngram=min_ngram,
    )
    table = Table(title="Mac generation benchmark")
    for col in ("mode", "tokens", "seconds", "tok/s", "ttft", "peak GB"):
        table.add_column(col, justify="right" if col != "mode" else "left")
    for r in results:
        table.add_row(
            r.name,
            str(r.tokens),
            f"{r.seconds:.2f}",
            f"{r.tok_s:.2f}",
            f"{(r.first_token_s or 0.0):.3f}",
            f"{r.peak_mem_gb:.2f}" if r.peak_mem_gb is not None else "n/a",
        )
    console.print(table)
    best = summarize_by_best(results)
    if len(best) > 1 and best[0].tok_s > 0:
        baseline = next((r for r in best if r.name == "baseline"), best[0])
        for r in best:
            if r.name != baseline.name and baseline.tok_s > 0:
                console.print(f"[green]{r.name} vs {baseline.name}: {r.tok_s / baseline.tok_s:.2f}×[/green]")
    if jsonl:
        write_jsonl(jsonl, results)


@app.command("mac-compare")
def mac_compare(
    model: str = typer.Argument(..., help="HF repo id or local MLX model path"),
    prompt: str = typer.Option("Return exactly: ready"),
    max_tokens: int = typer.Option(16),
    temperature: float = typer.Option(0.0),
    top_p: float = typer.Option(0.95),
    repeats: int = typer.Option(1),
    headroom_gb: float = typer.Option(4.0),
    jsonl: Optional[str] = typer.Option(None),
):
    """Compare MLX-LM direct generation against slam baseline generation."""
    results = compare_mlx_lm_vs_slam(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repeats=repeats,
        headroom_gb=headroom_gb,
    )
    table = Table(title="MLX-LM direct vs slam")
    for col in ("path", "tokens", "seconds", "tok/s", "ttft", "peak GB", "identity"):
        table.add_column(col, justify="right" if col != "path" else "left")
    for r in results:
        table.add_row(
            r.name,
            str(r.tokens),
            f"{r.seconds:.2f}",
            f"{r.tok_s:.2f}",
            f"{(r.first_token_s or 0.0):.3f}",
            f"{r.peak_mem_gb:.2f}" if r.peak_mem_gb is not None else "n/a",
            "yes" if r.metadata.get("token_identity") else "no",
        )
    console.print(table)
    for i in range(0, len(results), 2):
        direct = results[i]
        slam_result = results[i + 1] if i + 1 < len(results) else None
        if slam_result and direct.tok_s > 0:
            console.print(
                f"[green]repeat {direct.metadata.get('repeat')}: "
                f"slam/direct={slam_result.tok_s / direct.tok_s:.2f}×[/green]"
            )
    if jsonl:
        write_jsonl(jsonl, [r.to_record() for r in results])


@app.command("mac-kernel-bench")
def mac_kernel_bench(
    size: int = typer.Option(16_777_216, help="Number of elements"),
    dtype: str = typer.Option("float16", help="float16 | float32"),
    repeats: int = typer.Option(50),
    jsonl: Optional[str] = typer.Option(None),
):
    """Benchmark custom Metal kernels against MLX reference ops."""
    results = run_fused_silu_mul_bench(size=size, dtype=dtype, repeats=repeats)
    table = Table(title="Mac Metal kernel benchmark")
    table.add_column("kernel")
    table.add_column("seconds", justify="right")
    table.add_column("elem/s", justify="right")
    table.add_column("peak GB", justify="right")
    for r in results:
        table.add_row(
            r.name,
            f"{r.seconds:.4f}",
            f"{r.tok_s:.0f}",
            f"{r.peak_mem_gb:.2f}" if r.peak_mem_gb is not None else "n/a",
        )
    console.print(table)
    ref = next((r for r in results if r.name.startswith("mlx-reference")), None)
    custom = next((r for r in results if r.name.startswith("slam-metal")), None)
    if ref and custom and ref.tok_s > 0:
        console.print(f"[green]custom/ref: {custom.tok_s / ref.tok_s:.2f}×[/green]")
    if jsonl:
        write_jsonl(jsonl, results)


@app.command("mac-norm-bench")
def mac_norm_bench(
    rows: int = typer.Option(256, help="Rows/tokens in the benchmark input"),
    hidden: int = typer.Option(4096, help="Hidden dimension"),
    dtype: str = typer.Option("float16", help="float16 | float32"),
    eps: float = typer.Option(1e-6),
    repeats: int = typer.Option(50),
    jsonl: Optional[str] = typer.Option(None),
):
    """Benchmark the custom Metal RMSNorm kernel against MLX reference."""
    results = run_rms_norm_bench(
        rows=rows,
        hidden=hidden,
        dtype=dtype,
        eps=eps,
        repeats=repeats,
    )
    table = Table(title="Mac RMSNorm kernel benchmark")
    table.add_column("kernel")
    table.add_column("shape")
    table.add_column("seconds", justify="right")
    table.add_column("elem/s", justify="right")
    table.add_column("peak GB", justify="right")
    for r in results:
        table.add_row(
            r.name,
            f"{rows}x{hidden}",
            f"{r.seconds:.4f}",
            f"{r.tok_s:.0f}",
            f"{r.peak_mem_gb:.2f}" if r.peak_mem_gb is not None else "n/a",
        )
    console.print(table)
    ref = next((r for r in results if r.name.startswith("mlx-reference")), None)
    custom = next((r for r in results if r.name.startswith("slam-metal")), None)
    if ref and custom and ref.tok_s > 0:
        console.print(f"[green]custom/ref: {custom.tok_s / ref.tok_s:.2f}×[/green]")
    if jsonl:
        write_jsonl(jsonl, results)


@app.command("mac-linear-bench")
def mac_linear_bench(
    profile: str = typer.Option(
        "tiny",
        help="Shape profile: tiny | qwen2_7b | llama3_8b",
    ),
    shapes: Optional[str] = typer.Option(
        None,
        help="Override shapes as comma-separated MxKxN, e.g. 1x4096x4096,256x4096x14336",
    ),
    backends: str = typer.Option(
        "mlx-matmul,mlx-quantized-roundtrip",
        help="Comma-separated: mlx-matmul,mlx-quantized-roundtrip",
    ),
    dtype: str = typer.Option("float16", help="float16 | float32 | bfloat16"),
    repeats: int = typer.Option(20),
    seed: int = typer.Option(0),
    quant_group_size: int = typer.Option(64),
    quant_bits: int = typer.Option(4),
    jsonl: Optional[str] = typer.Option(None),
):
    """Run the Mac linear-kernel research harness on transformer shapes."""
    try:
        selected_shapes = parse_linear_shapes(shapes) if shapes else default_linear_shapes(profile)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    selected_backends = [b.strip() for b in backends.split(",") if b.strip()]

    cap_table = Table(title="Kernel capabilities")
    cap_table.add_column("Capability")
    cap_table.add_column("Available")
    cap_table.add_column("Reason")
    for cap in kernel_capabilities():
        cap_table.add_row(cap.name, "yes" if cap.available else "no", cap.reason)
    console.print(cap_table)

    results = run_linear_kernel_bench(
        shapes=selected_shapes,
        backends=selected_backends,
        dtype=dtype,
        repeats=repeats,
        seed=seed,
        quant_group_size=quant_group_size,
        quant_bits=quant_bits,
    )

    table = Table(title="Linear kernel benchmark")
    for col in (
        "shape",
        "kind",
        "backend",
        "dtype",
        "ms",
        "TFLOP/s",
        "peak GB",
        "allclose",
        "max err",
    ):
        table.add_column(col, justify="right" if col not in {"shape", "backend", "kind", "dtype"} else "left")
    for r in results:
        corr = r.correctness
        table.add_row(
            f"{r.shape.m}x{r.shape.k}x{r.shape.n}",
            r.shape.kind,
            r.backend,
            r.dtype,
            f"{r.matmul_s * 1000:.3f}",
            f"{r.tflops:.2f}",
            f"{r.peak_mem_gb:.2f}" if r.peak_mem_gb is not None else "n/a",
            "ref" if corr is None else ("yes" if corr.allclose else "no"),
            "ref" if corr is None else f"{corr.max_abs_error:.3g}",
        )
    console.print(table)

    if jsonl:
        write_jsonl(jsonl, [r.to_record() for r in results])


@app.command()
def bench(
    model: str = typer.Argument(..., help="HF repo id, local path, or GGUF file"),
    backend: str = typer.Option("auto", help="mlx | llama_cpp | auto"),
    system_prompt: str = typer.Option(
        "You are an expert Python programmer. Respond only with idiomatic, well-typed code. "
        * 20,
        help="Long shared prefix. Dominates prefill time.",
    ),
    user_a: str = typer.Option("Write a bubble sort.", help="First user turn"),
    user_b: str = typer.Option("Write a merge sort.", help="Second user turn"),
    max_tokens: int = typer.Option(40),
    temperature: float = typer.Option(0.7),
    headroom_gb: float = typer.Option(4.0),
    log: Optional[str] = typer.Option(None, help="Path to jsonl telemetry log"),
):
    """
    Benchmark prompt-cache reuse.

    Runs the SAME conversation twice:
    - Cold: fresh cache, full prompt prefilled every turn.
    - Warm: shared cache, turn 2 submits only the new user tokens.

    Reports TTFT delta — how much prefill you save on turn 2 by keeping state.
    """
    if backend == "auto":
        backend = "mlx" if sys.platform == "darwin" else "llama_cpp"
    be = MLXBackend() if backend == "mlx" else LlamaCppBackend()

    limits = auto_detect_limits(headroom_bytes=int(headroom_gb * 1024**3))
    budget = StaticBudget(limits)
    recorder = JsonlRecorder(log) if log else NullRecorder()

    console.print(f"[cyan]loading[/cyan] {model} via {backend}")
    be.load(model, None, budget)
    session = Session(backend=be, budget=budget, recorder=recorder)
    settings = GenerationSettings(max_tokens=max_tokens, temperature=temperature)

    def turn(label: str, prompt: str, cache):
        t0 = time.monotonic()
        ttft = None
        out_parts = []
        count = 0
        for tok in session.generate(prompt, settings, cache=cache):
            if ttft is None:
                ttft = time.monotonic() - t0
            out_parts.append(tok.text)
            count += 1
        elapsed = time.monotonic() - t0
        decode_tps = (count - 1) / (elapsed - ttft) if count > 1 and ttft else 0.0
        out = "".join(out_parts)
        console.print(f"[bold cyan]{label}[/bold cyan] ttft={ttft*1000:.0f}ms  total={elapsed:.2f}s  decode={decode_tps:.1f} t/s  tokens={count}")
        console.print(f"  [dim]{out[:120]}...[/dim]" if len(out) > 120 else f"  [dim]{out}[/dim]")
        return ttft or 0.0, out

    try:
        # --- Cold path: no cache reuse, each turn sends full prompt ---
        console.print("\n[bold]Cold (no cache reuse)[/bold]")
        ttft_c1, out_c1 = turn("t1 (full prompt)  ", system_prompt + "\nUser: " + user_a + "\nAssistant:", None)
        ttft_c2, _ = turn("t2 (full prompt)  ", system_prompt + "\nUser: " + user_a + "\nAssistant: " + out_c1 + "\nUser: " + user_b + "\nAssistant:", None)

        # --- Warm path: shared cache, t2 submits only the new tokens ---
        console.print("\n[bold]Warm (shared cache)[/bold]")
        cache = session.new_cache()
        if cache is None:
            console.print("[yellow]backend does not support prompt cache — skipping warm path[/yellow]")
            return
        ttft_w1, _ = turn("t1 (full prompt)  ", system_prompt + "\nUser: " + user_a + "\nAssistant:", cache)
        # Generated tokens from t1 are already in the cache. t2 submits only new user turn.
        ttft_w2, _ = turn("t2 (delta only)   ", "\nUser: " + user_b + "\nAssistant:", cache)

        console.print()
        console.print(f"[green]T2 TTFT  cold={ttft_c2*1000:.0f}ms  warm={ttft_w2*1000:.0f}ms  savings={(ttft_c2-ttft_w2)*1000:.0f}ms  speedup={(ttft_c2/ttft_w2 if ttft_w2 else float('inf')):.1f}×[/green]")
    finally:
        session.close()


@app.command()
def spec(
    model: str = typer.Argument(..., help="HF repo id or local path (MLX)"),
    draft: str = typer.Option(..., help="Draft model for speculation"),
    prompt: str = typer.Option("Write a Python bubble sort with type hints."),
    num_draft: int = typer.Option(2, help="Draft tokens per verify round"),
    max_tokens: int = typer.Option(96),
    temperature: float = typer.Option(0.0, help="0.0 for greedy (correctness-comparable)"),
    compare_baseline: bool = typer.Option(
        True, help="Also run non-speculative baseline and report speedup"
    ),
    headroom_gb: float = typer.Option(4.0),
    log: Optional[str] = typer.Option(None, help="Path to jsonl telemetry log"),
):
    """Speculative decoding with per-round telemetry.

    MLX-only. Handles hybrid-attention models (Qwen3.5 family) via sl4m's
    own snapshot/restore of ArraysCache + KVCache state.
    """
    be = MLXBackend()
    limits = auto_detect_limits(headroom_bytes=int(headroom_gb * 1024**3))
    budget = StaticBudget(limits)
    recorder = JsonlRecorder(log) if log else NullRecorder()

    console.print(f"[cyan]loading[/cyan] target={model}  draft={draft}")
    be.load(model, None, budget)
    be.load_draft(draft)

    session = Session(backend=be, budget=budget, recorder=recorder)
    settings = GenerationSettings(max_tokens=max_tokens, temperature=temperature)

    try:
        if compare_baseline:
            t0 = time.monotonic()
            base_tokens: list[int] = []
            base_text = []
            for tok in session.generate(prompt, settings):
                base_tokens.append(tok.token_id)
                base_text.append(tok.text)
            t_base = time.monotonic() - t0
            n_base = len(base_tokens)
            console.print(
                f"[bold]baseline[/bold] tokens={n_base} time={t_base:.2f}s "
                f"tps={n_base / t_base:.1f}"
            )

        t0 = time.monotonic()
        spec_tokens: list[int] = []
        spec_text = []
        accepted = 0
        n = 0
        for tok in session.generate_speculative(
            prompt, settings, num_draft=num_draft
        ):
            spec_tokens.append(tok.token_id)
            spec_text.append(tok.text)
            if tok.from_draft:
                accepted += 1
            if tok.token_id != -1:
                n += 1
        t_spec = time.monotonic() - t0
        tps = n / t_spec if t_spec > 0 else 0.0
        accept_rate = (accepted / n) * 100 if n else 0.0
        console.print(
            f"[bold]spec[/bold]     tokens={n} time={t_spec:.2f}s "
            f"tps={tps:.1f}  from_draft={accept_rate:.1f}%"
        )

        if compare_baseline:
            speedup = t_base / t_spec if t_spec > 0 else 0.0
            console.print(f"[green]speedup {speedup:.2f}×[/green]")
            if temperature == 0.0:
                m = min(len(base_tokens), len(spec_tokens))
                diverge = next((i for i in range(m) if base_tokens[i] != spec_tokens[i]), None)
                if diverge is None and len(base_tokens) == len(spec_tokens):
                    console.print(f"[green]correctness: {m} tokens identical[/green]")
                elif diverge is None:
                    console.print(
                        f"[yellow]correctness: prefix {m} match, lengths differ "
                        f"base={len(base_tokens)} spec={len(spec_tokens)}[/yellow]"
                    )
                else:
                    console.print(
                        f"[red]correctness FAIL at idx {diverge}: "
                        f"base={base_tokens[diverge]} spec={spec_tokens[diverge]}[/red]"
                    )

        console.print(f"\n[dim]{''.join(spec_text)[:200]}[/dim]")
    finally:
        session.close()


@app.command("spec-sweep")
def spec_sweep(
    model: str = typer.Argument(..., help="HF repo id or local path (MLX)"),
    draft: str = typer.Option(..., help="Draft model for speculation"),
    prompt: str = typer.Option(
        "Write a Python function that takes a list of integers and returns "
        "the median. Handle even-length lists by averaging the two middle "
        "elements. Include a docstring and type hints."
    ),
    num_drafts: str = typer.Option("1,2,4,6,8", help="Comma-separated num_draft values to sweep"),
    max_tokens: int = typer.Option(96),
    temperature: float = typer.Option(0.0),
    headroom_gb: float = typer.Option(4.0),
):
    """Sweep num_draft values, report which maximizes speedup for this model pair.

    Loads models once, reuses across runs. Prints a table keyed by num_draft.
    """
    be = MLXBackend()
    limits = auto_detect_limits(headroom_bytes=int(headroom_gb * 1024**3))
    budget = StaticBudget(limits)

    console.print(f"[cyan]loading[/cyan] target={model}  draft={draft}")
    be.load(model, None, budget)
    be.load_draft(draft)
    session = Session(backend=be, budget=budget, recorder=NullRecorder())
    settings = GenerationSettings(max_tokens=max_tokens, temperature=temperature)

    try:
        t0 = time.monotonic()
        n_base = 0
        for _ in session.generate(prompt, settings):
            n_base += 1
        t_base = time.monotonic() - t0
        tps_base = n_base / t_base if t_base > 0 else 0.0
        console.print(f"[bold]baseline[/bold] tokens={n_base} time={t_base:.2f}s tps={tps_base:.1f}")

        table = Table(title="num_draft sweep")
        table.add_column("n_draft", justify="right")
        table.add_column("tps", justify="right")
        table.add_column("speedup", justify="right")
        table.add_column("from_draft", justify="right")

        values = [int(x) for x in num_drafts.split(",")]
        for n in values:
            t0 = time.monotonic()
            n_tok = 0
            accepted = 0
            for tok in session.generate_speculative(prompt, settings, num_draft=n):
                if tok.token_id == -1:
                    continue
                n_tok += 1
                if tok.from_draft:
                    accepted += 1
            dt = time.monotonic() - t0
            tps = n_tok / dt if dt > 0 else 0.0
            speedup = tps / tps_base if tps_base > 0 else 0.0
            rate = (accepted / n_tok * 100) if n_tok else 0.0
            table.add_row(str(n), f"{tps:.1f}", f"{speedup:.2f}×", f"{rate:.1f}%")
        console.print(table)
    finally:
        session.close()


@app.command()
def pld(
    model: str = typer.Argument(..., help="HF repo id or local path (MLX)"),
    prompt: str = typer.Option(
        "Write a Python function that takes a list of integers and returns "
        "the median. Handle even-length lists by averaging the two middle "
        "elements. Include a docstring and type hints."
    ),
    num_draft: int = typer.Option(4, help="Max draft tokens per verify round (variable)"),
    max_ngram: int = typer.Option(3, help="Max n-gram size to match against history"),
    min_ngram: int = typer.Option(2, help="Min n-gram size before giving up"),
    max_tokens: int = typer.Option(128),
    temperature: float = typer.Option(0.0, help="0.0 enables correctness gate vs baseline"),
    compare_baseline: bool = typer.Option(
        True, help="Also run non-speculative baseline and report speedup"
    ),
    headroom_gb: float = typer.Option(4.0),
    log: Optional[str] = typer.Option(None, help="Path to jsonl telemetry log"),
):
    """Prompt Lookup Decoding — draftless speculation via n-gram match.

    MLX-only. No draft model. Drafts come from searching the prompt + what's
    been generated for the last n-gram's earlier occurrence and taking what
    followed. Fast path is code / refactor workloads with context-local repeats.
    """
    be = MLXBackend()
    limits = auto_detect_limits(headroom_bytes=int(headroom_gb * 1024**3))
    budget = StaticBudget(limits)
    recorder = JsonlRecorder(log) if log else NullRecorder()

    console.print(f"[cyan]loading[/cyan] {model}")
    be.load(model, None, budget)

    session = Session(backend=be, budget=budget, recorder=recorder)
    settings = GenerationSettings(max_tokens=max_tokens, temperature=temperature)

    try:
        if compare_baseline:
            t0 = time.monotonic()
            base_tokens: list[int] = []
            for tok in session.generate(prompt, settings):
                base_tokens.append(tok.token_id)
            t_base = time.monotonic() - t0
            n_base = len(base_tokens)
            console.print(
                f"[bold]baseline[/bold] tokens={n_base} time={t_base:.2f}s "
                f"tps={n_base / t_base:.1f}"
            )

        t0 = time.monotonic()
        pld_tokens: list[int] = []
        pld_text: list[str] = []
        accepted = 0
        n = 0
        for tok in session.generate_pld_speculative(
            prompt,
            settings,
            num_draft=num_draft,
            max_ngram_size=max_ngram,
            min_ngram_size=min_ngram,
        ):
            pld_tokens.append(tok.token_id)
            pld_text.append(tok.text)
            if tok.from_draft:
                accepted += 1
            if tok.token_id != -1:
                n += 1
        t_pld = time.monotonic() - t0
        tps = n / t_pld if t_pld > 0 else 0.0
        accept_rate = (accepted / n) * 100 if n else 0.0
        console.print(
            f"[bold]pld[/bold]      tokens={n} time={t_pld:.2f}s "
            f"tps={tps:.1f}  from_draft={accept_rate:.1f}%"
        )

        if compare_baseline:
            speedup = t_base / t_pld if t_pld > 0 else 0.0
            console.print(f"[green]speedup {speedup:.2f}×[/green]")
            if temperature == 0.0:
                m = min(len(base_tokens), len(pld_tokens))
                diverge = next((i for i in range(m) if base_tokens[i] != pld_tokens[i]), None)
                if diverge is None and len(base_tokens) == len(pld_tokens):
                    console.print(f"[green]correctness: {m} tokens identical[/green]")
                elif diverge is None:
                    console.print(
                        f"[yellow]correctness: prefix {m} match, lengths differ "
                        f"base={len(base_tokens)} pld={len(pld_tokens)}[/yellow]"
                    )
                else:
                    console.print(
                        f"[red]correctness FAIL at idx {diverge}: "
                        f"base={base_tokens[diverge]} pld={pld_tokens[diverge]}[/red]"
                    )

        console.print(f"\n[dim]{''.join(pld_text)[:200]}[/dim]")
    finally:
        session.close()


@app.command()
def hybrid(
    model: str = typer.Argument(..., help="Target model (MLX)"),
    draft: str = typer.Option(..., help="Draft model for fallback when PLD has no match"),
    prompt: str = typer.Option(
        "Write a Python function that takes a list of integers and returns "
        "the median. Handle even-length lists by averaging the two middle "
        "elements. Include a docstring and type hints."
    ),
    num_draft: int = typer.Option(4, help="Max draft tokens per verify round"),
    max_ngram: int = typer.Option(3, help="Max n-gram size for PLD lookup"),
    min_ngram: int = typer.Option(2, help="Min n-gram size for PLD lookup"),
    max_tokens: int = typer.Option(128),
    temperature: float = typer.Option(0.0, help="0.0 enables correctness gate vs baseline"),
    compare_baseline: bool = typer.Option(
        True, help="Also run non-speculative baseline and report speedup"
    ),
    headroom_gb: float = typer.Option(4.0),
    log: Optional[str] = typer.Option(None, help="Path to jsonl telemetry log"),
):
    """Hybrid speculative: PLD n-gram lookup first, draft model on no match.

    Fills the bottleneck PLD alone leaves (rounds with no n-gram match degrade
    to plain single-token steps). Uses the same verifier as `spec` and `pld`
    so correctness behaves identically at temp=0.
    """
    be = MLXBackend()
    limits = auto_detect_limits(headroom_bytes=int(headroom_gb * 1024**3))
    budget = StaticBudget(limits)
    recorder = JsonlRecorder(log) if log else NullRecorder()

    console.print(f"[cyan]loading[/cyan] target={model}  draft={draft}")
    be.load(model, None, budget)
    be.load_draft(draft)

    session = Session(backend=be, budget=budget, recorder=recorder)
    settings = GenerationSettings(max_tokens=max_tokens, temperature=temperature)

    try:
        if compare_baseline:
            t0 = time.monotonic()
            base_tokens: list[int] = []
            for tok in session.generate(prompt, settings):
                base_tokens.append(tok.token_id)
            t_base = time.monotonic() - t0
            n_base = len(base_tokens)
            console.print(
                f"[bold]baseline[/bold] tokens={n_base} time={t_base:.2f}s "
                f"tps={n_base / t_base:.1f}"
            )

        t0 = time.monotonic()
        hyb_tokens: list[int] = []
        hyb_text: list[str] = []
        accepted = 0
        n = 0
        for tok in session.generate_hybrid_speculative(
            prompt,
            settings,
            num_draft=num_draft,
            max_ngram_size=max_ngram,
            min_ngram_size=min_ngram,
        ):
            if tok.token_id != -1:
                hyb_tokens.append(tok.token_id)
                n += 1
            hyb_text.append(tok.text)
            if tok.from_draft:
                accepted += 1
        t_hyb = time.monotonic() - t0
        tps = n / t_hyb if t_hyb > 0 else 0.0
        accept_rate = (accepted / n) * 100 if n else 0.0
        console.print(
            f"[bold]hybrid[/bold]   tokens={n} time={t_hyb:.2f}s "
            f"tps={tps:.1f}  from_draft={accept_rate:.1f}%"
        )

        if compare_baseline:
            speedup = t_base / t_hyb if t_hyb > 0 else 0.0
            console.print(f"[green]speedup {speedup:.2f}×[/green]")
            if temperature == 0.0:
                m = min(len(base_tokens), len(hyb_tokens))
                diverge = next((i for i in range(m) if base_tokens[i] != hyb_tokens[i]), None)
                if diverge is None and len(base_tokens) == len(hyb_tokens):
                    console.print(f"[green]correctness: {m} tokens identical[/green]")
                elif diverge is None:
                    console.print(
                        f"[yellow]correctness: prefix {m} match, lengths differ "
                        f"base={len(base_tokens)} hybrid={len(hyb_tokens)}[/yellow]"
                    )
                else:
                    console.print(
                        f"[red]correctness FAIL at idx {diverge}: "
                        f"base={base_tokens[diverge]} hybrid={hyb_tokens[diverge]}[/red]"
                    )

        console.print(f"\n[dim]{''.join(hyb_text)[:200]}[/dim]")
    finally:
        session.close()


@app.command()
def techniques():
    """List registered techniques and their implementation status."""
    table = Table(title="Techniques")
    table.add_column("Name"); table.add_column("Target"); table.add_column("Status")
    table.add_row("expert_cache", "v0", "RFC drafted (docs/expert-cache-rfc.md), premise-check pending")
    table.add_row("spec_decode", "v1", "shipped (MLX, hybrid-attention supported)")
    table.add_row("kv_quant", "v2", "runtime knob shipped (--kv-bits); Technique interface TBD")
    table.add_row("layer_stream", "v2", "stub")
    table.add_row("lc_autoconfig", "linux", "shipped (slam lc-probe / lc-sweep)")
    console.print(table)


@app.command("lc-probe")
def lc_probe(
    model: str = typer.Argument(..., help="Path to GGUF model file"),
    ctx_target: int = typer.Option(32000, help="Target context length"),
    vram_mib: Optional[int] = typer.Option(
        None, help="Override free VRAM in MiB (otherwise probe nvidia-smi)"
    ),
    assume_total_vram: bool = typer.Option(
        True, help="Assume total VRAM (full GPU) instead of current free VRAM"
    ),
    safety_mib: int = typer.Option(200, help="VRAM safety margin"),
    top_n: int = typer.Option(8, help="Show top-N candidate configs"),
):
    """Probe a GGUF model + current GPU and print candidate llama-server configs.

    Use this before `lc-sweep` to sanity check what configs will be tried.
    Linux/CUDA only — MLX users should use `run` / `spec` etc.
    """
    from .llamacpp_config import (
        query_nvidia_gpus,
        read_gguf_info,
        suggest_configs,
    )

    info = read_gguf_info(model)
    console.print(
        f"[bold]{info.path.name}[/bold]  arch={info.architecture}  "
        f"blocks={info.block_count}  ctx_train={info.context_length}"
    )
    console.print(
        f"  heads={info.head_count}/{info.head_count_kv}  "
        f"embed={info.embedding_length}  size={info.file_size_bytes/(1024**3):.2f} GiB"
    )

    if vram_mib is not None:
        free = vram_mib
        console.print(f"[dim]using vram_mib override = {vram_mib} MiB[/dim]")
    else:
        gpus = query_nvidia_gpus()
        if not gpus:
            console.print("[red]no NVIDIA GPU detected; pass --vram-mib[/red]")
            raise typer.Exit(code=1)
        g = gpus[0]
        free = g.total_mib if assume_total_vram else g.free_mib
        label = "total" if assume_total_vram else "free"
        console.print(
            f"[dim]gpu[0]={g.name}  {label}={free} MiB "
            f"(current_free={g.free_mib}, total={g.total_mib})[/dim]"
        )

    cfgs = suggest_configs(info, free_vram_mib=free,
                           vram_safety_mib=safety_mib, ctx_target=ctx_target)
    table = Table(title=f"Candidate configs (budget={max(0, free - safety_mib)} MiB)")
    table.add_column("rank", justify="right")
    table.add_column("n_gpu_layers", justify="right")
    table.add_column("ctx", justify="right")
    table.add_column("batch", justify="right")
    table.add_column("est VRAM MiB", justify="right")
    table.add_column("notes")
    for i, c in enumerate(cfgs[:top_n], 1):
        table.add_row(
            str(i), str(c.n_gpu_layers), str(c.ctx_size), str(c.batch_size),
            str(c.est_vram_mib), c.notes,
        )
    console.print(table)

    if not cfgs:
        console.print("[yellow]No config fits in the given VRAM budget.[/yellow]")


@app.command("lc-sweep")
def lc_sweep(
    model: str = typer.Argument(..., help="Path to GGUF model file"),
    server_bin: str = typer.Option(
        "./build-linux/bin/llama-server",
        help="Path to llama-server binary",
    ),
    ctx_target: int = typer.Option(32000),
    vram_mib: Optional[int] = typer.Option(
        None, help="Override VRAM in MiB (otherwise use total VRAM)"
    ),
    safety_mib: int = typer.Option(200),
    max_configs: int = typer.Option(6, help="Max configs to try"),
    n_predict: int = typer.Option(96, help="Tokens to generate for timing"),
    port: int = typer.Option(8090),
    threads: Optional[int] = typer.Option(8),
    prompt: str = typer.Option(
        "Write a Python function that takes a list of integers and returns "
        "the median. Handle even-length lists by averaging the two middle "
        "elements. Include a docstring and type hints.",
    ),
    log_dir: Optional[str] = typer.Option(None, help="Dir for per-run server logs"),
    jsonl: Optional[str] = typer.Option(None, help="Write results as JSONL"),
):
    """Sweep llama-server configs to find the highest-tps config that fits.

    Each config launches a fresh llama-server, warms up, times a completion,
    then kills the server. Expect ~30-90s per config (dominated by load time).

    Before running: make sure nothing else is holding your GPU (stop any
    running llama-server on port 8080 etc.).
    """
    from pathlib import Path as _Path

    from .lc_sweep import auto_sweep

    console.print(f"[cyan]sweeping[/cyan] {model}")
    ld = _Path(log_dir) if log_dir else None
    rep = auto_sweep(
        model_path=model, server_bin=server_bin,
        ctx_target=ctx_target, vram_override_mib=vram_mib,
        vram_safety_mib=safety_mib, max_configs=max_configs,
        port=port, threads=threads, n_predict=n_predict,
        prompt=prompt, log_dir=ld,
    )

    table = Table(title=f"Sweep results — {_Path(model).name}")
    for col in ("ngl", "ctx", "batch", "tps", "prefill", "vram peak", "load s", "status"):
        table.add_column(col, justify="right")
    for r in rep.results:
        status = "ok" if r.ok else f"FAIL: {r.error[:40]}"
        tps = f"{r.tps:.1f}" if r.ok else "—"
        prefill = f"{r.prefill_tps:.0f}" if r.ok else "—"
        vram = f"{r.vram_peak_mib}" if r.ok else "—"
        load = f"{r.load_s:.1f}"
        table.add_row(
            str(r.config.n_gpu_layers), str(r.config.ctx_size),
            str(r.config.batch_size), tps, prefill, vram, load, status,
        )
    console.print(table)

    w = rep.winner
    if w is not None:
        console.print(
            f"\n[green]winner:[/green] ngl={w.config.n_gpu_layers} "
            f"ctx={w.config.ctx_size} batch={w.config.batch_size} "
            f"→ {w.tps:.1f} t/s (prefill {w.prefill_tps:.0f} t/s)"
        )
        console.print("\n[bold]Suggested llama-server invocation:[/bold]")
        cmd = [server_bin] + w.config.to_args(
            model_path=model, port=8080, host="0.0.0.0",
            threads=threads,
        )
        console.print(" ".join(cmd))
    else:
        console.print("[red]No config succeeded. Check --log-dir tail for errors.[/red]")

    if jsonl:
        rep.to_jsonl(_Path(jsonl))
        console.print(f"[dim]jsonl written: {jsonl}[/dim]")


if __name__ == "__main__":
    app()
