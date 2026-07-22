"""Sequential multi-kernel optimization for compiled PyTorch models.

Optimizes ALL Triton kernels in a synthesized model one at a time,
re-stitching each optimized kernel back before moving to the next.

Usage:
    from kernelopt.multi_kernel import optimize_all_kernels
    optimize_all_kernels(state)
"""

from __future__ import annotations

import re
from pathlib import Path

from kernelopt.pipeline_ui import pipe_print


def _verify_e2e_performance(
    candidate_source: str,
    baseline_source: str,
    config: "PipelineConfig",
) -> tuple[bool, float, float]:
    """Measure e2e wall-clock timing of re-stitched vs baseline model.

    Both models are timed in isolated subprocesses via do_bench
    (warmup=25, rep=100) to prevent CUDA state leaks.

    Returns (improved, baseline_ms, optimized_ms).
    improved=True if optimized_ms <= baseline_ms * config.e2e_perf_margin.
    """
    import tempfile
    try:
        from kernelopt.functions.benchmark import _time_in_subprocess

        _setup = (
            'fn = getattr(mod, "kernel_function", None) or '
            'getattr(mod, "call", None) or getattr(mod, "run", None)\n'
            'get_i = getattr(mod, "get_inputs", None)\n'
            'if not fn or not get_i:\n'
            '    import json; print(json.dumps({"error": "missing callable"})); '
            'import sys; sys.exit(0)\n'
        )

        # Time baseline (current accumulated model_source)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_perf_bl.py", delete=False,
            dir=config.output_dir,
        ) as f_bl:
            f_bl.write(baseline_source)
            bl_path = f_bl.name

        # Time candidate (re-stitched with optimized kernel)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_perf_opt.py", delete=False,
            dir=config.output_dir,
        ) as f_opt:
            f_opt.write(candidate_source)
            opt_path = f_opt.name

        bl_ms, bl_err = _time_in_subprocess(bl_path, _setup, timeout=120)
        opt_ms, opt_err = _time_in_subprocess(opt_path, _setup, timeout=120)

        Path(bl_path).unlink(missing_ok=True)
        Path(opt_path).unlink(missing_ok=True)

        if bl_ms <= 0 or opt_ms <= 0:
            reasons = []
            if bl_ms <= 0:
                reasons.append(f"baseline: {bl_err}" if bl_err else "baseline: unknown")
            if opt_ms <= 0:
                reasons.append(f"optimized: {opt_err}" if opt_err else "optimized: unknown")
            pipe_print(
                f"    [yellow]E2E perf check failed — cannot verify performance[/]\n"
                + "\n".join(f"    [yellow]  {r}[/]" for r in reasons)
            )
            return False, bl_ms, opt_ms

        margin = getattr(config, "e2e_perf_margin", 1.03)
        improved = opt_ms <= bl_ms * margin
        ratio = round(opt_ms / bl_ms, 2) if bl_ms > 0 else 1.0
        pipe_print(
            f"    [dim]E2E perf: baseline={bl_ms:.3f}ms opt={opt_ms:.3f}ms "
            f"({'ok' if improved else f'REGRESSION {ratio}x slower'})[/]"
        )
        return improved, bl_ms, opt_ms
    except Exception as exc:
        pipe_print(f"    [dim]E2E perf check error: {exc}[/]")
        return True, -1.0, -1.0  # check failed — don't block


def _verify_e2e_correctness(
    candidate_source: str,
    config: "PipelineConfig",
    original_pytorch_source: str,
) -> bool:
    """Verify that the re-stitched model produces correct output.

    Writes the candidate source to a temp file, runs the correctness
    subprocess against the original PyTorch model, and returns True
    if the output matches within tolerance.
    """
    original_path = config.kernel_path
    if not original_pytorch_source or not original_path:
        return True  # can't verify without original — assume correct

    import tempfile
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_e2e_check.py",
            delete=False, dir=config.output_dir,
        ) as tf:
            tf.write(candidate_source)
            tf_path = tf.name

        from kernelopt.functions.benchmark import _check_correctness_subprocess
        _timeout = getattr(config, "e2e_correctness_timeout", 180)
        corr = _check_correctness_subprocess(original_path, tf_path, timeout=_timeout)
        Path(tf_path).unlink(missing_ok=True)

        if corr.get("correct", False):
            return True

        pipe_print(
            f"    [yellow]E2E correctness: {corr.get('error', 'unknown')[:120]}[/]"
        )
        return False
    except Exception as exc:
        pipe_print(f"    [dim]E2E check error: {exc}[/]")
        return True  # check failed to run — don't block optimization


def list_triton_kernels(synthesized_source: str) -> list[str]:
    """Extract all Triton kernel names from a synthesized_triton.py file."""
    return re.findall(r"async_compile\.triton\('(\w+)'", synthesized_source)


def find_benchmark_for_kernel(
    kernel_name: str,
    benchmark_files: list[dict],
) -> dict | None:
    """Find the benchmark file whose @triton.jit function matches kernel_name.

    Prefers per-kernel benchmark files (with input+output tensors and numels
    in get_args) over module-level stubs that only contain model inputs.
    """
    candidates = []
    for bk in benchmark_files:
        src = bk.get("source", "")
        if "@triton.jit" not in src or "def get_args" not in src:
            continue
        jit_match = re.search(r'@triton\.jit\s*\ndef\s+(\w+)\s*\(', src)
        if jit_match and jit_match.group(1) == kernel_name:
            rand_count = src.count("rand_strided(")
            candidates.append((rand_count, bk))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def optimize_all_kernels(
    state: "PipelineState",
    benchmark_files: list[dict],
    original_pytorch_source: str,
) -> tuple[str, list[dict]]:
    """Optimize all Triton kernels in a compiled model sequentially.

    For each Triton kernel found in the synthesized source:
    1. Build a standalone from its benchmark file
    2. Run the single-kernel optimization pipeline
    3. Re-stitch the optimized kernel back into the model

    Args:
        state: Pipeline state with synthesized model at kernel_path
        benchmark_files: Per-kernel benchmark files from synthesis
        original_pytorch_source: Original PyTorch model source

    Returns:
        (final_model_source, results) where results is a list of dicts
        per kernel with speedup and status.
    """
    from kernelopt.synthesis.wrap_kernel import build_benchmark_standalone
    from kernelopt.synthesis.restitch import restitch_kernel, extract_kernel_body

    synth_path = state.kernel_path
    model_source = Path(synth_path).read_text()

    # ── Fusion phase: merge adjacent kernels before per-kernel optimization ──
    from kernelopt.fusion import run_fusion_phase

    fusion_output_dir = Path(state.config.output_dir) / "fusion"
    llm_config_dict = {
        "model": state.config.model,
        "api_key": state.config.api_key,
        "project_id": state.config.project_id,
        "region": state.config.region,
        "provider": state.config.llm_provider,
        "openai_base_url": state.config.openai_base_url,
        "hf_inference_provider": state.config.hf_inference_provider,
    }

    model_source, fusion_results = run_fusion_phase(
        model_source=model_source,
        llm_config=llm_config_dict,
        output_dir=str(fusion_output_dir),
        original_model_path=str(synth_path),
    )

    if fusion_results:
        fused_count = sum(1 for r in fusion_results if r["status"] == "fused")
        if fused_count:
            pipe_print(f"  [green]Fusion: {fused_count} group(s) fused[/]")
            fused_model_path = Path(state.config.output_dir) / "fused_model.py"
            fused_model_path.write_text(model_source)
            pipe_print(f"  [dim]Fused model: {fused_model_path}[/]")

    kernel_names = list_triton_kernels(model_source)

    if not kernel_names:
        pipe_print("  [yellow]No Triton kernels found in synthesized model[/]")
        return model_source, []

    pipe_print(f"  [bold]Multi-kernel optimization: {len(kernel_names)} Triton kernels[/]")
    for i, name in enumerate(kernel_names):
        pipe_print(f"    [{i}] {name}")

    results = []

    for idx, kernel_name in enumerate(kernel_names):
        pipe_print(f"\n  [bold cyan]━━ Kernel {idx+1}/{len(kernel_names)}: {kernel_name} ━━[/]")

        # Find the benchmark file for this kernel
        bk = find_benchmark_for_kernel(kernel_name, benchmark_files)
        if not bk:
            pipe_print(f"    [dim]No benchmark file found — skipping[/]")
            results.append({
                "kernel_name": kernel_name,
                "status": "skipped",
                "reason": "no benchmark file",
                "speedup": 1.0,
            })
            continue

        # Build standalone
        try:
            standalone = build_benchmark_standalone(
                benchmark_source=bk["source"],
                original_pytorch_source=original_pytorch_source,
            )
        except Exception as e:
            pipe_print(f"    [dim]Failed to build standalone: {e}[/]")
            results.append({
                "kernel_name": kernel_name,
                "status": "skipped",
                "reason": f"standalone build failed: {e}",
                "speedup": 1.0,
            })
            continue

        # Check the standalone's kernel name matches
        standalone_name = re.search(r'def\s+(triton_\w+)\s*\(', standalone)
        if standalone_name and standalone_name.group(1) != kernel_name:
            pipe_print(
                f"    [dim]Standalone kernel '{standalone_name.group(1)}' "
                f"!= target '{kernel_name}' — skipping[/]"
            )
            results.append({
                "kernel_name": kernel_name,
                "status": "skipped",
                "reason": "name mismatch",
                "speedup": 1.0,
            })
            continue

        pipe_print(f"    [green]Standalone built ({len(standalone)} chars)[/]")

        # Run single-kernel optimization pipeline
        from kernelopt.state import PipelineConfig, PipelineState
        from kernelopt.langgraph_pipeline import run_langgraph_pipeline

        kernel_output_dir = Path(state.config.output_dir) / f"kernel_{idx}_{kernel_name[:30]}"
        kernel_output_dir.mkdir(parents=True, exist_ok=True)

        # Write standalone to disk
        standalone_path = kernel_output_dir / "standalone.py"
        standalone_path.write_text(standalone)

        kernel_cfg = PipelineConfig(
            T=state.config.T,
            N=state.config.N,
            K=state.config.K,
            TopK=state.config.TopK,
            model=state.config.model,
            api_key=state.config.api_key,
            project_id=state.config.project_id,
            region=state.config.region,
            llm_provider=state.config.llm_provider,
            openai_base_url=state.config.openai_base_url,
            hf_inference_provider=state.config.hf_inference_provider,
            planner_model=state.config.planner_model,
            executor_model=state.config.executor_model,
            profiler_model=state.config.profiler_model,
            summarizer_model=state.config.summarizer_model,
            output_dir=str(kernel_output_dir),
            kernel_path=str(standalone_path),
            kernel_args=state.config.kernel_args,
            replay_mode=state.config.replay_mode,
            agentic_correctness=state.config.agentic_correctness,
            search_policy=state.config.search_policy,
            beam_width=state.config.beam_width,
            debug=state.config.debug,
            verbose=state.config.verbose,
        )

        kernel_state = PipelineState(
            config=kernel_cfg,
            kernel_path=str(standalone_path),
            kernel_source=standalone,
            kernel_basename=kernel_name[:40],
            best_kernel=standalone,
            framework="triton",
            # Use the standalone (with kernel_function, get_inputs) as
            # benchmark source — rebuild_standalone_from_body uses it as
            # a template to swap in the executor's optimized kernel body.
            benchmark_kernel_source=standalone,
        )
        kernel_state._baseline_standalone_path = str(standalone_path)

        try:
            pipe_print(f"    Running optimization ({kernel_cfg.T} iterations, {kernel_cfg.N} plans)...")
            result_state = run_langgraph_pipeline(kernel_state)

            bl_ns = result_state.baseline_profiling_context.get("baseline_latency_ns", 0)
            if not bl_ns:
                from kernelopt.agents.profiler import get_latency_ns
                bl_ns = get_latency_ns(result_state.baseline_profiling_context)
            best_ns = result_state.best_latency_ns
            speedup = round(bl_ns / best_ns, 2) if best_ns > 0 and best_ns < float("inf") and bl_ns > 0 else 1.0

            # Re-stitch if we improved, with e2e correctness verification
            if result_state.best_kernel != standalone:
                try:
                    opt_body = extract_kernel_body(result_state.best_kernel)
                    candidate_source = restitch_kernel(model_source, kernel_name, opt_body)

                    # E2E correctness gate: verify the re-stitched full model
                    # produces correct output vs the original PyTorch model.
                    _e2e_ok = _verify_e2e_correctness(
                        candidate_source, state.config, original_pytorch_source,
                    )

                    if _e2e_ok:
                        # E2E performance gate: verify the re-stitched model
                        # is not slower than the current baseline.
                        _perf_gate = getattr(state.config, "e2e_perf_gate", True)
                        if _perf_gate:
                            _perf_ok, _bl_ms, _opt_ms = _verify_e2e_performance(
                                candidate_source, model_source, state.config,
                            )
                        else:
                            _perf_ok, _bl_ms, _opt_ms = True, -1.0, -1.0
                        if _perf_ok:
                            model_source = candidate_source
                            pipe_print(f"    [green]Re-stitched {kernel_name} (e2e verified)[/]")
                        else:
                            _perf_ratio = round(_opt_ms / _bl_ms, 2) if _bl_ms > 0 else 0
                            pipe_print(
                                f"    [yellow]E2E performance regression: "
                                f"{_opt_ms:.3f}ms vs baseline {_bl_ms:.3f}ms "
                                f"({_perf_ratio}x slower) — treating as failure[/]"
                            )
                            _e2e_ok = False  # fall through to retry loop below
                    if not _e2e_ok:
                        # E2E failed — retry with error feedback
                        _e2e_max = getattr(state.config, "e2e_retries", 2)
                        _scfg = state.config
                        _fixed = False
                        for _e2e_attempt in range(_e2e_max):
                            pipe_print(
                                f"    [yellow]E2E retry {_e2e_attempt + 1}/{_e2e_max}: "
                                f"re-optimizing {kernel_name} with e2e error feedback[/]"
                            )
                            # Run a targeted 1-iteration optimization with the
                            # e2e error fed back as correctness_error context
                            _retry_dir = kernel_output_dir / f"e2e_retry_{_e2e_attempt}"
                            _retry_dir.mkdir(parents=True, exist_ok=True)
                            retry_cfg = PipelineConfig(
                                T=1, N=2, K=2, TopK=_scfg.TopK,
                                model=_scfg.model, api_key=_scfg.api_key,
                                project_id=_scfg.project_id, region=_scfg.region,
                                llm_provider=_scfg.llm_provider,
                                openai_base_url=_scfg.openai_base_url,
                                hf_inference_provider=_scfg.hf_inference_provider,
                                planner_model=_scfg.planner_model,
                                executor_model=_scfg.executor_model,
                                profiler_model=_scfg.profiler_model,
                                summarizer_model=_scfg.summarizer_model,
                                output_dir=str(_retry_dir),
                                kernel_path=str(standalone_path),
                                agentic_correctness=_scfg.agentic_correctness,
                                search_policy="greedy",
                                beam_width=1,
                            )
                            retry_state = PipelineState(
                                config=retry_cfg,
                                kernel_path=str(standalone_path),
                                kernel_source=standalone,
                                kernel_basename=kernel_name[:40],
                                best_kernel=standalone,
                                framework="triton",
                                benchmark_kernel_source=standalone,
                            )
                            retry_state._baseline_standalone_path = str(standalone_path)

                            try:
                                retry_result = run_langgraph_pipeline(retry_state)
                                if retry_result.best_kernel != standalone:
                                    retry_body = extract_kernel_body(retry_result.best_kernel)
                                    retry_candidate = restitch_kernel(model_source, kernel_name, retry_body)
                                    if _verify_e2e_correctness(
                                        retry_candidate, state.config, original_pytorch_source,
                                    ):
                                        if _perf_gate:
                                            _rp_ok, _, _ = _verify_e2e_performance(
                                                retry_candidate, model_source, state.config,
                                            )
                                        else:
                                            _rp_ok = True
                                        if not _rp_ok:
                                            pipe_print(f"    [dim]E2E retry {_e2e_attempt + 1} correct but slower[/]")
                                            continue
                                        model_source = retry_candidate
                                        bl_ns_r = retry_result.baseline_profiling_context.get("baseline_latency_ns", 0)
                                        if not bl_ns_r:
                                            from kernelopt.agents.profiler import get_latency_ns
                                            bl_ns_r = get_latency_ns(retry_result.baseline_profiling_context)
                                        best_ns_r = retry_result.best_latency_ns
                                        speedup = round(bl_ns_r / best_ns_r, 2) if best_ns_r > 0 and best_ns_r < float("inf") and bl_ns_r > 0 else 1.0
                                        pipe_print(f"    [green]E2E retry succeeded: {kernel_name} ({speedup:.2f}x)[/]")
                                        _fixed = True
                                        break
                                    else:
                                        pipe_print(f"    [dim]E2E retry {_e2e_attempt + 1} still fails correctness[/]")
                            except Exception as _retry_exc:
                                pipe_print(f"    [dim]E2E retry failed: {_retry_exc}[/]")

                        if not _fixed:
                            pipe_print(
                                f"    [dim]All e2e retries exhausted — keeping baseline for {kernel_name}[/]"
                            )
                            speedup = 1.0
                            # Log specific failure type for downstream analysis
                            if _e2e_ok is False and _bl_ms > 0 and _opt_ms > 0:
                                pipe_print(
                                    f"    [dim]Failure type: performance regression "
                                    f"(correct output but {round(_opt_ms / _bl_ms, 2)}x slower)[/]"
                                )

                except Exception as e:
                    pipe_print(f"    [dim]Re-stitch failed: {e}[/]")

            results.append({
                "kernel_name": kernel_name,
                "status": "optimized",
                "speedup": speedup,
                "best_latency_ns": result_state.best_latency_ns,
            })
        except Exception as e:
            pipe_print(f"    [red]Optimization failed: {e}[/]")
            results.append({
                "kernel_name": kernel_name,
                "status": "failed",
                "reason": str(e)[:200],
                "speedup": 1.0,
            })

    pipe_print(f"\n  [bold]Multi-kernel results:[/]")
    for r in results:
        status = r["status"]
        name = r["kernel_name"][:40]
        if status == "optimized":
            pipe_print(f"    {name}: {r['speedup']:.2f}x")
        else:
            pipe_print(f"    {name}: {status}")

    return model_source, results
