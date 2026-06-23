"""Kernel fusion phase — fuse adjacent Triton kernels before per-kernel optimization.

Entry points:
- run_fusion_phase() — tries pattern-based fusion first, falls back to legacy
- run_pattern_fusion_phase() — pattern-based cross-kernel fusion
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from kernelopt.fusion.graph import parse_call_graph
from kernelopt.fusion.detect import detect_fusion_groups
from kernelopt.fusion.generate import (
    generate_fused_kernel,
    generate_from_pattern_match,
    sanitize_llm_output,
    validate_fusion,
)
from kernelopt.pipeline_ui import pipe_print


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_kernel_sources(model_source: str, kernel_names: list[str]) -> dict[str, str]:
    """Extract the Triton source body (inside ''') for each kernel name.

    Uses the async_compile.triton block pattern to locate each kernel's source.
    Returns a mapping of kernel_name -> triton source (empty string if not found).
    """
    sources: dict[str, str] = {}
    for name in kernel_names:
        pattern = re.compile(
            re.escape(name)
            + r"""\s*=\s*async_compile\.triton\(\s*'"""
            + re.escape(name)
            + r"""'\s*,\s*'''(.*?)'''""",
            re.DOTALL,
        )
        m = pattern.search(model_source)
        sources[name] = m.group(1) if m else ""
    return sources


def _build_fused_async_compile_block(
    group_id: str,
    fused_source: str,
    fused_name: str = "fused_kernel",
) -> str:
    """Wrap fused source in an async_compile.triton() assignment block."""
    # Escape any triple-single-quote sequences in the source
    safe_source = fused_source.replace("'''", "\"\"\"")
    return (
        f"{fused_name} = async_compile.triton('{fused_name}', '''\n"
        f"{safe_source}\n"
        f"''', device_str='cuda')"
    )


def _build_fused_call_line(group, fused_name: str = "fused_kernel") -> str:
    """Build the .run() call line for the fused kernel."""
    # Combine all external inputs and outputs as positional args.
    # The fused kernel uses the same convention as other Triton kernels.
    all_args = list(group.input_buffers) + list(group.output_buffers)
    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for a in all_args:
        if a not in seen:
            seen.add(a)
            deduped.append(a)
    args_str = ", ".join(deduped)
    return f"{fused_name}.run({args_str}, stream=stream0)"


# ---------------------------------------------------------------------------
# Pattern-based fusion (new path)
# ---------------------------------------------------------------------------


def run_pattern_fusion_phase(
    model_source: str,
    llm_config: dict,
    output_dir: str | Path,
    original_model_path: str = "",
) -> tuple[str, list[dict]]:
    """Run pattern-based cross-kernel fusion.

    Uses the PatternRegistry to detect higher-level patterns
    (GEMM+epilogue, pointwise chains, residual+norm, conv+BN) in the
    Inductor kernel graph, then generates pattern-specific fused kernels
    via targeted LLM prompts.
    """
    from kernelopt.fusion.patterns import default_registry
    from kernelopt.multi_kernel import list_triton_kernels

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: collect kernel names and sources
    kernel_names = list_triton_kernels(model_source)
    if not kernel_names:
        return model_source, []

    kernel_sources = _extract_kernel_sources(model_source, kernel_names)

    # Step 2: parse call graph (extended — includes extern_kernels)
    try:
        graph = parse_call_graph(model_source)
    except Exception as exc:
        pipe_print(f"  [pattern-fusion] Graph parse failed: {exc}")
        return model_source, []

    # Step 3: pattern matching
    registry = default_registry()
    matches = registry.match_all(graph, kernel_sources)

    if not matches:
        return model_source, []

    pipe_print(f"  [pattern-fusion] Detected {len(matches)} pattern match(es).")
    for m in matches:
        node_names = [n.name for n in m.nodes]
        pipe_print(
            f"    {m.pattern_name} (profit={m.estimated_profit:.2f}): "
            + " + ".join(node_names[:3])
            + ("..." if len(node_names) > 3 else "")
        )

    results: list[dict] = []
    current_source = model_source

    # Step 4: process each match
    for match in matches:
        node_names = [n.name for n in match.nodes]

        # Skip annotation-only patterns (e.g., conv_epilogue)
        if match.metadata.get("annotation_only"):
            pipe_print(f"    {match.pattern_name}: annotated (no fusion)")
            results.append({
                "group_id": f"pattern_{match.pattern_name}_{node_names[0]}",
                "status": "annotated",
                "speedup": 1.0,
                "message": "annotation-only pattern",
                "kernels": node_names,
                "pattern": match.pattern_name,
            })
            continue

        group_output_dir = output_dir / f"pattern_{match.pattern_name}_{node_names[0]}"
        group_output_dir.mkdir(parents=True, exist_ok=True)

        fused_source: Optional[str] = None
        last_error: str = ""

        # Generate fused kernel (with one retry on failure)
        for attempt in range(2):
            if attempt == 1:
                # Prepend error feedback to the prompt
                pipe_print(f"    Retry with error feedback: {last_error[:100]}")

            fused_source = generate_from_pattern_match(match, llm_config)

            if fused_source is None:
                last_error = "LLM returned no source"
                continue

            # Validate
            orig_path = str(original_model_path) if original_model_path else ""
            try:
                ok, speedup, msg = validate_fusion(
                    original_model_path=orig_path,
                    fused_kernel_source=fused_source,
                    output_dir=str(group_output_dir),
                )
            except Exception as exc:
                ok, speedup, msg = False, 0.0, f"validate raised: {exc}"

            pipe_print(f"    Validation: ok={ok}, speedup={speedup:.2f}x, msg={msg!r}")

            if ok:
                break

            last_error = msg
            if attempt == 0:
                continue
            fused_source = None
            break

        if fused_source is None:
            pipe_print(f"    {match.pattern_name}: failed ({last_error[:80]})")
            results.append({
                "group_id": f"pattern_{match.pattern_name}_{node_names[0]}",
                "status": "failed",
                "speedup": 1.0,
                "message": last_error,
                "kernels": node_names,
                "pattern": match.pattern_name,
            })
            continue

        # Save and restitch
        fused_path = group_output_dir / "fused_kernel.py"
        fused_path.write_text(fused_source)

        from kernelopt.fusion.restitch import patch_call_for_fusion, restitch_fused
        from kernelopt.fusion.graph import ExternKernelNode

        fused_name = "fused_kernel"
        fused_block = _build_fused_async_compile_block(
            group_id=f"pattern_{match.pattern_name}",
            fused_source=fused_source,
            fused_name=fused_name,
        )

        # Separate Triton and extern kernel names for restitch
        triton_names = [n.name for n in match.nodes if not isinstance(n, ExternKernelNode)]
        extern_lines = [n.raw_line for n in match.nodes if isinstance(n, ExternKernelNode)]
        buf_names = [b.name for b in match.eliminated_buffers]

        # Build fused call line
        all_args = list(match.input_buffers) + list(match.output_buffers)
        seen: set[str] = set()
        deduped: list[str] = []
        for a in all_args:
            if a not in seen:
                seen.add(a)
                deduped.append(a)
        fused_call_line = f"{fused_name}.run({', '.join(deduped)}, stream=stream0)"

        insert_after = ""
        if match.eliminated_buffers:
            insert_after = f"{match.eliminated_buffers[-1].name} = empty_strided_cuda"

        first_triton = triton_names[0] if triton_names else None

        def _make_patch(
            triton_names=triton_names,
            buf_names=buf_names,
            fused_call_line=fused_call_line,
            insert_after=insert_after,
            extern_lines=extern_lines,
        ):
            def patch(source: str) -> str:
                return patch_call_for_fusion(
                    call_source=source,
                    kernels_to_remove=triton_names,
                    buffers_to_remove=buf_names,
                    fused_call_line=fused_call_line,
                    insert_after_line=insert_after,
                    extern_lines_to_remove=extern_lines,
                )
            return patch

        if first_triton:
            try:
                current_source = restitch_fused(
                    model_source=current_source,
                    kernel_names_to_remove=triton_names,
                    first_kernel_name=first_triton,
                    fused_block=fused_block,
                    call_patch_fn=_make_patch(),
                )
                pipe_print(f"    {match.pattern_name}: fused and stitched")
                results.append({
                    "group_id": f"pattern_{match.pattern_name}_{node_names[0]}",
                    "status": "fused",
                    "speedup": speedup,
                    "message": msg,
                    "kernels": node_names,
                    "pattern": match.pattern_name,
                })
            except Exception as exc:
                pipe_print(f"    Restitch failed: {exc}")
                results.append({
                    "group_id": f"pattern_{match.pattern_name}_{node_names[0]}",
                    "status": "restitch_failed",
                    "speedup": 1.0,
                    "message": str(exc),
                    "kernels": node_names,
                    "pattern": match.pattern_name,
                })

    return current_source, results


# ---------------------------------------------------------------------------
# Main entry point (legacy + pattern fallback)
# ---------------------------------------------------------------------------


def run_fusion_phase(
    model_source: str,
    llm_config: dict,
    output_dir: str | Path,
    original_model_path: str = "",
) -> tuple[str, list[dict]]:
    """Run the fusion phase over a synthesized model.

    Steps:
    1. Extract Triton kernel block sources from the model source.
    2. Parse the buffer dependency call graph.
    3. Detect fusion groups (chains of adjacent kernels).
    4. For each group: generate a fused kernel via LLM, validate (correctness +
       perf), and if valid restitch into the model source.
    5. On correctness failure, retry once with the error as feedback.

    Parameters
    ----------
    model_source:
        Full text of the synthesized_triton.py model file.
    llm_config:
        Dictionary with LLM config keys: model, api_key, project_id, region.
    output_dir:
        Directory for intermediate files and validation logs.
    original_model_path:
        Path to the original (pre-fusion) synthesized model file; used by
        validate_fusion for subprocess-based correctness checking.

    Returns
    -------
    (model_source, results_list)
        model_source -- potentially updated with fused kernels stitched in.
        results_list -- one dict per detected fusion group with keys:
            group_id, status, speedup, message, kernels.
    """
    from kernelopt.multi_kernel import list_triton_kernels

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Try pattern-based fusion first ────────────────────────────────────
    try:
        pattern_source, pattern_results = run_pattern_fusion_phase(
            model_source=model_source,
            llm_config=llm_config,
            output_dir=output_dir / "pattern_fusion",
            original_model_path=original_model_path,
        )
        if pattern_results:
            fused_count = sum(1 for r in pattern_results if r["status"] == "fused")
            annotated_count = sum(1 for r in pattern_results if r["status"] == "annotated")
            pipe_print(
                f"  [fusion] Pattern fusion: {fused_count} fused, "
                f"{annotated_count} annotated, {len(pattern_results)} total"
            )
            # Continue with legacy detection on the pattern-fused source
            model_source = pattern_source
    except Exception as exc:
        pipe_print(f"  [fusion] Pattern fusion phase failed: {exc} — continuing with legacy.")

    # ── Legacy path: Step 1: collect kernel names and sources ─────────────
    kernel_names = list_triton_kernels(model_source)
    if not kernel_names:
        pipe_print("  [fusion] No Triton kernels found — skipping fusion phase.")
        return model_source, []

    pipe_print(f"  [fusion] Found {len(kernel_names)} Triton kernels for fusion analysis.")
    kernel_sources = _extract_kernel_sources(model_source, kernel_names)

    # ── Step 2: parse call graph ──────────────────────────────────────────
    try:
        graph = parse_call_graph(model_source)
    except Exception as exc:
        pipe_print(f"  [fusion] Call graph parse failed: {exc} — skipping fusion phase.")
        return model_source, []

    # ── Step 3: detect fusion groups ─────────────────────────────────────
    try:
        groups = detect_fusion_groups(graph, kernel_sources)
    except Exception as exc:
        pipe_print(f"  [fusion] Fusion group detection failed: {exc} — skipping fusion phase.")
        return model_source, []

    if not groups:
        pipe_print("  [fusion] No fusible kernel groups detected.")
        return model_source, []

    pipe_print(f"  [fusion] Detected {len(groups)} fusion group(s).")

    results: list[dict] = []
    current_source = model_source

    # ── Step 4: process each fusion group ────────────────────────────────
    for group in groups:
        kernel_names_in_group = [k.name for k in group.kernels]
        pipe_print(
            f"\n  [fusion] Group {group.group_id} ({group.fusion_type}): "
            + " + ".join(kernel_names_in_group)
        )

        group_output_dir = output_dir / group.group_id
        group_output_dir.mkdir(parents=True, exist_ok=True)

        fused_source: Optional[str] = None
        last_error: str = ""

        # ── Step 4a: generate fused kernel (with one retry on correctness failure) ──
        for attempt in range(2):
            if attempt == 1:
                # Re-generate with error feedback attached to llm_config
                feedback_config = dict(llm_config)
                feedback_config["_retry_feedback"] = last_error
                pipe_print(f"    [fusion] Retry attempt 1 with error feedback...")
                # Temporarily patch the group sources to include error context
                # by injecting feedback into the prompt via a wrapper.
                # Since generate_fused_kernel takes llm_config, we add feedback
                # as a note in the config; generate_fused_kernel ignores unknown keys.
                fused_source = _generate_with_feedback(group, feedback_config)
            else:
                fused_source = generate_fused_kernel(group, llm_config)

            if fused_source is None:
                pipe_print(f"    [fusion] LLM generation failed (attempt {attempt + 1}).")
                last_error = "LLM returned no source"
                if attempt == 0:
                    continue
                break

            # ── Step 4b: validate ─────────────────────────────────────────
            orig_path = str(original_model_path) if original_model_path else ""
            try:
                ok, speedup, msg = validate_fusion(
                    original_model_path=orig_path,
                    fused_kernel_source=fused_source,
                    output_dir=str(group_output_dir),
                )
            except Exception as exc:
                ok, speedup, msg = False, 0.0, f"validate_fusion raised: {exc}"

            pipe_print(
                f"    [fusion] Validation: ok={ok}, speedup={speedup:.2f}x, msg={msg!r}"
            )

            if ok:
                break  # Successful — proceed to restitch

            # Correctness failure: record error and retry (only once)
            last_error = msg
            if attempt == 0:
                pipe_print("    [fusion] Correctness failure — retrying with error feedback.")
                continue
            # Second attempt also failed
            fused_source = None
            break

        # ── Step 4c: restitch if we have a validated fused kernel ─────────
        if fused_source is None:
            pipe_print(f"    [fusion] {group.group_id}: skipped (generation/validation failed).")
            results.append({
                "group_id": group.group_id,
                "status": "failed",
                "speedup": 1.0,
                "message": last_error,
                "kernels": kernel_names_in_group,
            })
            continue

        # Save fused kernel source to disk
        fused_path = group_output_dir / "fused_kernel.py"
        fused_path.write_text(fused_source)

        fused_name = "fused_kernel"

        # Build the async_compile block for the fused kernel
        fused_block = _build_fused_async_compile_block(
            group_id=group.group_id,
            fused_source=fused_source,
            fused_name=fused_name,
        )

        # Build the fused call line
        fused_call_line = _build_fused_call_line(group, fused_name=fused_name)

        # Build the call_patch_fn closure
        from kernelopt.fusion.restitch import patch_call_for_fusion, restitch_fused

        first_kernel_name = kernel_names_in_group[0]
        kernels_to_remove = kernel_names_in_group
        buffers_to_remove = [b.name for b in group.eliminated_buffers]

        # insert_after_line: line just before the first kernel's .run() call
        # We'll insert after the last buffer allocation for the first kernel's inputs,
        # or simply after the last allocation of any eliminated buffer.
        # Simplest safe choice: insert after the args unpacking line or the first
        # eliminated buffer allocation, whichever comes first in the call body.
        # Use a sentinel that reliably exists: insert after the first eliminated
        # buffer allocation line. If no eliminated buffers, use empty string
        # (which will not match — fused call won't be inserted automatically).
        insert_after = ""
        if group.eliminated_buffers:
            # Pick the last eliminated buffer name as anchor
            last_elim_buf = group.eliminated_buffers[-1].name
            # The actual line looks like: buf0 = empty_strided_cuda(...)
            # We'll use the buffer name as the needle inside patch_call_for_fusion
            insert_after = f"{last_elim_buf} = empty_strided_cuda"

        def _make_call_patch(
            kernels_to_remove=kernels_to_remove,
            buffers_to_remove=buffers_to_remove,
            fused_call_line=fused_call_line,
            insert_after=insert_after,
        ):
            def call_patch(source: str) -> str:
                return patch_call_for_fusion(
                    call_source=source,
                    kernels_to_remove=kernels_to_remove,
                    buffers_to_remove=buffers_to_remove,
                    fused_call_line=fused_call_line,
                    insert_after_line=insert_after,
                )
            return call_patch

        try:
            current_source = restitch_fused(
                model_source=current_source,
                kernel_names_to_remove=kernels_to_remove,
                first_kernel_name=first_kernel_name,
                fused_block=fused_block,
                call_patch_fn=_make_call_patch(),
            )
            pipe_print(f"    [fusion] {group.group_id}: fused and stitched into model.")
            results.append({
                "group_id": group.group_id,
                "status": "fused",
                "speedup": speedup,
                "message": msg,
                "kernels": kernel_names_in_group,
            })
        except Exception as exc:
            pipe_print(f"    [fusion] Restitch failed for {group.group_id}: {exc}")
            results.append({
                "group_id": group.group_id,
                "status": "restitch_failed",
                "speedup": 1.0,
                "message": str(exc),
                "kernels": kernel_names_in_group,
            })

    return current_source, results


# ---------------------------------------------------------------------------
# Retry helper (injects error feedback into the LLM prompt)
# ---------------------------------------------------------------------------


def _generate_with_feedback(group, llm_config: dict) -> Optional[str]:
    """Retry fused kernel generation, prepending the previous error to the prompt."""
    try:
        from kernelopt.llm import build_llm
    except ImportError as exc:
        pipe_print(f"[fusion] Cannot import build_llm for retry: {exc}")
        return None

    feedback = llm_config.get("_retry_feedback", "Unknown error")

    try:
        llm = build_llm(
            model=llm_config.get("model", "claude-sonnet-4-6"),
            api_key=llm_config.get("api_key"),
            max_tokens=llm_config.get("max_tokens", 8192),
            temperature=llm_config.get("temperature", 0.2),
            project_id=llm_config.get("project_id"),
            region=llm_config.get("region"),
        )
    except Exception as exc:
        pipe_print(f"[fusion] Failed to build LLM for retry: {exc}")
        return None

    from langchain_core.messages import HumanMessage, SystemMessage
    from kernelopt.fusion.generate import FUSION_SYSTEM_PROMPT, build_fusion_prompt

    base_prompt = build_fusion_prompt(group)
    retry_prompt = (
        f"PREVIOUS ATTEMPT FAILED WITH THE FOLLOWING ERROR:\n{feedback}\n\n"
        f"Please fix the issue and produce a corrected fused kernel.\n\n"
        + base_prompt
    )

    messages = [
        SystemMessage(content=FUSION_SYSTEM_PROMPT),
        HumanMessage(content=retry_prompt),
    ]

    try:
        response = llm.invoke(messages)
        source = response.content
        if not source or not source.strip():
            pipe_print("[fusion] LLM retry returned empty response.")
            return None
        return sanitize_llm_output(source.strip())
    except Exception as exc:
        pipe_print(f"[fusion] LLM retry invocation failed: {exc}")
        return None
