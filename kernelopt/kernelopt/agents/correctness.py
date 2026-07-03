"""Correctness agent: verify optimized kernel output against a CPU reference."""

from __future__ import annotations

import importlib.util
import logging
import tempfile
import types
from pathlib import Path

import torch
from langchain_core.messages import HumanMessage, SystemMessage

from kernelopt.llm import build_llm

logger = logging.getLogger(__name__)

from kernelopt.agents.correctness_prompts import CORRECTNESS_SYSTEM_PROMPT


def _make_input(inp, device: str):
    """Create a test input value from a TestInput spec.

    Returns a torch.Tensor (on *device*) for kind="tensor",
    or a plain Python scalar for kind="scalar".
    """
    if inp.kind == "scalar":
        val = inp.value if inp.value is not None else 1.0
        cast = {"int": int, "bool": bool, "float": float}.get(inp.scalar_dtype or "float", float)
        return cast(val)

    dtype_map = {
        "float32":  torch.float32,
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
        "int32":    torch.int32,
        "int64":    torch.int64,
    }
    dtype = dtype_map.get(inp.dtype or "float32", torch.float32)
    shape = inp.shape or [1]
    dist = inp.distribution
    if dist == "randn":
        return torch.randn(shape, dtype=dtype, device=device)
    elif dist == "rand":
        return torch.rand(shape, dtype=dtype, device=device)
    elif dist == "ones":
        return torch.ones(shape, dtype=dtype, device=device)
    elif dist == "zeros":
        return torch.zeros(shape, dtype=dtype, device=device)
    elif dist == "randint":
        lo = int(inp.low) if inp.low is not None else 0
        hi = int(inp.high) if inp.high is not None else 128
        return torch.randint(lo, hi, shape, dtype=dtype, device=device)
    return torch.randn(shape, dtype=dtype, device=device)


def run_correctness_agent(
    kernel_source: str,
    kernel_path: str,
    framework: str,
    api_key: str | None = None,
    model: str = "claude-sonnet-4-6",
    project_id: str | None = None,
    region: str | None = None,
    kernel_args: list[str] | None = None,
) -> tuple[bool, str]:
    """
    Verify an optimized kernel against a CPU reference.

    Flow:
      1. LLM returns a CorrectnessSpec (structured JSON):
           wrapper_function, reference_code, test_inputs, atol, rtol
      2. Python loads the optimized kernel via importlib (exact file path —
         no import patching, no cwd ambiguity).
      3. Python exec()s the reference_code to get a `reference(*args)` function.
      4. For 3 random seeds: generate inputs, call wrapper + reference,
         compare with torch.allclose.
      5. Return (passed, message).
    """
    from kernelopt.agents.schemas import CorrectnessSpec

    llm = build_llm(
        model=model,
        api_key=api_key,
        max_tokens=2048,
        temperature=0.1,
        project_id=project_id,
        region=region,
    )

    config_note = ""
    if kernel_args:
        config_note = (
            f"\n\n## Kernel Args\n"
            f"Profiled with: `{' '.join(kernel_args)}`\n"
            f"Use these exact dimensions for test_inputs shapes."
        )

    human_msg = (
        f"## Framework: {framework}\n\n"
        f"## Kernel Source ({Path(kernel_path).name})\n"
        f"```\n{kernel_source}\n```"
        f"{config_note}\n\n"
        f"Return the CorrectnessSpec JSON for this kernel."
    )

    # --- Step 1: LLM returns structured spec ---
    structured_llm = llm.with_structured_output(CorrectnessSpec)
    try:
        spec = structured_llm.invoke([
            SystemMessage(content=CORRECTNESS_SYSTEM_PROMPT),
            HumanMessage(content=human_msg),
        ])
    except Exception as e:
        return False, f"Correctness agent failed to produce spec: {e}"

    if spec.skip:
        reason = spec.skip_reason or "no reason given"
        logger.info("SKIP: %s", reason)
        return False, f"Correctness skipped: {reason}"

    logger.debug("wrapper=%r  atol=%s  rtol=%s", spec.wrapper_function, spec.atol, spec.rtol)
    logger.debug("reference_code:\n%s", spec.reference_code)

    # --- Step 2: Load optimized kernel via importlib (exact file, no ambiguity) ---
    kernel_basename = Path(kernel_path).stem
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.py', delete=False, dir='/tmp',
        prefix=f'{kernel_basename}_optimized_',
    ) as f:
        f.write(kernel_source)
        optimized_path = f.name

    try:
        mod_spec = importlib.util.spec_from_file_location("_opt_kernel", optimized_path)
        module = types.ModuleType("_opt_kernel")
        mod_spec.loader.exec_module(module)
    except Exception as e:
        Path(optimized_path).unlink(missing_ok=True)
        return False, f"Failed to import optimized kernel from {optimized_path}: {e}"

    wrapper_fn = getattr(module, spec.wrapper_function, None)
    if wrapper_fn is None:
        available = [n for n in dir(module) if not n.startswith("_")]
        Path(optimized_path).unlink(missing_ok=True)
        return False, (
            f"Wrapper function '{spec.wrapper_function}' not found. "
            f"Available names: {available}"
        )

    # --- Step 3: Build reference function from LLM-provided code ---
    ref_ns: dict = {"torch": torch}
    try:
        exec(spec.reference_code, ref_ns)  # noqa: S102
    except Exception as e:
        Path(optimized_path).unlink(missing_ok=True)
        return False, f"Failed to compile reference_code: {e}\n\nCode:\n{spec.reference_code}"

    reference_fn = ref_ns.get("reference")
    if reference_fn is None:
        Path(optimized_path).unlink(missing_ok=True)
        return False, (
            f"reference_code must define a function named 'reference'. "
            f"Defined names: {[k for k in ref_ns if not k.startswith('_')]}"
        )

    # --- Step 4: Run GPU vs CPU comparison for multiple seeds ---
    from kernelopt.memory import CORRECTNESS_SEEDS
    seeds = CORRECTNESS_SEEDS
    failures: list[str] = []

    for seed in seeds:
        torch.manual_seed(seed)
        try:
            inputs_cuda = [_make_input(inp, device="cuda") for inp in spec.test_inputs]
            inputs_cpu  = [t.cpu() if isinstance(t, torch.Tensor) else t for t in inputs_cuda]

            gpu_out = wrapper_fn(*inputs_cuda)
            ref_out = reference_fn(*inputs_cpu)

            # Normalize outputs to lists for multi-output comparison
            if not isinstance(gpu_out, (tuple, list)):
                gpu_out = [gpu_out]
            if not isinstance(ref_out, (tuple, list)):
                ref_out = [ref_out]

            for _oi, (_g, _r) in enumerate(zip(gpu_out, ref_out)):
                if not isinstance(_g, torch.Tensor) or not isinstance(_r, torch.Tensor):
                    continue
                gpu_f = _g.cpu().float()
                ref_f = _r.float()

                # Enforce minimum tolerance floor based on tensor size
                atol = max(spec.atol, 1e-3 if gpu_f.numel() > 100_000 else spec.atol)
                rtol = max(spec.rtol, 1e-3 if gpu_f.numel() > 100_000 else spec.rtol)
                if atol > spec.atol or rtol > spec.rtol:
                    logger.debug(
                        "tolerance overridden for large tensor (%d elements): "
                        "atol %s -> %s, rtol %s -> %s",
                        gpu_f.numel(), f"{spec.atol:.0e}", f"{atol:.0e}",
                        f"{spec.rtol:.0e}", f"{rtol:.0e}",
                    )

                passed = torch.allclose(gpu_f, ref_f, atol=atol, rtol=rtol)
                max_diff = (gpu_f - ref_f).abs().max().item()
                logger.debug("seed=%s output[%d]  passed=%s  max_diff=%.3e  atol=%s", seed, _oi, passed, max_diff, atol)

                if not passed:
                    failures.append(f"seed={seed} output[{_oi}]: max_diff={max_diff:.3e} > atol={atol}")

        except Exception as e:
            import traceback
            failures.append(f"seed={seed}: {e}\n{traceback.format_exc()}")

    Path(optimized_path).unlink(missing_ok=True)

    if failures:
        return False, "Correctness FAILED:\n" + "\n".join(failures)
    return True, f"Correctness passed (3 seeds, atol={spec.atol}, rtol={spec.rtol})"
