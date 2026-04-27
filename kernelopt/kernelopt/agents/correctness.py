"""Correctness agent: verify optimized kernel output against a CPU reference."""

from __future__ import annotations

import importlib.util
import tempfile
import types
from pathlib import Path

import torch
from langchain_core.messages import HumanMessage, SystemMessage

from kernelopt.llm import build_llm

CORRECTNESS_SYSTEM_PROMPT = """\
You are the Correctness Agent. Given an optimized GPU kernel, return a JSON
specification. Python will use that spec to load the kernel, run it, run a CPU
reference, and compare with torch.allclose. You do NOT write a test script.

---

## WHAT TO RETURN

### wrapper_function
The name of the plain Python wrapper function in the file — the function that
allocates the output tensor and calls the @triton.jit kernel with a grid.
Example: "run_triton_softmax", "matmul", "flash_attn_forward".
Do NOT name the @triton.jit kernel itself.

### reference_code
A self-contained Python string that defines a function named exactly `reference`.
It receives the same positional arguments as wrapper_function but as CPU tensors.
It must use only standard torch operations (no Triton, no CUDA).
Start with `import torch` if needed.

Example for row-wise softmax:
    import torch
    def reference(x):
        return torch.softmax(x, dim=-1)

### test_inputs
An ordered list of input tensor specs matching the arguments wrapper_function
expects. For each input specify:
- shape: exact list of ints (use kernel_args dimensions if provided)
- dtype: "float32", "float16", "bfloat16", "int32", or "int64"
- distribution: "randn" for most ops, "rand" for ops needing [0,1], "ones"/"zeros" for special cases

### atol / rtol
Tolerances for torch.allclose (matches torch.testing.assert_close defaults):
- FP32 (element-wise or reductions): atol=1e-5, rtol=1.3e-6
- FP16: atol=1e-3, rtol=1e-3
- BF16: atol=1e-5, rtol=1.6e-2

### skip
Set true ONLY if the operation cannot be tested automatically (custom RNG,
hardware-specific stateful op). Default: false.
"""


def _make_tensor(inp, device: str) -> torch.Tensor:
    dtype_map = {
        "float32":  torch.float32,
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
        "int32":    torch.int32,
        "int64":    torch.int64,
    }
    dtype = dtype_map.get(inp.dtype, torch.float32)
    dist = inp.distribution
    if dist == "randn":
        return torch.randn(inp.shape, dtype=dtype, device=device)
    elif dist == "rand":
        return torch.rand(inp.shape, dtype=dtype, device=device)
    elif dist == "ones":
        return torch.ones(inp.shape, dtype=dtype, device=device)
    elif dist == "zeros":
        return torch.zeros(inp.shape, dtype=dtype, device=device)
    elif dist == "randint":
        lo = int(inp.low) if inp.low is not None else 0
        hi = int(inp.high) if inp.high is not None else 128
        return torch.randint(lo, hi, inp.shape, dtype=dtype, device=device)
    return torch.randn(inp.shape, dtype=dtype, device=device)


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
        print(f"[correctness] SKIP: {reason}")
        return False, f"Correctness skipped: {reason}"

    print(f"[correctness] wrapper={spec.wrapper_function!r}  atol={spec.atol}  rtol={spec.rtol}")
    print(f"[correctness] reference_code:\n{spec.reference_code}")

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

    # --- Step 4: Run GPU vs CPU comparison for 3 seeds ---
    seeds = [0, 42, 1337]
    failures: list[str] = []

    for seed in seeds:
        torch.manual_seed(seed)
        try:
            inputs_cuda = [_make_tensor(inp, device="cuda") for inp in spec.test_inputs]
            inputs_cpu  = [t.cpu() for t in inputs_cuda]

            gpu_out = wrapper_fn(*inputs_cuda)
            ref_out = reference_fn(*inputs_cpu)

            # Unwrap tuples (some kernels return (output, lse) etc.)
            if isinstance(gpu_out, (tuple, list)):
                gpu_out = gpu_out[0]
            if isinstance(ref_out, (tuple, list)):
                ref_out = ref_out[0]

            gpu_f = gpu_out.cpu().float()
            ref_f = ref_out.float()

            passed = torch.allclose(gpu_f, ref_f, atol=spec.atol, rtol=spec.rtol)
            max_diff = (gpu_f - ref_f).abs().max().item()
            print(f"[correctness] seed={seed}  passed={passed}  max_diff={max_diff:.3e}")

            if not passed:
                failures.append(f"seed={seed}: max_diff={max_diff:.3e} > atol={spec.atol}")

        except Exception as e:
            import traceback
            failures.append(f"seed={seed}: {e}\n{traceback.format_exc()}")

    Path(optimized_path).unlink(missing_ok=True)

    if failures:
        return False, "Correctness FAILED:\n" + "\n".join(failures)
    return True, f"Correctness passed (3 seeds, atol={spec.atol}, rtol={spec.rtol})"
