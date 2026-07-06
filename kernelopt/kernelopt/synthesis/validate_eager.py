"""Validate a generated Triton kernel against PyTorch eager mode.

Runs the original PyTorch model and the generated kernel with identical
inputs, compares outputs with torch.allclose. All execution happens in
a subprocess to isolate GPU state.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def validate_against_eager(
    pytorch_path: str,
    generated_kernel_path: str,
    *,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    seeds: tuple[int, ...] = (0, 42),
    timeout: int = 300,
) -> tuple[bool, str]:
    """Compare generated kernel output against PyTorch eager.

    Args:
        pytorch_path: Path to original KernelBench .py with Model + get_inputs().
        generated_kernel_path: Path to LLM-generated kernel with kernel_function + get_inputs().
        atol, rtol: Tolerance for torch.allclose.
        seeds: Random seeds for reproducibility.
        timeout: Subprocess timeout in seconds.

    Returns:
        (passed, message) tuple.
    """
    pytorch_path = str(Path(pytorch_path).resolve())
    generated_kernel_path = str(Path(generated_kernel_path).resolve())

    script = f"""\
import json, sys, torch, importlib.util

torch.cuda.set_device(0)
results = []

# Load original PyTorch model
spec1 = importlib.util.spec_from_file_location("_orig", {pytorch_path!r})
mod1 = importlib.util.module_from_spec(spec1)
sys.modules["_orig"] = mod1
spec1.loader.exec_module(mod1)

model_cls = getattr(mod1, "Model", None)
get_init = getattr(mod1, "get_init_inputs", None)
get_inputs_orig = getattr(mod1, "get_inputs", None)

if model_cls is None or get_inputs_orig is None:
    print(json.dumps({{"passed": False, "error": "No Model or get_inputs in original"}}))
    sys.exit(0)

init_args = get_init() if get_init else []
model = model_cls(*init_args).cuda().eval()

# Load generated kernel
spec2 = importlib.util.spec_from_file_location("_gen", {generated_kernel_path!r})
mod2 = importlib.util.module_from_spec(spec2)
sys.modules["_gen"] = mod2
spec2.loader.exec_module(mod2)

kernel_fn = getattr(mod2, "kernel_function", None)
if kernel_fn is None:
    print(json.dumps({{"passed": False, "error": "No kernel_function in generated kernel"}}))
    sys.exit(0)

seeds = {seeds!r}
failures = []

for seed in seeds:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Generate inputs from the ORIGINAL model's get_inputs (CPU-based).
    # Move to CUDA and use the SAME tensors for both reference and generated
    # kernel to avoid CPU/CUDA RNG divergence.
    inputs = [t.cuda() if isinstance(t, torch.Tensor) else t for t in get_inputs_orig()]
    ref_inputs = [t.clone() if isinstance(t, torch.Tensor) else t for t in inputs]

    # For kernels that externalize weights (e.g., Conv3d where Inductor makes
    # weights explicit), the generated kernel expects MORE inputs than the
    # original model. Extract the model's actual weights and pass them to the
    # generated kernel so both use identical weight values.
    get_inputs_gen = getattr(mod2, "get_inputs", None)
    n_orig = len(inputs)
    if get_inputs_gen is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        gen_raw = [t.cuda() if isinstance(t, torch.Tensor) else t for t in get_inputs_gen()]
        n_gen = len(gen_raw)
        if n_gen > n_orig:
            # Extra args are likely externalized weights. Extract the model's
            # actual weight tensors and use them instead of random ones.
            n_extra = n_gen - n_orig
            model_weights = [p.data.clone() for p in model.parameters()]
            if len(model_weights) >= n_extra:
                # Use model's real weights for the extra args
                gen_inputs = model_weights[:n_extra] + \
                             [t.clone() if isinstance(t, torch.Tensor) else t for t in inputs]
            else:
                # Fallback: use generated kernel's random weights
                gen_inputs = [t.clone() if isinstance(t, torch.Tensor) else t for t in gen_raw[:n_extra]] + \
                             [t.clone() if isinstance(t, torch.Tensor) else t for t in inputs]
        elif n_gen == n_orig:
            # Same number of args — use original inputs directly (avoids
            # CPU/CUDA RNG divergence from different get_inputs implementations)
            gen_inputs = [t.clone() if isinstance(t, torch.Tensor) else t for t in inputs]
        else:
            gen_inputs = [t.clone() if isinstance(t, torch.Tensor) else t for t in inputs]
    else:
        gen_inputs = [t.clone() if isinstance(t, torch.Tensor) else t for t in inputs]

    with torch.no_grad():
        ref_out = model(*ref_inputs)
        gen_out = kernel_fn(*gen_inputs)

    # Normalize outputs to lists for multi-output comparison
    if not isinstance(ref_out, (tuple, list)):
        ref_out = [ref_out]
    if not isinstance(gen_out, (tuple, list)):
        gen_out = [gen_out]

    if len(ref_out) != len(gen_out):
        failures.append(f"seed={{seed}}: output count mismatch ref={{len(ref_out)}} gen={{len(gen_out)}}")
        continue

    for _oi, (_r, _g) in enumerate(zip(ref_out, gen_out)):
        if not isinstance(_r, torch.Tensor) or not isinstance(_g, torch.Tensor):
            continue
        ref_f = _r.float().cpu()
        gen_f = _g.float().cpu()

        if ref_f.shape != gen_f.shape:
            failures.append(f"seed={{seed}} output[{{_oi}}]: shape mismatch ref={{ref_f.shape}} gen={{gen_f.shape}}")
            continue

        max_diff = (ref_f - gen_f).abs().max().item()
        passed = torch.allclose(ref_f, gen_f, atol={atol}, rtol={rtol})

        if not passed:
            failures.append(f"seed={{seed}} output[{{_oi}}]: max_diff={{max_diff:.3e}} > atol={atol}")

if failures:
    print(json.dumps({{"passed": False, "failures": failures}}))
else:
    print(json.dumps({{"passed": True, "message": "Correctness passed ({{len(seeds)}} seeds, eager reference)"}}))
"""

    python = sys.executable
    pipeline_venv = Path(__file__).resolve().parents[3] / ".pipeline_venv" / "bin" / "python"
    if pipeline_venv.exists():
        python = str(pipeline_venv)

    import os
    env = {k: v for k, v in os.environ.items()}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONPATH", None)

    try:
        proc = subprocess.run(
            [python, "-c", script],
            capture_output=True, text=True, timeout=timeout,
            env=env, cwd="/tmp",
        )
        if proc.returncode == 0:
            for line in reversed(proc.stdout.strip().splitlines()):
                if line.strip().startswith("{"):
                    result = json.loads(line)
                    if result.get("passed"):
                        return True, result.get("message", "Passed")
                    failures = result.get("failures", [result.get("error", "unknown")])
                    return False, "Correctness FAILED:\n" + "\n".join(failures)
        return False, f"Subprocess failed (exit {proc.returncode}):\n{proc.stderr[-500:]}"
    except subprocess.TimeoutExpired:
        return False, f"Correctness check timed out ({timeout}s)"
    except Exception as e:
        return False, f"Correctness error: {e}"
