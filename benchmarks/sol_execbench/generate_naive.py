"""Generate naive Triton kernels for SOL-ExecBench problems.

Reads each JSON spec's reference code and sends it to Claude Sonnet 4
to produce a Triton kernel. Includes GPU validation with retry.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
import traceback
from pathlib import Path

from benchmarks.sol_execbench.harness import list_problems, load_problem, resolve_axes
from benchmarks.sol_execbench.wrap_kernel import wrap_kernel

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert GPU kernel engineer. Given a PyTorch reference function,
    write a correct Triton kernel that implements the same computation.

    Requirements:
    - Use @triton.jit for kernel functions
    - Write a Python `run()` function with the EXACT same signature as the reference
    - The `run()` function allocates outputs and launches the Triton kernel(s)
    - Use float32 accumulation for numerical stability
    - The file must be self-contained (only torch, triton, triton.language imports)
    - Handle edge cases (masking for non-power-of-2 sizes, etc.)

    Output ONLY the Python code, no explanation.""")

USER_TEMPLATE = textwrap.dedent("""\
    Write a Triton kernel implementing this PyTorch function:

    ```python
    {reference_code}
    ```

    Input specification:
    {input_spec}

    Axes (dimensions):
    {axes_spec}

    The kernel must define a `run()` function with the same signature.""")

RETRY_TEMPLATE = textwrap.dedent("""\
    Your previous Triton kernel crashed at runtime.

    Reference:
    ```python
    {reference_code}
    ```

    Previous kernel:
    ```python
    {previous_code}
    ```

    Error:
    ```
    {error}
    ```

    Fix it. Output ONLY the corrected Python code.""")

VALIDATE_TIMEOUT = 60


def _strip_fences(code: str) -> str:
    if code.startswith("```"):
        lines = code.split("\n")
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)
    return code


def _validate_kernel(kernel_path: str, gpu_id: int = 0) -> tuple[bool, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        proc = subprocess.run(
            [sys.executable, kernel_path, "--profile-only"],
            capture_output=True, text=True, timeout=VALIDATE_TIMEOUT, env=env,
        )
        if proc.returncode == 0:
            return True, ""
        return False, f"exit {proc.returncode}\n{proc.stderr[-1500:]}"
    except subprocess.TimeoutExpired:
        return False, f"timeout after {VALIDATE_TIMEOUT}s"
    except Exception as e:
        return False, str(e)


def generate_naive_kernel(
    problem_path: str,
    output_path: str,
    *,
    model: str = "claude-sonnet-4@20250514",
    project_id: str | None = None,
    region: str = "global",
    gpu_id: int = 0,
    max_retries: int = 3,
    validate: bool = True,
) -> dict:
    """Generate a naive Triton kernel for a SOL-ExecBench problem."""
    from anthropic import AnthropicVertex

    problem = load_problem(problem_path)
    client = AnthropicVertex(
        project_id=project_id or os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID", ""),
        region=region,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    input_spec = json.dumps(problem.inputs_spec, indent=2)
    axes_spec = json.dumps(problem.axes, indent=2)

    total_in = total_out = 0
    previous_code = None
    last_error = ""

    for attempt in range(1, max_retries + 1):
        try:
            if previous_code is None:
                content = USER_TEMPLATE.format(
                    reference_code=problem.reference_code,
                    input_spec=input_spec,
                    axes_spec=axes_spec,
                )
            else:
                content = RETRY_TEMPLATE.format(
                    reference_code=problem.reference_code,
                    previous_code=previous_code,
                    error=last_error,
                )

            response = client.messages.create(
                model=model, max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
            )
            code = _strip_fences(response.content[0].text)
            total_in += response.usage.input_tokens
            total_out += response.usage.output_tokens

            out.write_text(code)

            if not validate:
                return {"problem_id": problem.problem_id, "success": True, "error": "",
                        "tokens_in": total_in, "tokens_out": total_out, "attempts": attempt}

            wrapped = wrap_kernel(code, problem, 0, str(out.with_suffix(".wrapped.py")))
            ok, err = _validate_kernel(wrapped, gpu_id=gpu_id)
            if ok:
                return {"problem_id": problem.problem_id, "success": True, "error": "",
                        "tokens_in": total_in, "tokens_out": total_out, "attempts": attempt}

            previous_code = code
            last_error = err

        except Exception as e:
            last_error = f"{e}\n{traceback.format_exc()}"
            previous_code = None

    return {"problem_id": problem.problem_id, "success": False,
            "error": f"failed after {max_retries} attempts: {last_error}",
            "tokens_in": total_in, "tokens_out": total_out, "attempts": max_retries}


def main():
    parser = argparse.ArgumentParser(description="Generate naive Triton kernels for SOL-ExecBench")
    parser.add_argument("--problems", type=str, default=None, help="Comma-separated problem indices")
    parser.add_argument("--category", type=str, default=None,
                        help="Filter by category: gemm, rmsnorm, fused_rmsnorm, gqa, mla, moe")
    parser.add_argument("--output-dir", type=str, default="benchmarks/sol_execbench/naive_kernels")
    parser.add_argument("--model", type=str, default="claude-sonnet-4@20250514")
    parser.add_argument("--project-id", type=str, default=None)
    parser.add_argument("--region", type=str, default="global")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    problems = list_problems()

    if args.category:
        problems = [p for p in problems if p["category"] == args.category]
    if args.problems:
        ids = set(int(x) for x in args.problems.split(","))
        problems = [p for p in problems if int(p["problem_id"].split("_")[1]) in ids]

    print(f"Generating naive kernels for {len(problems)} SOL-ExecBench problems...")
    results = []

    for i, prob in enumerate(problems):
        pid = prob["problem_id"]
        out_path = Path(args.output_dir) / f"{pid}.py"
        print(f"  [{i+1}/{len(problems)}] {prob['name']}...", end=" ", flush=True)

        result = generate_naive_kernel(
            prob["path"], str(out_path),
            model=args.model, project_id=args.project_id, region=args.region,
            gpu_id=args.gpu, max_retries=args.max_retries, validate=not args.no_validate,
        )
        results.append(result)
        status = f"OK (attempt {result['attempts']})" if result["success"] else f"FAIL: {result['error'][:60]}"
        print(status)

    summary_path = Path(args.output_dir) / "generation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2))

    ok = sum(1 for r in results if r["success"])
    print(f"\nDone: {ok}/{len(results)} succeeded. Summary: {summary_path}")


if __name__ == "__main__":
    main()
