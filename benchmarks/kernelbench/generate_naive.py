"""Generate naive Triton kernels for KernelBench problems.

For each KernelBench problem, makes a single-shot LLM call to produce a
Triton kernel that implements the same operation as the PyTorch Model.
These serve as the common starting point for Track A (optimization-only).

Includes GPU validation: each kernel is run with --profile-only after
generation, and re-generated (with error feedback) if it crashes.
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

from benchmarks.kernelbench.harness import list_problems, load_problem

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert GPU kernel engineer. Given a PyTorch operation,
    write a correct Triton kernel that implements it.

    Requirements:
    - Use @triton.jit for the kernel function
    - Write a Python launcher function that allocates outputs and launches the kernel
    - The launcher must have the EXACT same signature as Model.forward (same arg names, same tensor shapes)
    - Include a `if __name__ == "__main__"` block with:
      - argparse with --profile-only flag
      - When --profile-only: allocate inputs, call the launcher once, synchronize
      - Without --profile-only: run correctness check against torch reference + simple benchmark
    - Use float32 accumulation for numerical stability
    - Use triton.next_power_of_2 for block sizes where appropriate
    - The file must be self-contained (no external imports beyond torch, triton, triton.language)

    Output ONLY the Python code, no explanation.""")

USER_TEMPLATE = textwrap.dedent("""\
    Write a Triton kernel that implements this PyTorch operation:

    ```python
    {source_code}
    ```

    The kernel should match the behavior of Model.forward().
    Input shapes and dtypes are determined by get_inputs().
    Model constructor args come from get_init_inputs().""")

RETRY_TEMPLATE = textwrap.dedent("""\
    Your previous Triton kernel for this operation crashed at runtime.

    Original PyTorch code:
    ```python
    {source_code}
    ```

    Your previous kernel:
    ```python
    {previous_code}
    ```

    Runtime error:
    ```
    {error}
    ```

    Fix the kernel so it runs without errors. Output ONLY the corrected Python code.""")

VALIDATE_TIMEOUT = 60


def _strip_markdown_fences(code: str) -> str:
    """Remove markdown code fences from LLM output."""
    if code.startswith("```"):
        lines = code.split("\n")
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)
    return code


def _validate_kernel(kernel_path: str, gpu_id: int = 0) -> tuple[bool, str]:
    """Run the kernel with --profile-only to check it doesn't crash.

    Returns (ok, error_msg).
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        proc = subprocess.run(
            [sys.executable, kernel_path, "--profile-only"],
            capture_output=True, text=True, timeout=VALIDATE_TIMEOUT, env=env,
        )
        if proc.returncode == 0:
            return True, ""
        stderr = proc.stderr[-2000:] if proc.stderr else ""
        stdout = proc.stdout[-500:] if proc.stdout else ""
        return False, f"exit code {proc.returncode}\nstderr:\n{stderr}\nstdout:\n{stdout}"
    except subprocess.TimeoutExpired:
        return False, f"kernel timed out after {VALIDATE_TIMEOUT}s"
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
    """Generate a naive Triton kernel for a single KernelBench problem.

    When validate=True, runs the kernel after generation and retries with
    error feedback if it crashes (up to max_retries attempts).

    Returns dict with keys: problem_id, output_path, success, error, attempts.
    """
    from anthropic import AnthropicVertex

    problem = load_problem(problem_path)
    source_code = Path(problem_path).read_text()

    client = AnthropicVertex(
        project_id=project_id or os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID", ""),
        region=region,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    total_in = 0
    total_out = 0
    previous_code = None
    last_error = ""

    for attempt in range(1, max_retries + 1):
        try:
            if previous_code is None:
                messages = [{"role": "user", "content": USER_TEMPLATE.format(source_code=source_code)}]
            else:
                messages = [{"role": "user", "content": RETRY_TEMPLATE.format(
                    source_code=source_code,
                    previous_code=previous_code,
                    error=last_error,
                )}]

            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
            code = _strip_markdown_fences(response.content[0].text)
            total_in += response.usage.input_tokens
            total_out += response.usage.output_tokens

            out.write_text(code)

            if not validate:
                return {
                    "problem_id": problem.problem_id,
                    "output_path": str(out),
                    "success": True,
                    "error": "",
                    "tokens_in": total_in,
                    "tokens_out": total_out,
                    "attempts": attempt,
                }

            ok, err = _validate_kernel(str(out), gpu_id=gpu_id)
            if ok:
                return {
                    "problem_id": problem.problem_id,
                    "output_path": str(out),
                    "success": True,
                    "error": "",
                    "tokens_in": total_in,
                    "tokens_out": total_out,
                    "attempts": attempt,
                }

            previous_code = code
            last_error = err

        except Exception as e:
            last_error = f"{e}\n{traceback.format_exc()}"
            previous_code = None

    return {
        "problem_id": problem.problem_id,
        "output_path": output_path,
        "success": False,
        "error": f"failed after {max_retries} attempts: {last_error}",
        "tokens_in": total_in,
        "tokens_out": total_out,
        "attempts": max_retries,
    }


def _generate_one(args_tuple):
    """Top-level wrapper for ProcessPoolExecutor (must be picklable)."""
    prob, out_path, model, project_id, region, gpu_id, max_retries, validate = args_tuple
    return generate_naive_kernel(
        prob["path"], str(out_path),
        model=model, project_id=project_id, region=region,
        gpu_id=gpu_id, max_retries=max_retries, validate=validate,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate naive Triton kernels for KernelBench")
    parser.add_argument("--levels", type=str, default="1,2,3", help="Comma-separated levels")
    parser.add_argument("--problems", type=str, default=None,
                        help="Problem range (e.g. 1-10) or comma-separated IDs")
    parser.add_argument("--output-dir", type=str, default="benchmarks/kernelbench/naive_kernels")
    parser.add_argument("--model", type=str, default="claude-sonnet-4@20250514")
    parser.add_argument("--project-id", type=str, default=None)
    parser.add_argument("--region", type=str, default="global")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id for validation")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max LLM attempts per kernel (with error feedback)")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip GPU validation (original single-shot behavior)")
    parser.add_argument("--only-failed", action="store_true",
                        help="Only regenerate kernels that previously failed validation")
    args = parser.parse_args()

    levels = [int(x) for x in args.levels.split(",")]
    problems = list_problems(levels=levels)

    if args.problems:
        if "-" in args.problems:
            lo, hi = args.problems.split("-")
            ids = set(range(int(lo), int(hi) + 1))
        else:
            ids = set(int(x) for x in args.problems.split(","))
        problems = [p for p in problems if int(p["path"].split("kernel_")[1].split(".")[0]) in ids]

    if args.only_failed:
        summary_path = Path(args.output_dir) / "generation_summary.json"
        if summary_path.exists():
            prev = json.loads(summary_path.read_text())
            failed_ids = {r["problem_id"] for r in prev if not r.get("success")}
            # Also check kernels that "succeeded" in generation but crash at runtime
            validate = not args.no_validate
            recheck = []
            for prob in problems:
                pid = prob["problem_id"]
                out_path = Path(args.output_dir) / f"{pid}.py"
                if pid in failed_ids:
                    recheck.append(prob)
                elif validate and out_path.exists():
                    ok, _ = _validate_kernel(str(out_path), gpu_id=args.gpu)
                    if not ok:
                        recheck.append(prob)
            problems = recheck
            print(f"Re-generating {len(problems)} previously failed kernels...")
        else:
            print("No previous summary found, generating all.")

    print(f"Generating naive kernels for {len(problems)} problems (max_retries={args.max_retries}, validate={not args.no_validate})...")
    results = []

    for i, prob in enumerate(problems):
        pid = prob["problem_id"]
        out_path = Path(args.output_dir) / f"{pid}.py"
        print(f"  [{i+1}/{len(problems)}] {pid}...", end=" ", flush=True)

        result = generate_naive_kernel(
            prob["path"], str(out_path),
            model=args.model,
            project_id=args.project_id,
            region=args.region,
            gpu_id=args.gpu,
            max_retries=args.max_retries,
            validate=not args.no_validate,
        )
        results.append(result)
        attempts = result.get("attempts", 1)
        if result["success"]:
            status = f"OK (attempt {attempts})" if attempts > 1 else "OK"
        else:
            status = f"FAIL after {attempts} attempts: {result['error'][:80]}"
        print(status)

    summary_path = Path(args.output_dir) / "generation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if args.only_failed and summary_path.exists():
        prev = json.loads(summary_path.read_text())
        prev_by_id = {r["problem_id"]: r for r in prev}
        for r in results:
            prev_by_id[r["problem_id"]] = r
        all_results = list(prev_by_id.values())
    else:
        all_results = results

    summary_path.write_text(json.dumps(all_results, indent=2))

    ok = sum(1 for r in all_results if r["success"])
    total = len(all_results)
    retried = sum(1 for r in results if r.get("attempts", 1) > 1)
    print(f"\nDone: {ok}/{total} succeeded ({retried} needed retries). Summary: {summary_path}")


if __name__ == "__main__":
    main()
