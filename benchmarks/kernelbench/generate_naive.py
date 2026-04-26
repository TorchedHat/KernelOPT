"""Generate naive Triton kernels for KernelBench problems.

For each KernelBench problem, makes a single-shot LLM call to produce a
Triton kernel that implements the same operation as the PyTorch Model.
These serve as the common starting point for Track A (optimization-only).
"""
from __future__ import annotations

import argparse
import json
import os
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


def generate_naive_kernel(
    problem_path: str,
    output_path: str,
    *,
    model: str = "claude-sonnet-4@20250514",
    project_id: str | None = None,
    region: str = "global",
) -> dict:
    """Generate a naive Triton kernel for a single KernelBench problem.

    Returns dict with keys: problem_id, output_path, success, error.
    """
    from anthropic import AnthropicVertex

    problem = load_problem(problem_path)
    source_code = Path(problem_path).read_text()

    client = AnthropicVertex(
        project_id=project_id or os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID", ""),
        region=region,
    )

    try:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": USER_TEMPLATE.format(source_code=source_code)}],
        )
        code = response.content[0].text

        if code.startswith("```"):
            lines = code.split("\n")
            lines = lines[1:] if lines[0].startswith("```") else lines
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = "\n".join(lines)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(code)

        return {
            "problem_id": problem.problem_id,
            "output_path": str(out),
            "success": True,
            "error": "",
            "tokens_in": response.usage.input_tokens,
            "tokens_out": response.usage.output_tokens,
        }

    except Exception as e:
        return {
            "problem_id": problem.problem_id,
            "output_path": output_path,
            "success": False,
            "error": f"{e}\n{traceback.format_exc()}",
            "tokens_in": 0,
            "tokens_out": 0,
        }


def main():
    parser = argparse.ArgumentParser(description="Generate naive Triton kernels for KernelBench")
    parser.add_argument("--levels", type=str, default="1,2,3", help="Comma-separated levels")
    parser.add_argument("--problems", type=str, default=None,
                        help="Problem range (e.g. 1-10) or comma-separated IDs")
    parser.add_argument("--output-dir", type=str, default="benchmarks/kernelbench/naive_kernels")
    parser.add_argument("--model", type=str, default="claude-sonnet-4@20250514")
    parser.add_argument("--project-id", type=str, default=None)
    parser.add_argument("--region", type=str, default="global")
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

    print(f"Generating naive kernels for {len(problems)} problems...")
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
        )
        results.append(result)
        status = "OK" if result["success"] else f"FAIL: {result['error'][:80]}"
        print(status)

    summary_path = Path(args.output_dir) / "generation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2))

    ok = sum(1 for r in results if r["success"])
    print(f"\nDone: {ok}/{len(results)} succeeded. Summary: {summary_path}")


if __name__ == "__main__":
    main()
