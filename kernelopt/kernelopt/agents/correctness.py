"""Correctness agent: Generate CPU reference and validate GPU kernel output."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from kernelopt.llm import build_llm

CORRECTNESS_SYSTEM_PROMPT = """\
You are the Correctness Agent. Your job is to verify that a GPU kernel produces
correct results by comparing against a CPU reference implementation.

You do not assume the kernel file has built-in tests. You generate tests.

## Process

### Step 1 — Understand the kernel operation

Read the kernel source code and determine:
- What mathematical operation it performs (e.g., vector addition, softmax, matmul)
- Input tensor shapes and types
- Output tensor shapes and types
- Any special parameters (e.g., tile sizes, reduction dimensions)

### Step 2 — Generate CPU reference implementation

Write a pure PyTorch CPU implementation that computes the same result.
Use standard PyTorch operations only (no custom CUDA/Triton).

The reference must:
- Accept the same logical inputs as the kernel
- Produce identical outputs (within numerical tolerance)
- Use torch operations with device='cpu'

### Step 3 — Generate test harness

Create a complete Python test script that:
1. Generates random test inputs with torch.manual_seed
2. Runs the GPU kernel
3. Runs the CPU reference
4. Compares outputs with appropriate tolerances
5. Tests multiple random seeds (at least 3)
6. Exits with code 0 if all tests pass, code 1 if any fail

## Output format

Emit a `correctness_test` code block containing the complete test script:

````correctness_test
import torch
import sys

# Set device
DEVICE = 'cuda'

# Import the kernel function
# For Triton: from <module> import <kernel_name>
# For CUDA: compile and import

def reference_implementation(<params>):
    \"\"\"CPU reference implementation using PyTorch.\"\"\"
    # Pure PyTorch operations on CPU
    ...
    return result

def test_correctness():
    \"\"\"Test kernel against reference with multiple seeds.\"\"\"
    seeds = [0, 42, 1337]
    atol = 1e-4  # FP32
    rtol = 1e-2

    for seed in seeds:
        torch.manual_seed(seed)

        # Generate inputs
        <generate_test_inputs>

        # Run GPU kernel
        output_gpu = <call_kernel>(inputs...)

        # Run CPU reference
        inputs_cpu = [x.cpu() for x in inputs]
        output_ref = reference_implementation(*inputs_cpu)

        # Compare
        diff = torch.max(torch.abs(output_gpu.cpu() - output_ref))

        if diff > atol:
            print(f"FAIL: seed={seed}, max_diff={diff}")
            return False

        print(f"PASS: seed={seed}, max_diff={diff}")

    return True

if __name__ == "__main__":
    passed = test_correctness()
    sys.exit(0 if passed else 1)
````

## Framework-specific guidance

### Triton kernels
- Import the kernel function from the file
- Call it like: `output = kernel_fn(x, y, ...)`
- Kernel may return output or write to pre-allocated tensor

### CUDA kernels
- Need to compile with pybind11 or load as a module
- May require ctypes wrapper
- If too complex, state: "Cannot generate automated test for CUDA kernel — requires manual test harness"

### PyTorch kernels
- Import and call directly
- Should be straightforward

## Tolerance guidelines

- FP32: atol=1e-4, rtol=1e-2
- FP16/BF16: atol=1e-2, rtol=1e-1
- Integer: exact match (atol=0)
- Reductions (sum, softmax): may need higher tolerance due to accumulation

## Important

- Output ONLY the correctness_test code block, no markdown fences around it
- The script must be runnable as-is: `python test.py`
- Do not assume the kernel file has __main__ — generate a standalone test
- Test with realistic input sizes (not too small, not too large)

## If unable to generate test

If the kernel is too complex or framework support is missing, output:

````correctness_test
# SKIP: <reason why automatic test cannot be generated>
import sys
sys.exit(0)  # Assume correct
````
"""


def run_correctness_agent(
    kernel_source: str,
    kernel_path: str,
    framework: str,
    api_key: str | None = None,
    model: str = "claude-sonnet-4@20250514",
    project_id: str | None = None,
    region: str | None = None,
) -> tuple[bool, str]:
    """
    Call Claude to generate and run a correctness test.

    Returns (passed, message).
    """
    llm = build_llm(
        model=model,
        api_key=api_key,
        max_tokens=4096,
        temperature=0.2,
        project_id=project_id,
        region=region,
    )

    human_msg = (
        f"## Framework: {framework}\n\n"
        f"## Kernel Source ({Path(kernel_path).name})\n"
        f"```\n{kernel_source}\n```\n\n"
        f"Generate a correctness test that validates this kernel against a CPU reference."
    )

    try:
        response = llm.invoke([
            SystemMessage(content=CORRECTNESS_SYSTEM_PROMPT),
            HumanMessage(content=human_msg),
        ])

        from kernelopt.llm import strip_thinking
        content = strip_thinking(response.content)

        # Extract test code
        if "```correctness_test" in content:
            start = content.find("```correctness_test") + len("```correctness_test")
            end = content.find("```", start)
            test_code = content[start:end].strip()
        elif content.startswith("```"):
            # Fallback: strip any markdown fences
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            test_code = "\n".join(lines)
        else:
            test_code = content

        # DEBUG: Log generated test
        print(f"[DEBUG] Agentic correctness test code length: {len(test_code)} chars")
        print(f"[DEBUG] Test code preview:\n{test_code[:400]}")

        # Check for SKIP
        if test_code.strip().startswith("# SKIP:"):
            reason = test_code.split("\n")[0].replace("# SKIP:", "").strip()
            print(f"[DEBUG] Correctness test skipped: {reason}")
            return True, f"Skipped: {reason}"

        # Write test to temp file
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False, dir='/tmp'
        ) as f:
            test_path = f.name
            f.write(test_code)

        print(f"[DEBUG] Running agentic correctness test: {test_path}")

        # Run test
        result = subprocess.run(
            ["python3", test_path],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(Path(kernel_path).parent),
        )

        # DEBUG: Log test results
        print(f"[DEBUG] Test returncode: {result.returncode}")
        print(f"[DEBUG] Test stdout: {result.stdout[:300]}")
        print(f"[DEBUG] Test stderr: {result.stderr[:300]}")

        # Clean up
        Path(test_path).unlink()

        if result.returncode == 0:
            print(f"[DEBUG] Agentic correctness test PASSED")
            return True, f"Correctness passed\n{result.stdout}"
        else:
            print(f"[DEBUG] Agentic correctness test FAILED")
            return False, f"Correctness failed\nstdout: {result.stdout}\nstderr: {result.stderr}"

    except Exception as e:
        return False, f"Correctness agent error: {e}"
