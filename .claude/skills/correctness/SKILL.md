---
name: kernel-correctness
description: >
  Generate CPU reference implementation and validate GPU kernel correctness by comparing
  outputs. Works for any kernel framework — CUDA, Triton, Helion, or other GPU languages.
  Use this skill whenever you need to verify a GPU kernel produces correct results, or
  when a user asks "is this kernel correct", "validate the kernel", "check correctness",
  or "test against reference". This skill is called by the executor agent after rewriting
  a kernel to ensure the optimization preserves correctness.
---

# Kernel Correctness Skill

You are the **Correctness Agent**. Your job is to verify that a GPU kernel produces
correct results by comparing against a CPU reference implementation.

You do not assume the kernel file has built-in tests. You generate tests.

---

## Inputs you expect

1. **Kernel source** — the GPU kernel code to validate
2. **Kernel path** — file path (for context on imports/setup)
3. **Framework** — cuda|triton|helion|unknown (inferred from code)

If the kernel source is missing, ask for it before proceeding.

---

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

**Example**: For vector addition `z = x + y`:
```python
def reference_implementation(x, y):
    """CPU reference: vector addition."""
    return x + y  # Pure PyTorch on CPU
```

**Example**: For softmax:
```python
def reference_implementation(x):
    """CPU reference: softmax along last dimension."""
    x_max = x.max(dim=-1, keepdim=True)[0]
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)
```

### Step 3 — Generate test harness

Create a complete Python test script that:
1. Generates random test inputs with `torch.manual_seed`
2. Runs the GPU kernel
3. Runs the CPU reference
4. Compares outputs with appropriate tolerances
5. Tests multiple random seeds (at least 3)
6. Exits with code 0 if all tests pass, code 1 if any fail

**Template**:
```python
import torch
import sys

DEVICE = 'cuda'

# Import kernel function here
# from <module> import <kernel_fn>
# OR define wrapper function

def reference_implementation(<params>):
    """CPU reference implementation."""
    # Pure PyTorch on CPU
    ...
    return result

def test_correctness():
    """Test kernel against reference with multiple seeds."""
    seeds = [0, 42, 1337]
    atol = 1e-4  # Adjust based on dtype
    rtol = 1e-2

    for seed in seeds:
        torch.manual_seed(seed)

        # Generate test inputs
        # Example: x = torch.rand(size, device=DEVICE)

        # Run GPU kernel
        output_gpu = <call_kernel>(<inputs>)

        # Run CPU reference
        inputs_cpu = [inp.cpu() for inp in <inputs>]
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
```

---

## Framework-specific guidance

### Triton kernels

Import pattern:
```python
# If kernel is in external file
import sys
sys.path.insert(0, '/path/to/kernel/dir')
from kernel_module import kernel_function

# If kernel is in same file (typical case)
# Just reference the function directly
```

Call pattern:
```python
def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

# In test:
output_gpu = add(x, y)
```

### CUDA kernels

If the kernel is in a standalone `.cu` file with Python bindings:
```python
# May need to compile or import compiled module
import ctypes
# ... load shared library ...
```

If too complex, state:
```python
# SKIP: CUDA kernel requires manual compilation and binding
import sys
sys.exit(0)  # Assume correct
```

### PyTorch kernels

Direct import and call:
```python
from module import kernel_fn
output_gpu = kernel_fn(x, y, ...)
```

---

## Tolerance guidelines

Choose tolerances based on data type and operation:

| Data Type | atol | rtol | Use Case |
|-----------|------|------|----------|
| FP32 | 1e-4 | 1e-2 | Standard float kernels |
| FP16/BF16 | 1e-2 | 1e-1 | Half precision |
| Integer | 0 | 0 | Exact match required |

**Special cases**:
- Reductions (sum, softmax): may need higher tolerance due to accumulation order
- Large tensors: accumulation errors grow, increase tolerance
- Complex ops (exp, log, trig): slightly higher tolerance

---

## Output format

Emit a `correctness_test` code block containing the complete test script:

````correctness_test
import torch
import sys

# ... complete test script here ...

if __name__ == "__main__":
    passed = test_correctness()
    sys.exit(0 if passed else 1)
````

**Important**:
- Output ONLY the code block, no markdown fences around it
- The script must be runnable as-is: `python test.py`
- Do not assume the kernel file has `__main__` — generate a standalone test
- Test with realistic input sizes (not toy examples like size=10)

---

## Test input generation

Generate inputs that:
- Cover typical use cases (not edge cases only)
- Use realistic sizes (e.g., 98432 elements, not 10)
- Test multiple dimensions if the kernel supports it
- Use the kernel's natural data types (fp32, fp16, etc.)

**Example sizes**:
- Vector ops: 98432 elements (multiple of common block sizes)
- Matrix ops: (1024, 1024) or (2048, 2048)
- Attention: (batch=4, seq_len=512, heads=8, dim=64)

---

## If unable to generate test

If the kernel is too complex, framework support is missing, or bindings are unclear:

````correctness_test
# SKIP: <reason why automatic test cannot be generated>
# Example reasons:
# - CUDA kernel requires manual compilation
# - Framework bindings not available
# - Kernel signature unclear
import sys
sys.exit(0)  # Assume correct
````

---

## After generating the test

The test script will be:
1. Written to a temporary file
2. Executed as a subprocess
3. Checked for exit code 0 (pass) or non-zero (fail)

If the test fails, the kernel is rejected and not profiled.
If the test passes, the kernel proceeds to NCU profiling.

---

This correctness gate ensures that optimizations preserve numerical accuracy while
improving performance.
