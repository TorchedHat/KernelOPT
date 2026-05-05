"""CUDA kernel compiler: wraps nvcc, formats errors."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path


def compile_cuda(cu_path: str, output_dir: str | None = None) -> tuple[str, str | None]:
    """Compile a .cu file with nvcc into an executable.

    If the .cu file doesn't have a main() function, generates a minimal host wrapper.

    Args:
        cu_path: Path to the .cu source file
        output_dir: Directory where the binary should be saved. If None, saves to same directory as source.

    Returns:
        (binary_path, None)           on success
        (binary_path, error_message)  on failure

    binary_path is always cu_path with .cu stripped (or in output_dir if provided).
    """
    p = Path(cu_path)
    if output_dir:
        binary = str(Path(output_dir) / p.stem)
    else:
        binary = str(p.parent / p.stem)

    # Check if source has main() function
    source = Path(cu_path).read_text()
    has_main = _has_main_function(source)

    if has_main:
        # Direct compilation
        compile_file = cu_path
    else:
        # Generate wrapper with main()
        wrapper_source = _generate_cuda_wrapper(source, cu_path)
        # Create temporary file in the same directory as output for proper includes
        temp_dir = Path(output_dir) if output_dir else p.parent
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.cu',
            dir=str(temp_dir),
            delete=False
        )
        temp_file.write(wrapper_source)
        temp_file.close()
        compile_file = temp_file.name

    try:
        result = subprocess.run(
            ["nvcc", "-O2", "-o", binary, compile_file],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        if not has_main:
            Path(compile_file).unlink(missing_ok=True)
        return binary, "nvcc timed out after 60 seconds."
    finally:
        if not has_main:
            Path(compile_file).unlink(missing_ok=True)

    if result.returncode != 0:
        return binary, _format_nvcc_errors(result.stderr)
    return binary, None


def _has_main_function(source: str) -> bool:
    """Check if CUDA source has a main() function."""
    # Match: int main(...) or void main(...)
    return bool(re.search(r'\b(?:int|void)\s+main\s*\(', source))


def _generate_cuda_wrapper(original_source: str, original_path: str) -> str:
    """Generate a CUDA source file with main() that calls the kernel.

    Extracts kernel parameters and wrapper function, then creates minimal host code.
    """
    # Extract wrapper function signature (extern "C" void funcname(...))
    wrapper_match = re.search(
        r'extern\s+"C"\s+void\s+(\w+)\s*\((.*?)\)',
        original_source,
        re.DOTALL
    )

    if not wrapper_match:
        # No wrapper found - create minimal main that just returns
        return original_source + "\n\nint main() { return 0; }\n"

    wrapper_name = wrapper_match.group(1)
    params_str = wrapper_match.group(2)

    # Parse parameters
    params = [p.strip() for p in params_str.split(',') if p.strip()]

    # Generate allocation and call code based on parameter types
    wrapper_code = f"""
// Auto-generated host wrapper for NCU profiling
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {{
    // Default size
    int N = 512;

    // Parse command line args
    for (int i = 1; i < argc; i++) {{
        if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {{
            N = atoi(argv[++i]);
        }}
    }}

"""

    # Allocate device memory for pointer parameters
    for i, param in enumerate(params):
        if '*' in param:
            # Pointer parameter - allocate device memory
            if 'const' in param:
                # Input pointer
                wrapper_code += f"    float *d_arg{i};\n"
                wrapper_code += f"    cudaMalloc(&d_arg{i}, N * N * sizeof(float));\n"
            else:
                # Output pointer
                wrapper_code += f"    float *d_arg{i};\n"
                wrapper_code += f"    cudaMalloc(&d_arg{i}, N * N * sizeof(float));\n"

    # Call the wrapper function
    wrapper_code += f"\n    {wrapper_name}("
    arg_list = []
    for i, param in enumerate(params):
        if '*' in param:
            arg_list.append(f"d_arg{i}")
        elif 'int' in param:
            arg_list.append("N")
        else:
            arg_list.append("0")
    wrapper_code += ", ".join(arg_list)
    wrapper_code += ");\n"

    # Free device memory
    for i, param in enumerate(params):
        if '*' in param:
            wrapper_code += f"    cudaFree(d_arg{i});\n"

    wrapper_code += """
    return 0;
}
"""

    return original_source + wrapper_code


def _format_nvcc_errors(stderr: str) -> str:
    """Strip temp-file path prefixes from nvcc stderr.

    Keeps lines matching: filename.cu(line): error/warning/note: message
    Strips any leading /tmp/... prefix so errors are readable.
    """
    if not stderr.strip():
        return ""
    lines = stderr.strip().splitlines()
    cleaned: list[str] = []
    for line in lines:
        # Strip leading /path/to/tmpXXX/ prefix before the filename
        line = re.sub(r"^[^\s]*?/([^/]+\.cu)", r"\1", line)
        cleaned.append(line)
    return "\n".join(cleaned)
